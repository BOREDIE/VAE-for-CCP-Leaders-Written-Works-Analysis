"""
CCP Leadership Writings — Unified VAE Pipeline
===============================================
Entry point. Edit config.py to set paths and defaults, or override via CLI flags.

Usage examples
--------------
# TF-IDF pipeline (CPU-friendly):
python main.py --pipeline tfidf --meta-csv data/metadata.csv --data-dir data/texts --out-dir outputs/

# Qwen3 embedding pipeline (GPU recommended):
python main.py --pipeline qwen3 --meta-csv data/metadata.csv --data-dir data/texts --out-dir outputs/

# Run hyperparameter sweep first, then train with defaults:
python main.py --pipeline tfidf --sweep --meta-csv data/metadata.csv --data-dir data/texts --out-dir outputs/
"""

import os
import warnings
import argparse
warnings.filterwarnings("ignore")

import torch

import config as cfg
from data import load_corpus, build_chunks
from features import extract_tfidf, extract_qwen3, extract_custom, build_tensors
from model import BetaVAE
from train import sanity_check, train_vae
from evaluate import compute_silhouette, project_pca, project_tsne, tfidf_interpretability, run_sweep
from visualize import plot_training_curves, plot_latent_projections, plot_dim_activations, plot_sweep
from checkpoint import save_checkpoint, save_latent_csv, save_hyperparams


def parse_args():
    p = argparse.ArgumentParser(description="CCP Leadership Writings — Unified β-VAE Pipeline")
    p.add_argument("--pipeline",   choices=["tfidf", "qwen3", "custom"], default=cfg.PIPELINE,
                   help="Feature extraction method (default: %(default)s)")
    p.add_argument("--meta-csv",   default=cfg.HP["meta_csv"],
                   help="Path to metadata CSV with columns: filename, author_english, date")
    p.add_argument("--data-dir",   default=cfg.HP["data_dir"],
                   help="Directory containing the raw text files")
    p.add_argument("--out-dir",    default=cfg.HP["out_dir"],
                   help="Directory for checkpoints, plots, and CSVs")
    p.add_argument("--epochs",     type=int, default=cfg.HP["epochs"])
    p.add_argument("--latent-dim", type=int, default=None,
                   help="Latent dimension z (default: 16 for tfidf, 32 for embedding)")
    p.add_argument("--beta",       type=float, default=None,
                   help="β regularisation weight (default: 0.5 for tfidf, 0.25 for embedding)")
    p.add_argument("--sweep",      action="store_true",
                   help="Run a hyperparameter sweep before main training run")
    p.add_argument("--no-sanity",  action="store_true",
                   help="Skip the forward-pass sanity check")
    return p.parse_args()


def apply_pipeline_defaults(pipeline: str, hp: dict, latent_dim, beta):
    if pipeline == "tfidf":
        hp["latent_dim"]    = latent_dim or 16
        hp["beta"]          = beta or 0.5
        hp["free_bits"]     = 0.0
        hp["learning_rate"] = 1e-3
        hp["lr_gamma"]      = 0.97
        recon_loss = "bce"
    else:
        hp["latent_dim"]    = latent_dim or 32
        hp["beta"]          = beta or 0.25
        hp["free_bits"]     = 0.05
        hp["learning_rate"] = 2e-4
        hp["lr_gamma"]      = 0.98
        recon_loss = "cosine"
    return recon_loss


def main():
    args     = parse_args()
    pipeline = args.pipeline
    device   = cfg.DEVICE
    seed     = cfg.SEED

    # Patch HP with CLI overrides
    hp = dict(cfg.HP)
    hp["meta_csv"] = args.meta_csv
    hp["data_dir"] = args.data_dir
    hp["out_dir"]  = args.out_dir
    hp["epochs"]   = args.epochs
    recon_loss     = apply_pipeline_defaults(pipeline, hp, args.latent_dim, args.beta)

    os.makedirs(hp["out_dir"], exist_ok=True)
    print(f"Pipeline : {pipeline}")
    print(f"Device   : {device}")
    print(f"z={hp['latent_dim']}  beta={hp['beta']}  free_bits={hp['free_bits']}")

    # ── 1. Load and chunk corpus ───────────────────────────────────────────
    print("\nLoading corpus ...", flush=True)
    df_docs = load_corpus(hp)
    print(f"  Documents : {len(df_docs)}", flush=True)

    print("Chunking ...", flush=True)
    df_chunks = build_chunks(df_docs, hp)
    print(f"  Chunks : {len(df_chunks)}", flush=True)
    print(df_chunks["author"].value_counts().to_string(), flush=True)

    # ── 2. Feature extraction ─────────────────────────────────────────────
    vec = None
    feature_names = None
    if pipeline == "tfidf":
        X_np, vec = extract_tfidf(df_chunks, hp)
        feature_names = vec.get_feature_names_out()
    elif pipeline == "qwen3":
        X_np, _ = extract_qwen3(df_chunks, hp)
    elif pipeline == "custom":
        X_np, _ = extract_custom(
            df_chunks, hp,
            cfg.CUSTOM_MODEL_ID, cfg.CUSTOM_EMBED_DIM, cfg.CUSTOM_EMBED_INSTRUCTION,
        )

    X_tensor, y_tensor, y_np, authors, a2i, loader = build_tensors(X_np, df_chunks, hp, device)
    print(f"\nInput dim : {hp['input_dim']}   Chunks : {len(X_np)}   Batches/epoch : {len(loader)}")
    print(f"Reconstruction loss : {recon_loss.upper()}")

    # ── 3. Optional hyperparameter sweep ─────────────────────────────────
    if args.sweep:
        print("\n── Hyperparameter Sweep ──", flush=True)
        df_sweep = run_sweep(pipeline, recon_loss, hp, loader, X_tensor, y_np, device, seed=seed)
        plot_sweep(df_sweep, pipeline, hp, hp["out_dir"])

    # ── 4. Build model ────────────────────────────────────────────────────
    model = BetaVAE(hp, recon_loss=recon_loss).to(device)
    print(
        f"\nBetaVAE — recon={recon_loss.upper()}  z={hp['latent_dim']}  "
        f"beta={hp['beta']}  free_bits={hp['free_bits']}"
    )
    print(f"  Input {hp['input_dim']} → Hidden {hp['hidden_dim']}×{hp['depth']} → Latent {hp['latent_dim']}")
    print(f"  Parameters : {model.count_params():,}")

    # ── 5. Sanity check ───────────────────────────────────────────────────
    if not args.no_sanity:
        sanity_check(model, loader, device)

    # ── 6. Train ──────────────────────────────────────────────────────────
    print(
        f"\nTraining  {pipeline}  z={hp['latent_dim']}  beta={hp['beta']}  "
        f"warmup={hp['kl_warmup_epochs']}ep  free_bits={hp['free_bits']}",
        flush=True,
    )
    history = train_vae(model, loader, hp, device, verbose=True)
    print("\nTraining complete ✓", flush=True)

    hp["recon_loss"] = recon_loss
    plot_training_curves(history, pipeline, hp, hp["out_dir"])

    # ── 7. Latent extraction & metrics ────────────────────────────────────
    print("Extracting mu vectors ...", flush=True)
    Z_mu = model.encode_mu(X_tensor)
    print(f"  Z_mu : {Z_mu.shape}", flush=True)

    if len(authors) > 1:
        compute_silhouette(Z_mu, X_np, y_np, seed=seed)

    Z_pca, var_ex = project_pca(Z_mu, seed=seed)
    Z_tsne, perp  = project_tsne(Z_mu, hp, seed=seed)

    # ── 8. Visualisations ─────────────────────────────────────────────────
    plot_latent_projections(Z_pca, Z_tsne, var_ex, perp,
                             df_chunks, authors, cfg.AUTHOR_META, pipeline, hp, hp["out_dir"])
    plot_dim_activations(Z_mu, df_chunks, authors, cfg.AUTHOR_META, pipeline, hp, hp["out_dir"])

    if pipeline == "tfidf":
        tfidf_interpretability(Z_mu, X_np, df_chunks, feature_names, authors, cfg.AUTHOR_META)

    # ── 9. Save outputs ───────────────────────────────────────────────────
    save_checkpoint(model, hp, history, authors, a2i, pipeline, recon_loss, hp["out_dir"], vec=vec)
    save_latent_csv(df_chunks, Z_pca, Z_tsne, Z_mu, hp, cfg.AUTHOR_META, pipeline, hp["out_dir"])
    save_hyperparams(hp, pipeline, hp["out_dir"])
    print("\nAll outputs saved ✓", flush=True)


if __name__ == "__main__":
    main()
