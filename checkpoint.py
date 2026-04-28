import os
import json

import torch
import pandas as pd


def save_checkpoint(model, hp: dict, history: dict, authors: list, a2i: dict,
                    pipeline: str, recon_loss: str, out_dir: str, vec=None) -> str:
    ckpt_path = os.path.join(out_dir, f"{pipeline}_vae_checkpoint.pt")
    save_dict = {
        "model_state_dict": model.state_dict(),
        "hyperparams":      hp,
        "history":          history,
        "authors":          authors,
        "a2i":              a2i,
        "pipeline":         pipeline,
        "recon_loss":       recon_loss,
    }
    if pipeline == "tfidf" and vec is not None:
        save_dict["vocab"] = vec.get_feature_names_out().tolist()
    torch.save(save_dict, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}", flush=True)
    return ckpt_path


def load_checkpoint(ckpt_path: str, model_cls, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    hp   = ckpt["hyperparams"]
    rl   = ckpt["recon_loss"]
    model = model_cls(hp, recon_loss=rl).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def save_latent_csv(df_chunks, Z_pca, Z_tsne, Z_mu,
                    hp: dict, author_meta: dict, pipeline: str, out_dir: str) -> str:
    df_out = df_chunks[["doc_id", "author", "date", "chunk_id"]].copy()
    df_out["pca_x"]  = Z_pca[:, 0]
    df_out["pca_y"]  = Z_pca[:, 1]
    df_out["tsne_x"] = Z_tsne[:, 0]
    df_out["tsne_y"] = Z_tsne[:, 1]
    for k in range(hp["latent_dim"]):
        df_out[f"z_{k}"] = Z_mu[:, k]
    df_out["faction"] = df_out["author"].map(
        lambda a: author_meta.get(a, {"faction": "Unknown"})["faction"]
    )
    csv_path = os.path.join(out_dir, f"{pipeline}_vae_latent_coords.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Latent CSV saved → {csv_path}  {df_out.shape}", flush=True)
    return csv_path


def save_hyperparams(hp: dict, pipeline: str, out_dir: str) -> str:
    hp_path = os.path.join(out_dir, f"{pipeline}_vae_hyperparams.json")
    with open(hp_path, "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in hp.items()}, f, indent=2, ensure_ascii=False)
    print(f"Hyperparams saved → {hp_path}", flush=True)
    return hp_path
