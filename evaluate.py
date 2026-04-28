import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def compute_silhouette(Z_mu, X_np, y_np, seed: int = 42):
    sil     = silhouette_score(Z_mu, y_np, sample_size=min(3000, len(y_np)), random_state=seed)
    sil_raw = silhouette_score(X_np, y_np, sample_size=min(3000, len(y_np)), random_state=seed)
    print(f"  Silhouette (VAE z)     : {sil:.4f}", flush=True)
    print(f"  Silhouette (raw input) : {sil_raw:.4f}", flush=True)
    print(f"  VAE compression gain   : {sil - sil_raw:+.4f}", flush=True)
    return sil, sil_raw


def project_pca(Z_mu, seed: int = 42):
    pca_2d = PCA(n_components=2, random_state=seed)
    Z_pca  = pca_2d.fit_transform(Z_mu)
    var_ex = pca_2d.explained_variance_ratio_
    print(f"PCA  PC1={var_ex[0]:.2%}  PC2={var_ex[1]:.2%}", flush=True)
    return Z_pca, var_ex


def project_tsne(Z_mu, hp: dict, seed: int = 42):
    perp   = min(hp["tsne_perplexity"], max(5, (len(Z_mu) - 1) // 3))
    print(f"t-SNE perplexity={perp} ...", flush=True)
    tsne   = TSNE(n_components=2, perplexity=perp, max_iter=hp["tsne_iter"],
                  random_state=seed, init="pca", learning_rate="auto", n_jobs=-1)
    Z_tsne = tsne.fit_transform(Z_mu)
    return Z_tsne, perp


def tfidf_interpretability(Z_mu, X_np, df_chunks, feature_names, authors, author_meta):
    author_means = {
        a: Z_mu[df_chunks["author"].values == a].mean(axis=0)
        for a in authors if (df_chunks["author"].values == a).any()
    }
    between_var = np.var(np.stack(list(author_means.values())), axis=0)
    top_dims    = np.argsort(between_var)[::-1][:8]

    print("Top 8 latent dimensions by between-author variance:", flush=True)
    for k in top_dims:
        z_k     = Z_mu[:, k]
        corr    = np.array([np.corrcoef(X_np[:, f], z_k)[0, 1] for f in range(X_np.shape[1])])
        top_pos = corr.argsort()[::-1][:8]
        top_neg = corr.argsort()[:8]
        print(f"  z[{k:2d}]  var={between_var[k]:.4f}", flush=True)
        print(
            f"    Author means: "
            + " ".join(f"{a.split()[0]}={author_means[a][k]:+.3f}" for a in authors if a in author_means),
            flush=True,
        )
        print(f"    High z: {' '.join(feature_names[top_pos])}", flush=True)
        print(f"    Low  z: {' '.join(feature_names[top_neg])}", flush=True)


def run_sweep(pipeline: str, recon_loss: str, hp: dict, loader, X_tensor, y_np, device, seed: int = 42):
    import pandas as pd
    from sklearn.metrics import silhouette_score as sil_score
    from model import BetaVAE
    from train import train_vae

    if pipeline == "tfidf":
        SWEEP_LATENT_DIMS = [8, 16, 32, 64]
        SWEEP_BETAS       = [0.5, 1.0, 2.0]
    else:
        SWEEP_LATENT_DIMS = [8, 16, 32]
        SWEEP_BETAS       = [0.25, 0.5, 1.0]
    SWEEP_EPOCHS = 40

    n_configs = len(SWEEP_LATENT_DIMS) * len(SWEEP_BETAS)
    print(f"Sweep: {n_configs} configs x {SWEEP_EPOCHS} epochs", flush=True)
    sweep_rows = []

    for z_dim in SWEEP_LATENT_DIMS:
        for beta_val in SWEEP_BETAS:
            hp_s = {**hp, "latent_dim": z_dim, "beta": beta_val, "epochs": SWEEP_EPOCHS}
            m_s  = BetaVAE(hp_s, recon_loss=recon_loss).to(device)
            hs   = train_vae(m_s, loader, hp_s, device, verbose=False)
            Zs   = m_s.encode_mu(X_tensor)
            sil  = (
                sil_score(Zs, y_np, sample_size=min(2000, len(y_np)), random_state=seed)
                if len(set(y_np.tolist())) > 1 else 0.0
            )
            sweep_rows.append({
                "latent_dim": z_dim, "beta": beta_val, "silhouette": sil,
                "recon": hs["recon"][-1], "kl": hs["kl"][-1],
            })
            print(
                f"  z={z_dim:>2}  beta={beta_val:.2f}  sil={sil:.4f}  "
                f"recon={hs['recon'][-1]:.4f}  kl={hs['kl'][-1]:.4f}",
                flush=True,
            )

    df_sweep = pd.DataFrame(sweep_rows)
    best = df_sweep.loc[df_sweep["silhouette"].idxmax()]
    print(f"\nBest: z={int(best['latent_dim'])}  beta={best['beta']}  sil={best['silhouette']:.4f}", flush=True)
    print(f"Suggest: HP['latent_dim']={int(best['latent_dim'])}  HP['beta']={best['beta']}", flush=True)
    return df_sweep
