import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


_BETA_COLORS = {0.25: "#F06292", 0.5: "#378ADD", 1.0: "#E24B4A", 2.0: "#1D9E75"}


def plot_training_curves(history: dict, pipeline: str, hp: dict, out_dir: str) -> str:
    recon_label = "BCE" if hp.get("recon_loss", "bce") == "bce" else "Cosine"
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), facecolor="#f8f7f4")
    fig.suptitle(
        f"Training curves  {pipeline.upper()}  z={hp['latent_dim']}  beta={hp['beta']}  "
        f"warmup={hp['kl_warmup_epochs']}ep  ep={hp['epochs']}",
        fontsize=12, fontweight="bold",
    )
    ep_x = range(1, len(history["total"]) + 1)
    for ax, key, title, col in zip(
        axes[:3],
        ["total", "recon", "kl"],
        ["Total ELBO", f"Reconstruction ({recon_label})", "KL divergence"],
        ["#378ADD", "#E24B4A", "#1D9E75"],
    ):
        ax.plot(ep_x, history[key], color=col, lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#fafaf8")

    axes[3].plot(ep_x, history["kl_weight"], color="#9C27B0", lw=2)
    axes[3].axhline(hp["beta"], color="#9C27B0", lw=1, ls="--", alpha=0.5,
                    label=f"target beta={hp['beta']}")
    axes[3].set_title("KL warmup schedule", fontsize=10)
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("KL weight")
    axes[3].legend(fontsize=8)
    axes[3].spines["top"].set_visible(False)
    axes[3].spines["right"].set_visible(False)
    axes[3].set_facecolor("#fafaf8")

    plt.tight_layout()
    out = os.path.join(out_dir, f"{pipeline}_vae_training_curves.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}", flush=True)

    kl_f = history["kl"][-1]
    if kl_f < 0.01:
        print("KL~0 — posterior collapse. Raise beta or lower lr.", flush=True)
    elif kl_f > 100:
        print("KL very high — lower beta.", flush=True)
    else:
        print(f"KL healthy: {kl_f:.4f}", flush=True)

    return out


def plot_latent_projections(Z_pca, Z_tsne, var_ex, perp: int,
                             df_chunks, authors: list, author_meta: dict,
                             pipeline: str, hp: dict, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#f8f7f4")
    fig.suptitle(
        f"Latent projections  {pipeline.upper()}  z={hp['latent_dim']}  beta={hp['beta']}",
        fontsize=13, fontweight="bold",
    )
    for ax, Z2d, title in zip(
        axes,
        [Z_pca, Z_tsne],
        [f"PCA ({var_ex[0]:.1%}+{var_ex[1]:.1%})", f"t-SNE (perp={perp})"],
    ):
        for a in authors:
            mask = df_chunks["author"].values == a
            if not mask.any():
                continue
            am  = author_meta.get(a, {"color": "#888", "marker": "o", "faction": "?"})
            pts = Z2d[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=am["color"], marker=am["marker"],
                       s=30, alpha=0.6, linewidths=0, label=f"{a} [{am['faction']}]")
            # centroid
            ax.scatter(pts[:, 0].mean(), pts[:, 1].mean(), c=am["color"], marker=am["marker"],
                       s=240, edgecolors="white", linewidths=1.8, zorder=6)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, framealpha=0.7, markerscale=1.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#fafaf8")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    plt.tight_layout()
    out = os.path.join(out_dir, f"{pipeline}_vae_latent_projections.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}", flush=True)
    return out


def plot_dim_activations(Z_mu, df_chunks, authors: list, author_meta: dict,
                          pipeline: str, hp: dict, out_dir: str) -> str:
    n_show = min(hp["latent_dim"], 12)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, n_show + 1),
                              facecolor="#f8f7f4", squeeze=False)
    fig.suptitle(
        f"Latent dimension activations  ({pipeline.upper()} VAE z={hp['latent_dim']})",
        fontsize=12, fontweight="bold",
    )
    for di in range(n_show):
        ax = axes[di][0]
        for a in authors:
            mask = df_chunks["author"].values == a
            if not mask.any():
                continue
            ax.hist(Z_mu[mask, di], bins=30, alpha=0.55, density=True,
                    color=author_meta.get(a, {"color": "#888"})["color"],
                    label=a if di == 0 else "")
        ax.set_ylabel(f"z[{di}]", fontsize=8, rotation=0, labelpad=36)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#fafaf8")
        ax.tick_params(labelsize=7)
        if di < n_show - 1:
            ax.set_xticks([])

    handles = [
        mpatches.Patch(color=author_meta.get(a, {"color": "#888"})["color"], label=a)
        for a in authors if (df_chunks["author"].values == a).any()
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.84, 1])
    out = os.path.join(out_dir, f"{pipeline}_vae_dim_activations.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}", flush=True)
    return out


def plot_sweep(df_sweep, pipeline: str, hp: dict, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#f8f7f4")
    fig.suptitle(
        f"{pipeline.upper()} VAE sweep  (warmup={hp['kl_warmup_epochs']}ep)",
        fontsize=13, fontweight="bold",
    )
    for bv in df_sweep["beta"].unique():
        sub = df_sweep[df_sweep["beta"] == bv].sort_values("latent_dim")
        c   = _BETA_COLORS.get(bv, "#888")
        axes[0].plot(sub["latent_dim"], sub["silhouette"], "o-",  color=c, lw=2, ms=8, label=f"beta={bv}")
        axes[1].plot(sub["latent_dim"], sub["recon"],      "s--", color=c, lw=2, ms=8, label=f"beta={bv}")
        axes[2].plot(sub["latent_dim"], sub["kl"],         "^-.", color=c, lw=2, ms=8, label=f"beta={bv}")
    for ax, yl, tl in zip(
        axes,
        ["Silhouette", "Recon loss", "KL divergence"],
        ["Separability vs z", "Recon quality vs z", "Regularisation vs z"],
    ):
        ax.set_xlabel("z")
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(tl, fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#fafaf8")

    plt.tight_layout()
    out = os.path.join(out_dir, f"{pipeline}_vae_sweep.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}", flush=True)
    return out
