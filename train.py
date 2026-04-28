import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


def sanity_check(model, loader, device) -> bool:
    model.eval()
    with torch.no_grad():
        x_sample           = next(iter(loader))[0].to(device)
        x_hat, mu, logv, z = model(x_sample)
        loss, recon, kl    = model.elbo_loss(x_sample, x_hat, mu, logv)

    print(f"Input  range [{x_sample.min():.4f}, {x_sample.max():.4f}]  nan={x_sample.isnan().any().item()}", flush=True)
    print(f"x_hat  range [{x_hat.min():.4f}, {x_hat.max():.4f}]   nan={x_hat.isnan().any().item()}", flush=True)
    print(f"mu     range [{mu.min():.4f}, {mu.max():.4f}]", flush=True)
    print(f"logv   range [{logv.min():.4f}, {logv.max():.4f}]", flush=True)
    print(f"Loss   recon={recon:.4f}  kl={kl:.4f}  total={loss:.4f}", flush=True)
    ok = not (loss.isnan() or loss.isinf())
    print(f"\n{'✓ Forward pass clean — safe to train.' if ok else '✗ NaN/Inf in loss — do NOT train.'}", flush=True)
    assert ok, "Fix the loss before training!"
    model.train()
    return ok


def train_vae(model, loader, hp: dict, device, verbose: bool = True) -> dict:
    optimiser     = Adam(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    scheduler     = ExponentialLR(optimiser, gamma=hp["lr_gamma"])
    warmup_epochs = hp.get("kl_warmup_epochs", 20)
    free_bits     = hp.get("free_bits", 0.0)
    history       = {"total": [], "recon": [], "kl": [], "kl_weight": []}
    t0 = time.time()

    for epoch in range(1, hp["epochs"] + 1):
        kl_weight = min(1.0, epoch / warmup_epochs) * hp["beta"]
        model.train()
        ep_total = ep_recon = ep_kl = 0.0

        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            optimiser.zero_grad()
            x_hat, mu, logv, _ = model(x_batch)
            loss, recon, kl    = model.elbo_loss(x_batch, x_hat, mu, logv,
                                                  kl_weight=kl_weight, free_bits=free_bits)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimiser.step()
            ep_total += loss.item()
            ep_recon += recon.item()
            ep_kl    += kl.item()

        scheduler.step()
        n = len(loader)
        history["total"].append(ep_total / n)
        history["recon"].append(ep_recon / n)
        history["kl"].append(ep_kl / n)
        history["kl_weight"].append(kl_weight)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Ep {epoch:>3}/{hp['epochs']}  "
                f"total={history['total'][-1]:9.4f}  "
                f"recon={history['recon'][-1]:9.4f}  "
                f"kl={history['kl'][-1]:7.4f}  "
                f"kl_w={kl_weight:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.1e}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    return history
