import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, depth: int, dropout: float):
        super().__init__()
        layers, in_d = [], input_dim
        for _ in range(depth):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        self.hidden  = nn.Sequential(*layers)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logv = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h    = self.hidden(x)
        mu   = self.fc_mu(h)
        logv = self.fc_logv(h).clamp(-10, 10)
        return mu, logv


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 depth: int, dropout: float, use_sigmoid: bool = False):
        super().__init__()
        layers, in_d = [], latent_dim
        for _ in range(depth):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        layers += [nn.Linear(hidden_dim, output_dim)]
        if use_sigmoid:
            layers += [nn.Sigmoid()]   # required for BCE loss
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BetaVAE(nn.Module):
    def __init__(self, hp: dict, recon_loss: str = "bce"):
        super().__init__()
        D, H, Z  = hp["input_dim"], hp["hidden_dim"], hp["latent_dim"]
        use_sig  = (recon_loss == "bce")
        self.encoder    = Encoder(D, H, Z, hp["depth"], hp["dropout"])
        self.decoder    = Decoder(Z, H, D, hp["depth"], hp["dropout"], use_sigmoid=use_sig)
        self.beta       = hp["beta"]
        self.recon_loss = recon_loss
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def reparameterise(self, mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logv).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):
        mu, logv = self.encoder(x)
        z        = self.reparameterise(mu, logv)
        x_hat    = self.decoder(z)
        return x_hat, mu, logv, z

    def elbo_loss(self, x, x_hat, mu, logv, kl_weight=None, free_bits: float = 0.0):
        if kl_weight is None:
            kl_weight = self.beta
        logv = logv.clamp(-10, 10)

        # Reconstruction loss
        if self.recon_loss == "bce":
            recon = F.binary_cross_entropy(x_hat, x, reduction="mean")
        else:  # cosine — for L2-normalised embedding vectors
            x_hat_n = F.normalize(x_hat, p=2, dim=1)
            x_n     = F.normalize(x,     p=2, dim=1)
            recon   = (1.0 - (x_n * x_hat_n).sum(dim=1)).mean()

        # KL divergence with optional free-bits regularisation
        kl_per_dim = -0.5 * (1 + logv - mu.pow(2) - logv.exp())
        if free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl = kl_per_dim.mean()

        return recon + kl_weight * kl, recon, kl

    @torch.no_grad()
    def encode_mu(self, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
        self.eval()
        mus = []
        for i in range(0, len(X), batch_size):
            mu, _ = self.encoder(X[i:i + batch_size])
            mus.append(mu.cpu().numpy())
        return np.vstack(mus)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
