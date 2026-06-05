"""
Amortised encoder  q_φ(z | x).

Optionally enriches the raw embedding x with an ArrowSpace-style
λ-fingerprint computed from a pre-built base Laplacian, replicating
the mechanistic interpretability workflow of the analysis notebooks.

Architecture
------------
    x  [λ-fingerprint]  →  MLP  →  (mu, log_var)  →  z (reparameterised)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


class WiringEncoder(nn.Module):
    """
    Parameters
    ----------
    input_dim : int
        Dimension of raw input embeddings (D).
    latent_dim : int
        Dimension of latent code z.
    hidden_dim : int
        Hidden layer width.
    use_lambda_features : bool
        If True, concatenate λ-fingerprint (n_lambda_bins dimensions)
        to input before encoding.
    n_lambda_bins : int
        Number of histogram bins in the λ-fingerprint.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        use_lambda_features: bool = True,
        n_lambda_bins: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_lambda_features = use_lambda_features
        self.n_lambda_bins = n_lambda_bins
        in_dim = input_dim + (n_lambda_bins if use_lambda_features else 0)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head      = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,                          # (B, D)
        lambda_fp: Optional[torch.Tensor] = None, # (B, n_lambda_bins)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z       : Tensor (B, latent_dim)   — reparameterised sample
        mu      : Tensor (B, latent_dim)
        log_var : Tensor (B, latent_dim)
        """
        if self.use_lambda_features and lambda_fp is not None:
            x = torch.cat([x, lambda_fp], dim=-1)  # (B, D + n_bins)

        h       = self.net(x)
        mu      = self.mu_head(h)
        log_var = self.log_var_head(h).clamp(-10.0, 4.0)  # numerical stability

        z = self._reparameterise(mu, log_var)
        return z, mu, log_var

    @staticmethod
    def _reparameterise(
        mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """z = mu + eps * std,  eps ~ N(0, I)."""
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(
        mu: torch.Tensor, log_var: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        KL( q(z|x) || N(0,I) ) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        """
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl = kl.sum(dim=-1)   # sum over latent dims
        return kl.mean() if reduction == "mean" else kl.sum()
