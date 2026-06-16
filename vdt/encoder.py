"""
Amortised encoder  q_phi(z | x).

Two encoder classes are provided:

  WiringEncoder   -- v1 MLP encoder (unchanged; preserved for backward compat).
  WiringEncoderV2 -- v2 VDT encoder using VibrationalStateBlock recurrence
                     and variational Gamma parameters (ModeWeightHead).

Both share the same reparameterise / kl_loss utilities.

Typical usage (v2)
------------------
    encoder = WiringEncoderV2(
        input_dim=512, latent_dim=64,
        n_nodes=128, feat_dim=32,
        n_layers=4, m_modes=16,
    )
    z, mu, log_var, log_a, log_b = encoder(
        x, L_f=L_f, eigvecs=U, lap=lap
    )

Typical usage (v1 legacy)
--------------------------
    encoder = WiringEncoder(input_dim=512, latent_dim=64)
    z, mu, log_var = encoder(x, lambda_fp=fp)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from vdt.vdt import VDT
from vdt.laplacian import DifferentiableLaplacian


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _reparameterise(
    mu: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """z = mu + eps * std,  eps ~ N(0, I)."""
    std = (0.5 * log_var).exp()
    return mu + torch.randn_like(std) * std


def kl_isotropic(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL( q(z|x) || N(0, I) ) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Parameters
    ----------
    mu, log_var : Tensor (B, latent_dim)
    reduction   : 'mean' or 'sum'
    """
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
    kl = kl.sum(dim=-1)
    return kl.mean() if reduction == "mean" else kl.sum()


# ---------------------------------------------------------------------------
# ModeWeightHead  (v2 only)
# ---------------------------------------------------------------------------

class ModeWeightHead(nn.Module):
    """
    Produce variational Gamma parameters (log_a, log_b) per latent mode.

    A Gamma distribution Gamma(a, b) is parameterised through its
    log-shape log_a and log-rate log_b.  The Gamma KL replaces the
    isotropic Gaussian KL when use_isotropic_kl=False in WiringEncoderV2.

    Parameters
    ----------
    hidden_dim : int
        Input dimension (VDT pooled output width).
    q : int
        Number of latent modes (= latent_dim).  Output is 2*q.
    """

    def __init__(self, hidden_dim: int, q: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_dim, 2 * q)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : Tensor (B, hidden_dim)

        Returns
        -------
        log_a : Tensor (B, q)
        log_b : Tensor (B, q)
        """
        out   = self.head(h)          # (B, 2q)
        log_a = out[:, : out.shape[-1] // 2]   # (B, q)
        log_b = out[:,   out.shape[-1] // 2:]  # (B, q)
        return log_a, log_b


# ---------------------------------------------------------------------------
# WiringEncoder  (v1 -- unchanged).
# ---------------------------------------------------------------------------

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
        If True, concatenate lambda-fingerprint (n_lambda_bins dimensions)
        to input before encoding.
    n_lambda_bins : int
        Number of histogram bins in the lambda-fingerprint.
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
        x: torch.Tensor,
        lambda_fp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z       : Tensor (B, latent_dim)
        mu      : Tensor (B, latent_dim)
        log_var : Tensor (B, latent_dim)
        """
        if self.use_lambda_features and lambda_fp is not None:
            x = torch.cat([x, lambda_fp], dim=-1)

        h       = self.net(x)
        mu      = self.mu_head(h)
        log_var = self.log_var_head(h).clamp(-10.0, 4.0)
        z       = _reparameterise(mu, log_var)
        return z, mu, log_var

    @staticmethod
    def _reparameterise(
        mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """z = mu + eps * std,  eps ~ N(0, I)."""
        return _reparameterise(mu, log_var)

    @staticmethod
    def kl_loss(
        mu: torch.Tensor,
        log_var: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        KL( q(z|x) || N(0, I) )
        """
        return kl_isotropic(mu, log_var, reduction)


# ---------------------------------------------------------------------------
# WiringEncoderV2  (v2 VDT encoder)
# ---------------------------------------------------------------------------

class WiringEncoderV2(nn.Module):
    """
    v2 encoder: VDT recurrence + ModeWeightHead.

    The encoder projects input x to an (N, d) node-feature matrix,
    runs K VDT blocks, and reads out:
      - (mu, log_var) from a linear head on the modal projection z,
      - (log_a, log_b) from ModeWeightHead for the variational Gamma KL.

    Parameters
    ----------
    input_dim : int
        Raw input embedding dimension D.
    latent_dim : int
        Latent code dimension q.
    n_nodes : int
        Graph node count N (must match the Laplacian passed at forward time).
    feat_dim : int
        Per-node feature channels d.  VDT operates on (N, d) state matrices.
    n_layers : int
        Number of VDT blocks K.
    m_modes : int or None
        Number of eigenmodes for the modal projection.  Defaults to feat_dim//4.
    n_heads : int
        Attention heads in each VibrationalStateBlock.
    use_lambda_features : bool
        If True, the lambda-fingerprint is concatenated to x before projection.
    n_lambda_bins : int
        Histogram bins in the lambda-fingerprint.
    use_isotropic_kl : bool
        If True, kl_loss() uses the isotropic Gaussian KL (v1 fallback).
        If False, the caller should use the Gamma KL from #24.
    dropout : float
        Dropout in VDT blocks and projection head.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_nodes: int,
        feat_dim: int,
        n_layers: int = 4,
        m_modes: Optional[int] = None,
        n_heads: int = 4,
        use_lambda_features: bool = True,
        n_lambda_bins: int = 16,
        use_isotropic_kl: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_lambda_features = use_lambda_features
        self.n_lambda_bins       = n_lambda_bins
        self.use_isotropic_kl    = use_isotropic_kl
        self.n_nodes             = n_nodes
        self.feat_dim            = feat_dim

        in_dim = input_dim + (n_lambda_bins if use_lambda_features else 0)

        # Project raw input to node-feature matrix initialisation.
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, n_nodes * feat_dim),
            nn.LayerNorm(n_nodes * feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # VDT recurrence stack.
        self.vdt = VDT(
            n_nodes=n_nodes,
            feat_dim=feat_dim,
            n_layers=n_layers,
            m_modes=m_modes,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Readout heads on the modal projection (feat_dim -> latent_dim).
        self.mu_head      = nn.Linear(feat_dim, latent_dim)
        self.log_var_head = nn.Linear(feat_dim, latent_dim)

        # Variational Gamma heads.
        self.mode_weight_head = ModeWeightHead(
            hidden_dim=feat_dim, q=latent_dim
        )

    def forward(
        self,
        x: torch.Tensor,                         # (B, D)
        L_f: torch.Tensor,                       # (B, N, N) or (N, N)
        eigvecs: torch.Tensor,                   # (N, *)
        lap: DifferentiableLaplacian,
        lambda_fp: Optional[torch.Tensor] = None,  # (B, n_lambda_bins)
    ) -> Tuple[
        torch.Tensor,  # z       (B, latent_dim)
        torch.Tensor,  # mu      (B, latent_dim)
        torch.Tensor,  # log_var (B, latent_dim)
        torch.Tensor,  # log_a   (B, latent_dim)
        torch.Tensor,  # log_b   (B, latent_dim)
    ]:
        """
        Parameters
        ----------
        x        : raw input embeddings  (B, D)
        L_f      : feature-space Laplacian  (B, N, N)
        eigvecs  : graph eigenvectors       (N, K_eig)
        lap      : DifferentiableLaplacian for CFL clamping
        lambda_fp: optional lambda-fingerprint  (B, n_lambda_bins)

        Returns
        -------
        z, mu, log_var, log_a, log_b  -- all (B, latent_dim)
        """
        # -- lambda-fingerprint concatenation (unchanged from v1) ---------
        if self.use_lambda_features and lambda_fp is not None:
            x = torch.cat([x, lambda_fp], dim=-1)   # (B, D + n_bins)

        B = x.shape[0]

        # -- Project to (B, N, d) -----------------------------------------
        X0 = self.input_proj(x)                     # (B, N*d)
        X0 = X0.view(B, self.n_nodes, self.feat_dim) # (B, N, d)

        # Expand L_f to batched if needed.
        if L_f.ndim == 2:
            L_f = L_f.unsqueeze(0).expand(B, -1, -1)

        # -- VDT recurrence -----------------------------------------------
        Q_K, _, _ = self.vdt(X0, L_f, eigvecs, lap)  # (B, N, d)

        # -- Modal projection z_modal = mean( Q_K @ U_m )  ----------------
        z_modal = self.vdt.modal_projection(Q_K, eigvecs)  # (B, d)

        # -- Readout -------------------------------------------------------
        mu      = self.mu_head(z_modal)                         # (B, latent_dim)
        log_var = self.log_var_head(z_modal).clamp(-10.0, 4.0)  # (B, latent_dim)
        z       = _reparameterise(mu, log_var)

        log_a, log_b = self.mode_weight_head(z_modal)  # (B, latent_dim) each

        return z, mu, log_var, log_a, log_b

    def kl_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Isotropic Gaussian KL fallback (used when use_isotropic_kl=True).
        The full Gamma KL from #24 should be used when use_isotropic_kl=False.
        """
        return kl_isotropic(mu, log_var, reduction)
