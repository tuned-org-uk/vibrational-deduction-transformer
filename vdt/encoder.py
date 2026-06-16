"""
Amortised encoder  q_phi(z | x).

Two encoder classes are provided:

  WiringEncoder --  VDT encoder using VibrationalStateBlock recurrence
                     and variational Gamma parameters (ModeWeightHead).

Both share the same reparameterise / kl_loss utilities.

Typical usage ()
------------------
    encoder = WiringEncoder(
        input_dim=512, latent_dim=64,
        n_nodes=128, feat_dim=32,
        n_layers=4, m_modes=16,
    )
    z, mu, log_var, log_a, log_b = encoder(
        x, L_f=L_f, eigvecs=U, lap=lap
    )

    The lambda-fingerprint is computed internally from 'lap' when
    use_lambda_features=True.  It is always derived from the fixed base
    graph topology (the frozen L(I)) -- never rebuilt from data at runtime.
    See docs/00-architecture.md, issue #34 Phase 1 guidance.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from vdt.vdt import VDT
from vdt.laplacian import DifferentiableLaplacian
from vdt.spectral import lambda_fingerprint


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
# ModeWeightHead  ( only)
# ---------------------------------------------------------------------------

class ModeWeightHead(nn.Module):
    """
    Produce variational Gamma parameters (log_a, log_b) per latent mode.

    A Gamma distribution Gamma(a, b) is parameterised through its
    log-shape log_a and log-rate log_b.  The Gamma KL replaces the
    isotropic Gaussian KL when use_isotropic_kl=False in WiringEncoder.

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
# WiringEncoder  ( VDT encoder)
# ---------------------------------------------------------------------------

class WiringEncoder(nn.Module):
    """
     encoder: VDT recurrence + ModeWeightHead.

    The encoder projects input x to an (N, d) node-feature matrix,
    runs K VDT blocks, and reads out:
      - (mu, log_var) from a linear head on the modal projection z,
      - (log_a, log_b) from ModeWeightHead for the variational Gamma KL.

    Lambda-fingerprint injection
    ----------------------------
    When use_lambda_features=True the lambda-fingerprint is computed
    INTERNALLY from the 'lap' DifferentiableLaplacian passed to forward().
    The fingerprint is derived from the fixed base graph topology (the
    frozen L(I)) and is NOT rebuilt from data at runtime.  This matches
    the architectural contract in docs/00-architecture.md and issue #34.

    input_proj is built for (input_dim + n_lambda_bins) when
    use_lambda_features=True, or for input_dim alone otherwise, so the
    projection dimension is always consistent with what forward() feeds in.

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
        If True, the lambda-fingerprint (computed from lap at forward time)
        is concatenated to x before the input projection.
    n_lambda_bins : int
        Histogram bins in the lambda-fingerprint.
    use_isotropic_kl : bool
        If True, kl_loss() uses the isotropic Gaussian KL (v1 fallback).
        If False, the caller should use the Gamma KL from issue #24.
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

        # input_proj width matches what forward() actually feeds in:
        #   D + n_lambda_bins  when use_lambda_features=True  (fingerprint always computed from lap)
        #   D                  when use_lambda_features=False
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
        L_f      : feature-space Laplacian  (B, N, N) or (N, N)
        eigvecs  : graph eigenvectors       (N, K_eig)
        lap      : DifferentiableLaplacian for CFL clamping and
                   lambda-fingerprint computation when use_lambda_features=True.
                   The fingerprint is derived from the fixed base graph
                   topology (lap.base_laplacian) -- not rebuilt from data.

        Returns
        -------
        z, mu, log_var, log_a, log_b  -- all (B, latent_dim)
        """
        B = x.shape[0]

        # -- lambda-fingerprint injection  --------------------------------
        # Compute the fingerprint from the fixed base Laplacian (L(I)) so
        # the encoder always receives a (B, D + n_bins) input when
        # use_lambda_features=True, consistent with how input_proj was built.
        if self.use_lambda_features:
            fp = lambda_fingerprint(
                lap.base_laplacian,
                tau_modes=self.n_lambda_bins,
                n_bins=self.n_lambda_bins,
            )  # (1, n_lambda_bins)
            fp = fp.expand(B, -1)                    # (B, n_lambda_bins)
            x = torch.cat([x, fp], dim=-1)           # (B, D + n_bins)

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
        The full Gamma KL from issue #24 should be used when
        use_isotropic_kl=False.
        """
        return kl_isotropic(mu, log_var, reduction)
