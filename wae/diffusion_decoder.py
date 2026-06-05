"""
Diffusion Decoder  —  L(z), E  →  x̂.

Given the learned Laplacian L(z) and a fixed embedding table E (N, D),
produces reconstructed embeddings by tau-mode diffusion.

The Gaussian likelihood is:
    log p(x | z) = -||x - x̂||^2 / (2 * sigma^2)

where sigma is a learnable scalar.

Architecture
------------
    L(z)  [shape (B, N, N)]  +  E [shape (N, D)]
        ↓  TauModeDiffusion  →  x̂_raw
        ↓  MLP refinement (per-node mode only, i.e. when node_idx is given)
        →  x̂

Output shape contract
---------------------
    node_idx given  →  x̂ has shape (B, D)      — per-node reconstruction
    node_idx=None   →  x̂ has shape (B, N, D)   — full-graph reconstruction
                       MLP refinement is NOT applied in this case because the
                       MLP expects (B, D) input. Callers that need per-node
                       semantics must always provide node_idx.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .spectral import TauModeDiffusion
from typing import Optional


class DiffusionDecoder(nn.Module):
    """
    Parameters
    ----------
    embedding_dim : int
        D — dimension of E and of x.
    hidden_dim : int
        Hidden width for optional MLP refinement (per-node mode only).
    tau_modes : int
        k — number of eigenvectors in tau-mode diffusion.
    diffusion_time : float
        Initial heat-kernel time t.
    use_mlp_refinement : bool
        If True, pass x̂_raw through a small MLP before outputting.
        The MLP is only applied when node_idx is provided (per-node mode).
        In full-graph mode (node_idx=None) it is always skipped.
    init_log_sigma : float
        Log of initial noise std for Gaussian likelihood.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        tau_modes: int = 16,
        diffusion_time: float = 1.0,
        use_mlp_refinement: bool = True,
        init_log_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        self.diffusion = TauModeDiffusion(
            tau_modes=tau_modes,
            diffusion_time=diffusion_time,
            learnable_time=True,
        )
        self.use_mlp_refinement = use_mlp_refinement
        if use_mlp_refinement:
            self.refine_mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma))

    def forward(
        self,
        L: torch.Tensor,          # (B, N, N)
        E: torch.Tensor,          # (N, D)
        node_idx: Optional[torch.Tensor] = None,  # (B,) → per-node; None → full-graph
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        L : Tensor  (B, N, N)   batch of Laplacians
        E : Tensor  (N, D)      embedding table (shared across batch)
        node_idx : Tensor or None
            Long tensor of shape (B,) selecting one node per sample.
            When provided the output is (B, D) and the MLP refinement
            is applied.  When None the output is (B, N, D) and the MLP
            is skipped (MLP is per-node only).

        Returns
        -------
        x_hat : Tensor
            Shape (B, D)    when node_idx is not None  (per-node mode)
            Shape (B, N, D) when node_idx is None      (full-graph mode)
        """
        x_raw = self.diffusion(L, E, node_idx=node_idx)
        # MLP refinement is per-node only: it expects (B, D) input.
        # Guard explicitly so full-graph callers never hit a shape error.
        if self.use_mlp_refinement and node_idx is not None:
            x_raw = x_raw + self.refine_mlp(x_raw)        # residual
        return x_raw

    def recon_loss(
        self,
        x: torch.Tensor,     # (B, D)  ground-truth
        x_hat: torch.Tensor, # (B, D)  reconstruction  (per-node mode only)
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Gaussian reconstruction loss (per-node mode only)::

            -log p(x|z) = ||x - x_hat||^2 / (2 sigma^2) + D * log sigma

        x_hat must be (B, D).  Do not call this in full-graph mode.
        """
        if x_hat.dim() != 2:
            raise ValueError(
                "recon_loss expects per-node x_hat with shape (B, D). "
                f"Got shape {tuple(x_hat.shape)}. "
                "Pass node_idx to forward() when computing training loss."
            )
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        sq_err = ((x - x_hat) ** 2).sum(dim=-1)    # (B,)
        D = x.shape[-1]
        nll = sq_err / (2 * sigma ** 2) + D * self.log_sigma
        return nll.mean() if reduction == "mean" else nll.sum()
