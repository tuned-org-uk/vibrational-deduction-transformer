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
        ↓  TauModeDiffusion  →  x̂_raw  (B, D)
        ↓  optional MLP refinement
        →  x̂  (B, D)
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
        Hidden width for optional MLP refinement.
    tau_modes : int
        k — number of eigenvectors in tau-mode diffusion.
    diffusion_time : float
        Initial heat-kernel time t.
    use_mlp_refinement : bool
        If True, pass x̂_raw through a small MLP before outputting.
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
        node_idx: Optional[torch.Tensor] = None,  # (B,) query nodes
    ) -> torch.Tensor:
        """
        Returns
        -------
        x_hat : Tensor  (B, D)
        """
        x_raw = self.diffusion(L, E, node_idx=node_idx)   # (B, D)
        if self.use_mlp_refinement:
            x_raw = x_raw + self.refine_mlp(x_raw)        # residual
        return x_raw

    def recon_loss(
        self,
        x: torch.Tensor,     # (B, D)  ground-truth
        x_hat: torch.Tensor, # (B, D)  reconstruction
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Gaussian reconstruction loss:
            -log p(x|z) = ||x - x̂||^2 / (2 sigma^2) + D * log sigma
        """
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        sq_err = ((x - x_hat) ** 2).sum(dim=-1)    # (B,)
        D = x.shape[-1]
        nll = sq_err / (2 * sigma ** 2) + D * self.log_sigma
        return nll.mean() if reduction == "mean" else nll.sum()
