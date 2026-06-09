"""
Diffusion Decoder  —  L(z), E  →  x̂.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .spectral import TauModeDiffusion
from typing import Optional


class DiffusionDecoder(nn.Module):
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
        L: torch.Tensor,
        E: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
        eig_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x_raw = self.diffusion(L, E, node_idx=node_idx, eig_cache=eig_cache)
        if self.use_mlp_refinement and node_idx is not None:
            x_raw = x_raw + self.refine_mlp(x_raw)
        return x_raw

    def recon_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if x_hat.dim() != 2:
            raise ValueError(
                "recon_loss expects per-node x_hat with shape (B, D). "
                f"Got shape {tuple(x_hat.shape)}. "
                "Pass node_idx to forward() when computing training loss."
            )
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        sq_err = ((x - x_hat) ** 2).sum(dim=-1)
        D = x.shape[-1]
        nll = sq_err / (2 * sigma ** 2) + D * self.log_sigma
        return nll.mean() if reduction == "mean" else nll.sum()
