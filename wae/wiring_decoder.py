"""
Wiring Decoder  —  z  →  edge weight adjustments  →  Laplacian L(z).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
from .laplacian import DifferentiableLaplacian


class WiringDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_edges: int,
        hidden_dim: int,
        n_heads: int,
        laplacian: DifferentiableLaplacian,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_edges = n_edges
        self.laplacian = laplacian

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_projs = nn.ModuleList([
            nn.Linear(hidden_dim, n_edges) for _ in range(n_heads)
        ])
        self.gate = nn.Linear(hidden_dim, n_heads)

    def forward(
        self,
        z: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        gates = self.gate(h).softmax(dim=-1)
        head_deltas = torch.stack([proj(h) for proj in self.head_projs], dim=1)
        delta = (gates.unsqueeze(-1) * head_deltas).sum(dim=1)
        L = self.laplacian(delta, node_idx=node_idx) if node_idx is not None else self.laplacian(delta)
        return L, delta
