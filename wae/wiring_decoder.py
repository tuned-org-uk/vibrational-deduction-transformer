"""
Wiring Decoder  —  z  →  edge weight adjustments  →  Laplacian L(z).

This is the key architectural novelty: the latent code z controls *how*
the graph is wired, not directly what the output is.

The decoder uses a mixture-of-experts head over n_heads base templates
(learnable prototypical edge patterns) to produce per-edge delta weights.
This gives the latent space a natural "wiring mode" interpretation:
each mixture head represents a distinct topological pattern.

Architecture
------------
    z  (B, latent_dim)
    ↓  Linear projection + GELU
    ↓  heads  (B, n_heads, E)
    ↓  softmax mixing  →  edge_delta  (B, E)
    ↓  DifferentiableLaplacian  →  L(z)  (B, N, N)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .laplacian import DifferentiableLaplacian


class WiringDecoder(nn.Module):
    """
    Parameters
    ----------
    latent_dim : int
        Dimension of z.
    n_edges : int
        Number of edges E in the base kNN graph.
    hidden_dim : int
        Hidden width.
    n_heads : int
        Number of mixture heads (each head learns a full edge-weight template).
    laplacian : DifferentiableLaplacian
        Pre-built differentiable Laplacian module (frozen topology).
    """

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

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # One head per mixture component: outputs per-edge delta logits
        self.head_projs = nn.ModuleList([
            nn.Linear(hidden_dim, n_edges) for _ in range(n_heads)
        ])
        # Mixing gate: predicts softmax weights over heads
        self.gate = nn.Linear(hidden_dim, n_heads)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : Tensor  shape (B, latent_dim)

        Returns
        -------
        L     : Tensor  (B, N, N)   learned Laplacian
        delta : Tensor  (B, E)      edge weight deltas (for diagnostics)
        """
        h = self.trunk(z)                                      # (B, H)
        gates = self.gate(h).softmax(dim=-1)                  # (B, n_heads)

        # Mixture of head deltas
        head_deltas = torch.stack(
            [proj(h) for proj in self.head_projs], dim=1
        )                                                      # (B, n_heads, E)
        delta = (gates.unsqueeze(-1) * head_deltas).sum(dim=1)  # (B, E)

        L = self.laplacian(delta)                              # (B, N, N)
        return L, delta
