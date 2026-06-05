"""
Differentiable graph Laplacian builder.

Mirrors ArrowSpaceBuilder.build() logic but implemented as a PyTorch layer so
gradients flow through edge weights into L(z), enabling end-to-end training.

API
---
    lap = DifferentiableLaplacian(n_nodes, knn_k, sigma, normalised=True)
    L   = lap(edge_logits)   # shape: (B, N, N) or (N, N)

Edge weights are produced by the WiringDecoder and represent soft, learned
perturbations on top of a base kNN affinity graph.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DifferentiableLaplacian(nn.Module):
    """
    Build a (batched) normalised Laplacian from soft edge logits.

    The base graph topology is fixed (precomputed kNN edges);
    the wiring decoder predicts per-edge weight adjustments that are
    combined with the base RBF affinities via a sigmoid gate.

    Parameters
    ----------
    n_nodes : int
        Number of nodes N in the graph.
    edge_index : torch.Tensor  shape (2, E)
        Pre-computed kNN edge indices (source, target).
    base_weights : torch.Tensor  shape (E,)
        Base RBF affinities for each edge (frozen).
    normalised : bool
        If True, return D^{-1/2} (A) D^{-1/2}; else combinatorial L = D - A.
    eps : float
        Numerical stability epsilon for degree normalisation.
    """

    def __init__(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        base_weights: torch.Tensor,
        normalised: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.normalised = normalised
        self.eps = eps
        self.register_buffer("edge_index", edge_index)          # (2, E)
        self.register_buffer("base_weights", base_weights)      # (E,)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, edge_delta: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_delta : Tensor  shape (B, E) or (E,)
            Per-edge weight adjustments predicted by WiringDecoder.
            Applied as a sigmoid gate on base_weights:
                w_ij = base_w_ij * sigmoid(delta_ij)

        Returns
        -------
        L : Tensor  shape (B, N, N) or (N, N)
            Symmetric (normalised) Laplacian.
        """
        batched = edge_delta.dim() == 2
        if not batched:
            edge_delta = edge_delta.unsqueeze(0)     # (1, E)

        B, E = edge_delta.shape
        N = self.n_nodes
        src, dst = self.edge_index[0], self.edge_index[1]   # (E,)

        # Soft edge weights: base affinity gated by learned delta
        w = self.base_weights.unsqueeze(0) * torch.sigmoid(edge_delta)  # (B, E)

        # Build dense symmetric adjacency  (B, N, N)
        A = torch.zeros(B, N, N, device=edge_delta.device, dtype=edge_delta.dtype)
        A[:, src, dst] += w
        A[:, dst, src] += w          # symmetrise

        if self.normalised:
            L = self._normalised_laplacian(A)
        else:
            deg = A.sum(dim=-1)      # (B, N)
            D   = torch.diag_embed(deg)
            L   = D - A

        if not batched:
            L = L.squeeze(0)
        return L

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalised_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """Return I - D^{-1/2} A D^{-1/2} from dense adjacency A (B, N, N)."""
        deg = A.sum(dim=-1).clamp(min=self.eps)       # (B, N)
        d_inv_sqrt = deg.pow(-0.5)                     # (B, N)
        # D^{-1/2} A D^{-1/2}
        normed = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)
        I = torch.eye(self.n_nodes, device=A.device, dtype=A.dtype).unsqueeze(0)
        return I - normed

    # ------------------------------------------------------------------
    # Class-method factory — build from raw embedding matrix
    # ------------------------------------------------------------------
    @classmethod
    def from_embeddings(
        cls,
        X: torch.Tensor,
        knn_k: int = 15,
        sigma: float = 0.5,
        normalised: bool = True,
    ) -> "DifferentiableLaplacian":
        """
        Build base kNN graph from embedding matrix X (N, D),
        compute RBF affinities, and return a DifferentiableLaplacian instance.

        This mirrors ArrowSpaceBuilder.build() in pyarrowspace.
        """
        from sklearn.neighbors import NearestNeighbors
        import numpy as np

        X_np = X.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=knn_k + 1, metric="cosine").fit(X_np)
        distances, indices = nbrs.kneighbors(X_np)

        # Drop self-loops (first neighbour = self)
        distances = distances[:, 1:]
        indices   = indices[:, 1:]

        N = X_np.shape[0]
        src_list, dst_list, w_list = [], [], []
        for i in range(N):
            for j_pos, j in enumerate(indices[i]):
                d = float(distances[i, j_pos])
                w = float(np.exp(-(d ** 2) / (2 * sigma ** 2)))
                src_list.append(i)
                dst_list.append(j)
                w_list.append(w)

        edge_index   = torch.tensor([src_list, dst_list], dtype=torch.long)
        base_weights = torch.tensor(w_list, dtype=torch.float32)

        return cls(
            n_nodes=N,
            edge_index=edge_index,
            base_weights=base_weights,
            normalised=normalised,
        )
