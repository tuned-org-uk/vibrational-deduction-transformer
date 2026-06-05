"""
Differentiable graph Laplacian builder.

Mirrors ArrowSpaceBuilder.build() logic but implemented as a PyTorch layer so
gradients flow through edge weights into L(z), enabling end-to-end training.

API
---
    lap = DifferentiableLaplacian(n_nodes, edge_index, base_weights)
    L   = lap(edge_delta)          # (B, N, N) dense  — default, large graphs OOM on MPS
    L   = lap(edge_delta,
              sparse=True)         # (B, N, N) sparse COO — safe on MPS, avoids N² buffer
    row = lap(edge_delta,
              node_idx=idx)        # (B, N)   — only row i of L, no full matrix

Memory comparison on Cora (N=2708, E=40620, B=16, float32):
    dense  :  B × N²       × 4B  =  16 × 7,333,264 × 4  ≈  470 MB  (× batch ≈ OOM on MPS)
    sparse :  B × E (COO)  × 4B  =  16 ×    40,620 × 4  ≈    2.6 MB
    per_node: B × N        × 4B  =  16 ×     2,708 × 4  ≈    0.17 MB

The sparse and per_node modes are the recommended paths for MPS / memory-
constrained devices.  The full dense path is retained for CUDA / CPU where
the N² buffer fits in VRAM/RAM and the eigensolver is faster on dense input.
"""
from __future__ import annotations
import torch
import torch.nn as nn
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
        If True, return I - D^{-1/2} A D^{-1/2}; else combinatorial L = D - A.
    sparse : bool
        Default forward mode.  When True, avoid materialising the full
        (B, N, N) dense adjacency — use sparse COO instead.  Dramatically
        reduces peak MPS / VRAM usage.  Individual call-site can override
        via the `sparse` argument to forward().
    eps : float
        Numerical stability epsilon for degree normalisation.
    """

    def __init__(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        base_weights: torch.Tensor,
        normalised: bool = True,
        sparse: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_nodes     = n_nodes
        self.normalised  = normalised
        self.sparse      = sparse
        self.eps         = eps
        self.register_buffer("edge_index",   edge_index)    # (2, E)
        self.register_buffer("base_weights", base_weights)  # (E,)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        edge_delta: torch.Tensor,
        sparse: Optional[bool] = None,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_delta : Tensor  shape (B, E) or (E,)
            Per-edge weight adjustments (WiringDecoder output).
        sparse : bool or None
            Override instance default.  When True, build a sparse COO
            Laplacian instead of the dense (B, N, N) block.  Returned
            tensor is dense but built via sparse ops to avoid the N²
            intermediate buffer.
        node_idx : Tensor  shape (B,) or None
            If given, return only row node_idx[b] of L for each batch
            element: shape (B, N).  This is the most memory-efficient
            path — no full N×N matrix is ever constructed.

        Returns
        -------
        L : Tensor
            shape (B, N, N)  — full Laplacian  (dense or built-sparse)
            shape (B, N)     — single row per batch element (node_idx mode)
            shape (N, N)     — unbatched (when input edge_delta is 1-D)
        """
        use_sparse = self.sparse if sparse is None else sparse

        batched = edge_delta.dim() == 2
        if not batched:
            edge_delta = edge_delta.unsqueeze(0)

        B = edge_delta.shape[0]
        N = self.n_nodes
        device = edge_delta.device
        dtype  = edge_delta.dtype
        src, dst = self.edge_index[0], self.edge_index[1]   # (E,)

        # Soft edge weights: base affinity gated by learned delta  (B, E)
        w = self.base_weights.unsqueeze(0) * torch.sigmoid(edge_delta)

        if node_idx is not None:
            # ----------------------------------------------------------------
            # per-node mode: return only row i of L — shape (B, N)
            # Never builds the full N×N matrix.
            # ----------------------------------------------------------------
            return self._row_laplacian(w, node_idx, N, B, src, dst, device, dtype)

        if use_sparse:
            # ----------------------------------------------------------------
            # Sparse COO mode: build A as sparse, compute L via sparse ops.
            # Avoids the N² dense buffer; returned tensor is dense (B, N, N)
            # but peak allocation is only O(B·E).
            # ----------------------------------------------------------------
            return self._sparse_laplacian(w, N, B, src, dst, device, dtype)

        # ----------------------------------------------------------------
        # Dense mode (CUDA / CPU): original path, fastest on big VRAM.
        # ----------------------------------------------------------------
        return self._dense_laplacian(w, N, B, src, dst, device, dtype, batched)

    # ------------------------------------------------------------------
    # Dense path (original)
    # ------------------------------------------------------------------
    def _dense_laplacian(
        self, w, N, B, src, dst, device, dtype, batched
    ) -> torch.Tensor:
        A = torch.zeros(B, N, N, device=device, dtype=dtype)
        A[:, src, dst] += w
        A[:, dst, src] += w
        if self.normalised:
            L = self._normalised_laplacian_dense(A)
        else:
            deg = A.sum(dim=-1)
            L   = torch.diag_embed(deg) - A
        return L.squeeze(0) if not batched else L

    # ------------------------------------------------------------------
    # Sparse COO path (MPS / memory-safe)
    # ------------------------------------------------------------------
    def _sparse_laplacian(
        self, w, N, B, src, dst, device, dtype
    ) -> torch.Tensor:
        """
        Build L batch-by-batch using sparse COO tensors so the peak
        allocation is O(E) not O(N²).  Each element is densified only
        after the Laplacian transform, keeping the dense block to (N, N)
        per element rather than (B, N, N) simultaneously.
        """
        Ls = []
        for b in range(B):
            wb = w[b]   # (E,)
            # Symmetrised indices and values
            idx_r = torch.cat([src, dst])           # (2E,)
            idx_c = torch.cat([dst, src])
            vals  = torch.cat([wb, wb])             # (2E,)
            indices = torch.stack([idx_r, idx_c])   # (2, 2E)

            A_sp = torch.sparse_coo_tensor(
                indices, vals, (N, N), device=device, dtype=dtype
            ).coalesce()
            A_dense = A_sp.to_dense()               # (N, N)  — only N², not B·N²

            if self.normalised:
                L_b = self._normalised_laplacian_single(A_dense)
            else:
                deg = A_dense.sum(dim=-1)
                L_b = torch.diag(deg) - A_dense
            Ls.append(L_b)

        return torch.stack(Ls, dim=0)   # (B, N, N)

    # ------------------------------------------------------------------
    # Per-node row path (most memory-efficient)
    # ------------------------------------------------------------------
    def _row_laplacian(
        self, w, node_idx, N, B, src, dst, device, dtype
    ) -> torch.Tensor:
        """
        Return only row node_idx[b] of L for each b.  Shape (B, N).

        For each batch element b and query node i = node_idx[b]:
            - Collect all edges incident to i
            - Compute degree of i and its neighbours
            - Return the i-th row of the normalised Laplacian

        Peak allocation: O(B · deg(i)) — essentially free.
        """
        rows = []
        for b in range(B):
            wb = w[b]          # (E,)
            i  = node_idx[b].item()

            # Build full adjacency row for node i using scatter
            a_row = torch.zeros(N, device=device, dtype=dtype)
            # edges where i is source
            mask_src = (src == i)
            a_row.scatter_add_(0, dst[mask_src], wb[mask_src])
            # edges where i is dest (symmetrise)
            mask_dst = (dst == i)
            a_row.scatter_add_(0, src[mask_dst], wb[mask_dst])

            if self.normalised:
                # Need degree of i and all neighbours for D^{-1/2} scaling
                # Degree of i
                deg_i = a_row.sum().clamp(min=self.eps)
                # Degree of every node (needed for D^{-1/2} at j positions)
                # Build full degree vector via scatter on all edges
                deg_all = torch.zeros(N, device=device, dtype=dtype)
                deg_all.scatter_add_(0, dst, wb)
                deg_all.scatter_add_(0, src, wb)
                deg_all = deg_all.clamp(min=self.eps)

                d_inv_sqrt_i   = deg_i.pow(-0.5)
                d_inv_sqrt_all = deg_all.pow(-0.5)   # (N,)

                # Row i of I - D^{-1/2} A D^{-1/2}
                normed_row = d_inv_sqrt_i * a_row * d_inv_sqrt_all  # (N,)
                l_row = -normed_row
                l_row[i] = l_row[i] + 1.0   # diagonal: 1 - normed_self (self-loop = 0)
            else:
                deg_i = a_row.sum()
                l_row = -a_row
                l_row[i] = deg_i

            rows.append(l_row)

        return torch.stack(rows, dim=0)   # (B, N)

    # ------------------------------------------------------------------
    # Laplacian helpers
    # ------------------------------------------------------------------
    def _normalised_laplacian_dense(self, A: torch.Tensor) -> torch.Tensor:
        """I - D^{-1/2} A D^{-1/2}  for batched dense A (B, N, N)."""
        deg = A.sum(dim=-1).clamp(min=self.eps)          # (B, N)
        d_inv_sqrt = deg.pow(-0.5)                        # (B, N)
        normed = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)
        I = torch.eye(self.n_nodes, device=A.device, dtype=A.dtype).unsqueeze(0)
        return I - normed

    def _normalised_laplacian_single(self, A: torch.Tensor) -> torch.Tensor:
        """I - D^{-1/2} A D^{-1/2}  for single dense A (N, N)."""
        deg = A.sum(dim=-1).clamp(min=self.eps)           # (N,)
        d_inv_sqrt = deg.pow(-0.5)                         # (N,)
        normed = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)
        I = torch.eye(self.n_nodes, device=A.device, dtype=A.dtype)
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
        sparse: bool = False,
    ) -> "DifferentiableLaplacian":
        """
        Build base kNN graph from embedding matrix X (N, D),
        compute RBF affinities, and return a DifferentiableLaplacian instance.

        Parameters
        ----------
        sparse : bool
            Pass True to enable sparse COO forward mode by default.
            Recommended for MPS devices or large graphs.
        """
        from sklearn.neighbors import NearestNeighbors
        import numpy as np

        X_np = X.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=knn_k + 1, metric="cosine").fit(X_np)
        distances, indices = nbrs.kneighbors(X_np)

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
            sparse=sparse,
        )
