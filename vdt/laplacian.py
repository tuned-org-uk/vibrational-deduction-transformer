"""
Differentiable graph Laplacian builder.

Mirrors ArrowSpaceBuilder.build() logic but implemented as a PyTorch layer so
gradients flow through edge weights into L(z), enabling end-to-end training.

API
---
    lap = DifferentiableLaplacian(n_nodes, edge_index, base_weights)
    L   = lap(edge_delta)          # (B, N, N) dense  -- default, large graphs OOM on MPS
    L   = lap(edge_delta,
              sparse=True)         # (B, N, N) sparse COO -- safe on MPS, avoids N^2 buffer
    row = lap(edge_delta,
              node_idx=idx)        # (B, N)   -- only row i of L, no full matrix

    #  class-method factory (used by SpectralLoadingDecoder, issue #26)
    L_batch = DifferentiableLaplacian.from_spectral_loading(W, L_base)

    # Retrieve the dense (N, N) base Laplacian for passing to
    # SpectralLoadingDecoder.forward() or for spectral analysis:
    L_base = lap.base_laplacian   # cached property, (N, N)

Memory comparison on Cora (N=2708, E=40620, B=16, float32):
    dense  :  B x N^2       x 4B  =  16 x 7,333,264 x 4  ~  470 MB  (OOM on MPS)
    sparse :  B x E (COO)   x 4B  =  16 x    40,620 x 4  ~    2.6 MB
    per_node: B x N         x 4B  =  16 x     2,708 x 4  ~    0.17 MB

The sparse and per_node modes are the recommended paths for MPS / memory-
constrained devices.  The full dense path is retained for CUDA / CPU where
the N^2 buffer fits in VRAM/RAM and the eigensolver is faster on dense input.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn


class MassMatrix:
    """
    Diagonal mass matrix derived from graph-Laplacian eigenvalues.

    The diagonal entries are defined as:

        M_ii = ( 1 - lambda_i^tau + eps )^{-1}

    where lambda_i are eigenvalues of the graph Laplacian, tau is a
    smoothing exponent, and eps guards against division by zero.

    A conditioning warning is emitted when max(M_ii)/min(M_ii) > 100
    (per docs//04-stability.md section 7).

    Parameters
    ----------
    eigenvalues : Tensor  shape (N,)
        Eigenvalues of the graph Laplacian, sorted ascending.
    tau : float
        Smoothing exponent (typically 0.5).
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        tau: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        self.eigenvalues = eigenvalues
        self.tau = tau
        self.eps = eps
        self._M_diag: Optional[torch.Tensor] = None

    @property
    def M_diag(self) -> torch.Tensor:
        """
        Diagonal of the mass matrix.  Shape (N,).  All entries > 0.

        Computed lazily and cached; set _M_diag = None to invalidate.
        Emits RuntimeWarning when max/min ratio exceeds 100.
        """
        if self._M_diag is not None:
            return self._M_diag

        lam = self.eigenvalues.clamp(min=0.0)
        denom = (1.0 - lam.pow(self.tau) + self.eps).abs().clamp(min=self.eps)
        M = denom.reciprocal()

        ratio = float(M.max() / M.clamp(min=self.eps).min())
        if ratio > 100.0:
            warnings.warn(
                f"MassMatrix conditioning ratio {ratio:.1f} > 100. "
                "The Laplacian may be poorly conditioned (see docs//04-stability.md S7).",
                RuntimeWarning,
                stacklevel=2,
            )

        self._M_diag = M
        return M

    def as_matrix(self) -> torch.Tensor:
        """Return the full diagonal matrix.  Shape (N, N)."""
        return torch.diag(self.M_diag)


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
        (B, N, N) dense adjacency -- use sparse COO instead.
    eps : float
        Numerical stability epsilon for degree normalisation.

    Attributes
    ----------
    base_laplacian : torch.Tensor  shape (N, N)
        Dense normalised symmetric Laplacian built from edge_index and
        base_weights (no edge delta applied).  Computed lazily on first
        access and cached as a plain tensor.  Use this to supply L_base
        to SpectralLoadingDecoder.forward() and from_spectral_loading().
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
        self.n_nodes = n_nodes
        self.normalised = normalised
        self.sparse = sparse
        self.eps = eps
        self.register_buffer("edge_index", edge_index)    # (2, E)
        self.register_buffer("base_weights", base_weights)  # (E,)
        self._lambda_max: Optional[float] = None
        self._base_laplacian: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # base_laplacian property
    # ------------------------------------------------------------------
    @property
    def base_laplacian(self) -> torch.Tensor:
        """
        Dense normalised symmetric Laplacian (N, N) built from the frozen
        edge_index and base_weights (no per-edge delta applied).

        Computed lazily on first access and cached in _base_laplacian.
        Call _invalidate_spectral_cache() to force recomputation after
        any manual update to base_weights.

        This is the tensor to pass as L_base to
        SpectralLoadingDecoder.forward() and
        DifferentiableLaplacian.from_spectral_loading().

        Returns
        -------
        torch.Tensor  shape (N, N), dtype float32
        """
        if self._base_laplacian is not None:
            return self._base_laplacian

        N = self.n_nodes
        src, dst = self.edge_index[0], self.edge_index[1]
        w = self.base_weights
        device, dtype = w.device, w.dtype

        A = torch.zeros(N, N, device=device, dtype=dtype)
        A[src, dst] += w
        A[dst, src] += w

        if self.normalised:
            deg = A.sum(dim=-1).clamp(min=self.eps)
            d_inv_sqrt = deg.pow(-0.5)
            normed = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)
            I = torch.eye(N, device=device, dtype=dtype)
            L = I - normed
        else:
            deg = A.sum(dim=-1)
            L = torch.diag(deg) - A

        self._base_laplacian = L
        return L

    # ------------------------------------------------------------------
    #  class-method factory -- used by SpectralLoadingDecoder (#26)
    # ------------------------------------------------------------------
    @classmethod
    def from_spectral_loading(
        cls,
        W: torch.Tensor,
        L_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Synthesise a per-batch differentiable Laplacian from a spectral
        loading matrix.  Fully differentiable through W so gradients flow
        back into SpectralLoadingDecoder.

        The per-batch edge weight update rule is:

            w_ij^{(b)} = base_w_ij * sigmoid( -||W_i^{(b)} - W_j^{(b)}||^2 )

        where base_w_ij is recovered from the off-diagonal of L_base.
        In the standard case d == N.

        Parameters
        ----------
        W : Tensor  shape (B, d, q)
            Spectral loading matrix from the decoder.  d must equal N.
        L_base : Tensor  shape (N, N)
            Frozen base Laplacian encoding the graph topology.
            Obtain via DifferentiableLaplacian.base_laplacian.

        Returns
        -------
        L : Tensor  shape (B, N, N)
            Per-batch normalised symmetric Laplacian.  Gradient flows
            through all operations back to W.

        Notes
        -----
        Zero row-sum and non-positive off-diagonal are guaranteed by
        construction of the normalised symmetric Laplacian.
        """
        B, d, q = W.shape
        N = L_base.shape[0]
        assert d == N, (
            f"from_spectral_loading: W.shape[1]={d} must equal L_base.shape[0]={N}."
        )
        device = W.device
        dtype = W.dtype
        eps = 1e-6

        # Recover base affinities from the off-diagonal of L_base.
        # For a normalised symmetric Laplacian, off-diagonal entries are
        # -D^{-1/2} A D^{-1/2}, so base affinities are their negations.
        eye = torch.eye(N, device=device, dtype=dtype)
        A_base = (-L_base * (1.0 - eye)).clamp(min=0.0)  # (N, N)

        # Pairwise squared distance in loading space: (B, N, N)
        # W: (B, N, q)  --  ||W_i - W_j||^2
        diff = W.unsqueeze(2) - W.unsqueeze(1)    # (B, N, N, q)
        sq_dist = (diff ** 2).sum(dim=-1)          # (B, N, N)

        # Soft gate in (0, 1): nearby nodes keep their base affinity
        gate = torch.sigmoid(-sq_dist)             # (B, N, N)

        # Updated adjacency, symmetrised
        A_updated = A_base.unsqueeze(0) * gate                         # (B, N, N)
        A_sym = 0.5 * (A_updated + A_updated.transpose(-1, -2))        # (B, N, N)

        # Build normalised symmetric Laplacian: I - D^{-1/2} A D^{-1/2}
        deg = A_sym.sum(dim=-1).clamp(min=eps)   # (B, N)
        d_inv_sqrt = deg.pow(-0.5)               # (B, N)
        normed = (
            d_inv_sqrt.unsqueeze(-1) * A_sym * d_inv_sqrt.unsqueeze(-2)
        )                                        # (B, N, N)
        I = eye.unsqueeze(0).expand(B, N, N)
        L = I - normed                           # (B, N, N)
        return L

    # ------------------------------------------------------------------
    # Spectral properties -- CFL helpers
    # ------------------------------------------------------------------
    @property
    def lambda_max(self) -> float:
        """
        Largest eigenvalue of the base Laplacian.

        Delegates to base_laplacian so the dense matrix is built only
        once and shared with SpectralLoadingDecoder.  Result is cached;
        call _invalidate_spectral_cache() after any edge-weight update.

        Returns
        -------
        float
            Largest eigenvalue lambda_max >= 0.
        """
        if self._lambda_max is not None:
            return self._lambda_max

        with torch.no_grad():
            eigs = torch.linalg.eigvalsh(self.base_laplacian)  # ascending, real
        self._lambda_max = float(eigs[-1].clamp(min=0.0))
        return self._lambda_max

    def _invalidate_spectral_cache(self) -> None:
        """Invalidate cached lambda_max and base_laplacian; both are recomputed on next access."""
        self._lambda_max = None
        self._base_laplacian = None

    def dt_max_cfl(self, safety: float = 1.0) -> float:
        """
        CFL-stable maximum time step for graph-diffusion integration.

        Defined as:

            dt_max = safety * sqrt( 2 / lambda_max )

        Parameters
        ----------
        safety : float
            Safety factor in (0, 1].  Use 0.9 for a 10 percent margin.

        Returns
        -------
        float
            Maximum stable time step.
        """
        lam = max(self.lambda_max, 1e-8)
        return safety * math.sqrt(2.0 / lam)

    # ------------------------------------------------------------------
    # Rayleigh quotient
    # ------------------------------------------------------------------
    def rayleigh_quotient(
        self,
        z: torch.Tensor,
        mass: Optional[MassMatrix] = None,
        eigenvalues: Optional[torch.Tensor] = None,
        tau: float = 0.5,
    ) -> torch.Tensor:
        """
        Generalised Rayleigh quotient  z^T L_f z / z^T M z.

        Used as a regulariser in Options 1-4 training objectives.  The
        return value is a non-negative scalar for any real z.

        Parameters
        ----------
        z : Tensor  shape (N,) or (B, N)
            Node embedding vector(s).
        mass : MassMatrix or None
            Pre-built mass matrix.  When None and eigenvalues is also None,
            identity mass is used (standard Rayleigh quotient z^T L z / z^T z).
        eigenvalues : Tensor  shape (N,) or None
            Eigenvalues used to build a MassMatrix on the fly.  Ignored
            when mass is provided.
        tau : float
            Tau exponent passed to MassMatrix when built on the fly.

        Returns
        -------
        Tensor  scalar >= 0
        """
        batched = z.dim() == 2
        if not batched:
            z = z.unsqueeze(0)  # (1, N)

        B, N = z.shape
        device, dtype = z.device, z.dtype

        L_f = self.base_laplacian.to(device=device, dtype=dtype)  # (N, N)

        # Mass matrix diagonal
        if mass is not None:
            M_diag = mass.M_diag.to(device=device, dtype=dtype)
        elif eigenvalues is not None:
            mm = MassMatrix(eigenvalues.to(device=device, dtype=dtype), tau=tau)
            M_diag = mm.M_diag
        else:
            M_diag = torch.ones(N, device=device, dtype=dtype)

        # Numerator: z^T L_f z; denominator: z^T M z
        Lz = (L_f.unsqueeze(0) @ z.unsqueeze(-1)).squeeze(-1)   # (B, N)
        numerator = (z * Lz).sum(dim=-1)                          # (B,)
        denominator = (z * (M_diag * z)).sum(dim=-1).clamp(min=self.eps)
        rq = (numerator / denominator).mean()
        return rq.clamp(min=0.0)

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
            Laplacian to avoid the N^2 intermediate buffer.
        node_idx : Tensor  shape (B,) or None
            If given, return only row node_idx[b] of L for each batch
            element: shape (B, N).

        Returns
        -------
        L : Tensor
            shape (B, N, N)  -- full Laplacian  (dense or built-sparse)
            shape (B, N)     -- single row per batch element (node_idx mode)
            shape (N, N)     -- unbatched (when input edge_delta is 1-D)
        """
        use_sparse = self.sparse if sparse is None else sparse

        batched = edge_delta.dim() == 2
        if not batched:
            edge_delta = edge_delta.unsqueeze(0)

        B = edge_delta.shape[0]
        N = self.n_nodes
        device = edge_delta.device
        dtype = edge_delta.dtype
        src, dst = self.edge_index[0], self.edge_index[1]

        w = self.base_weights.unsqueeze(0) * torch.sigmoid(edge_delta)

        if node_idx is not None:
            return self._row_laplacian(w, node_idx, N, B, src, dst, device, dtype)

        if use_sparse:
            return self._sparse_laplacian(w, N, B, src, dst, device, dtype)

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
            L = torch.diag_embed(deg) - A
        return L.squeeze(0) if not batched else L

    # ------------------------------------------------------------------
    # Sparse COO path (MPS / memory-safe)
    # ------------------------------------------------------------------
    def _sparse_laplacian(
        self, w, N, B, src, dst, device, dtype
    ) -> torch.Tensor:
        """
        Build L batch-by-batch using sparse COO tensors so the peak
        allocation is O(E) not O(N^2).  Each element is densified only
        after the Laplacian transform, keeping the dense block to (N, N)
        per element rather than (B, N, N) simultaneously.
        """
        Ls = []
        for b in range(B):
            wb = w[b]
            idx_r = torch.cat([src, dst])
            idx_c = torch.cat([dst, src])
            vals = torch.cat([wb, wb])
            indices = torch.stack([idx_r, idx_c])

            A_sp = torch.sparse_coo_tensor(
                indices, vals, (N, N), device=device, dtype=dtype
            ).coalesce()
            A_dense = A_sp.to_dense()

            if self.normalised:
                L_b = self._normalised_laplacian_single(A_dense)
            else:
                deg = A_dense.sum(dim=-1)
                L_b = torch.diag(deg) - A_dense
            Ls.append(L_b)

        return torch.stack(Ls, dim=0)  # (B, N, N)

    # ------------------------------------------------------------------
    # Per-node row path (most memory-efficient)
    # ------------------------------------------------------------------
    def _row_laplacian(
        self, w, node_idx, N, B, src, dst, device, dtype
    ) -> torch.Tensor:
        """
        Return only row node_idx[b] of L for each b.  Shape (B, N).
        Peak allocation: O(B * deg(i)).
        """
        rows = []
        for b in range(B):
            wb = w[b]
            i = node_idx[b].item()

            a_row = torch.zeros(N, device=device, dtype=dtype)
            mask_src = src == i
            a_row.scatter_add_(0, dst[mask_src], wb[mask_src])
            mask_dst = dst == i
            a_row.scatter_add_(0, src[mask_dst], wb[mask_dst])

            if self.normalised:
                deg_i = a_row.sum().clamp(min=self.eps)
                deg_all = torch.zeros(N, device=device, dtype=dtype)
                deg_all.scatter_add_(0, dst, wb)
                deg_all.scatter_add_(0, src, wb)
                deg_all = deg_all.clamp(min=self.eps)

                d_inv_sqrt_i = deg_i.pow(-0.5)
                d_inv_sqrt_all = deg_all.pow(-0.5)

                normed_row = d_inv_sqrt_i * a_row * d_inv_sqrt_all
                l_row = -normed_row
                l_row[i] = l_row[i] + 1.0
            else:
                deg_i = a_row.sum()
                l_row = -a_row
                l_row[i] = deg_i

            rows.append(l_row)

        return torch.stack(rows, dim=0)  # (B, N)

    # ------------------------------------------------------------------
    # Laplacian helpers
    # ------------------------------------------------------------------
    def _normalised_laplacian_dense(self, A: torch.Tensor) -> torch.Tensor:
        """I - D^{-1/2} A D^{-1/2}  for batched dense A (B, N, N)."""
        deg = A.sum(dim=-1).clamp(min=self.eps)
        d_inv_sqrt = deg.pow(-0.5)
        normed = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)
        I = torch.eye(self.n_nodes, device=A.device, dtype=A.dtype).unsqueeze(0)
        return I - normed

    def _normalised_laplacian_single(self, A: torch.Tensor) -> torch.Tensor:
        """I - D^{-1/2} A D^{-1/2}  for single dense A (N, N)."""
        deg = A.sum(dim=-1).clamp(min=self.eps)
        d_inv_sqrt = deg.pow(-0.5)
        normed = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)
        I = torch.eye(self.n_nodes, device=A.device, dtype=A.dtype)
        return I - normed

    # ------------------------------------------------------------------
    # Class-method factory -- build from raw embedding matrix
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
        indices = indices[:, 1:]

        N = X_np.shape[0]
        src_list, dst_list, w_list = [], [], []
        for i in range(N):
            for j_pos, j in enumerate(indices[i]):
                d = float(distances[i, j_pos])
                w = float(np.exp(-(d ** 2) / (2 * sigma ** 2)))
                src_list.append(i)
                dst_list.append(j)
                w_list.append(w)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        base_weights = torch.tensor(w_list, dtype=torch.float32)

        return cls(
            n_nodes=N,
            edge_index=edge_index,
            base_weights=base_weights,
            normalised=normalised,
            sparse=sparse,
        )
