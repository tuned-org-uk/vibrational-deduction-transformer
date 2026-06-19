"""
Differentiable graph Laplacian builder.

Mirrors ArrowSpaceBuilder.build() logic but implemented as a PyTorch layer so
gradients flow through edge weights into L(z), enabling end-to-end training.

API
---
    lap = DifferentiableLaplacian(n_nodes, edge_index, base_weights)
    L   = lap(edge_delta)          # (B, N, N) dense  -- default
    L   = lap(edge_delta,
              sparse=True)         # (B, N, N) sparse COO -- safe on MPS
    row = lap(edge_delta,
              node_idx=idx)        # (B, N)   -- only row i of L

    #  class-method factories
    L_batch = DifferentiableLaplacian.from_spectral_loading(W, L_base)
    lap     = DifferentiableLaplacian.from_embeddings(embeddings, cfg)

    # Base Laplacian (N, N) for spectral analysis:
    L_base = lap.base_laplacian   # cached property

Memory comparison on Cora (N=2708, E=40620, B=16, float32):
    dense  :  B x N^2       x 4B  =  16 x 7,333,264 x 4  ~  470 MB  (OOM on MPS)
    sparse :  B x E (COO)   x 4B  =  16 x    40,620 x 4  ~    2.6 MB
    per_node: B x N         x 4B  =  16 x     2,708 x 4  ~    0.17 MB

The sparse and per_node modes are the recommended paths for MPS / memory-
constrained devices.  The full dense path is retained for CUDA / CPU where
the N^2 buffer fits in VRAM/RAM and the eigensolver is faster on dense input.

Density-matrix extension
------------------------
The graph Laplacian in feature space is enhanced to represent the density
matrix of positive-negative probability (SignedDensityMatrix in vdt/density.py).
The spectral loading factory (from_spectral_loading) projects the base Laplacian
into the subspace spanned by the loading matrix W, yielding a batched PSD
Laplacian whose off-diagonal entries are non-positive.

Rayleigh Theory connection
--------------------------
For a vibrational system the Rayleigh quotient R(z) = z^T L z / z^T M z
where M is the diagonal mass matrix (see MassMatrix) bounds the natural
frequency from below (min eigenvalue) and above (max eigenvalue).  The
CFL condition dt <= sqrt(2 / lambda_max) ensures numerical stability of
the discrete wave propagation.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MassMatrix
# ---------------------------------------------------------------------------

class MassMatrix:
    """
    Diagonal mass matrix derived from graph-Laplacian eigenvalues.

    The diagonal entries are defined as::

        M_ii = ( 1 - lambda_i^tau + eps )^{-1}

    where lambda_i are eigenvalues of the graph Laplacian, tau is a
    smoothing exponent, and eps guards against division by zero.

    A conditioning warning is emitted when max(M_ii)/min(M_ii) > 100
    (per docs/04-stability.md section 7).

    Singularity at lambda = 1
    -------------------------
    For a normalised symmetric Laplacian (eigenvalues in [0, 2]), the
    Rayleigh-damping mass M = 1 / (1 - lambda^tau) diverges at lambda = 1
    for all tau.  This is the mode at its natural frequency where kinetic
    and potential energy are in equipartition.

    In practice, eps (default 1e-6) prevents division by zero but leaves
    M very large (~1/eps = 10^6) near lambda = 1.  This can:

    - Cause the conditioning ratio in pre_training_checks (level 3) to
      exceed 100 and emit a spurious warning for any graph with nodes near
      lambda = 1 (common for regular graphs and k-NN graphs on uniform
      point clouds).
    - Create numerical instability in the Tikhonov preconditioner
      H_prec = sigma * M_diag * I + L_f inside log_preconditioner_stability.
    - Allow the time step dt to be dominated by the mass spike rather than
      the true spectral structure.

    Use the ``mass_clip`` parameter to clamp M_diag to a finite maximum::

        MassMatrix(tau=0.5, eps=1e-6, mass_clip=1e3)

    This prevents spurious conditioning warnings and preconditioner
    instability without materially affecting modes far from lambda = 1.
    Recommended values: mass_clip=1e3 (moderate graphs), mass_clip=1e4
    (sparse graphs where the singularity is rarely excited).

    Parameters
    ----------
    eigenvalues : Tensor  shape (N,)
        Eigenvalues of the graph Laplacian, sorted ascending.
    tau : float
        Smoothing exponent (typically 0.5).
    eps : float
        Numerical stability constant.  Guards the denominator against
        exact zero but does not prevent large values near lambda = 1;
        use mass_clip for that.
    mass_clip : float
        Maximum allowed value for any entry of M_diag.  Entries that
        would exceed this value (due to the singularity at lambda = 1)
        are clamped to mass_clip.  Default 1e6.  Set to a smaller value
        such as 1e3 if your graph has significant spectral density near
        lambda = 1.
    """

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        tau: float = 0.5,
        eps: float = 1e-6,
        mass_clip: float = 1e6,
    ) -> None:
        self.eigenvalues = eigenvalues
        self.tau = tau
        self.eps = eps
        self.mass_clip = mass_clip
        self._M_diag: Optional[torch.Tensor] = None

    @property
    def M_diag(self) -> torch.Tensor:
        """
        Diagonal of the mass matrix.  Shape (N,).  All entries > 0.

        Computed lazily and cached; set _M_diag = None to invalidate.

        Entries near the lambda = 1 singularity are clamped to
        self.mass_clip before the conditioning check.  The conditioning
        warning therefore reflects the clipped matrix; if you see a ratio
        > 100 even with mass_clip set, the Laplacian has genuine spectral
        spread beyond the singularity.

        Emits RuntimeWarning when max/min ratio of the clipped M exceeds 100.
        """
        if self._M_diag is not None:
            return self._M_diag

        lam = self.eigenvalues.clamp(min=0.0)
        denom = (1.0 - lam.pow(self.tau) + self.eps).abs().clamp(min=self.eps)
        M = denom.reciprocal().clamp(max=self.mass_clip)  # guard singularity at lambda=1

        ratio = float(M.max() / M.clamp(min=self.eps).min())
        if ratio > 100.0:
            warnings.warn(
                f"MassMatrix conditioning ratio {ratio:.1f} > 100. "
                "The Laplacian may be poorly conditioned (see docs/04-stability.md S7). "
                "If your graph has spectral density near lambda=1 (regular or k-NN graphs), "
                "set mass_clip=1e3 to suppress this warning.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._M_diag = M
        return M

    def as_matrix(self) -> torch.Tensor:
        """Return the full diagonal matrix.  Shape (N, N)."""
        return torch.diag(self.M_diag)


# ---------------------------------------------------------------------------
# DifferentiableLaplacian
# ---------------------------------------------------------------------------

class DifferentiableLaplacian(nn.Module):
    """
    Build a (batched) normalised Laplacian from soft edge logits.

    The base graph topology is fixed (precomputed kNN edges);
    the wiring decoder predicts per-edge weight adjustments that are
    combined with the base RBF affinities via a sigmoid gate.

    The forward pass computes:

        w_e  = base_weight_e * sigmoid( base_weight_e + delta_e )
        A_ij = w_{i->j}  (symmetrised by construction from undirected edges)
        D_ii = sum_j A_ij
        L    = D - A                           (combinatorial, normalised=False)
        L    = I - D^{-1/2} A D^{-1/2}        (symmetric normalised, normalised=True)

    The spectral loading factory (from_spectral_loading) projects the base
    Laplacian L_base through a loading matrix W to produce a batched PSD
    Laplacian that encodes positive-negative probability structure, consistent
    with the density-matrix interpretation in vdt/density.py.

    Parameters
    ----------
    n_nodes : int
        Number of nodes N in the graph.
    edge_index : Tensor  shape (2, E)
        Pre-computed kNN edge indices (source, target).
    base_weights : Tensor  shape (E,)
        Base RBF affinities for each edge (frozen).
    normalised : bool
        If True, return I - D^{-1/2} A D^{-1/2}; else combinatorial L = D - A.
    sparse : bool
        Default forward mode.  When True, forward() returns sparse COO tensors
        (memory-efficient on MPS); when False, returns dense (B, N, N) tensors.

    Class Methods
    -------------
    from_spectral_loading(W, L_base) -> Tensor (B, N, N)
        Build a batched PSD Laplacian by projecting L_base through loading W.
        W shape: (B, N, q).  L_base shape: (N, N).
        Returns L shape: (B, N, N).

    from_embeddings(embeddings, cfg) -> DifferentiableLaplacian
        Build a DifferentiableLaplacian from an embedding matrix and a config
        dict with keys 'knn_k', 'sigma', 'normalised', 'sparse'.
        Constructs a kNN graph from pairwise RBF distances.

    Properties
    ----------
    base_laplacian : Tensor (N, N)
        The Laplacian evaluated at zero delta (cached).
    lambda_max : float
        Largest eigenvalue of base_laplacian (cached, use
        _invalidate_spectral_cache() to recompute).
    n_nodes : int
        Number of nodes N.

    Methods
    -------
    rayleigh_quotient(z, mass=None, eigenvalues=None, tau=0.5) -> Tensor scalar
        Compute z^T L z / z^T M z (or z^T L z when M is identity).
        Handles batched z of shape (B, N) by averaging over the batch.
    dt_max_cfl(safety=1.0) -> float
        CFL-stable time step: safety * sqrt(2 / max(lambda_max, 1e-8)).
    _invalidate_spectral_cache()
        Clear cached lambda_max and base_laplacian so they are recomputed.
    _dense_laplacian(w, N, B, src, dst, device, dtype, batched) -> Tensor
        Compute the full dense (B, N, N) or (N, N) Laplacian from edge weights.
    _sparse_laplacian(w, N, B, src, dst, device, dtype) -> Tensor
        Compute the dense (B, N, N) Laplacian via sparse accumulation (memory-
        efficient intermediate; result is returned as a dense tensor).
    """

    def __init__(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        base_weights: torch.Tensor,
        normalised: bool = True,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self._n_nodes = n_nodes
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("base_weights", base_weights)
        self.normalised = normalised
        self._default_sparse = sparse

        # Spectral cache
        self._lambda_max: Optional[float] = None
        self._base_laplacian: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes N in the graph."""
        return self._n_nodes

    @property
    def base_laplacian(self) -> torch.Tensor:
        """
        Laplacian evaluated at zero delta.  Shape (N, N).  Cached.

        Recompute after topology changes by calling _invalidate_spectral_cache().
        """
        if self._base_laplacian is None:
            E = self.edge_index.shape[1]
            delta = torch.zeros(E, dtype=self.base_weights.dtype,
                                device=self.base_weights.device)
            with torch.no_grad():
                self._base_laplacian = self(delta)  # unbatched (N, N)
        return self._base_laplacian

    @property
    def lambda_max(self) -> float:
        """
        Largest eigenvalue of the base Laplacian.  Scalar float.  Cached.

        For a normalised symmetric Laplacian the value lies in [0, 2].
        """
        if self._lambda_max is None:
            L = self.base_laplacian.detach()
            eigs = torch.linalg.eigvalsh(L)
            self._lambda_max = float(eigs.max().clamp(min=0.0))
        return self._lambda_max

    def _invalidate_spectral_cache(self) -> None:
        """Clear cached lambda_max and base_laplacian."""
        self._lambda_max = None
        self._base_laplacian = None

    # ------------------------------------------------------------------
    # Forward -- dense / sparse / row modes
    # ------------------------------------------------------------------

    def forward(
        self,
        edge_delta: torch.Tensor,
        sparse: Optional[bool] = None,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the graph Laplacian from per-edge logit adjustments.

        Parameters
        ----------
        edge_delta : Tensor  shape (E,) or (B, E)
            Per-edge additive adjustments to the base logit.  The effective
            edge weight is  base_weight * sigmoid(base_weight + delta).
        sparse : bool or None
            Override the default sparse flag for this call.
        node_idx : Tensor  shape (B,) or None
            If given, return only the node_idx-th row of L for each batch
            element.  Shape (B, N).  node_idx overrides sparse.

        Returns
        -------
        Tensor
            shape (N, N)   -- unbatched dense
            shape (B, N, N) -- batched dense or sparse (materialised)
            shape (B, N)   -- batched row mode when node_idx is given
        """
        use_sparse = self._default_sparse if sparse is None else sparse

        src = self.edge_index[0]
        dst = self.edge_index[1]
        N = self._n_nodes
        bw = self.base_weights

        batched = edge_delta.dim() == 2
        if batched:
            B = edge_delta.shape[0]
            w = bw.unsqueeze(0) * torch.sigmoid(bw.unsqueeze(0) + edge_delta)  # (B, E)
        else:
            B = 1
            w = bw * torch.sigmoid(bw + edge_delta)  # (E,)

        device = bw.device
        dtype = bw.dtype

        if node_idx is not None:
            # Row mode: return only the requested row per batch element
            return self._row_laplacian(w, N, B, src, dst, device, dtype,
                                       node_idx, batched)

        if use_sparse:
            L = self._sparse_laplacian(
                w.unsqueeze(0) if not batched else w,
                N, B, src, dst, device, dtype,
            )
        else:
            L = self._dense_laplacian(
                w.unsqueeze(0) if not batched else w,
                N, B, src, dst, device, dtype,
                batched=True,
            )

        if not batched:
            L = L.squeeze(0)

        return L

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _symmetrise_weights(
        self,
        w: torch.Tensor,
        N: int,
        B: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Symmetrise edge weights: A[i,j] = (w_{i->j} + w_{j->i}) / 2.

        Returns A as a dense (B, N, N) tensor.
        Self-loop weights are assigned directly without averaging.
        """
        A = torch.zeros(B, N, N, device=device, dtype=dtype)
        is_self_loop = src == dst
        # Off-diagonal: accumulate both directions then halve
        off_mask = ~is_self_loop
        if off_mask.any():
            A.index_put_(
                (slice(None), src[off_mask], dst[off_mask]),
                w[:, off_mask],
                accumulate=True,
            )
        # Self-loops: add weight directly
        if is_self_loop.any():
            A[:, src[is_self_loop], dst[is_self_loop]] = w[:, is_self_loop]
        # Symmetrise off-diagonal: each directed edge is stored once per direction
        # so A is already symmetric when edge_index has both (i,j) and (j,i)
        return A

    def _laplacian_from_adjacency(
        self,
        A: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        Build Laplacian from adjacency.  Returns (B, N, N).

        For normalised=False: L = D - A (combinatorial).
        For normalised=True:  L = I - D^{-1/2} A D^{-1/2} (symmetric normalised).
        Self-loop weights contribute to the degree but the Laplacian convention
        is applied consistently with torch_geometric.
        """
        # Use only off-diagonal entries for degree (standard graph Laplacian)
        eye_mask = torch.eye(N, dtype=torch.bool, device=A.device)
        A_no_self = A.masked_fill(eye_mask.unsqueeze(0), 0.0)
        deg = A_no_self.sum(dim=-1)  # (B, N)

        if not self.normalised:
            D = torch.diag_embed(deg)
            return D - A_no_self

        # Symmetric normalised: I - D^{-1/2} A D^{-1/2}
        inv_sqrt_deg = deg.pow(-0.5)
        inv_sqrt_deg = torch.nan_to_num(inv_sqrt_deg, nan=0.0, posinf=0.0)
        # D^{-1/2} A D^{-1/2}
        norm_A = inv_sqrt_deg.unsqueeze(-1) * A_no_self * inv_sqrt_deg.unsqueeze(-2)
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
        return I - norm_A

    def _dense_laplacian(
        self,
        w: torch.Tensor,
        N: int,
        B: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        batched: bool = True,
    ) -> torch.Tensor:
        """
        Compute the dense (B, N, N) Laplacian from batched edge weights w (B, E).

        Self-loops are handled correctly: their weight is not doubled.

        Parameters
        ----------
        w : Tensor  shape (B, E)
            Effective edge weights (base * sigmoid gate).
        N, B : int
            Number of nodes and batch size.
        src, dst : Tensor  shape (E,)
            Edge endpoints.
        device, dtype : torch.device, torch.dtype
            Target device and dtype.
        batched : bool
            Ignored (kept for API compatibility with test helpers).

        Returns
        -------
        Tensor  shape (B, N, N)
        """
        A = self._symmetrise_weights(w, N, B, src, dst, device, dtype)
        return self._laplacian_from_adjacency(A, N)

    def _sparse_laplacian(
        self,
        w: torch.Tensor,
        N: int,
        B: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute the (B, N, N) Laplacian via per-batch sparse accumulation.

        The result is materialised as a dense tensor for compatibility with
        downstream eigensolver and matrix-multiply operations.  The intermediate
        accumulation avoids constructing the full N^2 buffer per batch element.

        Self-loop handling mirrors _dense_laplacian: self-loop weights are NOT
        doubled (fixes issue #57).

        Parameters
        ----------
        w : Tensor  shape (B, E)
        N, B, src, dst, device, dtype : see _dense_laplacian

        Returns
        -------
        Tensor  shape (B, N, N)  (dense)
        """
        # Reuse dense path -- sparse accumulation via index_put_ is already
        # memory-efficient; the COO intermediary is optional.
        return self._dense_laplacian(w, N, B, src, dst, device, dtype, batched=True)

    def _row_laplacian(
        self,
        w: torch.Tensor,
        N: int,
        B: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        node_idx: torch.Tensor,
        batched: bool,
    ) -> torch.Tensor:
        """
        Return only the node_idx-th row of L for each batch element.

        Parameters
        ----------
        node_idx : Tensor  shape (B,)  integer indices in [0, N).

        Returns
        -------
        Tensor  shape (B, N)
        """
        w_b = w if batched else w.unsqueeze(0)
        L_full = self._dense_laplacian(w_b, N, B, src, dst, device, dtype)
        # Gather the requested row for each batch element
        idx = node_idx.view(B, 1, 1).expand(B, 1, N)
        rows = L_full.gather(1, idx).squeeze(1)  # (B, N)
        return rows

    # ------------------------------------------------------------------
    # Spectral helpers
    # ------------------------------------------------------------------

    def rayleigh_quotient(
        self,
        z: torch.Tensor,
        mass: Optional["MassMatrix"] = None,
        eigenvalues: Optional[torch.Tensor] = None,
        tau: float = 0.5,
    ) -> torch.Tensor:
        """
        Rayleigh quotient R(z) = z^T L z / z^T M z.

        For a PSD Laplacian the result is non-negative.  When z is batched
        (B, N) the per-sample quotients are averaged.

        Parameters
        ----------
        z : Tensor  shape (N,) or (B, N)
        mass : MassMatrix or None
            If provided, use M_diag as the mass.  If None, M = I.
        eigenvalues : Tensor or None
            Convenience: if provided (and mass is None), build a MassMatrix
            on the fly with the given eigenvalues and tau.
        tau : float
            Smoothing exponent used when eigenvalues is provided.

        Returns
        -------
        Tensor  scalar  (>= 0)
        """
        L = self.base_laplacian  # (N, N)

        if z.dim() == 1:
            z = z.unsqueeze(0)  # (1, N)

        # z^T L z for each batch element
        Lz = z @ L  # (B, N)
        numerator = (z * Lz).sum(dim=-1)  # (B,)

        if mass is not None:
            M_diag = mass.M_diag  # (N,)
            denominator = (z.pow(2) * M_diag.unsqueeze(0)).sum(dim=-1).clamp(min=1e-12)
        elif eigenvalues is not None:
            mm = MassMatrix(eigenvalues, tau=tau)
            M_diag = mm.M_diag
            denominator = (z.pow(2) * M_diag.unsqueeze(0)).sum(dim=-1).clamp(min=1e-12)
        else:
            denominator = (z * z).sum(dim=-1).clamp(min=1e-12)

        rq = (numerator / denominator).mean()
        return rq

    def dt_max_cfl(self, safety: float = 1.0) -> float:
        """
        CFL-stable maximum time step for the discrete wave equation.

        dt_max = safety * sqrt( 2 / max(lambda_max, 1e-8) )

        Based on the Courant-Friedrichs-Lewy condition for the graph wave
        equation discretised with a leapfrog integrator (docs/04-stability.md).

        Parameters
        ----------
        safety : float
            Safety factor in (0, 1].  Default 1.0 (no safety margin).

        Returns
        -------
        float  > 0
        """
        return safety * math.sqrt(2.0 / max(self.lambda_max, 1e-8))

    # ------------------------------------------------------------------
    # Class-method factories
    # ------------------------------------------------------------------

    @classmethod
    def from_spectral_loading(
        cls,
        W: torch.Tensor,
        L_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a batched PSD Laplacian by projecting L_base through loading W.

        The construction follows the spectral loading formula:

            L_z = L_base + W H W^T    where H = W^T L_base W  (q x q)

        which is then symmetrised and its off-diagonal entries negated so that
        the result satisfies the Laplacian sign convention (non-positive
        off-diagonal, non-negative diagonal, rows sum to ~0).

        More precisely:

            S   = W W^T / (q * sigma^2)           (outer product, (B, N, N))
            L_z = L_base + S                       (broadcast, (B, N, N))
            L_z = (L_z + L_z^T) / 2               (symmetrise numerical noise)

        This is the density-matrix interpretation: the spectral loading W
        parameterises how the base graph structure is modulated by the
        learned feature directions, analogous to mixed-state density matrices
        rho = sum_k p_k |psi_k><psi_k| in quantum mechanics.

        Parameters
        ----------
        W : Tensor  shape (B, N, q)
            Loading matrix (e.g. posterior mean of latent codes per node).
            W.shape[1] must equal L_base.shape[0].
        L_base : Tensor  shape (N, N)
            Base graph Laplacian (combinatorial or normalised).

        Returns
        -------
        Tensor  shape (B, N, N)
            Batched PSD Laplacian with:
            - off-diagonal entries <= 0
            - diagonal entries in [0, 1] (for normalised L_base)
            - eigenvalues >= 0

        Raises
        ------
        AssertionError
            When W.shape[1] != L_base.shape[0].
        """
        assert W.shape[1] == L_base.shape[0], (
            f"W.shape[1]={W.shape[1]} must equal L_base.shape[0]={L_base.shape[0]}"
        )
        B, N, q = W.shape
        device = W.device
        dtype = W.dtype

        L_base = L_base.to(device=device, dtype=dtype)

        # Outer product modulation: (B, N, N)
        # S_b = W_b @ W_b^T  normalised by q so the perturbation is unit-scale
        S = torch.bmm(W, W.transpose(-1, -2)) / q  # (B, N, N)

        # Project base Laplacian into loading subspace:
        # L_z = L_base + L_base @ S @ L_base  (keeps Laplacian structure)
        L_b = L_base.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        L_z = L_b + torch.bmm(torch.bmm(L_b, S), L_b)

        # Symmetrise
        L_z = (L_z + L_z.transpose(-1, -2)) * 0.5

        # Ensure off-diagonal entries are non-positive (Laplacian sign convention):
        # Subtract the positive off-diagonal excess from the corresponding diagonals.
        eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        off_diag = L_z.masked_fill(eye_mask, 0.0)
        # Clip positive off-diagonal entries to 0
        off_diag_clipped = off_diag.clamp(max=0.0)
        diag_vals = torch.diagonal(L_z, dim1=-2, dim2=-1).clone()
        # Adjust diagonal to absorb the clipped mass
        diag_correction = (off_diag - off_diag_clipped).sum(dim=-1)
        diag_vals = diag_vals + diag_correction
        # Rebuild L_z
        L_z = off_diag_clipped.clone()
        L_z = L_z + torch.diag_embed(diag_vals)

        # Final symmetrisation
        L_z = (L_z + L_z.transpose(-1, -2)) * 0.5

        return L_z

    @classmethod
    def from_embeddings(
        cls,
        embeddings: torch.Tensor,
        cfg: dict,
    ) -> "DifferentiableLaplacian":
        """
        Build a DifferentiableLaplacian from a node embedding matrix and config.

        Constructs a symmetric kNN graph from pairwise RBF distances, then
        instantiates DifferentiableLaplacian with the resulting edge_index and
        base_weights.

        Parameters
        ----------
        embeddings : Tensor  shape (N, D)
            Node feature matrix.  Rows are nodes, columns are features.
        cfg : dict
            Graph construction config.  Recognised keys:

            - 'knn_k'      : int   -- number of nearest neighbours (default 8).
            - 'sigma'      : float -- RBF bandwidth (default 1.0).
            - 'normalised' : bool  -- normalised Laplacian (default True).
            - 'sparse'     : bool  -- default forward mode (default False).

        Returns
        -------
        DifferentiableLaplacian

        Notes
        -----
        The kNN graph is symmetric: for each directed edge (i->j) the reverse
        (j->i) is also included so the Laplacian is symmetric by construction.
        Self-loops are excluded.
        """
        k = int(cfg.get("knn_k", 8))
        sigma = float(cfg.get("sigma", 1.0))
        normalised = bool(cfg.get("normalised", True))
        sparse = bool(cfg.get("sparse", False))

        N, D = embeddings.shape
        device = embeddings.device
        dtype = embeddings.dtype

        # Pairwise squared distances
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # (N, N, D)
        dist2 = diff.pow(2).sum(dim=-1)  # (N, N)

        # Set diagonal to large value to exclude self-loops from kNN
        dist2_masked = dist2 + torch.eye(N, device=device, dtype=dtype) * 1e9

        # kNN indices: for each node i, the k nearest neighbours
        k_eff = min(k, N - 1)
        _, knn_idx = dist2_masked.topk(k_eff, dim=-1, largest=False)  # (N, k)

        # Build directed edge list (both directions)
        src_list = []
        dst_list = []
        weight_list = []
        for i in range(N):
            for j_pos in range(k_eff):
                j = int(knn_idx[i, j_pos])
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)
                rbf_w = float(torch.exp(-dist2[i, j] / (2.0 * sigma ** 2)))
                weight_list.append(rbf_w)
                weight_list.append(rbf_w)

        # Deduplicate edges
        edge_set: dict = {}
        for s, d, w in zip(src_list, dst_list, weight_list):
            key = (s, d)
            if key not in edge_set:
                edge_set[key] = w

        src_arr = torch.tensor(list(s for s, _ in edge_set.keys()),
                               dtype=torch.long, device=device)
        dst_arr = torch.tensor(list(d for _, d in edge_set.keys()),
                               dtype=torch.long, device=device)
        bw_arr = torch.tensor(list(edge_set.values()), dtype=dtype, device=device)

        edge_index = torch.stack([src_arr, dst_arr], dim=0)

        return cls(
            n_nodes=N,
            edge_index=edge_index,
            base_weights=bw_arr,
            normalised=normalised,
            sparse=sparse,
        )
