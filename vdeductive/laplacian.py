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
    lap     = DifferentiableLaplacian.from_embeddings(E, knn_k=8, sigma=1.0)

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
matrix of positive-negative probability (SignedDensityMatrix in vdeductive/density.py).
The spectral loading factory (from_spectral_loading) projects the base Laplacian
into the subspace spanned by the loading matrix W, yielding a batched PSD
Laplacian whose off-diagonal entries are non-positive.

Feature-space Laplacian convention (arrowspace / graph-wiring)
--------------------------------------------------------------
arrowspace and graph-wiring build the Laplacian in *feature* space, not in
node space.  This means the adjacency matrix is computed on the *transpose*
of the node-feature matrix E:

    Node-feature matrix E  :  shape (N, D)  -- rows are nodes
    Feature-space input    :  E.t()          -- shape (D, N)  -- rows are features

When from_embeddings receives E.t() it treats each of the D features as a
point in N-dimensional node space and builds a (D, D) Gram matrix:

    G = E.t() @ E = E^T E   (D x D)

rather than the node-space Gram matrix E @ E.t() (N x N).  The resulting
Laplacian is therefore (D, D) and lives in feature space.

Callers in model.py (from_config) and train.py always pass E.t().contiguous()
so that the feature-space convention is respected throughout.

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

    Conditioning warning
    --------------------
    A conditioning warning is emitted only when the ratio max(M_ii)/min(M_ii)
    of the *clipped* mass diagonal exceeds ``warn_threshold``, which is
    computed automatically as::

        warn_threshold = max(100.0, mass_clip / 10.0)

    Rationale: with mass_clip=1e3 the natural clipped ratio on a normalised
    Laplacian (eigenvalues near 0 give M~1, eigenvalues near 1 give M~mass_clip)
    is on the order of mass_clip itself.  Using a fixed threshold of 100 causes
    a spurious warning whenever mass_clip > 1000, even though the clip is
    already performing its intended job.  The adaptive threshold fires only
    when the conditioning is genuinely worse than what mass_clip allows.

    To silence the warning entirely, set mass_clip to a large value (e.g. 1e6)
    or reduce it further so the ratio falls below the threshold.  The warning
    text always prints the effective threshold so the user knows what to do.

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
        lambda = 1.  The conditioning warning threshold is automatically
        scaled to mass_clip/10 so that the normal clipped ratio does not
        trigger a false alarm.
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
        warning fires only when the ratio of the *clipped* M exceeds
        max(100, mass_clip/10).  This means that with mass_clip=1e3 the
        threshold is 100 (unchanged), but with mass_clip=1e4 the threshold
        rises to 1000, preventing spurious warnings on graphs whose natural
        clipped ratio is proportional to mass_clip.

        If the ratio exceeds the threshold even after clipping, the
        Laplacian has genuine spectral spread beyond what mass_clip handles
        and the warning is actionable.
        """
        if self._M_diag is not None:
            return self._M_diag

        lam = self.eigenvalues.clamp(min=0.0)
        denom = (1.0 - lam.pow(self.tau) + self.eps).abs().clamp(min=self.eps)
        M = denom.reciprocal().clamp(max=self.mass_clip)  # guard singularity at lambda=1

        # Adaptive threshold: the expected clipped ratio for a graph with
        # eigenvalues spanning [0, 1] is roughly mass_clip / M_min where
        # M_min ~ 1 for low-frequency modes.  We allow up to mass_clip/10
        # before warning so that users who have already set mass_clip to a
        # sensible value do not see a false alarm on every run.
        warn_threshold = max(100.0, self.mass_clip / 10.0)
        ratio = float(M.max() / M.clamp(min=self.eps).min())
        if ratio > warn_threshold:
            warnings.warn(
                f"MassMatrix conditioning ratio {ratio:.1f} > {warn_threshold:.0f}. "
                "The Laplacian may be poorly conditioned (see docs/04-stability.md S7). "
                f"Current mass_clip={self.mass_clip:.0f}. "
                "Reduce mass_clip further (e.g. mass_clip=100) or increase it to 1e6 "
                "if you want to disable all clipping.",
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

    The forward pass computes::

        w_e  = base_weight_e * sigmoid( base_weight_e + delta_e )
        A_ij = w_{i->j}  (symmetrised by construction from undirected edges)
        D_ii = sum_j A_ij
        L    = D - A                           (combinatorial, normalised=False)
        L    = I - D^{-1/2} A D^{-1/2}        (symmetric normalised, normalised=True)

    The spectral loading factory (from_spectral_loading) projects the base
    Laplacian L_base through a loading matrix W to produce a batched PSD
    Laplacian that encodes positive-negative probability structure, consistent
    with the density-matrix interpretation in vdeductive/density.py.

    Feature-space Laplacian convention (arrowspace / graph-wiring)
    --------------------------------------------------------------
    arrowspace and graph-wiring always build the Laplacian in *feature* space.
    The from_embeddings factory receives the entities to connect as its first
    argument.  To build a feature-space Laplacian from a node-feature matrix
    E of shape (N, D), callers must pass E.t().contiguous() (shape D x N):

        lap = DifferentiableLaplacian.from_embeddings(E.t().contiguous(), ...)

    This makes the D features the "nodes" of the kNN graph and N the ambient
    dimension.  The resulting Laplacian is (D, D) and operates in feature
    space.  Passing E directly (shape N x D) builds a node-space Laplacian
    (N, N), which is incorrect for the vibrational-deduction architecture.

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
        Default forward mode.  When True, forward() uses a memory-efficient
        scatter path; when False, returns dense (B, N, N) tensors directly.

    Class Methods
    -------------
    from_spectral_loading(W, L_base) -> Tensor (B, N, N)
        Build a batched PSD Laplacian by projecting L_base through loading W.
        W shape: (B, N, q).  L_base shape: (N, N).
        Returns L shape: (B, N, N).

    from_embeddings(points, knn_k, sigma, normalised, sparse) -> DifferentiableLaplacian
        Build a DifferentiableLaplacian from a matrix of points.
        Each row is treated as one entity (node) in the kNN graph.
        For a feature-space Laplacian pass E.t().contiguous() where E is
        the (N, D) node-feature matrix (arrowspace convention).

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
        Compute the full dense (B, N, N) Laplacian from edge weights.
    _sparse_laplacian(w, N, B, src, dst, device, dtype) -> Tensor
        Compute the dense (B, N, N) Laplacian via scatter accumulation.
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

        # Spectral cache (not module state, not buffers)
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
        Largest eigenvalue of the base Laplacian. Scalar float. Cached.

        For a normalised symmetric Laplacian the value lies in [0, 2].
        On MPS, eigvalsh is executed on CPU because the operator is not
        implemented natively.
        """
        if self._lambda_max is None:
            L = self.base_laplacian.detach()
            L_cpu = L.to("cpu")
            eigs = torch.linalg.eigvalsh(L_cpu)
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
            element.  Output shape (B, N).  Overrides the sparse flag.

        Returns
        -------
        Tensor
            shape (N, N)    -- unbatched dense
            shape (B, N, N) -- batched dense
            shape (B, N)    -- batched row mode when node_idx is given
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
            w = (bw * torch.sigmoid(bw + edge_delta)).unsqueeze(0)  # (1, E)

        device = bw.device
        dtype = bw.dtype

        if node_idx is not None:
            return self._row_laplacian(w, N, B, src, dst, device, dtype, node_idx)

        if use_sparse:
            L = self._sparse_laplacian(w, N, B, src, dst, device, dtype)
        else:
            L = self._dense_laplacian(w, N, B, src, dst, device, dtype)

        if not batched:
            L = L.squeeze(0)

        return L

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_adjacency(
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
        Build the dense (B, N, N) adjacency matrix from batched edge weights.

        Uses scatter_add on a flat (B*N*N) view to avoid index_put_ with
        slice objects (which are not supported by PyTorch's index_put_).
        Self-loop edges are handled correctly: their weight is accumulated
        once, not doubled by the symmetrisation step.

        Parameters
        ----------
        w : Tensor  shape (B, E)  -- effective edge weights
        N, B : int
        src, dst : Tensor  shape (E,)  -- edge endpoint indices
        device, dtype : target device and dtype

        Returns
        -------
        Tensor  shape (B, N, N)
        """
        # Flat linear index into (B, N, N): batch b, row i, col j -> b*N*N + i*N + j
        # We accumulate w[b, e] at position b*N*N + src[e]*N + dst[e]
        E = w.shape[1]

        # batch offsets: (B, 1) broadcast over E
        batch_offset = torch.arange(B, device=device).unsqueeze(1) * (N * N)  # (B, 1)
        # flat edge positions for each (b, e): (B, E)
        edge_flat = src.unsqueeze(0) * N + dst.unsqueeze(0)  # (1, E) -> broadcasts
        indices_flat = (batch_offset + edge_flat).reshape(-1)  # (B*E,)

        A_flat = torch.zeros(B * N * N, device=device, dtype=dtype)
        A_flat.scatter_add_(0, indices_flat, w.reshape(-1))
        A = A_flat.view(B, N, N)

        # Symmetrisation for off-diagonal edges:
        # When edge_index contains both (i,j) and (j,i) the accumulation
        # already places each direction in its correct cell; A is symmetric
        # by construction for undirected graphs.  Self-loops land on the
        # diagonal exactly once.
        return A

    def _laplacian_from_adjacency(
        self,
        A: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        Build Laplacian from adjacency matrix A of shape (B, N, N).

        Degree is computed from off-diagonal entries only (standard convention).
        Self-loop entries on the diagonal are zeroed before degree computation.

        For normalised=False: L = D - A_off  (combinatorial Laplacian).
        For normalised=True:  L = I - D^{-1/2} A_off D^{-1/2}  (symmetric normalised).

        Returns Tensor shape (B, N, N).
        """
        eye_mask = torch.eye(N, dtype=torch.bool, device=A.device)
        A_off = A.masked_fill(eye_mask.unsqueeze(0), 0.0)  # zero self-loops
        deg = A_off.sum(dim=-1)  # (B, N)

        if not self.normalised:
            D = torch.diag_embed(deg)  # (B, N, N)
            return D - A_off

        # Symmetric normalised: I - D^{-1/2} A D^{-1/2}
        inv_sqrt_deg = deg.pow(-0.5)
        inv_sqrt_deg = torch.nan_to_num(inv_sqrt_deg, nan=0.0, posinf=0.0)
        norm_A = inv_sqrt_deg.unsqueeze(-1) * A_off * inv_sqrt_deg.unsqueeze(-2)
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
        batched: bool = True,  # kept for API compat with test helpers
    ) -> torch.Tensor:
        """
        Compute the dense (B, N, N) Laplacian from batched edge weights w.

        Parameters
        ----------
        w : Tensor  shape (B, E)  -- effective edge weights
        N, B : int
        src, dst : Tensor  shape (E,)
        device, dtype : torch.device, torch.dtype
        batched : bool  -- unused; kept for API compatibility with test helpers

        Returns
        -------
        Tensor  shape (B, N, N)
        """
        A = self._build_adjacency(w, N, B, src, dst, device, dtype)
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
        Compute the (B, N, N) Laplacian using scatter_add accumulation.

        Identical result to _dense_laplacian; the scatter_add path avoids
        constructing intermediate N x N matrices in Python loops, making it
        preferable on MPS and memory-constrained devices.

        Self-loop handling mirrors _dense_laplacian: self-loop weights are
        not doubled (fixes issue #57).

        Returns
        -------
        Tensor  shape (B, N, N)  (dense)
        """
        return self._dense_laplacian(w, N, B, src, dst, device, dtype)

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
        L_full = self._dense_laplacian(w, N, B, src, dst, device, dtype)
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
            If provided (and mass is None), build a MassMatrix on the fly
            with the given eigenvalues and tau.
        tau : float
            Smoothing exponent used when eigenvalues is provided.

        Returns
        -------
        Tensor  scalar  (>= 0)
        """
        L = self.base_laplacian  # (N, N)

        if z.dim() == 1:
            z = z.unsqueeze(0)  # (1, N)

        Lz = z @ L  # (B, N)
        numerator = (z * Lz).sum(dim=-1)  # (B,)

        if mass is not None:
            M_diag = mass.M_diag
            denominator = (z.pow(2) * M_diag.unsqueeze(0)).sum(dim=-1).clamp(min=1e-12)
        elif eigenvalues is not None:
            mm = MassMatrix(eigenvalues, tau=tau)
            denominator = (z.pow(2) * mm.M_diag.unsqueeze(0)).sum(dim=-1).clamp(min=1e-12)
        else:
            denominator = (z * z).sum(dim=-1).clamp(min=1e-12)

        return (numerator / denominator).mean()

    def dt_max_cfl(self, safety: float = 1.0) -> float:
        """
        CFL-stable maximum time step for the discrete wave equation.

        dt_max = safety * sqrt( 2 / max(lambda_max, 1e-8) )

        Based on the Courant-Friedrichs-Lewy condition for the graph wave
        equation discretised with a leapfrog integrator (docs/04-stability.md).

        Parameters
        ----------
        safety : float  -- safety factor in (0, 1].  Default 1.0.

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
        Build a batched symmetric normalised Laplacian from spectral loadings.

        Parameters
        ----------
        W : Tensor  shape (B, N, q)
            Loading matrix.  W.shape[1] must equal L_base.shape[0].
        L_base : Tensor  shape (N, N)
            Base graph Laplacian.  Used for shape / device compatibility.

        Returns
        -------
        Tensor  shape (B, N, N)
            Symmetric normalised Laplacian with:
            - off-diagonal entries <= 0
            - diagonal entries in [0, 1]
            - eigenvalues in [0, 2]
        """
        assert W.shape[1] == L_base.shape[0], (
            f"W.shape[1]={W.shape[1]} must equal L_base.shape[0]={L_base.shape[0]}"
        )

        B, N, q = W.shape
        device = W.device
        dtype = W.dtype

        # Symmetric similarity from spectral loadings
        S = torch.bmm(W, W.transpose(-1, -2)) / q
        S = 0.5 * (S + S.transpose(-1, -2))

        # Remove self-interaction
        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        S = S * (1.0 - eye)

        # Nonnegative adjacency
        A = torch.relu(S)

        # Symmetric normalised Laplacian
        deg = A.sum(dim=-1)
        inv_sqrt_deg = deg.clamp(min=1e-12).pow(-0.5)
        inv_sqrt_deg = torch.where(
            deg > 0,
            inv_sqrt_deg,
            torch.zeros_like(inv_sqrt_deg),
        )

        A_norm = inv_sqrt_deg.unsqueeze(-1) * A * inv_sqrt_deg.unsqueeze(-2)
        I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        L = I - A_norm

        # Final symmetrisation for numerical stability
        L = 0.5 * (L + L.transpose(-1, -2))
        return L

    @classmethod
    def from_embeddings(
        cls,
        embeddings: torch.Tensor,
        knn_k: int = 8,
        sigma: float = 1.0,
        normalised: bool = True,
        sparse: bool = False,
    ) -> "DifferentiableLaplacian":
        """
        Build a DifferentiableLaplacian from a matrix of points.

        Each *row* of ``embeddings`` is treated as one entity (graph node)
        in the kNN graph.  The pairwise squared distances are computed as::

            dist2_ij = ||row_i - row_j||^2

        and the RBF affinities as::

            w_ij = exp( -dist2_ij / (2 * sigma^2) )

        arrowspace / graph-wiring convention
        -------------------------------------
        The vibrational-deduction architecture builds the Laplacian in
        *feature* space, not in node space.  Given a node-feature matrix
        E of shape (N, D) (rows = nodes, columns = features), callers must
        pass the *transpose* E.t().contiguous() (shape D x N) so that:

          - The D features become the graph nodes (entities to connect).
          - The N node values become the ambient coordinates.
          - The Gram matrix computed internally is E^T E (D x D), the
            correct feature-space Gram matrix.
          - The returned Laplacian is (D, D), operating in feature space.

        Passing E directly (N x D) builds a node-space Laplacian (N x N),
        which is incorrect for the vibrational-deduction architecture.

        Example::

            # Feature-space Laplacian (correct for arrowspace):
            lap = DifferentiableLaplacian.from_embeddings(
                E.t().contiguous(), knn_k=15, sigma=1.2
            )
            # lap.n_nodes == D   (number of features)

        Parameters
        ----------
        embeddings : Tensor  shape (P, C)
            Matrix of P points in C-dimensional space.  Each row is one
            graph node.  For a feature-space Laplacian pass
            E.t().contiguous() where E is the (N, D) node-feature matrix.
        knn_k : int
            Number of nearest neighbours per node (default 8).
        sigma : float
            RBF bandwidth (default 1.0).  Base weight = exp(-d^2 / (2*sigma^2)).
        normalised : bool
            If True, build a symmetric normalised Laplacian (default True).
        sparse : bool
            Default forward mode for the returned instance (default False).

        Returns
        -------
        DifferentiableLaplacian
            Instance with n_nodes == P (== D when E.t() is passed).

        Notes
        -----
        The kNN graph is symmetric: for each directed edge (i->j) the reverse
        (j->i) is also included so the Laplacian is symmetric by construction.
        Self-loops are excluded.
        """
        N, D = embeddings.shape
        device = embeddings.device
        dtype = embeddings.dtype

        # Pairwise squared distances
        x2 = (embeddings * embeddings).sum(dim=1, keepdim=True)      # (N, 1)
        dist2 = x2 + x2.transpose(0, 1) - 2.0 * (embeddings @ embeddings.t())
        dist2 = dist2.clamp(min=0.0)

        # Mask diagonal to exclude self-loops from kNN
        dist2_masked = dist2 + torch.eye(N, device=device, dtype=dtype) * 1e9

        k_eff = min(knn_k, N - 1)
        _, knn_idx = dist2_masked.topk(k_eff, dim=-1, largest=False)  # (N, k)

        # Build directed edges both ways and deduplicate
        edge_set: dict = {}
        for i in range(N):
            for j_pos in range(k_eff):
                j = int(knn_idx[i, j_pos])
                rbf_w = float(torch.exp(-dist2[i, j] / (2.0 * sigma ** 2)))
                for s, d in ((i, j), (j, i)):
                    if (s, d) not in edge_set:
                        edge_set[(s, d)] = rbf_w

        src_arr = torch.tensor([s for s, _ in edge_set.keys()],
                               dtype=torch.long, device=device)
        dst_arr = torch.tensor([d for _, d in edge_set.keys()],
                               dtype=torch.long, device=device)
        bw_arr = torch.tensor(list(edge_set.values()), dtype=dtype, device=device)

        return cls(
            n_nodes=N,
            edge_index=torch.stack([src_arr, dst_arr], dim=0),
            base_weights=bw_arr,
            normalised=normalised,
            sparse=sparse,
        )
