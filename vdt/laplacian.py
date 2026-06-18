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
        Default forward mode.  
    """
    pass
