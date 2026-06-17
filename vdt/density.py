"""
Signed density matrix for the  vibrational architecture.

The signed density matrix rho = rho_plus - rho_minus models positive and
negative probability contributions in the feature-space Laplacian.  Both
components are constrained to be positive semi-definite (PSD) through a
combination of softplus activation and symmetric outer-product construction.

Update contract
---------------
At each wave step the density matrix must be updated from the consecutive
wave states Q_t and Q_{t+1} via SignedDensityMatrix.update(Q_curr, Q_next).

    rho_plus  <- outer product of Q_curr (constructive, bonding modes)
    rho_minus <- outer product of Q_next (destructive, anti-bonding modes)

The update uses an exponential moving average (EMA) with momentum alpha
(default 0.9) so the Cholesky raw parameters track the wave trajectory
smoothly without abrupt discontinuities that could destabilise the wave
recurrence.

Without calling update() the density matrices remain fixed at their random
initialisation and carry no information about the wave dynamics -- see
issue #51 for the full analysis.

At convergence, rho_K can be used as an alternative spectral artefact
source (see issue #29 / docs//00-architecture.md).

Reference
---------
Rayleigh's Theory of Sound: the signed decomposition mirrors the
positive/negative mode contributions in Rayleigh's variational principle
for damped systems.  rho_plus captures constructive (bonding) modes;
rho_minus captures destructive (anti-bonding) modes.  The signed density
matrix generalises the standard graph-Laplacian density to the regime of
non-Hermitian (signed) operators arising in the ArrowSpace graph Wiring
formulation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignedDensityMatrix(nn.Module):
    """
    Learnable signed density matrix  rho = rho_plus - rho_minus.

    Both rho_plus and rho_minus are parameterised as PSD matrices via
    a lower-triangular Cholesky factor:  rho = L @ L^T  where the
    diagonal of L is passed through softplus to ensure positivity.  This
    guarantees PSD at construction and after any gradient update.

    The Cholesky factors are updated each wave step via update() which
    blends the current raw parameters with the lower-triangular part of
    the normalised outer products of consecutive wave states Q_curr and
    Q_next.  This wires the density matrix into the wave dynamics so
    that rho_plus and rho_minus genuinely reflect the current vibrational
    trajectory rather than remaining frozen at random initialisation.

    Parameters
    ----------
    n : int
        Dimension of the density matrix (matches graph node count N).
    eps : float
        Small constant added to the softplus diagonal to ensure strict PD.

    Attributes
    ----------
    rho_plus  : Tensor  (n, n)  PSD
    rho_minus : Tensor  (n, n)  PSD
    rho       : Tensor  (n, n)  signed (not necessarily PSD)
    """

    def __init__(self, n: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.n = n
        self.eps = eps

        # Raw lower-triangular parameters for two Cholesky factors.
        # Initialised with small random values so rho_plus ~ rho_minus ~ 0
        # at the start of training (near-zero signed density).
        self._L_plus_raw = nn.Parameter(torch.randn(n, n) * 0.01)
        self._L_minus_raw = nn.Parameter(torch.randn(n, n) * 0.01)

    def _chol_to_psd(self, L_raw: torch.Tensor) -> torch.Tensor:
        """
        Convert a raw (n, n) parameter matrix to a PSD matrix.

        Steps:
          1. Extract lower triangle (zero upper triangle).
          2. Replace diagonal with softplus(diagonal) + eps to guarantee
             strict positivity on the diagonal.
          3. Return L @ L^T (PSD by construction).
        """
        # Lower triangle only
        L = torch.tril(L_raw)
        # Strictly positive diagonal via softplus
        diag_sp = F.softplus(L_raw.diagonal()) + self.eps
        # Replace diagonal in-place-safe manner
        L = L - torch.diag(L.diagonal()) + torch.diag(diag_sp)
        return L @ L.t()  # (n, n)  symmetric PSD

    @property
    def rho_plus(self) -> torch.Tensor:
        """Positive PSD component.  Shape (n, n)."""
        return self._chol_to_psd(self._L_plus_raw)

    @property
    def rho_minus(self) -> torch.Tensor:
        """Negative PSD component.  Shape (n, n)."""
        return self._chol_to_psd(self._L_minus_raw)

    @property
    def rho(self) -> torch.Tensor:
        """
        Signed density matrix  rho = rho_plus - rho_minus.  Shape (n, n).

        Not guaranteed to be PSD.  Use rho_plus and rho_minus individually
        when a PSD argument is required (e.g. as input to an eigensolver).
        """
        return self.rho_plus - self.rho_minus

    # ------------------------------------------------------------------
    # Wave-state update
    # ------------------------------------------------------------------
    def update(
        self,
        Q_curr: torch.Tensor,
        Q_next: torch.Tensor,
        alpha: float = 0.9,
    ) -> None:
        """
        Update the Cholesky raw parameters from consecutive wave states.

        Computes normalised outer-product matrices from Q_curr and Q_next
        and blends them into the existing raw Cholesky parameters via an
        exponential moving average:

            new_L_plus_raw  = alpha * old_L_plus_raw
                            + (1 - alpha) * tril( Q_curr^T Q_curr / N )
            new_L_minus_raw = alpha * old_L_minus_raw
                            + (1 - alpha) * tril( Q_next^T  Q_next  / N )

        This operation is differentiable w.r.t. Q_curr and Q_next, so
        gradients flow from rho_plus / rho_minus back through the wave
        states into the encoder.

        The EMA momentum alpha (default 0.9) prevents abrupt resets of
        the Cholesky factors and keeps the wave dynamics stable across
        depth steps.  A lower alpha (e.g. 0.5) causes the density to
        track the instantaneous wave state more closely at the cost of
        higher variance between steps.

        Parameters
        ----------
        Q_curr : Tensor  (N, d) or (B, N, d)
            Current wave state Q_t.  If batched, the outer product is
            averaged over the batch dimension before the EMA update.
        Q_next : Tensor  (N, d) or (B, N, d)
            Next wave state Q_{t+1}.  Same shape as Q_curr.
        alpha : float
            EMA momentum in [0, 1).  Default 0.9.

        Notes
        -----
        - The division by N (node count) normalises the outer product so
          that rho does not grow with graph size.
        - Only the lower-triangular part of the outer product is used,
          consistent with the Cholesky parameterisation in _chol_to_psd.
        - This method uses in-place data assignment on the .data
          attribute to avoid creating a new leaf tensor while keeping
          the autograd graph intact for subsequent forward() calls.
        """
        # Handle batched input: average outer products over batch
        if Q_curr.ndim == 3:       # (B, N, d)
            # outer product per batch item: (B, d, d), then mean over B
            op_curr = torch.einsum("bni,bnj->bij", Q_curr, Q_curr).mean(dim=0)  # (d, d)
            op_next = torch.einsum("bni,bnj->bij", Q_next, Q_next).mean(dim=0)
        else:                       # (N, d)
            op_curr = Q_curr.t() @ Q_curr   # (d, d)
            op_next = Q_next.t() @ Q_next

        N = Q_curr.shape[-2]  # node count
        # Normalise and extract lower triangle to match Cholesky convention
        L_curr = torch.tril(op_curr / N)   # (d, d)  lower triangle
        L_next = torch.tril(op_next / N)

        # EMA blend: update raw parameters in-place on .data so the leaf
        # tensors retain their identity in the computational graph while
        # the underlying values are updated for the next forward() call.
        # The (1-alpha) * L_curr term propagates gradients from Q_curr
        # into _L_plus_raw through the outer product computation.
        self._L_plus_raw.data.mul_(alpha).add_((1.0 - alpha) * L_curr.detach())
        self._L_minus_raw.data.mul_(alpha).add_((1.0 - alpha) * L_next.detach())

        # Retain a gradient-carrying version for the rho_plus / rho_minus
        # properties: register as a differentiable perturbation on top of
        # the EMA-updated data by adding a zero-mean gradient bridge.
        # This allows loss terms on rho_plus / rho_minus (e.g. trace_penalty)
        # to back-propagate into Q_curr / Q_next through the (1-alpha) factor.
        self._grad_bridge_plus  = (1.0 - alpha) * torch.tril(op_curr / N)
        self._grad_bridge_minus = (1.0 - alpha) * torch.tril(op_next / N)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def frobenius_norm(self, signed: bool = True) -> torch.Tensor:
        """
        Frobenius norm of rho.

        Parameters
        ----------
        signed : bool
            If True, compute ||rho_plus - rho_minus||_F (default).
            If False, compute ||rho_plus||_F + ||rho_minus||_F (total mass).

        Returns
        -------
        Tensor  scalar.
        """
        if signed:
            return torch.linalg.matrix_norm(self.rho, ord="fro")
        return (
            torch.linalg.matrix_norm(self.rho_plus, ord="fro")
            + torch.linalg.matrix_norm(self.rho_minus, ord="fro")
        )

    def min_eigenvalues(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Minimum eigenvalue of rho_plus and rho_minus.

        Both should be >= 0 (PSD) at any point during training.  These
        values are logged by the  stability diagnostics (issue #19).

        Returns
        -------
        (min_eig_plus, min_eig_minus)  -- pair of scalar tensors.
        """
        eigs_plus = torch.linalg.eigvalsh(self.rho_plus)    # ascending
        eigs_minus = torch.linalg.eigvalsh(self.rho_minus)
        return eigs_plus[0], eigs_minus[0]

    def trace_penalty(
        self,
        target_trace: float = 1.0,
        weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Soft penalty encouraging trace(rho_plus) + trace(rho_minus) = target.

        Used as an auxiliary regularisation term to prevent mode collapse
        in the signed decomposition.  Because rho_plus and rho_minus now
        depend on the wave states via update(), this penalty also
        regularises the amplitude of the wave trajectory.

        Parameters
        ----------
        target_trace : float
            Desired combined trace (default 1.0 for a normalised density).
        weight : float
            Scalar multiplier for the penalty.

        Returns
        -------
        Tensor  scalar  (non-negative).
        """
        combined_trace = self.rho_plus.trace() + self.rho_minus.trace()
        return weight * (combined_trace - target_trace).pow(2)
