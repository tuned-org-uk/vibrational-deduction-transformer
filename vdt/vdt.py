"""
Vibrational Discrete-time (VDT) recurrent encoder core.

The VDT implements Rayleigh's damped wave equation in discrete time:

    Q_{t+1} = 2*Q_t - Q_{t-1}
              - dt^2 * Q_t @ L_f^T
              - Gamma * (Q_t - Q_{t-1})
              + dt^2 * B_t

where:
  Q_t    -- (N, d) node-feature state matrix at step t
  L_f    -- (N, N) feature-space graph Laplacian (from laplacian.py)
  Gamma  -- (d,) per-feature damping vector (softplus > 0)
  B_t    -- (N, d) external forcing from the transformer mixing block
  dt     -- scalar time step, CFL-clamped each forward pass

CFL clamping
------------
The CFL condition for the discrete-time wave equation is:

    dt <= sqrt(2 / lambda_max(L_f))

where lambda_max is the spectral radius of the **dynamic** feature-space
Laplacian L_f, not the frozen base Laplacian L(I).  Using the base
Laplacian can silently violate the CFL constraint when the encoder
sharpens high-frequency modes early in training (issue #53).

The block computes the CFL bound from L_f using the Gershgorin circle
theorem (O(N^2), no eigendecomposition):

    lambda_max(L_f) <= max_i sum_j |L_f[i,j]|   (Gershgorin row bound)

This bound is always >= lambda_max so the resulting dt_max is always a
valid (conservative) CFL limit.  The base-Laplacian bound from
lap.dt_max_cfl() is also computed; the stricter of the two is used.

Set recompute_cfl=False in forward() to fall back to the frozen base
bound when O(N^2) overhead is unacceptable (e.g. very large graphs
during inference).

Density update contract
-----------------------
After each wave step the block calls

    block.density.update(Q_curr, Q_tp1, alpha=density_momentum)

so that rho_plus and rho_minus track the consecutive wave states
(Q_curr -> bonding / constructive modes, Q_tp1 -> anti-bonding /
destructive modes).  Without this call the density matrices would
remain frozen at random initialisation and carry no wave information.
See issue #51 and vdt/density.py for the full update derivation.

The module accumulates per-step density matrices (rho_plus, rho_minus)
from vdt/density.py, making the signed spectral energy available
downstream for the variational Gamma KL term (#24).

Reference
---------
Rayleigh's Theory of Sound: the update mirrors the Newmark-beta
time-stepping scheme for damped structural dynamics.  The damping
term Gamma*(Q_t - Q_{t-1}) corresponds to Rayleigh proportional
damping C = alpha*M + beta*K.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vdt.density import SignedDensityMatrix
from vdt.laplacian import DifferentiableLaplacian


# ---------------------------------------------------------------------------
# CFL helpers
# ---------------------------------------------------------------------------

def _gershgorin_lambda_max(L: torch.Tensor) -> torch.Tensor:
    """
    Gershgorin circle upper bound on lambda_max(L)  for a single (N, N)
    symmetric matrix.  Returns a scalar tensor on the same device as L.

    The bound is the maximum row absolute-sum:

        lambda_max(L) <= max_i  sum_j |L[i, j]|

    This is always >= the true lambda_max and requires only O(N^2)
    operations -- no eigendecomposition.  For a graph Laplacian the bound
    equals twice the maximum weighted degree, which is typically within a
    factor of 2-4 of the true spectral radius.

    Parameters
    ----------
    L : Tensor  (N, N)  symmetric matrix (must be 2-D, unbatched).

    Returns
    -------
    Tensor  scalar  -- the Gershgorin upper bound on lambda_max.
    """
    return L.abs().sum(dim=-1).max()  # scalar


# ---------------------------------------------------------------------------
# VibrationalStateBlock
# ---------------------------------------------------------------------------

class VibrationalStateBlock(nn.Module):
    """
    Single discrete damped-wave recurrence step with transformer mixing.

    The block operates on node-feature state matrices Q of shape (N, d):

        B_t     = TransformerMix(Q_t)   -- attention-based forcing
        Q_{t+1} = 2*Q_t - Q_{t-1}
                  - dt^2 * Q_t @ L_f^T
                  - gamma * (Q_t - Q_{t-1})
                  + dt^2 * B_t

    CFL constraint
    ~~~~~~~~~~~~~~
    By default (recompute_cfl=True) the CFL bound is computed fresh from
    the dynamic L_f at every forward pass using the Gershgorin circle
    theorem (O(N^2), no eigendecomposition).  This prevents silent CFL
    violations when the encoder sharpens high-frequency modes early in
    training (issue #53).

    The frozen base-Laplacian bound from lap.dt_max_cfl() is also
    computed; the stricter (smaller) of the two bounds is used.

    Pass recompute_cfl=False to disable the Gershgorin recomputation and
    fall back to the frozen base bound.  This is appropriate when the
    graph is large and the O(N^2) overhead is unacceptable during
    inference or profiling.

    Density update
    ~~~~~~~~~~~~~~
    After computing Q_{t+1} the block updates the SignedDensityMatrix:

        self.density.update(Q_t, Q_{t+1}, alpha=density_momentum)

    so that rho_plus captures the outer-product structure of the current
    state (constructive / bonding modes) and rho_minus captures the
    outer-product structure of the next state (destructive / anti-bonding
    modes).  This wires the density matrices into the wave dynamics and
    ensures they carry genuine spectral information rather than remaining
    frozen at random initialisation (issue #51).

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes N.
    feat_dim : int
        Feature dimension d (per-node channels).
    n_heads : int
        Number of attention heads in the transformer mixing block.
    dropout : float
        Dropout applied inside the mixing block.
    density_momentum : float
        EMA momentum alpha in [0, 1) for SignedDensityMatrix.update().
        Default 0.9.  Lower values cause the density to track the
        instantaneous wave state more closely at the cost of higher
        variance between depth steps.
    """

    def __init__(
        self,
        n_nodes: int,
        feat_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        density_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.n_nodes  = n_nodes
        self.feat_dim = feat_dim
        self.density_momentum = density_momentum

        # Learnable log-time-step: dt = exp(log_dt) so dt > 0.
        # Initialised near dt=0.1 (ln(0.1) = -2.3026).
        self.log_dt = nn.Parameter(torch.tensor(-2.3026))

        # Per-feature damping raw parameter; gamma = softplus(raw) > 0.
        self._gamma_raw = nn.Parameter(torch.zeros(feat_dim))

        # Per-step density matrix (n_nodes x n_nodes) accumulates energy.
        # Updated each forward pass from (Q_curr, Q_next) via density.update().
        self.density = SignedDensityMatrix(n=n_nodes)

        # Transformer mixing: MHA + FFN on (N, d) treated as a sequence
        # of N tokens each of dimension d.
        self.norm1  = nn.LayerNorm(feat_dim)
        self.attn   = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2  = nn.LayerNorm(feat_dim)
        self.ffn    = nn.Sequential(
            nn.Linear(feat_dim, 4 * feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * feat_dim, feat_dim),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def gamma(self) -> torch.Tensor:
        """Per-feature damping vector (d,); guaranteed > 0 via softplus."""
        return F.softplus(self._gamma_raw)  # (d,)

    # ------------------------------------------------------------------
    # CFL helpers
    # ------------------------------------------------------------------
    def _cfl_dt_Lf(
        self,
        L_f: torch.Tensor,
        lap: DifferentiableLaplacian,
    ) -> torch.Tensor:
        """
        Return CFL-clamped time step computed from the dynamic L_f.

        Combines two bounds and takes the stricter (smaller) one:

          1. Gershgorin bound on lambda_max(L_f):
                 dt_Lf = sqrt(2 / gershgorin_lambda_max(L_f[0]))
             O(N^2), no eigendecomposition, guaranteed safe.

          2. Base-Laplacian bound from lap.dt_max_cfl():
                 dt_base = sqrt(2 / lambda_max(L(I)))
             Pre-computed at construction; may be too loose if L_f
             has sharpened high-frequency modes beyond L(I).

        Parameters
        ----------
        L_f : Tensor  (N, N) or (B, N, N)
            Dynamic feature-space Laplacian for the current batch.
        lap : DifferentiableLaplacian
            Provides the frozen base-Laplacian CFL bound as a fallback.

        Returns
        -------
        Tensor  scalar  -- the minimum of dt_free, dt_Lf, and dt_base.
        """
        dt_free = self.log_dt.exp()  # learnable, unclamped

        # Gershgorin bound on the first (or only) batch element
        L_single = L_f[0] if L_f.dim() == 3 else L_f  # (N, N)
        with torch.no_grad():  # no gradient needed for the dt clamp itself
            lam_gershgorin = _gershgorin_lambda_max(L_single).clamp(min=1e-8)
        dt_Lf = (2.0 / lam_gershgorin).sqrt().to(dtype=dt_free.dtype, device=dt_free.device)

        # Frozen base-Laplacian bound
        dt_base = torch.tensor(
            lap.dt_max_cfl(), dtype=dt_free.dtype, device=dt_free.device
        )

        # Use the strictest (smallest) of all three bounds
        return torch.minimum(torch.minimum(dt_free, dt_Lf), dt_base)

    def _cfl_dt(
        self,
        lap: DifferentiableLaplacian,
    ) -> torch.Tensor:
        """
        Deprecated compatibility shim -- use _cfl_dt_Lf instead.

        This method returns the CFL bound from the frozen base Laplacian
        only.  It may be too loose when L_f has a larger lambda_max than
        L(I).  Use forward(recompute_cfl=True) (default) to enable the
        Gershgorin dynamic bound from L_f.

        .. deprecated::
            Will be removed in a future release.  Kept so external code
            that calls _cfl_dt() directly does not break silently.
        """
        warnings.warn(
            "_cfl_dt() uses the frozen base Laplacian CFL bound and may "
            "silently violate the CFL constraint when L_f has a larger "
            "lambda_max than L(I).  Use _cfl_dt_Lf(L_f, lap) instead, "
            "or call forward(recompute_cfl=True) (the default). "
            "See issue #53.",
            DeprecationWarning,
            stacklevel=2,
        )
        dt_free = self.log_dt.exp()
        dt_max  = torch.tensor(
            lap.dt_max_cfl(), dtype=dt_free.dtype, device=dt_free.device
        )
        return torch.minimum(dt_free, dt_max)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        Q_t:   torch.Tensor,          # (N, d) or (B, N, d)
        Q_tm1: torch.Tensor,          # (N, d) or (B, N, d)  -- previous step
        L_f:   torch.Tensor,          # (N, N) or (B, N, N)  -- feature Laplacian
        lap:   DifferentiableLaplacian,
        recompute_cfl: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one wave-step, update the density matrix, and return
        the next state together with the updated rho_plus / rho_minus.

        Parameters
        ----------
        Q_t   : current state   (N, d) or (B, N, d)
        Q_tm1 : previous state  (N, d) or (B, N, d)
        L_f   : feature Laplacian  (N, N) or (B, N, N)
        lap   : DifferentiableLaplacian instance
        recompute_cfl : bool  (default True)
            When True (default), compute the CFL dt bound from L_f using
            the Gershgorin circle theorem.  This prevents silent CFL
            violations when L_f has a larger lambda_max than the frozen
            base Laplacian (issue #53).

            When False, fall back to the frozen base-Laplacian bound from
            lap.dt_max_cfl().  Use only for performance-sensitive inference
            on very large graphs where the O(N^2) Gershgorin computation
            is measurably expensive.

        Returns
        -------
        Q_tp1      : Tensor  -- next state, same shape as Q_t
        rho_plus   : Tensor  (n_nodes, n_nodes)  PSD  -- constructive modes
        rho_minus  : Tensor  (n_nodes, n_nodes)  PSD  -- destructive modes

        Notes
        -----
        The density matrix is updated in-place after Q_tp1 is computed:

            self.density.update(Q_t, Q_tp1, alpha=self.density_momentum)

        rho_plus and rho_minus are therefore computed from the wave states
        of this step, not from random initialisation.  The EMA momentum
        (density_momentum) controls how quickly the density tracks changes
        in the wave trajectory across depth steps.
        """
        unbatched = Q_t.ndim == 2
        if unbatched:
            Q_t   = Q_t.unsqueeze(0)    # (1, N, d)
            Q_tm1 = Q_tm1.unsqueeze(0)
            L_f   = L_f.unsqueeze(0)    # (1, N, N)

        # -- Transformer mixing: B_t = FFN(MHA(Q_t)) ---------------------
        h, _   = self.attn(self.norm1(Q_t), self.norm1(Q_t), self.norm1(Q_t))
        Q_mix  = Q_t + h
        B_t    = Q_mix + self.ffn(self.norm2(Q_mix))  # (B, N, d)

        # -- CFL-clamped dt ----------------------------------------------
        if recompute_cfl:
            dt = self._cfl_dt_Lf(L_f, lap)
        else:
            # Fallback: frozen base-Laplacian bound only
            dt_free = self.log_dt.exp()
            dt_base = torch.tensor(
                lap.dt_max_cfl(), dtype=dt_free.dtype, device=dt_free.device
            )
            dt = torch.minimum(dt_free, dt_base)
        dt2 = dt * dt

        # -- Wave update -------------------------------------------------
        L_term = torch.bmm(L_f, Q_t)  # (B, N, N) x (B, N, d) -> (B, N, d)
        damp   = self.gamma.unsqueeze(0).unsqueeze(0) * (Q_t - Q_tm1)

        Q_tp1 = 2.0 * Q_t - Q_tm1 - dt2 * L_term - damp + dt2 * B_t

        # -- Density update: wire rho_plus / rho_minus to the wave states -
        self.density.update(Q_t, Q_tp1, alpha=self.density_momentum)

        if unbatched:
            Q_tp1 = Q_tp1.squeeze(0)

        return Q_tp1, self.density.rho_plus, self.density.rho_minus


# ---------------------------------------------------------------------------
# VDT  -- stacked VibrationalStateBlocks
# ---------------------------------------------------------------------------

class VDT(nn.Module):
    """
    Vibrational Discrete-Time (VDT) encoder.

    Stacks K VibrationalStateBlock layers.  Each block receives the
    output of the previous block as Q_t and the zero-init Q_{t-1}.
    A final modal projection compresses the output to a latent vector.

    CFL clamping
    ~~~~~~~~~~~~
    By default (recompute_cfl=True in forward()) the CFL bound is
    recomputed from the dynamic L_f at each block via the Gershgorin
    circle theorem (O(N^2) per block, no eigendecomposition).  This
    guarantees the CFL constraint is respected even when the encoder
    synthesises an L_f with sharper high-frequency modes than the frozen
    base Laplacian (issue #53).

    Density matrices
    ~~~~~~~~~~~~~~~~
    Each block maintains a SignedDensityMatrix updated from its own
    consecutive wave states (Q_curr, Q_next) via density.update().
    The rho_plus_list and rho_minus_list returned by forward() therefore
    contain K density matrices that track the vibrational energy at each
    depth, with rho_plus[k] reflecting constructive modes at layer k and
    rho_minus[k] reflecting destructive modes (issue #51).

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes N.
    feat_dim : int
        Per-node feature dimension d.  Input X0 must be (N, d).
    n_layers : int
        Number of stacked VibrationalStateBlocks K.
    m_modes : int
        Number of eigenvector modes used for modal projection.
        Defaults to feat_dim // 4 (at least 1).
    n_heads : int
        Attention heads inside each VibrationalStateBlock.
    dropout : float
        Dropout inside each VibrationalStateBlock.
    density_momentum : float
        EMA momentum forwarded to SignedDensityMatrix.update() in each
        block.  Default 0.9.  See VibrationalStateBlock for details.

    Notes
    -----
    Between blocks the previous state Q_{t-1} is carried as the block
    input Q_t from the layer above and zero for the first block:

        layer 0 : Q_t = X0,           Q_{t-1} = zeros_like(X0)
        layer k : Q_t = Q_{k-1},      Q_{t-1} = Q_{k-2} (or zeros at k=1)
    """

    def __init__(
        self,
        n_nodes:   int,
        feat_dim:  int,
        n_layers:  int  = 4,
        m_modes:   Optional[int] = None,
        n_heads:   int  = 4,
        dropout:   float = 0.1,
        density_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.n_nodes  = n_nodes
        self.feat_dim = feat_dim
        self.n_layers = n_layers
        self.m_modes  = m_modes if m_modes is not None else max(1, feat_dim // 4)

        self.blocks = nn.ModuleList([
            VibrationalStateBlock(
                n_nodes=n_nodes,
                feat_dim=feat_dim,
                n_heads=n_heads,
                dropout=dropout,
                density_momentum=density_momentum,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        X0:     torch.Tensor,
        L_f:    torch.Tensor,
        eigvecs: torch.Tensor,
        lap:    DifferentiableLaplacian,
        recompute_cfl: bool = True,
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        Tuple[List[torch.Tensor], List[torch.Tensor]],
    ]:
        """
        Run K vibrational steps and return the final state plus
        all intermediate states and density matrices.

        Each block's CFL bound is recomputed from L_f (Gershgorin) when
        recompute_cfl=True (default), preventing silent CFL violations
        (issue #53).  Each block's density matrix is updated from its
        wave states before rho_plus / rho_minus are appended to the
        output lists (issue #51).

        Parameters
        ----------
        X0      : initial node-feature matrix  (N, d) or (B, N, d)
        L_f     : feature-space Laplacian       (N, N) or (B, N, N)
        eigvecs : graph eigenvectors            (N, *) used for modal projection
        lap     : DifferentiableLaplacian instance
        recompute_cfl : bool  (default True)
            Forwarded to each VibrationalStateBlock.forward() call.
            Set False to use frozen base-Laplacian CFL bound (faster,
            less safe).

        Returns
        -------
        Q_K   : Tensor -- final state after K blocks
        Q_states : List[Tensor] -- [X0, Q_1, ..., Q_K]  length K+1
        (rho_plus_list, rho_minus_list) : Tuple[List, List]
                  -- one (n_nodes, n_nodes) PSD tensor per block,
                     updated from wave states (Q_curr, Q_next) at each depth.
        """
        Q_prev  = torch.zeros_like(X0)
        Q_curr  = X0

        Q_states:       List[torch.Tensor] = [X0]
        rho_plus_list:  List[torch.Tensor] = []
        rho_minus_list: List[torch.Tensor] = []

        for block in self.blocks:
            Q_next, rho_p, rho_m = block(
                Q_curr, Q_prev, L_f, lap, recompute_cfl=recompute_cfl
            )
            Q_states.append(Q_next)
            rho_plus_list.append(rho_p)
            rho_minus_list.append(rho_m)
            Q_prev = Q_curr
            Q_curr = Q_next

        Q_K = Q_curr
        return Q_K, Q_states, (rho_plus_list, rho_minus_list)

    def modal_projection(
        self,
        Q_K:    torch.Tensor,
        eigvecs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project Q_K onto the leading m_modes eigenvectors and mean-pool.

        z = mean_over_modes( Q_K @ U_m )   shape: (d,) or (B, d)

        where U_m = eigvecs[:, :m_modes]  (N, m_modes).
        """
        U_m = eigvecs[:, : self.m_modes]
        if Q_K.ndim == 2:
            z = (Q_K.t() @ U_m).mean(dim=-1)
        else:
            z = torch.einsum("bnd,nm->bdm", Q_K, U_m).mean(dim=-1)
        return z
