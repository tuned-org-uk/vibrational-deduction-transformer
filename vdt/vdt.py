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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vdt.density import SignedDensityMatrix
from vdt.laplacian import DifferentiableLaplacian


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

    After computing Q_{t+1} the block updates the SignedDensityMatrix:

        self.density.update(Q_t, Q_{t+1}, alpha=density_momentum)

    so that rho_plus captures the outer-product structure of the current
    state (constructive / bonding modes) and rho_minus captures the
    outer-product structure of the next state (destructive / anti-bonding
    modes).  This wires the density matrices into the wave dynamics and
    ensures they carry genuine spectral information rather than remaining
    frozen at random initialisation (issue #51).

    CFL constraint: dt = min(exp(log_dt), dt_max_cfl) is enforced at
    every forward call so the wave update remains stable.

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

        # Learnable log-time-step: dt = softplus(log_dt) in practice we
        # exponentiate so dt > 0.  Init near dt=0.1.
        self.log_dt = nn.Parameter(torch.tensor(-2.3026))  # ln(0.1)

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

    def _cfl_dt(self, lap: DifferentiableLaplacian) -> torch.Tensor:
        """
        Return CFL-clamped time step as a scalar tensor.

        dt = min( exp(log_dt), dt_max_cfl(lap) )

        Warning
        -------
        dt_max_cfl is computed from the frozen base Laplacian L(I), not
        from the dynamic feature-space Laplacian L_f passed to forward().
        If L_f has a larger lambda_max than the base Laplacian (common
        early in training), the true CFL constraint may be silently
        violated.  See issue #53 for the full fix using the Gershgorin
        bound on L_f.
        """
        dt_free   = self.log_dt.exp()
        dt_max    = torch.tensor(
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one wave-step and update the density matrix.

        Parameters
        ----------
        Q_t   : current state   (N, d) or (B, N, d)
        Q_tm1 : previous state  (N, d) or (B, N, d)
        L_f   : feature Laplacian  (N, N) or (B, N, N)
        lap   : DifferentiableLaplacian instance (for dt_max_cfl)

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
        # Q_t treated as (B, N, d) sequence of N tokens
        h, _   = self.attn(self.norm1(Q_t), self.norm1(Q_t), self.norm1(Q_t))
        Q_mix  = Q_t + h                               # residual
        B_t    = Q_mix + self.ffn(self.norm2(Q_mix))  # (B, N, d)

        # -- CFL-clamped dt ----------------------------------------------
        dt  = self._cfl_dt(lap)   # scalar tensor
        dt2 = dt * dt

        # -- Wave update -------------------------------------------------
        # Laplacian term: L_f @ Q_t  -- (B, N, d)
        L_term   = torch.bmm(L_f, Q_t)           # (B, N, N) x (B, N, d) -> (B, N, d)
        # Damping term: gamma (d,) broadcast over (B, N, d)
        damp     = self.gamma.unsqueeze(0).unsqueeze(0) * (Q_t - Q_tm1)

        Q_tp1 = 2.0 * Q_t - Q_tm1 - dt2 * L_term - damp + dt2 * B_t

        # -- Density update: wire rho_plus / rho_minus to the wave states -
        # Q_t   -> rho_plus  (constructive / bonding mode outer product)
        # Q_tp1 -> rho_minus (destructive / anti-bonding mode outer product)
        # The EMA momentum self.density_momentum controls the tracking speed.
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

    Density matrices
    ----------------
    Each block maintains a SignedDensityMatrix updated from its own
    consecutive wave states (Q_curr, Q_next) via density.update().
    The rho_plus_list and rho_minus_list returned by forward() therefore
    contain K density matrices that track the vibrational energy at each
    depth, with rho_plus[k] reflecting constructive modes at layer k and
    rho_minus[k] reflecting destructive modes.

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
        X0:     torch.Tensor,             # (N, d) or (B, N, d)  -- initial state
        L_f:    torch.Tensor,             # (N, N) or (B, N, N)
        eigvecs: torch.Tensor,            # (N, N) or (N, K_eig) -- graph eigenvectors
        lap:    DifferentiableLaplacian,  # for CFL clamping
    ) -> Tuple[
        torch.Tensor,          # Q_K       -- final state  (N, d) or (B, N, d)
        List[torch.Tensor],    # Q_states  -- list of K+1 states
        Tuple[List[torch.Tensor], List[torch.Tensor]],  # (rho_plus_list, rho_minus_list)
    ]:
        """
        Run K vibrational steps and return the final state plus
        all intermediate states and density matrices.

        Each block's density matrix is updated from its wave states before
        rho_plus / rho_minus are appended to the output lists, so the
        returned density matrices reflect the actual wave trajectory at
        each depth rather than random initialisations.

        Parameters
        ----------
        X0      : initial node-feature matrix  (N, d) or (B, N, d)
        L_f     : feature-space Laplacian       (N, N) or (B, N, N)
        eigvecs : graph eigenvectors            (N, *) used for modal projection
        lap     : DifferentiableLaplacian instance

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
            Q_next, rho_p, rho_m = block(Q_curr, Q_prev, L_f, lap)
            Q_states.append(Q_next)
            rho_plus_list.append(rho_p)
            rho_minus_list.append(rho_m)
            Q_prev = Q_curr
            Q_curr = Q_next

        Q_K = Q_curr  # (N, d) or (B, N, d)

        return Q_K, Q_states, (rho_plus_list, rho_minus_list)

    def modal_projection(
        self,
        Q_K:    torch.Tensor,   # (N, d) or (B, N, d)
        eigvecs: torch.Tensor,  # (N, K_eig)
    ) -> torch.Tensor:
        """
        Project Q_K onto the leading m_modes eigenvectors and mean-pool.

        z = mean_over_modes( Q_K @ U_m )   shape: (d,) or (B, d)

        where U_m = eigvecs[:, :m_modes]  (N, m_modes).
        """
        U_m = eigvecs[:, : self.m_modes]   # (N, m_modes)
        if Q_K.ndim == 2:                  # unbatched (N, d)
            z = (Q_K.t() @ U_m).mean(dim=-1)   # (d,)
        else:                              # batched (B, N, d)
            # (B, d, N) x (N, m_modes) -> (B, d, m_modes)
            z = torch.einsum("bnd,nm->bdm", Q_K, U_m).mean(dim=-1)  # (B, d)
        return z
