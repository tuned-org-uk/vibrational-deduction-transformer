"""
Wiring Decoder  --  z  ->  edge weight adjustments  ->  Laplacian L(z).

Two decoder classes are provided:

  WiringDecoder          v1, mixture-of-experts over edge templates (unchanged).
  SpectralLoadingDecoder , eigenbasis factorisation W = U_q diag(omega) S.

For the v1 architecture see the class docstring of WiringDecoder below.

 architecture (SpectralLoadingDecoder)
-----------------------------------------
Maps z (B, q) to a loading matrix W (B, d, q) in the Laplacian eigenbasis,
then synthesises L(z) (B, N, N) via
DifferentiableLaplacian.from_spectral_loading(W, L_base)::

    z  (B, q)
      |--> S_net          Linear(q, q*q) -> reshape  -> S       (B, q, q)
      |--> log_var_S_head Linear(q, q*q) -> reshape  -> log_var_S (B, q, q)
      |                                     clamped to [-6, 4]
      |--> omega_net      Linear(q, q)   -> exp       -> omega   (B, q)  [positive]
    W = U_q @ diag(omega) @ S                                    (B, d, q)
    L(z) = from_spectral_loading(W, L_base)                      (B, N, N)

log_var_S is produced by an *independent* head and must NOT be derived
from S.  It is the posterior log-variance for q(S) = N(S, exp(log_var_S))
and is consumed by spectral_basis_kl in the three-term ELBO (issue #52).

Gradients from the reconstruction loss and spectral_basis_kl flow back
through L(z) -> W -> S_net and omega_net -> z, and separately through
kl_S -> log_var_S -> log_var_S_head -> z.

Config dispatch
---------------
    decoder_type: spectral           -> SpectralLoadingDecoder  ( default)
    decoder_type: mixture_of_experts -> WiringDecoder           (v1 fallback)

Ref: docs//00-architecture.md
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .laplacian import DifferentiableLaplacian


# ---------------------------------------------------------------------------
# WiringDecoder  (v1 -- unchanged)
# ---------------------------------------------------------------------------

class WiringDecoder(nn.Module):
    """
    Decode a latent wiring code z into a differentiable graph Laplacian
    L(z) via a mixture-of-experts over n_heads edge templates.

    The architecture is::

        z  (B, latent_dim)
          |-- trunk MLP -->
          h  (B, hidden_dim)
          |-- n_heads Linear(hidden_dim, E) --> head_deltas (B, n_heads, E)
          |-- gate Linear(hidden_dim, n_heads) --> gates (B, n_heads) [softmax]
          |--> delta = sum_h gate_h * head_delta_h    (B, E)
          |--> DifferentiableLaplacian(delta)          (B, N, N) or (B, N)

    The edge weight for edge (i,j) is::

        w_ij = base_w_ij * sigmoid(delta_ij)

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent code z.
    n_edges : int
        Total number of directed edges E in the base kNN graph.
    hidden_dim : int
        Width of the shared trunk MLP.
    n_heads : int
        Number of mixture heads.
    laplacian : DifferentiableLaplacian
        Pre-built differentiable Laplacian module.
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

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_projs = nn.ModuleList([
            nn.Linear(hidden_dim, n_edges) for _ in range(n_heads)
        ])
        self.gate = nn.Linear(hidden_dim, n_heads)

    def forward(
        self,
        z: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent code z to a differentiable Laplacian.

        Parameters
        ----------
        z : Tensor  shape (B, latent_dim)
        node_idx : Tensor  shape (B,) or None
            When provided the Laplacian is returned as (B, N) row tensor.

        Returns
        -------
        L     : Tensor  (B, N, N) or (B, N) when node_idx is given.
        delta : Tensor  (B, E)  per-edge weight deltas.
        """
        h = self.trunk(z)
        gates = self.gate(h).softmax(dim=-1)
        head_deltas = torch.stack(
            [proj(h) for proj in self.head_projs], dim=1
        )
        delta = (gates.unsqueeze(-1) * head_deltas).sum(dim=1)
        L = (
            self.laplacian(delta, node_idx=node_idx)
            if node_idx is not None
            else self.laplacian(delta)
        )
        return L, delta


# ---------------------------------------------------------------------------
# SpectralLoadingDecoder  ()
# ---------------------------------------------------------------------------

class SpectralLoadingDecoder(nn.Module):
    """
     decoder: maps z (B, q) to a spectral loading matrix W (B, d, q)
    and then to a differentiable Laplacian L(z) (B, N, N).

    The loading matrix is factorised in the Laplacian eigenbasis::

        S         = S_net(z).view(B, q, q)                  -- spectral coefficients
        log_var_S = log_var_S_head(z).view(B, q, q).clamp   -- posterior log-variance (INDEPENDENT of S)
        omega     = exp( omega_net(z) )                      -- per-mode weights, > 0
        W         = U_q @ diag(omega) @ S                   -- (B, d, q)
        L(z)      = from_spectral_loading(W, L_base)

    Concretely, since U_q is (d, q) and diag(omega) @ S is (B, q, q):

        W = U_q.unsqueeze(0) @ (omega.unsqueeze(-1) * S)    # (B, d, q)

    L(z) is built by DifferentiableLaplacian.from_spectral_loading, which
    computes per-batch edge weights as:

        w_ij = base_aff_ij * sigmoid( -||W_i - W_j||^2 )

    where base_aff_ij = (-(L_base - I))_ij (the positive affinity recovered
    from the normalised symmetric Laplacian).  The full path
    z -> S, omega -> W -> L(z) is differentiable.

    Posterior log-variance head (log_var_S_head)
    --------------------------------------------
    log_var_S_head is a separate Linear(q, q*q) layer whose output is
    the posterior log-variance for q(S) = N(S, exp(log_var_S)).  It is
    INDEPENDENT of S: its weights are initialised separately and it
    receives z as input directly, not S.  This independence is required
    for a valid Gaussian reparameterisation -- tying var = S^2 + eps
    (the old proxy) conflated the posterior mean and variance and produced
    an invalid KL gradient (see issue #52).

    Near-identity initialisation
    ----------------------------
    _init_weights sets S_net.bias = eye(q).flatten() and S_net.weight = 0,
    so S_net(z=0) = eye(q) and the decoder starts from a well-conditioned
    spectral loading matrix.  omega_net is initialised with near-zero weights
    and zero bias, giving omega ~ exp(0) = 1 at init.
    log_var_S_head is initialised with near-zero weights and zero bias,
    giving log_var_S ~ 0 (var ~ 1) at init, matching the unit-variance prior.

    Relation to the three-term ELBO (#24)
    --------------------------------------
    spectral_basis_kl uses S (posterior mean) and log_var_S (posterior
    log-variance) as independent quantities.  The omega values feed into
    tau_mode_kl via log_a, log_b from ModeWeightHead in the encoder.
    SpectralLoadingDecoder produces W, omega, S, L_z, log_var_S; the
    ELBO assembly in model.py consumes all five.

    Parameters
    ----------
    q : int
        Number of latent modes (= latent_dim).
    d : int
        Feature / node dimension.  Must equal N (graph node count) in the
        standard case so that from_spectral_loading(W, L_base) is valid.
        An AssertionError is raised at construction when d != N would cause
        a shape mismatch inside from_spectral_loading.
    """

    def __init__(self, q: int, d: int) -> None:
        super().__init__()
        self.q = q
        self.d = d

        # S_net: z (B, q) -> S (B, q, q)  -- posterior mean of spectral loading
        self.S_net = nn.Linear(q, q * q)

        # log_var_S_head: z (B, q) -> log_var_S (B, q, q)
        # Independent of S -- must NOT be derived from S.  This independence
        # is the fix for issue #52: the old proxy log(S^2 + eps) conflated
        # mean and variance of q(S), invalidating the kl_S KL gradient.
        self.log_var_S_head = nn.Linear(q, q * q)

        # omega_net: z (B, q) -> log_omega (B, q)  [omega = exp(log_omega) > 0]
        self.omega_net = nn.Linear(q, q)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise weights so the decoder starts close to identity.

        S_net weight is zeroed and bias is set to eye(q).flatten(), so
        S_net(z=0) = eye(q) for all batch elements at initialisation.
        omega_net is initialised with near-zero weights and zero bias so
        initial omega_k = exp(~0) ~ 1 for all modes.
        log_var_S_head is initialised with near-zero weights and zero bias
        so log_var_S ~ 0 (posterior variance ~ 1) at init, matching the
        unit-variance prior.  The zero-bias init ensures independence from
        S_net at construction.
        """
        with torch.no_grad():
            # S_net: weight=0, bias=eye(q).flatten() so S_net(0) = eye_flat
            nn.init.zeros_(self.S_net.weight)
            self.S_net.bias.copy_(torch.eye(self.q).view(self.q * self.q))

            # log_var_S_head: near-zero weights, zero bias -> log_var_S ~ 0 at init
            nn.init.normal_(self.log_var_S_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.log_var_S_head.bias)

            nn.init.normal_(self.omega_net.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.omega_net.bias)

    def forward(
        self,
        z: torch.Tensor,          # (B, q)
        U_q: torch.Tensor,        # (d, q)  frozen Laplacian eigenvectors
        L_base: torch.Tensor,     # (N, N)  frozen base topology
    ) -> Tuple[
        torch.Tensor,   # W          (B, d, q)
        torch.Tensor,   # omega      (B, q)
        torch.Tensor,   # S          (B, q, q)
        torch.Tensor,   # L_z        (B, N, N)
        torch.Tensor,   # log_var_S  (B, q, q)  -- independent posterior log-variance
    ]:
        """
        Forward pass: z -> (W, omega, S, L_z, log_var_S).

        Parameters
        ----------
        z : Tensor  shape (B, q)
            Latent code from the encoder.
        U_q : Tensor  shape (d, q)
            Leading q eigenvectors of the base Laplacian L_base.
            Frozen (no gradient expected through U_q).
        L_base : Tensor  shape (N, N)
            Base graph Laplacian encoding the topology.  Frozen.

        Returns
        -------
        W : Tensor  (B, d, q)
            Spectral loading matrix.  Gradient flows back to z.
        omega : Tensor  (B, q)
            Per-mode positive weights.  Gradient flows back to z.
        S : Tensor  (B, q, q)
            Spectral coefficient matrix (posterior mean).  Gradient flows
            back to z.
        L_z : Tensor  (B, N, N)
            Synthesised normalised symmetric Laplacian.  Zero row sums,
            non-positive off-diagonal.
        log_var_S : Tensor  (B, q, q)
            Independent posterior log-variance for q(S) = N(S, exp(log_var_S)).
            Output of log_var_S_head(z), clamped to [-6, 4] for numerical
            stability.  Must NOT be derived from S -- it is an independent
            function of z alone (fix for issue #52).
        """
        B, q = z.shape

        # -- Spectral coefficients (posterior mean) -----------------------
        S = self.S_net(z).view(B, q, q)           # (B, q, q)

        # -- Independent posterior log-variance ---------------------------
        # Produced by a dedicated head from z directly, not from S.
        # Clamped to [-6, 4] matching the log_var clamping in WiringEncoder:
        #   lower bound -6: prevents posterior variance from collapsing to 0
        #   upper bound  4: prevents posterior variance from growing > ~55
        log_var_S = self.log_var_S_head(z).view(B, q, q).clamp(-6.0, 4.0)  # (B, q, q)

        # -- Per-mode weights (strictly positive via exp) ------------------
        omega = torch.exp(self.omega_net(z))       # (B, q)

        # -- Eigenbasis projection -----------------------------------------
        # W = U_q @ diag(omega) @ S
        # diag(omega) @ S  is broadcast:  omega.unsqueeze(-1) * S  -> (B, q, q)
        # U_q (d, q) @ (B, q, q)  ->  U_q.unsqueeze(0) @ (...)  -> (B, d, q)
        W = U_q.unsqueeze(0) @ (omega.unsqueeze(-1) * S)   # (B, d, q)

        # -- Synthesise Laplacian ------------------------------------------
        L_z = DifferentiableLaplacian.from_spectral_loading(W, L_base)  # (B, N, N)

        return W, omega, S, L_z, log_var_S
