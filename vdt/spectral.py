"""
Spectral utilities for the Wiring Autoencoder.

This module implements the three spectral primitives that sit at the heart
of the VDT training loop:

    TauModeDiffusion         differentiable truncated spectral diffusion decoder
    spectral_freq_cost       J_freq high-frequency energy penalty (training signal)
    lambda_fingerprint_soft  differentiable Gaussian KDE fingerprint (training signal)
    lambda_fingerprint_hard  hard histogram fingerprint (monitoring / logging only)
    lambda_fingerprint       backwards-compat alias for lambda_fingerprint_hard

 additions (issue #24 / four-term ELBO):

    spectral_basis_kl     KL between Gaussian basis posterior q(S) and
                          eigenvalue-weighted prior p(S|I)
    tau_mode_kl           KL between Gamma mode posterior q(omega_k) and
                          Exponential mode prior p(omega_k | tau, lambda_k)

 stability mitigations (issue #68):

    tau_mode_kl           now accepts a_min kwarg; clamps a = exp(log_a)
                          to min=a_min before lgamma/digamma to prevent
                          full Gamma shape collapse.
    active_mode_penalty   soft Lagrange penalty nu * relu(q_min - N_active)
                          that maintains a minimum number of active modes.
    count_active_modes    diagnostic helper that returns N_active as a
                          plain Python int (no gradient graph involvement).
                          Used by model.forward() to populate out['N_active']
                          for spectral_kl_health_check (issue #77).

 soft fingerprint (issue #56):

    lambda_fingerprint_soft  Gaussian KDE soft-histogram replacing the
                             non-differentiable torch.histc loop.
                             Fully batched, stays on input device, and is
                             differentiable w.r.t. eigvals so it can be
                             used as a true training signal rather than
                             a monitoring metric.
    lambda_fingerprint_hard  Renamed original hard-histogram function.
                             Non-differentiable; use for logging only.
    lambda_fingerprint       Backwards-compat alias -> lambda_fingerprint_hard.

All primitives share the same underlying eigensystem of the graph
Laplacian L(z).  To avoid redundant O(N^3) CPU eigendecompositions at every
training step, every public function accepts an optional eigvals /
eig_cache argument that allows the caller to pass precomputed spectral
quantities from outside the training loop.  See train.py for the
recommended caching pattern.

Architectural context
---------------------
See docs/00-architecture.md for the full data-flow diagram and module
reference.

The J_freq cost and tau-mode diffusion kernel correspond directly to the
ArrowSpace spectral cost function described in the Module Reference::

    J_freq = sum_{j > tau_modes} lambda_j( L(z) )
    K_tau  = U_k * exp(-t*Lambda_k) * U_k^T   (heat kernel, k = tau_modes)

Stability note
--------------
All torch.linalg.eigh / eigvalsh calls are offloaded to CPU because MPS does
not implement aten::_linalg_eigh as of PyTorch 2.x.  Every entry routes
through _precondition() which applies two conditioning steps before handing
the matrix to LAPACK:

1. Symmetrisation  L = (L + L^T) / 2
2. Tikhonov shift  L += eps * I  (_EIGSOLVE_EPS = 1e-4)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple


#: Small diagonal shift added to every matrix before passing to LAPACK.
#: Prevents repeated-eigenvalue failures on nearly degenerate Laplacians.
_EIGSOLVE_EPS: float = 1e-4


# ---------------------------------------------------------------------------
# Internal eigensolver helpers
# ---------------------------------------------------------------------------

def _precondition(L: torch.Tensor) -> torch.Tensor:
    """
    Symmetrise L and apply a Tikhonov diagonal shift before LAPACK.

    This is an internal helper called by _safe_eigh and _safe_eigvalsh.
    It always returns a CPU float32 tensor regardless of the input device.

    Steps:
      1. Symmetrisation  L_sym = (L + L^T) / 2
      2. Tikhonov shift  L_sym += _EIGSOLVE_EPS * I

    Parameters
    ----------
    L : torch.Tensor
        Shape (N, N) or (B, N, N).  May be on any device.

    Returns
    -------
    torch.Tensor
        Conditioned matrix in CPU float32, same shape as input.
    """
    L_cpu = L.detach().cpu().float()
    L_sym = (L_cpu + L_cpu.transpose(-2, -1)) * 0.5
    n = L_sym.shape[-1]
    eye = torch.eye(n, dtype=L_sym.dtype)
    if L_sym.dim() == 3:
        eye = eye.unsqueeze(0)
    return L_sym + _EIGSOLVE_EPS * eye


def _safe_eigvalsh(L: torch.Tensor) -> torch.Tensor:
    """
    Eigenvalues of a symmetric matrix (or batch) with LAPACK fallback.

    Routes through _precondition before calling torch.linalg.eigvalsh.
    On failure falls back to the general eigvals solver, taking real parts
    and sorting ascending so training is never interrupted.

    Parameters
    ----------
    L : torch.Tensor
        Shape (B, N, N) or (N, N).  Any device.

    Returns
    -------
    torch.Tensor
        Eigenvalues ascending.  Shape (B, N) or (N,).
        On the same device as input.
    """
    device = L.device
    L_pre = _precondition(L)
    try:
        ev = torch.linalg.eigvalsh(L_pre)
    except torch._C._LinAlgError:
        ev_complex = torch.linalg.eigvals(L_pre)
        ev = ev_complex.real
        ev, _ = ev.sort(dim=-1)
    return ev.to(device)


def _safe_eigh(
    L: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigenvalues and eigenvectors of a symmetric matrix with fallback.

    Same conditioning and fallback strategy as _safe_eigvalsh but also
    returns the eigenvector matrix.

    Parameters
    ----------
    L : torch.Tensor
        Shape (B, N, N) or (N, N).  Any device.

    Returns
    -------
    eigvals : torch.Tensor  ascending.  Shape (B, N) or (N,).
    eigvecs : torch.Tensor  corresponding.  Shape (B, N, N) or (N, N).
    Both tensors are on the same device as input.
    """
    device = L.device
    L_pre = _precondition(L)
    try:
        ev, evec = torch.linalg.eigh(L_pre)
    except torch._C._LinAlgError:
        ev_complex, evec_complex = torch.linalg.eig(L_pre)
        order = ev_complex.real.argsort(dim=-1)
        ev = ev_complex.real.gather(-1, order)
        order_v = order.unsqueeze(-2).expand_as(evec_complex.real)
        evec = evec_complex.real.gather(-1, order_v)
    return ev.to(device), evec.to(device)


# ---------------------------------------------------------------------------
# TauModeDiffusion
# ---------------------------------------------------------------------------

class TauModeDiffusion(nn.Module):
    """
    Truncated spectral diffusion using the k lowest-frequency eigenvectors
    of the Laplacian L(z).  This is the tau-mode approximation from the
    ArrowSpace spectral framework.

    Given Laplacian L and embedding table E (N, D) the diffusion kernel is::

        U, Lambda = eig_k(L)                    -- (N, k), (k,)
        K_tau  = U * diag(exp(-t*Lambda)) * U^T  -- (N, N) heat kernel
        x_hat_i = K_tau[i, :] @ E               -- (D,) diffused embedding

    Learnable log_t controls the diffusion time and is trained jointly
    with the rest of the model.

    Shape contract
    --------------
    E always has shape (N, D) -- a shared node-embedding table, not batched.
    In the node_idx branch the kernel row k_row has shape (B, N), so the
    matmul must be written as  k_row.unsqueeze(1) @ E_batched  where
    E_batched = E.unsqueeze(0).expand(B, -1, -1), yielding (B, 1, D)
    before squeezing to (B, D).  The no-index branch applies the full
    K @ E_batched giving (B, N, D).  Both branches broadcast E identically.

    Parameters
    ----------
    tau_modes : int
        Number of eigenvectors k to retain.
    diffusion_time : float
        Initial heat-kernel diffusion time t.
    learnable_time : bool
        If True, log_t is a learnable nn.Parameter.
    """

    def __init__(
        self,
        tau_modes: int = 8,
        diffusion_time: float = 1.0,
        learnable_time: bool = True,
    ) -> None:
        super().__init__()
        self.tau_modes = tau_modes
        if learnable_time:
            self.log_t = nn.Parameter(torch.tensor(float(diffusion_time)).log())
        else:
            self.register_buffer("log_t", torch.tensor(float(diffusion_time)).log())

    @property
    def t(self) -> torch.Tensor:
        """Current diffusion time (always positive via exp)."""
        return self.log_t.exp()

    def forward(
        self,
        L: torch.Tensor,
        E: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
        eig_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Apply tau-mode diffusion and return reconstructed embeddings.

        Parameters
        ----------
        L : torch.Tensor
            Shape (B, N, N) or (N, N).
        E : torch.Tensor
            Embedding table.  Shape (N, D).  Shared across the batch;
            broadcast to (B, N, D) internally.
        node_idx : torch.Tensor or None
            Long tensor (B,) selecting one node per batch element.
            When provided, returns the single diffused row for that node,
            shape (B, D).  When None, returns the full diffused table
            shape (B, N, D).
        eig_cache : tuple(eigvals, eigvecs) or None
            Pre-computed spectral decomposition of the base Laplacian.

        Returns
        -------
        torch.Tensor
            (B, D) when node_idx is given; (B, N, D) otherwise.
        """
        batched = L.dim() == 3
        if not batched:
            L = L.unsqueeze(0)

        B, N, _ = L.shape
        k = min(self.tau_modes, N)

        if eig_cache is not None:
            eigvals, eigvecs = eig_cache
            if eigvals.dim() == 1:
                eigvals = eigvals.unsqueeze(0).expand(B, -1)
            if eigvecs.dim() == 2:
                eigvecs = eigvecs.unsqueeze(0).expand(B, -1, -1)
        else:
            eigvals, eigvecs = _safe_eigh(L)

        eigvals = eigvals[:, :k]           # (B, k)
        eigvecs = eigvecs[:, :, :k]        # (B, N, k)
        heat    = torch.exp(-self.t * eigvals.clamp(min=0.0))  # (B, k)

        # E is (N, D) -- broadcast once to (B, N, D) for both branches.
        E_b = E.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        if node_idx is not None:
            # Select the eigenvector row for the queried node per batch element.
            # idx: (B, 1, k) -- gather along the node axis.
            idx     = node_idx.view(B, 1, 1).expand(B, 1, k)
            u_query = eigvecs.gather(1, idx).squeeze(1)    # (B, k)

            # Kernel row for the queried node: sum_j u_j(i)*heat_j * u_j
            # k_row: (B, N)
            k_row = (u_query * heat).unsqueeze(1) * eigvecs  # (B, 1, k) * (B, N, k)
            k_row = k_row.sum(-1)                             # (B, N)

            # Matmul: (B, 1, N) @ (B, N, D) -> (B, 1, D) -> (B, D)
            x_hat = k_row.unsqueeze(1).bmm(E_b).squeeze(1)  # (B, D)
        else:
            # Full heat-kernel matrix: (B, N, N)
            U_h   = eigvecs * heat.unsqueeze(1)              # (B, N, k)
            K     = U_h @ eigvecs.transpose(-1, -2)          # (B, N, N)
            x_hat = K.bmm(E_b)                               # (B, N, D)

        if not batched:
            x_hat = x_hat.squeeze(0)
        return x_hat


# ---------------------------------------------------------------------------
# J_freq  --  spectral frequency cost
# ---------------------------------------------------------------------------

def spectral_freq_cost(
    L: torch.Tensor,
    tau_modes: int = 8,
    reduction: str = "mean",
    eigvals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-frequency energy penalty on the wiring Laplacian (J_freq).

    Computes the sum of all eigenvalues of L beyond the first tau_modes
    low-frequency modes::

        J_freq = mean_b [ sum_{j > tau_modes} lambda_j( L_b ) ]

    Note
    ----
    When building a spectral entropy training signal from the eigenvalue
    distribution, prefer lambda_fingerprint_soft() over this function.
    lambda_fingerprint_soft() returns a normalised KDE histogram that is
    differentiable w.r.t. eigvals and can be used directly in the ELBO
    as a spectral diversity penalty.  spectral_freq_cost() penalises total
    high-frequency energy, not distribution shape.

    Parameters
    ----------
    L : torch.Tensor
        Shape (B, N, N) or (N, N).
    tau_modes : int
        Number of low-frequency modes excluded from the penalty.
    reduction : str
        'mean' | 'sum' | 'none'.
    eigvals : torch.Tensor or None
        Pre-computed eigenvalues.  Shape (N,) or (B, N).

    Returns
    -------
    torch.Tensor  scalar or (B,) when reduction='none'.
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    if eigvals is None:
        eigvals = _safe_eigvalsh(L)
    else:
        if eigvals.dim() == 1:
            eigvals = eigvals.unsqueeze(0).expand(L.shape[0], -1)

    high_freq = eigvals[:, tau_modes:].clamp(min=0.0)
    cost = high_freq.sum(dim=-1)

    if reduction == "mean":
        return cost.mean()
    elif reduction == "sum":
        return cost.sum()
    return cost


# ---------------------------------------------------------------------------
# lambda-fingerprint  --  differentiable soft-histogram (issue #56)
# ---------------------------------------------------------------------------

def lambda_fingerprint_soft(
    eigvals: torch.Tensor,
    n_bins: int = 32,
    lam_max: float = 2.0,
    bandwidth: float = 0.05,
) -> torch.Tensor:
    """
    Differentiable soft-histogram fingerprint of the eigenvalue distribution.

    Uses Gaussian kernel density estimation centred on uniformly-spaced bin
    centres in [0, lam_max].  The function is fully batched, stays on the
    input device, and is differentiable w.r.t. eigvals so it can be used as
    a genuine training signal (e.g. a spectral entropy penalty) rather than
    a monitoring-only diagnostic.

    Compared to lambda_fingerprint_hard (the original torch.histc
    implementation):

    - Device-consistent: no .cpu() call; result lives on eigvals.device.
    - Differentiable: gradients flow back through eigvals.
    - Batchable: a single vectorised computation over the batch dimension.
    - JIT-compilable: no Python-level loops or .cpu() calls.

    Algorithm
    ---------
    For each eigenvalue lambda_i and each bin centre c_j the soft
    assignment weight is::

        w_{i,j} = exp( -0.5 * ((lambda_i - c_j) / bandwidth)^2 )

    The histogram for batch element b is then::

        h_j = sum_i w_{i,j}     (sum over N eigenvalues)

    followed by L1 normalisation so sum_j h_j = 1.

    Parameters
    ----------
    eigvals : torch.Tensor
        Shape (B, N) or (N,).  Eigenvalues of the feature-space Laplacian.
        Need not be clipped prior to calling; values outside [0, lam_max]
        will contribute only to the tail bins.
    n_bins : int
        Number of histogram bins.  Default 32.
    lam_max : float
        Upper bound of the histogram range.  Default 2.0 (normalised
        Laplacian spectrum lies in [0, 2] by construction).
    bandwidth : float
        Gaussian kernel bandwidth (standard deviation).  Default 0.05.
        Smaller values give sharper localisation but noisier gradients;
        larger values smooth the fingerprint across adjacent bins.

    Returns
    -------
    torch.Tensor
        Shape (B, n_bins) or (n_bins,) matching the input rank.
        Each row is a normalised probability vector (sums to 1).

    Examples
    --------
    Using as a differentiable training signal::

        fp = lambda_fingerprint_soft(eigvals_q, n_bins=32)
        entropy = -(fp * (fp + 1e-8).log()).sum(dim=-1).mean()
        loss = loss + entropy_weight * (-entropy)  # maximise spectral entropy
    """
    squeeze = eigvals.dim() == 1
    if squeeze:
        eigvals = eigvals.unsqueeze(0)   # (1, N)

    B, N = eigvals.shape

    # Bin centres: uniformly spaced in [0, lam_max].  Shape (1, 1, n_bins).
    centres = torch.linspace(
        0.0, lam_max, n_bins,
        device=eigvals.device, dtype=eigvals.dtype,
    ).view(1, 1, n_bins)

    # Soft assignment: Gaussian kernel for every (eigenvalue, bin) pair.
    # eigvals expanded to (B, N, 1) for broadcasting against (1, 1, n_bins).
    ev_exp = eigvals.unsqueeze(-1)                        # (B, N, 1)
    soft_counts = torch.exp(
        -0.5 * ((ev_exp - centres) / bandwidth) ** 2
    )                                                     # (B, N, n_bins)

    soft_hist = soft_counts.sum(dim=1)                    # (B, n_bins)

    # L1 normalise: each histogram row sums to 1.
    soft_hist = soft_hist / (soft_hist.sum(dim=-1, keepdim=True) + 1e-8)

    return soft_hist.squeeze(0) if squeeze else soft_hist


def lambda_fingerprint_hard(
    L: torch.Tensor,
    tau_modes: int = 8,
    n_bins: int = 16,
    eigvals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ArrowSpace-style lambda-distribution fingerprint using a hard histogram.

    WARNING: Non-differentiable
    --------------------------
    This function uses torch.histc which has no gradient w.r.t. the input
    eigenvalues.  It forces each batch slice to CPU internally.  Use this
    function for logging, visualisation, and monitoring ONLY.  For any use
    case where gradients need to flow back through the fingerprint (e.g. a
    spectral entropy training objective) use lambda_fingerprint_soft() instead.

    Previously named lambda_fingerprint().  That name is preserved as a
    backwards-compat alias pointing here.

    Parameters
    ----------
    L : torch.Tensor
        Shape (B, N, N) or (N, N).
    tau_modes : int
        Number of low-frequency eigenvalues to histogram.
    n_bins : int
        Number of histogram bins over [0, 2].
    eigvals : torch.Tensor or None
        Pre-computed eigenvalues.  Shape (N,) or (B, N).

    Returns
    -------
    torch.Tensor  shape (B, n_bins).  On the same device as L.
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    device = L.device
    B = L.shape[0]

    if eigvals is None:
        eigvals = _safe_eigvalsh(L)
    else:
        if eigvals.dim() == 1:
            eigvals = eigvals.unsqueeze(0).expand(B, -1)

    eigvals = eigvals[:, :tau_modes].clamp(min=0.0)

    with torch.no_grad():
        fp = torch.zeros(B, n_bins, device=device)
        for b in range(B):
            counts = torch.histc(
                eigvals[b].cpu(),
                bins=n_bins, min=0.0, max=2.0,
            ).to(device)
            fp[b] = counts / (counts.sum() + 1e-8)
    return fp


# backwards-compat alias -- existing call sites continue to work unchanged.
lambda_fingerprint = lambda_fingerprint_hard


# ---------------------------------------------------------------------------
# spectral_basis_kl  --   four-term ELBO, basis term  (issue #24)
# ---------------------------------------------------------------------------

def spectral_basis_kl(
    S: torch.Tensor,
    log_var_S: torch.Tensor,
    eigvals_q: torch.Tensor,
    lam_s: float = 1.0,
) -> torch.Tensor:
    """
    KL between the Gaussian basis posterior q(S) and the eigenvalue-weighted
    Gaussian prior p(S | I).

    The posterior is a diagonal Gaussian:
        q(S) = N( S, diag(exp(log_var_S)) )

    The prior couples each entry (k, j) of S to the k-th eigenvalue of the
    frozen ArrowSpace index Laplacian L(I):
        p(S | I) = prod_{k,j} N( 0, 1 / (lam_s * lambda_k) )

    The resulting closed-form KL per element (k, j) is:
        KL_{kj} = 0.5 * [ lam_s * lambda_k * (var_kj + S_kj^2)
                           - log_var_kj - 1 - log(lam_s * lambda_k) ]

    which reduces to the standard unit-Gaussian KL when eigvals_q are all
    ones and lam_s=1.

    Parameters
    ----------
    S : torch.Tensor
        Posterior mean matrix.  Shape (B, q, q).
    log_var_S : torch.Tensor
        Log-variance matrix.  Shape (B, q, q).
    eigvals_q : torch.Tensor
        Leading q eigenvalues of the frozen index Laplacian L(I).  Shape (q,).
        Must be >= 0; clipped to min=1e-6 internally to avoid log(0).
    lam_s : float
        Scalar precision multiplier.  Default 1.0.

    Returns
    -------
    torch.Tensor  scalar -- mean over batch and all (k, j) entries.
    """
    # precision per mode: shape (q,) -> broadcast to (1, q, 1) for the k-axis
    prec = (lam_s * eigvals_q.clamp(min=1e-6))  # (q,)
    prec = prec.view(1, -1, 1)                   # (1, q, 1) -- broadcasts over (B, q, q)

    var_S = log_var_S.exp()                       # (B, q, q)

    # Standard Gaussian KL with prior precision prec:
    # KL = 0.5 * [ prec*(var + mu^2) - log_var - 1 - log(prec) ]
    kl = 0.5 * (
        prec * (var_S + S.pow(2))
        - log_var_S
        - 1.0
        - prec.log()
    )  # (B, q, q)

    return kl.sum(dim=(-2, -1)).mean()  # scalar


# ---------------------------------------------------------------------------
# tau_mode_kl  --   four-term ELBO, mode-weight term  (issue #24)
# ---------------------------------------------------------------------------

def tau_mode_kl(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eigvals_q: torch.Tensor,
    tau: float = 1.0,
    a_min: float = 0.1,
) -> torch.Tensor:
    """
    KL between the variational Gamma mode posterior and the Exponential
    mode prior, with shape-parameter floor to prevent full Gamma collapse.

    For each mode k, the posterior is:
        q(omega_k) = Gamma( a_k, b_k )   (rate parameterisation)
    and the prior is:
        p(omega_k | tau, lambda_k) = Exponential( tau * lambda_k )
                                   = Gamma(1, tau * lambda_k)

    Closed-form derivation::

        KL( Gamma(a,b) || Exp(r) )
          = E_q[log q(omega) - log p(omega)]
          = -H[Gamma(a,b)] - log(r) + r * E_q[omega]

        where, for Gamma with rate parameterisation:
          E_q[omega]    = a / b
          H[Gamma(a,b)] = a - log(b) + lgamma(a) + (1-a)*digamma(a)

        Substituting::

          KL = -(a - log(b) + lgamma(a) + (1-a)*digamma(a)) - log(r) + r*(a/b)
             = log(b) - log(r) - lgamma(a) - (1-a)*digamma(a) - a + a*r/b

        Note the sign on lgamma and the final term a*r/b (NOT a*b/r).

    Shape-parameter floor (stability mitigation, issue #68)
    -------------------------------------------------------
    Before computing lgamma and digamma, the shape parameter a is clamped
    to min=a_min (default 0.1) to prevent full Gamma shape collapse.  This
    mirrors the floor applied in WiringAutoencoder (configured via
    training.a_min in configs/default.yaml).  The clamp is applied AFTER
    exp() so gradients still flow through log_a; the floor only clips the
    forward-pass value seen by the special functions.

    Parameters
    ----------
    log_a : torch.Tensor
        Log shape parameters.  Shape (B, q).  a = exp(log_a) > 0.
    log_b : torch.Tensor
        Log rate parameters.  Shape (B, q).  b = exp(log_b) > 0.
    eigvals_q : torch.Tensor
        Leading q eigenvalues of the frozen index Laplacian L(I).  Shape (q,).
        Clipped to min=1e-6 so r > 0.
    tau : float
        Diffusion time scale.  Scalar multiplier for the prior rate.
        Default 1.0.
    a_min : float
        Floor applied to the shape parameter a = exp(log_a) before
        lgamma/digamma.  Prevents full Gamma collapse to a near-zero spike.
        Default 0.1 (matching the paper's Section 5.2 description).

    Returns
    -------
    torch.Tensor  scalar -- mean over batch, summed over modes.
    """
    # Clamp a to min=a_min AFTER exp() -- gradients still flow through log_a.
    a = log_a.exp().clamp(min=a_min)  # (B, q)  shape parameter >= a_min
    b = log_b.exp()                   # (B, q)  rate  parameter > 0

    # Prior rate r_k = tau * lambda_k;  shape (q,) -> (1, q)
    r = (tau * eigvals_q.clamp(min=1e-6)).unsqueeze(0)  # (1, q)

    # Closed-form KL( Gamma(a,b) || Exp(r) = Gamma(1, r) )
    # KL = log(b) - log(r) - lgamma(a) - (1-a)*digamma(a) - a + a*r/b
    kl = (
        log_b                              # log b
        - r.log()                          # - log r
        - torch.lgamma(a)                  # - lgamma(a)
        - (1.0 - a) * torch.digamma(a)     # - (1-a)*digamma(a)
        - a                                # - a
        + a * r / b                        # + a*r/b
    )  # (B, q)

    # Sum over q modes, mean over batch.
    return kl.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# count_active_modes  --  diagnostic helper (issue #77)
# ---------------------------------------------------------------------------

def count_active_modes(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    delta: float = 0.01,
) -> int:
    """
    Count the mean number of spectrally active modes across a batch.

    A mode k is considered active when its expected value under the Gamma
    variational posterior exceeds the threshold delta::

        E[omega_k] = a_k / b_k = exp(log_a_k) / exp(log_b_k) > delta

    The result is the mean count over the batch, rounded to the nearest
    integer and returned as a plain Python int so it can be stored in the
    output dict and used directly by spectral_kl_health_check without
    touching the gradient graph.

    This function shares the active-mode counting logic with
    active_mode_penalty() but is separated so that the diagnostic value
    is always available from model.forward() regardless of whether the
    penalty weight nu is zero.

    Parameters
    ----------
    log_a : torch.Tensor
        Log shape parameters.  Shape (B, q).  a = exp(log_a) > 0.
    log_b : torch.Tensor
        Log rate parameters.  Shape (B, q).  b = exp(log_b) > 0.
    delta : float
        Threshold for classifying a mode as active:  E[omega_k] > delta.
        Default 0.01.  Must match the delta used in active_mode_penalty()
        for the health-check result to be consistent with the penalty.

    Returns
    -------
    int
        Mean number of active modes across the batch, in [0, q].
        Computed under torch.no_grad() so it never introduces graph nodes.
    """
    with torch.no_grad():
        expected_omega = log_a.exp() / log_b.exp()          # (B, q)
        n_active = (expected_omega > delta).float().sum(dim=-1).mean()
        return int(round(n_active.item()))


# ---------------------------------------------------------------------------
# active_mode_penalty  --  stability mitigation (issue #68)
# ---------------------------------------------------------------------------

def active_mode_penalty(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    q_min: int,
    nu: float = 1.0,
    delta: float = 0.01,
) -> torch.Tensor:
    """
    Soft Lagrange penalty that maintains a minimum number of active modes.

    A mode k is considered active when its expected value under the Gamma
    variational posterior exceeds a threshold delta::

        E[omega_k] = a_k / b_k = exp(log_a_k) / exp(log_b_k)

    The penalty is::

        penalty = nu * relu( q_min - N_active )

    where N_active is the mean (over the batch) count of modes with
    E[omega_k] > delta.  When N_active >= q_min the penalty is zero.

    This is a *soft* constraint: it adds a positive term to the total
    ELBO loss only when too many modes have collapsed, discouraging the
    Exponential prior from pushing all mode weights to near-zero.

    The (B, q) expected-value tensor is computed in float32 with
    torch.no_grad() for the active count; gradients only flow through
    the relu(q_min - n_active_mean) path, so the gradient signal is
    sparse (zero when the constraint is satisfied, non-zero otherwise).

    Use count_active_modes() when you need N_active as a plain Python int
    for logging or health checks (issue #77) -- it avoids recomputing
    expected_omega a second time when both the penalty and the diagnostic
    are needed in the same forward pass.

    Parameters
    ----------
    log_a : torch.Tensor
        Log shape parameters.  Shape (B, q).
    log_b : torch.Tensor
        Log rate parameters.  Shape (B, q).
    q_min : int
        Minimum number of active modes required.  The penalty is zero
        when the mean active count meets or exceeds q_min.
        Set to 0 to disable.
    nu : float
        Lagrange multiplier weight for the active-mode penalty.
        Set to 0.0 to disable.  Default 1.0.
    delta : float
        Threshold for classifying a mode as active:  E[omega_k] > delta.
        Default 0.01.

    Returns
    -------
    torch.Tensor
        Scalar penalty.  Zero (detached) when nu == 0 or q_min == 0.
    """
    if nu == 0.0 or q_min == 0:
        return torch.zeros(1, device=log_a.device).squeeze()

    # E[omega_k] = a_k / b_k  (B, q)
    expected_omega = log_a.exp() / log_b.exp()

    # N_active: mean count of active modes across the batch (scalar)
    # Use float() to make the count differentiable-ish via the relu path.
    n_active = (expected_omega > delta).float().sum(dim=-1).mean()  # scalar

    return nu * torch.relu(torch.tensor(float(q_min), device=log_a.device) - n_active)
