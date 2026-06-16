"""
Spectral utilities for the Wiring Autoencoder.

This module implements the three spectral primitives that sit at the heart
of the VDT training loop:

    TauModeDiffusion    differentiable truncated spectral diffusion decoder
    spectral_freq_cost  J_freq high-frequency energy penalty (training signal)
    lambda_fingerprint  ArrowSpace-style lambda-distribution summary (encoder enrichment)

 additions (issue #24 / three-term ELBO):

    spectral_basis_kl   KL between Gaussian basis posterior q(S) and
                        eigenvalue-weighted prior p(S|I)
    tau_mode_kl         KL between Gamma mode posterior q(omega_k) and
                        Exponential mode prior p(omega_k | tau, lambda_k)

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
# lambda-fingerprint  --  ArrowSpace-style spectral feature
# ---------------------------------------------------------------------------

def lambda_fingerprint(
    L: torch.Tensor,
    tau_modes: int = 8,
    n_bins: int = 16,
    eigvals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ArrowSpace-style lambda-distribution fingerprint of the Laplacian spectrum.

    Computes a normalised histogram of the tau_modes lowest eigenvalues
    of L over [0, 2].  The result is concatenated to the encoder input to
    enrich the latent code z with the current wiring geometry.

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


# ---------------------------------------------------------------------------
# spectral_basis_kl  --   three-term ELBO, basis term  (issue #24)
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
# tau_mode_kl  --   three-term ELBO, mode-weight term  (issue #24)
# ---------------------------------------------------------------------------

def tau_mode_kl(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eigvals_q: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    KL between the variational Gamma mode posterior and the Exponential
    mode prior.

    For each mode k, the posterior is:
        q(omega_k) = Gamma( a_k, b_k )
    and the prior is:
        p(omega_k | tau, lambda_k) = Exponential( tau * lambda_k )
                                   = Gamma(1, tau * lambda_k)

    Closed-form KL (Gamma || Exponential)::

        KL( Gamma(a, b) || Exp(r) )
          = log(b) - log(r) + lgamma(a) + (1-a)*digamma(a) + a*b/r

    where r = tau * lambda_k.

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

    Returns
    -------
    torch.Tensor  scalar -- mean over batch, summed over modes.
    """
    a = log_a.exp()  # (B, q)  shape parameter > 0
    b = log_b.exp()  # (B, q)  rate  parameter > 0

    # Prior rate r_k = tau * lambda_k;  shape (q,) -> (1, q)
    r = (tau * eigvals_q.clamp(min=1e-6)).unsqueeze(0)  # (1, q)

    # Closed-form KL( Gamma(a,b) || Exp(r) = Gamma(1, r) )
    kl = (
        log_b                              # log b
        - r.log()                          # - log r
        + torch.lgamma(a)                  # lgamma(a)
        + (1.0 - a) * torch.digamma(a)     # (1 - a) * psi(a)
        + a * b / r                        # a * b / r
    )  # (B, q)

    # Sum over q modes, mean over batch.
    return kl.sum(dim=-1).mean()
