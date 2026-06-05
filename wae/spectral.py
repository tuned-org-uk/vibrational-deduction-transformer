"""
Spectral utilities for the Wiring Autoencoder.

Provides:
    TauModeDiffusion   — differentiable truncated spectral diffusion
    spectral_freq_cost — J_freq high-frequency energy penalty
    lambda_fingerprint — ArrowSpace-style λ-distribution summary

Device note
-----------
All torch.linalg.eigh / eigvalsh calls are offloaded to CPU (MPS does not
implement aten::_linalg_eigh as of PyTorch 2.x).  Every call routes through
_safe_eigh() / _safe_eigvalsh(), which apply two pre-conditioning steps before
handing the matrix to LAPACK:

    1. Symmetrise: L = (L + L.T) / 2
       The sparse COO -> dense path in DifferentiableLaplacian can leave
       tiny (~1e-7) asymmetry residuals from float32 scatter_add accumulation.
       LAPACK dsyevd assumes exact symmetry; violations cause error code 2707.

    2. Tikhonov shift: L += eps * I   (default eps=1e-4)
       Shifts all eigenvalues away from zero, giving LAPACK a well-separated
       spectrum and preventing repeated-eigenvalue failures.  eps=1e-4 is
       far below spectral resolution but well above the float32 noise floor.

A try/except fallback routes to torch.linalg.eig (non-symmetric solver) if
the symmetric solver still fails, returning real parts sorted ascending so
training is never interrupted by a single degenerate batch element.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Tikhonov regularisation strength applied before every eigensolver call.
_EIGSOLVE_EPS: float = 1e-4


# ---------------------------------------------------------------------------
# Safe eigensolver helpers
# ---------------------------------------------------------------------------

def _precondition(L: torch.Tensor) -> torch.Tensor:
    """
    Symmetrise L and add a small diagonal shift to prevent LAPACK
    convergence failures on ill-conditioned or nearly-degenerate matrices.

    Works on a single matrix (N, N) or a batch (B, N, N).
    Always returns a CPU tensor (caller is responsible for .to(device)).
    """
    L_cpu = L.detach().cpu().float()  # ensure float32, on CPU
    # 1. Symmetrise
    L_sym = (L_cpu + L_cpu.transpose(-2, -1)) * 0.5
    # 2. Tikhonov diagonal shift
    n = L_sym.shape[-1]
    eye = torch.eye(n, dtype=L_sym.dtype)  # (N, N)
    if L_sym.dim() == 3:
        eye = eye.unsqueeze(0)             # (1, N, N)
    return L_sym + _EIGSOLVE_EPS * eye


def _safe_eigvalsh(L: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a symmetric matrix (or batch) with fallback.

    Parameters
    ----------
    L : Tensor  (B, N, N) or (N, N)  — on any device

    Returns
    -------
    eigvals : Tensor  (B, N) or (N,)  — on the same device as input, ascending
    """
    device = L.device
    L_pre  = _precondition(L)   # CPU float32, symmetrised + shifted
    try:
        ev = torch.linalg.eigvalsh(L_pre)  # (B, N) or (N,) ascending
    except torch._C._LinAlgError:
        # Fallback: non-symmetric solver, take real parts
        ev_complex = torch.linalg.eigvals(L_pre)
        ev = ev_complex.real
        ev, _ = ev.sort(dim=-1)
    return ev.to(device)


def _safe_eigh(L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eigenvalues + eigenvectors of a symmetric matrix (or batch)
    with fallback.

    Parameters
    ----------
    L : Tensor  (B, N, N) or (N, N)  — on any device

    Returns
    -------
    eigvals  : Tensor  (B, N) or (N,)    — on same device as input, ascending
    eigvecs  : Tensor  (B, N, N) or (N, N) — on same device as input
    """
    device = L.device
    L_pre  = _precondition(L)   # CPU float32
    try:
        ev, evec = torch.linalg.eigh(L_pre)
    except torch._C._LinAlgError:
        # Fallback: general solver, sort by real part of eigenvalue
        ev_complex, evec_complex = torch.linalg.eig(L_pre)
        order = ev_complex.real.argsort(dim=-1)
        ev   = ev_complex.real.gather(-1, order)
        # gather eigvec columns: evec shape (..., N, N)
        order_v = order.unsqueeze(-2).expand_as(evec_complex.real)
        evec = evec_complex.real.gather(-1, order_v)
    return ev.to(device), evec.to(device)


# ---------------------------------------------------------------------------
# TauModeDiffusion
# ---------------------------------------------------------------------------
class TauModeDiffusion(nn.Module):
    """
    Truncated spectral diffusion using the k lowest-frequency eigenvectors
    of L(z) (tau-mode approximation).

    Given Laplacian L and embedding table E  (N, D)::

        U, Λ = eig_k(L)                   # (N, k), (k,)
        K_tau = U · exp(-tΛ) · Uᵀ         # (N, N)  diffusion kernel
        x̂_i  = K_tau[i] · E               # (D,)    diffused embedding

    The eigenvector computation is differentiable via torch.linalg.eigh.
    The decomposition is offloaded to CPU for MPS compatibility and moved
    back to the original device before any further computation.

    Parameters
    ----------
    tau_modes : int
        Number of eigenvectors k to retain.
    diffusion_time : float
        Diffusion time t (heat kernel scale).
    learnable_time : bool
        If True, t is a learnable scalar parameter.
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
        return self.log_t.exp()

    def forward(
        self,
        L: torch.Tensor,
        E: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        L        : (B, N, N) or (N, N)
        E        : (N, D)
        node_idx : (B,) or None — if given return (B, D), else (B, N, D)
        """
        batched = L.dim() == 3
        if not batched:
            L = L.unsqueeze(0)

        B, N, _ = L.shape
        device   = L.device
        k        = min(self.tau_modes, N)

        # Safe eigen-decomposition (symmetrise + shift + CPU offload)
        eigvals, eigvecs = _safe_eigh(L)        # both on `device`
        eigvals = eigvals[:, :k]                 # (B, k)
        eigvecs = eigvecs[:, :, :k]              # (B, N, k)

        heat = torch.exp(-self.t * eigvals.clamp(min=0.0))  # (B, k)

        if node_idx is not None:
            idx     = node_idx.view(B, 1, 1).expand(B, 1, k)
            u_query = eigvecs.gather(1, idx).squeeze(1)      # (B, k)
            k_row   = (u_query * heat).unsqueeze(1) * eigvecs # (B, N, k)
            k_row   = k_row.sum(-1)                           # (B, N)
            x_hat   = k_row @ E                               # (B, D)
        else:
            U_h   = eigvecs * heat.unsqueeze(1)               # (B, N, k)
            K     = U_h @ eigvecs.transpose(-1, -2)           # (B, N, N)
            x_hat = K @ E.unsqueeze(0).expand(B, -1, -1)      # (B, N, D)

        if not batched:
            x_hat = x_hat.squeeze(0)
        return x_hat


# ---------------------------------------------------------------------------
# J_freq — spectral frequency cost
# ---------------------------------------------------------------------------
def spectral_freq_cost(
    L: torch.Tensor,
    tau_modes: int = 8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    High-frequency energy penalty on the wiring Laplacian.

    J_freq = mean_b [ sum_{j > tau_modes} lambda_j( L_b ) ]

    Minimising J_freq encourages the learned wiring to concentrate energy
    in low-frequency modes (smooth wiring, low-entropy vibrational state).

    Parameters
    ----------
    L         : Tensor  (B, N, N) or (N, N)
    tau_modes : int     low-frequency modes excluded from penalty
    reduction : 'mean' | 'sum' | 'none'
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    device    = L.device
    eigvals   = _safe_eigvalsh(L)                        # (B, N) ascending, on device
    high_freq = eigvals[:, tau_modes:].clamp(min=0.0)    # (B, N-k)
    cost      = high_freq.sum(dim=-1)                    # (B,)

    if reduction == "mean":
        return cost.mean()
    elif reduction == "sum":
        return cost.sum()
    return cost


# ---------------------------------------------------------------------------
# λ-fingerprint (ArrowSpace-style feature)
# ---------------------------------------------------------------------------
def lambda_fingerprint(
    L: torch.Tensor,
    tau_modes: int = 8,
    n_bins: int = 16,
) -> torch.Tensor:
    """
    Compute an ArrowSpace-style λ-distribution fingerprint.

    Returns a binned histogram of the lowest tau_modes eigenvalues,
    normalised to sum to 1.  Used as additional input to the encoder
    to enrich z with spectral structure of the current wiring.

    Returns
    -------
    fp : Tensor  (B, n_bins)  on same device as input L
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    device  = L.device
    B       = L.shape[0]

    eigvals = _safe_eigvalsh(L)[:, :tau_modes].clamp(min=0.0)  # (B, k) on device

    with torch.no_grad():
        fp = torch.zeros(B, n_bins, device=device)
        for b in range(B):
            counts = torch.histc(
                eigvals[b].cpu(),
                bins=n_bins, min=0.0, max=2.0,
            ).to(device)
            fp[b] = counts / (counts.sum() + 1e-8)
    return fp
