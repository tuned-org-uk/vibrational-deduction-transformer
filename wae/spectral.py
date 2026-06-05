"""
Spectral utilities for the Wiring Autoencoder.

Provides:
    TauModeDiffusion   — differentiable truncated spectral diffusion
    spectral_freq_cost — J_freq high-frequency energy penalty
    lambda_fingerprint — ArrowSpace-style λ-distribution summary

Device note
-----------
All torch.linalg.eigh / eigvalsh calls are explicitly offloaded to CPU
before the decomposition and the result is moved back to the original
device afterwards.  This is required because aten::_linalg_eigh is not
implemented for Apple MPS (as of PyTorch 2.x) and raises
NotImplementedError at runtime.  The pattern:

    eigvals = torch.linalg.eigvalsh(L.cpu()).to(device)

does not affect gradient flow — autograd traces through .to() correctly
— and adds only a small host-device round-trip per forward pass.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


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
        tau_modes: int = 16,
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
        L: torch.Tensor,      # (B, N, N) or (N, N)
        E: torch.Tensor,      # (N, D)
        node_idx: Optional[torch.Tensor] = None,  # (B,) or None → return all
    ) -> torch.Tensor:
        """
        Returns
        -------
        x_hat : Tensor  shape (B, D) if node_idx given, else (B, N, D)
        """
        batched = L.dim() == 3
        if not batched:
            L = L.unsqueeze(0)

        B, N, _ = L.shape
        device = L.device
        k = min(self.tau_modes, N)

        # Truncated eigen-decomposition — offload to CPU for MPS compat.
        # autograd traces through .to() so gradients are preserved.
        eigvals, eigvecs = torch.linalg.eigh(L.cpu())
        eigvals = eigvals.to(device)[:, :k]    # (B, k)
        eigvecs = eigvecs.to(device)[:, :, :k] # (B, N, k)

        # Heat-kernel weights  exp(-t * lambda_j)
        heat = torch.exp(-self.t * eigvals.clamp(min=0.0))  # (B, k)

        if node_idx is not None:
            # Per-node mode: gather eigenvec rows for each batch element
            idx = node_idx.view(B, 1, 1).expand(B, 1, k)   # (B, 1, k)
            u_query = eigvecs.gather(1, idx).squeeze(1)     # (B, k)
            # k_row[b, j] = sum_m u_query[b,m] * heat[b,m] * eigvecs[b,j,m]
            k_row = (u_query * heat).unsqueeze(1) * eigvecs  # (B, N, k)
            k_row = k_row.sum(-1)                            # (B, N)
            x_hat = k_row @ E                                # (B, D)
        else:
            # Full-graph mode: all nodes  (B, N, D)
            U_h = eigvecs * heat.unsqueeze(1)                # (B, N, k)
            K   = U_h @ eigvecs.transpose(-1, -2)            # (B, N, N)
            x_hat = K @ E.unsqueeze(0).expand(B, -1, -1)     # (B, N, D)

        if not batched:
            x_hat = x_hat.squeeze(0)
        return x_hat


# ---------------------------------------------------------------------------
# J_freq — spectral frequency cost
# ---------------------------------------------------------------------------
def spectral_freq_cost(
    L: torch.Tensor,
    tau_modes: int = 16,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    High-frequency energy penalty on the wiring Laplacian.

    J_freq = mean over batch of  sum_{j > tau_modes} lambda_j(L)

    Minimising J_freq encourages the learned wiring to concentrate
    energy in low-frequency modes (smooth wiring).

    The eigensolver is offloaded to CPU for MPS compatibility.

    Parameters
    ----------
    L : Tensor  shape (B, N, N) or (N, N)
    tau_modes : int  k low-frequency modes to *exclude* from penalty
    reduction : 'mean' | 'sum' | 'none'
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    device = L.device
    # Offload eigensolver to CPU — not implemented on MPS
    eigvals = torch.linalg.eigvalsh(L.cpu()).to(device)  # (B, N) ascending
    high_freq = eigvals[:, tau_modes:].clamp(min=0.0)    # (B, N-k)
    cost = high_freq.sum(dim=-1)                          # (B,)

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
    tau_modes: int = 16,
    n_bins: int = 16,
) -> torch.Tensor:
    """
    Compute an ArrowSpace-style λ-distribution fingerprint.

    Returns a binned histogram of the lowest tau_modes eigenvalues,
    normalised to sum to 1, used as additional input to the encoder.

    The eigensolver is offloaded to CPU for MPS compatibility.
    Output is always returned on the same device as the input L.

    Returns
    -------
    fp : Tensor  shape (B, n_bins)
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    device = L.device
    B = L.shape[0]

    # Offload eigensolver to CPU — not implemented on MPS
    eigvals = torch.linalg.eigvalsh(L.cpu()).to(device)[:, :tau_modes].clamp(min=0.0)

    with torch.no_grad():
        fp = torch.zeros(B, n_bins, device=device)
        edges = torch.linspace(0.0, 2.0, n_bins + 1, device=device)
        for b in range(B):
            counts = torch.histc(
                eigvals[b].cpu(),   # histc not on MPS either
                bins=n_bins, min=0.0, max=2.0
            ).to(device)
            fp[b] = counts / (counts.sum() + 1e-8)
    return fp
