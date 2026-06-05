"""
Spectral utilities for the Wiring Autoencoder.

Provides:
    TauModeDiffusion   — differentiable truncated spectral diffusion
    spectral_freq_cost — J_freq high-frequency energy penalty (cf. tau-mode paper)
    lambda_fingerprint — ArrowSpace-style λ-distribution summary for a batch of nodes
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

    Given Laplacian L and embedding table E  (N, D):

        U, Λ = eig_k(L)                   # (N, k), (k,)
        K_tau = U · exp(-tΛ) · Uᵀ         # (N, N)  diffusion kernel
        x̂_i  = K_tau[i] · E               # (D,)    diffused embedding

    The eigenvector computation is differentiable via torch.linalg.eigh,
    which returns real eigenvalues for symmetric matrices.

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
        k = min(self.tau_modes, N)

        # Truncated eigen-decomposition — differentiable
        # eigh returns eigenvalues in ascending order (lambda_0 ~ 0 for conn. graph)
        eigvals, eigvecs = torch.linalg.eigh(L)   # (B, N), (B, N, N)
        eigvals = eigvals[:, :k]                   # (B, k)
        eigvecs = eigvecs[:, :, :k]                # (B, N, k)

        # Heat-kernel weights  exp(-t * lambda_j)
        heat = torch.exp(-self.t * eigvals.clamp(min=0.0))  # (B, k)

        # Diffusion kernel rows for the requested node(s)
        # K_tau[b, i, j] = sum_m U[b,i,m] * heat[b,m] * U[b,j,m]
        # We want K_tau[b, i, :] @ E  for the query node i

        if node_idx is not None:
            # Gather eigenvec rows for each batch element
            idx = node_idx.view(B, 1, 1).expand(B, 1, k)          # (B,1,k)
            u_query = eigvecs.gather(1, idx).squeeze(1)            # (B, k)
            # Weighted combination of all nodes via diffusion
            # k_row[b, j] = sum_m u_query[b,m] * heat[b,m] * eigvecs[b,j,m]
            k_row = (u_query * heat).unsqueeze(1) * eigvecs   # (B, N, k)
            k_row = k_row.sum(-1)                              # (B, N)
            # (B, N) @ (N, D) -> (B, D)
            x_hat = k_row @ E
        else:
            # Return all nodes  (B, N, D)
            # K_tau[b] = eigvecs[b] * heat[b] * eigvecs[b].T
            U_h = eigvecs * heat.unsqueeze(1)   # (B, N, k)
            K   = U_h @ eigvecs.transpose(-1, -2)  # (B, N, N)
            x_hat = K @ E.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

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
    energy in low-frequency modes (smooth wiring), matching the
    tau-mode truncation philosophy from the ArrowSpace cost function.

    Parameters
    ----------
    L : Tensor  shape (B, N, N) or (N, N)
    tau_modes : int  k low-frequency modes to *exclude* from penalty
    reduction : 'mean' | 'sum' | 'none'
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    eigvals = torch.linalg.eigvalsh(L)          # (B, N)  ascending
    high_freq = eigvals[:, tau_modes:].clamp(min=0.0)   # (B, N-k)
    cost = high_freq.sum(dim=-1)                         # (B,)

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
    Compute an ArrowSpace-style λ-distribution fingerprint for each graph in a batch.

    Returns a binned histogram of the lowest tau_modes eigenvalues,
    normalised to sum to 1, used as additional input to the encoder.

    Returns
    -------
    fp : Tensor  shape (B, n_bins)
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    B = L.shape[0]
    eigvals = torch.linalg.eigvalsh(L)[:, :tau_modes].clamp(min=0.0)  # (B, k)

    # Hard histogram — not differentiable, used only in encoder input
    with torch.no_grad():
        fp = torch.zeros(B, n_bins, device=L.device)
        edges = torch.linspace(0.0, 2.0, n_bins + 1, device=L.device)
        for b in range(B):
            counts = torch.histc(eigvals[b], bins=n_bins, min=0.0, max=2.0)
            fp[b] = counts / (counts.sum() + 1e-8)
    return fp
