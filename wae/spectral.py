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
handing the matrix to LAPACK.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

_EIGSOLVE_EPS: float = 1e-4


def _precondition(L: torch.Tensor) -> torch.Tensor:
    L_cpu = L.detach().cpu().float()
    L_sym = (L_cpu + L_cpu.transpose(-2, -1)) * 0.5
    n = L_sym.shape[-1]
    eye = torch.eye(n, dtype=L_sym.dtype)
    if L_sym.dim() == 3:
        eye = eye.unsqueeze(0)
    return L_sym + _EIGSOLVE_EPS * eye


def _safe_eigvalsh(L: torch.Tensor) -> torch.Tensor:
    device = L.device
    L_pre = _precondition(L)
    try:
        ev = torch.linalg.eigvalsh(L_pre)
    except torch._C._LinAlgError:
        ev_complex = torch.linalg.eigvals(L_pre)
        ev = ev_complex.real
        ev, _ = ev.sort(dim=-1)
    return ev.to(device)


def _safe_eigh(L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class TauModeDiffusion(nn.Module):
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
        eig_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
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

        eigvals = eigvals[:, :k]
        eigvecs = eigvecs[:, :, :k]
        heat = torch.exp(-self.t * eigvals.clamp(min=0.0))

        if node_idx is not None:
            idx = node_idx.view(B, 1, 1).expand(B, 1, k)
            u_query = eigvecs.gather(1, idx).squeeze(1)
            k_row = (u_query * heat).unsqueeze(1) * eigvecs
            k_row = k_row.sum(-1)
            x_hat = k_row @ E
        else:
            U_h = eigvecs * heat.unsqueeze(1)
            K = U_h @ eigvecs.transpose(-1, -2)
            x_hat = K @ E.unsqueeze(0).expand(B, -1, -1)

        if not batched:
            x_hat = x_hat.squeeze(0)
        return x_hat


def spectral_freq_cost(
    L: torch.Tensor,
    tau_modes: int = 8,
    reduction: str = "mean",
    eigvals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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


def lambda_fingerprint(
    L: torch.Tensor,
    tau_modes: int = 8,
    n_bins: int = 16,
    eigvals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
