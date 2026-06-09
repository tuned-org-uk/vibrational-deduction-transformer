"""
Spectral utilities for the Wiring Autoencoder.

This module implements the three spectral primitives that sit at the heart
of the WAE training loop:

    TauModeDiffusion    differentiable truncated spectral diffusion decoder
    spectral_freq_cost  J_freq high-frequency energy penalty (training signal)
    lambda_fingerprint  ArrowSpace-style λ-distribution summary (encoder enrichment)

All three primitives share the same underlying eigensystem of the graph
Laplacian L(z).  To avoid redundant O(N³) CPU eigendecompositions at every
training step, every public function accepts an optional ``eigvals`` /
``eig_cache`` argument that allows the caller to pass precomputed spectral
quantities from outside the training loop.  See ``train.py`` for the
recommended caching pattern.

Architectural context
---------------------
See `docs/00-architecture.md
<https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md>`_
for the full data-flow diagram and module reference.

The J_freq cost and tau-mode diffusion kernel correspond directly to the
ArrowSpace spectral cost function described in the Module Reference section::

    J_freq = sum_{j > tau_modes} λ_j( L(z) )
    K_tau  = U_k · exp(-tΛ_k) · U_k^T           (heat kernel, k = tau_modes)

Stability note
--------------
All torch.linalg.eigh / eigvalsh calls are offloaded to CPU because MPS does
not implement ``aten::_linalg_eigh`` as of PyTorch 2.x.  Every entry routes
through ``_precondition()`` which applies two conditioning steps before
handing the matrix to LAPACK:

1. **Symmetrisation** ``L = (L + L^T) / 2``
   The sparse COO → dense path in ``DifferentiableLaplacian`` can leave tiny
   (~1e-7) asymmetry residuals from float32 ``scatter_add`` accumulation.
   LAPACK dsyevd assumes exact symmetry; violations cause error code 2707.

2. **Tikhonov shift** ``L += eps * I``  (``_EIGSOLVE_EPS = 1e-4``)
   Shifts all eigenvalues away from zero, giving LAPACK a well-separated
   spectrum and preventing repeated-eigenvalue failures.  The shift is far
   below spectral resolution but well above the float32 noise floor.

For the full stability analysis of eigenvalue conditioning, CFL bounds, and
the relationship between λ_max and the Courant criterion see
`docs/04-stability.md Section 3
<https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/04-stability.md#3-numerical-stability-of-the-wave-update>`_.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple


#: Small diagonal shift added to every matrix before passing to LAPACK.
#: Prevents repeated-eigenvalue failures on nearly degenerate Laplacians.
#: See docs/04-stability.md Section 3.2 for the CFL derivation that motivates
#: this preconditioner.
_EIGSOLVE_EPS: float = 1e-4


# ---------------------------------------------------------------------------
# Internal eigensolver helpers
# ---------------------------------------------------------------------------

def _precondition(L: torch.Tensor) -> torch.Tensor:
    """
    Symmetrise *L* and apply a Tikhonov diagonal shift before LAPACK.

    This is an internal helper called by ``_safe_eigh`` and
    ``_safe_eigvalsh``.  It always returns a CPU float32 tensor regardless
    of the input device.

    The two steps it performs are:

    1. **Symmetrisation** ``L_sym = (L + L^T) / 2`` — removes float32
       accumulation residuals left by the sparse COO path in
       ``DifferentiableLaplacian._sparse_laplacian``.
    2. **Tikhonov shift** ``L_sym += _EIGSOLVE_EPS * I`` — prevents
       LAPACK convergence failures on matrices with repeated zero
       eigenvalues (e.g. a graph Laplacian with multiple connected
       components).

    Parameters
    ----------
    L : torch.Tensor
        Symmetric (or nearly symmetric) matrix or batch.
        Shape ``(N, N)`` or ``(B, N, N)``.  May be on any device.

    Returns
    -------
    torch.Tensor
        Conditioned matrix in CPU float32.
        Same shape as input.
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

    Routes through ``_precondition`` before calling
    ``torch.linalg.eigvalsh``.  On failure (ill-conditioned spectrum),
    falls back to the general ``torch.linalg.eigvals`` solver, taking
    real parts and sorting ascending so training is never interrupted by
    a single degenerate batch element.

    Used by ``spectral_freq_cost`` and ``lambda_fingerprint`` when no
    pre-cached eigenvalues are supplied.

    Parameters
    ----------
    L : torch.Tensor
        Shape ``(B, N, N)`` or ``(N, N)``.  Any device.

    Returns
    -------
    torch.Tensor
        Eigenvalues ascending.  Shape ``(B, N)`` or ``(N,)``.
        Returned on the **same device as input**.
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
    Eigenvalues **and** eigenvectors of a symmetric matrix with fallback.

    Same conditioning and fallback strategy as ``_safe_eigvalsh`` but also
    returns the eigenvector matrix.  This is the more expensive call and
    should only be invoked **once per training run** for the fixed base
    graph (see ``train.py``).

    In the recommended caching pattern the result is stored as
    ``spectral_cache = (base_eigvals, base_eigvecs)`` and threaded into
    ``TauModeDiffusion.forward`` via the ``eig_cache`` argument, completely
    eliminating per-step eigensolver calls inside the training loop.

    Parameters
    ----------
    L : torch.Tensor
        Shape ``(B, N, N)`` or ``(N, N)``.  Any device.

    Returns
    -------
    eigvals : torch.Tensor
        Ascending eigenvalues.  Shape ``(B, N)`` or ``(N,)``.
    eigvecs : torch.Tensor
        Corresponding eigenvectors.  Shape ``(B, N, N)`` or ``(N, N)``.
        Column ``[:,k]`` is the k-th eigenvector.
    Both tensors are on the **same device as input**.
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
    of the Laplacian L(z).  This is the **tau-mode approximation** from the
    ArrowSpace spectral framework.

    Given Laplacian L and embedding table E (N, D) the diffusion kernel is::

        U, Λ = eig_k(L)                   # (N, k), (k,)  — k = tau_modes
        K_tau  = U · diag(exp(-tΛ)) · U^T  # (N, N)  heat kernel at time t
        x_hat_i = K_tau[i, :] · E          # (D,)    diffused embedding for node i

    The operator ``exp(-tΛ)`` is a heat kernel: it exponentially damps
    high-frequency modes.  Low-frequency modes (small λ_k) survive longest.
    Learnable ``log_t`` controls the diffusion time and is trained jointly
    with the rest of the model.

    Connection to WAE architecture
    --------------------------------
    ``TauModeDiffusion`` is the inner computation of ``DiffusionDecoder``
    and is called once per forward pass.  In the full ELBO::

        L_WAE = E_q[log p(x|z)] - β KL - α J_freq

    the reconstruction term ``log p(x|z)`` is computed through this module.
    See `docs/00-architecture.md
    <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#elbo-derivation>`_
    for the full ELBO derivation.

    Performance: ``eig_cache``
    --------------------------
    By default, ``forward`` calls ``_safe_eigh(L)`` which is an O(N³) CPU
    LAPACK operation (N ≈ 2708 for Cora, ~3 hours/epoch without caching).
    Pass a pre-computed ``eig_cache = (eigvals, eigvecs)`` obtained from the
    fixed ``base_L`` to skip this call entirely::

        # Once before training
        base_eigvals, base_eigvecs = _safe_eigh(base_L)
        spectral_cache = (base_eigvals, base_eigvecs)

        # Each forward pass
        x_hat = diffusion(L, E, node_idx=idx, eig_cache=spectral_cache)

    See ``train.py`` for the canonical usage pattern.

    Parameters
    ----------
    tau_modes : int
        Number of eigenvectors k to retain.
        Corresponds to the "tau-mode" count in the ArrowSpace spectral cost.
        Higher values capture more high-frequency structure at greater cost.
        Default: 8.
    diffusion_time : float
        Initial heat-kernel diffusion time t.  Learnable if
        ``learnable_time=True``.
    learnable_time : bool
        If True, ``log_t`` is registered as a ``nn.Parameter`` and trained
        jointly with the model.  If False, it is a frozen buffer.
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
            Laplacian matrix.  Shape ``(B, N, N)`` (batched) or ``(N, N)``.
        E : torch.Tensor
            Embedding table shared across the batch.  Shape ``(N, D)``.
        node_idx : torch.Tensor or None
            Long tensor of shape ``(B,)`` selecting one node per batch
            element.  When provided the diffusion row ``K_tau[i, :]`` is
            extracted for only that node, giving output shape ``(B, D)``.
            When ``None``, the full kernel ``K_tau`` is applied to all N
            nodes, giving output shape ``(B, N, D)``.
        eig_cache : tuple(eigvals, eigvecs) or None
            Pre-computed spectral decomposition of the **base** Laplacian.
            When supplied, the O(N³) ``_safe_eigh`` call is skipped entirely
            and these eigenvectors are used as a frozen approximation.
            Shape of eigvals: ``(N,)``; eigvecs: ``(N, N)``.
            If the cache is unbatched (2-D), it is automatically broadcast
            to batch size B.

        Returns
        -------
        torch.Tensor
            ``(B, D)``  when ``node_idx`` is provided  (per-node mode).
            ``(B, N, D)`` when ``node_idx`` is ``None``  (full-graph mode).
            ``(N, D)`` or ``(D,)`` when input was unbatched.
        """
        batched = L.dim() == 3
        if not batched:
            L = L.unsqueeze(0)

        B, N, _ = L.shape
        k = min(self.tau_modes, N)

        if eig_cache is not None:
            eigvals, eigvecs = eig_cache
            # Broadcast unbatched cache to batch size B
            if eigvals.dim() == 1:
                eigvals = eigvals.unsqueeze(0).expand(B, -1)
            if eigvecs.dim() == 2:
                eigvecs = eigvecs.unsqueeze(0).expand(B, -1, -1)
        else:
            eigvals, eigvecs = _safe_eigh(L)

        eigvals = eigvals[:, :k]          # (B, k) lowest-frequency eigenvalues
        eigvecs = eigvecs[:, :, :k]       # (B, N, k) corresponding eigenvectors
        heat = torch.exp(-self.t * eigvals.clamp(min=0.0))  # (B, k)

        if node_idx is not None:
            # Per-node path: extract row i of K_tau without building (B, N, N)
            idx = node_idx.view(B, 1, 1).expand(B, 1, k)
            u_query = eigvecs.gather(1, idx).squeeze(1)       # (B, k)
            k_row = (u_query * heat).unsqueeze(1) * eigvecs   # (B, N, k)
            k_row = k_row.sum(-1)                              # (B, N)
            x_hat = k_row @ E                                  # (B, D)
        else:
            # Full-graph path: K_tau = U diag(heat) U^T, then apply to E
            U_h = eigvecs * heat.unsqueeze(1)                  # (B, N, k)
            K = U_h @ eigvecs.transpose(-1, -2)               # (B, N, N)
            x_hat = K @ E.unsqueeze(0).expand(B, -1, -1)      # (B, N, D)

        if not batched:
            x_hat = x_hat.squeeze(0)
        return x_hat


# ---------------------------------------------------------------------------
# J_freq  —  spectral frequency cost
# ---------------------------------------------------------------------------

def spectral_freq_cost(
    L: torch.Tensor,
    tau_modes: int = 8,
    reduction: str = "mean",
    eigvals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-frequency energy penalty on the wiring Laplacian (``J_freq``).

    Computes the sum of all eigenvalues of ``L`` beyond the first
    ``tau_modes`` low-frequency modes::

        J_freq = mean_b [ Σ_{j > tau_modes} λ_j( L_b ) ]

    Minimising ``J_freq`` during training encourages the learned wiring
    ``L(z)`` to concentrate energy in low-frequency modes, producing
    smooth, low-entropy vibrational states.  This is the direct spectral
    analogue of tau-mode truncation in the ArrowSpace cost function.

    This term appears as the ``α J_freq`` component of the WAE-ELBO::

        L_WAE = E_q[log p(x|z)] - β KL - α J_freq(L(z))

    See `docs/00-architecture.md § spectral_freq_cost
    <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#waespectraspy--taumodediffusion-spectral_freq_cost-lambda_fingerprint>`_
    and `docs/04-stability.md § 4.2
    <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/04-stability.md#42-key-stability-parameters-to-monitor>`_
    for the stability analysis of spectral costs during optimisation.

    Performance: ``eigvals``
    ------------------------
    By default this function calls ``_safe_eigvalsh(L)`` which is an O(N³)
    CPU LAPACK operation executed on every training step.  Supply
    ``eigvals`` (precomputed from ``base_L``) to skip it::

        # Once
        base_eigvals, _ = _safe_eigh(base_L)   # (N,)

        # Every step
        j_freq = spectral_freq_cost(L, tau_modes=k, eigvals=base_eigvals)

    Parameters
    ----------
    L : torch.Tensor
        Laplacian.  Shape ``(B, N, N)`` or ``(N, N)``.
    tau_modes : int
        Number of low-frequency modes excluded from the penalty.
        Should match the ``tau_modes`` used in ``TauModeDiffusion``.
    reduction : str
        ``'mean'`` (default) — average J_freq over the batch.
        ``'sum'``  — sum over the batch.
        ``'none'`` — return per-batch-element cost, shape ``(B,)``.
    eigvals : torch.Tensor or None
        Pre-computed eigenvalues.  Shape ``(N,)`` (unbatched, broadcast to
        all B elements) or ``(B, N)`` (per-element).
        When supplied the eigensolver is not called.

    Returns
    -------
    torch.Tensor
        Scalar when ``reduction in ('mean', 'sum')``; shape ``(B,)`` when
        ``reduction='none'``.
    """
    batched = L.dim() == 3
    if not batched:
        L = L.unsqueeze(0)

    if eigvals is None:
        eigvals = _safe_eigvalsh(L)
    else:
        if eigvals.dim() == 1:
            eigvals = eigvals.unsqueeze(0).expand(L.shape[0], -1)

    high_freq = eigvals[:, tau_modes:].clamp(min=0.0)   # (B, N - tau_modes)
    cost = high_freq.sum(dim=-1)                          # (B,)

    if reduction == "mean":
        return cost.mean()
    elif reduction == "sum":
        return cost.sum()
    return cost


# ---------------------------------------------------------------------------
# λ-fingerprint  —  ArrowSpace-style spectral feature
# ---------------------------------------------------------------------------

def lambda_fingerprint(
    L: torch.Tensor,
    tau_modes: int = 8,
    n_bins: int = 16,
    eigvals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ArrowSpace-style λ-distribution fingerprint of the Laplacian spectrum.

    Computes a binned histogram of the ``tau_modes`` lowest eigenvalues
    of ``L``, normalised to sum to 1.  The result is a compact spectral
    summary that is concatenated to the encoder input to enrich the latent
    code ``z`` with information about the current wiring geometry.

    Conceptually this mirrors the λ-fingerprint computed by ArrowSpace
    notebooks 01–05 and described in the Module Reference of
    `docs/00-architecture.md
    <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#waespectraspy--taumodediffusion-spectral_freq_cost-lambda_fingerprint>`_.

    The fingerprint is **non-differentiable** (computed inside
    ``torch.no_grad()``) and is used as a static input feature to the
    encoder, not as a training signal.

    Performance: caching
    --------------------
    ``base_L`` is fixed throughout training, so its fingerprint is a
    constant tensor.  Compute it **once** before the training loop and
    reuse::

        # Once
        base_eigvals, _ = _safe_eigh(base_L)
        cached_fp = lambda_fingerprint(
            base_L, tau_modes=k, eigvals=base_eigvals
        )  # shape (1, n_bins)

        # Each step — broadcast to batch size B
        batch_fp = cached_fp.expand(B, -1)

    See ``train.py`` for the canonical pattern.

    Parameters
    ----------
    L : torch.Tensor
        Laplacian.  Shape ``(B, N, N)`` or ``(N, N)``.
    tau_modes : int
        Number of low-frequency eigenvalues to histogram.
    n_bins : int
        Number of histogram bins over [0, 2].
    eigvals : torch.Tensor or None
        Pre-computed eigenvalues.  Shape ``(N,)`` or ``(B, N)``.
        When supplied the eigensolver is not called.

    Returns
    -------
    torch.Tensor
        Normalised eigenvalue histogram.  Shape ``(B, n_bins)``.
        On the same device as ``L``.
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

    eigvals = eigvals[:, :tau_modes].clamp(min=0.0)   # (B, k)

    with torch.no_grad():
        fp = torch.zeros(B, n_bins, device=device)
        for b in range(B):
            counts = torch.histc(
                eigvals[b].cpu(),
                bins=n_bins, min=0.0, max=2.0,
            ).to(device)
            fp[b] = counts / (counts.sum() + 1e-8)
    return fp
