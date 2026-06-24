"""
Spectral KL helpers, diffusion, and fingerprint utilities for the Wiring Autoencoder.

This module contains all penalty and KL-divergence functions that
operate on the Gamma-distributed spectral mode posterior (log_a, log_b)
and on the spectral loading matrix S, plus the heat-kernel diffusion
decoder and lambda-fingerprint helpers.

Public API
----------
build_laplacian          -- convenience wrapper around DifferentiableLaplacian
spectral_basis_kl        -- KL( q(S) || p(S|I) ), Term 3 of the ELBO
tau_mode_kl              -- KL( q(w) || p(w|tau,Lambda) ), Term 4
mode_entropy_penalty     -- Option D active-mode entropy ceiling penalty (#82)
active_mode_penalty      -- active-mode floor penalty (#68)
count_active_modes       -- diagnostic: mean E[omega_k] > delta count
spectral_kl_health_check -- health-check dict: mode_collapse / mode_explosion / kl_S_ok
TauModeDiffusion         -- learnable heat-kernel decoder nn.Module
spectral_freq_cost       -- frequency regularisation scalar for L(z)
lambda_fingerprint_hard  -- non-differentiable Laplacian spectrum histogram
lambda_fingerprint_soft  -- differentiable (KDE) Laplacian spectrum histogram
lambda_fingerprint       -- backwards-compat alias for lambda_fingerprint_hard
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# spectral_basis_kl  (Term 3: KL_S)
# ---------------------------------------------------------------------------

def spectral_basis_kl(
    S: torch.Tensor,
    log_var_S: torch.Tensor,
    eigvals_q: torch.Tensor,
    lam_s: float = 0.01,
) -> torch.Tensor:
    """
    KL divergence for the spectral loading matrix: KL( q(S) || p(S|I) ).

    The posterior q(S) is a diagonal Gaussian with mean S and log-variance
    log_var_S (shape (B, q, q)).  The prior p(S|I) is N(0, (1/lam_s) * I).

    KL = 0.5 * sum_ij [ lam_s * (S_ij^2 + exp(log_var_S_ij))
                         - log_var_S_ij - 1 - log(lam_s) ]

    The 0.5 * (-1 - log(1/lam_s)) term is a constant and is dropped.

    Parameters
    ----------
    S : torch.Tensor
        Posterior mean of the spectral loading matrix.  Shape (B, q, q).
    log_var_S : torch.Tensor
        Independent posterior log-variance from log_var_S_head.  Shape (B, q, q).
        Must NOT be derived from S (fix #52).
    eigvals_q : torch.Tensor
        Leading q eigenvalues of the index Laplacian.  Shape (q,).
        Not used in the current KL form but kept for API consistency.
    lam_s : float
        Precision of the spectral basis prior.  Default 0.01.

    Returns
    -------
    torch.Tensor
        Scalar KL value (mean over batch and all (i,j) pairs).
    """
    var_S = log_var_S.exp()
    kl = 0.5 * (lam_s * (S.pow(2) + var_S) - log_var_S).mean()
    return kl


# ---------------------------------------------------------------------------
# tau_mode_kl  (Term 4: KL_tau)
# ---------------------------------------------------------------------------

def tau_mode_kl(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eigvals_q: torch.Tensor,
    tau: float = 0.5,
    a_min: float = 0.1,
) -> torch.Tensor:
    """
    KL divergence for mode frequencies: KL( q(w) || p(w|tau, Lambda) ).

    Both q(w) and p(w) are Gamma distributions parameterised by shape (a)
    and rate (b).  The KL between two Gammas Gamma(a, b) and Gamma(a0, b0) is::

        KL = (a - a0) * psi(a) - log_gamma(a) + log_gamma(a0)
             + a0 * (log_b - log_b0) + a * (b0/b - 1)

    Prior: p(w) = Gamma(a0, b0) with a0 = tau * lambda_k, b0 = tau.
    Posterior: q(w) = Gamma(a, b) with a = exp(log_a).clamp(min=a_min),
                                        b = exp(log_b).

    The shape floor a_min clamps the forward value seen by lgamma/digamma
    while letting the gradient flow through log_a unchanged (issue #68).

    Parameters
    ----------
    log_a : torch.Tensor
        Log shape parameters of the Gamma posterior.  Shape (B, q).
    log_b : torch.Tensor
        Log rate parameters of the Gamma posterior.  Shape (B, q).
    eigvals_q : torch.Tensor
        Leading q eigenvalues of L(I).  Shape (q,).  Used as lambda_k.
    tau : float
        Diffusion time scale.  Prior is Gamma(tau*lambda_k, tau).  Default 0.5.
    a_min : float
        Floor applied to exp(log_a) before lgamma/digamma.  Gradient still
        flows through log_a; only the special-function arguments are clamped.
        Default 0.1.  Set to 0.0 to disable.

    Returns
    -------
    torch.Tensor
        Scalar KL value (mean over batch and modes).
    """
    a = log_a.exp().clamp(min=a_min)   # (B, q) -- floored for stability
    b = log_b.exp()                    # (B, q)

    # Prior parameters: Gamma(tau * lambda_k, tau)
    lam = eigvals_q.clamp(min=1e-6)    # (q,)  -- avoid division by zero
    a0 = tau * lam                     # (q,)
    b0 = torch.full_like(lam, tau)     # (q,)

    kl = (
        (a - a0) * torch.digamma(a)
        - torch.lgamma(a) + torch.lgamma(a0)
        + a0 * (log_b - b0.log())
        + a * (b0 / b - 1.0)
    )
    return kl.mean()


# ---------------------------------------------------------------------------
# mode_entropy_penalty  (Option D, issue #82)
# ---------------------------------------------------------------------------

def mode_entropy_penalty(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    nu_entropy: float,
) -> torch.Tensor:
    """
    Active-mode entropy ceiling penalty (Option D, issue #82).

    Computes the entropy of the softmax-normalised Gamma shape/rate contrast
    across modes and penalises HIGH entropy (i.e. uniform mode activation).
    This complements the existing active_mode_penalty floor
    (relu(q_min - N_active)) by adding an upper-bound pressure toward
    sparse, low-entropy mode distributions.

    The penalty is subtracted from the ELBO, i.e. added to the total loss::

        loss = recon + kl_z + kl_S + kl_tau + floor_penalty + entropy_penalty

    Mathematical form::

        pi_k = softmax_k(log_a - log_b)       normalised mode-weight proxy
        H    = -sum_k pi_k * log(pi_k + eps)  Shannon entropy over q modes
        loss = nu_entropy * mean_batch(H)

    When nu_entropy=0.0 the function returns zero and no gradient flows.
    The penalty is logged as 'entropy_S' in the training diagnostics.

    Design note (Rayleigh analogy)
    ------------------------------
    In Rayleigh's theory of sound, each vibrational mode k carries energy
    proportional to its eigenfrequency lambda_k.  The Gamma posterior ratio
    E[omega_k] = a_k / b_k acts as the estimated modal energy.  Uniform
    distribution across modes (high H) corresponds to equipartition -- a
    thermodynamic equilibrium that suppresses mode selection.  The entropy
    ceiling penalty breaks equipartition and drives the system toward a
    sparse, low-entropy mode spectrum, analogous to a physical oscillator
    forced to resonate in a dominant mode.

    Parameters
    ----------
    log_a : torch.Tensor
        Log shape parameters of the Gamma mode posterior.  Shape (B, q).
    log_b : torch.Tensor
        Log rate parameters of the Gamma mode posterior.  Shape (B, q).
    nu_entropy : float
        Penalty weight (config key nu_entropy, default 0.5).  Set to 0.0 to
        disable.  Larger values push the mode distribution toward sparser
        activation more aggressively.

    Returns
    -------
    torch.Tensor
        Scalar: nu_entropy * mean_batch(H).  Add to total loss (subtract from ELBO).
        Returns a zero tensor (no gradient) when nu_entropy == 0.0.
    """
    if nu_entropy == 0.0:
        return log_a.new_zeros(1).squeeze()
    pi = (log_a - log_b).softmax(dim=-1)          # (B, q)
    H  = -(pi * (pi + 1e-8).log()).sum(dim=-1)     # (B,)
    return nu_entropy * H.mean()


# ---------------------------------------------------------------------------
# active_mode_penalty  (floor, issue #68)
# ---------------------------------------------------------------------------

def active_mode_penalty(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    q_min: int = 4,
    nu: float = 1.0,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    Active-mode floor penalty (issue #68).

    Penalises the model when the mean number of active spectral modes falls
    below the required minimum q_min.  A mode is considered active when its
    expected value E[omega_k] = exp(log_a_k - log_b_k) exceeds delta.

    penalty = nu * relu(q_min - N_active)

    where N_active = mean_batch( sum_k [ E[omega_k] > delta ] ).

    The penalty is zero when N_active >= q_min.  Set nu=0 or q_min=0 to
    disable entirely.  This acts as a floor constraint; the complementary
    ceiling constraint is mode_entropy_penalty (Option D, issue #82).

    Parameters
    ----------
    log_a : torch.Tensor
        Log shape parameters of the Gamma posterior.  Shape (B, q).
    log_b : torch.Tensor
        Log rate parameters of the Gamma posterior.  Shape (B, q).
    q_min : int
        Minimum required number of active modes.  Default 4.
    nu : float
        Penalty weight.  Default 1.0.  Set to 0.0 to disable.
    delta : float
        Activation threshold for E[omega_k].  Default 0.1.

    Returns
    -------
    torch.Tensor
        Scalar penalty value.  Non-negative.
    """
    if nu == 0.0 or q_min == 0:
        return log_a.new_zeros(1).squeeze()
    with torch.no_grad():
        omega = (log_a - log_b).exp()              # E[omega_k] = a/b,  (B, q)
        n_active = (omega > delta).float().sum(dim=-1).mean()  # scalar
    return nu * torch.relu(torch.tensor(q_min, dtype=log_a.dtype) - n_active)


# ---------------------------------------------------------------------------
# count_active_modes  (diagnostic, issue #77)
# ---------------------------------------------------------------------------

def count_active_modes(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    delta: float = 0.1,
) -> int:
    """
    Return the mean number of active spectral modes across the batch.

    A mode is active when its expected value E[omega_k] = exp(log_a_k -
    log_b_k) exceeds the threshold delta.  The result is a plain Python
    int with no gradient graph involvement (computed under no_grad).

    This diagnostic is stored as out['N_active'] in WiringAutoencoder.forward()
    and consumed by spectral_kl_health_check in train.py.

    Parameters
    ----------
    log_a : torch.Tensor
        Log shape parameters.  Shape (B, q).
    log_b : torch.Tensor
        Log rate parameters.  Shape (B, q).
    delta : float
        Activation threshold.  Default 0.1.

    Returns
    -------
    int
        Mean active-mode count across the batch.
    """
    with torch.no_grad():
        omega = (log_a - log_b).exp()          # (B, q)
        return int((omega > delta).float().sum(dim=-1).mean().item())


# ---------------------------------------------------------------------------
# spectral_kl_health_check
# ---------------------------------------------------------------------------

def spectral_kl_health_check(
    kl_z: float,
    kl_S: float,
    kl_tau: float = 0.0,
    active_modes: int = -1,
    q: int = 1,
    epoch: int = 0,
    warmup_epochs: int = 5,
    kl_S_min: float = 0.05,
) -> dict:
    """
    Health check for the spectral KL terms after each training epoch.

    Returns a dict with boolean flags that can be printed or forwarded to a
    logging system::

        {
            'mode_collapse':   bool,
            'mode_explosion':  bool,
            'kl_S_ok':         bool,
        }

    The 'mode_explosion' flag is suppressed during early training
    (epoch < warmup_epochs) because all modes are typically active at
    initialisation and should not trigger a warning before the KL pressure
    has had time to prune them.

    Parameters
    ----------
    kl_z : float
        Isotropic latent KL scalar from the epoch.
    kl_S : float
        Spectral basis KL scalar from the epoch.
    kl_tau : float
        Tau-mode frequency KL scalar (unused in logic, present for logging).
    active_modes : int
        Mean active-mode count from count_active_modes / out['N_active'].
        Pass -1 to skip mode collapse / explosion checks.
    q : int
        Total number of spectral modes in the model.
    epoch : int
        Current 1-based epoch index.  mode_explosion is suppressed before
        warmup_epochs.
    warmup_epochs : int
        Number of warmup epochs before mode_explosion is flagged.
    kl_S_min : float
        Minimum acceptable kl_S value.  Default 0.05.

    Returns
    -------
    dict
        Health-check result dict with boolean values.
    """
    in_warmup = epoch < warmup_epochs
    return {
        "mode_collapse":  (active_modes == 0) if active_modes >= 0 else False,
        "mode_explosion": (active_modes >= q) and not in_warmup if active_modes >= 0 else False,
        "kl_S_ok":        kl_S > kl_S_min,
    }


# ---------------------------------------------------------------------------
# TauModeDiffusion  -- learnable heat-kernel decoder
# ---------------------------------------------------------------------------

class TauModeDiffusion(nn.Module):
    """
    Learnable heat-kernel diffusion over the leading tau_modes eigenvectors.

    Implements the truncated modal expansion from Rayleigh's Theory of Sound.
    Given the leading k = tau_modes eigenpairs (U_k, lambda_k) of the base
    graph Laplacian L(I), the diffusion kernel is::

        K_tau = U_k diag(exp(-t * lambda_k)) U_k^T

    where the scalar diffusion time t = exp(log_t) is a learnable parameter
    initialised to log(dt_init).  Applying K_tau to a node signal E
    smooths it along the graph's lowest-frequency eigenvectors, reconstructing
    the input x from the embedding table.

    Only the tau_modes lowest-frequency modes participate in the kernel;
    higher-frequency modes are implicitly set to zero weight.  When
    tau_modes == q, every retained mode contributes.

    Parameters
    ----------
    tau_modes : int
        Number of leading eigenvectors kept in the heat kernel.  Default 4.
    dt_init : float
        Initial diffusion time.  Stored as log_t = log(dt_init).  Default 0.01.
    """

    def __init__(self, tau_modes: int = 4, dt_init: float = 0.01) -> None:
        super().__init__()
        self.tau_modes = tau_modes
        self.log_t = nn.Parameter(torch.tensor(math.log(dt_init)))

    def forward(
        self,
        E: torch.Tensor,
        eigvals: torch.Tensor,
        eigvecs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the heat-kernel diffusion to the embedding table E.

        Parameters
        ----------
        E : torch.Tensor
            Node embedding table.  Shape (N, D).
        eigvals : torch.Tensor
            Eigenvalues of the base Laplacian.  Shape (N,) or (q,).
            Only the first tau_modes values are used.
        eigvecs : torch.Tensor
            Eigenvectors of the base Laplacian.  Shape (N, K) with K >= tau_modes.

        Returns
        -------
        torch.Tensor
            Smoothed embedding matrix.  Shape (N, D).
        """
        k   = self.tau_modes
        t   = self.log_t.exp()                            # scalar > 0
        lam = eigvals[:k]                                 # (k,)
        U_k = eigvecs[:, :k]                              # (N, k)
        diag = torch.exp(-t * lam)                        # (k,)
        # K_tau = U_k diag(exp(-t*lam)) U_k^T;  apply to E
        return U_k @ (diag.unsqueeze(-1) * (U_k.T @ E))  # (N, D)


# ---------------------------------------------------------------------------
# spectral_freq_cost
# ---------------------------------------------------------------------------

def spectral_freq_cost(
    L_z: torch.Tensor,
    tau_modes: int = 4,
) -> torch.Tensor:
    """
    Frequency regularisation scalar for the decoded Laplacian L(z).

    Computes the mean of the leading tau_modes eigenvalues of L(z) as a
    soft measure of the spectral frequency of the decoded graph.  Used as
    an auxiliary regularisation term to penalise high-frequency wirings.

    For a batched Laplacian L_z of shape (B, N, N), the cost is::

        cost = mean_batch( mean_k( lambda_k(L_z) ) )   k = 1 ... tau_modes

    The eigendecomposition is offloaded to CPU for MPS compatibility.

    Parameters
    ----------
    L_z : torch.Tensor
        Batched decoded Laplacian.  Shape (B, N, N).
    tau_modes : int
        Number of leading eigenvalues to average.  Default 4.

    Returns
    -------
    torch.Tensor
        Scalar cost value.
    """
    dev = L_z.device
    # Offload to CPU: linalg.eigh on MPS has accuracy issues for large N.
    eigvals = torch.linalg.eigvalsh(L_z.cpu())  # (B, N)
    eigvals = eigvals.to(dev)
    return eigvals[:, :tau_modes].mean()


# ---------------------------------------------------------------------------
# lambda_fingerprint_hard  (non-differentiable)
# ---------------------------------------------------------------------------

def lambda_fingerprint_hard(
    L: torch.Tensor,
    tau_modes: int = 16,
    n_bins: int = 16,
    lam_min: float = 0.0,
    lam_max: float = 2.0,
    eigvals: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Non-differentiable Laplacian spectrum histogram (hard bins).

    Computes a fixed-bin histogram of the leading tau_modes eigenvalues
    of L as a compact spectral fingerprint.  The histogram is normalised
    to a probability vector (sums to 1).

    This is the original 'lambda_fingerprint' implementation.  It is NOT
    differentiable with respect to L or the eigenvalues because it uses
    torch.histc, which has no gradient support.  Use lambda_fingerprint_soft
    when gradients through the fingerprint are required (issue #56).

    The eigensolver is offloaded to CPU for MPS compatibility.

    Parameters
    ----------
    L : torch.Tensor
        Laplacian matrix or batch.  Shape (N, N) or (B, N, N).
        Ignored when eigvals is provided.
    tau_modes : int
        Number of leading eigenvalues to include.  Default 16.
    n_bins : int
        Number of histogram bins.  Default 16.
    lam_min : float
        Left edge of the histogram range.  Default 0.0.
    lam_max : float
        Right edge of the histogram range.  Default 2.0 (normalised Laplacian).
    eigvals : torch.Tensor or None
        Pre-computed eigenvalues of shape (tau_modes,) or (B, tau_modes).
        When provided, the eigensolver is skipped.  Default None.

    Returns
    -------
    torch.Tensor
        Normalised histogram.  Shape (1, n_bins) for unbatched input or
        (B, n_bins) for batched input.  Each row sums to 1.
    """
    dev = L.device
    if eigvals is None:
        L_cpu = L.cpu()
        if L_cpu.ndim == 2:
            L_cpu = L_cpu.unsqueeze(0)
        ev = torch.linalg.eigvalsh(L_cpu)          # (B, N)
        ev = ev[:, :tau_modes]                     # (B, k)
    else:
        ev = eigvals.cpu()
        if ev.ndim == 1:
            ev = ev.unsqueeze(0)

    B = ev.shape[0]
    hists = []
    for b in range(B):
        h = torch.histc(
            ev[b].float(), bins=n_bins, min=lam_min, max=lam_max
        )  # (n_bins,)
        total = h.sum().clamp(min=1.0)
        hists.append(h / total)
    return torch.stack(hists, dim=0).to(dev)       # (B, n_bins)


# ---------------------------------------------------------------------------
# lambda_fingerprint_soft  (differentiable KDE, issue #56)
# ---------------------------------------------------------------------------

def lambda_fingerprint_soft(
    eigvals: torch.Tensor,
    n_bins: int = 16,
    lam_min: float = 0.0,
    lam_max: float = 2.0,
    bandwidth: float = 0.1,
) -> torch.Tensor:
    """
    Differentiable Laplacian spectrum histogram using Gaussian KDE.

    Replaces the hard torch.histc histogram with a soft assignment that
    supports gradient flow through the eigenvalues (issue #56).  Each
    eigenvalue contributes to each bin proportionally to a Gaussian kernel
    centred at the eigenvalue's position::

        bin_centres_j = lam_min + (j + 0.5) * bin_width,  j = 0 ... n_bins-1
        phi_kj        = exp( -0.5 * ((lambda_k - centre_j) / bandwidth)^2 )
        row_j         = sum_k phi_kj
        fingerprint   = row / row.sum()    (normalised to probability vector)

    The bandwidth parameter controls the kernel width in eigenvalue units.
    Smaller bandwidth gives sharper bins (closer to hard histogram); larger
    bandwidth gives smoother interpolation.

    This function is differentiable with respect to eigvals: gradients from
    downstream losses can flow back through the soft histogram into the
    eigensolver (or an amortised approximation thereof).

    Parameters
    ----------
    eigvals : torch.Tensor
        Eigenvalues.  Shape (q,) for unbatched or (B, q) for batched.
        The function auto-detects the shape.
    n_bins : int
        Number of histogram bins.  Default 16.
    lam_min : float
        Left edge of the spectral range.  Default 0.0.
    lam_max : float
        Right edge of the spectral range.  Default 2.0.
    bandwidth : float
        Gaussian kernel bandwidth in eigenvalue units.  Default 0.1.

    Returns
    -------
    torch.Tensor
        Normalised soft histogram.  Shape (n_bins,) for unbatched input or
        (B, n_bins) for batched input.  Each row sums to 1.
    """
    unbatched = eigvals.ndim == 1
    if unbatched:
        eigvals = eigvals.unsqueeze(0)   # (1, q)

    bin_width = (lam_max - lam_min) / n_bins
    centres = torch.linspace(
        lam_min + 0.5 * bin_width,
        lam_max - 0.5 * bin_width,
        n_bins,
        device=eigvals.device,
        dtype=eigvals.dtype,
    )  # (n_bins,)

    # Gaussian KDE: (B, q, 1) vs (1, 1, n_bins) -> (B, q, n_bins)
    diff = eigvals.unsqueeze(-1) - centres.unsqueeze(0).unsqueeze(0)
    phi  = torch.exp(-0.5 * (diff / bandwidth) ** 2)  # (B, q, n_bins)
    hist = phi.sum(dim=1)                              # (B, n_bins)
    fp   = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # normalise

    return fp.squeeze(0) if unbatched else fp


# ---------------------------------------------------------------------------
# lambda_fingerprint  -- backwards-compat alias for lambda_fingerprint_hard
# ---------------------------------------------------------------------------

lambda_fingerprint = lambda_fingerprint_hard
"""
Backwards-compatibility alias.  All new code should use
lambda_fingerprint_hard or lambda_fingerprint_soft explicitly.
"""


# ---------------------------------------------------------------------------
# build_laplacian  (convenience wrapper)
# ---------------------------------------------------------------------------

def build_laplacian(
    x: torch.Tensor,
    k: int = 15,
    sigma: float = 1.0,
    normalised: bool = True,
) -> "DifferentiableLaplacian":
    """
    Convenience wrapper: build a DifferentiableLaplacian from a feature
    matrix x using k-nearest-neighbour graph construction.

    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix.  Shape (N, D).
    k : int
        Number of nearest neighbours.  Default 15.
    sigma : float
        RBF bandwidth.  Default 1.0.
    normalised : bool
        Whether to return the normalised graph Laplacian.  Default True.

    Returns
    -------
    DifferentiableLaplacian
        Laplacian module, not yet on any device.
    """
    from .laplacian import DifferentiableLaplacian
    return DifferentiableLaplacian.from_embeddings(
        x, knn_k=k, sigma=sigma, normalised=normalised
    )
