"""
Spectral KL helpers for the Wiring Autoencoder.

This module contains all penalty and KL-divergence functions that
operate on the Gamma-distributed spectral mode posterior (log_a, log_b)
and on the spectral loading matrix S.

Public API
----------
build_laplacian          -- convenience wrapper around DifferentiableLaplacian
spectral_basis_kl        -- KL( q(S) || p(S|I) ), Term 3 of the ELBO
tau_mode_kl              -- KL( q(w) || p(w|tau,Lambda) ), Term 4
mode_entropy_penalty     -- Option D active-mode entropy ceiling penalty (#82)
active_mode_penalty      -- active-mode floor penalty (#68)
count_active_modes       -- diagnostic: mean E[omega_k] > delta count
spectral_kl_health_check -- health check dict: mode_collapse / mode_explosion / kl_S_ok
"""
from __future__ import annotations

import torch


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
    and rate (b).  The KL between two Gammas Gamma(a, b) and Gamma(a0, b0)
    is::

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

    where::

        ELBO = recon - kl_z - kl_S - kl_tau - nu_entropy * H

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
    N_active: int,
    q: int,
    epoch: int,
    kl_S_min: float = 0.05,
) -> dict:
    """
    Health check for the spectral KL terms after each training epoch.

    Returns a dict with three boolean flags that can be printed or
    forwarded to a logging system::

        {
            'mode_collapse':   bool,  -- all modes collapsed to zero
            'mode_explosion':  bool,  -- more active modes than q (impossible by design,
                                         but guards against float rounding)
            'kl_S_ok':         bool,  -- kl_S > kl_S_min (posterior non-trivial)
        }

    Parameters
    ----------
    kl_z : float
        Isotropic latent KL scalar from the epoch.
    kl_S : float
        Spectral basis KL scalar from the epoch.
    N_active : int
        Mean active-mode count from count_active_modes / out['N_active'].
    q : int
        Total number of spectral modes in the model.
    epoch : int
        Current epoch index (0-based).  Used for logging context.
    kl_S_min : float
        Minimum acceptable kl_S value.  Default 0.05.

    Returns
    -------
    dict
        Health-check result dict.
    """
    return {
        "mode_collapse":  N_active == 0,
        "mode_explosion": N_active > q,
        "kl_S_ok":        kl_S > kl_S_min,
    }


# ---------------------------------------------------------------------------
# build_laplacian  (convenience wrapper)
# ---------------------------------------------------------------------------

def build_laplacian(x: torch.Tensor, k: int = 15, sigma: float = 1.0,
                    normalised: bool = True) -> "DifferentiableLaplacian":
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
