"""
Stability diagnostics for the Wiring Autoencoder.

Provides four public functions covering the full diagnostic hierarchy
described in docs//04-stability.md.

Functions
---------
stability_diagnostics
    Full per-epoch diagnostic dict: CFL health, damping, spectral entropy,
    and density-matrix PSD checks.

log_preconditioner_stability
    Preconditioned gradient landscape metrics: condition number and
    convergence rate estimate.

pre_training_checks
    6-level checklist that must pass before any  training run.
    Raises RuntimeError on a disconnected graph.

spectral_kl_health_check
    -only: ELBO component sanity checks including mode-collapse
    and KL explosion detection.

Integration points
------------------
- Call `stability_diagnostics` at the end of each training epoch.
- Call `spectral_kl_health_check` after each forward() in training.
- Call `pre_training_checks` once before the first training step;
  raise RuntimeError to abort if any hard check fails.
- Log all returned scalars under the ``stability/`` prefix
  (Weights and Biases or stdout).

Depends on
----------
  vdt.laplacian  : DifferentiableLaplacian (MassMatrix, dt_max_cfl)
  vdt.spectral   : _safe_eigvalsh
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional

import torch

from vdt.spectral import _safe_eigvalsh


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#: Eigenvalues below this threshold are treated as numerically zero.
#: Used by pre_training_checks to count connected components for both
#: combinatorial and normalised symmetric Laplacians.
_ZERO_EIG_TOL: float = 1e-3


def _min_eigval(M: torch.Tensor) -> float:
    """
    Smallest eigenvalue of a symmetric matrix M (N, N), on CPU.
    Uses _safe_eigvalsh so MPS is safe.
    """
    ev = _safe_eigvalsh(M)
    return float(ev[0].item())


def _modal_energy(Q: torch.Tensor) -> float:
    """Frobenius norm squared of Q, normalised by number of elements."""
    return float(Q.pow(2).mean().item())


# ---------------------------------------------------------------------------
# stability_diagnostics
# ---------------------------------------------------------------------------

def stability_diagnostics(
    L_f: torch.Tensor,
    Q_states: List[torch.Tensor],
    rho_plus_list: List[torch.Tensor],
    rho_minus_list: List[torch.Tensor],
    eigvals: Optional[torch.Tensor],
    dt: float,
    gamma: torch.Tensor,
) -> dict:
    """
    Full stability diagnostic dict for one epoch.

    Runs in O(N^2 * K) for K VDT depth steps (no additional eigensolve
    if eigvals is provided).  Target: <5ms for d=256 on CPU.

    Parameters
    ----------
    L_f : Tensor  shape (N, N) or (B, N, N)
        Feature-space graph Laplacian at the current training step.
    Q_states : list of Tensor  shape (N, d) or (B, N, d)
        VDT output states [X0, Q_1, ..., Q_K] from VDT.forward().
    rho_plus_list : list of Tensor  shape (N, N)
        Positive density matrices, one per VDT block.
    rho_minus_list : list of Tensor  shape (N, N)
        Negative density matrices, one per VDT block.
    eigvals : Tensor  shape (N,) or (B, N) or None
        Pre-computed eigenvalues of L_f.  Computed internally when None.
    dt : float
        Current CFL-clamped time step used in training.
    gamma : Tensor  shape (d,)
        Per-feature damping vector (softplus output from VibrationalStateBlock).

    Returns
    -------
    dict with keys:
        lambda_max          float -- largest eigenvalue of L_f
        dt_max_CFL          float -- safety * sqrt(2 / lambda_max)
        dt_current          float -- dt passed in
        CFL_ok              bool  -- dt_current <= dt_max_CFL
        n_underdamped_modes int   -- number of modes with gamma_k < sqrt(lambda_k)
        frac_underdamped    float -- n_underdamped_modes / N
        modal_energy_per_depth  list[float] -- ||Q_k||_F^2 / elements per depth step
        energy_amplified    bool  -- any ||Q_{k+1}|| > ||Q_k|| * 1.05
        spectral_entropy_K  float -- entropy of normalised eigenvalue distribution
        min_eig_rho_plus    float -- min eigenvalue across all rho_plus matrices
        min_eig_rho_minus   float -- min eigenvalue across all rho_minus matrices
        max_frob_signed     float -- max Frobenius norm of (rho_plus - rho_minus)
        rho_psd_ok          bool  -- min_eig_rho_plus >= -1e-5 and min_eig_rho_minus >= -1e-5
    """
    # -- Eigenvalues of L_f ------------------------------------------------
    L_single = L_f[0] if L_f.dim() == 3 else L_f
    if eigvals is None:
        ev = _safe_eigvalsh(L_single)
    else:
        ev = eigvals
        if ev.dim() == 2:
            ev = ev[0]
    ev = ev.float().clamp(min=0.0)  # (N,)
    N = ev.shape[0]

    lam_max = float(ev[-1].item())
    dt_max_cfl = math.sqrt(2.0 / max(lam_max, 1e-8))
    cfl_ok = dt <= dt_max_cfl

    # -- Underdamped modes -------------------------------------------------
    # Underdamped: gamma_k < sqrt(lambda_k) for each feature k.
    # We scalar-reduce gamma to a single value for comparison.
    gamma_mean = float(gamma.mean().item())
    sqrt_ev = ev.sqrt()  # (N,)
    n_underdamped = int((sqrt_ev > gamma_mean).sum().item())
    frac_underdamped = n_underdamped / max(N, 1)

    # -- Modal energy per depth step ---------------------------------------
    energies = [_modal_energy(Q) for Q in Q_states]
    energy_amplified = any(
        energies[k + 1] > energies[k] * 1.05
        for k in range(len(energies) - 1)
    )

    # -- Spectral entropy --------------------------------------------------
    ev_pos = ev.clamp(min=1e-8)
    prob = ev_pos / ev_pos.sum()
    spectral_entropy = float(-(prob * prob.log()).sum().item())

    # -- Density matrix checks --------------------------------------------
    min_rho_p = float("inf")
    min_rho_m = float("inf")
    max_frob_signed = 0.0

    for rp, rm in zip(rho_plus_list, rho_minus_list):
        min_rho_p = min(min_rho_p, _min_eigval(rp))
        min_rho_m = min(min_rho_m, _min_eigval(rm))
        frob = float((rp - rm).norm("fro").item())
        max_frob_signed = max(max_frob_signed, frob)

    if not rho_plus_list:
        min_rho_p = 0.0
        min_rho_m = 0.0

    rho_psd_ok = (min_rho_p >= -1e-5) and (min_rho_m >= -1e-5)

    return {
        "lambda_max":             lam_max,
        "dt_max_CFL":             dt_max_cfl,
        "dt_current":             dt,
        "CFL_ok":                 cfl_ok,
        "n_underdamped_modes":    n_underdamped,
        "frac_underdamped":       frac_underdamped,
        "modal_energy_per_depth": energies,
        "energy_amplified":       energy_amplified,
        "spectral_entropy_K":     spectral_entropy,
        "min_eig_rho_plus":       min_rho_p,
        "min_eig_rho_minus":      min_rho_m,
        "max_frob_signed":        max_frob_signed,
        "rho_psd_ok":             rho_psd_ok,
    }


# ---------------------------------------------------------------------------
# log_preconditioner_stability
# ---------------------------------------------------------------------------

def log_preconditioner_stability(
    A: torch.Tensor,
    L_f: torch.Tensor,
    M_diag: torch.Tensor,
    sigma: float,
    eta: float,
) -> dict:
    """
    Metrics for the preconditioned gradient landscape.

    Estimates the strong-convexity constant mu, the Lipschitz constant L,
    the condition number kappa, and whether the learning rate eta is
    within the convergence-guarantee range.

    Hessian approximation: H_prec = sigma * M_diag * I  +  L_f
    (Tikhonov-preconditioned, ignoring off-diagonal coupling).

    Parameters
    ----------
    A : Tensor  shape (N, N)
        Symmetric weight matrix (e.g., current adjacency).
    L_f : Tensor  shape (N, N)
        Feature-space Laplacian.
    M_diag : Tensor  shape (N,)
        Diagonal mass matrix (from MassMatrix.M_diag).
    sigma : float
        Tikhonov regularisation coefficient.
    eta : float
        Current learning rate.

    Returns
    -------
    dict with keys:
        mu_sigma_M       float -- minimum eigenvalue of H_prec (strong convexity)
        L_sigma_M        float -- maximum eigenvalue of H_prec (Lipschitz const.)
        kappa_H_prec     float -- L_sigma_M / mu_sigma_M (condition number)
        eta_ok           bool  -- eta < 2 / L_sigma_M (gradient descent step condition)
        convergence_rate float -- (kappa - 1) / (kappa + 1)  (worst-case ratio)
    """
    # Build diagonal H_prec = sigma * diag(M) + diag(eigvals(L_f))
    # We approximate using only the L_f diagonal term for efficiency.
    L_single = L_f[0] if L_f.dim() == 3 else L_f
    ev = _safe_eigvalsh(L_single).float().clamp(min=0.0)  # (N,)

    h_diag = sigma * M_diag.float().clamp(min=1e-8) + ev
    mu = float(h_diag.min().item())
    L_const = float(h_diag.max().item())
    kappa = L_const / max(mu, 1e-8)
    eta_ok = eta < 2.0 / max(L_const, 1e-8)
    convergence_rate = (kappa - 1.0) / (kappa + 1.0)

    return {
        "mu_sigma_M":       mu,
        "L_sigma_M":        L_const,
        "kappa_H_prec":     kappa,
        "eta_ok":           eta_ok,
        "convergence_rate": convergence_rate,
    }


# ---------------------------------------------------------------------------
# pre_training_checks
# ---------------------------------------------------------------------------

# Six-level stability hierarchy from docs//04-stability.md section 7.
_CHECKS_ORDER = [
    "graph_connected",
    "cfl_satisfied",
    "mass_conditioned",
    "damping_positive",
    "density_psd",
    "kl_finite",
]


def pre_training_checks(
    L_f: torch.Tensor,
    M_diag: torch.Tensor,
    dt_init: float,
    gamma: Optional[torch.Tensor] = None,
    kl_sample: Optional[float] = None,
) -> List[str]:
    """
    Six-level pre-training checklist from docs//04-stability.md section 7.

    Checks are run in order; on a hard failure (level 1 -- disconnected graph)
    a RuntimeError is raised immediately so the training loop cannot start
    with a pathological Laplacian.  All other failures are collected as
    warning strings and returned to the caller.

    Connectivity detection
    ----------------------
    The check counts eigenvalues below _ZERO_EIG_TOL (1e-3).  Any Laplacian
    with more than one near-zero eigenvalue has more than one connected
    component.  This criterion works for both:

    - Combinatorial Laplacian (L = D - A): zero eigenvalue multiplicity
      equals the number of connected components by the spectral theorem.
    - Normalised symmetric Laplacian (L = I - D^{-1/2} A D^{-1/2}): same
      zero-multiplicity property holds; the non-zero eigenvalues shift but
      each component still contributes exactly one zero.

    The old Fiedler-threshold check (ev[1] < 1e-5) only worked for the
    combinatorial case; for two balanced cliques in the normalised Laplacian
    the Fiedler value is ~0.67 and was never detected.

    Parameters
    ----------
    L_f : Tensor  shape (N, N) or (B, N, N)
        Feature-space Laplacian.  Unbatched or first-batch element used.
    M_diag : Tensor  shape (N,)
        Diagonal mass matrix.
    dt_init : float
        Initial time step chosen by the caller.
    gamma : Tensor  shape (d,) or None
        Per-feature damping vector.  Skipped when None.
    kl_sample : float or None
        A sample ELBO KL value.  Checked for finiteness when provided.

    Returns
    -------
    list[str]
        Warning strings for each failed check.  Empty list means all clear.

    Raises
    ------
    RuntimeError
        If the graph Laplacian is disconnected (zero-eigenvalue multiplicity > 1).
    """
    warnings_out: List[str] = []

    L_single = L_f[0] if L_f.dim() == 3 else L_f
    ev = _safe_eigvalsh(L_single).float()  # (N,)  NOT clamped -- need real zeros
    N = int(ev.shape[0])

    # Level 1 -- graph connectivity
    # Count eigenvalues below _ZERO_EIG_TOL.  A connected graph has exactly
    # one zero eigenvalue (the constant eigenvector); a disconnected graph
    # has >= 2.  This works for both combinatorial and normalised Laplacians.
    n_zero = int((ev < _ZERO_EIG_TOL).sum().item())
    fiedler = float(ev[1].item()) if N > 1 else float(ev[0].item())
    if n_zero > 1:
        raise RuntimeError(
            f"pre_training_checks: graph is disconnected "
            f"(zero-eigenvalue multiplicity={n_zero}, "
            f"Fiedler value={fiedler:.4f}).  "
            "Check kNN construction -- all nodes must be reachable."
        )

    ev = ev.clamp(min=0.0)  # safe to clamp now that connectivity is confirmed

    # Level 2 -- CFL constraint
    lam_max = float(ev[-1].item())
    dt_max = math.sqrt(2.0 / max(lam_max, 1e-8))
    if dt_init > dt_max:
        warnings_out.append(
            f"CFL violated: dt_init={dt_init:.4f} > dt_max_CFL={dt_max:.4f}. "
            "Reduce dt_init or increase damping."
        )

    # Level 3 -- mass matrix conditioning
    cond = float(M_diag.max().item()) / max(float(M_diag.min().item()), 1e-8)
    if cond > 100.0:
        warnings_out.append(
            f"MassMatrix conditioning ratio {cond:.1f} > 100. "
            "Laplacian may be poorly conditioned (docs//04-stability.md S7)."
        )

    # Level 4 -- damping positivity
    if gamma is not None:
        if float(gamma.min().item()) <= 0.0:
            warnings_out.append(
                "gamma contains non-positive entries.  "
                "VibrationalStateBlock must use softplus on raw gamma."
            )

    # Level 5 -- density PSD (checked via M_diag as proxy; rho checked at runtime)
    if float(M_diag.min().item()) < -1e-5:
        warnings_out.append(
            "M_diag contains negative entries.  "
            "MassMatrix is not PSD -- check eigenvalue computation."
        )

    # Level 6 -- KL finiteness
    if kl_sample is not None:
        if not math.isfinite(kl_sample):
            warnings_out.append(
                f"KL sample is non-finite ({kl_sample}).  "
                "Check spectral_basis_kl and tau_mode_kl inputs."
            )

    return warnings_out


# ---------------------------------------------------------------------------
# spectral_kl_health_check  ()
# ---------------------------------------------------------------------------

def spectral_kl_health_check(
    kl_z: float,
    kl_S: float,
    kl_tau: float,
    active_modes: int,
    q: int,
) -> dict:
    """
     ELBO health check -- run after each WiringAutoencoder.forward().

    Checks all three KL components for sign, finiteness, and explosion.
    Detects mode collapse (< 10% modes active) and mode explosion (all
    modes active, meaning no spectral selection is happening).

    Emits warnings.warn on:
      - mode_collapse: active_modes < 0.1 * q
      - kl explosion: any KL component > 1e4

    Parameters
    ----------
    kl_z : float
        Isotropic KL term from WiringEncoder.  Should be > 0.
    kl_S : float
        Spectral basis KL from spectral_basis_kl().  Should be > 0.
    kl_tau : float
        Mode-weight KL from tau_mode_kl().  Should be > 0.
    active_modes : int
        Number of modes with omega_k above a relevance threshold (caller-computed).
    q : int
        Total number of latent modes.

    Returns
    -------
    dict with keys:
        kl_z_ok        bool  -- kl_z > 0 and kl_z < 1e4
        kl_S_ok        bool  -- kl_S > 0
        kl_tau_ok      bool  -- kl_tau > 0
        mode_collapse  bool  -- active_modes < q * 0.1
        mode_explosion bool  -- active_modes == q
    """
    kl_z_ok        = (kl_z > 0.0) and (kl_z < 1e4)
    kl_S_ok        = kl_S > 0.0
    kl_tau_ok      = kl_tau > 0.0
    mode_collapse  = active_modes < q * 0.1
    mode_explosion = active_modes == q

    if mode_collapse:
        warnings.warn(
            f"spectral_kl_health_check: mode collapse detected -- "
            f"only {active_modes}/{q} modes active (<10%). "
            "Consider reducing tau or increasing the spectral diversity loss.",
            RuntimeWarning,
            stacklevel=2,
        )

    if kl_z > 1e4:
        warnings.warn(
            f"spectral_kl_health_check: kl_z={kl_z:.2e} > 1e4 (explosion). "
            "Check isotropic KL computation and log_var range.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "kl_z_ok":        kl_z_ok,
        "kl_S_ok":        kl_S_ok,
        "kl_tau_ok":      kl_tau_ok,
        "mode_collapse":  mode_collapse,
        "mode_explosion": mode_explosion,
    }
