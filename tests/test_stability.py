"""
tests/test_stability.py  --  acceptance tests for vdt/stability.py (#19).

AC1  stability_diagnostics: all expected keys present; CFL_ok=True on well-formed input
AC2  stability_diagnostics: PSD-violating rho_plus -> rho_psd_ok=False
AC3  log_preconditioner_stability: condition number and eta_ok boundary
AC4  pre_training_checks: connected ring graph -> no warnings
AC5  pre_training_checks: disconnected graph -> RuntimeError
AC6  pre_training_checks: CFL-violating dt -> warning string
AC7  spectral_kl_health_check: healthy ELBO values -> all flags correct
AC8  spectral_kl_health_check: mode_collapse when active_modes < 10% of q
AC9  spectral_kl_health_check: kl_z > 1e4 triggers RuntimeWarning
"""
from __future__ import annotations
import math
import warnings
import torch
import pytest
from vdt.stability import (
    stability_diagnostics,
    log_preconditioner_stability,
    pre_training_checks,
    spectral_kl_health_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ring_laplacian(N: int) -> torch.Tensor:
    """Normalised symmetric Laplacian of a connected ring graph."""
    A = torch.zeros(N, N)
    for i in range(N):
        j = (i + 1) % N
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(-1).clamp(min=1e-6)
    d_inv = deg.pow(-0.5)
    normed = d_inv.unsqueeze(-1) * A * d_inv.unsqueeze(-2)
    return torch.eye(N) - normed


def _disconnected_laplacian(N: int) -> torch.Tensor:
    """Two isolated cliques: nodes 0..N//2-1 and N//2..N-1."""
    half = N // 2
    A = torch.zeros(N, N)
    # clique 1
    for i in range(half):
        for j in range(i + 1, half):
            A[i, j] = A[j, i] = 1.0
    # clique 2
    for i in range(half, N):
        for j in range(i + 1, N):
            A[i, j] = A[j, i] = 1.0
    deg = A.sum(-1).clamp(min=1e-6)
    d_inv = deg.pow(-0.5)
    normed = d_inv.unsqueeze(-1) * A * d_inv.unsqueeze(-2)
    return torch.eye(N) - normed


def _psd_matrix(N: int, min_eig: float = 0.1) -> torch.Tensor:
    """Random PSD matrix with minimum eigenvalue >= min_eig."""
    Z = torch.randn(N, N)
    M = Z @ Z.t() / N + min_eig * torch.eye(N)
    return M


def _psd_violating_matrix(N: int) -> torch.Tensor:
    """Non-PSD matrix with a negative eigenvalue."""
    M = _psd_matrix(N)
    M[0, 0] -= 10.0  # force a negative eigenvalue
    return M


# ---------------------------------------------------------------------------
# AC1  --  stability_diagnostics: expected keys + CFL_ok
# ---------------------------------------------------------------------------

class TestStabilityDiagnosticsKeys:
    """
    AC1: stability_diagnostics returns all expected keys and CFL_ok=True
    when dt is well within the CFL bound.

    EXPECTED_KEYS includes the four dynamic-Lf CFL keys added in issue #53:
      lambda_max_Lf, dt_max_CFL_Lf, cfl_margin, cfl_Lf_ok.
    """

    EXPECTED_KEYS = {
        # Base-Laplacian CFL (backward-compatible)
        "lambda_max", "dt_max_CFL", "dt_current", "CFL_ok",
        # Dynamic L_f CFL (issue #53)
        "lambda_max_Lf", "dt_max_CFL_Lf", "cfl_margin", "cfl_Lf_ok",
        # Damping
        "n_underdamped_modes", "frac_underdamped",
        # Energy
        "modal_energy_per_depth", "energy_amplified",
        # Spectral entropy
        "spectral_entropy_K",
        # Density matrices
        "min_eig_rho_plus", "min_eig_rho_minus", "max_frob_signed", "rho_psd_ok",
    }

    def test_all_keys_present(self) -> None:
        N, d, K = 12, 8, 3
        L_f = _ring_laplacian(N)
        Q_states = [torch.randn(N, d) for _ in range(K + 1)]
        rho_p = [_psd_matrix(N) for _ in range(K)]
        rho_m = [_psd_matrix(N) for _ in range(K)]
        gamma = torch.ones(d) * 0.5
        diag = stability_diagnostics(
            L_f, Q_states, rho_p, rho_m,
            eigvals=None, dt=0.05, gamma=gamma,
        )
        assert self.EXPECTED_KEYS == set(diag.keys())

    def test_cfl_ok_true(self) -> None:
        N, d = 10, 4
        L_f = _ring_laplacian(N)
        gamma = torch.ones(d) * 0.5
        lam_max = float(torch.linalg.eigvalsh(L_f)[-1].item())
        dt_safe = 0.5 * math.sqrt(2.0 / max(lam_max, 1e-8))
        diag = stability_diagnostics(
            L_f, [torch.randn(N, d)], [], [],
            eigvals=None, dt=dt_safe, gamma=gamma,
        )
        assert diag["CFL_ok"] is True

    def test_cfl_ok_false(self) -> None:
        N, d = 10, 4
        L_f = _ring_laplacian(N)
        gamma = torch.ones(d) * 0.5
        diag = stability_diagnostics(
            L_f, [torch.randn(N, d)], [], [],
            eigvals=None, dt=999.0, gamma=gamma,
        )
        assert diag["CFL_ok"] is False


# ---------------------------------------------------------------------------
# AC2  --  rho_psd_ok=False when rho_plus is not PSD
# ---------------------------------------------------------------------------

class TestRhoPSDViolation:
    """
    AC2: stability_diagnostics must report rho_psd_ok=False when rho_plus
    has a negative eigenvalue.
    """

    def test_psd_violation_detected(self) -> None:
        N, d = 8, 4
        L_f = _ring_laplacian(N)
        gamma = torch.ones(d) * 0.3
        rho_bad = [_psd_violating_matrix(N)]
        rho_good = [_psd_matrix(N)]
        diag = stability_diagnostics(
            L_f, [torch.randn(N, d), torch.randn(N, d)],
            rho_bad, rho_good,
            eigvals=None, dt=0.05, gamma=gamma,
        )
        assert diag["rho_psd_ok"] is False
        assert diag["min_eig_rho_plus"] < -1e-5


# ---------------------------------------------------------------------------
# AC3  --  log_preconditioner_stability
# ---------------------------------------------------------------------------

class TestLogPreconditionerStability:
    """
    AC3: kappa > 0, convergence_rate in [0, 1), eta_ok correct boundary.
    """

    def test_kappa_positive(self) -> None:
        N = 8
        L_f = _ring_laplacian(N)
        M_diag = torch.ones(N) * 2.0
        A = _psd_matrix(N)
        result = log_preconditioner_stability(A, L_f, M_diag, sigma=1.0, eta=1e-3)
        assert result["kappa_H_prec"] > 0
        assert 0.0 <= result["convergence_rate"] < 1.0

    def test_eta_ok_boundary(self) -> None:
        N = 6
        L_f = _ring_laplacian(N)
        M_diag = torch.ones(N)
        A = _psd_matrix(N)
        result = log_preconditioner_stability(A, L_f, M_diag, sigma=0.1, eta=1e-10)
        assert result["eta_ok"] is True

        result_bad = log_preconditioner_stability(A, L_f, M_diag, sigma=0.1, eta=1e6)
        assert result_bad["eta_ok"] is False


# ---------------------------------------------------------------------------
# AC4  --  pre_training_checks: connected graph -> no warnings
# ---------------------------------------------------------------------------

class TestPreTrainingChecksConnected:
    """
    AC4: A well-formed connected graph with valid dt returns an empty warning list.
    """

    def test_no_warnings_on_valid_input(self) -> None:
        N = 12
        L_f = _ring_laplacian(N)
        M_diag = torch.ones(N) * 1.5
        lam_max = float(torch.linalg.eigvalsh(L_f)[-1].item())
        dt_safe = 0.5 * math.sqrt(2.0 / max(lam_max, 1e-8))
        gamma = torch.ones(4) * 0.5
        warnings_list = pre_training_checks(L_f, M_diag, dt_safe, gamma=gamma)
        assert warnings_list == [], f"Unexpected warnings: {warnings_list}"


# ---------------------------------------------------------------------------
# AC5  --  pre_training_checks: disconnected graph -> RuntimeError
# ---------------------------------------------------------------------------

class TestPreTrainingChecksDisconnected:
    """
    AC5: A disconnected Laplacian (Fiedler value ~ 0) must raise RuntimeError.
    """

    def test_disconnected_raises(self) -> None:
        N = 8
        L_f = _disconnected_laplacian(N)
        M_diag = torch.ones(N)
        with pytest.raises(RuntimeError, match="disconnected"):
            pre_training_checks(L_f, M_diag, dt_init=0.01)


# ---------------------------------------------------------------------------
# AC6  --  pre_training_checks: CFL-violating dt -> warning string
# ---------------------------------------------------------------------------

class TestPreTrainingChecksCFL:
    """
    AC6: dt_init > dt_max_CFL produces a non-empty warning list with
    a message mentioning 'CFL'.
    """

    def test_cfl_warning_returned(self) -> None:
        N = 10
        L_f = _ring_laplacian(N)
        M_diag = torch.ones(N)
        warnings_list = pre_training_checks(L_f, M_diag, dt_init=1e6)
        assert any("CFL" in w for w in warnings_list), (
            f"Expected CFL warning, got: {warnings_list}"
        )


# ---------------------------------------------------------------------------
# AC7  --  spectral_kl_health_check: healthy ELBO
# ---------------------------------------------------------------------------

class TestKLHealthCheckHealthy:
    """
    AC7: Healthy ELBO values (all KLs positive and bounded, moderate active
    modes) return all-clear flags.
    """

    def test_healthy_returns_all_clear(self) -> None:
        result = spectral_kl_health_check(
            kl_z=2.5, kl_S=1.1, kl_tau=0.8,
            active_modes=5, q=8,
        )
        assert result["kl_z_ok"] is True
        assert result["kl_S_ok"] is True
        assert result["kl_tau_ok"] is True
        assert result["mode_collapse"] is False
        assert result["mode_explosion"] is False


# ---------------------------------------------------------------------------
# AC8  --  spectral_kl_health_check: mode_collapse
# ---------------------------------------------------------------------------

class TestKLHealthCheckModeCollapse:
    """
    AC8: mode_collapse=True when active_modes < 10% of q.
    Also verifies that warnings.warn is emitted.
    """

    def test_mode_collapse_flagged(self) -> None:
        q = 20
        active = 1  # 5% < 10% threshold
        result = spectral_kl_health_check(
            kl_z=1.0, kl_S=1.0, kl_tau=1.0,
            active_modes=active, q=q,
        )
        assert result["mode_collapse"] is True

    def test_mode_collapse_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spectral_kl_health_check(
                kl_z=1.0, kl_S=1.0, kl_tau=1.0,
                active_modes=0, q=16,
            )
        assert any("mode collapse" in str(wi.message).lower() for wi in w), (
            "Expected mode collapse warning"
        )


# ---------------------------------------------------------------------------
# AC9  --  spectral_kl_health_check: KL explosion warning
# ---------------------------------------------------------------------------

class TestKLHealthCheckExplosion:
    """
    AC9: kl_z > 1e4 triggers RuntimeWarning about explosion.
    """

    def test_kl_explosion_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = spectral_kl_health_check(
                kl_z=2e4, kl_S=1.0, kl_tau=1.0,
                active_modes=5, q=8,
            )
        assert result["kl_z_ok"] is False
        assert any("explosion" in str(wi.message).lower() for wi in w), (
            "Expected KL explosion warning"
        )
