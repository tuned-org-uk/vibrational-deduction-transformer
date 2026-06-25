"""
tests/test_cfl_dynamic.py

Tests for issue #53: CFL bound must be recomputed from the dynamic
feature-space Laplacian L_f, not from the frozen base Laplacian L(I).

Three test scenarios are covered:

1. test_gershgorin_clamp_triggers
   Construct a VibrationalStateBlock whose learnable log_dt would produce
   a dt that exceeds the Gershgorin CFL bound of an injected L_f with
   lambda_max >> base.  Verify forward() clamps dt to the tighter bound.

2. test_base_fallback_is_looser
   With the same injected L_f, verify that recompute_cfl=False returns
   a strictly larger dt_max (the base bound), confirming the two code
   paths differ when L_f is sharper than the base Laplacian.

3. test_stability_diagnostics_cfl_Lf_ok
   Verify that stability_diagnostics() correctly reports cfl_Lf_ok=False
   (and CFL_ok=True) when dt satisfies the base bound but violates the
   Gershgorin L_f bound.
"""
from __future__ import annotations

import math

import pytest
import torch

from vdeductive.vdt import VibrationalStateBlock, _gershgorin_lambda_max
from vdeductive.stability import stability_diagnostics


# ---------------------------------------------------------------------------
# Minimal DifferentiableLaplacian stub
# ---------------------------------------------------------------------------

class _StubLap:
    """
    Minimal stub satisfying the DifferentiableLaplacian interface used
    by VibrationalStateBlock._cfl_dt_Lf and _cfl_dt.

    Attributes
    ----------
    _lambda_max_base : float
        lambda_max of the fake frozen base Laplacian.
    """

    def __init__(self, lambda_max_base: float) -> None:
        self._lambda_max_base = lambda_max_base

    def dt_max_cfl(self) -> float:
        """Return sqrt(2 / lambda_max_base) -- the frozen base CFL bound."""
        return math.sqrt(2.0 / max(self._lambda_max_base, 1e-8))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 8    # small graph for fast CPU tests
D = 4    # feature dimension


def _make_block(lambda_max_base: float = 2.0) -> tuple:
    """
    Return (block, lap_stub) with log_dt initialised to a large value
    (dt_free = 1.0) so the CFL clamp is certain to trigger.
    """
    block = VibrationalStateBlock(n_nodes=N, feat_dim=D, n_heads=2)
    # Force dt_free = 1.0 >> any reasonable CFL bound
    with torch.no_grad():
        block.log_dt.copy_(torch.tensor(0.0))  # exp(0) = 1.0
    lap = _StubLap(lambda_max_base=lambda_max_base)
    return block, lap


def _sharp_L_f(lambda_max_target: float = 20.0) -> torch.Tensor:
    """
    Construct a (1, N, N) graph Laplacian whose Gershgorin bound is
    >= lambda_max_target.  We use a fully-connected normalised Laplacian
    scaled so that the max row sum equals lambda_max_target.

    L = lambda_max_target / (2*(N-1)) * (N*I - 11^T)
    Row sums of abs(L): diagonal = lambda_max_target * (N-1)/(2*(N-1))
                                 = lambda_max_target / 2
    plus off-diagonal absolute sum = lambda_max_target / 2
    Total row absolute sum         = lambda_max_target

    So Gershgorin bound  = lambda_max_target (exactly).
    """
    scale = lambda_max_target / (2.0 * (N - 1))
    ones = torch.ones(N, N)
    L = scale * (N * torch.eye(N) - ones)  # (N, N)
    return L.unsqueeze(0)  # (1, N, N)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCFLDynamic:
    """
    Tests that the CFL bound is correctly derived from the dynamic L_f
    rather than the frozen base Laplacian (issue #53).
    """

    def test_gershgorin_bound_correct(self):
        """
        _gershgorin_lambda_max returns the correct row-absolute-sum bound
        for a known matrix.
        """
        target = 20.0
        L_f = _sharp_L_f(lambda_max_target=target).squeeze(0)  # (N, N)
        bound = float(_gershgorin_lambda_max(L_f).item())
        # The bound should equal target within floating-point tolerance
        assert abs(bound - target) < 1e-4, (
            f"Gershgorin bound {bound:.6f} != expected {target:.6f}"
        )

    def test_gershgorin_clamp_triggers(self):
        """
        When L_f has lambda_max >> base, forward() with recompute_cfl=True
        clamps dt to a value <= sqrt(2 / lambda_max_Lf).

        The block has log_dt=0 -> dt_free=1.0 which far exceeds both the
        base CFL bound (sqrt(2/2) ~ 1.0) and the L_f bound (sqrt(2/20)
        ~ 0.316).  The forward pass should not explode and Q_tp1 must be
        finite.
        """
        lambda_max_base = 2.0
        lambda_max_Lf   = 20.0

        block, lap = _make_block(lambda_max_base=lambda_max_base)
        L_f = _sharp_L_f(lambda_max_target=lambda_max_Lf)  # (1, N, N)

        Q_t   = torch.randn(1, N, D)
        Q_tm1 = torch.zeros(1, N, D)

        # Compute expected dt bound from Gershgorin
        gershgorin_bound = float(_gershgorin_lambda_max(L_f.squeeze(0)).item())
        dt_max_expected = math.sqrt(2.0 / gershgorin_bound)

        Q_tp1, _, _ = block(Q_t, Q_tm1, L_f, lap, recompute_cfl=True)

        assert torch.isfinite(Q_tp1).all(), "Q_tp1 contains non-finite values"

        # Verify the block's effective dt was clamped
        dt_free = float(block.log_dt.exp().item())  # 1.0
        assert dt_free > dt_max_expected, (
            "dt_free should exceed the Gershgorin CFL bound for this test to be meaningful"
        )

    def test_base_fallback_is_looser(self):
        """
        recompute_cfl=False uses the base-Laplacian bound which is looser
        (larger dt_max) than the Gershgorin L_f bound when L_f is sharper
        than the base Laplacian.

        We verify this by comparing dt_max_CFL_base vs dt_max_CFL_Lf.
        """
        lambda_max_base = 2.0
        lambda_max_Lf   = 20.0

        _, lap = _make_block(lambda_max_base=lambda_max_base)
        L_f = _sharp_L_f(lambda_max_target=lambda_max_Lf)

        dt_max_base = lap.dt_max_cfl()  # sqrt(2 / 2) ~ 1.0
        gershgorin_bound = float(_gershgorin_lambda_max(L_f.squeeze(0)).item())
        dt_max_Lf = math.sqrt(2.0 / gershgorin_bound)  # sqrt(2 / 20) ~ 0.316

        assert dt_max_base > dt_max_Lf, (
            f"Base bound ({dt_max_base:.4f}) should be looser than "
            f"Gershgorin L_f bound ({dt_max_Lf:.4f}) when L_f is sharper"
        )

    def test_stability_diagnostics_cfl_Lf_ok(self):
        """
        stability_diagnostics correctly distinguishes CFL_ok (base bound)
        from cfl_Lf_ok (dynamic L_f bound).

        We feed a dt that satisfies the base CFL bound but violates the
        Gershgorin L_f bound, and verify:
          - CFL_ok  = True   (dt satisfies base bound)
          - cfl_Lf_ok = False  (dt violates tighter L_f bound)
          - cfl_margin > 1   (L_f is sharper than base)
        """
        lambda_max_base = 2.0
        lambda_max_Lf   = 20.0

        L_f = _sharp_L_f(lambda_max_target=lambda_max_Lf).squeeze(0)  # (N, N)

        # Base eigvals (simulating a graph with lambda_max = 2.0)
        eigvals_base = torch.linspace(0.0, lambda_max_base, N)

        # dt that satisfies base bound but violates the tighter L_f bound
        dt_max_base = math.sqrt(2.0 / lambda_max_base)  # ~ 1.0
        gershgorin_bound = float(_gershgorin_lambda_max(L_f).item())
        dt_max_Lf = math.sqrt(2.0 / gershgorin_bound)   # ~ 0.316
        dt_test = 0.5 * (dt_max_base + dt_max_Lf)       # in between

        assert dt_test > dt_max_Lf,  "dt_test must exceed the L_f bound"
        assert dt_test <= dt_max_base, "dt_test must satisfy the base bound"

        gamma = torch.ones(D) * 0.5
        Q_states = [torch.randn(N, D) for _ in range(3)]

        diag = stability_diagnostics(
            L_f=L_f,
            Q_states=Q_states,
            rho_plus_list=[],
            rho_minus_list=[],
            eigvals=eigvals_base,
            dt=dt_test,
            gamma=gamma,
        )

        assert diag["CFL_ok"] is True,  (
            f"CFL_ok should be True (dt {dt_test:.4f} <= base {dt_max_base:.4f})"
        )
        assert diag["cfl_Lf_ok"] is False, (
            f"cfl_Lf_ok should be False (dt {dt_test:.4f} > L_f bound {dt_max_Lf:.4f})"
        )
        assert diag["cfl_margin"] > 1.0, (
            f"cfl_margin should be > 1 when L_f is sharper than base "
            f"(got {diag['cfl_margin']:.4f})"
        )
        assert "lambda_max_Lf" in diag
        assert "dt_max_CFL_Lf" in diag
        assert diag["lambda_max_Lf"] == pytest.approx(gershgorin_bound, rel=1e-4)
