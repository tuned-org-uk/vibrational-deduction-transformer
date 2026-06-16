"""
Unit tests for wae/spectral.py -- v2 additions: spectral_basis_kl, tau_mode_kl.

Acceptance criteria from issue #24
-----------------------------------
AC1  spectral_basis_kl reduces to isotropic KL when eigvals_q are all
     ones and lam_s = 1.0.
AC2  spectral_basis_kl is >= 0, scales with lam_s, and is differentiable
     through S and log_var_S.
AC3  tau_mode_kl matches a Monte Carlo numerical estimate on (B=4, q=8)
     input to 3 significant figures.
AC4  No regressions on existing spectral.py symbols.
"""
from __future__ import annotations
import math
import pytest
import torch

from wae.spectral import (
    spectral_basis_kl,
    tau_mode_kl,
    TauModeDiffusion,
    spectral_freq_cost,
    lambda_fingerprint,
)

torch.manual_seed(0)

B, Q = 4, 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_inputs(b: int = B, q: int = Q):
    S = torch.randn(b, q, q)
    log_var_S = torch.randn(b, q, q)
    eigvals_q = torch.rand(q).abs() + 0.1  # strictly positive
    return S, log_var_S, eigvals_q


# ---------------------------------------------------------------------------
# AC1 -- reduces to isotropic KL when eigvals = 1, lam_s = 1
# ---------------------------------------------------------------------------

class TestSpectralBasisKlIsotropicLimit:
    """spectral_basis_kl must equal the standard VAE KL when the prior is unit Gaussian."""

    def test_matches_isotropic_kl_scalar(self):
        """Mean KL equals the analytic isotropic KL reference."""
        S, log_var_S, _ = _random_inputs()
        eigvals_ones = torch.ones(Q)

        got = spectral_basis_kl(S, log_var_S, eigvals_ones, lam_s=1.0)

        # Reference: standard Gaussian KL = 0.5*(var + mu^2 - log_var - 1)
        var_S = log_var_S.exp()
        ref = 0.5 * (var_S + S.pow(2) - log_var_S - 1.0)
        ref_scalar = ref.sum(dim=(-2, -1)).mean()

        assert torch.allclose(got, ref_scalar, atol=1e-5), (
            f"Expected {ref_scalar.item():.6f}, got {got.item():.6f}"
        )

    def test_matches_isotropic_kl_zero_mean(self):
        """Zero-mean posterior with unit variance gives KL = 0."""
        S = torch.zeros(B, Q, Q)
        log_var_S = torch.zeros(B, Q, Q)  # var = 1
        eigvals_ones = torch.ones(Q)

        got = spectral_basis_kl(S, log_var_S, eigvals_ones, lam_s=1.0)
        assert got.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# AC2 -- non-negativity, monotonicity in lam_s, differentiability
# ---------------------------------------------------------------------------

class TestSpectralBasisKlProperties:

    def test_non_negative(self):
        S, log_var_S, eigvals_q = _random_inputs()
        val = spectral_basis_kl(S, log_var_S, eigvals_q)
        assert val.item() >= -1e-6, f"KL was negative: {val.item()}"

    def test_scales_with_lam_s(self):
        """Larger lam_s means tighter prior, should increase KL."""
        S, log_var_S, eigvals_q = _random_inputs()
        kl_low = spectral_basis_kl(S, log_var_S, eigvals_q, lam_s=0.5)
        kl_high = spectral_basis_kl(S, log_var_S, eigvals_q, lam_s=4.0)
        # KL is not guaranteed monotone for all realisations, but on
        # average (B=4, q=8) with large posterior spread it should hold.
        assert kl_high.item() > kl_low.item()

    def test_gradient_flows_through_S(self):
        S = torch.randn(B, Q, Q, requires_grad=True)
        log_var_S = torch.randn(B, Q, Q)
        eigvals_q = torch.rand(Q).abs() + 0.1
        kl = spectral_basis_kl(S, log_var_S, eigvals_q)
        kl.backward()
        assert S.grad is not None
        assert torch.isfinite(S.grad).all()

    def test_gradient_flows_through_log_var_S(self):
        S = torch.randn(B, Q, Q)
        log_var_S = torch.randn(B, Q, Q, requires_grad=True)
        eigvals_q = torch.rand(Q).abs() + 0.1
        kl = spectral_basis_kl(S, log_var_S, eigvals_q)
        kl.backward()
        assert log_var_S.grad is not None
        assert torch.isfinite(log_var_S.grad).all()

    def test_output_is_scalar(self):
        S, log_var_S, eigvals_q = _random_inputs()
        val = spectral_basis_kl(S, log_var_S, eigvals_q)
        assert val.shape == torch.Size([])


# ---------------------------------------------------------------------------
# AC3 -- tau_mode_kl matches Monte Carlo estimate to 3 sig figs
# ---------------------------------------------------------------------------

class TestTauModeKlMonteCarlo:
    """
    Verifies tau_mode_kl against a Monte Carlo numerical reference.

    MC estimator of KL( Gamma(a,b) || Exp(r) ):
      1. Draw N_s samples omega ~ Gamma(a, b)  (PyTorch parameterisation:
         Gamma(concentration=a, rate=b))
      2. log q = log Gamma-pdf(omega; a, b)
      3. log p = log Exp-pdf(omega; r) = log(r) - r*omega
      4. KL ~= mean(log q - log p)
    """

    N_SAMPLES = 200_000
    REL_TOL = 5e-3  # 0.5 % -- well within 3 sig-fig requirement

    def _mc_kl_gamma_exp(self, a: torch.Tensor, b: torch.Tensor, r: torch.Tensor) -> float:
        """
        Monte Carlo estimate of mean_batch sum_modes KL( Gamma(a,b) || Exp(r) ).
        a, b, r all shape (B, q).
        """
        total = 0.0
        n = a.shape[0] * a.shape[1]
        for bi in range(a.shape[0]):
            kl_modes = 0.0
            for ki in range(a.shape[1]):
                ai = a[bi, ki].item()
                bi_ = b[bi, ki].item()
                ri = r[bi, ki].item()
                dist_q = torch.distributions.Gamma(
                    concentration=torch.tensor(ai),
                    rate=torch.tensor(bi_),
                )
                omega = dist_q.sample((self.N_SAMPLES,))
                log_q = dist_q.log_prob(omega)
                log_p = torch.log(torch.tensor(ri)) - ri * omega
                kl_modes += (log_q - log_p).mean().item()
            total += kl_modes
        # mean over batch, sum over modes
        return total / a.shape[0]

    def test_matches_mc_estimate(self):
        torch.manual_seed(42)
        log_a = torch.randn(B, Q) * 0.3         # a ~ exp(small noise) => a close to 1
        log_b = torch.randn(B, Q) * 0.3
        eigvals_q = torch.rand(Q).abs() + 0.2   # strictly > 0
        tau = 1.0

        closed = tau_mode_kl(log_a, log_b, eigvals_q, tau=tau).item()

        a = log_a.exp()
        b = log_b.exp()
        r = (tau * eigvals_q.clamp(min=1e-6)).unsqueeze(0).expand(B, -1)
        mc = self._mc_kl_gamma_exp(a, b, r)

        rel_err = abs(closed - mc) / (abs(mc) + 1e-8)
        assert rel_err < self.REL_TOL, (
            f"tau_mode_kl closed={closed:.6f} MC={mc:.6f} "
            f"rel_err={rel_err:.4f} > {self.REL_TOL}"
        )


# ---------------------------------------------------------------------------
# AC3 (additional) -- tau_mode_kl properties
# ---------------------------------------------------------------------------

class TestTauModeKlProperties:

    def test_output_is_scalar(self):
        log_a = torch.randn(B, Q)
        log_b = torch.randn(B, Q)
        eigvals_q = torch.rand(Q).abs() + 0.1
        val = tau_mode_kl(log_a, log_b, eigvals_q)
        assert val.shape == torch.Size([])

    def test_gradient_flows_through_log_a_log_b(self):
        log_a = torch.randn(B, Q, requires_grad=True)
        log_b = torch.randn(B, Q, requires_grad=True)
        eigvals_q = torch.rand(Q).abs() + 0.1
        val = tau_mode_kl(log_a, log_b, eigvals_q)
        val.backward()
        assert log_a.grad is not None and torch.isfinite(log_a.grad).all()
        assert log_b.grad is not None and torch.isfinite(log_b.grad).all()

    def test_finite_on_extreme_params(self):
        """Very small and very large (a, b) should not produce nan/inf."""
        log_a = torch.tensor([[math.log(0.1), math.log(10.0)] * (Q // 2)] * B)
        log_b = torch.tensor([[math.log(0.1), math.log(10.0)] * (Q // 2)] * B)
        eigvals_q = torch.ones(Q)
        val = tau_mode_kl(log_a, log_b, eigvals_q)
        assert torch.isfinite(val)


# ---------------------------------------------------------------------------
# AC4 -- no regressions on existing symbols
# ---------------------------------------------------------------------------

class TestNoRegressions:

    def test_tau_mode_diffusion_importable(self):
        assert TauModeDiffusion is not None
        m = TauModeDiffusion(tau_modes=4)
        assert hasattr(m, "log_t")

    def test_spectral_freq_cost_importable(self):
        L = torch.eye(6).unsqueeze(0).repeat(2, 1, 1)
        cost = spectral_freq_cost(L, tau_modes=2)
        assert torch.isfinite(cost)

    def test_lambda_fingerprint_importable(self):
        L = torch.eye(8).unsqueeze(0)
        fp = lambda_fingerprint(L, tau_modes=4, n_bins=8)
        assert fp.shape == (1, 8)
