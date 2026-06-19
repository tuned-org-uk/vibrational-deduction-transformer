"""
Unit tests for vdt/spectral.py --  additions: spectral_basis_kl, tau_mode_kl,
lambda_fingerprint_soft.

Acceptance criteria from issue #24
-----------------------------------
AC1  spectral_basis_kl reduces to isotropic KL when eigvals_q are all
     ones and lam_s = 1.0.
AC2  spectral_basis_kl is >= 0, scales with lam_s, and is differentiable
     through S and log_var_S.
AC3  tau_mode_kl matches a Monte Carlo numerical estimate on (B=4, q=8)
     input to 3 significant figures.
AC4  No regressions on existing spectral.py symbols.

Acceptance criteria from issue #56
-----------------------------------
AC5  lambda_fingerprint_soft is differentiable: eigvals.grad is not None
     and finite after .backward().
AC6  lambda_fingerprint_soft output lives on the same device as eigvals
     (no silent CPU migration).
AC7  lambda_fingerprint_soft output shape is (B, n_bins) for batched input
     and (n_bins,) for unbatched input; each row sums to 1.
AC8  lambda_fingerprint_soft peak bin agrees with lambda_fingerprint_hard
     to within +/-1 bin on a synthetic well-separated spectrum.
AC9  lambda_fingerprint alias still resolves (backwards-compat).
"""
from __future__ import annotations
import math
import pytest
import torch

from vdt.spectral import (
    spectral_basis_kl,
    tau_mode_kl,
    TauModeDiffusion,
    spectral_freq_cost,
    lambda_fingerprint,
    lambda_fingerprint_hard,
    lambda_fingerprint_soft,
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
        """Backwards-compat alias must still resolve and return correct shape."""
        L = torch.eye(8).unsqueeze(0)
        fp = lambda_fingerprint(L, tau_modes=4, n_bins=8)
        assert fp.shape == (1, 8)


# ---------------------------------------------------------------------------
# AC5 -- lambda_fingerprint_soft is differentiable w.r.t. eigvals
# ---------------------------------------------------------------------------

class TestLambdaFingerprintSoftDifferentiability:
    """
    Gradient must flow back from the soft-histogram output through eigvals.
    This is the primary requirement from issue #56: the fingerprint must
    be usable as a genuine training signal.
    """

    def test_grad_flows_through_eigvals_batched(self):
        eigvals = torch.rand(B, Q, requires_grad=True)
        fp = lambda_fingerprint_soft(eigvals, n_bins=16)
        # Use entropy as a scalar loss to trigger .backward()
        loss = -(fp * (fp + 1e-8).log()).sum()
        loss.backward()
        assert eigvals.grad is not None, "No gradient on eigvals after backward()"
        assert torch.isfinite(eigvals.grad).all(), "Non-finite gradient on eigvals"

    def test_grad_flows_through_eigvals_unbatched(self):
        eigvals = torch.rand(Q, requires_grad=True)
        fp = lambda_fingerprint_soft(eigvals, n_bins=16)
        loss = fp.sum()
        loss.backward()
        assert eigvals.grad is not None
        assert torch.isfinite(eigvals.grad).all()

    def test_grad_is_nonzero(self):
        """Gradient should not be identically zero for a generic input."""
        eigvals = torch.rand(B, Q, requires_grad=True)
        fp = lambda_fingerprint_soft(eigvals, n_bins=16)
        fp.sum().backward()
        assert eigvals.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# AC6 -- device consistency
# ---------------------------------------------------------------------------

class TestLambdaFingerprintSoftDevice:
    """
    Output device must match input device.  The original torch.histc
    implementation forced CPU; lambda_fingerprint_soft must not.
    """

    def test_output_on_cpu_for_cpu_input(self):
        eigvals = torch.rand(B, Q)
        fp = lambda_fingerprint_soft(eigvals, n_bins=16)
        assert fp.device == eigvals.device

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_output_on_cuda_for_cuda_input(self):
        eigvals = torch.rand(B, Q).cuda()
        fp = lambda_fingerprint_soft(eigvals, n_bins=16)
        assert fp.device.type == "cuda"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_output_on_mps_for_mps_input(self):
        eigvals = torch.rand(B, Q).to("mps")
        fp = lambda_fingerprint_soft(eigvals, n_bins=16)
        assert fp.device.type == "mps"


# ---------------------------------------------------------------------------
# AC7 -- output shape and normalisation
# ---------------------------------------------------------------------------

class TestLambdaFingerprintSoftShape:

    def test_batched_shape(self):
        eigvals = torch.rand(B, Q)
        fp = lambda_fingerprint_soft(eigvals, n_bins=32)
        assert fp.shape == (B, 32), f"Expected ({B}, 32), got {fp.shape}"

    def test_unbatched_shape(self):
        eigvals = torch.rand(Q)
        fp = lambda_fingerprint_soft(eigvals, n_bins=16)
        assert fp.shape == (16,), f"Expected (16,), got {fp.shape}"

    def test_each_row_sums_to_one(self):
        eigvals = torch.rand(B, Q)
        fp = lambda_fingerprint_soft(eigvals, n_bins=32)
        row_sums = fp.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), (
            f"Row sums not all 1.0: {row_sums}"
        )

    def test_unbatched_sums_to_one(self):
        eigvals = torch.rand(Q)
        fp = lambda_fingerprint_soft(eigvals, n_bins=16)
        assert fp.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_all_non_negative(self):
        eigvals = torch.rand(B, Q)
        fp = lambda_fingerprint_soft(eigvals, n_bins=32)
        assert (fp >= 0).all()


# ---------------------------------------------------------------------------
# AC8 -- soft histogram peak agrees with hard histogram on well-separated
#        synthetic spectrum
# ---------------------------------------------------------------------------

class TestLambdaFingerprintSoftVsHard:
    """
    On a synthetic eigenvalue cluster placed at a known frequency, the
    dominant bin in lambda_fingerprint_soft should agree with the dominant
    bin in lambda_fingerprint_hard to within +/-1 bin.

    Synthetic spectrum: N=32 eigenvalues clustered tightly around 1.0
    (mid-range of the normalised Laplacian [0, 2]).  The majority of
    eigenvalue mass should map to bins near the midpoint of the range.
    """

    def test_peak_bin_agreement(self):
        N_BINS = 32
        # Cluster all eigenvalues near 1.0 with tiny noise
        torch.manual_seed(7)
        eigvals_1d = torch.ones(16) + torch.randn(16) * 0.02  # cluster at lam ~ 1.0
        eigvals_1d = eigvals_1d.clamp(0.0, 2.0)

        # Soft fingerprint (differentiable)
        fp_soft = lambda_fingerprint_soft(
            eigvals_1d.unsqueeze(0), n_bins=N_BINS, lam_max=2.0, bandwidth=0.05
        ).squeeze(0)  # (N_BINS,)

        # Hard fingerprint -- wrap as (1, N, N) dummy Laplacian and pass
        # pre-computed eigvals to skip the eigensolver.
        dummy_L = torch.eye(16).unsqueeze(0)  # (1, 16, 16)
        fp_hard = lambda_fingerprint_hard(
            dummy_L, tau_modes=16, n_bins=N_BINS,
            eigvals=eigvals_1d.unsqueeze(0),
        ).squeeze(0)  # (N_BINS,)

        peak_soft = fp_soft.argmax().item()
        peak_hard = fp_hard.argmax().item()

        assert abs(peak_soft - peak_hard) <= 1, (
            f"Soft peak bin={peak_soft} and hard peak bin={peak_hard} "
            f"differ by more than 1 bin on a well-separated spectrum."
        )


# ---------------------------------------------------------------------------
# AC9 -- backwards-compat alias
# ---------------------------------------------------------------------------

class TestBackwardsCompatAlias:

    def test_lambda_fingerprint_is_hard(self):
        """lambda_fingerprint must still resolve as lambda_fingerprint_hard."""
        assert lambda_fingerprint is lambda_fingerprint_hard

    def test_hard_importable_separately(self):
        assert lambda_fingerprint_hard is not None

    def test_soft_importable_separately(self):
        assert lambda_fingerprint_soft is not None
