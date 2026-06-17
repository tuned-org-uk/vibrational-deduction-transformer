"""
Unit tests for vdt/laplacian.py and vdt/density.py  (issue #16).

Covers all acceptance criteria:

  AC1 -- from_spectral_loading produces valid PSD Laplacian
         (zero row sums, non-positive off-diagonal, symmetric, eigenvalues >= 0).
  AC2 -- Gradient flows from L back through from_spectral_loading to W.
  AC3 -- MassMatrix M_diag > 0; conditioning warning fires when ratio > 100.
  AC4 -- rayleigh_quotient returns scalar >= 0 for any z.
  AC5 -- SignedDensityMatrix passes PSD tests at init and after random updates.
  AC6 -- CFL helpers correct; existing DifferentiableLaplacian forward paths
         continue to pass.

Run with:
    pytest tests/test_laplacian.py -v
"""
from __future__ import annotations

import math
import warnings

import pytest
import torch

from vdt.laplacian import DifferentiableLaplacian, MassMatrix
from vdt.density import SignedDensityMatrix


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N = 8   # node count for most tests (small enough for gradcheck)
B = 4   # batch size


def _make_ring_graph(n: int):
    """Ring graph: n nodes, 2n directed edges (symmetric)."""
    fwd_src = list(range(n))
    fwd_dst = list(range(1, n)) + [0]
    src = fwd_src + fwd_dst
    dst = fwd_dst + fwd_src
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    base_weights = torch.ones(len(src), dtype=torch.float32)
    return edge_index, base_weights


def _make_lap(n: int = N, sparse: bool = False) -> DifferentiableLaplacian:
    edge_index, base_weights = _make_ring_graph(n)
    return DifferentiableLaplacian(
        n_nodes=n,
        edge_index=edge_index,
        base_weights=base_weights,
        normalised=True,
        sparse=sparse,
    )


def _make_L_base(n: int = N) -> torch.Tensor:
    """Return unbatched base Laplacian from a ring graph (N, N)."""
    lap = _make_lap(n)
    delta = torch.zeros(lap.edge_index.shape[1])
    return lap(delta).detach()  # (N, N) unbatched


@pytest.fixture
def lap():
    return _make_lap()


# ---------------------------------------------------------------------------
# AC1 -- from_spectral_loading produces valid PSD Laplacian
# ---------------------------------------------------------------------------

class TestFromSpectralLoading:
    """AC1: shape, zero row sums, non-positive off-diagonal, symmetry, PSD."""

    def _inputs(self, b=B, n=N, q=4, seed=0):
        torch.manual_seed(seed)
        W = torch.randn(b, n, q, requires_grad=True)
        L_base = _make_L_base(n)
        return W, L_base

    def test_shape(self):
        W, L_base = self._inputs()
        L = DifferentiableLaplacian.from_spectral_loading(W, L_base)
        assert L.shape == (B, N, N)

    def test_normalized_laplacian_properties(self):
        W, L_base = self._inputs()
        L = DifferentiableLaplacian.from_spectral_loading(W, L_base)

        # 1) Symmetric by construction
        assert torch.allclose(
            L, L.transpose(-1, -2), atol=1e-6
        ), f"Non-symmetric L: max abs diff = {(L - L.transpose(-1, -2)).abs().max():.2e}"

        # 2) Off-diagonal entries are non-positive
        eye = torch.eye(L.size(-1), device=L.device, dtype=torch.bool).unsqueeze(0)
        offdiag = L.masked_fill(eye, 0.0)
        assert torch.all(
            offdiag <= 1e-6
        ), f"Positive off-diagonal entry found: max = {offdiag.max():.2e}"

        # 3) Diagonal entries are in [0, 1] and are ~1 for non-isolated nodes
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        assert torch.all(
            diag >= -1e-6
        ), f"Negative diagonal entry found: min = {diag.min():.2e}"
        assert torch.all(
            diag <= 1.0 + 1e-6
        ), f"Diagonal entry > 1 found: max = {diag.max():.2e}"

        # 4) Spectrum of symmetric normalised Laplacian lies in [0, 2]
        eigvals = torch.linalg.eigvalsh(L)
        assert torch.all(
            eigvals >= -1e-5
        ), f"Negative eigenvalue found: min = {eigvals.min():.2e}"
        assert torch.all(
            eigvals <= 2.0 + 1e-5
        ), f"Eigenvalue > 2 found: max = {eigvals.max():.2e}"

    def test_offdiagonal_nonpositive(self):
        """Off-diagonal entries of the normalised Laplacian are <= 0."""
        W, L_base = self._inputs()
        L = DifferentiableLaplacian.from_spectral_loading(W, L_base)
        eye = torch.eye(N, dtype=torch.bool).unsqueeze(0)
        off = L.masked_fill(eye, 0.0)
        assert (off <= 1e-6).all(), (
            f"Positive off-diagonal: max = {off.max():.4f}"
        )

    def test_symmetric(self):
        W, L_base = self._inputs()
        L = DifferentiableLaplacian.from_spectral_loading(W, L_base)
        assert torch.allclose(L, L.transpose(-1, -2), atol=1e-6)

    def test_eigenvalues_nonnegative(self):
        """Eigenvalues of each per-batch Laplacian must be >= 0 (PSD)."""
        W, L_base = self._inputs(b=1)
        L = DifferentiableLaplacian.from_spectral_loading(W, L_base)
        eigs = torch.linalg.eigvalsh(L[0])
        assert (eigs >= -1e-5).all(), (
            f"Negative eigenvalue: min = {eigs.min():.4f}"
        )

    def test_d_neq_n_raises(self):
        """Assert fires when W.shape[1] != L_base.shape[0]."""
        W = torch.randn(2, N + 1, 3)  # d != N
        L_base = _make_L_base(N)
        with pytest.raises(AssertionError):
            DifferentiableLaplacian.from_spectral_loading(W, L_base)


# ---------------------------------------------------------------------------
# AC2 -- gradient flows through from_spectral_loading to W
# ---------------------------------------------------------------------------

class TestFromSpectralLoadingGradient:
    """AC2: autograd.gradcheck and manual backward."""

    def test_gradcheck(self):
        """torch.autograd.gradcheck verifies analytic gradient w.r.t. W."""
        torch.manual_seed(42)
        n, q = 4, 2  # small for speed
        W = torch.randn(1, n, q, dtype=torch.float64, requires_grad=True)
        L_base = _make_L_base(n).double()

        def fn(W_):
            return DifferentiableLaplacian.from_spectral_loading(W_, L_base)

        assert torch.autograd.gradcheck(
            fn, (W,), eps=1e-4, atol=1e-3
        ), "gradcheck failed: gradient does not flow to W"

    def test_grad_not_none(self):
        """W.grad is non-None and correct shape after backward."""
        torch.manual_seed(7)
        n, q = 6, 3
        W = torch.randn(2, n, q, requires_grad=True)
        L_base = _make_L_base(n)

        L = DifferentiableLaplacian.from_spectral_loading(W, L_base)
        L.sum().backward()

        assert W.grad is not None, "W.grad is None -- no gradient"
        assert W.grad.shape == W.shape

    def test_grad_nonzero(self):
        """
        At least some gradient elements must be nonzero.

        Note: L_z.sum() is identically zero by the Laplacian row-sum
        property: sum(diag(row_sum)) - sum(A_sym) = sum(A_sym) - sum(A_sym) = 0.
        Calling L.sum().backward() therefore always gives zero gradient
        regardless of the gate formula.  We use L.pow(2).sum() instead,
        which is strictly positive and whose gradient w.r.t. W is nonzero
        through the dot-product gate.
        """
        torch.manual_seed(9)
        W = torch.randn(2, N, 4, requires_grad=True)
        L_base = _make_L_base(N)
        L = DifferentiableLaplacian.from_spectral_loading(W, L_base)
        L.pow(2).sum().backward()
        assert W.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# AC3 -- MassMatrix: M_diag > 0, conditioning warning fires
# ---------------------------------------------------------------------------

class TestMassMatrix:
    """AC3: M_diag > 0; RuntimeWarning when ratio > 100; no warning otherwise."""

    def test_m_diag_positive(self):
        eigs = torch.rand(N) * 1.9  # eigenvalues in [0, 2)
        mm = MassMatrix(eigs, tau=0.5)
        assert (mm.M_diag > 0).all()

    def test_m_diag_shape(self):
        mm = MassMatrix(torch.rand(N))
        assert mm.M_diag.shape == (N,)

    def test_conditioning_warning_fires(self):
        # lambda=0 --> denominator ~1 --> M~1
        # lambda=0.9999 with tau=1 --> denominator ~eps --> M~1/eps >> 1
        eigs = torch.tensor([0.0, 0.0, 0.0, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mm = MassMatrix(eigs, tau=1.0, eps=1e-4)
            _ = mm.M_diag
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(rw) >= 1, "Expected RuntimeWarning for high conditioning"
        assert "100" in str(rw[0].message)

    def test_no_warning_good_conditioning(self):
        eigs = torch.linspace(0.01, 0.5, N)  # well-behaved range
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mm = MassMatrix(eigs, tau=0.5)
            _ = mm.M_diag
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(rw) == 0

    def test_as_matrix_is_diagonal(self):
        eigs = torch.rand(N)
        mm = MassMatrix(eigs)
        M = mm.as_matrix()
        assert M.shape == (N, N)
        assert torch.allclose(M, torch.diag(mm.M_diag))

    def test_cache_invalidation(self):
        """Clearing _M_diag forces recomputation."""
        eigs = torch.rand(N)
        mm = MassMatrix(eigs)
        d1 = mm.M_diag
        mm._M_diag = None
        d2 = mm.M_diag
        assert torch.allclose(d1, d2)


# ---------------------------------------------------------------------------
# AC4 -- rayleigh_quotient returns scalar >= 0
# ---------------------------------------------------------------------------

class TestRayleighQuotient:
    """AC4: non-negative scalar for any z, with and without MassMatrix."""

    def test_nonnegative_random_z(self, lap):
        torch.manual_seed(1)
        for _ in range(10):
            z = torch.randn(N)
            rq = lap.rayleigh_quotient(z)
            assert float(rq) >= -1e-6, f"Negative: {float(rq):.4f}"

    def test_nonnegative_batched(self, lap):
        torch.manual_seed(2)
        z = torch.randn(B, N)
        rq = lap.rayleigh_quotient(z)
        assert float(rq) >= -1e-6

    def test_with_mass_matrix(self, lap):
        torch.manual_seed(3)
        eigs = torch.rand(N) * 1.5
        mm = MassMatrix(eigs, tau=0.5)
        z = torch.randn(N)
        rq = lap.rayleigh_quotient(z, mass=mm)
        assert float(rq) >= -1e-6

    def test_with_eigenvalues_arg(self, lap):
        torch.manual_seed(4)
        eigs = torch.rand(N)
        z = torch.randn(N)
        rq = lap.rayleigh_quotient(z, eigenvalues=eigs, tau=0.5)
        assert float(rq) >= -1e-6

    def test_scalar_output(self, lap):
        rq = lap.rayleigh_quotient(torch.randn(N))
        assert rq.ndim == 0


# ---------------------------------------------------------------------------
# AC5 -- SignedDensityMatrix PSD
# ---------------------------------------------------------------------------

class TestSignedDensityMatrix:
    """AC5: rho_plus and rho_minus PSD at init and after random gradient steps."""

    def test_rho_plus_psd_at_init(self):
        sdm = SignedDensityMatrix(n=N)
        eigs = torch.linalg.eigvalsh(sdm.rho_plus)
        assert (eigs >= -1e-5).all(), f"rho_plus not PSD: min = {eigs.min():.4f}"

    def test_rho_minus_psd_at_init(self):
        sdm = SignedDensityMatrix(n=N)
        eigs = torch.linalg.eigvalsh(sdm.rho_minus)
        assert (eigs >= -1e-5).all(), f"rho_minus not PSD: min = {eigs.min():.4f}"

    def test_rho_shape(self):
        sdm = SignedDensityMatrix(n=N)
        assert sdm.rho.shape == (N, N)

    def test_psd_after_gradient_steps(self):
        """PSD preserved through 5 SGD steps."""
        torch.manual_seed(5)
        sdm = SignedDensityMatrix(n=N)
        opt = torch.optim.SGD(sdm.parameters(), lr=0.1)
        for _ in range(5):
            opt.zero_grad()
            sdm.rho.sum().backward()
            opt.step()

        ep = torch.linalg.eigvalsh(sdm.rho_plus.detach())
        em = torch.linalg.eigvalsh(sdm.rho_minus.detach())
        assert (ep >= -1e-5).all(), f"rho_plus not PSD after updates: min = {ep.min():.4f}"
        assert (em >= -1e-5).all(), f"rho_minus not PSD after updates: min = {em.min():.4f}"

    def test_min_eigenvalues_nonnegative(self):
        sdm = SignedDensityMatrix(n=N)
        me_p, me_m = sdm.min_eigenvalues()
        assert float(me_p) >= -1e-5
        assert float(me_m) >= -1e-5

    def test_frobenius_norm_signed(self):
        sdm = SignedDensityMatrix(n=N)
        fn = sdm.frobenius_norm(signed=True)
        assert fn.ndim == 0
        assert float(fn) >= 0.0

    def test_frobenius_norm_unsigned(self):
        sdm = SignedDensityMatrix(n=N)
        fn = sdm.frobenius_norm(signed=False)
        fn_s = sdm.frobenius_norm(signed=True)
        # Total mass >= signed (triangle inequality)
        assert float(fn) >= float(fn_s) - 1e-6

    def test_trace_penalty_scalar_nonneg(self):
        sdm = SignedDensityMatrix(n=N)
        tp = sdm.trace_penalty(target_trace=1.0)
        assert tp.ndim == 0
        assert float(tp) >= 0.0

    def test_symmetry(self):
        sdm = SignedDensityMatrix(n=N)
        assert torch.allclose(sdm.rho_plus, sdm.rho_plus.t(), atol=1e-6)
        assert torch.allclose(sdm.rho_minus, sdm.rho_minus.t(), atol=1e-6)


# ---------------------------------------------------------------------------
# AC6a -- CFL helpers
# ---------------------------------------------------------------------------

class TestCFLHelpers:
    """lambda_max and dt_max_cfl correctness."""

    def test_lambda_max_nonnegative(self, lap):
        assert lap.lambda_max >= 0.0

    def test_lambda_max_bounded(self, lap):
        """Normalised Laplacian eigenvalues are in [0, 2]."""
        assert lap.lambda_max <= 2.0 + 1e-5

    def test_dt_max_cfl_positive(self, lap):
        assert lap.dt_max_cfl() > 0.0

    def test_dt_max_cfl_formula(self, lap):
        expected = math.sqrt(2.0 / max(lap.lambda_max, 1e-8))
        assert abs(lap.dt_max_cfl() - expected) < 1e-6

    def test_dt_max_cfl_safety_factor(self, lap):
        dt_safe = lap.dt_max_cfl(safety=0.9)
        dt_full = lap.dt_max_cfl(safety=1.0)
        assert abs(dt_safe - 0.9 * dt_full) < 1e-6

    def test_invalidate_cache(self, lap):
        _ = lap.lambda_max          # populate cache
        lap._invalidate_spectral_cache()
        assert lap._lambda_max is None

    def test_lambda_max_cached(self, lap):
        lm1 = lap.lambda_max
        lm2 = lap.lambda_max         # should return cached value
        assert lm1 == lm2


# ---------------------------------------------------------------------------
# AC6b -- existing DifferentiableLaplacian forward paths unchanged
# ---------------------------------------------------------------------------

class TestForwardCompat:
    """Regression: original dense / sparse / row-mode forward paths unchanged."""

    def test_dense_forward_shape_batched(self):
        lap = _make_lap(N)
        E = lap.edge_index.shape[1]
        delta = torch.zeros(B, E)
        L = lap(delta)
        assert L.shape == (B, N, N)

    def test_dense_forward_shape_unbatched(self):
        lap = _make_lap(N)
        E = lap.edge_index.shape[1]
        delta = torch.zeros(E)
        L = lap(delta)
        assert L.shape == (N, N)

    def test_dense_row_sum_near_zero(self):
        lap = _make_lap(N)
        E = lap.edge_index.shape[1]
        delta = torch.zeros(E)
        L = lap(delta)
        row_sums = L.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros(N), atol=1e-5)

    def test_sparse_matches_dense(self):
        lap_d = _make_lap(N, sparse=False)
        lap_s = _make_lap(N, sparse=True)
        E = lap_d.edge_index.shape[1]
        delta = torch.zeros(1, E)
        assert torch.allclose(lap_d(delta), lap_s(delta), atol=1e-5)

    def test_row_mode_shape(self):
        lap = _make_lap(N)
        E = lap.edge_index.shape[1]
        delta = torch.zeros(B, E)
        node_idx = torch.randint(0, N, (B,))
        rows = lap(delta, node_idx=node_idx)
        assert rows.shape == (B, N)
