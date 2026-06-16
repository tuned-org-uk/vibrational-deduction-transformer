"""
Unit tests for vdt/vdt.py (VibrationalStateBlock, VDT) and the
WiringEncoder / ModeWeightHead additions to vdt/encoder.py  (issue #17).

Acceptance criteria
-------------------
AC1  VDT.forward() returns (Q_K, Q_states, (rho_plus_list, rho_minus_list)).
AC2  dt is never above dt_max_cfl in any forward pass.
AC3  gamma > 0 always (softplus constraint enforced).
AC4  WiringEncoder.forward() returns (z, mu, log_var, log_a, log_b)
     with all shapes (B, latent_dim).
AC5  Old WiringEncoder-based training loop still works (no regression).
AC6  Unit tests: shape checks, CFL clamp, Q_states length K.

Run with:
    pytest tests/test_vdt.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vdt.vdt import VDT, VibrationalStateBlock
from vdt.encoder import ModeWeightHead, WiringEncoder, WiringEncoder
from vdt.laplacian import DifferentiableLaplacian


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N   = 8    # nodes
D   = 4    # feat_dim
B   = 3    # batch
Q   = 8    # latent_dim
INP = 16   # raw input dim


def _make_ring(n: int):
    fwd_src = list(range(n))
    fwd_dst = list(range(1, n)) + [0]
    src = fwd_src + fwd_dst
    dst = fwd_dst + fwd_src
    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    base_weights = torch.ones(len(src))
    return edge_index, base_weights


@pytest.fixture
def lap():
    edge_index, base_weights = _make_ring(N)
    return DifferentiableLaplacian(
        n_nodes=N, edge_index=edge_index,
        base_weights=base_weights, normalised=True,
    )


@pytest.fixture
def L_f(lap):
    """Batched (B, N, N) base Laplacian."""
    delta = torch.zeros(B, lap.edge_index.shape[1])
    return lap(delta).detach()  # (B, N, N)


@pytest.fixture
def eigvecs():
    """(N, N) random orthonormal -- realistic placeholder."""
    A = torch.randn(N, N)
    Q_mat, _ = torch.linalg.qr(A)
    return Q_mat  # (N, N) orthonormal columns


@pytest.fixture
def vdt_block(lap):
    return VibrationalStateBlock(n_nodes=N, feat_dim=D, n_heads=2)


@pytest.fixture
def vdt_model():
    return VDT(n_nodes=N, feat_dim=D, n_layers=3, m_modes=2, n_heads=2)


@pytest.fixture
def enc():
    return WiringEncoder(
        input_dim=INP, latent_dim=Q,
        n_nodes=N, feat_dim=D,
        n_layers=2, m_modes=2,
        n_heads=2, use_lambda_features=False,
    )


# ---------------------------------------------------------------------------
# AC2 / AC3  VibrationalStateBlock
# ---------------------------------------------------------------------------

class TestVibrationalStateBlock:
    """Shape, CFL clamp, gamma positivity, density shapes."""

    def test_output_shape_batched(self, vdt_block, L_f, lap):
        Q_t   = torch.randn(B, N, D)
        Q_tm1 = torch.zeros(B, N, D)
        Q_tp1, rp, rm = vdt_block(Q_t, Q_tm1, L_f, lap)
        assert Q_tp1.shape == (B, N, D)

    def test_output_shape_unbatched(self, vdt_block, L_f, lap):
        Q_t   = torch.randn(N, D)
        Q_tm1 = torch.zeros(N, D)
        Q_tp1, rp, rm = vdt_block(Q_t, Q_tm1, L_f[0], lap)
        assert Q_tp1.shape == (N, D)

    def test_cfl_clamp_dt_le_dt_max(self, vdt_block, lap):
        """AC2: dt must be <= dt_max_cfl after clamping."""
        dt   = vdt_block._cfl_dt(lap).item()
        dtmax = lap.dt_max_cfl()
        assert dt <= dtmax + 1e-6, (
            f"CFL violated: dt={dt:.4f} > dt_max={dtmax:.4f}"
        )

    def test_cfl_clamp_large_log_dt(self, vdt_block, lap):
        """Force log_dt very large; dt must still be <= dt_max_cfl."""
        with torch.no_grad():
            vdt_block.log_dt.fill_(10.0)   # exp(10) >> dt_max
        dt    = vdt_block._cfl_dt(lap).item()
        dtmax = lap.dt_max_cfl()
        assert dt <= dtmax + 1e-6

    def test_gamma_positive_at_init(self, vdt_block):
        """AC3: gamma > 0 via softplus."""
        assert (vdt_block.gamma > 0).all()

    def test_gamma_positive_after_updates(self, vdt_block, L_f, lap):
        """AC3: gamma stays > 0 after 5 gradient steps."""
        opt = torch.optim.SGD(vdt_block.parameters(), lr=0.1)
        Q_t   = torch.randn(B, N, D)
        Q_tm1 = torch.zeros(B, N, D)
        for _ in range(5):
            opt.zero_grad()
            out, _, _ = vdt_block(Q_t, Q_tm1, L_f, lap)
            out.sum().backward()
            opt.step()
        assert (vdt_block.gamma > 0).all()

    def test_density_shapes(self, vdt_block, L_f, lap):
        """rho_plus and rho_minus are (N, N)."""
        Q_t   = torch.randn(B, N, D)
        Q_tm1 = torch.zeros(B, N, D)
        _, rp, rm = vdt_block(Q_t, Q_tm1, L_f, lap)
        assert rp.shape == (N, N)
        assert rm.shape == (N, N)


# ---------------------------------------------------------------------------
# AC1 / AC6  VDT
# ---------------------------------------------------------------------------

class TestVDT:
    """AC1 return type, Q_states length K+1, shapes."""

    def test_forward_return_type(self, vdt_model, L_f, eigvecs, lap):
        """AC1: returns (Q_K, Q_states, (rho_plus_list, rho_minus_list))."""
        X0 = torch.randn(B, N, D)
        result = vdt_model(X0, L_f, eigvecs, lap)
        assert len(result) == 3
        Q_K, Q_states, (rp_list, rm_list) = result
        assert isinstance(Q_states, list)
        assert isinstance(rp_list, list)
        assert isinstance(rm_list, list)

    def test_q_states_length(self, vdt_model, L_f, eigvecs, lap):
        """AC6: Q_states has length K+1."""
        X0 = torch.randn(B, N, D)
        _, Q_states, _ = vdt_model(X0, L_f, eigvecs, lap)
        assert len(Q_states) == vdt_model.n_layers + 1

    def test_rho_lists_length(self, vdt_model, L_f, eigvecs, lap):
        """rho_plus_list and rho_minus_list each have length K."""
        X0 = torch.randn(B, N, D)
        _, _, (rp_list, rm_list) = vdt_model(X0, L_f, eigvecs, lap)
        assert len(rp_list)  == vdt_model.n_layers
        assert len(rm_list)  == vdt_model.n_layers

    def test_Q_K_shape_batched(self, vdt_model, L_f, eigvecs, lap):
        X0 = torch.randn(B, N, D)
        Q_K, _, _ = vdt_model(X0, L_f, eigvecs, lap)
        assert Q_K.shape == (B, N, D)

    def test_Q_K_shape_unbatched(self, vdt_model, eigvecs, lap):
        """Unbatched (N, d) input."""
        edge_index, bw = _make_ring(N)
        L_f_unb = DifferentiableLaplacian(
            n_nodes=N, edge_index=edge_index,
            base_weights=bw, normalised=True,
        )(torch.zeros(edge_index.shape[1])).detach()  # (N, N)
        X0 = torch.randn(N, D)
        Q_K, _, _ = vdt_model(X0, L_f_unb, eigvecs, lap)
        assert Q_K.shape == (N, D)

    def test_modal_projection_shape(self, vdt_model, L_f, eigvecs, lap):
        X0 = torch.randn(B, N, D)
        Q_K, _, _ = vdt_model(X0, L_f, eigvecs, lap)
        z = vdt_model.modal_projection(Q_K, eigvecs)
        assert z.shape == (B, D)

    def test_gradient_flows_to_X0(self, vdt_model, L_f, eigvecs, lap):
        X0 = torch.randn(B, N, D, requires_grad=True)
        Q_K, _, _ = vdt_model(X0, L_f, eigvecs, lap)
        Q_K.sum().backward()
        assert X0.grad is not None
        assert X0.grad.shape == X0.shape


# ---------------------------------------------------------------------------
# AC4  WiringEncoder
# ---------------------------------------------------------------------------

class TestWiringEncoder:
    """AC4: 5-tuple output, correct shapes, kl_loss non-negative."""

    def test_forward_returns_5_tuple(self, enc, L_f, eigvecs, lap):
        x = torch.randn(B, INP)
        out = enc(x, L_f, eigvecs, lap)
        assert len(out) == 5

    def test_output_shapes(self, enc, L_f, eigvecs, lap):
        x = torch.randn(B, INP)
        z, mu, log_var, log_a, log_b = enc(x, L_f, eigvecs, lap)
        for name, t in [("z", z), ("mu", mu), ("log_var", log_var),
                        ("log_a", log_a), ("log_b", log_b)]:
            assert t.shape == (B, Q), (
                f"{name} shape {t.shape} != ({B}, {Q})"
            )

    def test_kl_loss_nonnegative(self, enc, L_f, eigvecs, lap):
        x = torch.randn(B, INP)
        _, mu, log_var, _, _ = enc(x, L_f, eigvecs, lap)
        kl = enc.kl_loss(mu, log_var)
        assert float(kl) >= 0.0

    def test_kl_loss_scalar(self, enc, L_f, eigvecs, lap):
        x = torch.randn(B, INP)
        _, mu, log_var, _, _ = enc(x, L_f, eigvecs, lap)
        assert enc.kl_loss(mu, log_var).ndim == 0

    def test_gradient_flows_to_input(self, enc, L_f, eigvecs, lap):
        x = torch.randn(B, INP, requires_grad=True)
        z, _, _, _, _ = enc(x, L_f, eigvecs, lap)
        z.sum().backward()
        assert x.grad is not None

    def test_L_f_unbatched_broadcast(self, enc, eigvecs, lap):
        """2-D L_f is automatically broadcast to batch dimension."""
        edge_index, bw = _make_ring(N)
        L_f_2d = DifferentiableLaplacian(
            n_nodes=N, edge_index=edge_index,
            base_weights=bw, normalised=True,
        )(torch.zeros(edge_index.shape[1])).detach()  # (N, N)
        x = torch.randn(B, INP)
        out = enc(x, L_f_2d, eigvecs, lap)
        assert len(out) == 5


# ---------------------------------------------------------------------------
# AC5  WiringEncoder regression
# ---------------------------------------------------------------------------

class TestWiringEncoderRegression:
    """AC5: WiringEncoder unchanged API and behaviour."""

    @pytest.fixture
    def enc_v1(self):
        return WiringEncoder(
            input_dim=INP, latent_dim=Q,
            hidden_dim=32, use_lambda_features=False,
        )

    def test_forward_returns_3_tuple(self, enc_v1):
        x = torch.randn(B, INP)
        out = enc_v1(x)
        assert len(out) == 3

    def test_output_shapes(self, enc_v1):
        x = torch.randn(B, INP)
        z, mu, log_var = enc_v1(x)
        for name, t in [("z", z), ("mu", mu), ("log_var", log_var)]:
            assert t.shape == (B, Q), f"{name} shape mismatch"

    def test_kl_loss_nonneg(self, enc_v1):
        x = torch.randn(B, INP)
        _, mu, lv = enc_v1(x)
        assert float(WiringEncoder.kl_loss(mu, lv)) >= 0.0

    def test_gradient_flows(self, enc_v1):
        x = torch.randn(B, INP, requires_grad=True)
        z, _, _ = enc_v1(x)
        z.sum().backward()
        assert x.grad is not None

    def test_lambda_fp_concatenation(self, enc_v1):
        enc_v1.use_lambda_features = True
        x  = torch.randn(B, INP)
        fp = torch.randn(B, enc_v1.n_lambda_bins)
        # Rebuild with lambda features to get correct in_dim
        enc_lf = WiringEncoder(
            input_dim=INP, latent_dim=Q,
            hidden_dim=32, use_lambda_features=True,
        )
        z, mu, lv = enc_lf(x, lambda_fp=fp)
        assert z.shape == (B, Q)


# ---------------------------------------------------------------------------
# ModeWeightHead standalone
# ---------------------------------------------------------------------------

class TestModeWeightHead:
    def test_output_shapes(self):
        head = ModeWeightHead(hidden_dim=D, q=Q)
        h    = torch.randn(B, D)
        log_a, log_b = head(h)
        assert log_a.shape == (B, Q)
        assert log_b.shape == (B, Q)

    def test_gradient_flows(self):
        head = ModeWeightHead(hidden_dim=D, q=Q)
        h    = torch.randn(B, D, requires_grad=True)
        log_a, log_b = head(h)
        (log_a + log_b).sum().backward()
        assert h.grad is not None
