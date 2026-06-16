"""
Unit tests for vdt/encoder.py -- WiringEncoder, ModeWeightHead.

Acceptance criteria from issue #25
-----------------------------------
AC1  WiringEncoder.forward returns (z, mu, log_var, log_a, log_b)
     all (B, latent_dim) on (B=4, D=32, q=8) input.
AC2  ModeWeightHead outputs finite log_a, log_b on random input.
AC3  kl_isotropic (the only kl_z path in ) is non-negative.
AC4  kl_z scalar is differentiable through mu (gradcheck).
AC5  WiringEncoder still passes its 3-tuple contract.

VDT is stubbed to avoid requiring a live graph fixture.
All gradient paths use float64 for gradcheck precision.
"""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from vdt.encoder import WiringEncoder, ModeWeightHead, kl_isotropic

torch.manual_seed(7)

B      = 4
D_IN   = 32
Q      = 8       # latent_dim
N      = 16      # n_nodes (small for fast test)
FEAT   = 16      # feat_dim


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_encoder(**kwargs) -> WiringEncoder:
    defaults = dict(
        input_dim=D_IN,
        latent_dim=Q,
        n_nodes=N,
        feat_dim=FEAT,
        n_layers=2,
        m_modes=Q,
        n_heads=2,
        use_lambda_features=False,
        use_isotropic_kl=True,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return WiringEncoder(**defaults)


class _StubLap:
    """Minimal DifferentiableLaplacian stub -- only dt_max_cfl is needed."""
    def dt_max_cfl(self, safety: float = 0.5):
        return torch.tensor(0.1)


def _stub_vdt_forward(self, X0, L_f, eigvecs, lap):
    """
    Replaces VDT.forward.  Returns (Q_K, P, S) with the right shapes and
    live autograd graph so backward passes work.
    """
    B_, N_, d = X0.shape
    Q_K = X0 + 0.0   # identity -- keeps gradient alive
    P   = X0
    S   = X0
    return Q_K, P, S


def _stub_modal_projection(self, Q_K, eigvecs):
    """Replaces VDT.modal_projection.  Returns mean over nodes."""
    return Q_K.mean(dim=1)  # (B, feat_dim)


# ---------------------------------------------------------------------------
# AC1 -- forward returns correct 5-tuple shapes
# ---------------------------------------------------------------------------

class TestWiringEncoderForward:

    def _forward(self, encoder, x):
        L_f    = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        eigvecs = torch.eye(N)[:, :Q]        # (N, Q)
        lap     = _StubLap()
        with (
            patch.object(type(encoder.vdt), "forward",      _stub_vdt_forward),
            patch.object(type(encoder.vdt), "modal_projection", _stub_modal_projection),
        ):
            return encoder(x, L_f=L_f, eigvecs=eigvecs, lap=lap)

    def test_returns_5_tuple(self):
        enc = _make_encoder()
        x   = torch.randn(B, D_IN)
        out = self._forward(enc, x)
        assert len(out) == 5, f"Expected 5-tuple, got {len(out)}-tuple"

    def test_output_shapes(self):
        enc = _make_encoder()
        x   = torch.randn(B, D_IN)
        z, mu, log_var, log_a, log_b = self._forward(enc, x)
        assert z.shape       == (B, Q), f"z shape: {z.shape}"
        assert mu.shape      == (B, Q), f"mu shape: {mu.shape}"
        assert log_var.shape == (B, Q), f"log_var shape: {log_var.shape}"
        assert log_a.shape   == (B, Q), f"log_a shape: {log_a.shape}"
        assert log_b.shape   == (B, Q), f"log_b shape: {log_b.shape}"

    def test_output_finite(self):
        enc = _make_encoder()
        x   = torch.randn(B, D_IN)
        for t in self._forward(enc, x):
            assert torch.isfinite(t).all(), f"Non-finite output in {t.shape}"

    def test_log_var_clamped(self):
        """log_var must not exceed 4.0 (hard clamp from encoder)."""
        enc = _make_encoder()
        x   = torch.randn(B, D_IN) * 100
        _, _, log_var, _, _ = self._forward(enc, x)
        assert (log_var <= 4.01).all()


# ---------------------------------------------------------------------------
# AC2 -- ModeWeightHead outputs finite (log_a, log_b)
# ---------------------------------------------------------------------------

class TestModeWeightHead:

    def test_output_shapes(self):
        head  = ModeWeightHead(hidden_dim=64, q=Q)
        h     = torch.randn(B, 64)
        log_a, log_b = head(h)
        assert log_a.shape == (B, Q)
        assert log_b.shape == (B, Q)

    def test_output_finite(self):
        head  = ModeWeightHead(hidden_dim=64, q=Q)
        h     = torch.randn(B, 64)
        log_a, log_b = head(h)
        assert torch.isfinite(log_a).all()
        assert torch.isfinite(log_b).all()

    def test_gradient_flows(self):
        head  = ModeWeightHead(hidden_dim=64, q=Q)
        h     = torch.randn(B, 64, requires_grad=True)
        log_a, log_b = head(h)
        (log_a.sum() + log_b.sum()).backward()
        assert h.grad is not None and torch.isfinite(h.grad).all()


# ---------------------------------------------------------------------------
# AC3 -- kl_isotropic is non-negative
# ---------------------------------------------------------------------------

class TestKlIsotropicNonNegative:

    def test_non_negative_random(self):
        mu      = torch.randn(B, Q)
        log_var = torch.randn(B, Q)
        kl = kl_isotropic(mu, log_var)
        assert kl.item() >= -1e-6, f"kl_isotropic was {kl.item()}"

    def test_zero_at_prior(self):
        """mu=0, log_var=0 (var=1) => KL = 0."""
        mu      = torch.zeros(B, Q)
        log_var = torch.zeros(B, Q)
        kl = kl_isotropic(mu, log_var)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_only_kl_path_in(self):
        """WiringEncoder.kl_loss must call kl_isotropic (checked by value)."""
        enc     = _make_encoder(use_isotropic_kl=True)
        mu      = torch.zeros(B, Q)
        log_var = torch.zeros(B, Q)
        val     = enc.kl_loss(mu, log_var)
        assert val.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# AC4 -- kl_z differentiable through mu via gradcheck
# ---------------------------------------------------------------------------

class TestKlGradCheck:

    def test_gradcheck_mu(self):
        """gradcheck on kl_isotropic w.r.t. mu in float64."""
        mu      = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        log_var = torch.zeros(2, 4, dtype=torch.float64)

        def fn(mu_):
            return kl_isotropic(mu_, log_var)

        assert torch.autograd.gradcheck(fn, (mu,), eps=1e-5, atol=1e-4)

    def test_gradcheck_log_var(self):
        """gradcheck on kl_isotropic w.r.t. log_var in float64."""
        mu      = torch.zeros(2, 4, dtype=torch.float64)
        log_var = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

        def fn(lv):
            return kl_isotropic(mu, lv)

        assert torch.autograd.gradcheck(fn, (log_var,), eps=1e-5, atol=1e-4)

