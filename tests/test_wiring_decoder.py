"""
Tests for SpectralLoadingDecoder -- with focus on the log_var_S
independence fix (issue #52).

Test inventory
--------------

test_log_var_S_shape
    Verifies that log_var_S has shape (B, q, q) and that all values are
    within the [-6, 4] clamp range for a batch of random z inputs.

test_log_var_S_independent_of_S
    Core regression test for issue #52.  Checks that log_var_S is NOT
    a near-deterministic function of S.pow(2) by verifying that the
    Pearson correlation between log_var_S.flatten() and
    (S.pow(2) + 1e-6).log().flatten() is below 0.9 across 20 random
    initialisations.  If the old proxy were re-introduced the correlation
    would be >= 0.99.

test_log_var_S_gradient
    Verifies that gradients flow back through log_var_S to z
    independently of the S gradient path.  Specifically, zeroing the
    S_net weights (so dL/dS = 0 for the S path) must still leave a
    non-zero gradient on z through log_var_S_head.

test_forward_returns_five_outputs
    Smoke test: forward() returns exactly 5 tensors with the expected
    shapes (W, omega, S, L_z, log_var_S).

test_near_identity_init
    Verifies the near-identity initialisation contract:
    S_net(z=0) == eye(q).flatten() and log_var_S(z=0) ~ 0 for all entries.

test_wiring_decoder_v1_unchanged
    Smoke test for WiringDecoder (v1) to confirm it is unaffected by the
    SpectralLoadingDecoder changes.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vdt.wiring_decoder import SpectralLoadingDecoder, WiringDecoder
from vdt.laplacian import DifferentiableLaplacian


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 12   # graph nodes
q = 4    # spectral modes
B = 6    # batch size


@pytest.fixture()
def decoder():
    """Fresh SpectralLoadingDecoder at default random init."""
    return SpectralLoadingDecoder(q=q, d=N)


@pytest.fixture()
def dummy_inputs():
    """Random (z, U_q, L_base) compatible with (N=12, q=4)."""
    torch.manual_seed(0)
    z = torch.randn(B, q)
    # U_q: orthonormal columns (d=N, q)
    U_raw = torch.randn(N, q)
    U_q, _ = torch.linalg.qr(U_raw)
    U_q = U_q[:, :q]   # (N, q)
    # L_base: random symmetric positive semi-definite Laplacian stand-in
    A = torch.rand(N, N)
    A = (A + A.t()) / 2.0
    A.fill_diagonal_(0.0)
    D = A.sum(dim=1)
    L_base = torch.diag(D) - A   # combinatorial Laplacian
    return z, U_q, L_base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLogVarSIndependence:
    """Regression tests for issue #52: log_var_S must not be a proxy of S^2."""

    def test_log_var_S_shape(self, decoder, dummy_inputs):
        """
        log_var_S must have shape (B, q, q) and lie within the [-6, 4] clamp.
        """
        z, U_q, L_base = dummy_inputs
        _, _, _, _, log_var_S = decoder(z, U_q, L_base)
        assert log_var_S.shape == (B, q, q), (
            f"Expected log_var_S shape ({B}, {q}, {q}), got {log_var_S.shape}"
        )
        assert log_var_S.min().item() >= -6.0 - 1e-5, (
            f"log_var_S min {log_var_S.min().item():.4f} below clamp lower bound -6"
        )
        assert log_var_S.max().item() <= 4.0 + 1e-5, (
            f"log_var_S max {log_var_S.max().item():.4f} above clamp upper bound 4"
        )

    def test_log_var_S_independent_of_S(self):
        """
        Core regression for issue #52.

        log_var_S must NOT be a near-deterministic function of S^2.  We
        verify this by checking that the Pearson correlation between
        log_var_S.flatten() and (S^2 + 1e-6).log().flatten() is below 0.9
        for a freshly initialised decoder across 20 random z inputs.

        If the old proxy ``log_var_S = (S.pow(2) + 1e-6).log()`` were
        re-introduced into forward(), the correlation would be 1.0 by
        construction and this test would fail.
        """
        torch.manual_seed(42)
        correlations = []
        for seed in range(20):
            # New random init each iteration to sample across weight space
            dec = SpectralLoadingDecoder(q=q, d=N)
            z = torch.randn(B, q)
            U_q = torch.eye(N)[:, :q]   # simple orthonormal basis
            L_base = torch.zeros(N, N)  # flat Laplacian (shape only)

            with torch.no_grad():
                _, _, S, _, log_var_S = dec(z, U_q, L_base)

            proxy = (S.pow(2) + 1e-6).log().flatten()
            actual = log_var_S.flatten()

            # Pearson correlation: (x - mean_x) @ (y - mean_y) / (||...|| * ||...||)
            px = proxy - proxy.mean()
            ay = actual - actual.mean()
            denom = px.norm() * ay.norm()
            if denom.item() < 1e-8:
                # Both constant -- correlation undefined; treat as 0.
                corr = 0.0
            else:
                corr = (px @ ay / denom).item()
            correlations.append(abs(corr))

        max_corr = max(correlations)
        assert max_corr < 0.9, (
            f"log_var_S appears to be derived from S^2: max |correlation| = "
            f"{max_corr:.4f} >= 0.9.  This indicates the proxy may have been "
            f"re-introduced (issue #52)."
        )

    def test_log_var_S_gradient(self, decoder, dummy_inputs):
        """
        Gradients must flow to z through log_var_S_head independently
        of the S_net path.

        Strategy: zero out S_net weights so d(loss)/d(S path) = 0, then
        compute a loss on log_var_S only and verify grad on z is non-zero.
        """
        z, U_q, L_base = dummy_inputs
        z = z.detach().requires_grad_(True)

        # Freeze S_net: zero weight and bias so S_net contributes nothing
        with torch.no_grad():
            nn.init.zeros_(decoder.S_net.weight)
            nn.init.zeros_(decoder.S_net.bias)

        # Also freeze omega_net
        with torch.no_grad():
            nn.init.zeros_(decoder.omega_net.weight)
            nn.init.zeros_(decoder.omega_net.bias)

        # Forward: only log_var_S_head has non-zero weights now
        _, _, _, _, log_var_S = decoder(z, U_q, L_base)
        loss = log_var_S.sum()
        loss.backward()

        assert z.grad is not None, "No gradient on z after log_var_S.sum().backward()"
        assert z.grad.abs().sum().item() > 0.0, (
            "z.grad is all-zero: log_var_S_head is not connected to z"
        )


class TestForwardContract:
    """Shape and initialisation contracts for SpectralLoadingDecoder."""

    def test_forward_returns_five_outputs(self, decoder, dummy_inputs):
        """
        forward() must return exactly 5 tensors: W, omega, S, L_z, log_var_S.
        """
        z, U_q, L_base = dummy_inputs
        outputs = decoder(z, U_q, L_base)
        assert len(outputs) == 5, (
            f"Expected 5 outputs from SpectralLoadingDecoder.forward(), got {len(outputs)}"
        )
        W, omega, S, L_z, log_var_S = outputs
        assert W.shape == (B, N, q),          f"W shape mismatch: {W.shape}"
        assert omega.shape == (B, q),          f"omega shape mismatch: {omega.shape}"
        assert S.shape == (B, q, q),           f"S shape mismatch: {S.shape}"
        assert L_z.shape == (B, N, N),         f"L_z shape mismatch: {L_z.shape}"
        assert log_var_S.shape == (B, q, q),   f"log_var_S shape mismatch: {log_var_S.shape}"

    def test_near_identity_init(self):
        """
        At z=0, S_net should output eye(q) and log_var_S should be ~0.

        This tests the _init_weights contract:
          S_net.weight = 0, S_net.bias = eye(q).flatten()  =>  S_net(0) = eye_flat
          log_var_S_head.weight ~ N(0, 0.01), bias = 0     =>  output ~ 0 at z=0
        """
        dec = SpectralLoadingDecoder(q=q, d=N)
        z_zero = torch.zeros(1, q)
        U_q = torch.eye(N)[:, :q]
        L_base = torch.zeros(N, N)

        with torch.no_grad():
            _, _, S, _, log_var_S = dec(z_zero, U_q, L_base)

        # S should be eye(q)
        expected_S = torch.eye(q).unsqueeze(0)   # (1, q, q)
        assert torch.allclose(S, expected_S, atol=1e-5), (
            f"S at z=0 is not eye(q): max deviation {(S - expected_S).abs().max():.6f}"
        )

        # log_var_S should be close to zero (weight init ~ N(0, 0.01), bias=0)
        # Tolerance is generous: small weights can still give ~0.01*q output
        assert log_var_S.abs().max().item() < 0.1, (
            f"log_var_S at z=0 too large: max abs = {log_var_S.abs().max().item():.4f}. "
            f"Expected near-zero from zero-bias init."
        )


class TestV1Unchanged:
    """Smoke tests confirming WiringDecoder (v1) is unaffected by the fix."""

    def test_wiring_decoder_v1_forward(self):
        """
        WiringDecoder.forward() must still return (L, delta) with no changes.

        n_edges is derived from lap.edge_index.shape[1] after construction
        so the test remains valid for any knn_k value (knn_k=3 on 8 nodes
        gives 8*3=24 directed edges, not 16).
        """
        torch.manual_seed(7)
        n_nodes = 8
        latent_dim = 8
        hidden_dim = 16
        n_heads = 2

        # Build a minimal DifferentiableLaplacian
        E_small = torch.randn(n_nodes, 4)
        lap = DifferentiableLaplacian.from_embeddings(
            E_small, knn_k=3, sigma=1.0, normalised=True, sparse=False
        )
        # Derive n_edges from the actual graph topology so this test
        # remains correct for any knn_k or graph size.
        n_edges = lap.edge_index.shape[1]

        dec = WiringDecoder(
            latent_dim=latent_dim,
            n_edges=n_edges,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            laplacian=lap,
        )
        z = torch.randn(B, latent_dim)
        outputs = dec(z)
        assert len(outputs) == 2, f"WiringDecoder must return 2 outputs, got {len(outputs)}"
        L, delta = outputs
        assert delta.shape == (B, n_edges), f"delta shape mismatch: {delta.shape}"
