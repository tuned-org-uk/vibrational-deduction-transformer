"""
Unit tests for the Wiring Autoencoder core modules.

Run with:  pytest tests/ -v
"""
from __future__ import annotations
import torch
import pytest

from wae.laplacian import DifferentiableLaplacian
from wae.spectral import TauModeDiffusion, spectral_freq_cost, lambda_fingerprint
from wae.encoder import WiringEncoder
from wae.wiring_decoder import WiringDecoder
from wae.diffusion_decoder import DiffusionDecoder
from wae.model import WiringAutoencoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_graph():
    """Tiny 20-node, 3-NN graph for fast unit tests."""
    N, D, k = 20, 16, 3
    E = torch.randn(N, D)
    lap = DifferentiableLaplacian.from_embeddings(E, knn_k=k, sigma=1.0)
    return lap, E, N, D


@pytest.fixture
def wae_model(small_graph):
    lap, E, N, D = small_graph
    model = WiringAutoencoder(
        input_dim=D, latent_dim=8, hidden_dim=32,
        n_wiring_heads=2, tau_modes=4,
        beta=1.0, alpha=0.1, laplacian=lap,
    )
    return model, E


# ---------------------------------------------------------------------------
# DifferentiableLaplacian
# ---------------------------------------------------------------------------
class TestDifferentiableLaplacian:
    def test_symmetric(self, small_graph):
        lap, E, N, _ = small_graph
        E_val = torch.zeros(lap.base_weights.shape[0])  # zero delta -> base wiring
        L = lap(E_val)
        assert L.shape == (N, N)
        assert torch.allclose(L, L.T, atol=1e-5), "Laplacian must be symmetric"

    def test_positive_semidefinite(self, small_graph):
        lap, _, _, _ = small_graph
        delta = torch.zeros(lap.base_weights.shape[0])
        L = lap(delta)
        eigvals = torch.linalg.eigvalsh(L)
        assert eigvals.min() > -1e-4, "Normalised Laplacian must be PSD"

    def test_batched(self, small_graph):
        lap, _, N, _ = small_graph
        B = 4
        delta = torch.randn(B, lap.base_weights.shape[0])
        L = lap(delta)
        assert L.shape == (B, N, N)

    def test_gradient_flows(self, small_graph):
        lap, _, _, _ = small_graph
        delta = torch.randn(lap.base_weights.shape[0], requires_grad=True)
        L = lap(delta)
        loss = L.sum()
        loss.backward()
        assert delta.grad is not None


# ---------------------------------------------------------------------------
# TauModeDiffusion
# ---------------------------------------------------------------------------
class TestTauModeDiffusion:
    def test_output_shape_all_nodes(self, small_graph):
        lap, E, N, D = small_graph
        delta = torch.zeros(lap.base_weights.shape[0])
        L = lap(delta).unsqueeze(0)   # (1, N, N)
        diff = TauModeDiffusion(tau_modes=4, learnable_time=False)
        out = diff(L, E)
        assert out.shape == (1, N, D)

    def test_output_shape_node_idx(self, small_graph):
        lap, E, N, D = small_graph
        B = 5
        delta = torch.randn(B, lap.base_weights.shape[0])
        L = lap(delta)
        node_idx = torch.randint(0, N, (B,))
        diff = TauModeDiffusion(tau_modes=4, learnable_time=False)
        out = diff(L, E, node_idx=node_idx)
        assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# spectral_freq_cost
# ---------------------------------------------------------------------------
def test_freq_cost_positive(small_graph):
    lap, _, N, _ = small_graph
    delta = torch.randn(2, lap.base_weights.shape[0])
    L = lap(delta)
    cost = spectral_freq_cost(L, tau_modes=4)
    assert cost >= 0, "J_freq must be non-negative"


# ---------------------------------------------------------------------------
# WiringEncoder
# ---------------------------------------------------------------------------
class TestWiringEncoder:
    def test_output_shapes(self):
        enc = WiringEncoder(input_dim=32, latent_dim=8, hidden_dim=64, use_lambda_features=False)
        x = torch.randn(16, 32)
        z, mu, log_var = enc(x)
        assert z.shape == mu.shape == log_var.shape == (16, 8)

    def test_kl_positive(self):
        enc = WiringEncoder(input_dim=32, latent_dim=8, hidden_dim=64, use_lambda_features=False)
        x = torch.randn(16, 32)
        _, mu, log_var = enc(x)
        kl = WiringEncoder.kl_loss(mu, log_var)
        assert kl >= 0


# ---------------------------------------------------------------------------
# WiringAutoencoder end-to-end
# ---------------------------------------------------------------------------
class TestWiringAutoencoder:
    def test_forward_shapes(self, wae_model, small_graph):
        model, E = wae_model
        lap, _, N, D = small_graph
        B = 6
        x        = torch.randn(B, D)
        node_idx = torch.randint(0, N, (B,))
        out = model(x, E, node_idx=node_idx)
        assert out["x_hat"].shape  == (B, D)
        assert out["z"].shape       == (B, 8)
        assert out["L"].shape       == (B, N, N)

    def test_loss_keys(self, wae_model, small_graph):
        model, E = wae_model
        lap, _, N, D = small_graph
        x = torch.randn(4, D)
        node_idx = torch.randint(0, N, (4,))
        out = model(x, E, node_idx=node_idx)
        for key in ("loss", "recon_loss", "kl_loss", "freq_loss"):
            assert key in out, f"Missing key: {key}"

    def test_backward(self, wae_model, small_graph):
        model, E = wae_model
        lap, _, N, D = small_graph
        x = torch.randn(4, D)
        node_idx = torch.randint(0, N, (4,))
        out = model(x, E, node_idx=node_idx)
        out["loss"].backward()
        # At least some parameters should have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_generate(self, wae_model, small_graph):
        model, E = wae_model
        lap, _, N, D = small_graph
        gen = model.generate(E, n_samples=3)
        assert gen["z"].shape    == (3, 8)
        assert gen["L"].shape    == (3, N, N)
        assert gen["x_hat"].shape == (3, D)
