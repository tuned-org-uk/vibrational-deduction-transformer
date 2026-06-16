"""
Unit tests for the Wiring Autoencoder core modules.

Run with:  pytest tests/ -v
"""
from __future__ import annotations
import torch
import pytest

from vdt.laplacian import DifferentiableLaplacian
from vdt.spectral import TauModeDiffusion, spectral_freq_cost, lambda_fingerprint
from vdt.encoder import WiringEncoder
from vdt.wiring_decoder import WiringDecoder
from vdt.diffusion_decoder import DiffusionDecoder
from vdt.model import WiringAutoencoder


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
def vdt_model(small_graph):
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
# DiffusionDecoder — explicit shape contract tests
# ---------------------------------------------------------------------------
class TestDiffusionDecoder:
    def test_per_node_shape(self, small_graph):
        """With node_idx, output must be (B, D)."""
        lap, E, N, D = small_graph
        B = 4
        delta = torch.randn(B, lap.base_weights.shape[0])
        L = lap(delta)
        node_idx = torch.randint(0, N, (B,))
        dec = DiffusionDecoder(embedding_dim=D, hidden_dim=32, tau_modes=4)
        out = dec(L, E, node_idx=node_idx)
        assert out.shape == (B, D), f"Expected ({B}, {D}), got {tuple(out.shape)}"

    def test_full_graph_shape(self, small_graph):
        """Without node_idx, output must be (B, N, D)."""
        lap, E, N, D = small_graph
        B = 3
        delta = torch.randn(B, lap.base_weights.shape[0])
        L = lap(delta)
        dec = DiffusionDecoder(embedding_dim=D, hidden_dim=32, tau_modes=4)
        out = dec(L, E, node_idx=None)
        assert out.shape == (B, N, D), f"Expected ({B}, {N}, {D}), got {tuple(out.shape)}"

    def test_recon_loss_guards_full_graph(self, small_graph):
        """recon_loss must raise if x_hat is not (B, D)."""
        lap, E, N, D = small_graph
        B = 2
        delta = torch.randn(B, lap.base_weights.shape[0])
        L = lap(delta)
        dec = DiffusionDecoder(embedding_dim=D, hidden_dim=32, tau_modes=4)
        x_hat_full = dec(L, E, node_idx=None)       # (B, N, D)
        x = torch.randn(B, D)
        with pytest.raises(ValueError, match="per-node"):
            dec.recon_loss(x, x_hat_full)


# ---------------------------------------------------------------------------
# WiringAutoencoder end-to-end
# ---------------------------------------------------------------------------
class TestWiringAutoencoder:
    def test_forward_shapes(self, vdt_model, small_graph):
        model, E = vdt_model
        lap, _, N, D = small_graph
        B = 6
        x        = torch.randn(B, D)
        node_idx = torch.randint(0, N, (B,))
        out = model(x, E, node_idx=node_idx)
        assert out["x_hat"].shape  == (B, D)
        assert out["z"].shape       == (B, 8)
        assert out["L"].shape       == (B, N, N)

    def test_loss_keys(self, vdt_model, small_graph):
        model, E = vdt_model
        lap, _, N, D = small_graph
        x = torch.randn(4, D)
        node_idx = torch.randint(0, N, (4,))
        out = model(x, E, node_idx=node_idx)
        for key in ("loss", "recon_loss", "kl_loss", "freq_loss"):
            assert key in out, f"Missing key: {key}"

    def test_backward(self, vdt_model, small_graph):
        model, E = vdt_model
        lap, _, N, D = small_graph
        x = torch.randn(4, D)
        node_idx = torch.randint(0, N, (4,))
        out = model(x, E, node_idx=node_idx)
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    # ------------------------------------------------------------------
    # generate() — the core of issue #3
    # ------------------------------------------------------------------
    def test_generate_per_node_default(self, vdt_model, small_graph):
        """
        Default generate() call: mode='per_node', no node_idx supplied.
        x_hat must be (n_samples, D) — same shape as reconstruction path.
        node_idx key must be present and have shape (n_samples,).
        """
        model, E = vdt_model
        _, _, N, D = small_graph
        n = 5
        gen = model.generate(E, n_samples=n)
        assert gen["z"].shape        == (n, 8),    f"z shape wrong: {gen['z'].shape}"
        assert gen["L"].shape        == (n, N, N), f"L shape wrong: {gen['L'].shape}"
        assert gen["node_idx"].shape == (n,),      f"node_idx shape wrong: {gen['node_idx'].shape}"
        assert gen["x_hat"].shape    == (n, D),    f"x_hat shape wrong: {gen['x_hat'].shape}"

    def test_generate_per_node_explicit_idx(self, vdt_model, small_graph):
        """
        generate() with explicit node_idx in per_node mode.
        Output x_hat must match the supplied node indices' geometry.
        """
        model, E = vdt_model
        _, _, N, D = small_graph
        n = 4
        node_idx = torch.arange(n)   # nodes 0,1,2,3
        gen = model.generate(E, n_samples=n, node_idx=node_idx, mode="per_node")
        assert gen["x_hat"].shape == (n, D)
        assert torch.equal(gen["node_idx"], node_idx)

    def test_generate_full_graph(self, vdt_model, small_graph):
        """
        generate() in full_graph mode.
        x_hat must be (n_samples, N, D).
        node_idx must be None.
        """
        model, E = vdt_model
        _, _, N, D = small_graph
        n = 3
        gen = model.generate(E, n_samples=n, mode="full_graph")
        assert gen["x_hat"].shape == (n, N, D), f"Expected ({n},{N},{D}), got {tuple(gen['x_hat'].shape)}"
        assert gen["node_idx"] is None

    def test_generate_invalid_mode(self, vdt_model, small_graph):
        """Passing an unknown mode must raise ValueError."""
        model, E = vdt_model
        with pytest.raises(ValueError, match="mode must be"):
            model.generate(E, n_samples=2, mode="bad_mode")

    def test_generate_node_idx_shape_mismatch(self, vdt_model, small_graph):
        """Wrong node_idx length must raise ValueError."""
        model, E = vdt_model
        with pytest.raises(ValueError, match="node_idx must have shape"):
            model.generate(E, n_samples=4, node_idx=torch.arange(3))

    def test_generate_sanity_encode_decode_round_trip(self, vdt_model, small_graph):
        """
        Encode a real node, then generate from that z in per_node mode.
        The generated embedding for the same node_idx should be finite
        and have the correct dtype.
        """
        model, E = vdt_model
        _, _, N, D = small_graph
        model.eval()
        # pick a real node
        node_idx = torch.tensor([0])
        x = E[node_idx]         # (1, D)  ground-truth embedding
        # forward pass to get z
        with torch.no_grad():
            out = model(x, E, node_idx=node_idx)
        z_real = out["z"]       # (1, latent)
        # generate from same z via per-node mode
        with torch.no_grad():
            L, _ = model.wiring_decoder(z_real)
            x_hat = model.diffusion_decoder(L, E, node_idx=node_idx)
        assert x_hat.shape == (1, D)
        assert torch.isfinite(x_hat).all(), "Generated embedding contains NaN/Inf"
        assert x_hat.dtype == torch.float32
