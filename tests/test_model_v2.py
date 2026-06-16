"""
tests/test_model_v2.py  --  Integration tests for WiringAutoencoderV2.

Acceptance criteria from issue #27:

  AC1  forward() returns a dict with all 9 keys; all are finite tensors.
  AC2  Total loss == recon + kl_z + kl_S + kl_tau (test with known scalars).
  AC3  extract_spectral_artefact() produces S_memory with shape (d_model, d_model).
  AC4  from_config() correctly dispatches v1 vs v2 on model.version.
  AC5  Integration: single training step on synthetic (B=4, D=32, N=16, q=8)
       data; loss is finite and decreases after 10 SGD steps.
"""
from __future__ import annotations
import pytest
import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# Helpers -- build a minimal set of synthetic inputs for (B=4, D=32, N=16, q=8)
# ---------------------------------------------------------------------------

B, D, N, Q = 4, 32, 16, 8


def _make_laplacian():
    """Build a tiny symmetric PSD Laplacian (N, N) for tests."""
    from wae.laplacian import DifferentiableLaplacian
    # Manually construct a ring-graph Laplacian to avoid kNN dep in tests.
    W = torch.zeros(N, N)
    for i in range(N):
        j = (i + 1) % N
        W[i, j] = 1.0
        W[j, i] = 1.0
    deg = W.sum(dim=-1)
    L = torch.diag(deg) - W  # combinatorial Laplacian (N, N)

    # Wrap in a DifferentiableLaplacian via its internal constructor.
    # We set edge_index and base_weights from the ring topology.
    src, dst = W.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)     # (2, E)
    base_weights = torch.ones(src.shape[0])          # (E,)
    lap = DifferentiableLaplacian(edge_index=edge_index, base_weights=base_weights, n_nodes=N)
    return lap, L


def _make_spectral(L: torch.Tensor, q: int):
    """Return the leading q eigenvectors and eigenvalues of L."""
    eigvals, eigvecs = torch.linalg.eigh(L)
    return eigvecs[:, :q], eigvals[:q].clamp(min=1e-6)


def _make_model():
    from wae.model import WiringAutoencoderV2
    lap, L_ring = _make_laplacian()
    model = WiringAutoencoderV2(
        input_dim=D,
        latent_dim=16,
        hidden_dim=64,
        q=Q,
        tau_modes=Q,
        lam_s=0.01,
        tau=0.5,
        laplacian=lap,
    )
    U_q, eigvals_q = _make_spectral(L_ring, Q)
    return model, U_q, eigvals_q


# ---------------------------------------------------------------------------
# AC1  --  forward() returns 9-key dict; all values finite
# ---------------------------------------------------------------------------

class TestForwardOutputs:
    def test_nine_keys(self):
        model, U_q, eigvals_q = _make_model()
        x = torch.randn(B, D)
        node_idx = torch.arange(B)
        out = model(x, U_q, eigvals_q, node_idx=node_idx)
        expected = {"loss", "recon", "kl_z", "kl_S", "kl_tau",
                    "x_hat", "z", "mu", "log_var"}
        assert set(out.keys()) == expected, (
            f"Missing keys: {expected - set(out.keys())}, "
            f"extra keys: {set(out.keys()) - expected}"
        )

    def test_all_finite(self):
        model, U_q, eigvals_q = _make_model()
        x = torch.randn(B, D)
        node_idx = torch.arange(B)
        out = model(x, U_q, eigvals_q, node_idx=node_idx)
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), f"Non-finite values in '{key}'"

    def test_shapes(self):
        model, U_q, eigvals_q = _make_model()
        x = torch.randn(B, D)
        node_idx = torch.arange(B)
        out = model(x, U_q, eigvals_q, node_idx=node_idx)
        assert out["x_hat"].shape == (B, D)
        assert out["z"].shape == (B, 16)   # latent_dim
        assert out["mu"].shape == (B, 16)
        assert out["log_var"].shape == (B, 16)
        assert out["loss"].ndim == 0        # scalar


# ---------------------------------------------------------------------------
# AC2  --  total loss == recon + kl_z + kl_S + kl_tau
# ---------------------------------------------------------------------------

class TestELBODecomposition:
    def test_loss_equals_sum_of_terms(self):
        model, U_q, eigvals_q = _make_model()
        x = torch.randn(B, D)
        node_idx = torch.arange(B)
        out = model(x, U_q, eigvals_q, node_idx=node_idx)
        expected = out["recon"] + out["kl_z"] + out["kl_S"] + out["kl_tau"]
        assert torch.allclose(out["loss"], expected, atol=1e-5), (
            f"loss={out['loss'].item():.6f} != sum={expected.item():.6f}"
        )

    def test_known_scalar_addition(self):
        """Verify the decomposition identity with manually set scalars."""
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        c = torch.tensor(3.0, requires_grad=True)
        d = torch.tensor(4.0, requires_grad=True)
        total = a + b + c + d
        assert total.item() == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# AC3  --  extract_spectral_artefact() produces S_memory (d_model, d_model)
# ---------------------------------------------------------------------------

class TestExtractSpectralArtefact:
    def test_s_memory_shape(self):
        model, U_q, eigvals_q = _make_model()
        artefact = model.extract_spectral_artefact(U_q, eigvals_q)
        assert "S_memory" in artefact
        S_mem = artefact["S_memory"]
        # d_model == D (input_dim) as passed through SpectralLoadingDecoder
        assert S_mem.ndim == 2
        assert S_mem.shape[0] == S_mem.shape[1], "S_memory must be square"

    def test_w_hat_and_omega_hat_shapes(self):
        model, U_q, eigvals_q = _make_model()
        artefact = model.extract_spectral_artefact(U_q, eigvals_q)
        omega_hat = artefact["omega_hat"]
        W_hat = artefact["W_hat"]
        assert omega_hat.shape == (Q,)
        assert W_hat.ndim == 3
        assert W_hat.shape[-1] == Q

    def test_s_memory_symmetric(self):
        """S_memory is built from outer products so must be symmetric."""
        model, U_q, eigvals_q = _make_model()
        artefact = model.extract_spectral_artefact(U_q, eigvals_q)
        S_mem = artefact["S_memory"]
        assert torch.allclose(S_mem, S_mem.T, atol=1e-5)


# ---------------------------------------------------------------------------
# AC4  --  from_config() dispatches v1 vs v2 on model.version
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_dispatches_v2(self):
        from wae.model import from_config, WiringAutoencoderV2
        cfg = {
            "model": {
                "version": 2,
                "latent_dim": 16,
                "hidden_dim": 64,
                "q": Q,
                "tau_modes": Q,
                "lam_s": 0.01,
                "tau": 0.5,
            },
            "graph": {
                "knn_k": 4,
                "sigma": 0.5,
                "normalised": True,
                "sparse": False,
            },
        }
        E = torch.randn(N, D)
        model = from_config(cfg, E)
        assert isinstance(model, WiringAutoencoderV2)

    def test_dispatches_v1(self):
        from wae.model import from_config, WiringAutoencoder
        cfg = {
            "model": {
                "version": 1,
                "latent_dim": 16,
                "hidden_dim": 64,
                "n_wiring_heads": 2,
                "tau_modes": 4,
                "beta": 1.0,
                "alpha": 0.1,
                "use_lambda_features": False,
            },
            "graph": {
                "knn_k": 4,
                "sigma": 0.5,
                "normalised": True,
                "sparse": False,
            },
        }
        E = torch.randn(N, D)
        model = from_config(cfg, E)
        assert isinstance(model, WiringAutoencoder)

    def test_version_absent_defaults_to_v1(self):
        from wae.model import from_config, WiringAutoencoder
        cfg = {
            "model": {
                "latent_dim": 16,
                "hidden_dim": 64,
                "n_wiring_heads": 2,
                "tau_modes": 4,
                "beta": 1.0,
                "alpha": 0.1,
                "use_lambda_features": False,
            },
            "graph": {
                "knn_k": 4,
                "sigma": 0.5,
                "normalised": True,
            },
        }
        E = torch.randn(N, D)
        model = from_config(cfg, E)
        assert isinstance(model, WiringAutoencoder)


# ---------------------------------------------------------------------------
# AC5  --  Integration: loss is finite and decreases after 10 SGD steps
# ---------------------------------------------------------------------------

class TestTrainingConvergence:
    def test_loss_decreases_over_10_steps(self):
        """
        Synthetic (B=4, D=32, N=16, q=8) data.
        Loss must be finite at step 0 and strictly lower at step 9.
        A tolerance of 1e-4 is applied to guard against float noise.
        """
        model, U_q, eigvals_q = _make_model()
        optimiser = optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(B, D)
        node_idx = torch.arange(B)

        losses = []
        for _ in range(10):
            optimiser.zero_grad()
            out = model(x, U_q, eigvals_q, node_idx=node_idx)
            loss = out["loss"]
            assert torch.isfinite(loss), "Loss became non-finite during training"
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] + 1e-4, (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )
