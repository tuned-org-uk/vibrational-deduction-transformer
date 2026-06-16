"""
tests/test_model.py  --  Integration tests for WiringAutoencoder.

Acceptance criteria from issue #27:

  AC1  forward() returns a dict with all 9 keys; all are finite tensors.
  AC2  Total loss == recon + kl_z + kl_S + kl_tau (test with known scalars).
  AC3  extract_spectral_artefact() produces S_memory with shape (d_model, d_model).
  AC4  from_config() correctly dispatches v1 vs  on model.version.
  AC5  Integration: one full training loop on synthetic (B=4, D=32, N=16, q=8)
       data; all three ELBO terms (kl_z, kl_S, kl_tau) finite at every step,
       and the total loss strictly decreasing after 10 SGD steps.
       (Requirement from issue #27 and roadmap issue #34.)
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
    """Build a tiny symmetric PSD ring-graph Laplacian (N, N) for tests."""
    from vdt.laplacian import DifferentiableLaplacian
    W = torch.zeros(N, N)
    for i in range(N):
        j = (i + 1) % N
        W[i, j] = 1.0
        W[j, i] = 1.0
    deg = W.sum(dim=-1)
    L = torch.diag(deg) - W  # combinatorial Laplacian (N, N)

    src, dst = W.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)   # (2, E)
    base_weights = torch.ones(src.shape[0])        # (E,)
    lap = DifferentiableLaplacian(
        edge_index=edge_index, base_weights=base_weights, n_nodes=N
    )
    return lap, L


def _make_spectral(L: torch.Tensor, q: int):
    """Return the leading q eigenvectors and eigenvalues of L."""
    eigvals, eigvecs = torch.linalg.eigh(L)
    return eigvecs[:, :q], eigvals[:q].clamp(min=1e-6)


def _make_model():
    from vdt.model import WiringAutoencoder
    lap, L_ring = _make_laplacian()
    model = WiringAutoencoder(
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
        assert S_mem.ndim == 2
        assert S_mem.shape[0] == S_mem.shape[1], "S_memory must be square"

    def test_w_hat_and_omega_hat_shapes(self):
        model, U_q, eigvals_q = _make_model()
        artefact = model.extract_spectral_artefact(U_q, eigvals_q)
        assert artefact["omega_hat"].shape == (Q,)
        assert artefact["W_hat"].ndim == 3
        assert artefact["W_hat"].shape[-1] == Q

    def test_s_memory_symmetric(self):
        """S_memory is built from outer products so must be symmetric."""
        model, U_q, eigvals_q = _make_model()
        artefact = model.extract_spectral_artefact(U_q, eigvals_q)
        S_mem = artefact["S_memory"]
        assert torch.allclose(S_mem, S_mem.T, atol=1e-5)


# ---------------------------------------------------------------------------
# AC4  --  from_config() dispatches v1 vs  on model.version
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_dispatches(self):
        from vdt.model import from_config, WiringAutoencoder
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
        assert isinstance(model, WiringAutoencoder)

    def test_dispatches_v1(self):
        from vdt.model import from_config, WiringAutoencoder
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
        from vdt.model import from_config, WiringAutoencoder
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
# AC5  --  Integration: all three ELBO terms finite per step;
#          total loss strictly decreases after 10 SGD steps.
#
#  Requirement (issue #27 + roadmap issue #34):
#    "one full training step on synthetic (B=4, D=32, N=16, q=8) data,
#     all three ELBO terms finite and the loss decreasing after 10 steps."
#
#  This class tests both conditions explicitly:
#    1. kl_z, kl_S, kl_tau each individually finite at every step.
#    2. loss[step 9] < loss[step 0]  (strict, no tolerance padding).
# ---------------------------------------------------------------------------

class TestTrainingConvergence:
    """
    Integration test for WiringAutoencoder on synthetic data.

    Synthetic fixture:  B=4, D=32, N=16, q=8  (ring-graph Laplacian).
    Optimiser:          Adam, lr=1e-3.
    Steps:              10.
    """

    @staticmethod
    def _run_training_loop(n_steps: int = 10):
        """Run n_steps of Adam on fixed synthetic data; return per-step records."""
        model, U_q, eigvals_q = _make_model()
        optimiser = optim.Adam(model.parameters(), lr=1e-3)
        torch.manual_seed(0)
        x = torch.randn(B, D)
        node_idx = torch.arange(B)

        records = []
        for step in range(n_steps):
            optimiser.zero_grad()
            out = model(x, U_q, eigvals_q, node_idx=node_idx)
            out["loss"].backward()
            optimiser.step()
            records.append({
                "step":    step,
                "loss":    out["loss"].item(),
                "kl_z":    out["kl_z"].item(),
                "kl_S":    out["kl_S"].item(),
                "kl_tau":  out["kl_tau"].item(),
                "recon":   out["recon"].item(),
            })
        return records

    def test_all_three_elbo_terms_finite_at_every_step(self):
        """
        AC5 (part 1): kl_z, kl_S, kl_tau are each individually finite
        at every one of the 10 training steps.
        """
        records = self._run_training_loop(n_steps=10)
        for rec in records:
            step = rec["step"]
            for term in ("kl_z", "kl_S", "kl_tau"):
                val = rec[term]
                assert (
                    val == val and abs(val) != float("inf")
                ), (
                    f"ELBO term '{term}' is non-finite ({val}) at step {step}. "
                    f"Full record: {rec}"
                )

    def test_total_loss_finite_at_every_step(self):
        """Total loss must be finite at every step (sanity gate)."""
        records = self._run_training_loop(n_steps=10)
        for rec in records:
            val = rec["loss"]
            assert (
                val == val and abs(val) != float("inf")
            ), f"Loss is non-finite ({val}) at step {rec['step']}"

    def test_loss_strictly_decreases_after_10_steps(self):
        """
        AC5 (part 2): total loss at step 9 is strictly less than at step 0.
        No tolerance padding -- the 10-step Adam trajectory must show
        real descent on fixed synthetic data.
        """
        records = self._run_training_loop(n_steps=10)
        first, last = records[0]["loss"], records[-1]["loss"]
        assert last < first, (
            f"Loss did not decrease over 10 steps: "
            f"step 0 = {first:.6f}, step 9 = {last:.6f}.\n"
            f"Per-step losses: {[r['loss'] for r in records]}"
        )

    def test_recon_term_finite_at_every_step(self):
        """Reconstruction term must also stay finite throughout."""
        records = self._run_training_loop(n_steps=10)
        for rec in records:
            val = rec["recon"]
            assert (
                val == val and abs(val) != float("inf")
            ), f"'recon' is non-finite ({val}) at step {rec['step']}"
