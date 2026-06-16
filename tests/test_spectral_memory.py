"""
tests/test_spectral_memory.py  --  Tests for SpectralAssociativeMemory (issue #28)

Acceptance criteria from issue #28:

  AC1  forward() output shape matches query shape.
  AC2  Retrieval SNR test: with orthonormal keys, retrieve a stored pattern
       with cosine similarity > 0.95.
  AC3  delta_update increases the memory's response to the written (key, value)
       pair without degrading responses to previously stored patterns
       (measure before/after cosine similarity on a toy 4-pattern store).
  AC4  from_vdt() classmethod constructs the object from a trained
       WiringAutoencoder end-to-end.
  AC5  Module is importable and serialisable via torch.save / torch.load.
"""
from __future__ import annotations
import io
import pytest
import torch
import torch.nn.functional as F

from vdt.spectral_memory import SpectralAssociativeMemory

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_MODEL = 64    # memory dimensionality for most tests
Q       = 8     # number of stored spectral keys
B       = 4     # batch size
N       = 16    # graph nodes (for from_vdt fixture)
D       = 32    # input feature dim (for from_vdt fixture)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _orthonormal_keys(q: int, d: int) -> torch.Tensor:
    """Return a (q, d) matrix of orthonormal rows via QR decomposition."""
    A = torch.randn(d, q)
    Q_mat, _ = torch.linalg.qr(A)
    return Q_mat.T  # (q, d)


def _outer_product_memory(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Build an outer-product Hopfield memory: S = sum_k v_k (x) k_k^T.
    keys:   (q, d)
    values: (q, d)
    returns: (d, d)
    """
    return torch.einsum("kd,ke->de", values, keys)  # (d, d)


def _make_vdt_v2_and_spectral():
    """Minimal WiringAutoencoder with ring-graph Laplacian for from_vdt tests."""
    from vdt.laplacian import DifferentiableLaplacian
    from vdt.model import WiringAutoencoder

    W = torch.zeros(N, N)
    for i in range(N):
        j = (i + 1) % N
        W[i, j] = 1.0
        W[j, i] = 1.0
    deg = W.sum(dim=-1)
    L   = torch.diag(deg) - W
    src, dst = W.nonzero(as_tuple=True)
    lap = DifferentiableLaplacian(
        edge_index=torch.stack([src, dst], dim=0),
        base_weights=torch.ones(src.shape[0]),
        n_nodes=N,
    )
    model = WiringAutoencoder(
        input_dim=D, latent_dim=16, hidden_dim=64,
        q=Q, tau_modes=Q, lam_s=0.01, tau=0.5, laplacian=lap,
    )
    eigvals, eigvecs = torch.linalg.eigh(L)
    U_q       = eigvecs[:, :Q]
    eigvals_q = eigvals[:Q].clamp(min=1e-6)
    return model, U_q, eigvals_q


# ---------------------------------------------------------------------------
# AC1  --  forward() output shape matches query shape
# ---------------------------------------------------------------------------

class TestForwardShape:
    def test_output_shape_matches_query(self):
        S = torch.eye(D_MODEL)
        mem = SpectralAssociativeMemory(S, d_model=D_MODEL)
        query  = torch.randn(B, D_MODEL)
        output = mem(query)
        assert output.shape == query.shape, (
            f"Expected output shape {query.shape}, got {output.shape}"
        )

    def test_single_query(self):
        S   = torch.eye(D_MODEL)
        mem = SpectralAssociativeMemory(S, d_model=D_MODEL)
        query  = torch.randn(1, D_MODEL)
        output = mem(query)
        assert output.shape == (1, D_MODEL)

    def test_large_batch(self):
        S   = torch.eye(D_MODEL)
        mem = SpectralAssociativeMemory(S, d_model=D_MODEL)
        query  = torch.randn(32, D_MODEL)
        output = mem(query)
        assert output.shape == (32, D_MODEL)

    def test_rejects_wrong_input_dim(self):
        S   = torch.eye(D_MODEL)
        mem = SpectralAssociativeMemory(S, d_model=D_MODEL)
        with pytest.raises(ValueError):
            mem(torch.randn(B, D_MODEL + 1))

    def test_rejects_non_square_S_memory(self):
        with pytest.raises(ValueError):
            SpectralAssociativeMemory(torch.randn(D_MODEL, D_MODEL + 1), d_model=D_MODEL)

    def test_rejects_d_model_mismatch(self):
        with pytest.raises(ValueError):
            SpectralAssociativeMemory(torch.eye(D_MODEL), d_model=D_MODEL + 1)


# ---------------------------------------------------------------------------
# AC2  --  retrieval SNR: orthonormal keys, cosine similarity > 0.95
# ---------------------------------------------------------------------------

class TestRetrievalSNR:
    def test_orthonormal_key_retrieval_cos_sim_above_095(self):
        """
        Build a Hopfield memory with q orthonormal keys and q unit-norm values.
        Query with the first key; the retrieved output should have cosine
        similarity > 0.95 with the first stored value.

        This holds when d >> q so the cross-term interference is small:
        with d=64, q=4 the SNR is d_eff/q >= 16/4 = 4, well above threshold.
        """
        torch.manual_seed(0)
        q_patterns = 4
        keys   = _orthonormal_keys(q_patterns, D_MODEL)  # (q, d)
        values = _orthonormal_keys(q_patterns, D_MODEL)  # (q, d)  independent

        S_mem = _outer_product_memory(keys, values)   # (d, d)
        mem   = SpectralAssociativeMemory(S_mem, d_model=D_MODEL)

        # Query with the first stored key (exactly on the key manifold).
        query  = keys[0:1]   # (1, d)
        output = mem(query)  # (1, d)

        cos_sim = F.cosine_similarity(output, values[0:1], dim=-1).item()
        assert cos_sim > 0.95, (
            f"Retrieval cosine similarity {cos_sim:.4f} is below 0.95. "
            f"Hopfield retrieval quality is insufficient."
        )


# ---------------------------------------------------------------------------
# AC3  --  delta_update: new pattern written; prior patterns not degraded
# ---------------------------------------------------------------------------

class TestDeltaUpdate:
    def test_delta_update_increases_response_to_new_pattern(self):
        """
        Write a new (key, value) pair via delta_update.
        The memory's response to that key (cosine similarity between
        forward(key) and value) must increase after the update.
        """
        torch.manual_seed(1)
        q_patterns = 4
        keys   = _orthonormal_keys(q_patterns, D_MODEL)
        values = _orthonormal_keys(q_patterns, D_MODEL)

        S_mem = _outer_product_memory(keys, values)
        mem   = SpectralAssociativeMemory(S_mem, d_model=D_MODEL)

        # New pattern not in the original store.
        new_key   = F.normalize(torch.randn(D_MODEL), dim=0)
        new_value = F.normalize(torch.randn(D_MODEL), dim=0)

        # Response before update.
        cos_before = F.cosine_similarity(
            mem(new_key.unsqueeze(0)), new_value.unsqueeze(0), dim=-1
        ).item()

        mem.delta_update(new_key, new_value)

        # Response after update.
        cos_after = F.cosine_similarity(
            mem(new_key.unsqueeze(0)), new_value.unsqueeze(0), dim=-1
        ).item()

        assert cos_after > cos_before, (
            f"delta_update did not increase memory response: "
            f"before={cos_before:.4f}, after={cos_after:.4f}"
        )

    def test_delta_update_does_not_degrade_prior_patterns(self):
        """
        After writing a new pattern, all 4 original patterns should still
        have cosine similarity >= 0.80 with their stored values.

        The tolerance is set to 0.80 (rather than 0.95) because one
        non-orthogonal delta update introduces bounded cross-talk.
        """
        torch.manual_seed(2)
        q_patterns = 4
        keys   = _orthonormal_keys(q_patterns, D_MODEL)
        values = _orthonormal_keys(q_patterns, D_MODEL)

        S_mem = _outer_product_memory(keys, values)
        mem   = SpectralAssociativeMemory(S_mem, d_model=D_MODEL)

        # Measure retrieval quality of all prior patterns before update.
        cos_before = [
            F.cosine_similarity(
                mem(keys[i:i+1]), values[i:i+1], dim=-1
            ).item()
            for i in range(q_patterns)
        ]

        # Write one new pattern.
        new_key   = F.normalize(torch.randn(D_MODEL), dim=0)
        new_value = F.normalize(torch.randn(D_MODEL), dim=0)
        mem.delta_update(new_key, new_value)

        # Measure again.
        cos_after = [
            F.cosine_similarity(
                mem(keys[i:i+1]), values[i:i+1], dim=-1
            ).item()
            for i in range(q_patterns)
        ]

        for i, (before, after) in enumerate(zip(cos_before, cos_after)):
            assert after >= 0.80, (
                f"Pattern {i} cos_sim dropped to {after:.4f} after delta_update "
                f"(was {before:.4f}).  Interference too high."
            )

    def test_delta_update_rejects_wrong_key_shape(self):
        S   = torch.eye(D_MODEL)
        mem = SpectralAssociativeMemory(S, d_model=D_MODEL)
        with pytest.raises(ValueError):
            mem.delta_update(torch.randn(D_MODEL + 1), torch.randn(D_MODEL))

    def test_delta_update_rejects_wrong_value_shape(self):
        S   = torch.eye(D_MODEL)
        mem = SpectralAssociativeMemory(S, d_model=D_MODEL)
        with pytest.raises(ValueError):
            mem.delta_update(torch.randn(D_MODEL), torch.randn(D_MODEL + 1))

    def test_delta_update_does_not_require_grad(self):
        """
        delta_update must not affect the gradient graph of S_memory.
        After the update, forward().backward() must not raise.
        """
        S   = torch.eye(D_MODEL)
        mem = SpectralAssociativeMemory(S, d_model=D_MODEL)
        new_key   = F.normalize(torch.randn(D_MODEL), dim=0)
        new_value = F.normalize(torch.randn(D_MODEL), dim=0)
        mem.delta_update(new_key, new_value)

        # forward + backward must succeed without error.
        query  = torch.randn(B, D_MODEL, requires_grad=True)
        output = mem(query)
        output.sum().backward()
        assert query.grad is not None


# ---------------------------------------------------------------------------
# AC4  --  from_vdt() end-to-end construction
# ---------------------------------------------------------------------------

class TestFromvdt:
    def test_from_vdt_returns_spectral_associative_memory(self):
        model, U_q, eigvals_q = _make_vdt_v2_and_spectral()
        artefact = model.extract_spectral_artefact(U_q, eigvals_q)
        d_model  = artefact["S_memory"].shape[0]
        mem = SpectralAssociativeMemory.from_vdt(model, U_q, eigvals_q, d_model=d_model)
        assert isinstance(mem, SpectralAssociativeMemory)

    def test_from_vdt_S_memory_square(self):
        model, U_q, eigvals_q = _make_vdt_v2_and_spectral()
        artefact = model.extract_spectral_artefact(U_q, eigvals_q)
        d_model  = artefact["S_memory"].shape[0]
        mem = SpectralAssociativeMemory.from_vdt(model, U_q, eigvals_q, d_model=d_model)
        S   = mem.S_memory
        assert S.shape[0] == S.shape[1], "S_memory from from_vdt must be square"

    def test_from_vdt_forward_shape(self):
        model, U_q, eigvals_q = _make_vdt_v2_and_spectral()
        artefact = model.extract_spectral_artefact(U_q, eigvals_q)
        d_model  = artefact["S_memory"].shape[0]
        mem    = SpectralAssociativeMemory.from_vdt(model, U_q, eigvals_q, d_model=d_model)
        query  = torch.randn(B, d_model)
        output = mem(query)
        assert output.shape == query.shape


# ---------------------------------------------------------------------------
# AC5  --  serialisability via torch.save / torch.load
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_save_and_load_state_dict(self):
        """
        torch.save / torch.load on the full module must restore S_memory
        and produce identical forward() outputs.
        """
        S   = torch.randn(D_MODEL, D_MODEL)
        S   = (S + S.T) / 2.0        # symmetrise for a well-defined memory
        mem = SpectralAssociativeMemory(S, d_model=D_MODEL)

        query  = torch.randn(B, D_MODEL)
        out_before = mem(query).detach().clone()

        # Round-trip through BytesIO (no filesystem access needed in CI).
        buf = io.BytesIO()
        torch.save(mem, buf)
        buf.seek(0)
        mem_loaded = torch.load(buf, weights_only=False)

        out_after = mem_loaded(query)
        assert torch.allclose(out_before, out_after, atol=1e-5), (
            "Forward output changed after torch.save/load round-trip."
        )

    def test_save_and_load_state_dict_only(self):
        """state_dict() / load_state_dict() must also work correctly."""
        S1  = torch.randn(D_MODEL, D_MODEL)
        S1  = (S1 + S1.T) / 2.0
        mem1 = SpectralAssociativeMemory(S1, d_model=D_MODEL)

        S2   = torch.zeros(D_MODEL, D_MODEL)
        mem2 = SpectralAssociativeMemory(S2, d_model=D_MODEL)

        # Load mem1's weights into mem2.
        mem2.load_state_dict(mem1.state_dict())

        query = torch.randn(B, D_MODEL)
        assert torch.allclose(mem1(query), mem2(query), atol=1e-5)
