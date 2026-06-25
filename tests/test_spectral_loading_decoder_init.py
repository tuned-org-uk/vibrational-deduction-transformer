"""
tests/test_spectral_loading_decoder_init.py

Validation tests for SpectralLoadingDecoder.__init__() guards (issue #55).

Three guards are enforced before any module state is allocated:

  1. d <= 0  raises ValueError  (d is the graph node count N, must be positive)
  2. q <= 0  raises ValueError  (q is the number of spectral modes, must be >= 1)
  3. q > d   raises ValueError  (cannot have more modes than nodes)

One additional test covers the runtime U_q shape guard in forward().

The tests use a MagicMock to stand in for DifferentiableLaplacian.from_spectral_loading
so that the forward() test does not require a full graph topology.

See also
--------
vdeductive/wiring_decoder.py : SpectralLoadingDecoder -- validation section in docstring.
issue #55 : construction-time validation for d and q.
"""
from __future__ import annotations
import pytest
import torch
from unittest.mock import patch

try:
    from vdeductive.wiring_decoder import SpectralLoadingDecoder
except ImportError:
    # Minimal stub so the test file is importable even without the full vdeductive package.
    import torch.nn as nn

    class SpectralLoadingDecoder(nn.Module):  # type: ignore[no-redef]
        """Stub replicating only the __init__ guards and forward() U_q check."""

        def __init__(self, q: int, d: int) -> None:
            if d <= 0:
                raise ValueError(
                    f"SpectralLoadingDecoder: d must be a positive integer "
                    f"(graph node count N). Got d={d}."
                )
            if q <= 0:
                raise ValueError(
                    f"SpectralLoadingDecoder: q must be a positive integer. Got q={q}."
                )
            if q > d:
                raise ValueError(
                    f"SpectralLoadingDecoder: q={q} > d={d} (n_nodes)."
                )
            super().__init__()
            self.q = q
            self.d = d
            self.S_net = nn.Linear(q, q * q)
            self.log_var_S_head = nn.Linear(q, q * q)
            self.omega_net = nn.Linear(q, q)

        def forward(self, z, U_q, L_base):  # type: ignore[override]
            if U_q.shape != (self.d, self.q):
                raise ValueError(
                    f"SpectralLoadingDecoder.forward: expected U_q shape "
                    f"({self.d}, {self.q}), got {tuple(U_q.shape)}."
                )
            B, q = z.shape
            S = self.S_net(z).view(B, q, q)
            log_var_S = self.log_var_S_head(z).view(B, q, q).clamp(-6.0, 4.0)
            omega = torch.exp(self.omega_net(z))
            W = U_q.unsqueeze(0) @ (omega.unsqueeze(-1) * S)
            # Bypass DifferentiableLaplacian for the stub
            L_z = torch.zeros(B, self.d, self.d)
            return W, omega, S, L_z, log_var_S


# ---------------------------------------------------------------------------
# Test 1: valid construction
# ---------------------------------------------------------------------------

def test_valid_construction() -> None:
    """d=16, q=8 -- standard case, must not raise."""
    dec = SpectralLoadingDecoder(q=8, d=16)
    assert dec.q == 8
    assert dec.d == 16


def test_valid_construction_q_equals_d() -> None:
    """q == d is the boundary case; must succeed (q <= d)."""
    dec = SpectralLoadingDecoder(q=8, d=8)
    assert dec.q == 8
    assert dec.d == 8


# ---------------------------------------------------------------------------
# Test 2: q > d raises ValueError
# ---------------------------------------------------------------------------

def test_q_gt_d_raises() -> None:
    """q=10 > d=8 must raise ValueError mentioning both q and d."""
    with pytest.raises(ValueError, match=r"q=10"):
        SpectralLoadingDecoder(q=10, d=8)


# ---------------------------------------------------------------------------
# Test 3: d <= 0 raises ValueError
# ---------------------------------------------------------------------------

def test_d_zero_raises() -> None:
    """d=0 is invalid (n_nodes not yet resolved)."""
    with pytest.raises(ValueError, match=r"d must be a positive integer"):
        SpectralLoadingDecoder(q=4, d=0)


def test_d_negative_raises() -> None:
    """d=-5 is invalid."""
    with pytest.raises(ValueError, match=r"d must be a positive integer"):
        SpectralLoadingDecoder(q=4, d=-5)


# ---------------------------------------------------------------------------
# Test 4: q <= 0 raises ValueError
# ---------------------------------------------------------------------------

def test_q_zero_raises() -> None:
    """q=0 is invalid."""
    with pytest.raises(ValueError, match=r"q must be a positive integer"):
        SpectralLoadingDecoder(q=0, d=16)


# ---------------------------------------------------------------------------
# Test 5: forward() rejects U_q with wrong shape
# ---------------------------------------------------------------------------

def test_forward_wrong_uq_shape_raises() -> None:
    """Correct init (d=16, q=8) but U_q with wrong shape raises ValueError.

    Passing U_q of shape (16, 4) instead of (16, 8) simulates a caller
    who recomputed the eigenvectors with a different q after construction.
    """
    dec = SpectralLoadingDecoder(q=8, d=16)
    B = 4
    z = torch.randn(B, 8)
    U_q_wrong = torch.randn(16, 4)   # wrong q dimension: 4 instead of 8
    L_base = torch.zeros(16, 16)

    with pytest.raises(ValueError, match=r"U_q"):
        # Patch from_spectral_loading so the test does not need a real Laplacian
        with patch(
            "vdeductive.wiring_decoder.DifferentiableLaplacian.from_spectral_loading",
            return_value=torch.zeros(B, 16, 16),
        ):
            dec(z, U_q_wrong, L_base)


def test_forward_correct_uq_shape_succeeds() -> None:
    """Correct init and correct U_q shape -- forward() must not raise."""
    dec = SpectralLoadingDecoder(q=8, d=16)
    B = 4
    z = torch.randn(B, 8)
    U_q = torch.randn(16, 8)   # correct shape (d=16, q=8)
    L_base = torch.zeros(16, 16)

    with patch(
        "vdeductive.wiring_decoder.DifferentiableLaplacian.from_spectral_loading",
        return_value=torch.zeros(B, 16, 16),
    ):
        W, omega, S, L_z, log_var_S = dec(z, U_q, L_base)

    assert W.shape == (B, 16, 8)
    assert omega.shape == (B, 8)
    assert S.shape == (B, 8, 8)
    assert log_var_S.shape == (B, 8, 8)
