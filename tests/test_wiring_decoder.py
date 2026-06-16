"""
tests/test_wiring_decoder.py  --  acceptance tests for SpectralLoadingDecoder (#26).

AC1  forward shapes
AC2  L_z structural constraints (symmetry, zero row sums, non-positive off-diagonal)
AC3  gradient flows from L_z back to z
AC4  omega strictly positive
AC5  near-identity init (S ~ eye(q), omega ~ ones(q))
AC6  v1 WiringDecoder backward compat
"""
from __future__ import annotations
import torch
import torch.nn as nn
import pytest
from wae.wiring_decoder import SpectralLoadingDecoder, WiringDecoder
from wae.laplacian import DifferentiableLaplacian


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_base_lap(N: int, device: str = "cpu") -> torch.Tensor:
    """Build a simple symmetric normalised Laplacian for a ring graph."""
    A = torch.zeros(N, N)
    for i in range(N):
        j = (i + 1) % N
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(-1).clamp(min=1e-6)
    d_inv = deg.pow(-0.5)
    normed = d_inv.unsqueeze(-1) * A * d_inv.unsqueeze(-2)
    return (torch.eye(N) - normed).to(device)


def _make_U_q(L_base: torch.Tensor, q: int) -> torch.Tensor:
    """Return the q leading eigenvectors of L_base (smallest eigenvalues)."""
    with torch.no_grad():
        eigvals, eigvecs = torch.linalg.eigh(L_base)   # ascending
    return eigvecs[:, :q].contiguous()  # (N, q)


# ---------------------------------------------------------------------------
# AC1 -- forward shapes
# ---------------------------------------------------------------------------

class TestForwardShapes:
    """
    AC1: SpectralLoadingDecoder.forward() returns tensors with correct shapes.
    """

    @pytest.mark.parametrize("B,N,q", [
        (1, 8, 3),
        (4, 8, 3),
        (2, 16, 5),
    ])
    def test_shapes(self, B: int, N: int, q: int) -> None:
        d = N  # standard case d == N
        decoder = SpectralLoadingDecoder(q=q, d=d)
        L_base = _make_base_lap(N)
        U_q = _make_U_q(L_base, q)  # (d, q)

        z = torch.randn(B, q)
        W, omega, S, L_z = decoder(z, U_q, L_base)

        assert W.shape    == (B, d, q),  f"W shape {W.shape}"
        assert omega.shape == (B, q),    f"omega shape {omega.shape}"
        assert S.shape    == (B, q, q),  f"S shape {S.shape}"
        assert L_z.shape  == (B, N, N),  f"L_z shape {L_z.shape}"


# ---------------------------------------------------------------------------
# AC2 -- Laplacian structural constraints
# ---------------------------------------------------------------------------

class TestLaplacianConstraints:
    """
    AC2: L_z satisfies the three structural properties of a normalised
    symmetric Laplacian:
      (a) symmetry
      (b) zero row sums
      (c) non-positive off-diagonal entries
    """

    def _get_L_z(self, B: int = 3, N: int = 10, q: int = 4) -> torch.Tensor:
        decoder = SpectralLoadingDecoder(q=q, d=N)
        L_base = _make_base_lap(N)
        U_q = _make_U_q(L_base, q)
        z = torch.randn(B, q)
        _, _, _, L_z = decoder(z, U_q, L_base)
        return L_z

    def test_symmetry(self) -> None:
        L_z = self._get_L_z()
        err = (L_z - L_z.transpose(-1, -2)).abs().max().item()
        assert err < 1e-5, f"Symmetry error {err:.2e}"

    def test_zero_row_sums(self) -> None:
        L_z = self._get_L_z()
        row_sums = L_z.sum(dim=-1).abs().max().item()
        assert row_sums < 1e-5, f"Row-sum error {row_sums:.2e}"

    def test_off_diagonal_non_positive(self) -> None:
        L_z = self._get_L_z(B=2, N=8, q=3)
        B, N, _ = L_z.shape
        mask = ~torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
        off_diag = L_z[mask]
        assert (off_diag <= 1e-5).all(), (
            f"Off-diagonal max: {off_diag.max().item():.4f} (expected <= 0)"
        )


# ---------------------------------------------------------------------------
# AC3 -- gradient flows back to z
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """
    AC3: Gradients flow from L_z back to z.
    Verified both with manual backward and with torch.autograd.gradcheck on
    a small (N=d=q=4, B=1) instance.
    """

    def test_manual_backward(self) -> None:
        B, N, q = 2, 8, 3
        decoder = SpectralLoadingDecoder(q=q, d=N)
        L_base = _make_base_lap(N)
        U_q = _make_U_q(L_base, q)

        z = nn.Parameter(torch.randn(B, q))
        W, omega, S, L_z = decoder(z, U_q, L_base)
        loss = L_z.pow(2).mean()
        loss.backward()

        assert z.grad is not None, "No gradient on z"
        assert z.grad.shape == z.shape
        assert not torch.isnan(z.grad).any(), "NaN in z.grad"

    def test_gradcheck(self) -> None:
        """gradcheck on a tiny graph (N=d=q=4, B=1) in double precision."""
        N, q = 4, 4
        decoder = SpectralLoadingDecoder(q=q, d=N).double()
        L_base = _make_base_lap(N).double()
        U_q = _make_U_q(L_base, q).double()

        def fn(z: torch.Tensor) -> torch.Tensor:
            _, _, _, L_z = decoder(z, U_q, L_base)
            return L_z

        z = torch.randn(1, q, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(fn, (z,), eps=1e-5, atol=1e-4, rtol=1e-3), (
            "gradcheck failed for SpectralLoadingDecoder"
        )


# ---------------------------------------------------------------------------
# AC4 -- omega strictly positive
# ---------------------------------------------------------------------------

class TestOmegaPositive:
    """
    AC4: omega = exp(omega_net(z)) is strictly positive for any z.
    """

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_positive_various_seeds(self, seed: int) -> None:
        torch.manual_seed(seed)
        B, N, q = 4, 8, 4
        decoder = SpectralLoadingDecoder(q=q, d=N)
        L_base = _make_base_lap(N)
        U_q = _make_U_q(L_base, q)
        z = torch.randn(B, q) * 10  # large magnitude to stress-test exp gate
        _, omega, _, _ = decoder(z, U_q, L_base)
        assert (omega > 0).all(), f"omega has non-positive entries: {omega.min().item()}"


# ---------------------------------------------------------------------------
# AC5 -- near-identity init
# ---------------------------------------------------------------------------

class TestNearIdentityInit:
    """
    AC5: At initialisation (z=0), S is close to eye(q) and omega is close to
    ones(q).  This ensures the decoder starts from a well-conditioned point.
    """

    def test_S_close_to_identity(self) -> None:
        q, N = 5, 10
        decoder = SpectralLoadingDecoder(q=q, d=N)
        z_zero = torch.zeros(1, q)
        L_base = _make_base_lap(N)
        U_q = _make_U_q(L_base, q)
        _, _, S, _ = decoder(z_zero, U_q, L_base)
        # S[0] should equal eye(q) exactly at z=0 given _init_weights
        err = (S[0] - torch.eye(q)).abs().max().item()
        assert err < 1e-5, f"S init error {err:.2e}; expected close to eye({q})"

    def test_omega_close_to_ones(self) -> None:
        q, N = 5, 10
        decoder = SpectralLoadingDecoder(q=q, d=N)
        z_zero = torch.zeros(1, q)
        L_base = _make_base_lap(N)
        U_q = _make_U_q(L_base, q)
        _, omega, _, _ = decoder(z_zero, U_q, L_base)
        # omega = exp(~0) ~ 1  with small weight init
        err = (omega[0] - torch.ones(q)).abs().max().item()
        assert err < 0.1, f"omega init error {err:.4f}; expected close to 1"


# ---------------------------------------------------------------------------
# AC6 -- v1 WiringDecoder backward compat
# ---------------------------------------------------------------------------

class TestWiringDecoderV1Compat:
    """
    AC6: WiringDecoder (v1) is unaffected by the addition of
    SpectralLoadingDecoder.  Forward signature and output shapes must be
    unchanged.
    """

    def _make_wiring_decoder(
        self, N: int = 16, E: int = 48, latent_dim: int = 8, n_heads: int = 2
    ) -> WiringDecoder:
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        edge_index = torch.stack([src, dst])
        base_weights = torch.rand(E).abs() + 0.1
        lap = DifferentiableLaplacian(
            n_nodes=N,
            edge_index=edge_index,
            base_weights=base_weights,
        )
        return WiringDecoder(
            latent_dim=latent_dim,
            n_edges=E,
            hidden_dim=32,
            n_heads=n_heads,
            laplacian=lap,
        )

    def test_v1_output_shapes(self) -> None:
        B, latent_dim, N, E = 3, 8, 16, 48
        decoder = self._make_wiring_decoder(N=N, E=E, latent_dim=latent_dim)
        z = torch.randn(B, latent_dim)
        L, delta = decoder(z)
        assert L.shape     == (B, N, N)
        assert delta.shape == (B, E)

    def test_v1_gradient(self) -> None:
        B, latent_dim = 2, 8
        decoder = self._make_wiring_decoder(latent_dim=latent_dim)
        z = nn.Parameter(torch.randn(B, latent_dim))
        L, _ = decoder(z)
        loss = L.pow(2).mean()
        loss.backward()
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
