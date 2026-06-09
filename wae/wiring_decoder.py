"""
Wiring Decoder  —  z  →  edge weight adjustments  →  Laplacian L(z).

This module implements the **key architectural novelty** of the Wiring
Autoencoder: the latent code ``z`` controls *how the graph is wired*, not
directly what the output is.  The learned Laplacian ``L(z)`` is then fed
into the diffusion decoder, making every downstream computation spectral.

Architectural context
---------------------
``WiringDecoder`` occupies the second stage of the WAE data flow::

    z  (B, latent_dim)
    |  Linear projection + GELU
    |  n_heads learnable edge templates
    |  softmax mixing gate
    v
    edge_delta  (B, E)          per-edge weight adjustments
    |  DifferentiableLaplacian
    v
    L(z)  (B, N, N)             differentiable Laplacian

Gradients from the reconstruction loss and J_freq flow back through
``L(z)`` to ``edge_delta`` and then to ``z``.  This is the full
differentiable wiring path described in
`docs/00-architecture.md \u00a7 WiringDecoder
<https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#waewiring_decoderpy--wiringdecoder>`_.

Mixture-of-experts design
-------------------------
Each of ``n_heads`` heads learns a full prototype edge-weight template
over all E edges.  The gate network predicts per-sample softmax weights
over the heads, so the mixture ``\u03a3_h gate_h \u00b7 template_h`` selects and
interpolates between qualitatively different wiring modes.  This gives
the latent space a natural "wiring mode" interpretation: each head
corresponds to a distinct topological prototype.

Connection to learning algorithms
----------------------------------
The wiring decoder is shared across all six algorithm variants described
in `docs/03-branching.md
<https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/03-branching.md>`_.
For Option 1 (Deterministic AE) the decoder output feeds directly into
the spectral reconstruction term.  For Options 2\u20136 the same ``L(z)`` is
used as the graph Laplacian that governs wave dynamics, energy, or
classification.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
from .laplacian import DifferentiableLaplacian


class WiringDecoder(nn.Module):
    """
    Decode a latent wiring code ``z`` into a differentiable graph Laplacian
    ``L(z)`` via a mixture-of-experts over ``n_heads`` edge templates.

    The architecture is::

        z  (B, latent_dim)
          |-- trunk MLP -->
          h  (B, hidden_dim)
          |-- n_heads Linear(hidden_dim, E) --> head_deltas (B, n_heads, E)
          |-- gate Linear(hidden_dim, n_heads) --> gates (B, n_heads) [softmax]
          |--> delta = sum_h gate_h * head_delta_h    (B, E)
          |--> DifferentiableLaplacian(delta)         (B, N, N) or (B, N)

    The edge weight for edge (i,j) is::

        w_ij = base_w_ij * sigmoid(delta_ij)

    where ``base_w_ij`` is the frozen RBF affinity from the kNN graph.
    The gated sigmoid keeps all weights in (0, base_w_ij), preserving
    non-negativity and ensuring the Laplacian remains positive semi-definite.

    For the full derivation of why differentiability through ``L(z)`` is
    critical to end-to-end learning see
    `docs/00-architecture.md \u00a7 DifferentiableLaplacian
    <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#waelaplacianpy--differentiablelaplacian>`_.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent code ``z``.
    n_edges : int
        Total number of directed edges E in the base kNN graph.
        Obtained from ``laplacian.base_weights.shape[0]``.
    hidden_dim : int
        Width of the shared trunk MLP.
    n_heads : int
        Number of mixture heads.  More heads = richer wiring vocabulary
        at higher parameter cost.  Typical range: 2\u20138.
    laplacian : DifferentiableLaplacian
        Pre-built differentiable Laplacian module holding the frozen graph
        topology (edge_index, base_weights) as registered buffers.
    """

    def __init__(
        self,
        latent_dim: int,
        n_edges: int,
        hidden_dim: int,
        n_heads: int,
        laplacian: DifferentiableLaplacian,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_edges = n_edges
        self.laplacian = laplacian

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_projs = nn.ModuleList([
            nn.Linear(hidden_dim, n_edges) for _ in range(n_heads)
        ])
        self.gate = nn.Linear(hidden_dim, n_heads)

    def forward(
        self,
        z: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent code ``z`` to a differentiable Laplacian.

        Parameters
        ----------
        z : torch.Tensor
            Latent wiring code.  Shape ``(B, latent_dim)``.
        node_idx : torch.Tensor or None
            Long tensor of shape ``(B,)`` of query node indices.
            When provided the Laplacian is returned as a row tensor
            ``(B, N)`` using the memory-efficient per-node path of
            ``DifferentiableLaplacian`` (no N\u00d7N matrix materialised).
            When ``None`` the full ``(B, N, N)`` Laplacian is returned.

            .. note::
                Passing ``node_idx`` here avoids constructing and storing
                the full ``(B, N, N)`` matrix during training, which is
                the dominant memory and compute bottleneck on MPS/CPU.
                See `issue #22
                <https://github.com/tuned-org-uk/wiring-autoencoder/issues/22>`_.

        Returns
        -------
        L : torch.Tensor
            ``(B, N, N)`` when ``node_idx`` is ``None``.
            ``(B, N)``   when ``node_idx`` is provided.
        delta : torch.Tensor
            ``(B, E)`` per-edge weight deltas (useful for diagnostics
            and visualisation of learned wiring patterns).
        """
        h = self.trunk(z)                                        # (B, H)
        gates = self.gate(h).softmax(dim=-1)                     # (B, n_heads)
        head_deltas = torch.stack(
            [proj(h) for proj in self.head_projs], dim=1
        )                                                        # (B, n_heads, E)
        delta = (gates.unsqueeze(-1) * head_deltas).sum(dim=1)   # (B, E)
        L = (
            self.laplacian(delta, node_idx=node_idx)
            if node_idx is not None
            else self.laplacian(delta)
        )
        return L, delta
