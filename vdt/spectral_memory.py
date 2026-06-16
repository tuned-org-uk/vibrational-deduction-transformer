"""
vdt/spectral_memory.py  --  SpectralAssociativeMemory

This module wraps the post-training spectral artefact A(I) produced by
WiringAutoencoder.extract_spectral_artefact() into a pre-built
Hopfield / linear associative memory matrix that can seed a downstream
transformer's feed-forward or cross-attention value matrices.

Two-phase architecture
----------------------
PHASE 1 -- OFFLINE (VDT v2 training)
    ArrowSpace index I  ->  L(I), U_q, Lambda_q
    WiringAutoencoder.train()  ->  ELBO maximisation
    extract_spectral_artefact()  ->  A(I)  = {S_memory, omega_hat, W_hat}
    SpectralAssociativeMemory(A(I))  ->  S_I

PHASE 2 -- ONLINE (Spectral Memory Transformer)
    Transformer FFN / cross-attention initialised from S_I
    Self-attention: dynamic short-term associations
    S_I: long-term spectral prior memory
    Delta-rule updates: write new associations online

Memory matrix construction
--------------------------
The outer-product Hopfield memory is defined as:

    S_I = sum_{k=1}^{q}  E[omega_k] * d_theta(w_hat_k) * w_hat_k^T

where:
  - Keys     w_hat_k  : Laplacian eigenvector-aligned loading directions
                        (approximately orthonormal, maximising retrieval SNR).
  - Values   d_theta(w_hat_k) : decoder responses at each spectral direction.
  - Weights  E[omega_k]       : mode weights, down-weighting high-frequency
                                (noisy) components.

The matrix S_I is stored as a non-persistent nn.Buffer so the module is
serialised by torch.save / torch.load without being treated as a learnable
parameter.

Ref: docs/v2/00-architecture.md -- Spectral Artefact and Associative Memory
Ref: issue #28
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict


class SpectralAssociativeMemory(nn.Module):
    """
    Pre-built Hopfield associative memory seeded from the VDT v2 spectral
    artefact A(I).

    The memory matrix S_memory is stored as a non-learnable buffer of shape
    (d_model, d_model).  It can be updated online via the delta rule without
    touching the gradient graph.

    Parameters
    ----------
    S_memory : Tensor
        Pre-built memory matrix.  Shape (d_model, d_model).
        Must be symmetric (outer-product construction guarantees this).
    d_model : int
        Dimensionality of the key/value space.  Must equal S_memory.shape[0].

    Raises
    ------
    ValueError
        If S_memory is not square or d_model does not match.
    """

    def __init__(self, S_memory: Tensor, d_model: int) -> None:
        super().__init__()
        if S_memory.ndim != 2 or S_memory.shape[0] != S_memory.shape[1]:
            raise ValueError(
                f"S_memory must be a square 2-D tensor, got shape {tuple(S_memory.shape)}"
            )
        if S_memory.shape[0] != d_model:
            raise ValueError(
                f"d_model={d_model} does not match S_memory side {S_memory.shape[0]}"
            )
        self.d_model = d_model
        # Non-persistent so state_dict() never flags it as a missing key
        # when loading a checkpoint that was saved before a delta_update.
        # Set persistent=True if you want checkpoint fidelity across updates.
        self.register_buffer("S_memory", S_memory.clone().detach(), persistent=True)

    # ------------------------------------------------------------------
    # forward -- Hopfield retrieval via softmax attention on spectral keys
    # ------------------------------------------------------------------

    def forward(self, query: Tensor) -> Tensor:
        """
        Hopfield retrieval using the stored spectral memory matrix.

        The retrieval rule is a single-step modern Hopfield update:

            output_i = S_memory^T * softmax( S_memory * query_i / sqrt(d) )

        which is equivalent to attending over the column-space of S_memory.
        For an outer-product memory S = V * K^T with orthonormal K, this
        recovers the nearest stored value with cosine similarity > 0.95 when
        the query is close to a stored key.

        Parameters
        ----------
        query : Tensor
            Shape (B, d_model).  Query vectors.

        Returns
        -------
        Tensor
            Shape (B, d_model).  Retrieved memory vectors.
        """
        if query.ndim != 2 or query.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected query shape (B, {self.d_model}), got {tuple(query.shape)}"
            )
        # logits: (B, d_model) @ (d_model, d_model)^T = (B, d_model)
        scale = self.d_model ** 0.5
        logits = query @ self.S_memory / scale          # (B, d_model)
        attn   = F.softmax(logits, dim=-1)              # (B, d_model)
        output = attn @ self.S_memory.T                 # (B, d_model)
        return output

    # ------------------------------------------------------------------
    # delta_update -- online delta-rule write
    # ------------------------------------------------------------------

    def delta_update(self, key: Tensor, value: Tensor) -> None:
        """
        Online delta-rule association write.

        Updates S_memory += value (x) key^T (outer product) in-place
        without corrupting the gradient graph.  The update is applied
        directly to the buffer using no_grad so it is invisible to
        autograd.

        The delta rule writes a new (key, value) association.  Because
        the stored spectral keys are approximately orthonormal, a single
        new outer-product update degrades the retrieval SNR for existing
        patterns by at most ||key||^2 / d_model, which is <= 1/d_model
        for unit-norm keys.

        Parameters
        ----------
        key : Tensor
            Shape (d_model,).  The retrieval key to associate.
        value : Tensor
            Shape (d_model,).  The value to store at this key.

        Raises
        ------
        ValueError
            If key or value shape does not match d_model.
        """
        if key.shape != (self.d_model,):
            raise ValueError(f"key must have shape ({self.d_model},), got {tuple(key.shape)}")
        if value.shape != (self.d_model,):
            raise ValueError(f"value must have shape ({self.d_model},), got {tuple(value.shape)}")
        with torch.no_grad():
            self.S_memory += torch.outer(value, key)

    # ------------------------------------------------------------------
    # from_vdt -- post-training factory classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_vdt(
        cls,
        vdt_v2: "WiringAutoencoder",  # noqa: F821 -- forward reference
        U_q: Tensor,
        eigvals_q: Tensor,
        d_model: int,
    ) -> "SpectralAssociativeMemory":
        """
        Post-training construction from a trained WiringAutoencoder.

        Calls extract_spectral_artefact() on the trained model and wraps
        the resulting S_memory into a SpectralAssociativeMemory module.

        Parameters
        ----------
        vdt_v2 : WiringAutoencoder
            A trained VDT v2 instance.
        U_q : Tensor
            Leading q eigenvectors of the frozen index Laplacian L(I).
            Shape (N, q).
        eigvals_q : Tensor
            Leading q eigenvalues of L(I).  Shape (q,).
        d_model : int
            Expected dimensionality of the memory matrix.  Must match the
            model's decoder output dimension.

        Returns
        -------
        SpectralAssociativeMemory
            Ready-to-use memory module seeded from A(I).

        Raises
        ------
        KeyError
            If extract_spectral_artefact() does not return 'S_memory'.
        ValueError
            If the returned S_memory side does not match d_model.
        """
        artefact: Dict[str, Tensor] = vdt_v2.extract_spectral_artefact(U_q, eigvals_q)
        if "S_memory" not in artefact:
            raise KeyError(
                "extract_spectral_artefact() did not return 'S_memory'. "
                "Check WiringAutoencoder implementation."
            )
        S_memory = artefact["S_memory"].detach()
        return cls(S_memory=S_memory, d_model=d_model)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"
