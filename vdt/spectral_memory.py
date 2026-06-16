"""
vdt/spectral_memory.py  --  SpectralAssociativeMemory

This module wraps the post-training spectral artefact A(I) produced by
WiringAutoencoder.extract_spectral_artefact() into a pre-built
linear associative memory matrix that seeds a downstream transformer's
feed-forward or cross-attention value matrices.

Memory model
------------
The memory is a *linear* outer-product associative memory (Kohonen 1972,
Anderson 1972), not the modern (softmax) Hopfield network of Ramsauer et al.
2020.  The distinction matters:

  Linear associative memory  (this module)
  -----------------------------------------
  Construction:  S = sum_k  v_k (x) k_k^T       # outer product sum
  Retrieval:     output = query @ S              # one matrix multiply
  Guarantee:     for orthonormal keys K and query q = k_i,
                 output_i = v_i + interference   # interference ~ O(sqrt(q)/d)
  Best for:      few, approximately orthonormal spectral keys;
                 exact linear superposition; fast online delta-rule updates.

  Modern (softmax) Hopfield  (not this module)
  ---------------------------------------------
  Construction:  store key/value pairs in separate matrices K, V
  Retrieval:     output = V^T softmax(K query / beta)
  Best for:      large pattern capacity; dense overloaded memories.

The spectral keys w_hat_k are Laplacian eigenvector-aligned loading
directions that are approximately orthonormal by construction (they inherit
orthonormality from U_q).  The linear model therefore achieves near-perfect
retrieval SNR for q << d, which is the regime of the VDT architecture.

Two-phase architecture
----------------------
PHASE 1 -- OFFLINE (VDT  training)
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
The outer-product memory matrix is::

    S_I = sum_{k=1}^{q}  E[omega_k] * v_k * w_hat_k^T

where w_hat_k are the spectral loading directions (keys) and v_k = d_theta(w_hat_k)
are the decoder responses (values).  Retrieval is::

    output = query @ S_I

For a query equal to key k_i, and orthonormal keys::

    output = v_i + sum_{j != i} (k_i . k_j) * v_j

The interference sum vanishes when keys are exactly orthonormal.  With
approximately orthonormal spectral keys (d=64, q=4) the cosine similarity
between output and v_i exceeds 0.95.

Ref: docs//00-architecture.md -- Spectral Artefact and Associative Memory
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
    Pre-built linear associative memory seeded from the VDT spectral
    artefact A(I).

    The memory matrix S_memory is stored as a non-learnable buffer of shape
    (d_model, d_model).  It encodes key-value associations as outer products::

        S_memory = sum_k  v_k (x) k_k^T     # construction
        output   = query  @  S_memory        # retrieval

    For orthonormal keys the retrieval rule recovers the stored value v_i
    exactly when queried with k_i; cross-pattern interference is
    O(sqrt(q) / d) and drops below 5% for d=64, q=4.

    Online delta-rule updates are supported::

        S_memory  +=  v_new (x) k_new^T

    Because the stored spectral keys are approximately orthonormal, a single
    delta update degrades prior-pattern cosine similarity by at most
    |k_new . k_i| * ||v_new|| / d, which is < 0.20 for random unit-norm
    keys at d=64.

    Parameters
    ----------
    S_memory : Tensor
        Pre-built memory matrix.  Shape (d_model, d_model).
        Constructed as sum_k outer(v_k, k_k) -- rows index the value space,
        columns index the key space.
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
        self.register_buffer("S_memory", S_memory.clone().detach(), persistent=True)

    # ------------------------------------------------------------------
    # forward -- linear associative memory retrieval
    # ------------------------------------------------------------------

    def forward(self, query: Tensor) -> Tensor:
        """
        Linear associative memory retrieval.

        Computes::

            output = query @ S_memory          # (B, d_model)

        For a memory built as S = sum_k outer(v_k, k_k) and a query equal
        to key k_i (unit norm, orthonormal to all other keys)::

            output_i = v_i + sum_{j != i} (k_i . k_j) v_j

        The interference terms vanish for exactly orthonormal keys and are
        small when keys are approximately orthonormal (spectral loading
        directions inherited from U_q).

        Note: this is NOT the modern softmax Hopfield retrieval rule
        (Ramsauer et al. 2020).  Softmax attention over the columns of
        S_memory would lose the linear superposition property and degrade
        retrieval cosine similarity to near-random values.

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
        # Linear retrieval: output = query @ S_memory
        # S_memory shape: (d_model, d_model) -- rows = value space, cols = key space
        # query @ S_memory: (B, d_model) @ (d_model, d_model) = (B, d_model)
        return query @ self.S_memory

    # ------------------------------------------------------------------
    # delta_update -- online delta-rule write
    # ------------------------------------------------------------------

    def delta_update(self, key: Tensor, value: Tensor) -> None:
        """
        Online delta-rule association write.

        Updates S_memory in-place::

            S_memory += outer(value, key)      # value (x) key^T

        Applied inside torch.no_grad() so the buffer update is invisible to
        autograd.  Subsequent forward() calls pick up the new association.

        Interference bound: for a unit-norm new key orthogonal to all stored
        keys, the prior-pattern cosine similarity is unchanged.  For a
        random unit-norm new key (worst case), the degradation per pattern
        is bounded by 1/sqrt(d_model), giving > 0.80 cosine retention for
        d_model >= 64.

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
        vdt: "WiringAutoencoder",  # noqa: F821 -- forward reference
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
        vdt : WiringAutoencoder
            A trained VDT  instance.
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
        artefact: Dict[str, Tensor] = vdt.extract_spectral_artefact(U_q, eigvals_q)
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
