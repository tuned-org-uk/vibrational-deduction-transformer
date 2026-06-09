"""
Diffusion Decoder  —  L(z), E  →  x_hat.

This module implements the final decoding stage of the Wiring Autoencoder:
given the learned Laplacian ``L(z)`` and the shared embedding table ``E``,
it produces reconstructed embeddings via tau-mode spectral diffusion
followed by an optional per-node MLP refinement.

Architectural context
---------------------
``DiffusionDecoder`` is the last block in the WAE data flow::

    L(z)  (B, N, N)  +  E  (N, D)
      |  TauModeDiffusion  (heat kernel K_tau = U exp(-tΛ) U^T)
      |  [optional MLP refinement  — per-node mode only]
      v
    x_hat  (B, D)   (per-node, normal training path)

The Gaussian likelihood is::

    log p(x | z) = -||x - x_hat||² / (2σ²) - D log σ

where ``σ = exp(log_sigma)`` is a learnable scalar.

Output shape contract
---------------------
``node_idx`` provided  →  ``x_hat`` shape ``(B, D)``   (per-node reconstruction)
``node_idx = None``    →  ``x_hat`` shape ``(B, N, D)`` (full-graph diagnostic)

Always pass ``node_idx`` during training.  The full-graph path bypasses
the MLP refinement step (which expects ``(B, D)`` input) and is intended
only for post-training visualisation and probing.

Spectral context
----------------
The diffusion time ``t`` is learnable.  At convergence it encodes the
optimal spectral scale for reconstruction: a small ``t`` keeps many modes,
a large ``t`` blurs toward the graph mean.  The interaction between ``t``
and the CFL bound on Δt is analysed in
`docs/04-stability.md § 3
<https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/04-stability.md#3-numerical-stability-of-the-wave-update>`_.

See also `docs/00-architecture.md § DiffusionDecoder
<https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#waediffusion_decoderpy--diffusiondecoder>`_
for the full module description.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .spectral import TauModeDiffusion
from typing import Optional, Tuple


class DiffusionDecoder(nn.Module):
    """
    Decode a batch of Laplacians ``L(z)`` and an embedding table ``E``
    into reconstructed node embeddings ``x_hat`` via tau-mode spectral
    diffusion.

    Internally delegates the spectral step to ``TauModeDiffusion`` and
    optionally refines the per-node result with a small residual MLP.

    Performance: ``eig_cache``
    --------------------------
    The dominant cost in the training forward pass is the O(N³) CPU
    eigendecomposition inside ``TauModeDiffusion``.  This can be eliminated
    by passing ``eig_cache``::

        # Once before training loop in train.py
        base_eigvals, base_eigvecs = _safe_eigh(base_L)
        spectral_cache = (base_eigvals, base_eigvecs)

        # Each training step
        x_hat = decoder(L, E, node_idx=idx, eig_cache=spectral_cache)

    See `issue #22
    <https://github.com/tuned-org-uk/wiring-autoencoder/issues/22>`_
    for the full analysis.

    Parameters
    ----------
    embedding_dim : int
        ``D`` — dimension of each node embedding in ``E`` and in ``x``.
    hidden_dim : int
        Hidden width for the optional MLP refinement network.
    tau_modes : int
        ``k`` — number of eigenvectors kept in tau-mode diffusion.
        Should match the ``tau_modes`` used in ``spectral_freq_cost``
        and in the encoder's ``lambda_fingerprint`` call.
    diffusion_time : float
        Initial value of the learnable diffusion time ``t``.
    use_mlp_refinement : bool
        If ``True`` (default), apply a residual two-layer MLP to
        ``x_hat_raw`` in per-node mode.  Skipped automatically in full-graph
        mode because the MLP expects ``(B, D)`` input.
    init_log_sigma : float
        Initial value of ``log_sigma``.  ``σ = exp(log_sigma)`` is the
        noise standard deviation in the Gaussian likelihood.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        tau_modes: int = 16,
        diffusion_time: float = 1.0,
        use_mlp_refinement: bool = True,
        init_log_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        self.diffusion = TauModeDiffusion(
            tau_modes=tau_modes,
            diffusion_time=diffusion_time,
            learnable_time=True,
        )
        self.use_mlp_refinement = use_mlp_refinement
        if use_mlp_refinement:
            self.refine_mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma))

    def forward(
        self,
        L: torch.Tensor,
        E: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
        eig_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Run tau-mode diffusion and optionally refine with the MLP.

        Parameters
        ----------
        L : torch.Tensor
            Batch of Laplacians.  Shape ``(B, N, N)``.
        E : torch.Tensor
            Embedding table.  Shape ``(N, D)``.
        node_idx : torch.Tensor or None
            Long tensor ``(B,)`` selecting one node per sample.
            When given the output is ``(B, D)`` and MLP refinement is
            applied.  When ``None`` the output is ``(B, N, D)`` and the
            MLP is skipped (MLP expects ``(B, D)`` — see contract above).
        eig_cache : tuple(eigvals, eigvecs) or None
            Pre-computed spectral cache from the fixed ``base_L``.
            Passed directly to ``TauModeDiffusion.forward`` to skip the
            per-step eigensolver call.
            ``eigvals`` shape ``(N,)``; ``eigvecs`` shape ``(N, N)``.

        Returns
        -------
        torch.Tensor
            ``(B, D)``    when ``node_idx`` is not ``None``.
            ``(B, N, D)`` when ``node_idx`` is ``None``.
        """
        x_raw = self.diffusion(L, E, node_idx=node_idx, eig_cache=eig_cache)
        if self.use_mlp_refinement and node_idx is not None:
            x_raw = x_raw + self.refine_mlp(x_raw)   # residual connection
        return x_raw

    def recon_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood reconstruction loss.

        Computes the per-node ELBO reconstruction term::

            -log p(x | z) = ||x - x_hat||² / (2σ²) + D log σ

        This is the ``E_q[log p(x|z)]`` term in the WAE-ELBO::

            L_WAE = E_q[log p(x|z)] - β KL - α J_freq

        See `docs/00-architecture.md § ELBO Derivation
        <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#elbo-derivation>`_.

        Parameters
        ----------
        x : torch.Tensor
            Ground-truth embeddings.  Shape ``(B, D)``.
        x_hat : torch.Tensor
            Reconstructed embeddings.  Must be ``(B, D)``; call this only
            in per-node mode.  Raises ``ValueError`` if ``x_hat.dim() != 2``.
        reduction : str
            ``'mean'`` (default) or ``'sum'``.

        Returns
        -------
        torch.Tensor
            Scalar reconstruction loss.
        """
        if x_hat.dim() != 2:
            raise ValueError(
                "recon_loss expects per-node x_hat with shape (B, D). "
                f"Got shape {tuple(x_hat.shape)}. "
                "Pass node_idx to forward() when computing training loss."
            )
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        sq_err = ((x - x_hat) ** 2).sum(dim=-1)   # (B,)
        D = x.shape[-1]
        nll = sq_err / (2 * sigma ** 2) + D * self.log_sigma
        return nll.mean() if reduction == "mean" else nll.sum()
