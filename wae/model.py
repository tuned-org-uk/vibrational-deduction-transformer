"""
WiringAutoencoder  ---  full model assembling all WAE modules.

This is the top-level model class.  It wires together the four sub-modules
defined in ``wae/encoder.py``, ``wae/wiring_decoder.py``,
``wae/diffusion_decoder.py``, and ``wae/laplacian.py`` and exposes a
single ``forward()`` method that returns all ELBO components.

ELBO
----
The full Wiring Autoencoder objective is::

    L(th, phi; x, i) = E_{q_phi(z|x)}[ log p_th(x | z, i) ]
                     - beta * KL( q_phi(z|x) || p(z) )
                     - alpha * J_freq( L(z) )

where:
    x       raw embedding of query node i
    z       latent wiring code sampled via reparameterisation
    L(z)    differentiable Laplacian of the learned graph wiring
    beta, alpha   KL and frequency regularisation weights (from config)

See `docs/00-architecture.md S ELBO Derivation
<https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#elbo-derivation>`_
for the full derivation and the relationship between J_freq and ArrowSpace
tau-mode truncation.

For the six algorithm variants that build on top of this ELBO (deterministic
AE, energy-based, latent diffusion, variational Laplace, forecasting,
classifier) see `docs/03-branching.md
<https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/03-branching.md>`_.

Performance
-----------
The training forward pass involves three spectral operations that are
potentially O(N^3) per call:

1. ``lambda_fingerprint(base_L)``   -- encoder enrichment
2. ``TauModeDiffusion(L)``          -- eigenvectors for diffusion kernel
3. ``spectral_freq_cost(L)``        -- eigenvalues for J_freq

All three can be eliminated from the training loop by precomputing
spectral quantities from the fixed ``base_L`` once and passing them as
arguments.  The ``WiringAutoencoder.forward`` signature accepts
``lambda_fp``, ``spectral_cache``, and ``freq_eigvals`` for exactly this
purpose.  See ``train.py`` for the canonical caching pattern and
`issue #22
<https://github.com/tuned-org-uk/wiring-autoencoder/issues/22>`_ for the
full performance analysis.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal, Optional, Any, Tuple

from .encoder import WiringEncoder
from .wiring_decoder import WiringDecoder
from .diffusion_decoder import DiffusionDecoder
from .laplacian import DifferentiableLaplacian
from .spectral import spectral_freq_cost, lambda_fingerprint


class WiringAutoencoder(nn.Module):
    """
    Full Wiring Autoencoder.

    Assembles ``WiringEncoder``, ``WiringDecoder``, and ``DiffusionDecoder``
    into a single trainable model and computes the three-term WAE-ELBO.

    Data flow summary::

        x  (B, D)  [+ lambda-fingerprint]
          --> WiringEncoder          z, mu, log sigma^2  (B, latent_dim)
          --> WiringDecoder          L(z)                (B, N, N)
          --> DiffusionDecoder       x_hat               (B, D)
          --> recon_loss + KL + J_freq

    The full data-flow diagram is in
    `docs/00-architecture.md S Data Flow Diagram
    <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#data-flow-diagram>`_.

    Connection to ArrowSpace
    ------------------------
    The ``DifferentiableLaplacian`` embedded in ``WiringDecoder`` mirrors
    ``ArrowSpaceBuilder.build()`` from the ``arrowspace`` library.  The
    correspondence between ArrowSpace concepts and WAE components is
    documented in
    `docs/00-architecture.md S Connection to ArrowSpace
    <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/00-architecture.md#connection-to-arrowspace--graph-wiring>`_.

    Stability
    ---------
    Key quantities to monitor during training:

    - ``lambda_max(L(z))``  -- governs the CFL bound on diffusion time ``t``.
    - ``J_freq``             -- should decrease as wiring becomes smoother.
    - ``KL``                 -- should stabilise after warm-up.
    - Spectral entropy       -- should remain high (no representation collapse).

    See `docs/04-stability.md S 7
    <https://github.com/tuned-org-uk/wiring-autoencoder/blob/main/docs/04-stability.md#7-evaluation-protocol-stability-metrics-checklist>`_
    for the full stability metrics checklist.

    Parameters
    ----------
    input_dim : int
        ``D`` -- embedding dimension of each node.
    latent_dim : int
        ``k`` -- dimension of the latent wiring code ``z``.
    hidden_dim : int
        Hidden width shared by encoder, wiring decoder, and MLP refinement.
    n_wiring_heads : int
        Number of mixture heads in the wiring decoder.
    tau_modes : int
        Number of eigenvectors kept in tau-mode diffusion and used in
        ``J_freq`` and ``lambda_fingerprint``.
    beta : float
        KL weight ``beta`` in the ELBO.
    alpha : float
        J_freq weight ``alpha`` in the ELBO.
    laplacian : DifferentiableLaplacian
        Pre-built Laplacian module (frozen topology + base weights).  Built
        from the embedding table ``E`` via
        ``DifferentiableLaplacian.from_embeddings``.
    use_lambda_features : bool
        If ``True``, concatenate the lambda-fingerprint of ``base_L`` to the
        encoder input.  Controlled by ``model.use_lambda_features`` in the
        YAML config.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_wiring_heads: int,
        tau_modes: int,
        beta: float,
        alpha: float,
        laplacian: DifferentiableLaplacian,
        use_lambda_features: bool = True,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.tau_modes = tau_modes

        n_edges = laplacian.base_weights.shape[0]

        self.encoder = WiringEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_lambda_features=use_lambda_features,
        )
        self.wiring_decoder = WiringDecoder(
            latent_dim=latent_dim,
            n_edges=n_edges,
            hidden_dim=hidden_dim,
            n_heads=n_wiring_heads,
            laplacian=laplacian,
        )
        self.diffusion_decoder = DiffusionDecoder(
            embedding_dim=input_dim,
            hidden_dim=hidden_dim,
            tau_modes=tau_modes,
        )
        self._laplacian = laplacian

    def forward(
        self,
        x: torch.Tensor,
        E: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
        base_L: Optional[torch.Tensor] = None,
        lambda_fp: Optional[torch.Tensor] = None,
        spectral_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        freq_eigvals: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Single forward pass returning all ELBO components.

        The three optional cache arguments (``lambda_fp``, ``spectral_cache``,
        ``freq_eigvals``) together eliminate all per-step O(N^3) CPU
        eigendecompositions from the training loop.  See ``train.py`` for the
        canonical pattern of precomputing these once before the epoch loop.

        Parameters
        ----------
        x : torch.Tensor
            Query node embeddings.  Shape ``(B, D)``.
        E : torch.Tensor
            Full embedding table shared by all batch elements.  Shape
            ``(N, D)``.
        node_idx : torch.Tensor or None
            Long tensor ``(B,)`` of query node indices within ``E``.
            Required for per-node reconstruction loss (training path).
            When ``None`` a full-graph reconstruction is attempted but
            ``recon_loss`` will raise unless node_idx is provided.
        base_L : torch.Tensor or None
            Fixed base Laplacian ``(N, N)`` used as fallback to compute
            ``lambda_fingerprint`` on-the-fly if ``lambda_fp`` is not
            supplied.  Ignored when ``lambda_fp`` is given.
        lambda_fp : torch.Tensor or None
            Pre-computed lambda-fingerprint of ``base_L``.  Shape
            ``(1, n_bins)`` or ``(B, n_bins)``; broadcast to
            ``(B, n_bins)`` automatically.  When supplied, avoids calling
            ``lambda_fingerprint`` inside the forward pass.
        spectral_cache : tuple(eigvals, eigvecs) or None
            Pre-computed eigendecomposition of ``base_L`` for
            ``TauModeDiffusion``.
            ``eigvals`` shape ``(N,)``; ``eigvecs`` shape ``(N, N)``.
            When supplied, avoids the O(N^3) eigensolver in
            ``TauModeDiffusion``.
        freq_eigvals : torch.Tensor or None
            Pre-computed eigenvalues of ``base_L`` for ``J_freq``.
            Shape ``(N,)``.
            When supplied, avoids the O(N^3) eigensolver in
            ``spectral_freq_cost``.

        Returns
        -------
        dict with keys:
            ``loss``       -- total ELBO (scalar, minimise this)
            ``recon_loss`` -- Gaussian NLL reconstruction term
            ``kl_loss``    -- KL(q(z|x) || N(0, I))
            ``freq_loss``  -- J_freq spectral regulariser
            ``x_hat``      -- (B, D) reconstructed embeddings
            ``L``          -- (B, N, N) learned Laplacian
            ``z``          -- (B, latent_dim) latent samples
            ``mu``         -- (B, latent_dim) posterior means
            ``log_var``    -- (B, latent_dim) posterior log-variances
        """
        # --- lambda-fingerprint (encoder enrichment) ------------------------
        # Use cached fp if provided; fall back to on-the-fly computation only
        # when no cache is available (e.g. evaluation without precomputation).
        #
        # Broadcast rule (fixes issue #15):
        #   lambda_fingerprint() may return shape (1, n_bins) when base_L is
        #   a plain 2-D (N, N) tensor (single graph, not a batch).
        #   view(1, -1) normalises both the 1-D (n_bins,) and 2-D (1, n_bins)
        #   cases to (1, n_bins) before expand broadcasts to (B, n_bins).
        #   .contiguous() ensures torch.cat in the encoder receives a
        #   contiguous tensor.
        lam_fp = lambda_fp
        if self.encoder.use_lambda_features and lam_fp is None and base_L is not None:
            with torch.no_grad():
                lam_fp = lambda_fingerprint(base_L, tau_modes=self.tau_modes)
                lam_fp = lam_fp.view(1, -1).expand(x.shape[0], -1).contiguous()

        # Also broadcast any externally supplied lambda_fp to batch size B.
        # Handles the case where the caller passes a cached (1, n_bins) tensor.
        if lam_fp is not None and lam_fp.shape[0] != x.shape[0]:
            lam_fp = lam_fp.view(1, -1).expand(x.shape[0], -1).contiguous()

        # --- Encode ---------------------------------------------------------
        z, mu, log_var = self.encoder(x, lambda_fp=lam_fp)   # (B, latent_dim)

        # --- Wiring decode --------------------------------------------------
        # Full (B, N, N) Laplacian -- node_idx-aware row path is the next step
        # (see issue #22 for the planned optimisation).
        L, _delta = self.wiring_decoder(z, node_idx=None)    # (B, N, N)

        # --- Diffusion decode -----------------------------------------------
        x_hat = self.diffusion_decoder(
            L, E, node_idx=node_idx, eig_cache=spectral_cache
        )                                                     # (B, D)

        # --- ELBO components ------------------------------------------------
        recon = self.diffusion_decoder.recon_loss(x, x_hat)
        kl = WiringEncoder.kl_loss(mu, log_var)
        j_freq = spectral_freq_cost(
            L, tau_modes=self.tau_modes, eigvals=freq_eigvals
        )
        loss = recon + self.beta * kl + self.alpha * j_freq

        return {
            "loss": loss,
            "recon_loss": recon,
            "kl_loss": kl,
            "freq_loss": j_freq,
            "x_hat": x_hat,
            "L": L,
            "z": z,
            "mu": mu,
            "log_var": log_var,
        }

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        E: torch.Tensor,
    ) -> "WiringAutoencoder":
        """
        Convenience factory: build a ``WiringAutoencoder`` from a parsed
        YAML config dict and an embedding table.

        Usage::

            with open('configs/default.yaml') as f:
                cfg = yaml.safe_load(f)
            data = load_dataset('cora', ...)
            model = WiringAutoencoder.from_config(cfg, data['E'])

        The config structure mirrors ``configs/default.yaml``::

            model:
              latent_dim: 32
              hidden_dim: 256
              n_wiring_heads: 4
              tau_modes: 8
              beta: 1.0
              alpha: 0.1
              use_lambda_features: true
            graph:
              knn_k: 15
              sigma: 0.5
              normalised: true
              sparse: true

        Parameters
        ----------
        cfg : dict
            Parsed YAML config.  Must contain ``model`` and ``graph`` keys.
        E : torch.Tensor
            Embedding table ``(N, D)`` used to build the kNN base graph and
            to set ``input_dim = D``.

        Returns
        -------
        WiringAutoencoder
            Fully initialised model (not yet moved to a device; call
            ``.to(device)`` after).
        """
        mc = cfg["model"]
        gc = cfg["graph"]

        lap = DifferentiableLaplacian.from_embeddings(
            E,
            knn_k=gc["knn_k"],
            sigma=gc["sigma"],
            normalised=gc["normalised"],
            sparse=gc.get("sparse", False),
        )
        return cls(
            input_dim=E.shape[1],
            latent_dim=mc["latent_dim"],
            hidden_dim=mc["hidden_dim"],
            n_wiring_heads=mc["n_wiring_heads"],
            tau_modes=mc["tau_modes"],
            beta=mc["beta"],
            alpha=mc["alpha"],
            laplacian=lap,
            use_lambda_features=mc["use_lambda_features"],
        )
