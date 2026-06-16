"""
Wiring Autoencoder  --   only.

This module provides the single top-level model class::

    WiringAutoencoder   , three-term ELBO (recon + kl_z + kl_S + kl_tau).

 architecture (three-term ELBO, issue #27)
--------------------------------------------
Assembles WiringEncoder, SpectralLoadingDecoder, and DiffusionDecoder
under the three-term variational objective::

    L_VDTv2 = E_q[log p(x|z,W)]
             - KL( q(z)  || N(0,I)          )   # kl_z  -- isotropic
             - KL( q(S)  || p(S|I)          )   # kl_S  -- spectral basis
             - KL( q(w)  || p(w | tau, L)   )   # kl_tau -- mode frequency

Data flow::

    x  (B, D),  U_q (N, q),  eigvals_q (q,),  L_f (B, N, N)
      --> WiringEncoder      z, mu, log_var, log_a, log_b
      --> SpectralLoadingDecoder  W (B, d, q), omega (B, q), S (B, q, q),
                                  L_z (B, N, N)
      --> DiffusionDecoder   uses self.embedding (N, D) as the node table
                             x_hat (B, D)
      --> three-term ELBO

The L_z key is NOT included in the return dict (PR #35).

Embedding table
---------------
DiffusionDecoder.forward() requires an embedding table E of shape (N, D)
(the full per-node feature matrix) so that the heat-kernel row
k_row (B, N) can be multiplied against E_b (B, N, D).  The per-batch
query x (B, D) must NOT be used as E -- its leading dimension is B, not N.

WiringAutoencoder therefore stores ``self.embedding`` as an nn.Parameter
of shape (n_nodes, input_dim), initialised to zeros.  Callers should
populate it (e.g. with pre-trained node embeddings) before training.
forward() passes self.embedding as E by default; an explicit
``embedding_table`` kwarg overrides it for inference on a new graph.

v1 (WiringAutoencoder) has been removed.  Only version 2 is supported.

Config dispatch
---------------
    model.version: 2  -> WiringAutoencoder   ()

Ref: docs//00-architecture.md
Ref: docs//05-Code.md
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Any, Tuple

from .encoder import WiringEncoder
from .diffusion_decoder import DiffusionDecoder
from .laplacian import DifferentiableLaplacian


# ---------------------------------------------------------------------------
# -specific imports  (SpectralLoadingDecoder + KL helpers)
# ---------------------------------------------------------------------------

from .wiring_decoder import SpectralLoadingDecoder
from .spectral import spectral_basis_kl, tau_mode_kl


# ---------------------------------------------------------------------------
# WiringAutoencoder  ( -- three-term ELBO, issue #27)
# ---------------------------------------------------------------------------

class WiringAutoencoder(nn.Module):
    """
    Wiring Autoencoder  -- three-term ELBO with spectral mode priors.

    Assembles WiringEncoder, SpectralLoadingDecoder, and DiffusionDecoder
    under the three-term variational objective::

        L_VDTv2 = E_q[log p(x|z,W)]
                 - KL( q(z)  || N(0,I)          )   # kl_z  -- isotropic
                 - KL( q(S)  || p(S|I)          )   # kl_S  -- spectral basis
                 - KL( q(w)  || p(w|tau,Lambda) )   # kl_tau -- mode frequency

    Note: the Laplacian-precision latent KL (Term 2, kl_lap) has been
    removed per PR #35.  The L_z tensor is also not returned from forward()
    for the same reason.

    Embedding table
    ---------------
    ``self.embedding`` is a learnable nn.Parameter of shape
    ``(n_nodes, input_dim)``.  It is the canonical node embedding table E
    passed to DiffusionDecoder at every forward call.  Initialised to zeros;
    callers should overwrite it with pre-trained embeddings before training::

        model.embedding.data.copy_(pretrained_E)

    The per-batch query ``x`` (shape ``(B, D)``) is the reconstruction
    target and encoder input; it is NOT the embedding table.

    Data flow::

        x  (B, D),  U_q (N, q),  eigvals_q (q,),  L_f (B, N, N)
          --> WiringEncoder
                z        (B, latent_dim)
                mu       (B, latent_dim)
                log_var  (B, latent_dim)
                log_a    (B, q)           -- Gamma shape log-params
                log_b    (B, q)           -- Gamma rate  log-params
          --> SpectralLoadingDecoder
                W        (B, feat_dim, q) -- spectral loading matrix
                omega    (B, q)           -- mode weights
                S        (B, q, q)        -- rotation matrix
                L_z      (B, N, N)        -- synthesised Laplacian (internal)
          --> DiffusionDecoder
                uses self.embedding  (N, D) as the node embedding table
                x_hat    (B, D)
          --> three-term ELBO

    Parameters
    ----------
    input_dim : int
        D -- node embedding dimension.
    latent_dim : int
        Dimension of the isotropic VAE latent z and the number of latent
        modes passed to WiringEncoder as latent_dim.
    hidden_dim : int
        Per-node feature channel width (feat_dim) used inside WiringEncoder
        and also the MLP width of DiffusionDecoder.
    q : int
        Number of spectral modes; must match U_q.shape[1] at runtime.
        Passed to WiringEncoder as latent_dim so that mode-weight heads
        produce exactly q outputs.
    n_nodes : int or None
        Graph node count N.  Inferred from laplacian.n_nodes when None.
    tau_modes : int
        Number of eigenvectors kept by DiffusionDecoder.
    lam_s : float
        Weight for kl_S (spectral basis KL).  Default 0.01.
    tau : float
        Diffusion time scale for kl_tau (mode frequency KL).  Default 0.5.
    laplacian : DifferentiableLaplacian
        Base Laplacian module shared with SpectralLoadingDecoder.
        Its dense base Laplacian (laplacian.base_laplacian) is passed as
        L_base to SpectralLoadingDecoder.forward() at each step.
    n_layers : int
        Number of VDT blocks inside WiringEncoder.  Default 4.
    n_heads : int
        Attention heads per VDT block.  Default 4.
    dropout : float
        Dropout probability inside WiringEncoder.  Default 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        q: int,
        tau_modes: int,
        lam_s: float,
        tau: float,
        laplacian: DifferentiableLaplacian,
        n_nodes: Optional[int] = None,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.q = q
        self.lam_s = lam_s
        self.tau = tau
        self.tau_modes = tau_modes

        # Resolve n_nodes from the laplacian when not supplied explicitly.
        if n_nodes is None:
            n_nodes = laplacian.n_nodes
        self._n_nodes = n_nodes

        # Canonical node embedding table E of shape (N, D).
        # DiffusionDecoder requires the full (N, D) table, not the per-batch
        # query x (B, D).  Initialised to zeros; overwrite with pre-trained
        # embeddings before training: model.embedding.data.copy_(E_pretrained)
        self.embedding = nn.Parameter(
            torch.zeros(n_nodes, input_dim), requires_grad=True
        )

        # WiringEncoder  signature:
        #   input_dim, latent_dim, n_nodes, feat_dim, n_layers, ...
        # hidden_dim plays the role of feat_dim (per-node feature channels).
        # q spectral modes are matched by setting latent_dim=q inside the
        # encoder so that ModeWeightHead produces q outputs.
        self.encoder = WiringEncoder(
            input_dim=input_dim,
            latent_dim=q,          # encoder latent_dim == number of modes
            n_nodes=n_nodes,
            feat_dim=hidden_dim,   # hidden_dim -> per-node feature channels
            n_layers=n_layers,
            n_heads=n_heads,
            use_isotropic_kl=True,
            dropout=dropout,
        )
        # SpectralLoadingDecoder takes (q, d) at init; L_base is supplied
        # at forward() time via self._laplacian.base_laplacian.
        self.wiring_decoder = SpectralLoadingDecoder(
            q=q,
            d=n_nodes,
        )
        self.diffusion_decoder = DiffusionDecoder(
            embedding_dim=input_dim,
            hidden_dim=hidden_dim,
            tau_modes=tau_modes,
        )
        self._laplacian = laplacian

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        U_q: torch.Tensor,
        eigvals_q: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
        spectral_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        L_f: Optional[torch.Tensor] = None,
        embedding_table: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Single forward pass returning all three-term ELBO components.

        Parameters
        ----------
        x : torch.Tensor
            Query node embeddings used as encoder input and reconstruction
            target.  Shape (B, D).  NOTE: this is NOT the embedding table
            passed to DiffusionDecoder -- see ``embedding_table`` below.
        U_q : torch.Tensor
            Leading q eigenvectors of the frozen index Laplacian L(I).
            Shape (N, q).  Passed to both WiringEncoder (as eigvecs) and
            SpectralLoadingDecoder.
        eigvals_q : torch.Tensor
            Corresponding q eigenvalues of L(I).  Shape (q,).
            Used by spectral_basis_kl and tau_mode_kl priors.
        node_idx : torch.Tensor or None
            Long tensor (B,) selecting one node per sample.  Required for
            per-node reconstruction loss; pass None only for diagnostics.
        spectral_cache : tuple(eigvals, eigvecs) or None
            Pre-computed full eigendecomposition of the base Laplacian for
            DiffusionDecoder.  Avoids an O(N^3) eigensolver at each step.
        L_f : torch.Tensor or None
            Feature-space Laplacian (B, N, N) or (N, N) for WiringEncoder.
            When None, a uniform Laplacian is synthesised from U_q and
            eigvals_q on the fly.
        embedding_table : torch.Tensor or None
            Full node embedding table of shape (N, D) to pass to
            DiffusionDecoder instead of the stored self.embedding buffer.
            Use this to override the buffer at inference time on a new graph.
            When None (default), self.embedding is used.

        Returns
        -------
        dict with exactly 9 keys:
            loss     -- total ELBO scalar (minimise this)
            recon    -- Gaussian NLL reconstruction term
            kl_z     -- isotropic KL  KL(q(z) || N(0,I))
            kl_S     -- spectral basis KL  KL(q(S) || p(S|I))
            kl_tau   -- mode frequency KL  KL(q(w) || p(w|tau,L))
            x_hat    -- (B, D) reconstructed embeddings
            z        -- (B, latent_dim) latent samples
            mu       -- (B, latent_dim) posterior means
            log_var  -- (B, latent_dim) posterior log-variances
        """
        B = x.shape[0]

        # Build a feature-space Laplacian from the spectral basis when the
        # caller does not supply one explicitly.
        # L_f = U_q diag(eigvals_q) U_q^T  expanded to (B, N, N).
        if L_f is None:
            L_f_base = U_q @ torch.diag(eigvals_q) @ U_q.t()   # (N, N)
            L_f = L_f_base.unsqueeze(0).expand(B, -1, -1)       # (B, N, N)
        elif L_f.ndim == 2:
            L_f = L_f.unsqueeze(0).expand(B, -1, -1)

        # --- Encode -------------------------------------------------------
        # WiringEncoder.forward() returns (z, mu, log_var, log_a, log_b).
        # eigvecs argument accepts (N, K_eig); we pass the full U_q.
        z, mu, log_var, log_a, log_b = self.encoder(
            x,
            L_f=L_f,
            eigvecs=U_q,
            lap=self._laplacian,
        )

        # --- Spectral decode (wiring) -------------------------------------
        # SpectralLoadingDecoder.forward() signature: (z, U_q, L_base)
        # L_base is the dense base Laplacian from self._laplacian.
        W, omega, S, L_z = self.wiring_decoder(
            z, U_q, self._laplacian.base_laplacian
        )

        # Embedding table E must be (N, D) -- the full node feature matrix.
        # Using x (shape (B, D)) here would cause a shape mismatch inside
        # TauModeDiffusion when k_row (B, N) is bmm'd against E_b (B, N, D).
        E = embedding_table if embedding_table is not None else self.embedding

        # --- Diffusion decode ---------------------------------------------
        x_hat = self.diffusion_decoder(
            L_z, E, node_idx=node_idx, eig_cache=spectral_cache
        )  # (B, D)

        # --- ELBO terms ---------------------------------------------------
        # Term 1: reconstruction  E_q[log p(x|z,W)]
        recon = self.diffusion_decoder.recon_loss(x, x_hat)

        # Term 2: isotropic KL  KL(q(z) || N(0,I))
        kl_z = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        # Term 3: spectral basis KL  KL(q(S) || p(S|I))
        # log_var_S is derived from S as log(S^2 + eps) to stay on the
        # differentiable path without an extra output head.
        log_var_S = (S.pow(2) + 1e-6).log()  # (B, q, q)
        kl_S = spectral_basis_kl(S, log_var_S, eigvals_q, lam_s=self.lam_s)

        # Term 4: mode frequency KL  KL(q(w) || p(w|tau,Lambda))
        kl_tau = tau_mode_kl(log_a, log_b, eigvals_q, tau=self.tau)

        # Total loss (all KL terms are non-negative by construction)
        loss = recon + kl_z + kl_S + kl_tau

        return {
            "loss": loss,
            "recon": recon,
            "kl_z": kl_z,
            "kl_S": kl_S,
            "kl_tau": kl_tau,
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "log_var": log_var,
        }

    # ------------------------------------------------------------------
    # extract_spectral_artefact
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_spectral_artefact(
        self,
        U_q: torch.Tensor,
        eigvals_q: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Produce the spectral artefact dict used for post-training memory
        construction (consumed by SpectralAssociativeMemory, issue #28).

        The model is called at the posterior mode (z = mu = 0, the prior
        mean) to obtain the mean loading matrix W_hat and mode weights
        omega_hat.  The outer-product Hopfield memory matrix S_memory is
        then assembled as::

            S_memory = sum_k E[omega_k] * d_theta(w_hat_k) * w_hat_k^T

        where:
          - w_hat_k   is the k-th column of W_hat  (feat_dim,)
          - d_theta(.) is the decoder response at spectral direction w_hat_k
            (approximated as W_hat[:, k] for a linear decoder)
          - E[omega_k] = omega_hat[:, k].mean() for the scalar weight

        Shape:
          W_hat      (1, feat_dim, q)
          omega_hat  (q,)  -- averaged over the single-sample batch
          S_memory   (feat_dim, feat_dim)

        Parameters
        ----------
        U_q : torch.Tensor
            Leading q eigenvectors of L(I).  Shape (N, q).
        eigvals_q : torch.Tensor
            Leading q eigenvalues of L(I).  Shape (q,).

        Returns
        -------
        dict with keys: W_hat, omega_hat, S_memory
        """
        device = next(self.parameters()).device
        # Sample from the prior mean z = 0 to obtain the posterior-mode artefact.
        z_prior = torch.zeros(1, self.q, device=device)

        W_hat, omega_raw, S, _L_z = self.wiring_decoder(
            z_prior, U_q, self._laplacian.base_laplacian
        )
        # W_hat : (1, feat_dim, q)
        # omega_raw : (1, q)

        omega_hat = omega_raw.squeeze(0)   # (q,)
        W = W_hat.squeeze(0)               # (feat_dim, q)
        d_model = W.shape[0]

        S_memory = torch.zeros(d_model, d_model, device=device)
        for k in range(self.q):
            w_k = W[:, k]                  # (feat_dim,)
            S_memory += omega_hat[k] * torch.outer(w_k, w_k)

        return {
            "W_hat": W_hat,          # (1, feat_dim, q)
            "omega_hat": omega_hat,  # (q,)
            "S_memory": S_memory,    # (feat_dim, feat_dim)
        }

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        U_q: torch.Tensor,
        eigvals_q: torch.Tensor,
        E: torch.Tensor,
        n_samples: int = 1,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample z ~ N(0, I) and decode to reconstructed embeddings.

        Parameters
        ----------
        U_q : torch.Tensor
            Shape (N, q).
        eigvals_q : torch.Tensor
            Shape (q,).
        E : torch.Tensor
            Full embedding table (N, D).
        n_samples : int
            Number of samples to generate.
        node_idx : torch.Tensor or None
            Long tensor (n_samples,) selecting one node per sample.

        Returns
        -------
        torch.Tensor  (n_samples, D) or (n_samples, N, D)
        """
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.q, device=device)
        W, omega, S, L_z = self.wiring_decoder(
            z, U_q, self._laplacian.base_laplacian
        )
        return self.diffusion_decoder(L_z, E, node_idx=node_idx)


# ---------------------------------------------------------------------------
# from_config  --  -only factory
# ---------------------------------------------------------------------------

def from_config(
    cfg: dict[str, Any],
    E: torch.Tensor,
) -> "WiringAutoencoder":
    """
    Build WiringAutoencoder from a parsed YAML config dict.

    Only version 2 is supported.  Passing any other version raises
    ValueError.

     YAML config example::

        model:
          version: 2
          latent_dim: 32   # used as q (number of spectral modes)
          hidden_dim: 256  # feat_dim inside WiringEncoder
          q: 16
          lam_s: 0.01
          tau: 0.5
          tau_modes: 16
          n_layers: 4
          n_heads: 4
        graph:
          knn_k: 15
          sigma: 0.5
          normalised: true
          sparse: false

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.  Must contain a 'model' key.
    E : torch.Tensor
        Embedding table (N, D).  Also used to initialise the model's
        internal self.embedding buffer.

    Returns
    -------
    WiringAutoencoder
    """
    mc = cfg["model"]
    version = int(mc.get("version", 2))

    if version != 2:
        raise ValueError(
            f"Only model version 2 is supported; got version={version}.  "
            "WiringAutoencoder (v1) has been removed."
        )

    gc = cfg.get("graph", {})
    lap = DifferentiableLaplacian.from_embeddings(
        E,
        knn_k=gc.get("knn_k", 15),
        sigma=gc.get("sigma", 0.5),
        normalised=gc.get("normalised", True),
        sparse=gc.get("sparse", False),
    )
    model = WiringAutoencoder(
        input_dim=E.shape[1],
        latent_dim=mc["latent_dim"],
        hidden_dim=mc.get("hidden_dim", 256),
        q=mc["q"],
        tau_modes=mc.get("tau_modes", mc["q"]),
        lam_s=mc.get("lam_s", 0.01),
        tau=mc.get("tau", 0.5),
        laplacian=lap,
        n_layers=mc.get("n_layers", 4),
        n_heads=mc.get("n_heads", 4),
        dropout=mc.get("dropout", 0.1),
    )
    # Initialise the embedding buffer with the supplied table.
    model.embedding.data.copy_(E)
    return model
