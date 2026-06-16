"""
Wiring Autoencoder  --  full model assembling all VDT modules.

This module provides two top-level model classes:

  WiringAutoencoder     v1, original two-term ELBO (recon + kl + J_freq).
  WiringAutoencoderV2   v2, three-term ELBO (recon + kl_z + kl_S + kl_tau).

v1 architecture
---------------
The full Wiring Autoencoder objective is::

    L_VDT(th, phi; x, i) = E_{q_phi(z|x)}[ log p_th(x | z, i) ]
                          - beta  * KL( q_phi(z|x) || N(0,I) )
                          - alpha * J_freq( L(z) )

See docs/00-architecture.md -- ELBO Derivation for the full derivation.

v2 architecture (three-term ELBO, issue #27)
--------------------------------------------
Assembles WiringEncoderV2, SpectralLoadingDecoder, and DiffusionDecoder
under the three-term objective (PR #35 -- Laplacian-precision KL removed)::

    L_VDTv2 = E_q[log p(x|z,W)]
             - KL( q(z)  || N(0,I)          )   # kl_z  -- isotropic
             - KL( q(S)  || p(S|I)          )   # kl_S  -- spectral basis
             - KL( q(w)  || p(w | tau, L)   )   # kl_tau -- mode frequency

Data flow::

    x  (B, D),  U_q (N, q),  eigvals_q (q,)
      --> WiringEncoderV2      z, mu, log_var, log_a, log_b
      --> SpectralLoadingDecoder  W (B, d, q), omega (B, q), S (B, q, q),
                                  L_z (B, N, N)
      --> DiffusionDecoder     x_hat (B, D)
      --> three-term ELBO

The L_z key is NOT included in the return dict (PR #35).

Config dispatch
---------------
    model.version: 1  -> WiringAutoencoder     (v1)
    model.version: 2  -> WiringAutoencoderV2   (v2)

Ref: docs/v2/00-architecture.md
Ref: docs/v2/05-Code.md
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


# ---------------------------------------------------------------------------
# WiringAutoencoder  (v1 -- unchanged)
# ---------------------------------------------------------------------------

class WiringAutoencoder(nn.Module):
    """
    Full Wiring Autoencoder (v1).

    Assembles WiringEncoder, WiringDecoder, and DiffusionDecoder into a
    single trainable model and computes the two-term VDT-ELBO::

        L = E_q[log p(x|z)] - beta*KL(q(z)||N(0,I)) - alpha*J_freq(L(z))

    Data flow::

        x  (B, D)  [+ lambda-fingerprint]
          --> WiringEncoder          z, mu, log sigma^2  (B, latent_dim)
          --> WiringDecoder          L(z)                (B, N, N)
          --> DiffusionDecoder       x_hat               (B, D)
          --> recon_loss + KL + J_freq

    See docs/00-architecture.md for the full data-flow diagram and
    connection to the ArrowSpace library.

    Stability
    ---------
    Key quantities to monitor during training:

    - lambda_max(L(z))  -- governs the CFL bound on diffusion time t.
    - J_freq             -- should decrease as wiring becomes smoother.
    - KL                 -- should stabilise after warm-up.
    - Spectral entropy   -- should remain high (no representation collapse).

    Parameters
    ----------
    input_dim : int
        D -- embedding dimension of each node.
    latent_dim : int
        k -- dimension of the latent wiring code z.
    hidden_dim : int
        Hidden width shared by encoder, wiring decoder, and MLP refinement.
    n_wiring_heads : int
        Number of mixture heads in the wiring decoder.
    tau_modes : int
        Number of eigenvectors kept in tau-mode diffusion and J_freq.
    beta : float
        KL weight in the ELBO.
    alpha : float
        J_freq weight in the ELBO.
    laplacian : DifferentiableLaplacian
        Pre-built Laplacian module (frozen topology + base weights).
    use_lambda_features : bool
        If True, concatenate the lambda-fingerprint of base_L to the
        encoder input.
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

        The three optional cache arguments (lambda_fp, spectral_cache,
        freq_eigvals) together eliminate all per-step O(N^3) CPU
        eigendecompositions from the training loop.  See train.py.

        Parameters
        ----------
        x : torch.Tensor
            Query node embeddings.  Shape (B, D).
        E : torch.Tensor
            Full embedding table.  Shape (N, D).
        node_idx : torch.Tensor or None
            Long tensor (B,) of query node indices within E.
        base_L : torch.Tensor or None
            Fixed base Laplacian (N, N) used as fallback for lambda_fp.
        lambda_fp : torch.Tensor or None
            Pre-computed lambda-fingerprint of base_L.  Shape (1, n_bins)
            or (B, n_bins); broadcast to (B, n_bins) automatically.
        spectral_cache : tuple(eigvals, eigvecs) or None
            Pre-computed eigendecomposition of base_L for TauModeDiffusion.
        freq_eigvals : torch.Tensor or None
            Pre-computed eigenvalues of base_L for J_freq.  Shape (N,).

        Returns
        -------
        dict with keys:
            loss, recon_loss, kl_loss, freq_loss, x_hat, L, z, mu, log_var
        """
        lam_fp = lambda_fp
        if self.encoder.use_lambda_features and lam_fp is None and base_L is not None:
            with torch.no_grad():
                lam_fp = lambda_fingerprint(base_L, tau_modes=self.tau_modes)
                lam_fp = lam_fp.view(1, -1).expand(x.shape[0], -1).contiguous()

        if lam_fp is not None and lam_fp.shape[0] != x.shape[0]:
            lam_fp = lam_fp.view(1, -1).expand(x.shape[0], -1).contiguous()

        z, mu, log_var = self.encoder(x, lambda_fp=lam_fp)
        L, _delta = self.wiring_decoder(z, node_idx=None)
        x_hat = self.diffusion_decoder(
            L, E, node_idx=node_idx, eig_cache=spectral_cache
        )

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
        Build a WiringAutoencoder (v1) from a parsed YAML config dict.

        Parameters
        ----------
        cfg : dict
            Parsed YAML config.  Must contain 'model' and 'graph' keys.
        E : torch.Tensor
            Embedding table (N, D).

        Returns
        -------
        WiringAutoencoder
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


# ---------------------------------------------------------------------------
# WiringAutoencoderV2  (v2 -- three-term ELBO, issue #27)
# ---------------------------------------------------------------------------

try:
    from .encoder import WiringEncoderV2
    from .wiring_decoder import SpectralLoadingDecoder
    from .spectral import spectral_basis_kl, tau_mode_kl
    _V2_IMPORTS_OK = True
except ImportError:
    _V2_IMPORTS_OK = False


class WiringAutoencoderV2(nn.Module):
    """
    Wiring Autoencoder v2 -- three-term ELBO with spectral mode priors.

    Assembles WiringEncoderV2, SpectralLoadingDecoder, and DiffusionDecoder
    under the three-term variational objective::

        L_VDTv2 = E_q[log p(x|z,W)]
                 - KL( q(z)  || N(0,I)          )   # kl_z  -- isotropic
                 - KL( q(S)  || p(S|I)          )   # kl_S  -- spectral basis
                 - KL( q(w)  || p(w|tau,Lambda) )   # kl_tau -- mode frequency

    Note: the Laplacian-precision latent KL (Term 2, kl_lap) has been
    removed per PR #35.  The L_z tensor is also not returned from forward()
    for the same reason.

    Data flow::

        x  (B, D),  U_q (N, q),  eigvals_q (q,)
          --> WiringEncoderV2
                z        (B, latent_dim)
                mu       (B, latent_dim)
                log_var  (B, latent_dim)
                log_a    (B, q)           -- Gamma shape log-params
                log_b    (B, q)           -- Gamma rate  log-params
          --> SpectralLoadingDecoder
                W        (B, d_model, q)  -- spectral loading matrix
                omega    (B, q)           -- mode weights
                S        (B, q, q)        -- rotation matrix
                L_z      (B, N, N)        -- synthesised Laplacian (internal)
          --> DiffusionDecoder
                x_hat    (B, D)
          --> three-term ELBO

    Parameters
    ----------
    input_dim : int
        D -- node embedding dimension.
    latent_dim : int
        Dimension of the isotropic VAE latent z.
    hidden_dim : int
        Hidden width for encoder trunk and diffusion decoder MLP.
    q : int
        Number of spectral modes; must match U_q.shape[1] at runtime.
    tau_modes : int
        Number of eigenvectors kept by DiffusionDecoder.
    lam_s : float
        Weight for kl_S (spectral basis KL).  Default 0.01.
    tau : float
        Diffusion time scale for kl_tau (mode frequency KL).  Default 0.5.
    laplacian : DifferentiableLaplacian
        Base Laplacian module shared with SpectralLoadingDecoder.
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
    ) -> None:
        if not _V2_IMPORTS_OK:
            raise ImportError(
                "WiringAutoencoderV2 requires WiringEncoderV2 and "
                "SpectralLoadingDecoder.  Ensure phase-1 modules are present."
            )
        super().__init__()
        self.q = q
        self.lam_s = lam_s
        self.tau = tau
        self.tau_modes = tau_modes

        self.encoder = WiringEncoderV2(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            q=q,
        )
        self.wiring_decoder = SpectralLoadingDecoder(
            latent_dim=latent_dim,
            q=q,
            laplacian=laplacian,
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
    ) -> dict[str, torch.Tensor]:
        """
        Single forward pass returning all three-term ELBO components.

        Parameters
        ----------
        x : torch.Tensor
            Query node embeddings.  Shape (B, D).
        U_q : torch.Tensor
            Leading q eigenvectors of the frozen index Laplacian L(I).
            Shape (N, q).  Passed to SpectralLoadingDecoder to synthesise
            L_z via DifferentiableLaplacian.from_spectral_loading.
        eigvals_q : torch.Tensor
            Corresponding q eigenvalues of L(I).  Shape (q,).
            Used by spectral_basis_kl and tau_mode_kl priors.
        node_idx : torch.Tensor or None
            Long tensor (B,) selecting one node per sample.  Required for
            per-node reconstruction loss; pass None only for diagnostics.
        spectral_cache : tuple(eigvals, eigvecs) or None
            Pre-computed full eigendecomposition of the base Laplacian for
            DiffusionDecoder.  Avoids an O(N^3) eigensolver at each step.

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
        # --- Encode -------------------------------------------------------
        # WiringEncoderV2.forward() returns (z, mu, log_var, log_a, log_b)
        z, mu, log_var, log_a, log_b = self.encoder(x)

        # --- Spectral decode (wiring) -------------------------------------
        # SpectralLoadingDecoder.forward() returns (W, omega, S, L_z)
        W, omega, S, L_z = self.wiring_decoder(z, U_q)

        # Embed the node-level embedding table from the laplacian's
        # base_weights context -- use the static embedding table passed in
        # via x (the caller owns E).  Build a per-batch E from x alone for
        # the diffusion step; callers wanting full-graph diffusion should
        # pass E separately.  For training we always use per-node mode.
        #
        # Convention: in v2 the caller passes x as both the query and the
        # embedding table for the per-node diffusion step.  Full-graph mode
        # requires an explicit E argument (see generate()).
        E = x  # (B, D) -- per-node path; node_idx is applied inside decoder

        # --- Diffusion decode ---------------------------------------------
        x_hat = self.diffusion_decoder(
            L_z, E, node_idx=node_idx, eig_cache=spectral_cache
        )  # (B, D)

        # --- ELBO terms ---------------------------------------------------
        # Term 1: reconstruction  E_q[log p(x|z,W)]
        recon = self.diffusion_decoder.recon_loss(x, x_hat)

        # Term 2: isotropic KL  KL(q(z) || N(0,I))
        # Standard closed-form: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
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
          - w_hat_k   is the k-th column of W_hat  (d_model,)
          - d_theta(.) is the decoder response at spectral direction w_hat_k
            (approximated as W_hat[:, k] for a linear decoder)
          - E[omega_k] = omega_hat[:, k].mean() for the scalar weight

        Shape:
          W_hat      (1, d_model, q)
          omega_hat  (q,)  -- averaged over the single-sample batch
          S_memory   (d_model, d_model)

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
        z_prior = torch.zeros(1, self.encoder.latent_dim, device=device)

        # Decode wiring at the prior mean
        W_hat, omega_raw, S, _L_z = self.wiring_decoder(z_prior, U_q)
        # W_hat : (1, d_model, q)
        # omega_raw : (1, q)

        omega_hat = omega_raw.squeeze(0)          # (q,)
        W = W_hat.squeeze(0)                      # (d_model, q)
        d_model = W.shape[0]

        # Build the outer-product Hopfield memory.
        # S_memory = sum_k omega_k * w_k w_k^T
        S_memory = torch.zeros(d_model, d_model, device=device)
        for k in range(self.q):
            w_k = W[:, k]                         # (d_model,)
            S_memory += omega_hat[k] * torch.outer(w_k, w_k)

        return {
            "W_hat": W_hat,          # (1, d_model, q)
            "omega_hat": omega_hat,  # (q,)
            "S_memory": S_memory,    # (d_model, d_model)
        }

    # ------------------------------------------------------------------
    # generate  (v1 interface preserved)
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
        z = torch.randn(n_samples, self.encoder.latent_dim, device=device)
        W, omega, S, L_z = self.wiring_decoder(z, U_q)
        return self.diffusion_decoder(L_z, E, node_idx=node_idx)


# ---------------------------------------------------------------------------
# from_config  --  unified factory dispatching on model.version
# ---------------------------------------------------------------------------

def from_config(
    cfg: dict[str, Any],
    E: torch.Tensor,
) -> "WiringAutoencoder | WiringAutoencoderV2":
    """
    Build the appropriate VDT model from a parsed YAML config dict.

    Dispatches on cfg['model']['version']:
      version: 1  (or absent)  ->  WiringAutoencoder     (v1)
      version: 2               ->  WiringAutoencoderV2   (v2)

    v2 YAML config example::

        model:
          version: 2
          latent_dim: 32
          q: 16
          lam_s: 0.01
          tau: 0.5
          decoder_type: spectral
        encoder:
          use_isotropic_kl: true

    v1 YAML config example::

        model:
          version: 1        # or omit; defaults to 1
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
        Parsed YAML config.  Must contain a 'model' key.
    E : torch.Tensor
        Embedding table (N, D).

    Returns
    -------
    WiringAutoencoder or WiringAutoencoderV2
    """
    mc = cfg["model"]
    version = int(mc.get("version", 1))

    if version == 2:
        gc = cfg.get("graph", {})
        lap = DifferentiableLaplacian.from_embeddings(
            E,
            knn_k=gc.get("knn_k", 15),
            sigma=gc.get("sigma", 0.5),
            normalised=gc.get("normalised", True),
            sparse=gc.get("sparse", False),
        )
        return WiringAutoencoderV2(
            input_dim=E.shape[1],
            latent_dim=mc["latent_dim"],
            hidden_dim=mc.get("hidden_dim", 256),
            q=mc["q"],
            tau_modes=mc.get("tau_modes", mc["q"]),
            lam_s=mc.get("lam_s", 0.01),
            tau=mc.get("tau", 0.5),
            laplacian=lap,
        )

    # version == 1 (default)
    gc = cfg["graph"]
    lap = DifferentiableLaplacian.from_embeddings(
        E,
        knn_k=gc["knn_k"],
        sigma=gc["sigma"],
        normalised=gc["normalised"],
        sparse=gc.get("sparse", False),
    )
    return WiringAutoencoder(
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
