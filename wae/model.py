"""
WiringAutoencoder — full model assembling all modules.

ELBO:
    ℒ(θ, φ; x, i) = E_{q_φ(z|x)}[log p_θ(x | z, i)]
                    - β · KL(q_φ(z|x) || p(z))
                    - α · J_freq(L(z))

where:
    x   ... raw embedding of query node i
    z   ... latent wiring code
    L(z)... learned Laplacian
    β, α.. KL and frequency regularisation weights from config
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal, Optional, Any

from .encoder import WiringEncoder
from .wiring_decoder import WiringDecoder
from .diffusion_decoder import DiffusionDecoder
from .laplacian import DifferentiableLaplacian
from .spectral import spectral_freq_cost, lambda_fingerprint


class WiringAutoencoder(nn.Module):
    """
    Full Wiring Autoencoder.

    Parameters
    ----------
    input_dim : int          D  — embedding dimension
    latent_dim : int         k  — latent code dimension
    hidden_dim : int         MLP hidden width
    n_wiring_heads : int     mixture heads in WiringDecoder
    tau_modes : int          eigenvectors retained in diffusion decoder
    beta : float             KL weight
    alpha : float            J_freq weight
    laplacian : DifferentiableLaplacian
    use_lambda_features : bool   enrich encoder with λ-fingerprint
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
        self.beta  = beta
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

        # Base Laplacian for λ-fingerprint (no gradient through this path)
        self._laplacian = laplacian

    # ------------------------------------------------------------------
    # Forward — return ELBO components
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,          # (B, D)  query embeddings
        E: torch.Tensor,          # (N, D)  full embedding table
        node_idx: Optional[torch.Tensor] = None,  # (B,)  query node indices
        base_L: Optional[torch.Tensor] = None,    # (N, N) fixed L for λ-fp
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys: loss, recon_loss, kl_loss, freq_loss, x_hat, L, z, mu, log_var

        x_hat shape
        -----------
        (B, D)  — per-node reconstruction when node_idx is provided (normal training).
        (B, N, D) — full-graph reconstruction when node_idx is None (diagnostic only;
                    recon_loss will still average over the B dimension but compares
                    only the first D-slice; pass node_idx during training).
        """
        # Optional λ-fingerprint enrichment for encoder
        lam_fp = None
        if self.encoder.use_lambda_features and base_L is not None:
            with torch.no_grad():
                lam_fp = lambda_fingerprint(base_L, tau_modes=self.tau_modes)
                # Repeat fingerprint for batch
                if lam_fp.dim() == 1:
                    lam_fp = lam_fp.unsqueeze(0).expand(x.shape[0], -1)

        # Encode
        z, mu, log_var = self.encoder(x, lambda_fp=lam_fp)  # (B, latent)

        # Wiring decode
        L, _delta = self.wiring_decoder(z)                  # (B, N, N)

        # Diffusion decode  — shape: (B, D) when node_idx given, else (B, N, D)
        x_hat = self.diffusion_decoder(L, E, node_idx=node_idx)

        # ELBO components
        recon  = self.diffusion_decoder.recon_loss(x, x_hat)
        kl     = WiringEncoder.kl_loss(mu, log_var)
        j_freq = spectral_freq_cost(L, tau_modes=self.tau_modes)

        loss = recon + self.beta * kl + self.alpha * j_freq

        return {
            "loss":       loss,
            "recon_loss": recon,
            "kl_loss":    kl,
            "freq_loss":  j_freq,
            "x_hat":      x_hat,
            "L":          L,
            "z":          z,
            "mu":         mu,
            "log_var":    log_var,
        }

    # ------------------------------------------------------------------
    # generate() — two explicit, stable modes
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        E: torch.Tensor,
        n_samples: int = 8,
        node_idx: Optional[torch.Tensor] = None,
        mode: Literal["per_node", "full_graph"] = "per_node",
    ) -> dict[str, torch.Tensor]:
        """
        Sample z ~ N(0, I) and decode to wiring + embeddings.

        Modes
        -----
        ``per_node`` (default)
            Reconstructs one node per sample, matching the training
            reconstruction path exactly.

            * If ``node_idx`` is provided it must have shape ``(n_samples,)``
              and selects which node to reconstruct for each sample.
            * If ``node_idx`` is None, node indices are drawn uniformly at
              random from ``{0, …, N-1}``.

            Output shapes::

                z        : (n_samples, latent_dim)
                L        : (n_samples, N, N)
                node_idx : (n_samples,)          ← always returned so callers
                                                    can trace which nodes were
                                                    reconstructed
                x_hat    : (n_samples, D)        ← stable, matches forward()

        ``full_graph``
            Reconstructs *all* N nodes simultaneously for each sample.
            The MLP refinement step is bypassed (it is per-node only).
            Useful for graph-level visualisation or probing the entire
            wiring space.

            Output shapes::

                z        : (n_samples, latent_dim)
                L        : (n_samples, N, N)
                node_idx : None
                x_hat    : (n_samples, N, D)

        Parameters
        ----------
        E : Tensor  (N, D)  embedding table
        n_samples : int     number of samples
        node_idx : Tensor or None
            Optional ``(n_samples,)`` long tensor of node indices.
            Only used in ``per_node`` mode; ignored in ``full_graph`` mode.
        mode : 'per_node' | 'full_graph'
            Generation semantics (see above).

        Returns
        -------
        dict with keys: z, L, node_idx, x_hat
        """
        if mode not in ("per_node", "full_graph"):
            raise ValueError(f"mode must be 'per_node' or 'full_graph', got {mode!r}")

        device = next(self.parameters()).device
        N = E.shape[0]

        z = torch.randn(n_samples, self.encoder.mu_head.out_features, device=device)
        L, _ = self.wiring_decoder(z)   # (n_samples, N, N)

        if mode == "per_node":
            if node_idx is None:
                node_idx = torch.randint(0, N, (n_samples,), device=device)
            else:
                node_idx = node_idx.to(device)
                if node_idx.shape != (n_samples,):
                    raise ValueError(
                        f"node_idx must have shape ({n_samples},), "
                        f"got {tuple(node_idx.shape)}"
                    )
            # Use DiffusionDecoder normally — MLP refinement applied, shape (B, D)
            x_hat = self.diffusion_decoder(L, E, node_idx=node_idx)

        else:  # full_graph
            # Bypass MLP refinement: it expects (B, D) input, not (B, N, D).
            # Call TauModeDiffusion directly with node_idx=None → (B, N, D).
            x_hat = self.diffusion_decoder.diffusion(L, E, node_idx=None)
            node_idx = None

        return {"z": z, "L": L, "node_idx": node_idx, "x_hat": x_hat}

    # ------------------------------------------------------------------
    # Factory — build WAE from config dict and embedding table
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        E: torch.Tensor,
    ) -> "WiringAutoencoder":
        """
        Convenience factory.

            wae = WiringAutoencoder.from_config(cfg, E)

        where cfg is the parsed YAML dict (see configs/default.yaml).
        """
        mc = cfg["model"]
        gc = cfg["graph"]

        lap = DifferentiableLaplacian.from_embeddings(
            E,
            knn_k=gc["knn_k"],
            sigma=gc["sigma"],
            normalised=gc["normalised"],
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
