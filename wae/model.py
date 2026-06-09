"""
WiringAutoencoder — full model assembling all modules.
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
        spectral_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        freq_eigvals: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        lam_fp = lambda_fp
        if self.encoder.use_lambda_features and lam_fp is None and base_L is not None:
            with torch.no_grad():
                lam_fp = lambda_fingerprint(base_L, tau_modes=self.tau_modes)
                lam_fp = lam_fp.expand(x.shape[0], -1).contiguous()
                if lam_fp.dim() == 1:
                    lam_fp = lam_fp.unsqueeze(0).expand(x.shape[0], -1)

        z, mu, log_var = self.encoder(x, lambda_fp=lam_fp)
        L, _delta = self.wiring_decoder(z, node_idx=None)
        x_hat = self.diffusion_decoder(L, E, node_idx=node_idx, eig_cache=spectral_cache)

        recon = self.diffusion_decoder.recon_loss(x, x_hat)
        kl = WiringEncoder.kl_loss(mu, log_var)
        j_freq = spectral_freq_cost(L, tau_modes=self.tau_modes, eigvals=freq_eigvals)
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
