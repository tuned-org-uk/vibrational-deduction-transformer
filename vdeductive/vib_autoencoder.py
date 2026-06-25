"""
vdeductive/vib_autoencoder.py  --  Deterministic Vibrational Autoencoder (Option 1).

LEGACY / ABLATION BASELINE
---------------------------
This module is NOT the canonical model.  The canonical variational model is
WiringAutoencoder in vdeductive/model.py.

This file is retained exclusively as an ablation baseline for:
  - Option 1 (#20, #30): deterministic AE regression check vs WiringAutoencoder.
  - Option 6 (#18, #29): classifier ablation that consumes the spectral
    artefact (W_hat, omega_hat, S_memory) produced by extract_spectral_artefact().

Do not import these classes in new production code.  Use WiringAutoencoder
from vdeductive/model.py instead.
---------------------------

Two classes are provided:

  VibrationalAutoencoder    v1, deterministic AE with J_freq hard penalty.
  DeterministicSpectralAE   , SpectralLoadingDecoder in deterministic mode
                                with spectral_penalty: hard | soft.

v1 objective::

    L = ||X0 - X_hat_0||_F^2 + alpha * J_freq(L(z)) + beta * R_M(z)

 objective (spectral_penalty: hard)::

    L = ||X0 - X_hat_0||_F^2 + alpha * spectral_freq_cost(L_z) + beta * R_M(z)

 objective (spectral_penalty: soft)::

    L = ||X0 - X_hat_0||_F^2 + alpha * tau_mode_kl(log_a, log_b, eigvals, tau)
                              + beta * R_M(z)

In deterministic mode omega is fixed to ones (no KL loss term).  The full
gradient path is::

    x -> VDT encoder -> z -> SpectralLoadingDecoder (omega=1) -> L_z
      -> DiffusionDecoder -> x_hat -> recon + spectral penalty

Post-training, call extract_spectral_artefact() to produce the artefact
dict (W_hat, omega_hat, S_memory) for downstream Option 6 use.

Ref: docs//03-branching.md -- Option 1
Depends on: vdeductive/vdeductive.py (#17), vdeductive/wiring_decoder.py (#26), vdeductive/spectral.py (#24)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

from .vdeductive import VDT
from .wiring_decoder import SpectralLoadingDecoder
from .diffusion_decoder import DiffusionDecoder
from .spectral import spectral_freq_cost, tau_mode_kl
from .metrics import spectral_entropy, active_modes


# ---------------------------------------------------------------------------
# VibrationalAutoencoder  (v1 -- retained from legacy spec)
# ---------------------------------------------------------------------------

class VibrationalAutoencoder(nn.Module):
    """
    Deterministic vibrational autoencoder (v1).

    Encodes x via a two-layer MLP + VDT bottleneck, then decodes through
    a WiringDecoder + DiffusionDecoder.  The objective adds a hard spectral
    frequency cost (J_freq) and an optional mass-matrix regulariser R_M.

    This class is the v1 reference implementation.  It must not be
    modified when  changes are introduced; its tests must continue to
    pass after DeterministicSpectralAE is added.

    Objective::

        L = ||X0 - X_hat||_F^2 / (B*D)
              + alpha * J_freq(L_z, tau_modes=tau_modes)
              + beta  * ||z||_2^2 / (B * latent_dim)   # R_M

    Parameters
    ----------
    input_dim : int
        D -- node embedding dimension.
    latent_dim : int
        Dimension of the deterministic latent code z.
    hidden_dim : int
        Hidden width for encoder MLP and diffusion decoder.
    tau_modes : int
        Number of eigenmodes kept for J_freq and DiffusionDecoder.
    alpha : float
        J_freq penalty weight.
    beta : float
        Mass-matrix regulariser weight.
    laplacian : DifferentiableLaplacian
        Pre-built Laplacian module.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        tau_modes: int,
        alpha: float,
        beta: float,
        laplacian,      # DifferentiableLaplacian -- avoid circular import
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.tau_modes  = tau_modes
        self.alpha      = alpha
        self.beta       = beta

        # Encoder: MLP -> VDT bottleneck
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Wiring decoder (v1 mixture-of-experts)
        from .wiring_decoder import WiringDecoder
        n_edges = laplacian.base_weights.shape[0]
        self.wiring_decoder = WiringDecoder(
            latent_dim=latent_dim,
            n_edges=n_edges,
            hidden_dim=hidden_dim,
            n_heads=4,
            laplacian=laplacian,
        )

        self.diffusion_decoder = DiffusionDecoder(
            embedding_dim=input_dim,
            hidden_dim=hidden_dim,
            tau_modes=tau_modes,
        )

    def forward(
        self,
        x: torch.Tensor,           # (B, D)
        E: torch.Tensor,           # (N, D)
        node_idx: Optional[torch.Tensor] = None,
        freq_eigvals: Optional[torch.Tensor] = None,
        spectral_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass returning all loss components.

        Parameters
        ----------
        x : Tensor  (B, D)  query node embeddings
        E : Tensor  (N, D)  full embedding table
        node_idx : Tensor (B,) or None
        freq_eigvals : Tensor (N,) or None  pre-computed for J_freq
        spectral_cache : (eigvals, eigvecs) or None  pre-computed for DiffusionDecoder

        Returns
        -------
        dict: loss, recon, freq_loss, reg_loss, x_hat, z, L
        """
        z   = self.encoder_mlp(x)                              # (B, latent_dim)
        L, _delta = self.wiring_decoder(z)
        x_hat = self.diffusion_decoder(
            L, E, node_idx=node_idx, eig_cache=spectral_cache
        )

        recon    = ((x - x_hat) ** 2).sum(dim=-1).mean()
        j_freq   = spectral_freq_cost(L, tau_modes=self.tau_modes,
                                       eigvals=freq_eigvals)
        reg_loss = (z ** 2).sum(dim=-1).mean()
        loss     = recon + self.alpha * j_freq + self.beta * reg_loss

        return {
            "loss":     loss,
            "recon":    recon,
            "freq_loss": j_freq,
            "reg_loss": reg_loss,
            "x_hat":    x_hat,
            "z":        z,
            "L":        L,
        }


# ---------------------------------------------------------------------------
# DeterministicSpectralAE  ( -- Option 1, issue #20)
# ---------------------------------------------------------------------------

class DeterministicSpectralAE(nn.Module):
    """
     deterministic autoencoder using SpectralLoadingDecoder (Option 1,
    issue #20).

    Replaces the v1 WiringDecoder with SpectralLoadingDecoder in
    deterministic mode: omega is fixed to ones after the forward pass so
    no KL loss term is added.  The spectral penalty term is selected by
    the ``spectral_penalty`` config flag::

        hard -> spectral_freq_cost  (v1 J_freq, hard eigenvalue penalty)
        soft -> tau_mode_kl with omega=ones treated as learned scalars

    The gradient path is fully intact through SpectralLoadingDecoder ->
    DifferentiableLaplacian.from_spectral_loading.

    Regression guarantee
    --------------------
    With spectral_penalty='hard' and a compatible eigenbasis U_q, the
    per-step reconstruction MSE must stay within 2% of the v1
    VibrationalAutoencoder baseline (acceptance criterion from #20).

    Post-training artefact
    ----------------------
    Call extract_spectral_artefact(U_q, eigvals_q) after training to
    produce the (W_hat, omega_hat, S_memory) dict that Option 6 (#18)
    consumes via SpectralAssociativeMemory.from_vdeductive().

    Parameters
    ----------
    input_dim : int
        D -- node embedding dimension.
    latent_dim : int
        q -- latent / spectral-mode dimension (= SpectralLoadingDecoder.q).
    hidden_dim : int
        Hidden width for encoder MLP and DiffusionDecoder.
    tau_modes : int
        Modes kept by DiffusionDecoder.
    alpha : float
        Spectral penalty weight.
    beta : float
        Mass-matrix regulariser weight (||z||^2 / (B * q)).
    tau : float
        Diffusion time for tau_mode_kl when spectral_penalty='soft'.
    spectral_penalty : str
        'hard' -> spectral_freq_cost  |  'soft' -> tau_mode_kl.
    use_density_bottleneck : bool
        If True, a SignedDensityMatrix is instantiated and its trace
        penalty is added as an additional regularisation term.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        tau_modes: int,
        alpha: float,
        beta: float,
        tau: float = 0.5,
        spectral_penalty: Literal["hard", "soft"] = "hard",
        use_density_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim          = latent_dim
        self.q                   = latent_dim   # alias -- q = latent_dim for 
        self.tau_modes           = tau_modes
        self.alpha               = alpha
        self.beta                = beta
        self.tau                 = tau
        self.spectral_penalty    = spectral_penalty
        self.use_density_bottleneck = use_density_bottleneck

        # Encoder: MLP -> q-dimensional latent code
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        #  spectral decoder -- deterministic mode (omega fixed to 1 at
        # inference; here omega_net weights are trained but post-hoc clamped)
        # SpectralLoadingDecoder takes (q, d); d = input_dim in this context
        self.spectral_decoder = SpectralLoadingDecoder(
            q=latent_dim,
            d=input_dim,
        )

        self.diffusion_decoder = DiffusionDecoder(
            embedding_dim=input_dim,
            hidden_dim=hidden_dim,
            tau_modes=tau_modes,
        )

        # Optional density bottleneck (N is not known at init; lazy)
        self._density = None
        if use_density_bottleneck:
            # Will be initialised on first forward() call
            pass

    def forward(
        self,
        x: torch.Tensor,               # (B, D)
        U_q: torch.Tensor,             # (D, q)  eigenvectors of base L
        L_base: torch.Tensor,          # (N, N)  frozen base topology
        E: torch.Tensor,               # (N, D)  full embedding table
        eigvals_q: Optional[torch.Tensor] = None,  # (q,) for soft penalty
        node_idx: Optional[torch.Tensor] = None,
        spectral_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the deterministic spectral AE.

        Parameters
        ----------
        x : Tensor  (B, D)
        U_q : Tensor  (D, q)
        L_base : Tensor  (N, N)
        E : Tensor  (N, D)
        eigvals_q : Tensor (q,) or None  -- required for spectral_penalty='soft'
        node_idx : Tensor (B,) or None
        spectral_cache : (eigvals, eigvecs) or None

        Returns
        -------
        dict: loss, recon, spectral_loss, reg_loss, x_hat, z, L_z,
              omega, S, W, H_lambda, active_mode_count
        """
        B, D = x.shape
        q    = self.q

        # --- Encode -------------------------------------------------------
        z = self.encoder_mlp(x)                       # (B, q)

        # --- Spectral decode (deterministic: omega fixed to 1) -----------
        W, omega_raw, S, L_z = self.spectral_decoder(z, U_q, L_base)
        # Deterministic mode: detach omega from the gradient path and
        # replace with ones.  omega_raw is still used for soft-penalty.
        omega_det = torch.ones_like(omega_raw)         # (B, q)  fixed

        # --- Diffusion decode --------------------------------------------
        x_hat = self.diffusion_decoder(
            L_z, E, node_idx=node_idx, eig_cache=spectral_cache
        )                                              # (B, D)

        # --- Reconstruction loss -----------------------------------------
        recon = ((x - x_hat) ** 2).sum(dim=-1).mean()

        # --- Spectral penalty --------------------------------------------
        if self.spectral_penalty == "hard":
            spectral_loss = spectral_freq_cost(
                L_z, tau_modes=self.tau_modes
            )
        else:  # soft
            if eigvals_q is None:
                raise ValueError(
                    "eigvals_q is required for spectral_penalty='soft'"
                )
            # For soft penalty: treat log(omega_raw) as log_a, zeros as log_b
            log_a = omega_raw.log().clamp(min=-10.0)
            log_b = torch.zeros_like(log_a)
            spectral_loss = tau_mode_kl(
                log_a, log_b, eigvals_q, tau=self.tau
            )

        # --- Mass-matrix regulariser -------------------------------------
        reg_loss = (z ** 2).sum(dim=-1).mean()

        # --- Optional density bottleneck ---------------------------------
        density_loss = x.new_zeros(1)
        if self.use_density_bottleneck:
            N = L_base.shape[0]
            if self._density is None:
                self._density = __import__(
                    "vdeductive.density", fromlist=["SignedDensityMatrix"]
                ).SignedDensityMatrix(n=N).to(x.device)
            rho = self._density.rho
            density_loss = self._density.trace_penalty()

        # --- Total loss --------------------------------------------------
        loss = (
            recon
            + self.alpha * spectral_loss
            + self.beta  * reg_loss
            + density_loss
        )

        # --- Diagnostics (no grad) ---------------------------------------
        with torch.no_grad():
            # Spectral entropy H(Lambda) using mean omega across batch
            omega_mean   = omega_raw.mean(dim=0)      # (q,)
            h_lambda     = spectral_entropy(omega_mean)
            n_active     = active_modes(omega_mean)

        return {
            "loss":              loss,
            "recon":             recon,
            "spectral_loss":     spectral_loss,
            "reg_loss":          reg_loss,
            "x_hat":             x_hat,
            "z":                 z,
            "L_z":               L_z,
            "omega":             omega_raw,
            "S":                 S,
            "W":                 W,
            "H_lambda":          h_lambda,
            "active_mode_count": n_active,
        }

    # ------------------------------------------------------------------
    # extract_spectral_artefact
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_spectral_artefact(
        self,
        U_q: torch.Tensor,         # (D, q)
        L_base: torch.Tensor,      # (N, N)
        eigvals_q: torch.Tensor,   # (q,)
    ) -> dict[str, torch.Tensor]:
        """
        Produce the spectral artefact consumed by SpectralAssociativeMemory
        and Option 6 ablation (#18, #29).

        Called at the prior mean z=0 to obtain the mean loading matrix
        W_hat and mode weights omega_hat.  Builds the outer-product
        Hopfield memory matrix S_memory = sum_k omega_k * w_k w_k^T.

        Parameters
        ----------
        U_q : Tensor  (D, q)
        L_base : Tensor  (N, N)
        eigvals_q : Tensor  (q,)

        Returns
        -------
        dict:
            W_hat      (1, D, q)
            omega_hat  (q,)
            S_memory   (D, D)
            eigvals_q  (q,)   -- echoed for downstream convenience
        """
        device = next(self.parameters()).device
        z_prior = torch.zeros(1, self.q, device=device)

        W_hat, omega_raw, _S, _L_z = self.spectral_decoder(
            z_prior, U_q.to(device), L_base.to(device)
        )
        # W_hat : (1, D, q)
        omega_hat = omega_raw.squeeze(0)          # (q,)
        W         = W_hat.squeeze(0)              # (D, q)
        D_dim, q  = W.shape

        S_memory = torch.zeros(D_dim, D_dim, device=device)
        for k in range(q):
            w_k = W[:, k]
            S_memory += omega_hat[k] * torch.outer(w_k, w_k)

        return {
            "W_hat":     W_hat,
            "omega_hat": omega_hat,
            "S_memory":  S_memory,
            "eigvals_q": eigvals_q.to(device),
        }
