"""
Molecular / Spectral Graph-Generation Demo  (resolves issue #13)
=================================================================

Demonstrates the VDT as a **generative spectral prior** for graph topology
and vibrational-mode modelling, using a synthetic spring-network dataset.

Conceptual grounding
--------------------
Rayleigh's Theory of Sound treats small oscillations of a coupled system via
the eigenvalue problem  K*u = omega^2 * M * u, where K is the stiffness
matrix and M the mass matrix.  For uniform masses, K is proportional to the
graph Laplacian L, and omega^2 are its eigenvalues -- the vibrational modes.

In the VDT framework:
  * Low-entropy state (ordered oscillation)  <->  few non-zero eigenvalues
    (smooth, low-frequency wiring).
  * High-entropy state (thermal rest)        <->  flat eigenvalue spectrum
    (dense, high-frequency wiring).

Three-term ELBO (v2, issue #13)
--------------------------------
The model optimises a three-term variational bound:

    L(theta, phi) = E_q[ log p(x|z) ]        (reconstruction)
                  - beta * KL( q(S) || p(S|I) )   (spectral-basis KL)
                  - beta * KL( q(omega) || Exp(tau*L) ) (mode-weight KL)

All three terms are logged per epoch and exported to training_log.csv.

Usage
-----
    python demos/spectral_generation_demo.py --n-graphs 400 --epochs 60
    python demos/spectral_generation_demo.py --help

Outputs (all written to results/spectral_demo/)
-----------------------------------------------
    spectral_demo_results.csv     -- per-sample generation metrics
    entropy_control_results.csv   -- entropy-targeting experiment
    training_log.csv              -- epoch-level ELBO / loss components
    figures/training_curves.png   -- recon + kl_S + kl_tau + total loss
    figures/entropy_distribution.png
    figures/spectral_distance.png
    figures/latent_entropy.png
    figures/mode_shapes.png
    figures/entropy_target_error.png
    figures/mode_weights.png      -- per-mode mean weight bar chart (NEW)
    figures/memory_heatmap.png    -- W_proj spectral memory heatmap (NEW)
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from vdt.metrics import spectral_entropy as v2_spectral_entropy
from vdt.metrics import memory_snr as v2_memory_snr
from vdt.metrics import active_modes as v2_active_modes
from vdt.metrics import compute_kl_S, compute_kl_tau

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===========================================================================
# 1.  SYNTHETIC SPRING-NETWORK DATASET
# ===========================================================================

def make_spring_graph(
    n_nodes: int, k_neighbours: int, sigma: float = 1.0, noise: float = 0.05
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Build one random spring-network graph.

    Nodes are placed on a circle (plus small Gaussian noise), connected to
    their k nearest neighbours.  Edge weights are RBF-kernel stiffness
    constants.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the ring.
    k_neighbours : int
        Base number of nearest neighbours to connect.
    sigma : float
        RBF bandwidth for edge-weight calculation.
    noise : float
        Standard deviation of node-position jitter.

    Returns
    -------
    L     : (N, N) normalised Laplacian (symmetric PSD)
    modes : (N, N) eigenvectors (columns = mode shapes)
    eigvals : (N,) eigenvalues in ascending order
    pos   : (N, 2) node positions
    """
    theta = torch.linspace(0, 2 * math.pi, n_nodes + 1)[:-1]
    pos   = torch.stack([theta.cos(), theta.sin()], dim=1)
    pos  += noise * torch.randn_like(pos)

    diff  = pos.unsqueeze(1) - pos.unsqueeze(0)      # (N, N, 2)
    dist2 = (diff ** 2).sum(-1)                       # (N, N)

    knn_idx = dist2.argsort(dim=1)[:, 1: k_neighbours + 1]  # (N, k)
    A = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in knn_idx[i]:
            w = math.exp(-dist2[i, j].item() / (2 * sigma ** 2))
            A[i, j] = w
            A[j, i] = w

    deg = A.sum(dim=1).clamp(min=1e-8)
    D_inv_sqrt = torch.diag(deg ** -0.5)
    L = torch.eye(n_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    L = (L + L.T) / 2

    eigvals, eigvecs = torch.linalg.eigh(L)
    return L, eigvecs, eigvals, pos


def build_dataset(
    n_graphs: int, n_nodes: int = 20, k_neighbours: int = 4,
    tau_modes: int = 6
) -> dict:
    """
    Generate n_graphs random spring-network graphs.

    Node features x_i are the i-th rows of the first tau_modes eigenvectors
    (i.e. the projection of node i onto the lowest vibrational mode shapes).

    Parameters
    ----------
    n_graphs : int
        Number of graphs to generate.
    n_nodes : int
        Nodes per graph.
    k_neighbours : int
        Maximum kNN degree.
    tau_modes : int
        Number of vibrational modes to use as node features.

    Returns
    -------
    dict with keys: X, X_graph, Ls, eigvals, graph_id,
                    n_graphs, n_nodes, tau_modes
    """
    Xs, Ls, Evs = [], [], []
    for g in range(n_graphs):
        k = random.randint(2, k_neighbours + 2)
        L, modes, eigvals, _ = make_spring_graph(n_nodes, k_neighbours=k)
        x = modes[:, :tau_modes]   # (N, tau_modes)
        Xs.append(x)
        Ls.append(L)
        Evs.append(eigvals)

    X     = torch.stack(Xs)            # (G, N, tau_modes)
    Ls_t  = torch.stack(Ls)            # (G, N, N)
    Evs_t = torch.stack(Evs)           # (G, N, N)
    graph_id = torch.arange(n_graphs).repeat_interleave(n_nodes)
    X_flat   = X.view(-1, tau_modes)
    return dict(X=X_flat, X_graph=X, Ls=Ls_t, eigvals=Evs_t,
                graph_id=graph_id, n_graphs=n_graphs,
                n_nodes=n_nodes, tau_modes=tau_modes)


# ===========================================================================
# 2.  MODEL (three-term ELBO, issue #13)
# ===========================================================================

class DiffLaplacian(nn.Module):
    """Differentiable normalised Laplacian from learned edge-weight logits."""

    def __init__(self, L_base: Tensor):
        super().__init__()
        N = L_base.shape[0]
        self.N = N
        idx = torch.triu_indices(N, N, offset=1)
        self.register_buffer("row", idx[0])
        self.register_buffer("col", idx[1])
        A_base  = torch.eye(N) - L_base
        A_base  = A_base.clamp(min=0)
        base_w  = A_base[idx[0], idx[1]].clamp(min=1e-6)
        self.log_base_w = nn.Parameter(base_w.log())

    def forward(self, delta: Tensor) -> Tensor:
        """
        Construct a batch of normalised Laplacians from per-edge logits.

        Parameters
        ----------
        delta : (E,) or (B, E) edge-weight logit adjustments

        Returns
        -------
        L : (N, N) or (B, N, N) normalised Laplacian
        """
        base_w  = self.log_base_w.exp()
        batched = delta.dim() == 2
        if not batched:
            delta = delta.unsqueeze(0)
        B = delta.shape[0]
        N = self.N
        w = base_w.unsqueeze(0) * torch.sigmoid(delta)  # (B, E)
        A = torch.zeros(B, N, N, device=w.device)
        A[:, self.row, self.col] = w
        A[:, self.col, self.row] = w
        deg        = A.sum(-1).clamp(min=1e-8)
        D_inv_sqrt = torch.diag_embed(deg ** -0.5)
        I          = torch.eye(N, device=w.device).unsqueeze(0)
        L          = I - D_inv_sqrt @ A @ D_inv_sqrt
        L          = (L + L.transpose(-1, -2)) / 2
        return L.squeeze(0) if not batched else L


def _spectral_entropy_batch(eigvals: Tensor, eps: float = 1e-8) -> Tensor:
    """Batched Shannon entropy of normalised eigenvalue distribution."""
    lam = eigvals.clamp(min=eps)
    p   = lam / lam.sum(dim=-1, keepdim=True)
    return -(p * (p + eps).log()).sum(dim=-1)


def freq_cost(L: Tensor, tau_modes: int) -> Tensor:
    """Spectral frequency cost: sum of eigenvalues beyond the tau_modes-th."""
    eigvals = torch.linalg.eigvalsh(L).clamp(min=0)  # (B, N)
    return eigvals[:, tau_modes:].sum(dim=-1).mean()


class Encoder(nn.Module):
    """
    Amortised encoder for the SpectralVDT three-term ELBO.

    Outputs the standard VAE parameters (mu, log_var, z) plus the
    spectral-loading posterior (S_mean, log_var_S) and mode-weight
    posterior (log_a, log_b) needed for kl_S and kl_tau respectively.

    Parameters
    ----------
    input_dim : int
        Node feature dimension (tau_modes).
    latent_dim : int
        VAE latent dimension.
    tau_modes : int
        Number of spectral modes to parameterise.
    hidden : int
        Hidden layer width.
    """

    def __init__(self, input_dim: int, latent_dim: int, tau_modes: int,
                 hidden: int = 128):
        super().__init__()
        self.tau_modes = tau_modes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.GELU(),
        )
        self.mu_head      = nn.Linear(hidden, latent_dim)
        self.log_var_head = nn.Linear(hidden, latent_dim)
        # Spectral loading posterior
        s_dim = tau_modes * tau_modes
        self.S_mu_head    = nn.Linear(hidden, s_dim)
        self.S_lv_head    = nn.Linear(hidden, s_dim)
        # Mode-weight posterior (Gamma-like: parameterised by log_a, log_b)
        self.la_head      = nn.Linear(hidden, tau_modes)
        self.lb_head      = nn.Linear(hidden, tau_modes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h   = self.net(x)
        mu  = self.mu_head(h)
        lv  = self.log_var_head(h)
        std = (0.5 * lv).exp()
        z   = mu + std * torch.randn_like(std)
        B, q = x.shape[0], self.tau_modes
        S_mean = self.S_mu_head(h).view(B, q, q)
        S_lv   = self.S_lv_head(h).view(B, q, q)
        log_a  = self.la_head(h)   # (B, q)
        log_b  = self.lb_head(h)   # (B, q)
        return dict(z=z, mu=mu, log_var=lv,
                    S_mean=S_mean, log_var_S=S_lv,
                    log_a=log_a, log_b=log_b)

    @staticmethod
    def kl_gaussian(mu: Tensor, log_var: Tensor) -> Tensor:
        """Standard diagonal-Gaussian KL: KL( N(mu, sigma^2) || N(0,1) )."""
        return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()


class WiringDecoder(nn.Module):
    """Maps z to per-edge delta logits, then builds L(z) via DiffLaplacian."""

    def __init__(self, latent_dim: int, n_edges: int, hidden: int = 128,
                 n_heads: int = 4, laplacian: DiffLaplacian = None):
        super().__init__()
        self.lap   = laplacian
        self.heads = nn.ModuleList([
            nn.Linear(latent_dim, n_edges) for _ in range(n_heads)
        ])
        self.gate  = nn.Linear(latent_dim, n_heads)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (L, delta) where L is (B,N,N) and delta is (B,E)."""
        gates  = F.softmax(self.gate(z), dim=-1)                     # (B, H)
        deltas = torch.stack([h(z) for h in self.heads], dim=1)      # (B, H, E)
        delta  = (gates.unsqueeze(-1) * deltas).sum(dim=1)           # (B, E)
        return self.lap(delta), delta


class DiffusionDecoder(nn.Module):
    """L(z), E -> x_hat via tau-mode truncated heat kernel."""

    def __init__(self, tau_modes: int, feat_dim: int, out_dim: int,
                 learnable_t: bool = True):
        super().__init__()
        self.tau   = tau_modes
        self.log_t = nn.Parameter(torch.zeros(1)) if learnable_t else None
        self.mlp   = nn.Sequential(
            nn.Linear(feat_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(
        self, L: Tensor, E: Tensor, node_idx: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Diffuse node embeddings through the spectral heat kernel.

        Parameters
        ----------
        L        : (B, N, N) batch of Laplacians
        E        : (N, D) node embedding table
        node_idx : (B,) optional node indices for row selection

        Returns
        -------
        x_hat : (B, D) or (B, N, D) decoded features
        K     : (B, N, N) truncated heat kernel
        """
        B, N, _ = L.shape
        eigvals, eigvecs = torch.linalg.eigh(L)          # (B,N), (B,N,N)
        lam_k = eigvals[:, :self.tau]                    # (B, k)
        U_k   = eigvecs[:, :, :self.tau]                 # (B, N, k)
        t     = self.log_t.exp() if self.log_t is not None else torch.ones(1, device=L.device)
        heat  = torch.exp(-t * lam_k)                   # (B, k)
        K     = (U_k * heat.unsqueeze(1)) @ U_k.transpose(-1, -2)  # (B, N, N)
        feat  = torch.bmm(K, E.unsqueeze(0).expand(B, -1, -1))     # (B, N, D)
        if node_idx is not None:
            feat = feat[torch.arange(B, device=L.device), node_idx]  # (B, D)
        return self.mlp(feat), K


class SpectralVDT(nn.Module):
    """
    Spectral Wiring Autoencoder with three-term ELBO (issue #13).

    The three ELBO terms are:

        term_recon  = E_q[ log p(x | z) ]          (node feature reconstruction)
        term_kl_S   = KL( q(S) || p(S | I) )       (spectral loading matrix)
        term_kl_tau = KL( q(omega) || Exp(tau*L) ) (mode weight posterior)

    The total loss is:  term_recon + beta * (term_kl_S + term_kl_tau)

    Parameters
    ----------
    feat_dim : int
        Node feature dimension (= tau_modes).
    latent_dim : int
        VAE latent dimension.
    n_nodes : int
        Number of nodes per graph.
    tau_modes : int
        Number of spectral modes retained by the decoder.
    L_base : Tensor (N, N)
        Mean training Laplacian used to initialise DiffLaplacian edge weights.
    beta : float
        Weight on the combined KL term.
    alpha : float
        Weight on the frequency regularisation cost J_freq.
    hidden : int
        MLP hidden layer width.
    n_heads : int
        Number of mixture heads in WiringDecoder.
    """

    def __init__(
        self,
        feat_dim: int, latent_dim: int, n_nodes: int, tau_modes: int,
        L_base: Tensor,
        beta: float = 1.0, alpha: float = 0.1,
        hidden: int = 128, n_heads: int = 4,
    ):
        super().__init__()
        n_edges = n_nodes * (n_nodes - 1) // 2
        self.laplacian = DiffLaplacian(L_base)
        self.encoder   = Encoder(feat_dim, latent_dim, tau_modes, hidden)
        self.wdecoder  = WiringDecoder(latent_dim, n_edges, hidden, n_heads,
                                       laplacian=self.laplacian)
        self.ddecoder  = DiffusionDecoder(tau_modes, feat_dim, feat_dim)
        self.beta, self.alpha = beta, alpha
        self.tau = tau_modes
        # W_proj: (latent_dim, tau_modes) -- spectral memory projection
        self.W_proj = nn.Parameter(
            torch.randn(latent_dim, tau_modes) / math.sqrt(latent_dim)
        )

    def forward(self, x: Tensor, E: Tensor) -> Dict[str, Tensor]:
        """
        Full forward pass with three-term ELBO.

        Parameters
        ----------
        x : (B, D) node features
        E : (N, D) or (B, N, D) node embedding table

        Returns
        -------
        dict with keys: loss, recon, kl_S, kl_tau, jfreq,
                        x_hat, L, z, mu, log_var
        """
        B = x.shape[0]
        enc = self.encoder(x)
        z, mu, lv = enc["z"], enc["mu"], enc["log_var"]
        L, delta  = self.wdecoder(z)                    # (B, N, N)
        x_hat, K  = self.ddecoder(L, E)                 # (B, N, D)
        recon = F.mse_loss(x_hat, E.unsqueeze(0).expand(B, -1, -1))

        # Frozen spectral basis from mean L for KL computation
        with torch.no_grad():
            eigvals_all, _ = torch.linalg.eigh(L.detach().mean(0))
            eigvals_q = eigvals_all[:self.tau].to(x.device)

        kl_s  = compute_kl_S(enc["S_mean"], enc["log_var_S"], eigvals_q)
        kl_t  = compute_kl_tau(enc["log_a"], enc["log_b"], eigvals_q)
        jf    = freq_cost(L, self.tau)
        loss  = recon + self.beta * (kl_s + kl_t) + self.alpha * jf

        return dict(
            loss=loss, recon=recon, kl_S=kl_s, kl_tau=kl_t, jfreq=jf,
            x_hat=x_hat, L=L, z=z, mu=mu, log_var=lv,
            log_a=enc["log_a"], log_b=enc["log_b"],
        )

    @torch.no_grad()
    def generate(
        self, E: Tensor, n_samples: int = 1, z: Tensor = None
    ) -> dict:
        """
        Sample novel wirings from the prior N(0, I).

        Parameters
        ----------
        E         : (N, D) embedding table
        n_samples : int  number of samples when z is not provided
        z         : (S, latent_dim) optional pre-sampled latent codes

        Returns
        -------
        dict: z, L, x_hat, eigvals, entropy
        """
        if z is None:
            z = torch.randn(
                n_samples, self.encoder.mu_head.out_features, device=E.device
            )
        L, _     = self.wdecoder(z)
        x_hat, _ = self.ddecoder(L, E)
        eigvals  = torch.linalg.eigvalsh(L)
        H        = _spectral_entropy_batch(eigvals)
        return dict(z=z, L=L, x_hat=x_hat, eigvals=eigvals, entropy=H)

    @torch.no_grad()
    def mode_weights(self, n_samples: int = 256, device: str = "cpu") -> Tensor:
        """
        Estimate per-mode mean weight E[omega_k] by sampling from N(0,I).

        Samples n_samples latent codes, decodes each to a Laplacian, and
        computes the normalised eigenvalue distribution (soft mode weights).
        Returns the mean across samples.

        Parameters
        ----------
        n_samples : int  number of prior samples to average over
        device    : str

        Returns
        -------
        omega_bar : (tau_modes,) mean mode weights in [0, 1]
        """
        z   = torch.randn(n_samples, self.encoder.mu_head.out_features).to(device)
        L, _ = self.wdecoder(z)
        eigvals = torch.linalg.eigvalsh(L).clamp(min=0)  # (S, N)
        lam_k   = eigvals[:, :self.tau]                   # (S, tau)
        total   = lam_k.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (lam_k / total).mean(dim=0).cpu()          # (tau,)


# ===========================================================================
# 3.  TRAINING LOOP
# ===========================================================================

def train(
    model: SpectralVDT, loader: DataLoader,
    optimizer: torch.optim.Optimizer, device: str
) -> Dict[str, float]:
    """One training epoch; returns mean loss components."""
    model.train()
    totals: Dict[str, float] = dict(loss=0., recon=0., kl_S=0., kl_tau=0., jfreq=0.)
    n = 0
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        out = model(x_batch, x_batch)
        optimizer.zero_grad()
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for k in totals:
            totals[k] += out[k].item() * x_batch.shape[0]
        n += x_batch.shape[0]
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: SpectralVDT, loader: DataLoader, device: str
) -> Dict[str, float]:
    """One evaluation pass; returns mean loss components."""
    model.eval()
    totals: Dict[str, float] = dict(loss=0., recon=0., kl_S=0., kl_tau=0., jfreq=0.)
    n = 0
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        out = model(x_batch, x_batch)
        for k in totals:
            totals[k] += out[k].item() * x_batch.shape[0]
        n += x_batch.shape[0]
    return {k: v / n for k, v in totals.items()}


# ===========================================================================
# 4.  SPECTRAL-DISTANCE METRICS
# ===========================================================================

def spectral_distance(L_a: Tensor, L_b: Tensor) -> Tensor:
    """Frobenius distance between Laplacians.  Batch-safe."""
    diff = L_a - L_b
    return (diff * diff).sum(dim=(-1, -2)).sqrt()


# ===========================================================================
# 5.  ENTROPY-CONTROLLED GENERATION
# ===========================================================================

def generate_at_target_entropy(
    model: SpectralVDT, E: Tensor, target_H: float,
    n_candidates: int = 500, tol: float = 0.05
) -> dict:
    """
    Sample latent codes from N(0,I) and keep those closest to target_H.

    Parameters
    ----------
    model       : SpectralVDT in eval mode
    E           : (N, D) embedding table
    target_H    : float  target spectral entropy
    n_candidates : int  number of prior samples to draw
    tol         : float  tolerance window for match-rate calculation

    Returns
    -------
    dict: target, mean_entropy, std_entropy, match_rate, best_err,
          best_L, best_eigvals
    """
    gen = model.generate(E, n_samples=n_candidates)
    H   = gen["entropy"]
    err = (H - target_H).abs()
    return dict(
        target=target_H,
        mean_entropy=H.mean().item(),
        std_entropy=H.std().item(),
        match_rate=(err < tol).float().mean().item(),
        best_err=err.min().item(),
        best_L=gen["L"][err.argmin().item()],
        best_eigvals=gen["eigvals"][err.argmin().item()],
    )


# ===========================================================================
# 6.  FIGURES  (extended for issue #13: mode_weights + memory_heatmap)
# ===========================================================================

def save_figures(
    out_dir: Path, train_log: List[dict], gen_results: dict,
    entropy_results: List[dict], dataset: dict,
    model: SpectralVDT, device: str,
) -> None:
    """
    Save all demo figures.

    New outputs added for issue #13:
      * mode_weights.png   -- bar chart of per-mode mean weight from prior samples
      * memory_heatmap.png -- W_proj (latent_dim x tau_modes) spectral memory heatmap

    Parameters
    ----------
    out_dir        : Path  output directory root
    train_log      : list of per-epoch dicts with loss components
    gen_results    : dict from the generation phase
    entropy_results : list from entropy-targeting experiment
    dataset        : original dataset dict
    model          : trained SpectralVDT
    device         : device string
    """
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    epochs = [r["epoch"] for r in train_log]

    # --- (A) Three-term ELBO training curves ---
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, key, title in zip(
        axes,
        ["train_loss",  "train_recon", "train_kl_S",
         "train_kl_tau", "train_jfreq"],
        ["Total Loss",  "Recon MSE",   "KL_S (spectral basis)",
         "KL_tau (mode weights)", "J_freq spectral cost"],
    ):
        val_key = "val_" + key[6:]
        ax.plot(epochs, [r[key]    for r in train_log], label="train", lw=2)
        ax.plot(epochs, [r[val_key] for r in train_log], label="val",
                lw=2, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Three-term ELBO Training Curves", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(figs / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- (B) Spectral entropy distribution ---
    H_vals = gen_results["entropies"]
    H_data = gen_results["data_entropies"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(H_data, bins=30, alpha=0.6, color="#01696f", label="Dataset Laplacians")
    ax.hist(H_vals, bins=30, alpha=0.6, color="#964219", label="Generated Laplacians")
    ax.set_xlabel("Spectral Entropy H(Lambda)", fontsize=11)
    ax.set_title("Spectral Entropy: Dataset vs VDT Generated", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "entropy_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- (C) Spectral distance ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gen_results["spectral_distances"], bins=30, color="#a12c7b", alpha=0.85)
    ax.axvline(
        gen_results["mean_spectral_dist"], color="#a12c7b",
        linestyle="--", lw=2,
        label=f"mean = {gen_results['mean_spectral_dist']:.3f}",
    )
    ax.set_xlabel("Frobenius distance ||L_nearest - L_gen||_F", fontsize=10)
    ax.set_title("Spectral Distance to Nearest Training Laplacian", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "spectral_distance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- (D) Latent entropy scatter (PCA) ---
    try:
        from sklearn.decomposition import PCA  # type: ignore
        Z  = gen_results["latent_z"].numpy()
        H  = gen_results["entropies"]
        Z2 = PCA(n_components=2).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=H, cmap="viridis", s=15, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Spectral Entropy")
        ax.set_title("VDT Latent Space coloured by Spectral Entropy", fontsize=12)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(figs / "latent_entropy.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass

    # --- (E) Sample vibrational mode shapes ---
    N   = dataset["n_nodes"]
    tau = dataset["tau_modes"]
    sample_L = dataset["Ls"][0]
    _, sample_modes = torch.linalg.eigh(sample_L)
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    n_plot = min(4, tau)
    fig, axes = plt.subplots(1, n_plot, figsize=(14, 3))
    if n_plot == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.plot(theta.numpy(), sample_modes[:, idx].numpy(),
                lw=2, color=cm.viridis(idx / tau))
        ax.set_title(f"Mode {idx + 1}  (lambda={dataset['eigvals'][0, idx]:.3f})")
        ax.set_xlabel("theta (node position)"); ax.grid(alpha=0.3)
    plt.suptitle("Vibrational Mode Shapes of Sample Spring Network", y=1.02)
    plt.tight_layout()
    plt.savefig(figs / "mode_shapes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- (F) Entropy target error ---
    targets  = [r["target"]     for r in entropy_results]
    best_err = [r["best_err"]   for r in entropy_results]
    match_rt = [r["match_rate"] for r in entropy_results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(targets, best_err, marker="o", color="#01696f", lw=2)
    ax1.set_xlabel("Target Spectral Entropy", fontsize=11)
    ax1.set_ylabel("|H_gen - H_target|", fontsize=11)
    ax1.set_title("Entropy Targeting: Best-of-500 Error", fontsize=12)
    ax1.grid(alpha=0.3)
    ax2.plot(targets, match_rt, marker="s", color="#964219", lw=2)
    ax2.set_xlabel("Target Spectral Entropy", fontsize=11)
    ax2.set_ylabel("Match Rate  (|err| < 0.05)", fontsize=11)
    ax2.set_title("Entropy Targeting: Match Rate", fontsize=12)
    ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "entropy_target_error.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- (G) NEW: Per-mode mean weight bar chart ---
    model.eval()
    omega_bar = model.mode_weights(n_samples=256, device=device)  # (tau_modes,)
    tau_modes = omega_bar.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, tau_modes * 0.8), 4))
    mode_ids = list(range(1, tau_modes + 1))
    bar_colors = [cm.viridis(i / max(tau_modes - 1, 1)) for i in range(tau_modes)]
    bars = ax.bar(mode_ids, omega_bar.numpy(), color=bar_colors, edgecolor="white")
    ax.set_xlabel("Spectral Mode k", fontsize=11)
    ax.set_ylabel("Mean weight E[omega_k]", fontsize=11)
    ax.set_title(
        "Per-mode Mean Weight from Prior Samples\n"
        "(normalised eigenvalue distribution averaged over N(0,I) prior)",
        fontsize=10,
    )
    ax.set_xticks(mode_ids)
    for bar, val in zip(bars, omega_bar.numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "mode_weights.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- (H) NEW: W_proj spectral memory heatmap ---
    W = model.W_proj.detach().cpu().numpy()   # (latent_dim, tau_modes)
    fig, ax = plt.subplots(
        figsize=(max(5, tau_modes * 0.6), max(5, W.shape[0] * 0.15))
    )
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r",
                   vmin=-abs(W).max(), vmax=abs(W).max())
    plt.colorbar(im, ax=ax, label="Weight magnitude")
    ax.set_xlabel("Spectral Mode k", fontsize=11)
    ax.set_ylabel("Latent dimension j", fontsize=11)
    ax.set_title(
        "W_proj Spectral Memory Heatmap\n"
        "(rows = latent dims, cols = spectral modes)",
        fontsize=10,
    )
    ax.set_xticks(range(tau_modes))
    ax.set_xticklabels([f"k={i+1}" for i in range(tau_modes)], fontsize=8)
    plt.tight_layout()
    plt.savefig(figs / "memory_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[demo] Figures saved to {figs}/")


# ===========================================================================
# 7.  EXPORT CSV
# ===========================================================================

def export_csvs(
    out_dir: Path, train_log: List[dict], gen_results: dict,
    entropy_results: List[dict]
) -> None:
    """
    Export training log and generation results to CSV.

    training_log.csv now includes kl_S and kl_tau columns in addition
    to the original recon, jfreq, and total loss columns.
    """
    # Training log (includes all three ELBO terms)
    with open(out_dir / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=train_log[0].keys())
        w.writeheader(); w.writerows(train_log)

    # Per-sample generation metrics
    n = len(gen_results["entropies"])
    rows = [
        dict(
            sample_id=i,
            spectral_entropy=float(gen_results["entropies"][i]),
            spectral_distance=(
                float(gen_results["spectral_distances"][i])
                if i < len(gen_results["spectral_distances"])
                else float("nan")
            ),
        )
        for i in range(n)
    ]
    with open(out_dir / "spectral_demo_results.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["sample_id", "spectral_entropy", "spectral_distance"]
        )
        w.writeheader(); w.writerows(rows)

    # Entropy control results
    with open(out_dir / "entropy_control_results.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["target", "mean_entropy", "std_entropy",
                           "match_rate", "best_err"]
        )
        w.writeheader()
        w.writerows([
            {k: r[k] for k in ["target", "mean_entropy", "std_entropy",
                                "match_rate", "best_err"]}
            for r in entropy_results
        ])
    print(f"[demo] CSVs saved to {out_dir}/")


# ===========================================================================
# 8.  MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectral Graph-Generation Demo for VDT (issue #13)"
    )
    parser.add_argument("--n-graphs",   type=int,   default=400)
    parser.add_argument("--n-nodes",    type=int,   default=16)
    parser.add_argument("--k-neigh",    type=int,   default=3)
    parser.add_argument("--tau-modes",  type=int,   default=6)
    parser.add_argument("--latent-dim", type=int,   default=12)
    parser.add_argument("--hidden",     type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--beta",       type=float, default=0.5)
    parser.add_argument("--alpha",      type=float, default=0.05)
    parser.add_argument("--n-gen",      type=int,   default=500)
    parser.add_argument("--output",     default="results/spectral_demo")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = args.device

    print("[demo] Building spring-network dataset...")
    dataset = build_dataset(
        args.n_graphs, args.n_nodes, args.k_neigh, args.tau_modes
    )
    N = args.n_nodes
    D = args.tau_modes

    L_base = dataset["Ls"].mean(0)  # (N, N)
    print(
        f"[demo] Dataset: {args.n_graphs} graphs x {N} nodes, "
        f"feat_dim={D}, tau_modes={args.tau_modes}"
    )

    X_all  = dataset["X"]                      # (G*N, D)
    idx_all = torch.arange(X_all.shape[0])
    ds = TensorDataset(X_all, idx_all)
    n_train  = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    print("[demo] Building SpectralVDT (three-term ELBO)...")
    model = SpectralVDT(
        feat_dim=D, latent_dim=args.latent_dim, n_nodes=N,
        tau_modes=args.tau_modes, L_base=L_base,
        beta=args.beta, alpha=args.alpha,
        hidden=args.hidden, n_heads=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    print(f"[demo] Training for {args.epochs} epochs...")
    train_log: List[dict] = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t = train(model, train_loader, optimizer, device)
        v = evaluate(model, val_loader, device)
        scheduler.step()
        row = dict(epoch=epoch)
        row.update({"train_" + k: tv for k, tv in t.items()})
        row.update({"val_"   + k: vv for k, vv in v.items()})
        train_log.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}  "
                f"loss={t['loss']:.4f}  "
                f"recon={t['recon']:.4f}  "
                f"kl_S={t['kl_S']:.4f}  "
                f"kl_tau={t['kl_tau']:.4f}  "
                f"jfreq={t['jfreq']:.4f}"
            )
        if v["loss"] < best_val_loss:
            best_val_loss = v["loss"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    print("[demo] Generating novel wirings...")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    model.eval()

    E_mean = dataset["X_graph"].mean(0).to(device)   # (N, D)
    gen    = model.generate(E_mean, n_samples=args.n_gen)

    Ls_data = dataset["Ls"].to(device)               # (G, N, N)
    sp_dists = [
        spectral_distance(
            Ls_data, gen["L"][i].unsqueeze(0).expand_as(Ls_data)
        ).min().item()
        for i in range(gen["L"].shape[0])
    ]

    data_eig = torch.linalg.eigvalsh(Ls_data)
    H_data   = _spectral_entropy_batch(data_eig).tolist()

    gen_results = dict(
        entropies=gen["entropy"].tolist(),
        spectral_distances=sp_dists,
        mean_spectral_dist=float(np.mean(sp_dists)),
        data_entropies=H_data,
        latent_z=gen["z"].cpu(),
    )

    print("[demo] Entropy-controlled generation experiment...")
    H_range = np.linspace(
        float(np.percentile(H_data, 10)),
        float(np.percentile(H_data, 90)),
        num=10,
    )
    entropy_results: List[dict] = []
    for target_H in H_range:
        res = generate_at_target_entropy(
            model, E_mean, float(target_H),
            n_candidates=500, tol=0.05,
        )
        entropy_results.append(res)
        print(
            f"  target H={target_H:.3f}  "
            f"best_err={res['best_err']:.4f}  "
            f"match_rate={res['match_rate']:.3f}"
        )

    print("[demo] Exporting CSVs and figures...")
    export_csvs(out_dir, train_log, gen_results, entropy_results)
    save_figures(out_dir, train_log, gen_results, entropy_results,
                 dataset, model, device)

    # v2 metrics summary via vdt.metrics
    omega_bar  = model.mode_weights(n_samples=256, device=device)
    n_active   = v2_active_modes(omega_bar, delta=0.01)
    W_keys     = model.W_proj.T.detach().cpu()  # (tau_modes, latent_dim)
    snr        = v2_memory_snr(W_keys)
    H_gen_mean = float(np.mean(gen_results["entropies"]))
    H_gen_std  = float(np.std(gen_results["entropies"]))

    # Print summary
    print("\n" + "=" * 62)
    print("DEMO SUMMARY (v2 metrics)")
    print("=" * 62)
    print(f"  Dataset            : {args.n_graphs} graphs, {N} nodes")
    print(f"  Best val ELBO loss : {best_val_loss:.5f}")
    print(f"  Mean spectral dist : {gen_results['mean_spectral_dist']:.4f}")
    print(f"  Generated H(Lambda): {H_gen_mean:.3f} +/- {H_gen_std:.3f}")
    print(f"  Active modes       : {n_active} / {args.tau_modes}")
    print(f"  Memory SNR (W_proj): {snr:.4f}")
    mean_match = float(np.mean([r['match_rate'] for r in entropy_results]))
    print(f"  Entropy match rate (tol=0.05): {mean_match:.3f}")
    print("=" * 62)
    print(f"  Results: {out_dir}/")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
