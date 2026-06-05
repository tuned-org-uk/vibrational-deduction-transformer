"""
Molecular / Spectral Graph-Generation Demo  (resolves issue #13)
=================================================================

Demonstrates the WAE as a **generative spectral prior** for graph topology
and vibrational-mode modelling, using a synthetic spring-network dataset.

Conceptual grounding
--------------------
Rayleigh's Theory of Sound treats small oscillations of a coupled system via
the eigenvalue problem  K·u = ω²·M·u, where K is the stiffness matrix and
M the mass matrix.  For uniform masses, K is proportional to the graph
Laplacian L, and ω² are its eigenvalues — the *vibrational modes*.

In the WAE framework:
  • Low-entropy state (ordered oscillation)  ↔  few non-zero eigenvalues
    (smooth, low-frequency wiring).
  • High-entropy state (thermal rest / collapsed wave-function)  ↔  flat
    eigenvalue spectrum (dense, high-frequency wiring).

The demo trains on *node feature vectors derived from vibrational mode shapes*
(eigenvectors of L), then shows that the WAE latent space encodes spectral
entropy, and that we can sample novel wirings matching a target entropy.

Usage
-----
    python demos/spectral_generation_demo.py --n-graphs 400 --epochs 60
    python demos/spectral_generation_demo.py --help

Outputs (all written to results/spectral_demo/)
-----------------------------------------------
    spectral_demo_results.csv     — per-sample generation metrics
    entropy_control_results.csv   — entropy-targeting experiment
    training_log.csv              — epoch-level ELBO / loss components
    figures/training_curves.png
    figures/entropy_distribution.png
    figures/spectral_distance.png
    figures/latent_entropy.png
    figures/mode_shapes.png
    figures/entropy_target_error.png
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def make_spring_graph(n_nodes: int, k_neighbours: int, sigma: float = 1.0,
                      noise: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build one random spring-network graph.

    Nodes are placed on a circle (plus small noise), connected to their
    k nearest neighbours.  Edge weights are RBF-kernel stiffness constants.

    Returns
    -------
    L : (N, N) normalised Laplacian (symmetric PSD)
    modes : (N, N) matrix of eigenvectors (columns = mode shapes)
    """
    theta = torch.linspace(0, 2 * math.pi, n_nodes + 1)[:-1]
    pos = torch.stack([theta.cos(), theta.sin()], dim=1)  # (N, 2)
    pos += noise * torch.randn_like(pos)

    # pairwise squared distances
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 2)
    dist2 = (diff ** 2).sum(-1)                  # (N, N)

    # kNN connectivity
    knn_idx = dist2.argsort(dim=1)[:, 1: k_neighbours + 1]  # (N, k)
    A = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in knn_idx[i]:
            w = math.exp(-dist2[i, j].item() / (2 * sigma ** 2))
            A[i, j] = w
            A[j, i] = w

    # Normalised Laplacian  L = I - D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1).clamp(min=1e-8)
    D_inv_sqrt = torch.diag(deg ** -0.5)
    L = torch.eye(n_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    L = (L + L.T) / 2  # enforce exact symmetry

    eigvals, eigvecs = torch.linalg.eigh(L)  # ascending eigenvalues
    return L, eigvecs, eigvals, pos


def build_dataset(n_graphs: int, n_nodes: int = 20, k_neighbours: int = 4,
                  tau_modes: int = 6) -> dict:
    """
    Generate `n_graphs` random spring-network graphs.

    Node features x_i are the i-th rows of the first `tau_modes` eigenvectors
    (i.e. the projection of node i onto the lowest vibrational mode shapes).

    Returns a dict with:
        X       : (n_graphs * N, tau_modes)  node feature matrix
        Ls      : (n_graphs, N, N)  Laplacians
        eigvals : (n_graphs, N)     eigenvalue spectra
        graph_id: (n_graphs * N,)   graph membership index
    """
    Xs, Ls, Evs = [], [], []
    for g in range(n_graphs):
        k = random.randint(2, k_neighbours + 2)
        L, modes, eigvals, _ = make_spring_graph(n_nodes, k_neighbours=k)
        # node features = first tau_modes mode-shape components
        x = modes[:, :tau_modes]  # (N, tau_modes)
        Xs.append(x)
        Ls.append(L)
        Evs.append(eigvals)

    X = torch.stack(Xs)               # (G, N, tau_modes)
    Ls_t = torch.stack(Ls)            # (G, N, N)
    Evs_t = torch.stack(Evs)          # (G, N, N)
    graph_id = torch.arange(n_graphs).repeat_interleave(n_nodes)

    # Flatten to (G*N, tau_modes) for node-level encoding
    X_flat = X.view(-1, tau_modes)
    return dict(X=X_flat, X_graph=X, Ls=Ls_t, eigvals=Evs_t,
                graph_id=graph_id, n_graphs=n_graphs,
                n_nodes=n_nodes, tau_modes=tau_modes)


# ===========================================================================
# 2.  MINIMAL WAE  (self-contained, no wae/ package import needed for demo)
# ===========================================================================

class DiffLaplacian(nn.Module):
    """Differentiable normalised Laplacian from learned edge-weight logits."""

    def __init__(self, L_base: torch.Tensor):
        super().__init__()
        N = L_base.shape[0]
        self.N = N
        # Recover base adjacency from L_base
        D_base = 1 - L_base.diag()  # approximate degree proxy; use A directly
        # Store upper-triangle edge indices for a fully-connected base graph
        idx = torch.triu_indices(N, N, offset=1)
        self.register_buffer("row", idx[0])
        self.register_buffer("col", idx[1])
        # Base weights from L_base (for warm-start)
        A_base = torch.eye(N) - L_base
        A_base = A_base.clamp(min=0)
        base_w = A_base[idx[0], idx[1]].clamp(min=1e-6)
        self.log_base_w = nn.Parameter(base_w.log())

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta : (E,) or (B, E) edge-weight logit adjustments
        Returns L : (N, N) or (B, N, N)
        """
        base_w = self.log_base_w.exp()
        batched = delta.dim() == 2
        if not batched:
            delta = delta.unsqueeze(0)
        B = delta.shape[0]
        N, E_sz = self.N, self.row.shape[0]

        w = base_w.unsqueeze(0) * torch.sigmoid(delta)  # (B, E)
        # Build adjacency
        A = torch.zeros(B, N, N, device=w.device)
        A[:, self.row, self.col] = w
        A[:, self.col, self.row] = w
        # Normalised Laplacian
        deg = A.sum(-1).clamp(min=1e-8)               # (B, N)
        D_inv_sqrt = torch.diag_embed(deg ** -0.5)    # (B, N, N)
        I = torch.eye(N, device=w.device).unsqueeze(0)
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
        L = (L + L.transpose(-1, -2)) / 2
        return L.squeeze(0) if not batched else L


def spectral_entropy(eigvals: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Shannon entropy of the normalised eigenvalue distribution.
    eigvals : (..., N)
    """
    lam = eigvals.clamp(min=eps)
    p = lam / lam.sum(dim=-1, keepdim=True)
    return -(p * (p + eps).log()).sum(dim=-1)


def freq_cost(L: torch.Tensor, tau_modes: int) -> torch.Tensor:
    """
    J_freq = sum of eigenvalues beyond the first tau_modes.
    Penalises high-frequency wiring energy.
    """
    eigvals = torch.linalg.eigvalsh(L).clamp(min=0)  # (B, N)
    return eigvals[:, tau_modes:].sum(dim=-1).mean()


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.GELU(),
        )
        self.mu_head      = nn.Linear(hidden, latent_dim)
        self.log_var_head = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu, lv = self.mu_head(h), self.log_var_head(h)
        std = (0.5 * lv).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, lv

    @staticmethod
    def kl(mu, log_var):
        return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()


class WiringDecoder(nn.Module):
    """Maps z to per-edge delta logits, then builds L(z)."""

    def __init__(self, latent_dim: int, n_edges: int, hidden: int = 128,
                 n_heads: int = 4, laplacian: DiffLaplacian = None):
        super().__init__()
        self.lap = laplacian
        self.heads = nn.ModuleList([
            nn.Linear(latent_dim, n_edges) for _ in range(n_heads)
        ])
        self.gate = nn.Linear(latent_dim, n_heads)

    def forward(self, z):
        # Mixture-of-experts over n_heads edge templates
        gates = F.softmax(self.gate(z), dim=-1)             # (B, H)
        deltas = torch.stack([h(z) for h in self.heads], dim=1)  # (B, H, E)
        delta = (gates.unsqueeze(-1) * deltas).sum(dim=1)   # (B, E)
        return self.lap(delta), delta


class DiffusionDecoder(nn.Module):
    """L(z), E -> x̂  via tau-mode heat kernel."""

    def __init__(self, tau_modes: int, feat_dim: int, out_dim: int,
                 learnable_t: bool = True):
        super().__init__()
        self.tau = tau_modes
        self.log_t = nn.Parameter(torch.zeros(1)) if learnable_t else None
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(self, L: torch.Tensor, E: torch.Tensor,
                node_idx: torch.Tensor = None) -> torch.Tensor:
        """
        L  : (B, N, N)
        E  : (N, D) embedding table
        node_idx : (B,) — which node to decode; if None decode all
        """
        B, N, _ = L.shape
        eigvals, eigvecs = torch.linalg.eigh(L)            # (B,N), (B,N,N)
        lam_k = eigvals[:, :self.tau]                      # (B, k)
        U_k   = eigvecs[:, :, :self.tau]                   # (B, N, k)

        t = self.log_t.exp() if self.log_t is not None else torch.ones(1, device=L.device)
        heat = torch.exp(-t * lam_k)                       # (B, k)
        # K_tau = U_k diag(heat) U_k^T  ->  (B, N, N)
        K = U_k * heat.unsqueeze(1)                        # (B, N, k)
        K = K @ U_k.transpose(-1, -2)                      # (B, N, N)

        if node_idx is not None:
            # Select rows for target nodes
            row_sel = K[torch.arange(B, device=K.device), node_idx]  # (B, N)
            feat = row_sel @ E.unsqueeze(0).expand(B, -1, -1).squeeze()  # needs care
            # safe batched matmul
            feat = torch.bmm(row_sel.unsqueeze(1),
                             E.unsqueeze(0).expand(B, -1, -1)).squeeze(1)  # (B, D)
        else:
            feat = torch.bmm(K, E.unsqueeze(0).expand(B, -1, -1))         # (B, N, D)

        return self.mlp(feat), K


class SpectralWAE(nn.Module):
    def __init__(self, feat_dim: int, latent_dim: int, n_nodes: int,
                 tau_modes: int, L_base: torch.Tensor,
                 beta: float = 1.0, alpha: float = 0.1,
                 hidden: int = 128, n_heads: int = 4):
        super().__init__()
        n_edges = n_nodes * (n_nodes - 1) // 2
        self.laplacian = DiffLaplacian(L_base)
        self.encoder   = Encoder(feat_dim, latent_dim, hidden)
        self.wdecoder  = WiringDecoder(latent_dim, n_edges, hidden, n_heads,
                                       laplacian=self.laplacian)
        self.ddecoder  = DiffusionDecoder(tau_modes, feat_dim, feat_dim)
        self.beta, self.alpha = beta, alpha
        self.tau = tau_modes

    def forward(self, x: torch.Tensor, E: torch.Tensor,
                node_idx: torch.Tensor = None):
        B = x.shape[0]
        z, mu, lv = self.encoder(x)
        L, delta  = self.wdecoder(z)               # L : (B, N, N)
        x_hat, K  = self.ddecoder(L, E, node_idx)  # x_hat : (B,D) or (B,N,D)

        # Losses
        if node_idx is not None:
            recon = F.mse_loss(x_hat, x)
        else:
            recon = F.mse_loss(x_hat, E.unsqueeze(0).expand(B, -1, -1))

        kl   = Encoder.kl(mu, lv)
        jf   = freq_cost(L, self.tau)
        loss = recon + self.beta * kl + self.alpha * jf

        return dict(loss=loss, recon=recon, kl=kl, jfreq=jf,
                    x_hat=x_hat, L=L, z=z, mu=mu, lv=lv)

    def generate(self, E: torch.Tensor, n_samples: int = 1,
                 z: torch.Tensor = None):
        with torch.no_grad():
            if z is None:
                z = torch.randn(n_samples, self.encoder.mu_head.out_features,
                                device=E.device)
            L, _ = self.wdecoder(z)
            x_hat, _ = self.ddecoder(L, E)
            eigvals = torch.linalg.eigvalsh(L)
            H = spectral_entropy(eigvals)
        return dict(z=z, L=L, x_hat=x_hat, eigvals=eigvals, entropy=H)


# ===========================================================================
# 3.  TRAINING LOOP
# ===========================================================================

def train(model: SpectralWAE, loader: DataLoader, optimizer: torch.optim.Optimizer,
          device: str) -> dict:
    model.train()
    totals = dict(loss=0., recon=0., kl=0., jfreq=0.)
    n = 0
    for x_batch, node_idx in loader:
        x_batch  = x_batch.to(device)
        node_idx = node_idx.to(device)
        E        = model.wdecoder.lap.log_base_w  # proxy — use x_batch as E too
        # Use x_batch as both node features and the embedding table slice
        # (full E is passed via the graph-level tensor set outside the loop)
        out = model(x_batch, x_batch, node_idx=None)
        optimizer.zero_grad()
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for k in totals:
            totals[k] += out[k].item() * x_batch.shape[0]
        n += x_batch.shape[0]
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model: SpectralWAE, loader: DataLoader, device: str) -> dict:
    model.eval()
    totals = dict(loss=0., recon=0., kl=0., jfreq=0.)
    n = 0
    for x_batch, node_idx in loader:
        x_batch  = x_batch.to(device)
        out = model(x_batch, x_batch, node_idx=None)
        for k in totals:
            totals[k] += out[k].item() * x_batch.shape[0]
        n += x_batch.shape[0]
    return {k: v / n for k, v in totals.items()}


# ===========================================================================
# 4.  SPECTRAL-DISTANCE METRIC
# ===========================================================================

def spectral_distance(L_a: torch.Tensor, L_b: torch.Tensor) -> torch.Tensor:
    """
    Frobenius distance between two Laplacians (batch-safe).
    L_a, L_b : (B, N, N) or (N, N)
    """
    diff = L_a - L_b
    return (diff * diff).sum(dim=(-1, -2)).sqrt()


def eigenvalue_distance(lam_a: torch.Tensor, lam_b: torch.Tensor) -> torch.Tensor:
    """
    L2 distance between sorted eigenvalue spectra.
    """
    return (lam_a - lam_b).pow(2).sum(dim=-1).sqrt()


# ===========================================================================
# 5.  ENTROPY-CONTROLLED GENERATION
# ===========================================================================

def generate_at_target_entropy(model: SpectralWAE, E: torch.Tensor,
                               target_H: float, n_candidates: int = 500,
                               tol: float = 0.05) -> dict:
    """
    Sample z vectors from N(0,I), compute entropy of L(z), and keep those
    closest to `target_H`.  Returns statistics on how well the WAE can
    produce wirings at specified spectral entropy.
    """
    gen = model.generate(E, n_samples=n_candidates)
    H   = gen["entropy"]                      # (n_candidates,)
    err = (H - target_H).abs()
    matched = (err < tol).float().mean().item()
    best_idx = err.argmin().item()
    return dict(
        target=target_H,
        mean_entropy=H.mean().item(),
        std_entropy=H.std().item(),
        match_rate=matched,
        best_err=err.min().item(),
        best_L=gen["L"][best_idx],
        best_eigvals=gen["eigvals"][best_idx],
    )


# ===========================================================================
# 6.  FIGURES
# ===========================================================================

def save_figures(out_dir: Path, train_log: list, gen_results: dict,
                 entropy_results: list, dataset: dict,
                 model: SpectralWAE, device: str) -> None:
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # --- Training curves ---
    epochs = [r["epoch"] for r in train_log]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, key, title in zip(axes,
            ["train_loss", "train_recon", "train_kl", "train_jfreq"],
            ["Total ELBO Loss", "Recon MSE", "KL Divergence", "J_freq Spectral Cost"]):
        ax.plot(epochs, [r[key] for r in train_log], label="train", lw=2)
        ax.plot(epochs, [r["val_" + key[6:]] for r in train_log],
                label="val", lw=2, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Generated spectral entropy distribution ---
    H_vals = gen_results["entropies"]
    H_data = gen_results["data_entropies"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(H_data, bins=30, alpha=0.6, color="#01696f", label="Dataset Laplacians")
    ax.hist(H_vals, bins=30, alpha=0.6, color="#964219", label="Generated Laplacians")
    ax.set_xlabel("Spectral Entropy H(Λ)", fontsize=11)
    ax.set_title("Spectral Entropy: Dataset vs WAE Generated", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "entropy_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Spectral distance ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gen_results["spectral_distances"], bins=30, color="#a12c7b", alpha=0.85)
    ax.axvline(gen_results["mean_spectral_dist"], color="#a12c7b",
               linestyle="--", lw=2, label=f"mean = {gen_results['mean_spectral_dist']:.3f}")
    ax.set_xlabel("Frobenius distance ||L_nearest_data − L_gen||_F", fontsize=10)
    ax.set_title("Spectral Distance to Nearest Training Laplacian", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "spectral_distance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Latent entropy scatter (PCA) ---
    from sklearn.decomposition import PCA  # type: ignore
    Z = gen_results["latent_z"].numpy()
    H = gen_results["entropies"]
    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=H, cmap="viridis", s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Spectral Entropy")
    ax.set_title("WAE Latent Space coloured by Spectral Entropy", fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(figs / "latent_entropy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Sample vibrational mode shapes ---
    N = dataset["n_nodes"]
    tau = dataset["tau_modes"]
    sample_L = dataset["Ls"][0]
    _, sample_modes = torch.linalg.eigh(sample_L)
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    fig, axes = plt.subplots(1, min(4, tau), figsize=(14, 3))
    for idx, ax in enumerate(axes):
        ax.plot(theta.numpy(), sample_modes[:, idx].numpy(),
                lw=2, color=cm.viridis(idx / tau))
        ax.set_title(f"Mode {idx + 1}  (λ={dataset['eigvals'][0, idx]:.3f})")
        ax.set_xlabel("θ (node position)"); ax.grid(alpha=0.3)
    plt.suptitle("Vibrational Mode Shapes of Sample Spring Network", y=1.02)
    plt.tight_layout()
    plt.savefig(figs / "mode_shapes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Entropy target error ---
    targets  = [r["target"]   for r in entropy_results]
    best_err = [r["best_err"] for r in entropy_results]
    match_rt = [r["match_rate"] for r in entropy_results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(targets, best_err, marker="o", color="#01696f", lw=2)
    ax1.set_xlabel("Target Spectral Entropy", fontsize=11)
    ax1.set_ylabel("|H_generated − H_target|", fontsize=11)
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
    print(f"[demo] Figures saved to {figs}/")


# ===========================================================================
# 7.  EXPORT CSV RESULTS
# ===========================================================================

def export_csvs(out_dir: Path, train_log: list, gen_results: dict,
                entropy_results: list) -> None:
    import csv

    # Training log
    with open(out_dir / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=train_log[0].keys())
        w.writeheader(); w.writerows(train_log)

    # Per-sample generation metrics
    rows = []
    n = len(gen_results["entropies"])
    for i in range(n):
        rows.append(dict(
            sample_id=i,
            spectral_entropy=float(gen_results["entropies"][i]),
            spectral_distance=float(gen_results["spectral_distances"][i])
            if i < len(gen_results["spectral_distances"]) else float("nan"),
        ))
    with open(out_dir / "spectral_demo_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "spectral_entropy",
                                           "spectral_distance"])
        w.writeheader(); w.writerows(rows)

    # Entropy control results
    with open(out_dir / "entropy_control_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "mean_entropy", "std_entropy",
                                           "match_rate", "best_err"])
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

def main():
    parser = argparse.ArgumentParser(
        description="Molecular/Spectral Graph-Generation Demo for WAE (issue #13)")
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

    # ------------------------------------------------------------------
    print("[demo] Building spring-network dataset...")
    dataset = build_dataset(args.n_graphs, args.n_nodes, args.k_neigh,
                            args.tau_modes)
    N   = args.n_nodes
    D   = args.tau_modes        # feature dimension = number of mode shapes

    # Use the mean Laplacian as the base graph for DiffLaplacian
    L_base = dataset["Ls"].mean(0)  # (N, N)
    print(f"[demo] Dataset: {args.n_graphs} graphs × {N} nodes, "
          f"feat_dim={D}, tau_modes={args.tau_modes}")

    # Flatten dataset for node-level DataLoader
    X_all  = dataset["X"]        # (G*N, D)
    idx_all = torch.arange(X_all.shape[0])
    ds = TensorDataset(X_all, idx_all)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # ------------------------------------------------------------------
    print("[demo] Building SpectralWAE...")
    model = SpectralWAE(
        feat_dim=D, latent_dim=args.latent_dim, n_nodes=N,
        tau_modes=args.tau_modes, L_base=L_base,
        beta=args.beta, alpha=args.alpha,
        hidden=args.hidden, n_heads=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    print(f"[demo] Training for {args.epochs} epochs...")
    train_log = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t_metrics = train(model, train_loader, optimizer, device)
        v_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        row = dict(epoch=epoch)
        row.update({"train_" + k: v for k, v in t_metrics.items()})
        row.update({"val_"   + k: v for k, v in v_metrics.items()})
        train_log.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  "
                  f"loss={t_metrics['loss']:.4f}  "
                  f"recon={t_metrics['recon']:.4f}  "
                  f"kl={t_metrics['kl']:.4f}  "
                  f"jfreq={t_metrics['jfreq']:.4f}")
        if v_metrics["loss"] < best_val_loss:
            best_val_loss = v_metrics["loss"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    # ------------------------------------------------------------------
    print("[demo] Generating novel wirings...")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    model.eval()

    # Embedding table = mean feature per node position across all graphs
    E_mean = dataset["X_graph"].mean(0).to(device)    # (N, D)
    gen = model.generate(E_mean, n_samples=args.n_gen)

    # Spectral distances: for each generated L, find nearest data L
    Ls_data = dataset["Ls"].to(device)   # (G, N, N)
    sp_dists = []
    for i in range(gen["L"].shape[0]):
        d = spectral_distance(
            Ls_data, gen["L"][i].unsqueeze(0).expand_as(Ls_data)
        )  # (G,)
        sp_dists.append(d.min().item())

    # Dataset Laplacian entropies for comparison
    data_eig = torch.linalg.eigvalsh(Ls_data)
    H_data   = spectral_entropy(data_eig).tolist()

    gen_results = dict(
        entropies=gen["entropy"].tolist(),
        spectral_distances=sp_dists,
        mean_spectral_dist=float(np.mean(sp_dists)),
        data_entropies=H_data,
        latent_z=gen["z"].cpu(),
    )

    # ------------------------------------------------------------------
    print("[demo] Entropy-controlled generation experiment...")
    H_range = np.linspace(
        float(np.percentile(H_data, 10)),
        float(np.percentile(H_data, 90)),
        num=10
    )
    entropy_results = []
    for target_H in H_range:
        res = generate_at_target_entropy(
            model, E_mean, float(target_H),
            n_candidates=500, tol=0.05)
        entropy_results.append(res)
        print(f"  target H={target_H:.3f}  "
              f"best_err={res['best_err']:.4f}  "
              f"match_rate={res['match_rate']:.3f}")

    # ------------------------------------------------------------------
    print("[demo] Exporting CSVs and figures...")
    export_csvs(out_dir, train_log, gen_results, entropy_results)
    save_figures(out_dir, train_log, gen_results, entropy_results,
                 dataset, model, device)

    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"  Dataset            : {args.n_graphs} spring-network graphs, {N} nodes")
    print(f"  Best val ELBO loss : {best_val_loss:.5f}")
    print(f"  Mean spectral dist : {gen_results['mean_spectral_dist']:.4f}")
    mean_H = float(np.mean(gen_results['entropies']))
    std_H  = float(np.std(gen_results['entropies']))
    print(f"  Generated H(Λ)     : {mean_H:.3f} ± {std_H:.3f}")
    mean_match = float(np.mean([r['match_rate'] for r in entropy_results]))
    print(f"  Mean entropy match rate (tol=0.05): {mean_match:.3f}")
    print("=" * 60)
    print(f"  Results: {out_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
