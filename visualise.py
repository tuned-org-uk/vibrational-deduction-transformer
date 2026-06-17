"""
Visualisations for Wiring Autoencoder experiments.

Aligned with WiringAutoencoder v2 API (vdt/model.py).

Changes from the stale version (issue #63):

  - plot_spectral: replaced broken model.generate(E) call with the
    correct v2 signature model.generate(U_q, eigvals_q, E).  generate()
    returns a plain tensor (n_samples, D), not a dict with 'L'.
    Spectral entropy is now computed by running the prior-sampled z
    through the wiring decoder directly to extract L_z (B, N, N).

  - plot_latent: added the full spectral preamble (base Laplacian ->
    _safe_eigh -> U_q, eigvals_q, spectral_cache) so the forward call
    receives all arguments required by the v2 signature.  Confirmed
    out['mu'] is still the correct latent-mean key.

  - Added plot_taumode and --mode taumode: runs the test loader,
    collects per-sample Gamma shape/rate parameters (log_a, log_b) from
    the encoder's ModeWeightHead, derives expected mode weights
    omega_k = exp(log_a) / exp(log_b) per sample, and plots a per-class
    KDE of the mean omega distribution.

Reads CSV files produced by train.py / benchmark.py and renders:
    1. Training curves  (loss, recon, kl_z, kl_S, kl_tau)
    2. Benchmark comparison bar chart
    3. Latent space 2D PCA coloured by class label
    4. Spectral entropy of generated Laplacians across latent space
    5. Taumode weight distribution per class (KDE)

Designed to work with pluot (https://github.com/keller-mark/pluot) for
interactive visualisation; falls back to matplotlib for static output.

Usage
-----
    python visualise.py --mode training  --log checkpoints/training_log.csv
    python visualise.py --mode benchmark --csv results/benchmark_cora.csv
    python visualise.py --mode latent    --checkpoint checkpoints/best.pt --dataset cora
    python visualise.py --mode spectral  --checkpoint checkpoints/best.pt --dataset cora
    python visualise.py --mode taumode   --checkpoint checkpoints/best.pt --dataset cora
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Internal spectral preamble helper
# ---------------------------------------------------------------------------

def _build_spectral_preamble(cfg: dict, E, device: str = "cpu"):
    """
    Build the spectral quantities required by WiringAutoencoder.forward() v2.

    Computes the base Laplacian from the embedding table E, runs a full
    eigendecomposition once, and returns the sliced (U_q, eigvals_q) pair
    together with the full spectral_cache for DiffusionDecoder.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config dict (must contain 'model' and 'graph' keys).
    E : torch.Tensor
        Embedding table shape (N, D).  Used to build the base Laplacian.
    device : str
        Target device string.

    Returns
    -------
    base_L       : torch.Tensor (N, N) -- dense base Laplacian
    U_q          : torch.Tensor (N, q) -- leading q eigenvectors
    eigvals_q    : torch.Tensor (q,)   -- leading q eigenvalues
    spectral_cache : tuple (full_eigvals (N,), full_eigvecs (N, N))
    """
    import torch
    from vdt.laplacian import DifferentiableLaplacian
    from vdt.spectral import _safe_eigh

    gc = cfg.get("graph", {})
    base_lap = DifferentiableLaplacian.from_embeddings(
        E,
        knn_k=gc.get("knn_k", 15),
        sigma=gc.get("sigma", 0.5),
        normalised=gc.get("normalised", True),
        sparse=False,   # always dense for visualisation
    )
    with torch.no_grad():
        base_L = base_lap(base_lap.base_weights.unsqueeze(0)).squeeze(0)  # (N, N)
        full_eigvals, full_eigvecs = _safe_eigh(base_L)                   # (N,), (N, N)

    q = cfg["model"].get("q", cfg["model"].get("tau_modes", 8))
    U_q       = full_eigvecs[:, :q]
    eigvals_q = full_eigvals[:q]
    spectral_cache = (full_eigvals, full_eigvecs)
    return base_L, U_q, eigvals_q, spectral_cache


# ---------------------------------------------------------------------------
# Mode: training curves
# ---------------------------------------------------------------------------

def plot_training(log_csv: str, out_dir: str) -> None:
    """
    Plot training and validation curves from the CSV log produced by train.py.

    Reads columns: epoch, train_loss, val_loss, train_recon, val_recon,
    train_kl_z, val_kl_z, train_kl_S, val_kl_S, train_kl_tau, val_kl_tau.
    Aligned with the v2 csv_fields written by train.py (issue #62).

    Parameters
    ----------
    log_csv : str
        Path to the CSV log file.
    out_dir : str
        Directory in which to save training_curves.png and
        training_curves.json.
    """
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    metrics = [
        ("train_loss",   "val_loss",    "Total ELBO Loss"),
        ("train_recon",  "val_recon",   "Reconstruction Loss"),
        ("train_kl_z",   "val_kl_z",    "KL_z (isotropic)"),
        ("train_kl_S",   "val_kl_S",    "KL_S (spectral basis)"),
        ("train_kl_tau", "val_kl_tau",  "KL_tau (mode frequency)"),
    ]
    for ax, (tr, vl, title) in zip(axes, metrics):
        ax.plot(df["epoch"], df[tr], label="train", linewidth=2)
        ax.plot(df["epoch"], df[vl], label="val",   linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(out_dir) / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    pluot_data = {
        "type": "line",
        "series": [
            {"name": title, "train": df[tr].tolist(), "val": df[vl].tolist()}
            for tr, vl, title in metrics
        ],
        "x": df["epoch"].tolist()
    }
    with open(Path(out_dir) / "training_curves.json", "w") as f:
        json.dump(pluot_data, f)


# ---------------------------------------------------------------------------
# Mode: benchmark comparison
# ---------------------------------------------------------------------------

def plot_benchmark(csv_path: str, out_dir: str) -> None:
    """
    Render a grouped bar chart comparing models on benchmark metrics.

    Expects a CSV with columns: model, recon_mse, kl, linear_probe.

    Parameters
    ----------
    csv_path : str
        Path to the benchmark CSV (e.g. results/benchmark_cora.csv).
    out_dir : str
        Output directory for benchmark_comparison.png and .json.
    """
    df = pd.read_csv(csv_path)
    metrics  = ["recon_mse", "kl", "linear_probe"]
    labels   = ["Recon MSE down", "KL Div down", "Linear Probe Acc up"]
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#01696f", "#964219", "#a12c7b"]
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=row["model"],
                      color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_title("Benchmark: VDT vs Baseline VAE vs Linear AE", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = Path(out_dir) / "benchmark_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    with open(Path(out_dir) / "benchmark_comparison.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f)


# ---------------------------------------------------------------------------
# Mode: latent space
# ---------------------------------------------------------------------------

def plot_latent(checkpoint_path: str, dataset: str, out_dir: str) -> None:
    """
    Project the test-set latent means to 2D PCA and colour by class label.

    Loads a checkpoint, rebuilds the model, runs the full spectral preamble
    (base Laplacian -> _safe_eigh -> U_q, eigvals_q, spectral_cache), then
    collects out['mu'] from each test batch using the v2 forward signature.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt file saved by train.py (keys: 'cfg', 'model').
    dataset : str
        Dataset name override (e.g. 'cora').
    out_dir : str
        Output directory for latent_space.png and latent_space.json.
    """
    import torch
    from vdt import WiringAutoencoder
    from vdt.dataset import load_dataset, make_loaders
    from sklearn.decomposition import PCA

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]
    cfg["dataset"]["name"] = dataset
    data = load_dataset(dataset, root=cfg["dataset"]["root"], device="cpu")
    E    = data["E"]

    model = WiringAutoencoder.from_config(cfg, E)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Full spectral preamble -- required by v2 forward() signature.
    # Builds U_q (N, q), eigvals_q (q,), and spectral_cache for DiffusionDecoder.
    _base_L, U_q, eigvals_q, spectral_cache = _build_spectral_preamble(cfg, E)

    loaders = make_loaders(data, batch_size=512)
    zs, ys = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            out = model(
                batch["x"],
                U_q,
                eigvals_q,
                node_idx=batch["node_idx"],
                spectral_cache=spectral_cache,
            )
            zs.append(out["mu"].numpy())
            ys.append(batch["label"].numpy())
    Z = np.concatenate(zs)
    Y = np.concatenate(ys)

    pca = PCA(n_components=2)
    Z2  = pca.fit_transform(Z)
    n_classes = data["meta"]["n_classes"]
    cmap = plt.get_cmap("tab10", n_classes)

    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(n_classes):
        mask = Y == c
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=12, alpha=0.6,
                   color=cmap(c), label=f"class {c}")
    ax.set_title("VDT Latent Space (PCA-2D) -- coloured by class", fontsize=12)
    ax.legend(markerscale=2, fontsize=9, ncol=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = Path(out_dir) / "latent_space.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    pluot_data = {
        "type": "scatter",
        "x": Z2[:, 0].tolist(),
        "y": Z2[:, 1].tolist(),
        "label": Y.tolist(),
    }
    with open(Path(out_dir) / "latent_space.json", "w") as f:
        json.dump(pluot_data, f)


# ---------------------------------------------------------------------------
# Mode: spectral entropy of generated Laplacians
# ---------------------------------------------------------------------------

def plot_spectral(
    checkpoint_path: str,
    dataset: str,
    out_dir: str,
    n_samples: int = 200,
) -> None:
    """
    Plot the distribution of spectral entropy over Laplacians generated by
    sampling z ~ N(0, I) and decoding through the wiring decoder.

    The v2 generate() method returns a plain tensor of reconstructed
    embeddings (n_samples, D), not a dict with 'L'.  To obtain the
    Laplacians, this function samples z from the prior and runs the
    wiring decoder directly to extract L_z (n_samples, N, N) before
    computing spectral entropy.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt file saved by train.py.
    dataset : str
        Dataset name override.
    out_dir : str
        Output directory for spectral_entropy.png and spectral_entropy.json.
    n_samples : int
        Number of prior samples to generate.  Default 200.
    """
    import torch
    from vdt import WiringAutoencoder
    from vdt.dataset import load_dataset

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]
    cfg["dataset"]["name"] = dataset
    data = load_dataset(dataset, root=cfg["dataset"]["root"], device="cpu")
    E    = data["E"]

    model = WiringAutoencoder.from_config(cfg, E)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _base_L, U_q, eigvals_q, _spectral_cache = _build_spectral_preamble(cfg, E)

    with torch.no_grad():
        # Sample z from the isotropic prior, project to spectral mode space,
        # and decode through the wiring decoder to obtain L_z directly.
        # This mirrors the internals of model.generate() but exposes L_z.
        z    = torch.randn(n_samples, model.latent_dim)   # (n_samples, latent_dim)
        z_q  = model.z_to_q(z)                            # (n_samples, q)
        # wiring_decoder returns (W, omega, S, L_z, log_var_S)
        _W, _omega, _S, L_z, _lv = model.wiring_decoder(
            z_q, U_q, model._laplacian.base_laplacian
        )  # L_z: (n_samples, N, N)

        # Spectral entropy H(Lambda) = -sum_i p_i log(p_i)
        # where p_i = lambda_i / sum(lambda_j), lambda_i = max(eigval_i, 0).
        eigvals_all = torch.linalg.eigvalsh(L_z).clamp(min=1e-8)  # (n_samples, N)
        p = eigvals_all / eigvals_all.sum(dim=-1, keepdim=True)
        entropy = -(p * p.log()).sum(dim=-1).numpy()  # (n_samples,)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(entropy, bins=30, color="#01696f", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Spectral Entropy H(Lambda)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Spectral Entropy in Generated Laplacians", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(out_dir) / "spectral_entropy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    pluot_data = {
        "type": "histogram",
        "values": entropy.tolist(),
        "metric": "spectral_entropy",
    }
    with open(Path(out_dir) / "spectral_entropy.json", "w") as f:
        json.dump(pluot_data, f)


# ---------------------------------------------------------------------------
# Mode: taumode weight distribution
# ---------------------------------------------------------------------------

def plot_taumode(checkpoint_path: str, dataset: str, out_dir: str) -> None:
    """
    Plot the per-class distribution of expected spectral mode weights omega_k.

    For each test sample the encoder's ModeWeightHead produces log_a and
    log_b (shape (B, q)), parameterising a Gamma posterior q(omega_k).  The
    expected mode weight for sample i is:

        omega_i = mean_k( exp(log_a_i) / exp(log_b_i) )

    i.e. the mean expected weight averaged over all q modes.  This scalar
    summarises how strongly the wiring decoder relies on spectral structure
    for that sample.

    A per-class KDE of this distribution surfaces mode-selectivity
    differences between classes -- a key diagnostic for the VDT architecture.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt file saved by train.py.
    dataset : str
        Dataset name override.
    out_dir : str
        Output directory for taumode_distribution.png and
        taumode_distribution.json.
    """
    import torch
    from vdt import WiringAutoencoder
    from vdt.dataset import load_dataset, make_loaders
    from scipy.stats import gaussian_kde

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]
    cfg["dataset"]["name"] = dataset
    data = load_dataset(dataset, root=cfg["dataset"]["root"], device="cpu")
    E    = data["E"]

    model = WiringAutoencoder.from_config(cfg, E)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _base_L, U_q, eigvals_q, spectral_cache = _build_spectral_preamble(cfg, E)

    loaders = make_loaders(data, batch_size=512)
    omegas, ys = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            out = model(
                batch["x"],
                U_q,
                eigvals_q,
                node_idx=batch["node_idx"],
                spectral_cache=spectral_cache,
            )
            # The encoder attaches log_a and log_b to the output dict via
            # WiringEncoder.  They are not in the top-level forward() return
            # dict -- re-run the encoder directly to access them.
            z, mu, log_var, log_a, log_b = model.encoder(
                batch["x"],
                L_f=None,
                eigvecs=U_q,
                lap=model._laplacian,
            )
            # Expected mode weight per sample: mean over q modes of a_k/b_k.
            a = log_a.exp()   # (B, q)
            b = log_b.exp()   # (B, q)
            omega_mean = (a / b).mean(dim=-1)  # (B,)
            omegas.append(omega_mean.numpy())
            ys.append(batch["label"].numpy())

    Omega = np.concatenate(omegas)  # (N_test,)
    Y     = np.concatenate(ys)      # (N_test,)

    n_classes = data["meta"]["n_classes"]
    cmap = plt.get_cmap("tab10", n_classes)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_grid  = np.linspace(Omega.min() - 0.1, Omega.max() + 0.1, 300)
    kde_data = {}
    for c in range(n_classes):
        vals = Omega[Y == c]
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method="scott")
        density = kde(x_grid)
        ax.plot(x_grid, density, color=cmap(c), linewidth=2, label=f"class {c}")
        ax.fill_between(x_grid, density, alpha=0.12, color=cmap(c))
        kde_data[str(c)] = {"x": x_grid.tolist(), "density": density.tolist()}

    ax.set_xlabel("Mean expected mode weight  E[omega_k]", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Taumode weight distribution per class  (VDT spectral selectivity)",
        fontsize=12,
    )
    ax.legend(markerscale=2, fontsize=9, ncol=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = Path(out_dir) / "taumode_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    pluot_data = {
        "type": "kde",
        "classes": kde_data,
        "metric": "mean_omega",
    }
    with open(Path(out_dir) / "taumode_distribution.json", "w") as f:
        json.dump(pluot_data, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["training", "benchmark", "latent", "spectral", "taumode"],
        required=True,
    )
    p.add_argument("--log",        default="checkpoints/training_log.csv")
    p.add_argument("--csv",        default="results/benchmark_cora.csv")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--dataset",    default="cora")
    p.add_argument("--output",     default="results/")
    p.add_argument("--n-samples",  type=int, default=200)
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.mode == "training":
        plot_training(args.log, args.output)
    elif args.mode == "benchmark":
        plot_benchmark(args.csv, args.output)
    elif args.mode == "latent":
        plot_latent(args.checkpoint, args.dataset, args.output)
    elif args.mode == "spectral":
        plot_spectral(args.checkpoint, args.dataset, args.output,
                      n_samples=args.n_samples)
    elif args.mode == "taumode":
        plot_taumode(args.checkpoint, args.dataset, args.output)


if __name__ == "__main__":
    main()
