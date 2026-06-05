"""
Visualisations for Wiring Autoencoder experiments.

Reads CSV files produced by train.py / benchmark.py and renders:
    1. Training curves (loss, recon, KL, freq)
    2. Benchmark comparison bar chart
    3. Latent space 2D UMAP / PCA coloured by class label
    4. Spectral entropy of generated Laplacians across latent space

Designed to work with pluot (https://github.com/keller-mark/pluot) for
interactive visualisation; falls back to matplotlib for static output.

Usage
-----
    python visualise.py --mode training --log checkpoints/training_log.csv
    python visualise.py --mode benchmark --csv results/benchmark_cora.csv
    python visualise.py --mode latent    --checkpoint checkpoints/best.pt --dataset cora
    python visualise.py --mode spectral  --checkpoint checkpoints/best.pt --dataset cora
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
# Mode: training curves
# ---------------------------------------------------------------------------
def plot_training(log_csv: str, out_dir: str) -> None:
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    metrics = [
        ("train_loss",  "val_loss",  "Total ELBO Loss"),
        ("train_recon", "val_recon", "Reconstruction Loss"),
        ("train_kl",    "val_kl",    "KL Divergence"),
        ("train_freq",  "val_freq",  "J_freq (Spectral Cost)"),
    ]
    for ax, (tr, vl, title) in zip(axes, metrics):
        ax.plot(df["epoch"], df[tr], label="train", linewidth=2)
        ax.plot(df["epoch"], df[vl], label="val",   linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(out_dir) / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Also export JSON for pluot
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
    df = pd.read_csv(csv_path)
    metrics  = ["recon_mse", "kl", "linear_probe"]
    labels   = ["Recon MSE ↓", "KL Div ↓", "Linear Probe Acc ↑"]
    n_models = len(df)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#01696f", "#964219", "#a12c7b"]
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=row["model"], color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_title("Benchmark: WAE vs Baseline VAE vs Linear AE", fontsize=13)
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
    import torch
    from wae import WiringAutoencoder
    from wae.dataset import load_dataset, make_loaders
    from sklearn.decomposition import PCA

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]
    cfg["dataset"]["name"] = dataset
    data = load_dataset(dataset, root=cfg["dataset"]["root"], device="cpu")
    E = data["E"]

    model = WiringAutoencoder.from_config(cfg, E)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from wae.laplacian import DifferentiableLaplacian
    base_lap = DifferentiableLaplacian.from_embeddings(E)
    with torch.no_grad():
        base_L = base_lap(base_lap.base_weights.unsqueeze(0)).squeeze(0)

    loaders = make_loaders(data, batch_size=512)
    zs, ys = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            out = model(batch["x"], E, node_idx=batch["node_idx"], base_L=base_L)
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
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=12, alpha=0.6, color=cmap(c), label=f"class {c}")
    ax.set_title("WAE Latent Space (PCA-2D) — coloured by class", fontsize=12)
    ax.legend(markerscale=2, fontsize=9, ncol=2)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = Path(out_dir) / "latent_space.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    pluot_data = {"type": "scatter", "x": Z2[:, 0].tolist(), "y": Z2[:, 1].tolist(), "label": Y.tolist()}
    with open(Path(out_dir) / "latent_space.json", "w") as f:
        json.dump(pluot_data, f)


# ---------------------------------------------------------------------------
# Mode: spectral entropy of generated Laplacians
# ---------------------------------------------------------------------------
def plot_spectral(checkpoint_path: str, dataset: str, out_dir: str, n_samples: int = 200) -> None:
    import torch
    from wae import WiringAutoencoder
    from wae.dataset import load_dataset

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]
    cfg["dataset"]["name"] = dataset
    data = load_dataset(dataset, root=cfg["dataset"]["root"], device="cpu")
    E = data["E"]

    model = WiringAutoencoder.from_config(cfg, E)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        gen = model.generate(E, n_samples=n_samples)
        Ls  = gen["L"]   # (n_samples, N, N)
        eigvals_all = torch.linalg.eigvalsh(Ls).clamp(min=1e-8)  # (n_samples, N)

    # Spectral entropy per sample
    p = eigvals_all / eigvals_all.sum(dim=-1, keepdim=True)
    entropy = -(p * p.log()).sum(dim=-1).numpy()  # (n_samples,)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(entropy, bins=30, color="#01696f", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Spectral Entropy H(Λ)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Spectral Entropy in Generated Laplacians", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(out_dir) / "spectral_entropy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    pluot_data = {"type": "histogram", "values": entropy.tolist(), "metric": "spectral_entropy"}
    with open(Path(out_dir) / "spectral_entropy.json", "w") as f:
        json.dump(pluot_data, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["training", "benchmark", "latent", "spectral"], required=True)
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
        plot_spectral(args.checkpoint, args.dataset, args.output, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
