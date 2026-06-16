"""
Interactive Visualisation for Spectral Demo Results  (issue #13)
================================================================

Reads the CSV files produced by `demos/spectral_generation_demo.py` and
renders interactive visualisations.  Written to be compatible with
pluot (https://github.com/keller-mark/pluot) for browser-based exploration;
also outputs static PNG fallbacks via matplotlib.

Usage
-----
    python demos/visualise_spectral_demo.py --results results/spectral_demo

Outputs
-------
    results/spectral_demo/figures/interactive_entropy.json
    results/spectral_demo/figures/interactive_spectral_dist.json
    results/spectral_demo/figures/interactive_training.json
    results/spectral_demo/figures/interactive_entropy_control.json
    results/spectral_demo/figures/pluot_manifest.json   <- load this in pluot
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [pluot] {path}")


# ---------------------------------------------------------------------------
# 1. Spectral entropy histogram (dataset vs generated)
# ---------------------------------------------------------------------------
def vis_entropy(results_df: pd.DataFrame, figs: Path) -> None:
    H = results_df["spectral_entropy"].tolist()

    # pluot-compatible JSON
    save_json({
        "type": "histogram",
        "title": "Generated Spectral Entropy Distribution",
        "x_label": "Spectral Entropy H(Λ)",
        "series": [{"name": "Generated", "values": H}]
    }, figs / "interactive_entropy.json")

    # static PNG
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(H, bins=30, color="#964219", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Spectral Entropy H(Λ)", fontsize=11)
    ax.set_title("Generated Spectral Entropy Distribution", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "vis_entropy.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Spectral distance scatter
# ---------------------------------------------------------------------------
def vis_spectral_dist(results_df: pd.DataFrame, figs: Path) -> None:
    H  = results_df["spectral_entropy"].tolist()
    SD = results_df["spectral_distance"].tolist()

    save_json({
        "type": "scatter",
        "title": "Spectral Entropy vs Nearest-Neighbour Distance",
        "x": H, "y": SD,
        "x_label": "Spectral Entropy H(Λ)",
        "y_label": "Frobenius Distance to Nearest Training L",
    }, figs / "interactive_spectral_dist.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(H, SD, s=12, alpha=0.6, color="#a12c7b")
    ax.set_xlabel("Spectral Entropy H(Λ)", fontsize=11)
    ax.set_ylabel("NN Frobenius Distance", fontsize=11)
    ax.set_title("Spectral Entropy vs Nearest-Neighbour Spectral Distance", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "vis_spectral_dist.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Training curves
# ---------------------------------------------------------------------------
def vis_training(log_df: pd.DataFrame, figs: Path) -> None:
    metrics = [
        ("train_loss",   "val_loss",   "Total ELBO Loss"),
        ("train_recon",  "val_recon",  "Reconstruction MSE"),
        ("train_kl",     "val_kl",     "KL Divergence"),
        ("train_jfreq",  "val_jfreq",  "J_freq Spectral Cost"),
    ]
    series = []
    for tr, vl, title in metrics:
        if tr in log_df.columns and vl in log_df.columns:
            series.append({"name": title,
                           "train": log_df[tr].tolist(),
                           "val":   log_df[vl].tolist()})

    save_json({
        "type": "line",
        "title": "VDT Training Curves — Spring Network Demo",
        "x": log_df["epoch"].tolist(),
        "x_label": "Epoch",
        "series": series
    }, figs / "interactive_training.json")

    fig, axes = plt.subplots(1, len(series), figsize=(5 * len(series), 4))
    if len(series) == 1:
        axes = [axes]
    for ax, s in zip(axes, series):
        ax.plot(log_df["epoch"], s["train"], label="train", lw=2)
        ax.plot(log_df["epoch"], s["val"],   label="val",   lw=2, ls="--")
        ax.set_title(s["name"], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "vis_training.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Entropy-targeting results
# ---------------------------------------------------------------------------
def vis_entropy_control(ctrl_df: pd.DataFrame, figs: Path) -> None:
    targets    = ctrl_df["target"].tolist()
    best_err   = ctrl_df["best_err"].tolist()
    match_rate = ctrl_df["match_rate"].tolist()

    save_json({
        "type": "line",
        "title": "Entropy-Controlled Generation",
        "x": targets,
        "x_label": "Target Spectral Entropy",
        "series": [
            {"name": "Best Error",   "values": best_err},
            {"name": "Match Rate",   "values": match_rate},
        ]
    }, figs / "interactive_entropy_control.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(targets, best_err, marker="o", color="#01696f", lw=2)
    ax1.set_xlabel("Target Entropy"); ax1.set_ylabel("|err|")
    ax1.set_title("Best-of-500 Error"); ax1.grid(alpha=0.3)
    ax2.plot(targets, match_rate, marker="s", color="#964219", lw=2)
    ax2.set_xlabel("Target Entropy"); ax2.set_ylabel("Match Rate")
    ax2.set_ylim(0, 1); ax2.set_title("Match Rate (tol=0.05)"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "vis_entropy_control.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. pluot manifest
# ---------------------------------------------------------------------------
def write_pluot_manifest(figs: Path) -> None:
    manifest = {
        "title": "VDT Spectral Generation Demo",
        "description": "Interactive pluot views for the spring-network spectral demo.",
        "views": [
            {"id": "training",        "file": "interactive_training.json",        "type": "line"},
            {"id": "entropy",         "file": "interactive_entropy.json",         "type": "histogram"},
            {"id": "spectral_dist",   "file": "interactive_spectral_dist.json",   "type": "scatter"},
            {"id": "entropy_control", "file": "interactive_entropy_control.json", "type": "line"},
        ]
    }
    save_json(manifest, figs / "pluot_manifest.json")
    print(f"\n  Load {figs}/pluot_manifest.json in pluot for full interactive view.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/spectral_demo")
    args = p.parse_args()

    rd   = Path(args.results)
    figs = rd / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("[vis] Loading CSVs...")
    results_df = load_csv(rd / "spectral_demo_results.csv")
    log_df     = load_csv(rd / "training_log.csv")
    ctrl_df    = load_csv(rd / "entropy_control_results.csv")

    print("[vis] Generating visualisations...")
    vis_training(log_df, figs)
    vis_entropy(results_df, figs)
    vis_spectral_dist(results_df, figs)
    vis_entropy_control(ctrl_df, figs)
    write_pluot_manifest(figs)
    print("[vis] Done.")


if __name__ == "__main__":
    main()
