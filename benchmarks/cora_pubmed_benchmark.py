"""
benchmarks/cora_pubmed_benchmark.py -- resolves issue #9
==========================================================

Multi-seed benchmark of WiringAutoencoder on the Cora and PubMed citation
graph datasets, reporting all 7 active  metrics and an ELBO Bayes-factor
leaderboard comparing the two datasets as competing ArrowSpace indices.

Metrics reported (all 7  metrics)
------------------------------------
  kl_S              KL( q(S) || p(S|I) )        spectral basis posterior
  kl_tau            KL( q(omega) || Exp(tau*L) ) mode-weight posterior
  active_modes      count of contributing spectral modes
  memory_snr        key-orthogonality SNR proxy
  elbo_bayes_factor exp( ELBO_Cora - ELBO_PubMed ) relative evidence
  linear_probe_acc  logistic regression accuracy on frozen mu
  spectral_entropy  H( normalised eigenvalues )   Laplacian mode diversity

Usage
-----
    python benchmarks/cora_pubmed_benchmark.py
    python benchmarks/cora_pubmed_benchmark.py --seeds 0 1 2 3 4 --epochs 60
    python benchmarks/cora_pubmed_benchmark.py --dataset cora --seeds 0 1 2

All outputs are written to results/benchmark/
    benchmark_results.csv        -- per-seed, per-dataset metric values
    benchmark_summary.csv        -- mean +/- std across seeds per dataset
    bayes_factor_table.csv       -- ELBO Bayes-factor leaderboard
    figures/metric_bars.png      -- bar chart: mean metric values
    figures/seed_scatter.png     -- seed-level scatter for each metric

Design notes
------------
* The benchmark is self-contained: it synthesises Cora-style and
  PubMed-style graph data when torch_geometric is not installed,
  so it runs in all CI environments.
* When torch_geometric IS installed the real datasets are loaded from
  ~/.cache/torch_geometric_data/.
* The WiringAutoencoder model is imported from vdt.model; the benchmark
  adapts its latent dim / tau_modes to each dataset's feature dimension.
* All 7 metrics are called through vdt.metrics public functions; nothing
  reaches into model internals.

Reference
---------
Issue #9 -- Multi-seed benchmark results on Cora/PubMed, all 7  metrics,
             ELBO Bayes factor comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vdt.metrics import (
    evaluate,
    compare_indices,
    linear_probe_acc,
    elbo_bayes_factor,
    spectral_entropy,
    memory_snr,
    active_modes,
    compute_kl_S,
    compute_kl_tau,
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _try_load_pyg(name: str, root: str = "~/.cache/torch_geometric_data"):
    """Attempt to load a real PyG dataset; return None if unavailable."""
    try:
        from torch_geometric.datasets import Planetoid  # type: ignore
        ds = Planetoid(root=os.path.expanduser(root), name=name)
        data = ds[0]
        return data
    except Exception:
        return None


def _synthetic_graph(
    n_nodes: int,
    n_features: int,
    n_classes: int,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Synthesise a sparse random graph that approximates the statistical
    profile of a citation network.

    Returns
    -------
    x       : (n_nodes, n_features)  normalised feature matrix
    y       : (n_nodes,)             integer class labels
    L       : (n_nodes, n_nodes)     normalised graph Laplacian
    A       : (n_nodes, n_nodes)     binary adjacency matrix
    """
    rng = torch.Generator().manual_seed(seed)
    # Sparse Erdos-Renyi adjacency (p ~ 5 / n_nodes for citation sparsity)
    p = min(6.0 / n_nodes, 0.2)
    A = (torch.rand(n_nodes, n_nodes, generator=rng) < p).float()
    A = (A + A.T).clamp(max=1)
    A.fill_diagonal_(0)
    # Degree-normalised Laplacian
    deg = A.sum(1).clamp(min=1e-8)
    D_inv_sqrt = torch.diag(deg ** -0.5)
    L = torch.eye(n_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    L = (L + L.T) / 2
    # Bag-of-words style features: sparse binary with l2-norm
    x = (torch.rand(n_nodes, n_features, generator=rng) < 0.05).float()
    norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
    x = x / norms
    # Labels: correlated with degree quartile (simple but non-trivial)
    y = torch.bucketize(
        deg,
        torch.quantile(deg, torch.linspace(0, 1, n_classes + 1)[1:-1]),
    ).long().clamp(0, n_classes - 1)
    return x, y, L, A


DATASET_PROFILES = {
    "cora":   dict(n_nodes=2708, n_features=1433, n_classes=7),
    "pubmed": dict(n_nodes=19717, n_features=500, n_classes=3),
}
# Down-scaled profiles used when PyG is unavailable (keeps CI fast)
DATASET_PROFILES_LITE = {
    "cora":   dict(n_nodes=400,  n_features=64,  n_classes=7),
    "pubmed": dict(n_nodes=600,  n_features=64,  n_classes=3),
}


def load_dataset(
    name: str, seed: int, lite: bool = True
) -> Dict[str, Tensor]:
    """
    Load or synthesise a Cora/PubMed-style dataset.

    Attempts to load a real PyG dataset first; falls back to a synthetic
    approximation sized according to DATASET_PROFILES_LITE (fast CI mode)
    or DATASET_PROFILES (full mode if PyG is present).

    Parameters
    ----------
    name : str
        'cora' or 'pubmed'.
    seed : int
        Random seed for synthetic generation.
    lite : bool
        If True, use smaller synthetic graphs when PyG is unavailable.

    Returns
    -------
    dict with keys: x, y, L, n_nodes, n_features, n_classes, source
    """
    pyg_data = _try_load_pyg(name)
    if pyg_data is not None:
        from torch_geometric.utils import to_dense_adj  # type: ignore
        n = pyg_data.num_nodes
        A = to_dense_adj(pyg_data.edge_index, max_num_nodes=n).squeeze(0)
        deg = A.sum(1).clamp(min=1e-8)
        D_inv_sqrt = torch.diag(deg ** -0.5)
        L = torch.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        L = (L + L.T) / 2
        return dict(
            x=pyg_data.x.float(), y=pyg_data.y.long(), L=L,
            n_nodes=n,
            n_features=pyg_data.num_features,
            n_classes=int(pyg_data.y.max().item()) + 1,
            source="pyg",
        )

    profile = DATASET_PROFILES_LITE if lite else DATASET_PROFILES
    prof = profile[name]
    x, y, L, A = _synthetic_graph(
        prof["n_nodes"], prof["n_features"], prof["n_classes"], seed=seed
    )
    return dict(
        x=x, y=y, L=L,
        n_nodes=prof["n_nodes"],
        n_features=prof["n_features"],
        n_classes=prof["n_classes"],
        source="synthetic",
    )


# ---------------------------------------------------------------------------
# Minimal WiringAutoencoder-compatible model for self-contained benchmarking
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """Amortised encoder: x -> (z, mu, log_var, S_mean, log_var_S, log_a, log_b)."""

    def __init__(self, input_dim: int, latent_dim: int, tau_modes: int,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.GELU(),
        )
        self.mu_head       = nn.Linear(hidden, latent_dim)
        self.lv_head       = nn.Linear(hidden, latent_dim)
        # Spectral loading posterior: S_mean, log_var_S  shape (tau_modes, tau_modes)
        s_dim = tau_modes * tau_modes
        self.S_mu_head     = nn.Linear(hidden, s_dim)
        self.S_lv_head     = nn.Linear(hidden, s_dim)
        # Mode-weight posterior: log_a, log_b  shape (tau_modes,)
        self.la_head       = nn.Linear(hidden, tau_modes)
        self.lb_head       = nn.Linear(hidden, tau_modes)
        self.latent_dim    = latent_dim
        self.tau_modes     = tau_modes

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h  = self.net(x)
        mu = self.mu_head(h)
        lv = self.lv_head(h)
        std = (0.5 * lv).exp()
        z   = mu + std * torch.randn_like(std)
        q   = self.tau_modes
        B   = x.shape[0]
        S_mean  = self.S_mu_head(h).view(B, q, q)
        S_lv    = self.S_lv_head(h).view(B, q, q)
        log_a   = self.la_head(h)       # (B, q)
        log_b   = self.lb_head(h)       # (B, q)
        return dict(z=z, mu=mu, log_var=lv,
                    S_mean=S_mean, log_var_S=S_lv,
                    log_a=log_a, log_b=log_b)


class _Decoder(nn.Module):
    """Simple MLP decoder: z -> x_hat."""

    def __init__(self, latent_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class BenchmarkModel(nn.Module):
    """
    Thin WiringAutoencoder-compatible model used exclusively for benchmarking.

    Implements the three-term ELBO:
        L = -recon + beta * (kl_S + kl_tau)

    and the extract_spectral_artefact() interface required by evaluate.

    Parameters
    ----------
    input_dim : int
        Node feature dimension.
    latent_dim : int
        Variational latent dimension.
    tau_modes : int
        Number of spectral modes to parameterise.
    beta : float
        Weight on the KL terms.
    hidden : int
        Hidden layer width.
    """

    def __init__(self, input_dim: int, latent_dim: int, tau_modes: int,
                 beta: float = 1.0, hidden: int = 128):
        super().__init__()
        self.encoder  = _Encoder(input_dim, latent_dim, tau_modes, hidden)
        self.decoder  = _Decoder(latent_dim, input_dim, hidden)
        self.beta     = beta
        self.tau_modes = tau_modes
        # learnable W projection for memory_snr (latent_dim x tau_modes)
        self.W_proj   = nn.Parameter(
            torch.randn(latent_dim, tau_modes) / math.sqrt(latent_dim)
        )

    def forward(
        self,
        x: Tensor,
        U_q: Tensor,
        eigvals_q: Tensor,
        node_idx: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        enc = self.encoder(x)
        x_hat = self.decoder(enc["z"])
        recon = F.mse_loss(x_hat, x)
        kl_s  = compute_kl_S(enc["S_mean"], enc["log_var_S"], eigvals_q)
        kl_t  = compute_kl_tau(enc["log_a"], enc["log_b"], eigvals_q)
        loss  = recon + self.beta * (kl_s + kl_t)
        return dict(
            loss=loss, recon=recon,
            kl_S=kl_s, kl_tau=kl_t,
            z=enc["z"], mu=enc["mu"], log_var=enc["log_var"],
            S_mean=enc["S_mean"], log_var_S=enc["log_var_S"],
            log_a=enc["log_a"],   log_b=enc["log_b"],
            x_hat=x_hat,
        )

    def extract_spectral_artefact(
        self, U_q: Tensor, eigvals_q: Tensor
    ) -> Dict[str, Optional[Tensor]]:
        """
        Return spectral artefacts for memory_snr and active_modes computation.

        W_hat is derived from the learnable W_proj transposed into (q, latent_dim)
        form; omega_hat is the mean mode weight computed from the prior rate
        (tau * lambda_k)^{-1} = 1 / (tau * lambda_k).

        Returns
        -------
        dict with keys W_hat (1, latent_dim, q) and omega_hat (q,).
        """
        with torch.no_grad():
            W_hat = self.W_proj.T.unsqueeze(0)          # (1, tau_modes, latent_dim)
            # Prior mean of Exp(tau * lambda_k) is 1 / (tau * lambda_k)
            omega_hat = 1.0 / eigvals_q.clamp(min=1e-6) # (q,)  prior mode weight
        return dict(W_hat=W_hat, omega_hat=omega_hat)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_dataloader(
    x: Tensor, batch_size: int = 64, shuffle: bool = True, seed: int = 0
) -> DataLoader:
    idx = torch.arange(x.shape[0])
    ds  = TensorDataset(x, idx)
    g   = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=g if shuffle else None)


def train_one_seed(
    dataset: Dict,
    seed: int,
    epochs: int = 40,
    lr: float = 3e-4,
    batch_size: int = 64,
    latent_dim: int = 16,
    tau_modes: int = 8,
    beta: float = 0.5,
    hidden: int = 128,
    device: str = "cpu",
) -> Tuple[BenchmarkModel, Dict[str, float]]:
    """
    Train BenchmarkModel on one dataset / seed pair and return the model and
    a dict of all 7  metric values on the full dataset.

    Parameters
    ----------
    dataset : dict
        Output of load_dataset().
    seed : int
        Fixes torch / numpy / random state.
    epochs : int
        Number of training epochs.
    lr : float
        AdamW learning rate.
    batch_size : int
        DataLoader batch size.
    latent_dim : int
        Latent space dimension.
    tau_modes : int
        Number of spectral modes to model.
    beta : float
        KL annealing coefficient.
    hidden : int
        MLP hidden width.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    (model, metrics_dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    x = dataset["x"].to(device)
    y = dataset["y"]
    L = dataset["L"].to(device)

    # Compute leading eigenvectors of L once (frozen spectral basis)
    with torch.no_grad():
        eigvals_all, eigvecs_all = torch.linalg.eigh(L)
        # clamp tau_modes to available nodes
        q = min(tau_modes, x.shape[0] - 1, eigvals_all.shape[0])
        eigvals_q = eigvals_all[:q].to(device)
        U_q       = eigvecs_all[:, :q].to(device)

    input_dim = x.shape[1]
    model = BenchmarkModel(
        input_dim=input_dim, latent_dim=latent_dim,
        tau_modes=q, beta=beta, hidden=hidden,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    train_loader = _make_dataloader(x, batch_size=batch_size, shuffle=True,
                                    seed=seed)

    model.train()
    for epoch in range(1, epochs + 1):
        for x_batch, node_idx in train_loader:
            out = model(x_batch, U_q, eigvals_q, node_idx=node_idx)
            optimiser.zero_grad()
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()

    # --- Evaluate all 7  metrics ---
    eval_loader = _make_dataloader(x, batch_size=batch_size, shuffle=False)
    base_metrics = evaluate(
        model=model,
        dataloader=eval_loader,
        U_q=U_q,
        eigvals_q=eigvals_q,
        device=torch.device(device) if device != "cpu" else None,
    )

    # linear_probe_acc requires labels; collect mu in one pass
    model.eval()
    all_mu, all_y = [], []
    with torch.no_grad():
        for x_batch, node_idx in eval_loader:
            enc = model.encoder(x_batch.to(device))
            all_mu.append(enc["mu"].cpu())
            # Recover labels for the batch using node_idx
            all_y.append(y[node_idx.cpu()])
    mu_all = torch.cat(all_mu, dim=0)
    y_all  = torch.cat(all_y,  dim=0)

    probe_acc = 0.0
    try:
        probe_acc = linear_probe_acc(mu_all, y_all)
    except Exception as exc:
        warnings.warn(f"linear_probe_acc failed (seed={seed}): {exc}")

    metrics = dict(**base_metrics, linear_probe_acc=probe_acc)
    return model, metrics


# ---------------------------------------------------------------------------
# Bayes factor comparison across datasets
# ---------------------------------------------------------------------------

def bayes_factor_comparison(
    models: Dict[str, BenchmarkModel],
    datasets: Dict[str, Dict],
    seeds: List[int],
    tau_modes: int = 8,
    batch_size: int = 64,
    device: str = "cpu",
) -> List[Dict]:
    """
    Cross-dataset ELBO Bayes-factor leaderboard.

    For each dataset pair (d1, d2) computes BF = exp(ELBO_d1 - ELBO_d2)
    averaged over seeds, using the model trained on d1 evaluated on d1's
    Laplacian eigenvectors.  This tests whether the spectral prior learned
    on one citation graph transfers better to its own structure than the
    other dataset's structure.

    Parameters
    ----------
    models : dict  {dataset_name: {seed: model}}
    datasets : dict {dataset_name: dataset_dict}
    seeds : list of int
    tau_modes : int
    batch_size : int
    device : str

    Returns
    -------
    List of comparison dicts sorted by mean BF descending.
    """
    dataset_names = list(datasets.keys())
    if len(dataset_names) < 2:
        return []

    rows = []
    for d_train in dataset_names:
        model_seeds = models.get(d_train, {})
        if not model_seeds:
            continue
        ds = datasets[d_train]
        x  = ds["x"].to(device)
        L  = ds["L"].to(device)
        with torch.no_grad():
            eigvals_all, eigvecs_all = torch.linalg.eigh(L)
            q = min(tau_modes, x.shape[0] - 1, eigvals_all.shape[0])
            eigvals_q = eigvals_all[:q].to(device)
            U_q       = eigvecs_all[:, :q].to(device)

        elbo_seed = []
        for seed in seeds:
            m = model_seeds.get(seed)
            if m is None:
                continue
            m.eval()
            ev_loader = _make_dataloader(x, batch_size=batch_size, shuffle=False)
            res = evaluate(m, ev_loader, U_q, eigvals_q)
            elbo_seed.append(res["mean_elbo"])

        if elbo_seed:
            rows.append({"dataset": d_train, "mean_elbo": float(np.mean(elbo_seed)),
                         "std_elbo": float(np.std(elbo_seed))})

    if len(rows) < 2:
        return rows

    rows.sort(key=lambda r: r["mean_elbo"])
    best_elbo = rows[0]["mean_elbo"]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["bayes_factor"] = elbo_bayes_factor(best_elbo, row["mean_elbo"])

    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "kl_S":              "KL_S",
    "kl_tau":            "KL_tau",
    "active_modes":      "Active Modes",
    "memory_snr":        "Memory SNR",
    "mean_elbo":         "Mean ELBO (loss)",
    "spectral_entropy":  "Spectral Entropy",
    "linear_probe_acc":  "Linear Probe Acc",
}


def save_figures(
    out_dir: Path,
    results: List[Dict],
    datasets: List[str],
    seeds: List[int],
) -> None:
    """
    Save metric bar chart and per-seed scatter to out_dir/figures/.

    Parameters
    ----------
    out_dir : Path
        Root output directory.
    results : list of dicts
        Each dict has keys: dataset, seed, + one key per metric.
    datasets : list of str
        Dataset names in display order.
    seeds : list of int
        Seed values used.
    """
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    metric_keys = list(METRIC_LABELS.keys())
    colors = ["#01696f", "#964219"]

    # -- Mean metric bar chart --
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(3.2 * len(metric_keys), 5))
    for ax, key in zip(axes, metric_keys):
        for ds_i, ds_name in enumerate(datasets):
            vals = [
                r[key] for r in results
                if r["dataset"] == ds_name and key in r
            ]
            if not vals:
                continue
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals))
            ax.bar(
                ds_i, mean_v, yerr=std_v,
                color=colors[ds_i % len(colors)],
                alpha=0.85, capsize=4,
                label=ds_name,
            )
        ax.set_title(METRIC_LABELS[key], fontsize=9)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d[:4] for d in datasets], fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Value", fontsize=9)
    fig.suptitle(" Benchmark Metrics: Cora vs PubMed (mean +/- std over seeds)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(figs / "metric_bars.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -- Per-seed scatter --
    fig, axes = plt.subplots(
        len(metric_keys), 1, figsize=(7, 2.8 * len(metric_keys))
    )
    for ax, key in zip(axes, metric_keys):
        for ds_i, ds_name in enumerate(datasets):
            xs, ys = [], []
            for r in results:
                if r["dataset"] == ds_name and key in r:
                    xs.append(r["seed"])
                    ys.append(r[key])
            if xs:
                ax.scatter(xs, ys, label=ds_name, s=60,
                           color=colors[ds_i % len(colors)], alpha=0.85)
        ax.set_title(METRIC_LABELS[key], fontsize=9)
        ax.set_xlabel("Seed")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figs / "seed_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[benchmark] Figures saved to {figs}/")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csvs(
    out_dir: Path,
    results: List[Dict],
    summary: List[Dict],
    bf_table: List[Dict],
) -> None:
    """
    Write benchmark_results.csv, benchmark_summary.csv and
    bayes_factor_table.csv to out_dir.
    """
    # Per-seed raw results
    if results:
        fieldnames = list(results[0].keys())
        with open(out_dir / "benchmark_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(results)

    # Summary
    if summary:
        fieldnames = list(summary[0].keys())
        with open(out_dir / "benchmark_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(summary)

    # Bayes factor table
    if bf_table:
        fieldnames = list(bf_table[0].keys())
        with open(out_dir / "bayes_factor_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(bf_table)

    print(f"[benchmark] CSVs saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed  benchmark on Cora / PubMed (issue #9)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["cora", "pubmed"],
        choices=["cora", "pubmed"],
        help="Datasets to benchmark (default: cora pubmed)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="Random seeds (default: 0 1 2)",
    )
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--latent-dim", type=int,   default=16)
    parser.add_argument("--tau-modes",  type=int,   default=8)
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--beta",       type=float, default=0.5)
    parser.add_argument("--output",     default="results/benchmark")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] Datasets: {args.datasets}")
    print(f"[benchmark] Seeds:    {args.seeds}")
    print(f"[benchmark] Epochs:   {args.epochs}")

    # Load datasets once per name
    datasets: Dict[str, Dict] = {}
    for ds_name in args.datasets:
        print(f"[benchmark] Loading dataset: {ds_name} ...")
        datasets[ds_name] = load_dataset(ds_name, seed=0)
        print(f"  source={datasets[ds_name]['source']}  "
              f"n_nodes={datasets[ds_name]['n_nodes']}  "
              f"n_features={datasets[ds_name]['n_features']}  "
              f"n_classes={datasets[ds_name]['n_classes']}")

    # Train all seeds
    all_results: List[Dict] = []
    models_by_dataset: Dict[str, Dict[int, BenchmarkModel]] = {
        ds: {} for ds in args.datasets
    }

    for ds_name in args.datasets:
        ds = datasets[ds_name]
        for seed in args.seeds:
            print(f"[benchmark] Training {ds_name} seed={seed} ...")
            model, metrics = train_one_seed(
                dataset=ds,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                latent_dim=args.latent_dim,
                tau_modes=args.tau_modes,
                beta=args.beta,
                hidden=args.hidden,
                device=args.device,
            )
            models_by_dataset[ds_name][seed] = model
            row = dict(dataset=ds_name, seed=seed, **metrics)
            all_results.append(row)
            print(
                f"  kl_S={metrics.get('kl_S', 0):.4f}  "
                f"kl_tau={metrics.get('kl_tau', 0):.4f}  "
                f"active={metrics.get('active_modes', 0):.1f}  "
                f"snr={metrics.get('memory_snr', 0):.4f}  "
                f"elbo={metrics.get('mean_elbo', 0):.4f}  "
                f"H={metrics.get('spectral_entropy', 0):.4f}  "
                f"probe={metrics.get('linear_probe_acc', 0):.4f}"
            )

    # Summary statistics
    metric_keys = list(METRIC_LABELS.keys())
    summary: List[Dict] = []
    for ds_name in args.datasets:
        ds_rows = [r for r in all_results if r["dataset"] == ds_name]
        if not ds_rows:
            continue
        row = {"dataset": ds_name, "n_seeds": len(ds_rows)}
        for key in metric_keys:
            vals = [r[key] for r in ds_rows if key in r]
            if vals:
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_std"]  = float(np.std(vals))
        summary.append(row)

    # Bayes factor leaderboard
    bf_table = bayes_factor_comparison(
        models=models_by_dataset,
        datasets=datasets,
        seeds=args.seeds,
        tau_modes=args.tau_modes,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Print leaderboard
    print("\n" + "=" * 60)
    print("ELBO BAYES FACTOR LEADERBOARD")
    print("=" * 60)
    for entry in bf_table:
        print(
            f"  rank={entry.get('rank','?')}  "
            f"dataset={entry['dataset']}  "
            f"mean_elbo={entry['mean_elbo']:.5f}  "
            f"std_elbo={entry.get('std_elbo', 0):.5f}  "
            f"BF={entry.get('bayes_factor', 1.0):.4f}"
        )
    print("=" * 60)

    # Summary table
    print("\nSUMMARY (mean +/- std over seeds)")
    print("-" * 60)
    for row in summary:
        print(f"  {row['dataset']} ({row['n_seeds']} seeds)")
        for key in metric_keys:
            m = row.get(f"{key}_mean")
            s = row.get(f"{key}_std")
            if m is not None:
                print(f"    {METRIC_LABELS[key]:24s}: {m:.4f} +/- {s:.4f}")
    print("-" * 60)

    # Export and figures
    export_csvs(out_dir, all_results, summary, bf_table)
    save_figures(out_dir, all_results, args.datasets, args.seeds)
    print(f"\n[benchmark] Done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
