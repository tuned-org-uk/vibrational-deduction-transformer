"""
experiments/option6_ablation.py  --  Option 6 spectral memory ablation (#18, #29).

Runs four conditions of the VibrationalClassifier (#18) that isolate the
contribution of the SpectralAssociativeMemory pre-initialisation produced
by DeterministicSpectralAE.extract_spectral_artefact() (#20).

Conditions
----------
random_init          No pre-training.  key_matrix kept at orthogonal init.
spectral_memory      key_matrix init from S_I artefact then *frozen*.
spectral_memory_ft   key_matrix init from S_I artefact then fine-tuned.
fine_tune_all        Full VibrationalClassifier unfrozen, no spectral init.

for each condition the script:
  1. Builds a VibrationalClassifier from scratch.
  2. (if spectral_memory*) loads artefact from checkpoint and calls
     classifier.init_from_spectral_memory(memory, freeze=...).
  3. Trains for n_epochs using Adam, logs per-epoch CE + accuracy.
  4. Writes per-epoch metrics to results/option6_<condition>_metrics.csv
     and a final summary to results/option6_summary.json.

CLI
---
    python -m experiments.option6_ablation \\
        --data_path <pt file with dict x, y, L_f> \\
        --artefact_path <pt file from extract_spectral_artefact> \\
        --out_dir results \\
        --n_epochs 50 --batch_size 32 --lr 3e-4

The data file must be a .pt dict with keys:
    x  : (N_total, n_nodes, D)  float32 node feature matrices
    y  : (N_total,)             long    class labels
    L_f: (n_nodes, n_nodes)     float32 frozen index Laplacian

The artefact file is the .pt dict returned by
DeterministicSpectralAE.extract_spectral_artefact().

Ref: docs/v2/03-branching.md -- Option 6 / Track A ablation
Depends on: wae/classifier.py (#18), wae/spectral_memory.py (#28)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wae.classifier import VibrationalClassifier


# ---------------------------------------------------------------------------
# Condition config
# ---------------------------------------------------------------------------

@dataclass
class ConditionConfig:
    """
    Configuration for a single ablation condition.

    Attributes
    ----------
    name : str
        Condition label used in filenames and the summary JSON.
    use_spectral_init : bool
        If True, load key_matrix from the S_I artefact before training.
    freeze_key_matrix : bool
        If True (spectral_memory condition), key_matrix.requires_grad = False.
        Ignored when use_spectral_init is False.
    freeze_vdt : bool
        If True, all VDT weights are frozen except the classification head.
    """
    name: str
    use_spectral_init: bool = False
    freeze_key_matrix: bool = False
    freeze_vdt: bool = False


CONDITIONS: List[ConditionConfig] = [
    ConditionConfig(
        name="random_init",
        use_spectral_init=False,
        freeze_key_matrix=False,
        freeze_vdt=False,
    ),
    ConditionConfig(
        name="spectral_memory",
        use_spectral_init=True,
        freeze_key_matrix=True,
        freeze_vdt=True,
    ),
    ConditionConfig(
        name="spectral_memory_ft",
        use_spectral_init=True,
        freeze_key_matrix=False,
        freeze_vdt=False,
    ),
    ConditionConfig(
        name="fine_tune_all",
        use_spectral_init=False,
        freeze_key_matrix=False,
        freeze_vdt=False,
    ),
]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class AblationLogger:
    """
    Per-condition CSV logger and JSON summary writer.

    Writes one CSV row per epoch with columns::

        epoch, train_ce, train_acc, val_ce, val_acc

    and a final summary JSON with per-condition best_val_acc and
    best_val_epoch.

    Parameters
    ----------
    out_dir : Path
    condition_name : str
    """

    def __init__(self, out_dir: Path, condition_name: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path  = out_dir / f"option6_{condition_name}_metrics.csv"
        self._fh  = open(csv_path, "w", newline="")
        self._csv = csv.writer(self._fh)
        self._csv.writerow(["epoch", "train_ce", "train_acc", "val_ce", "val_acc"])
        self.rows: List[dict] = []

    def log(self, epoch: int, train_ce: float, train_acc: float,
            val_ce: float, val_acc: float) -> None:
        """Write one row."""
        self._csv.writerow([epoch, train_ce, train_acc, val_ce, val_acc])
        self._fh.flush()
        self.rows.append(dict(epoch=epoch, train_ce=train_ce,
                              train_acc=train_acc, val_ce=val_ce,
                              val_acc=val_acc))

    def close(self) -> None:
        """Flush and close the file handle."""
        self._fh.close()

    @property
    def best_val_acc(self) -> float:
        """Best validation accuracy across all logged epochs."""
        if not self.rows:
            return 0.0
        return max(r["val_acc"] for r in self.rows)

    @property
    def best_val_epoch(self) -> int:
        """Epoch at which best_val_acc was achieved."""
        if not self.rows:
            return -1
        return max(self.rows, key=lambda r: r["val_acc"])["epoch"]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: VibrationalClassifier,
    loader: DataLoader,
    L_f: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate cross-entropy and top-1 accuracy on a DataLoader.

    Returns
    -------
    (mean_ce, accuracy) : both floats in [0, inf) and [0, 1].
    """
    model.eval()
    total_ce  = 0.0
    total_ok  = 0
    total_n   = 0
    L_f_dev   = L_f.to(device)

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out     = model(x_batch, L_f_dev)
        logits  = out["logits"]                       # (B, n_classes)
        import torch.nn.functional as F
        ce      = F.cross_entropy(logits, y_batch).item()
        preds   = logits.argmax(dim=-1)
        total_ce  += ce * x_batch.shape[0]
        total_ok  += (preds == y_batch).sum().item()
        total_n   += x_batch.shape[0]

    model.train()
    return total_ce / total_n, total_ok / total_n


def run_ablation_condition(
    cond: ConditionConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    L_f: torch.Tensor,
    n_classes: int,
    input_dim: int,
    d_model: int,
    depth: int,
    n_nodes: int,
    artefact: Optional[dict],
    n_epochs: int,
    lr: float,
    out_dir: Path,
    device: torch.device,
) -> dict:
    """
    Train one condition and return its result summary dict.

    Parameters
    ----------
    cond : ConditionConfig
    train_loader, val_loader : DataLoader
    L_f : Tensor  (n_nodes, n_nodes)  frozen Laplacian on CPU
    n_classes, input_dim, d_model, depth, n_nodes : int
    artefact : dict or None  from DeterministicSpectralAE.extract_spectral_artefact
    n_epochs, lr : int, float
    out_dir : Path
    device : torch.device

    Returns
    -------
    dict: name, best_val_acc, best_val_epoch
    """
    print(f"\n[option6] condition: {cond.name}")

    model = VibrationalClassifier(
        input_dim=input_dim,
        d_model=d_model,
        n_classes=n_classes,
        depth=depth,
        n_nodes=n_nodes,
    ).to(device)

    # Spectral memory initialisation
    if cond.use_spectral_init and artefact is not None:
        # Build a lightweight wrapper so init_from_spectral_memory works
        class _MemoryProxy:
            def __init__(self, art: dict) -> None:
                self.S_memory = art["S_memory"]

        memory = _MemoryProxy(artefact)
        model.init_from_spectral_memory(
            memory, freeze=cond.freeze_key_matrix
        )
        print(f"  key_matrix init from artefact, freeze={cond.freeze_key_matrix}")

    # Optionally freeze VDT weights
    if cond.freeze_vdt:
        for param in model.vdt.parameters():
            param.requires_grad_(False)
        print("  VDT weights frozen")

    # Only optimise parameters that require grad
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=lr)
    logger    = AblationLogger(out_dir, cond.name)
    L_f_dev   = L_f.to(device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_ce_acc  = 0.0
        train_ok      = 0
        train_total   = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimiser.zero_grad(set_to_none=True)

            out  = model(x_batch, L_f_dev)
            loss = model.compute_loss(out, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimiser.step()

            import torch.nn.functional as F
            ce = F.cross_entropy(out["logits"], y_batch).item()
            train_ce_acc  += ce * x_batch.shape[0]
            train_ok      += (out["logits"].argmax(-1) == y_batch).sum().item()
            train_total   += x_batch.shape[0]

        train_ce  = train_ce_acc / train_total
        train_acc = train_ok / train_total

        val_ce, val_acc = evaluate(model, val_loader, L_f, device)
        logger.log(epoch, train_ce, train_acc, val_ce, val_acc)

        if epoch % max(1, n_epochs // 5) == 0:
            print(
                f"  epoch {epoch:>3d}/{n_epochs}  "
                f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
            )

    # Save checkpoint
    ckpt_path = out_dir / f"option6_{cond.name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  checkpoint -> {ckpt_path}")

    logger.close()
    return {
        "name":           cond.name,
        "best_val_acc":   logger.best_val_acc,
        "best_val_epoch": logger.best_val_epoch,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse CLI args and run all four ablation conditions."""
    parser = argparse.ArgumentParser(
        description="Option 6 spectral memory ablation (#18/#29)"
    )
    parser.add_argument(
        "--data_path", required=True,
        help=".pt file with keys: x (N_total, n_nodes, D), y (N_total,), L_f (n_nodes, n_nodes)"
    )
    parser.add_argument(
        "--artefact_path", default=None,
        help=".pt artefact from DeterministicSpectralAE.extract_spectral_artefact()"
    )
    parser.add_argument("--out_dir",    default="results")
    parser.add_argument("--n_epochs",   type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--d_model",    type=int,   default=64)
    parser.add_argument("--depth",      type=int,   default=4)
    parser.add_argument("--val_frac",   type=float, default=0.2)
    parser.add_argument(
        "--conditions", nargs="+",
        default=[c.name for c in CONDITIONS],
        help="Subset of conditions to run (default: all four)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="torch device string (default: cpu)"
    )
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data --------------------------------------------------------
    data  = torch.load(args.data_path, map_location="cpu")
    x_all = data["x"].float()       # (N_total, n_nodes, D)
    y_all = data["y"].long()         # (N_total,)
    L_f   = data["L_f"].float()     # (n_nodes, n_nodes)

    N_total, n_nodes, input_dim = x_all.shape
    n_classes = int(y_all.max().item()) + 1

    n_val   = max(1, int(N_total * args.val_frac))
    n_train = N_total - n_val
    perm    = torch.randperm(N_total)
    tr_idx  = perm[:n_train]
    va_idx  = perm[n_train:]

    train_ds = TensorDataset(x_all[tr_idx], y_all[tr_idx])
    val_ds   = TensorDataset(x_all[va_idx], y_all[va_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # --- Load artefact (optional) ----------------------------------------
    artefact: Optional[dict] = None
    if args.artefact_path is not None:
        artefact = torch.load(args.artefact_path, map_location="cpu")
        print(f"Loaded artefact from {args.artefact_path}")

    # --- Filter conditions -----------------------------------------------
    active_conditions = [
        c for c in CONDITIONS if c.name in args.conditions
    ]

    # --- Run ablation -----------------------------------------------------
    summary: List[dict] = []
    for cond in active_conditions:
        result = run_ablation_condition(
            cond=cond,
            train_loader=train_loader,
            val_loader=val_loader,
            L_f=L_f,
            n_classes=n_classes,
            input_dim=input_dim,
            d_model=args.d_model,
            depth=args.depth,
            n_nodes=n_nodes,
            artefact=artefact,
            n_epochs=args.n_epochs,
            lr=args.lr,
            out_dir=out_dir,
            device=device,
        )
        summary.append(result)

    # --- Write summary JSON -----------------------------------------------
    summary_path = out_dir / "option6_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary -> {summary_path}")
    for r in summary:
        print(
            f"  {r['name']:<25s}  "
            f"best_val_acc={r['best_val_acc']:.4f}  "
            f"best_epoch={r['best_val_epoch']}"
        )


if __name__ == "__main__":
    main()
