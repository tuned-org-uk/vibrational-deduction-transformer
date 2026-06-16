"""
experiments/option6_ablation.py  --  Option 6 spectral memory ablation (#18, #29).

Runs three conditions of the VibrationalClassifier (#18) and a K-depth sweep
that together isolate the contribution of the SpectralAssociativeMemory
pre-initialisation produced by DeterministicSpectralAE.extract_spectral_artefact()
(#20).

Conditions (3)
--------------
random_init          No pre-training.  key_matrix kept at orthogonal init.
spectral_memory      key_matrix init from S_I artefact then *frozen*.
spectral_memory_ft   key_matrix init from S_I artefact then fine-tuned.

K-depth sweep
-------------
For each condition above, the VDT depth is swept over K in {2, 4, 8, 16}.
The sweep is controlled by --k_values (default: 2 4 8 16).

Artefact loading
----------------
The spectral artefact can be supplied in two ways:

  1. --artefact_path <path>  : a .pt dict produced directly by
     DeterministicSpectralAE.extract_spectral_artefact() and saved to
     checkpoints/option1_artefact.pt by option1_train.py.

  2. --ckpt_path <path>      : a WiringAutoencoderV2 checkpoint (.pt) that
     contains a full model state_dict.  The script rebuilds the model and
     calls model.extract_spectral_artefact() to produce the artefact.
     Requires --ckpt_dim (latent_dim used when saving the checkpoint).

Exactly one of --artefact_path or --ckpt_path must be supplied when any
spectral_memory* condition is enabled.

For each condition + K combination the script:
  1. Builds a VibrationalClassifier with depth=K.
  2. (if spectral_memory*) loads artefact and calls
     classifier.init_from_spectral_memory(memory, freeze=...).
  3. Trains for n_epochs using Adam, logs per-epoch CE + accuracy.
  4. Writes per-epoch metrics to results/option6_<condition>_K<k>_metrics.csv
     and a final summary to results/option6_summary.json.

CLI
---
    python -m experiments.option6_ablation \\
        --data_path <pt file with dict x, y, L_f> \\
        --artefact_path checkpoints/option1_artefact.pt \\
        --out_dir results \\
        --n_epochs 50 --batch_size 32 --lr 3e-4 \\
        --k_values 2 4 8 16

    # -- or -- load artefact from a full checkpoint:
    python -m experiments.option6_ablation \\
        --data_path <pt file> \\
        --ckpt_path checkpoints/option1_best.pt \\
        --ckpt_dim 16 \\
        --out_dir results

The data file must be a .pt dict with keys:
    x  : (N_total, n_nodes, D)  float32 node feature matrices
    y  : (N_total,)             long    class labels
    L_f: (n_nodes, n_nodes)     float32 frozen index Laplacian

Ref: docs/v2/03-branching.md -- Option 6 / Track A ablation
Depends on: wae/classifier.py (#18), wae/spectral_memory.py (#28), wae/vib_autoencoder.py (#20)
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

    Three conditions are defined (fine_tune_all is excluded -- it lacks
    a clear spectral hypothesis and confounds the ablation):

        random_init        No spectral pre-init.  Orthogonal key_matrix.
        spectral_memory    Spectral init, key_matrix frozen.
        spectral_memory_ft Spectral init, key_matrix fine-tuned.

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


# Three conditions: fine_tune_all removed (#29).
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
]

# Default K-depth sweep values (#29 AC).
DEFAULT_K_VALUES: List[int] = [2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class AblationLogger:
    """
    Per-condition CSV logger and JSON summary writer.

    Writes one CSV row per epoch with columns::

        epoch, train_ce, train_acc, val_ce, val_acc

    and a final summary JSON with per-condition best_val_acc,
    best_val_epoch, condition, and K.

    Parameters
    ----------
    out_dir : Path
    condition_name : str
    k : int  VDT depth for this run
    """

    def __init__(self, out_dir: Path, condition_name: str, k: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path  = out_dir / f"option6_{condition_name}_K{k}_metrics.csv"
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
# Artefact loading helpers
# ---------------------------------------------------------------------------

def load_artefact_from_path(artefact_path: str) -> dict:
    """
    Load a spectral artefact dict directly from a .pt file.

    The file must have been produced by
    DeterministicSpectralAE.extract_spectral_artefact() and saved to
    ``checkpoints/option1_artefact.pt`` by option1_train.py.

    Parameters
    ----------
    artefact_path : str  path to the .pt artefact file

    Returns
    -------
    dict with keys: W_hat, omega_hat, S_memory, eigvals_q
    """
    artefact = torch.load(artefact_path, map_location="cpu")
    required = {"W_hat", "omega_hat", "S_memory", "eigvals_q"}
    missing  = required - set(artefact.keys())
    if missing:
        raise KeyError(
            f"Artefact file {artefact_path} is missing keys: {missing}. "
            f"Re-generate using option1_train.py."
        )
    return artefact


def load_artefact_from_checkpoint(
    ckpt_path: str,
    ckpt_dim: int,
    L_f: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Rebuild a DeterministicSpectralAE from a WiringAutoencoderV2
    checkpoint and call extract_spectral_artefact().

    Parameters
    ----------
    ckpt_path : str   path to the .pt model checkpoint
    ckpt_dim  : int   latent_dim used when the checkpoint was saved
    L_f : Tensor  (N, N)  frozen base Laplacian (CPU)
    device : torch.device

    Returns
    -------
    dict with keys: W_hat, omega_hat, S_memory, eigvals_q
    """
    from wae.vib_autoencoder import DeterministicSpectralAE

    state = torch.load(ckpt_path, map_location="cpu")
    N     = L_f.shape[0]

    # Infer input_dim from the first Linear weight in the checkpoint
    input_dim = None
    for key, val in state.items():
        if "encoder" in key and val.ndim == 2:
            input_dim = val.shape[1]
            break
    if input_dim is None:
        raise RuntimeError(
            f"Cannot infer input_dim from checkpoint {ckpt_path}. "
            f"Inspect state_dict keys manually."
        )

    model = DeterministicSpectralAE(
        input_dim=input_dim,
        latent_dim=ckpt_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Compute eigenbasis for extract_spectral_artefact
    eigvals_full, eigvecs_full = torch.linalg.eigh(L_f.to(device))
    eigvals_q = eigvals_full[1:ckpt_dim + 1]
    U_q       = eigvecs_full[:, 1:ckpt_dim + 1]

    with torch.no_grad():
        artefact = model.extract_spectral_artefact(
            U_q=U_q,
            L_base=L_f.to(device),
            eigvals_q=eigvals_q,
        )
    return artefact


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
    import torch.nn.functional as F
    model.eval()
    total_ce = total_ok = total_n = 0
    L_f_dev  = L_f.to(device)

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out     = model(x_batch, L_f_dev)
        logits  = out["logits"]
        ce      = F.cross_entropy(logits, y_batch).item()
        preds   = logits.argmax(dim=-1)
        total_ce  += ce * x_batch.shape[0]
        total_ok  += (preds == y_batch).sum().item()
        total_n   += x_batch.shape[0]

    model.train()
    return total_ce / total_n, total_ok / total_n


def run_ablation_condition(
    cond: ConditionConfig,
    k: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    L_f: torch.Tensor,
    n_classes: int,
    input_dim: int,
    d_model: int,
    n_nodes: int,
    artefact: Optional[dict],
    n_epochs: int,
    lr: float,
    out_dir: Path,
    device: torch.device,
) -> dict:
    """
    Train one (condition, K) cell and return its result summary dict.

    Parameters
    ----------
    cond : ConditionConfig
    k : int   VDT depth for this run (from the K-sweep)
    train_loader, val_loader : DataLoader
    L_f : Tensor  (n_nodes, n_nodes)  frozen Laplacian on CPU
    n_classes, input_dim, d_model, n_nodes : int
    artefact : dict or None  from DeterministicSpectralAE.extract_spectral_artefact
    n_epochs, lr : int, float
    out_dir : Path
    device : torch.device

    Returns
    -------
    dict: condition, k, best_val_acc, best_val_epoch
    """
    print(f"\n[option6] condition={cond.name}  K={k}")

    model = VibrationalClassifier(
        input_dim=input_dim,
        d_model=d_model,
        n_classes=n_classes,
        depth=k,
        n_nodes=n_nodes,
    ).to(device)

    # Spectral memory initialisation
    if cond.use_spectral_init and artefact is not None:
        class _MemoryProxy:
            def __init__(self, art: dict) -> None:
                self.S_memory = art["S_memory"]
        model.init_from_spectral_memory(
            _MemoryProxy(artefact), freeze=cond.freeze_key_matrix
        )
        print(f"  key_matrix init from artefact, freeze={cond.freeze_key_matrix}")

    if cond.freeze_vdt:
        for param in model.vdt.parameters():
            param.requires_grad_(False)
        print("  VDT weights frozen")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=lr)
    logger    = AblationLogger(out_dir, cond.name, k)
    L_f_dev   = L_f.to(device)

    for epoch in range(1, n_epochs + 1):
        import torch.nn.functional as F
        model.train()
        train_ce_acc = train_ok = train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimiser.zero_grad(set_to_none=True)

            out  = model(x_batch, L_f_dev)
            loss = model.compute_loss(out, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimiser.step()

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

    ckpt_path = out_dir / f"option6_{cond.name}_K{k}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  checkpoint -> {ckpt_path}")

    logger.close()
    return {
        "condition":      cond.name,
        "k":              k,
        "best_val_acc":   logger.best_val_acc,
        "best_val_epoch": logger.best_val_epoch,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse CLI args and run all conditions x K-sweep cells."""
    parser = argparse.ArgumentParser(
        description="Option 6 spectral memory ablation (#18/#29)"
    )
    parser.add_argument(
        "--data_path", required=True,
        help=".pt file with keys: x (N_total, n_nodes, D), y (N_total,), L_f (n_nodes, n_nodes)"
    )

    artefact_group = parser.add_mutually_exclusive_group()
    artefact_group.add_argument(
        "--artefact_path", default=None,
        help="Direct .pt artefact from DeterministicSpectralAE.extract_spectral_artefact() "
             "(saved to checkpoints/option1_artefact.pt by option1_train.py)."
    )
    artefact_group.add_argument(
        "--ckpt_path", default=None,
        help="WiringAutoencoderV2 checkpoint (.pt).  Rebuilt and "
             "extract_spectral_artefact() called internally.  Requires --ckpt_dim."
    )
    parser.add_argument(
        "--ckpt_dim", type=int, default=None,
        help="latent_dim used when saving --ckpt_path.  Required with --ckpt_path."
    )

    parser.add_argument("--out_dir",    default="results")
    parser.add_argument("--n_epochs",   type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--d_model",    type=int,   default=64)
    parser.add_argument("--val_frac",   type=float, default=0.2)
    parser.add_argument(
        "--k_values", nargs="+", type=int, default=DEFAULT_K_VALUES,
        help="VDT depth values to sweep over (default: 2 4 8 16)."
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=[c.name for c in CONDITIONS],
        help="Subset of conditions to run (default: all three).",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Validate ckpt_path usage
    if args.ckpt_path is not None and args.ckpt_dim is None:
        parser.error("--ckpt_dim is required when --ckpt_path is supplied.")

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data -------------------------------------------------------
    data  = torch.load(args.data_path, map_location="cpu")
    x_all = data["x"].float()       # (N_total, n_nodes, D)
    y_all = data["y"].long()         # (N_total,)
    L_f   = data["L_f"].float()     # (n_nodes, n_nodes)

    N_total, n_nodes, input_dim = x_all.shape
    n_classes = int(y_all.max().item()) + 1

    n_val    = max(1, int(N_total * args.val_frac))
    n_train  = N_total - n_val
    perm     = torch.randperm(N_total)
    tr_idx   = perm[:n_train]
    va_idx   = perm[n_train:]

    train_loader = DataLoader(
        TensorDataset(x_all[tr_idx], y_all[tr_idx]),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_all[va_idx], y_all[va_idx]),
        batch_size=args.batch_size, shuffle=False
    )

    # --- Load artefact ---------------------------------------------------
    artefact: Optional[dict] = None
    active_conditions = [c for c in CONDITIONS if c.name in args.conditions]
    needs_artefact = any(c.use_spectral_init for c in active_conditions)

    if needs_artefact:
        if args.artefact_path is not None:
            artefact = load_artefact_from_path(args.artefact_path)
            print(f"Loaded artefact from {args.artefact_path}")
        elif args.ckpt_path is not None:
            artefact = load_artefact_from_checkpoint(
                args.ckpt_path, args.ckpt_dim, L_f, device
            )
            print(f"Extracted artefact from checkpoint {args.ckpt_path}")
        else:
            raise ValueError(
                "Conditions requiring spectral init are active but neither "
                "--artefact_path nor --ckpt_path was supplied."
            )
    else:
        print("No spectral-init conditions active; artefact not loaded.")

    # --- Run K-sweep x conditions ----------------------------------------
    summary: List[dict] = []
    for k in args.k_values:
        for cond in active_conditions:
            result = run_ablation_condition(
                cond=cond,
                k=k,
                train_loader=train_loader,
                val_loader=val_loader,
                L_f=L_f,
                n_classes=n_classes,
                input_dim=input_dim,
                d_model=args.d_model,
                n_nodes=n_nodes,
                artefact=artefact,
                n_epochs=args.n_epochs,
                lr=args.lr,
                out_dir=out_dir,
                device=device,
            )
            summary.append(result)

    # --- Write summary JSON ----------------------------------------------
    summary_path = out_dir / "option6_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary -> {summary_path}")
    print(f"{'condition':<25s}  {'K':>4}  {'best_val_acc':>12}  {'best_epoch':>10}")
    print("-" * 60)
    for r in summary:
        print(
            f"  {r['condition']:<23s}  "
            f"K={r['k']:<3d}  "
            f"best_val_acc={r['best_val_acc']:.4f}  "
            f"best_epoch={r['best_val_epoch']}"
        )


if __name__ == "__main__":
    main()
