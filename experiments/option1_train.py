"""
experiments/option1_train.py  --  Option 1 deterministic AE trainer (#20, #30).

Trains DeterministicSpectralAE on a node-embedding dataset and saves:

    checkpoints/option1_best.pt          -- model state dict at best val loss
    results/option1_metrics.csv          -- per-epoch train/val loss breakdown
    results/option1_artefact.pt          -- spectral artefact for Option 6

The artefact file is produced by DeterministicSpectralAE.extract_spectral_artefact()
and contains W_hat, omega_hat, S_memory, eigvals_q.  Pass it to
option6_ablation.py via --artefact_path.

CLI
---
    python -m experiments.option1_train \\
        --data_path <pt file>  \\
        --out_dir results       \\
        --ckpt_dir checkpoints  \\
        --n_epochs 100          \\
        --batch_size 64         \\
        --lr 3e-4               \\
        --latent_dim 16         \\
        --spectral_penalty hard

The data file must be a .pt dict with keys:
    x    : (N_total, D)   float32 node embeddings
    E    : (N_nodes, D)   float32 full embedding table
    L_f  : (N_nodes, N_nodes) float32 frozen base Laplacian

Optionally include:
    eigvals : (N_nodes,) float32  pre-computed eigenvalues of L_f
    eigvecs : (N_nodes, N_nodes)  pre-computed eigenvectors of L_f

Ref: docs/v2/03-branching.md -- Option 1
Depends on: wae/vib_autoencoder.py (#20), wae/wiring_decoder.py
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wae.vib_autoencoder import DeterministicSpectralAE


# ---------------------------------------------------------------------------
# Eigenbasis helper
# ---------------------------------------------------------------------------

def compute_eigenbasis(
    L: torch.Tensor,
    q: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the leading q eigenvectors and eigenvalues of a symmetric
    Laplacian L (N, N).

    Uses torch.linalg.eigh which returns eigenvalues in ascending order.
    The first eigenvalue of a connected graph is ~0 (constant mode);
    modes 1..q are returned.

    Parameters
    ----------
    L : Tensor  (N, N)  symmetric Laplacian
    q : int     number of modes to extract (must be < N)

    Returns
    -------
    eigvals : Tensor  (q,)    in [0, lambda_max]
    eigvecs : Tensor  (N, q)  orthonormal columns
    """
    eigvals_full, eigvecs_full = torch.linalg.eigh(L)
    # Skip the trivial constant eigenvector at index 0
    eigvals = eigvals_full[1:q + 1]    # (q,)
    eigvecs = eigvecs_full[:, 1:q + 1] # (N, q)
    return eigvals, eigvecs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_option1(
    model: DeterministicSpectralAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    U_q: torch.Tensor,
    L_base: torch.Tensor,
    eigvals_q: Optional[torch.Tensor],
    n_epochs: int,
    lr: float,
    ckpt_path: Path,
    metrics_path: Path,
    device: torch.device,
    E: torch.Tensor,
    val_frac: float = 0.2,
) -> None:
    """
    Full Option 1 training loop.

    Saves the best checkpoint to ckpt_path and logs per-epoch metrics
    to metrics_path as a CSV with columns::

        epoch, train_loss, train_recon, train_spectral, val_loss, val_recon

    Parameters
    ----------
    model : DeterministicSpectralAE
    train_loader, val_loader : DataLoader
    U_q : Tensor  (D, q)  leading eigenvectors of L_base (frozen)
    L_base : Tensor  (N, N)  frozen base Laplacian
    eigvals_q : Tensor (q,) or None  -- required for spectral_penalty='soft'
    n_epochs : int
    lr : float
    ckpt_path : Path  checkpoint output path
    metrics_path : Path  CSV output path
    device : torch.device
    E : Tensor  (N, D)  full embedding table
    val_frac : float  unused here (split already done by caller)
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    L_base_dev  = L_base.to(device)
    U_q_dev     = U_q.to(device)
    E_dev       = E.to(device)
    eig_q_dev   = eigvals_q.to(device) if eigvals_q is not None else None

    best_val    = math.inf
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "epoch", "train_loss", "train_recon", "train_spectral",
            "val_loss", "val_recon", "H_lambda", "active_modes"
        ])

        for epoch in range(1, n_epochs + 1):
            # --- Train ---
            model.train()
            t_loss = t_recon = t_spectral = 0.0
            n_train = 0

            for x_batch, in train_loader:
                x_batch = x_batch.to(device)
                B       = x_batch.shape[0]
                optimiser.zero_grad(set_to_none=True)

                out  = model(
                    x_batch, U_q_dev, L_base_dev, E_dev,
                    eigvals_q=eig_q_dev,
                )
                out["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

                t_loss     += out["loss"].item()     * B
                t_recon    += out["recon"].item()    * B
                t_spectral += out["spectral_loss"].item() * B
                n_train    += B

            scheduler.step()

            # --- Validate ---
            model.eval()
            v_loss = v_recon = 0.0
            n_val  = 0
            h_lambda_mean = 0.0
            active_mean   = 0.0

            with torch.no_grad():
                for x_batch, in val_loader:
                    x_batch = x_batch.to(device)
                    B       = x_batch.shape[0]
                    out     = model(
                        x_batch, U_q_dev, L_base_dev, E_dev,
                        eigvals_q=eig_q_dev,
                    )
                    v_loss     += out["loss"].item()  * B
                    v_recon    += out["recon"].item() * B
                    h_lambda_mean += out["H_lambda"].item()     * B
                    active_mean   += out["active_mode_count"]   * B
                    n_val      += B

            tr_loss = t_loss / n_train
            vl_loss = v_loss / n_val

            writer.writerow([
                epoch,
                tr_loss,
                t_recon / n_train,
                t_spectral / n_train,
                vl_loss,
                v_recon / n_val,
                h_lambda_mean / n_val,
                active_mean / n_val,
            ])
            fh.flush()

            if epoch % max(1, n_epochs // 10) == 0:
                print(
                    f"epoch {epoch:>4d}/{n_epochs}  "
                    f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
                    f"H(lambda)={h_lambda_mean/n_val:.3f}"
                )

            # Checkpoint on improvement
            if vl_loss < best_val:
                best_val = vl_loss
                torch.save(model.state_dict(), ckpt_path)

    print(f"Training complete. Best val_loss={best_val:.4f}")
    print(f"Checkpoint -> {ckpt_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse CLI args, build model, train, extract artefact."""
    parser = argparse.ArgumentParser(
        description="Option 1: train DeterministicSpectralAE (#20/#30)"
    )
    parser.add_argument(
        "--data_path", required=True,
        help=".pt file with keys: x (N_total, D), E (N_nodes, D), "
             "L_f (N_nodes, N_nodes) [+optional eigvals, eigvecs]"
    )
    parser.add_argument("--out_dir",    default="results")
    parser.add_argument("--ckpt_dir",   default="checkpoints")
    parser.add_argument("--n_epochs",   type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--latent_dim", type=int,   default=16,
                        help="q -- spectral mode count (= latent_dim)")
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--tau_modes",  type=int,   default=8)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--beta",       type=float, default=0.01)
    parser.add_argument("--tau",        type=float, default=0.5)
    parser.add_argument("--spectral_penalty", choices=["hard", "soft"],
                        default="hard")
    parser.add_argument("--val_frac",   type=float, default=0.2)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir  = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data -------------------------------------------------------
    data  = torch.load(args.data_path, map_location="cpu")
    x_all = data["x"].float()       # (N_total, D)
    E     = data["E"].float()        # (N_nodes, D)
    L_f   = data["L_f"].float()     # (N_nodes, N_nodes)

    D      = x_all.shape[1]
    N      = L_f.shape[0]
    q      = args.latent_dim

    # --- Eigenbasis (recompute if not cached) ----------------------------
    if "eigvals" in data and "eigvecs" in data:
        eigvals_q = data["eigvals"][:q].float()       # (q,)
        U_q       = data["eigvecs"][:, :q].float()    # (N, q) or (D, q)
        print("Using cached eigenbasis from data file.")
    else:
        print(f"Computing eigenbasis (q={q}) from L_f...")
        eigvals_q, U_q = compute_eigenbasis(L_f, q)
        print(f"  eigvals range [{eigvals_q.min():.4f}, {eigvals_q.max():.4f}]")

    # --- Train/val split ------------------------------------------------
    N_total = x_all.shape[0]
    n_val   = max(1, int(N_total * args.val_frac))
    n_train = N_total - n_val
    perm    = torch.randperm(N_total)
    tr_idx  = perm[:n_train]
    va_idx  = perm[n_train:]

    train_loader = DataLoader(
        TensorDataset(x_all[tr_idx]),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_all[va_idx]),
        batch_size=args.batch_size, shuffle=False
    )

    # --- Build model -----------------------------------------------------
    model = DeterministicSpectralAE(
        input_dim=D,
        latent_dim=q,
        hidden_dim=args.hidden_dim,
        tau_modes=args.tau_modes,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        spectral_penalty=args.spectral_penalty,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeterministicSpectralAE  params={n_params:,}")

    # --- Train -----------------------------------------------------------
    train_option1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        U_q=U_q,
        L_base=L_f,
        eigvals_q=eigvals_q,
        n_epochs=args.n_epochs,
        lr=args.lr,
        ckpt_path=ckpt_dir / "option1_best.pt",
        metrics_path=out_dir / "option1_metrics.csv",
        device=device,
        E=E,
    )

    # --- Extract spectral artefact for Option 6 --------------------------
    print("\nExtracting spectral artefact...")
    model.load_state_dict(
        torch.load(ckpt_dir / "option1_best.pt", map_location=device)
    )
    artefact = model.extract_spectral_artefact(
        U_q=U_q.to(device),
        L_base=L_f.to(device),
        eigvals_q=eigvals_q.to(device),
    )
    artefact_path = out_dir / "option1_artefact.pt"
    torch.save(artefact, artefact_path)
    print(f"Artefact -> {artefact_path}")
    print(
        f"  W_hat:     {tuple(artefact['W_hat'].shape)}"
        f"  omega_hat: {tuple(artefact['omega_hat'].shape)}"
        f"  S_memory:  {tuple(artefact['S_memory'].shape)}"
    )


if __name__ == "__main__":
    main()
