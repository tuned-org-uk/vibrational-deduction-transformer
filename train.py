"""
Wiring Autoencoder -- training entry point.

Aligned with WiringAutoencoder v2 API (vdt/model.py).  The forward()
signature changed substantially between the initial train.py and the
current package:

  Old (stale):
    model(x, E, node_idx, base_L, lambda_fp, spectral_cache, freq_eigvals)
    returns: loss, recon_loss, kl_loss, freq_loss, ...

  Current (v2):
    model(x, U_q, eigvals_q, node_idx, spectral_cache, L_f)
    returns: loss, recon, kl_z, kl_S, kl_tau, x_hat, z, mu, log_var

Spectral pre-computation builds (eigvals, eigvecs) from the base Laplacian
once at startup and passes them as spectral_cache=(eigvals, eigvecs) into
every forward call -- avoiding repeated O(N^3) eigh inside the model.
"""
from __future__ import annotations
import argparse
import time
import yaml
import csv
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from vdt import WiringAutoencoder, get_device
from vdt.dataset import load_dataset, make_loaders
from vdt.laplacian import DifferentiableLaplacian


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Wiring Autoencoder")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", default=None, help="Override dataset.name in config")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None,
                   help="Force device: 'mps', 'cuda', 'cpu'. Default: auto-detect.")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_batch_size(cfg: dict, device: torch.device) -> int:
    tc = cfg["training"]
    if device.type == "mps" and "mps_batch_size" in tc:
        bs = tc["mps_batch_size"]
        print(f"[VDT] MPS device detected -- using mps_batch_size={bs} (override batch_size={tc['batch_size']})")
        return bs
    return tc["batch_size"]


def train_one_epoch(
    model: WiringAutoencoder,
    loader,
    optimizer,
    U_q: torch.Tensor,
    eigvals_q: torch.Tensor,
    device: torch.device,
    grad_clip: float = 1.0,
    spectral_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    L_f: torch.Tensor | None = None,
) -> dict:
    """
    Run one training epoch.

    Parameters
    ----------
    model        : WiringAutoencoder
    loader       : DataLoader yielding batches with keys 'x' and 'node_idx'
    optimizer    : torch optimizer
    U_q          : (N, q) leading eigenvectors of the base Laplacian
    eigvals_q    : (q,) corresponding eigenvalues
    device       : target device
    grad_clip    : gradient clipping norm (default 1.0)
    spectral_cache : pre-computed (eigvals, eigvecs) tuple for DiffusionDecoder
    L_f          : optional (N, N) feature-space Laplacian; None lets forward()
                   reconstruct it from U_q and eigvals_q on the fly

    Returns
    -------
    dict with mean loss, recon, kl_z, kl_S, kl_tau over the epoch.
    """
    model.train()
    # Keys match WiringAutoencoder.forward() return dict exactly.
    totals = {"loss": 0.0, "recon": 0.0, "kl_z": 0.0, "kl_S": 0.0, "kl_tau": 0.0}
    n = 0
    for batch in loader:
        x        = batch["x"].to(device)
        node_idx = batch["node_idx"].to(device)
        optimizer.zero_grad()
        out = model(
            x,
            U_q,
            eigvals_q,
            node_idx=node_idx,
            spectral_cache=spectral_cache,
            L_f=L_f,
        )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = x.shape[0]
        for k in totals:
            totals[k] += out[k].item() * bs
        n += bs
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model: WiringAutoencoder,
    loader,
    U_q: torch.Tensor,
    eigvals_q: torch.Tensor,
    device: torch.device,
    spectral_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    L_f: torch.Tensor | None = None,
) -> dict:
    """
    Evaluate the model on one epoch without gradient computation.

    Parameters
    ----------
    model        : WiringAutoencoder
    loader       : DataLoader yielding batches with keys 'x' and 'node_idx'
    U_q          : (N, q) leading eigenvectors of the base Laplacian
    eigvals_q    : (q,) corresponding eigenvalues
    device       : target device
    spectral_cache : pre-computed (eigvals, eigvecs) tuple for DiffusionDecoder
    L_f          : optional (N, N) feature-space Laplacian

    Returns
    -------
    dict with mean loss, recon, kl_z, kl_S, kl_tau over the epoch.
    """
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl_z": 0.0, "kl_S": 0.0, "kl_tau": 0.0}
    n = 0
    for batch in loader:
        x        = batch["x"].to(device)
        node_idx = batch["node_idx"].to(device)
        out = model(
            x,
            U_q,
            eigvals_q,
            node_idx=node_idx,
            spectral_cache=spectral_cache,
            L_f=L_f,
        )
        bs = x.shape[0]
        for k in totals:
            totals[k] += out[k].item() * bs
        n += bs
    return {k: v / n for k, v in totals.items()}


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.seed:
        cfg["training"]["seed"] = args.seed
    if args.no_wandb:
        cfg["logging"]["use_wandb"] = False

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    device = get_device(force=args.device, verbose=True)
    batch_size = resolve_batch_size(cfg, device)
    print(f"[VDT] device={device}, dataset={cfg['dataset']['name']}, batch_size={batch_size}")

    tc = cfg["training"]
    dc = cfg["dataset"]
    gc = cfg["graph"]

    data    = load_dataset(dc["name"], root=dc["root"], device=device)
    loaders = make_loaders(data, batch_size=batch_size)
    E       = data["E"]
    meta    = data["meta"]
    print(f"[VDT] nodes={meta['n_nodes']}, feat_dim={meta['feat_dim']}, classes={meta['n_classes']}")

    # Build model via factory -- also wires up the internal DifferentiableLaplacian.
    model   = WiringAutoencoder.from_config(cfg, E).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VDT] trainable params: {n_params:,}")

    # One-time spectral pre-computation for the fixed base graph.
    # Builds:
    #   U_q          (N, q) -- leading eigenvectors passed to every forward() call
    #   eigvals_q    (q,)   -- corresponding eigenvalues
    #   spectral_cache      -- (full_eigvals, full_eigvecs) for DiffusionDecoder
    #
    # This avoids repeated O(N^3) eigh inside the model at each training step.
    # The base Laplacian L_base is computed once from the frozen base weights;
    # it does not change during training.
    t_spec = time.time()
    with torch.no_grad():
        use_sparse = gc.get("sparse", device.type == "mps")
        base_lap = DifferentiableLaplacian.from_embeddings(
            E,
            knn_k=gc["knn_k"],
            sigma=gc["sigma"],
            normalised=gc["normalised"],
            sparse=use_sparse,
        ).to(device)
        base_L      = base_lap(base_lap.base_weights.unsqueeze(0)).squeeze(0)  # (N, N)
        full_eigvals, full_eigvecs = torch.linalg.eigh(base_L)                 # (N,), (N, N)
        q           = cfg["model"].get("q", cfg["model"].get("tau_modes", 8))
        U_q         = full_eigvecs[:, :q].to(device)    # (N, q)
        eigvals_q   = full_eigvals[:q].to(device)       # (q,)
        spectral_cache = (full_eigvals.to(device), full_eigvecs.to(device))
    print(f"[VDT] cached base spectral quantities in {time.time() - t_spec:.1f}s  (q={q})")

    optimizer = optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-5)
    )
    scheduler = None
    if tc.get("scheduler") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"])

    lc = cfg["logging"]
    if lc.get("use_wandb"):
        import wandb
        wandb.init(project=lc["project"], config=cfg)

    save_dir = Path(lc.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training_log.csv"
    # CSV fields aligned with forward() return dict keys.
    csv_fields = [
        "epoch",
        "train_loss", "train_recon", "train_kl_z", "train_kl_S", "train_kl_tau",
        "val_loss",   "val_recon",   "val_kl_z",   "val_kl_S",   "val_kl_tau",
    ]
    csv_f = open(log_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_writer.writeheader()

    best_val = float("inf")
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer,
            U_q, eigvals_q, device,
            grad_clip=tc.get("grad_clip", 1.0),
            spectral_cache=spectral_cache,
        )
        val_metrics = eval_epoch(
            model, loaders["val"],
            U_q, eigvals_q, device,
            spectral_cache=spectral_cache,
        )
        if scheduler:
            scheduler.step()

        row = {
            "epoch":       epoch,
            "train_loss":  train_metrics["loss"],
            "train_recon": train_metrics["recon"],
            "train_kl_z":  train_metrics["kl_z"],
            "train_kl_S":  train_metrics["kl_S"],
            "train_kl_tau":train_metrics["kl_tau"],
            "val_loss":    val_metrics["loss"],
            "val_recon":   val_metrics["recon"],
            "val_kl_z":    val_metrics["kl_z"],
            "val_kl_S":    val_metrics["kl_S"],
            "val_kl_tau":  val_metrics["kl_tau"],
        }
        csv_writer.writerow(row)
        csv_f.flush()

        if lc.get("use_wandb"):
            import wandb
            wandb.log(row)

        if epoch % lc.get("log_every", 10) == 0 or epoch == 1:
            dt = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{tc['epochs']}  "
                f"train={train_metrics['loss']:.4f}  "
                f"val={val_metrics['loss']:.4f}  "
                f"recon={val_metrics['recon']:.4f}  "
                f"kl_z={val_metrics['kl_z']:.4f}  "
                f"kl_S={val_metrics['kl_S']:.4f}  "
                f"kl_tau={val_metrics['kl_tau']:.4f}  "
                f"({dt:.1f}s)"
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                save_dir / "best.pt",
            )

    csv_f.close()
    print(f"[VDT] Training complete. Best val loss: {best_val:.4f}")
    print(f"[VDT] Log saved to {log_path}")


if __name__ == "__main__":
    main()
