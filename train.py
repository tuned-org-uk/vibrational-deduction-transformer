"""
Wiring Autoencoder — training entry point.
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

from wae import WiringAutoencoder, get_device
from wae.dataset import load_dataset, make_loaders
from wae.spectral import _safe_eigh, lambda_fingerprint


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
        print(f"[WAE] MPS device detected — using mps_batch_size={bs} (override batch_size={tc['batch_size']})")
        return bs
    return tc["batch_size"]


def train_one_epoch(
    model: WiringAutoencoder,
    loader,
    optimizer,
    E: torch.Tensor,
    base_L,
    device: torch.device,
    grad_clip: float = 1.0,
    lambda_fp: torch.Tensor | None = None,
    spectral_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    freq_eigvals: torch.Tensor | None = None,
) -> dict:
    model.train()
    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "freq_loss": 0.0}
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        node_idx = batch["node_idx"].to(device)
        optimizer.zero_grad()
        batch_lambda_fp = None if lambda_fp is None else lambda_fp.expand(x.shape[0], -1)
        out = model(
            x,
            E,
            node_idx=node_idx,
            base_L=base_L,
            lambda_fp=batch_lambda_fp,
            spectral_cache=spectral_cache,
            freq_eigvals=freq_eigvals,
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
    E: torch.Tensor,
    base_L,
    device: torch.device,
    lambda_fp: torch.Tensor | None = None,
    spectral_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    freq_eigvals: torch.Tensor | None = None,
) -> dict:
    model.eval()
    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "freq_loss": 0.0}
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        node_idx = batch["node_idx"].to(device)
        batch_lambda_fp = None if lambda_fp is None else lambda_fp.expand(x.shape[0], -1)
        out = model(
            x,
            E,
            node_idx=node_idx,
            base_L=base_L,
            lambda_fp=batch_lambda_fp,
            spectral_cache=spectral_cache,
            freq_eigvals=freq_eigvals,
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
    print(f"[WAE] device={device}, dataset={cfg['dataset']['name']}, batch_size={batch_size}")

    tc = cfg["training"]
    dc = cfg["dataset"]
    gc = cfg["graph"]

    data = load_dataset(dc["name"], root=dc["root"], device=device)
    loaders = make_loaders(data, batch_size=batch_size)
    E = data["E"]
    meta = data["meta"]
    print(f"[WAE] nodes={meta['n_nodes']}, feat_dim={meta['feat_dim']}, classes={meta['n_classes']}")

    model = WiringAutoencoder.from_config(cfg, E).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[WAE] trainable params: {n_params:,}")

    from wae.laplacian import DifferentiableLaplacian
    use_sparse = gc.get("sparse", device.type == "mps")
    base_lap = DifferentiableLaplacian.from_embeddings(
        E,
        knn_k=gc["knn_k"],
        sigma=gc["sigma"],
        normalised=gc["normalised"],
        sparse=use_sparse,
    ).to(device)
    with torch.no_grad():
        base_L = base_lap(base_lap.base_weights.unsqueeze(0)).squeeze(0)

    # One-time spectral precomputation for the fixed base graph.
    # This avoids repeated O(N^3) CPU eigensolver calls during each forward pass.
    t_spec = time.time()
    with torch.no_grad():
        base_eigvals, base_eigvecs = _safe_eigh(base_L)
        cached_lambda_fp = lambda_fingerprint(
            base_L,
            tau_modes=cfg["model"]["tau_modes"],
            eigvals=base_eigvals,
        )
    spectral_cache = (base_eigvals, base_eigvecs)
    freq_eigvals = base_eigvals
    print(f"[WAE] cached base spectral quantities in {time.time() - t_spec:.1f}s")

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
    csv_fields = ["epoch", "train_loss", "train_recon", "train_kl", "train_freq",
                  "val_loss", "val_recon", "val_kl", "val_freq"]
    csv_f = open(log_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_writer.writeheader()

    best_val = float("inf")
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            E,
            base_L,
            device,
            tc.get("grad_clip", 1.0),
            lambda_fp=cached_lambda_fp,
            spectral_cache=spectral_cache,
            freq_eigvals=freq_eigvals,
        )
        val_metrics = eval_epoch(
            model,
            loaders["val"],
            E,
            base_L,
            device,
            lambda_fp=cached_lambda_fp,
            spectral_cache=spectral_cache,
            freq_eigvals=freq_eigvals,
        )
        if scheduler:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_recon": train_metrics["recon_loss"],
            "train_kl": train_metrics["kl_loss"],
            "train_freq": train_metrics["freq_loss"],
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon_loss"],
            "val_kl": val_metrics["kl_loss"],
            "val_freq": val_metrics["freq_loss"],
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
                f"kl={val_metrics['kl_loss']:.4f}  "
                f"freq={val_metrics['freq_loss']:.4f}  "
                f"({dt:.1f}s)"
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                       save_dir / "best.pt")

    csv_f.close()
    print(f"[WAE] Training complete. Best val loss: {best_val:.4f}")
    print(f"[WAE] Log saved to {log_path}")


if __name__ == "__main__":
    main()
