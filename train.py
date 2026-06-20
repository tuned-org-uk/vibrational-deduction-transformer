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

Stability safeguards (issue #75)
----------------------------------
Three safeguards added that were previously dead code:

1.  pre_training_checks -- called once before epoch 1 from main(), after
    the spectral pre-computation block.  A CFL failure raises RuntimeError
    so training is aborted before any weights are dirtied.  MassMatrix is
    now wired (issue #74 merged), so mass_diag is supplied.

2.  log_var saturation counter -- train_one_epoch wraps every forward call
    in warnings.catch_warnings and counts batches that emit the
    'log_var clamp active' RuntimeWarning from WiringEncoder.  The count
    is returned in the metrics dict as 'log_var_saturations' and written
    to the CSV and W&B log.

3.  spectral_kl_health_check -- called at the end of each epoch after
    val_metrics is computed.  Failures are printed as [WARN] lines; they
    do not abort training.

active_modes wiring (issue #77)
--------------------------------
spectral_kl_health_check now receives the real N_active from
val_metrics['N_active'] (returned by WiringAutoencoder.forward() since
issue #77 was closed).  The fallback val_metrics.get('N_active', _q) is
still present for backward compatibility with older checkpoints.

mode_explosion warmup (stability.py fix)
-----------------------------------------
spectral_kl_health_check accepts epoch and warmup_epochs so the
mode_explosion warning is suppressed during early training when all modes
are still active by initialisation.  warmup_epochs is read from
cfg['training']['kl_warmup_epochs'] (default 5).

Checkpoint strategy
-------------------
Two files are written to save_dir on every epoch:

  last.pt   -- unconditional snapshot after every epoch.  Contains
               model.state_dict(), optimizer.state_dict(),
               scheduler.state_dict() (when a scheduler is active),
               cfg, epoch, and best_val.  Use --resume to restart from
               this file after a crash or pre-emption.

  best.pt   -- written only when val_loss strictly improves.  Same
               payload as last.pt so it can also serve as a resume point.

Resuming training::

    python train.py --config configs/mps.yaml --resume checkpoints/last.pt

The resume path restores model weights, optimizer and scheduler state,
the epoch counter (training continues from epoch+1), and best_val so
the best.pt gate is not reset to inf.

Periodic checkpoints (save_every_n_epochs) are also supported::

    # in your config under logging:
    save_every_n_epochs: 10

This writes checkpoints/epoch_{N:04d}.pt every N epochs in addition to
last.pt and best.pt.
"""
from __future__ import annotations
import argparse
import time
import warnings
import yaml
import csv
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from vdt import WiringAutoencoder, get_device
from vdt.dataset import load_dataset, make_loaders
from vdt.laplacian import DifferentiableLaplacian, MassMatrix
from vdt.stability import pre_training_checks, spectral_kl_health_check


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
    p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to a checkpoint file (last.pt or best.pt) to resume training from. "
            "Restores model weights, optimizer state, scheduler state, epoch counter, "
            "and best_val so training continues seamlessly."
        ),
    )
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


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
    best_val: float,
) -> None:
    """
    Save a full training checkpoint to path.

    The checkpoint dict contains:

    - epoch        : int   -- the epoch just completed
    - model        : dict  -- model.state_dict()
    - optimizer    : dict  -- optimizer.state_dict()
    - scheduler    : dict or None  -- scheduler.state_dict() when active
    - cfg          : dict  -- full config used for this run
    - best_val     : float -- best validation loss seen so far

    Saving is atomic: the payload is first written to a sibling .tmp file
    and then renamed to path so a crash mid-write never leaves a corrupt
    checkpoint.

    Parameters
    ----------
    path      : destination file (e.g. save_dir / 'last.pt')
    epoch     : epoch number just completed
    model     : the model being trained
    optimizer : the optimizer
    scheduler : LR scheduler or None
    cfg       : config dict
    best_val  : best validation loss seen so far in this run
    """
    payload = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "cfg":       cfg,
        "best_val":  best_val,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    """
    Load a checkpoint and restore model, optimizer, and scheduler state.

    Parameters
    ----------
    path      : checkpoint file produced by save_checkpoint()
    model     : model instance (must already be on device)
    optimizer : optimizer instance
    scheduler : LR scheduler or None
    device    : target device for map_location

    Returns
    -------
    (start_epoch, best_val)
        start_epoch -- epoch to resume FROM (checkpoint epoch + 1)
        best_val    -- best validation loss persisted in the checkpoint
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_val    = float(ckpt.get("best_val", float("inf")))
    print(
        f"[VDT] Resumed from {path}  "
        f"(epoch {ckpt['epoch']}, best_val={best_val:.4f})"
    )
    return start_epoch, best_val


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
    dict with mean loss, recon, kl_z, kl_S, kl_tau, log_var_saturations,
    and N_active over the epoch.  log_var_saturations counts the number of
    batches that emitted a 'log_var clamp active' RuntimeWarning from
    WiringEncoder (issue #75).  N_active is the mean number of spectrally
    active modes per batch as returned by the model forward pass; it is
    present in the dict only when the model exposes out['N_active']
    (issue #77).
    """
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl_z": 0.0, "kl_S": 0.0, "kl_tau": 0.0}
    n = 0
    log_var_saturations = 0
    n_active_sum = 0
    n_active_batches = 0
    for batch in loader:
        x        = batch["x"].to(device)
        node_idx = batch["node_idx"].to(device)
        optimizer.zero_grad()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = model(
                x,
                U_q,
                eigvals_q,
                node_idx=node_idx,
                spectral_cache=spectral_cache,
                L_f=L_f,
            )
        log_var_saturations += sum(
            1 for w in caught if "log_var clamp active" in str(w.message)
        )
        if "N_active" in out:
            n_active_sum += int(out["N_active"])
            n_active_batches += 1
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = x.shape[0]
        for k in totals:
            totals[k] += out[k].item() * bs
        n += bs
    metrics = {k: v / n for k, v in totals.items()}
    metrics["log_var_saturations"] = log_var_saturations
    if n_active_batches > 0:
        metrics["N_active"] = n_active_sum / n_active_batches
    return metrics


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
    N_active is included when the model forward pass exposes out['N_active']
    (issue #77); the value is the mean over batches in this split.
    """
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl_z": 0.0, "kl_S": 0.0, "kl_tau": 0.0}
    n = 0
    n_active_sum = 0
    n_active_batches = 0
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
        if "N_active" in out:
            n_active_sum += int(out["N_active"])
            n_active_batches += 1
        bs = x.shape[0]
        for k in totals:
            totals[k] += out[k].item() * bs
        n += bs
    metrics = {k: v / n for k, v in totals.items()}
    if n_active_batches > 0:
        metrics["N_active"] = n_active_sum / n_active_batches
    return metrics


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

    model   = WiringAutoencoder.from_config(cfg, E).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VDT] trainable params: {n_params:,}")

    # ------------------------------------------------------------------
    # Spectral pre-computation (one-time, frozen base graph)
    # ------------------------------------------------------------------
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
        base_L      = base_lap(base_lap.base_weights.unsqueeze(0)).squeeze(0)
        base_L_cpu  = base_L.detach().to("cpu")
        full_eigvals, full_eigvecs = torch.linalg.eigh(base_L_cpu)
        full_eigvals = full_eigvals.to(device)
        full_eigvecs = full_eigvecs.to(device)
        q           = cfg["model"].get("q", cfg["model"].get("tau_modes", 8))
        U_q         = full_eigvecs[:, :q].to(device)
        eigvals_q   = full_eigvals[:q].to(device)
        spectral_cache = (full_eigvals.to(device), full_eigvecs.to(device))
        L_f = model.build_L_f(full_eigvals, full_eigvecs).to(device)
    print(f"[VDT] cached base spectral quantities in {time.time() - t_spec:.1f}s  (q={q})")

    # ------------------------------------------------------------------
    # Optimiser and scheduler -- built before checkpoint restore so that
    # load_checkpoint() can populate their state dicts in-place.
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-5)
    )
    scheduler = None
    if tc.get("scheduler") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"])

    # ------------------------------------------------------------------
    # Resume from checkpoint (--resume flag)
    # ------------------------------------------------------------------
    # start_epoch is 1 for a fresh run; load_checkpoint() returns
    # ckpt['epoch'] + 1 so training continues from the next epoch.
    # best_val is also restored so the best.pt gate is not reset to inf.
    start_epoch = 1
    best_val    = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume path not found: {resume_path}")
        start_epoch, best_val = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )

    # ------------------------------------------------------------------
    # Pre-flight stability check (issue #75)
    # ------------------------------------------------------------------
    mass_clip = float(cfg["model"].get("mass_clip", 1e3))
    eps       = float(cfg["model"].get("eps"))
    tau       = float(cfg["model"].get("tau", 0.5))
    lam_s     = float(cfg["model"].get("lam_s", 0.1))
    print(f"[VDT] eps: {eps}, tau: {tau}, mass_clip: {mass_clip}, lam_s: {lam_s}")
    assert eps is not None, "eps should be set in config -> model"
    _mass_for_check = MassMatrix(
        full_eigvals.cpu(),
        tau=tau,
        eps=eps,
        mass_clip=mass_clip,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        stability_issues = pre_training_checks(
            L_f.cpu(),
            _mass_for_check.M_diag,
            tc.get("dt_init", 0.005),
        )
        print(f"[VDT] dt_init: {tc.get('dt_init', 0.005)}")
    if stability_issues:
        for msg in stability_issues:
            print(f"[VDT][STABILITY] {msg}")
        raise RuntimeError(
            "CFL pre-flight check failed. "
            "Reduce dt_init or grad_clip in the config before training."
        )
    print("[VDT] Pre-flight stability checks passed.")

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    lc = cfg["logging"]
    if lc.get("use_wandb"):
        import wandb
        wandb.init(project=lc["project"], config=cfg)

    save_dir = Path(lc.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training_log.csv"

    csv_fields = [
        "epoch",
        "train_loss", "train_recon", "train_kl_z", "train_kl_S", "train_kl_tau",
        "val_loss",   "val_recon",   "val_kl_z",   "val_kl_S",   "val_kl_tau",
        "log_var_saturations",
        "N_active",
    ]
    # When resuming, append to the existing CSV rather than overwriting it.
    csv_mode = "a" if (args.resume and log_path.exists()) else "w"
    csv_f = open(log_path, csv_mode, newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    if csv_mode == "w":
        csv_writer.writeheader()

    _q          = cfg["model"].get("q", 16)
    _kl_warmup  = int(tc.get("kl_warmup_epochs", 5))
    save_every  = int(lc.get("save_every_n_epochs", 0))  # 0 = disabled

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, tc["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer,
            U_q, eigvals_q, device,
            grad_clip=tc.get("grad_clip", 1.0),
            spectral_cache=spectral_cache,
            L_f=L_f,
        )
        val_metrics = eval_epoch(
            model, loaders["val"],
            U_q, eigvals_q, device,
            spectral_cache=spectral_cache,
            L_f=L_f,
        )
        if scheduler:
            scheduler.step()

        log_var_sat = train_metrics.get("log_var_saturations", 0)
        if log_var_sat > 0:
            print(
                f"  [WARN] log_var saturated in {log_var_sat} batch(es) this epoch -- "
                "consider reducing lr or increasing log_var_max in WiringEncoder."
            )

        kl_health = spectral_kl_health_check(
            val_metrics["kl_z"],
            val_metrics["kl_S"],
            val_metrics["kl_tau"],
            active_modes=val_metrics.get("N_active", _q),
            q=_q,
            epoch=epoch,
            warmup_epochs=_kl_warmup,
        )
        if not all(kl_health.values()):
            failed = [k for k, v in kl_health.items() if not v]
            print(f"  [WARN] KL health check failed: {failed}")
            if "mode_explosion" in failed:
                print(
                    f"  [WARN] mode_explosion detected -- current lam_s={lam_s:.4f}. "
                    "Consider increasing lam_s in the config."
                )

        row = {
            "epoch":                epoch,
            "train_loss":           train_metrics["loss"],
            "train_recon":          train_metrics["recon"],
            "train_kl_z":           train_metrics["kl_z"],
            "train_kl_S":           train_metrics["kl_S"],
            "train_kl_tau":         train_metrics["kl_tau"],
            "val_loss":             val_metrics["loss"],
            "val_recon":            val_metrics["recon"],
            "val_kl_z":             val_metrics["kl_z"],
            "val_kl_S":             val_metrics["kl_S"],
            "val_kl_tau":           val_metrics["kl_tau"],
            "log_var_saturations":  log_var_sat,
            "N_active":             val_metrics.get("N_active", ""),
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

        # --------------------------------------------------------------
        # Checkpoint saves
        # --------------------------------------------------------------
        # 1. Always write last.pt (atomic rename via .tmp).
        save_checkpoint(
            save_dir / "last.pt",
            epoch, model, optimizer, scheduler, cfg, best_val,
        )

        # 2. Write best.pt when val_loss strictly improves.
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                save_dir / "best.pt",
                epoch, model, optimizer, scheduler, cfg, best_val,
            )

        # 3. Periodic snapshot (optional; controlled by save_every_n_epochs).
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:04d}.pt",
                epoch, model, optimizer, scheduler, cfg, best_val,
            )

    csv_f.close()
    print(f"[VDT] Training complete. Best val loss: {best_val:.4f}")
    print(f"[VDT] Checkpoints saved to {save_dir}")
    print(f"[VDT] Log saved to {log_path}")


if __name__ == "__main__":
    main()
