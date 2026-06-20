"""
Benchmark script -- compare Wiring Autoencoder (VDT) against:
    1. Plain VAE (same architecture, no wiring / graph path)
    2. Linear Autoencoder (AE) -- PCA-equivalent reconstruction baseline

Metrics reported (saved to CSV + printed as table):
    - Reconstruction MSE (lower is better)
    - Latent KL divergence (lower ~ better posterior collapse diagnostic)
    - Downstream classification accuracy on frozen latent z (linear probe)
    - Spectral entropy H(Lambda) of the generated Laplacians (VDT only)

Usage
-----
    python benchmark.py --dataset cora --output results/
    python benchmark.py --dataset cora --device mps     # force MPS
    python benchmark.py --dataset cora --device cpu     # force CPU
    python benchmark.py --dataset cora --batch-size 8   # override batch

Device / config defaults
------------------------
    When device resolves to 'mps' and no --config flag is passed, the
    script automatically uses configs/mps.yaml instead of
    configs/default.yaml.  That profile sets hidden_dim=32, n_layers=2,
    and batch_size=4 which keeps MHA activations under 1 GB on Cora.

    To run the full default profile on MPS (requires >=32 GB RAM)::

        python benchmark.py --config configs/default.yaml
"""
from __future__ import annotations
import argparse
import csv
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

from vdt import WiringAutoencoder, get_device
from vdt.encoder import WiringEncoder
from vdt.dataset import load_dataset, make_loaders
from vdt.laplacian import DifferentiableLaplacian
from vdt.spectral import spectral_freq_cost


# ---------------------------------------------------------------------------
# Baseline VAE (no wiring decoder, direct Gaussian decoder)
# ---------------------------------------------------------------------------
class BaselineVAE(nn.Module):
    """Standard VAE with MLP encoder and MLP decoder for comparison."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> dict:
        h = self.encoder_net(x)
        mu, log_var = self.mu_head(h), self.log_var_head(h).clamp(-10, 4)
        std = (0.5 * log_var).exp()
        z   = mu + std * torch.randn_like(std)
        x_hat = self.decoder_net(z)
        recon = ((x - x_hat) ** 2).sum(dim=-1).mean()
        kl    = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=-1).mean()
        return {"loss": recon + kl, "recon_loss": recon, "kl_loss": kl,
                "freq_loss": torch.tensor(0.0), "mu": mu, "z": z}


# ---------------------------------------------------------------------------
# Baseline Linear AE (PCA-equivalent)
# ---------------------------------------------------------------------------
class LinearAE(nn.Module):
    """Linear autoencoder: x -> z = Wx, x_hat = W^T z (PCA-equivalent)."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.W   = nn.Linear(input_dim, latent_dim, bias=False)
        self.dec = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.W(x)
        x_hat = self.dec(z)
        recon = ((x - x_hat) ** 2).sum(dim=-1).mean()
        return {"loss": recon, "recon_loss": recon,
                "kl_loss": torch.tensor(0.0),
                "freq_loss": torch.tensor(0.0), "mu": z, "z": z}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_model(
    model, loaders, device, epochs, lr,
    # VDT-only spectral arguments (None for baselines)
    U_q=None, eigvals_q=None, L_f=None, eig_cache=None,
    is_vdt=False,
):
    """Train model for the given number of epochs.

    For VDT (is_vdt=True) the current WiringAutoencoder.forward() signature
    is used::

        model(x, U_q, eigvals_q, node_idx=node_idx,
              L_f=L_f, spectral_cache=eig_cache)

    U_q, eigvals_q, L_f, and eig_cache must be pre-computed by the caller
    (see spectral pre-computation block in main()) and passed here.  They
    are ignored for baseline models.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch in loaders["train"]:
            x        = batch["x"].to(device)
            node_idx = batch["node_idx"].to(device)
            optimizer.zero_grad()
            if is_vdt:
                out = model(
                    x, U_q, eigvals_q,
                    node_idx=node_idx,
                    L_f=L_f,
                    spectral_cache=eig_cache,
                )
            else:
                out = model(x)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


@torch.no_grad()
def extract_latents(
    model, loader, device,
    U_q=None, eigvals_q=None, L_f=None, eig_cache=None,
    is_vdt=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract latent means (mu) and labels for a linear probe.

    VDT output uses the 'mu' key returned by WiringAutoencoder.forward().
    Baseline outputs also return 'mu'; no key remapping is needed.
    """
    model.eval()
    zs, ys = [], []
    for batch in loader:
        x        = batch["x"].to(device)
        node_idx = batch["node_idx"].to(device)
        if is_vdt:
            out = model(
                x, U_q, eigvals_q,
                node_idx=node_idx,
                L_f=L_f,
                spectral_cache=eig_cache,
            )
        else:
            out = model(x)
        zs.append(out["mu"].cpu().numpy())
        ys.append(batch["label"].cpu().numpy())
    return np.concatenate(zs), np.concatenate(ys)


@torch.no_grad()
def compute_test_metrics(
    model, loader, device,
    U_q=None, eigvals_q=None, L_f=None, eig_cache=None,
    is_vdt=False,
) -> dict[str, float]:
    """Accumulate per-batch metrics over a loader and return means.

    Key mapping for VDT (WiringAutoencoder output keys -> benchmark keys)::

        'recon'  -> recon_loss
        'kl_z'   -> kl_loss    (isotropic VAE KL)
        'kl_S'   -> freq_loss  (spectral basis KL, analogous to old freq cost)

    Baseline models still return 'recon_loss', 'kl_loss', 'freq_loss'
    directly; their branch is unchanged.
    """
    model.eval()
    totals = {"recon_loss": 0.0, "kl_loss": 0.0, "freq_loss": 0.0}
    n = 0
    for batch in loader:
        x        = batch["x"].to(device)
        node_idx = batch["node_idx"].to(device)
        if is_vdt:
            out = model(
                x, U_q, eigvals_q,
                node_idx=node_idx,
                L_f=L_f,
                spectral_cache=eig_cache,
            )
            # Remap current output keys to benchmark accumulator keys.
            totals["recon_loss"] += out["recon"].item() * x.shape[0]
            totals["kl_loss"]    += out["kl_z"].item()  * x.shape[0]
            totals["freq_loss"]  += out["kl_S"].item()  * x.shape[0]
        else:
            out = model(x)
            bs = x.shape[0]
            for k in totals:
                totals[k] += out[k].item() * bs
        n += x.shape[0]
    return {k: v / n for k, v in totals.items()}


def linear_probe_accuracy(
    z_train: np.ndarray, y_train: np.ndarray,
    z_test:  np.ndarray, y_test:  np.ndarray,
) -> float:
    scaler = StandardScaler()
    z_tr = scaler.fit_transform(z_train)
    z_te = scaler.transform(z_test)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(z_tr, y_train)
    return accuracy_score(y_test, clf.predict(z_te))


# ---------------------------------------------------------------------------
# Batch-size resolution helper
# ---------------------------------------------------------------------------

def _resolve_batch_size(cfg: dict, device: str, cli_override: int | None) -> int:
    """Return the batch size to use for all loaders.

    Priority order (highest first):

    1. --batch-size CLI flag (explicit override)
    2. cfg['training']['mps_batch_size']  when device == 'mps'
    3. cfg['training']['batch_size']
    4. Hard fallback: 4 (safe on any device)

    The MPS path reads 'mps_batch_size' first so that mps.yaml can keep
    a small safe default without touching the generic 'batch_size' key
    (which train.py may read independently).
    """
    if cli_override is not None:
        return cli_override
    tc = cfg.get("training", {})
    if str(device) == "mps" and "mps_batch_size" in tc:
        return int(tc["mps_batch_size"])
    return int(tc.get("batch_size", 4))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark WiringAutoencoder vs. VAE and LinearAE baselines."
    )
    p.add_argument("--dataset",    default="cora")
    p.add_argument("--output",     default="results/")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--latent",     type=int,   default=32)
    p.add_argument("--hidden",     type=int,   default=256)
    p.add_argument("--config",     default=None,
                   help="Path to YAML config.  Defaults to configs/mps.yaml "
                        "when device is MPS, configs/default.yaml otherwise.")
    p.add_argument("--device",     default=None,
                   help="Force device: 'mps', 'cuda', 'cpu'. Default: auto-detect.")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="Override batch size for all loaders.  When omitted, "
                        "the value is read from the config file "
                        "(mps_batch_size for MPS, batch_size otherwise).")
    args = p.parse_args()

    # ------------------------------------------------------------------ #
    # Device selection -- MPS -> CUDA -> CPU                              #
    # ------------------------------------------------------------------ #
    device = get_device(force=args.device, verbose=True)
    print(f"[Benchmark] device={device}, dataset={args.dataset}")

    # ------------------------------------------------------------------ #
    # Config selection -- auto-pick mps.yaml on MPS when not overridden   #
    # ------------------------------------------------------------------ #
    if args.config is not None:
        config_path = args.config
    elif str(device) == "mps" and Path("configs/mps.yaml").exists():
        config_path = "configs/mps.yaml"
        print("[Benchmark] MPS device detected -- using configs/mps.yaml "
              "(pass --config configs/default.yaml to override).")
    else:
        config_path = "configs/default.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["name"] = args.dataset

    # ------------------------------------------------------------------ #
    # Batch size -- read from config, override via CLI                    #
    # ------------------------------------------------------------------ #
    batch_size = _resolve_batch_size(cfg, device, args.batch_size)
    print(f"[Benchmark] batch_size={batch_size}  (config: {config_path})")

    data    = load_dataset(args.dataset, root=cfg["dataset"]["root"], device=device)
    loaders = make_loaders(data, batch_size=batch_size)
    E       = data["E"]
    meta    = data["meta"]
    D       = meta["feat_dim"]
    print(f"[Benchmark] N={meta['n_nodes']}, D={D}, classes={meta['n_classes']}")

    # ------------------------------------------------------------------ #
    # Build WiringAutoencoder and run spectral pre-computation once.      #
    #                                                                      #
    # WiringAutoencoder.forward() requires:                               #
    #   U_q       -- top-q eigenvectors of the base Laplacian  (N, q)    #
    #   eigvals_q -- corresponding eigenvalues                  (q,)      #
    #   L_f       -- mass-weighted feature-space Laplacian      (N, N)   #
    #   eig_cache -- full (eigvals, eigvecs) for DiffusionDecoder        #
    #                                                                      #
    # linalg.eigh runs on CPU to avoid the MPS linalg restriction; the   #
    # results are moved to device afterwards.                             #
    # ------------------------------------------------------------------ #
    vdt = WiringAutoencoder.from_config(cfg, E).to(device)

    base_lap = vdt._laplacian
    with torch.no_grad():
        base_L_dense = base_lap(base_lap.base_weights.unsqueeze(0)).squeeze(0)
        base_L_cpu   = base_L_dense.cpu()
        full_eigvals_cpu, full_eigvecs_cpu = torch.linalg.eigh(base_L_cpu)
        full_eigvals = full_eigvals_cpu.to(device)
        full_eigvecs = full_eigvecs_cpu.to(device)

    q = vdt.q
    U_q       = full_eigvecs[:, :q]   # (N, q)
    eigvals_q = full_eigvals[:q]      # (q,)

    # Mass-weighted feature-space Laplacian, built once (issue #74).
    L_f = vdt.build_L_f(full_eigvals_cpu, full_eigvecs_cpu).to(device)  # (N, N)

    # Full eigdecomposition cache for DiffusionDecoder.
    eig_cache = (full_eigvals, full_eigvecs)

    results = []

    # -------------------------------------------------------------------
    # 1. Wiring Autoencoder
    # -------------------------------------------------------------------
    print("\n[1/3] Training Wiring Autoencoder...")
    vdt = train_model(
        vdt, loaders, device, args.epochs, args.lr,
        U_q=U_q, eigvals_q=eigvals_q, L_f=L_f, eig_cache=eig_cache,
        is_vdt=True,
    )
    vdt_metrics = compute_test_metrics(
        vdt, loaders["test"], device,
        U_q=U_q, eigvals_q=eigvals_q, L_f=L_f, eig_cache=eig_cache,
        is_vdt=True,
    )
    z_tr, y_tr = extract_latents(
        vdt, loaders["train"], device,
        U_q=U_q, eigvals_q=eigvals_q, L_f=L_f, eig_cache=eig_cache,
        is_vdt=True,
    )
    z_te, y_te = extract_latents(
        vdt, loaders["test"], device,
        U_q=U_q, eigvals_q=eigvals_q, L_f=L_f, eig_cache=eig_cache,
        is_vdt=True,
    )
    vdt_acc = linear_probe_accuracy(z_tr, y_tr, z_te, y_te)
    results.append({"model": "WiringAE",
                    "recon_mse":    vdt_metrics["recon_loss"],
                    "kl":           vdt_metrics["kl_loss"],
                    "freq_cost":    vdt_metrics["freq_loss"],
                    "linear_probe": vdt_acc})
    print(f"  VDT  -> recon={vdt_metrics['recon_loss']:.4f}  "
          f"kl={vdt_metrics['kl_loss']:.4f}  probe={vdt_acc:.4f}")

    # -------------------------------------------------------------------
    # 2. Baseline VAE  -- uses same hidden_dim as VDT config for fairness
    # -------------------------------------------------------------------
    vdt_hidden = cfg["model"].get("hidden_dim", args.hidden)
    print("[2/3] Training Baseline VAE...")
    vae = BaselineVAE(D, args.latent, vdt_hidden)
    vae = train_model(vae, loaders, device, args.epochs, args.lr, is_vdt=False)
    vae_metrics = compute_test_metrics(vae, loaders["test"], device, is_vdt=False)
    z_tr, y_tr = extract_latents(vae, loaders["train"], device)
    z_te, y_te = extract_latents(vae, loaders["test"],  device)
    vae_acc    = linear_probe_accuracy(z_tr, y_tr, z_te, y_te)
    results.append({"model": "BaselineVAE",
                    "recon_mse":    vae_metrics["recon_loss"],
                    "kl":           vae_metrics["kl_loss"],
                    "freq_cost":    0.0,
                    "linear_probe": vae_acc})
    print(f"  VAE  -> recon={vae_metrics['recon_loss']:.4f}  "
          f"kl={vae_metrics['kl_loss']:.4f}  probe={vae_acc:.4f}")

    # -------------------------------------------------------------------
    # 3. Linear AE
    # -------------------------------------------------------------------
    print("[3/3] Training Linear AE (PCA baseline)...")
    lin_ae = LinearAE(D, args.latent)
    lin_ae = train_model(lin_ae, loaders, device, args.epochs, args.lr, is_vdt=False)
    lin_metrics = compute_test_metrics(lin_ae, loaders["test"], device)
    z_tr, y_tr  = extract_latents(lin_ae, loaders["train"], device)
    z_te, y_te  = extract_latents(lin_ae, loaders["test"],  device)
    lin_acc     = linear_probe_accuracy(z_tr, y_tr, z_te, y_te)
    results.append({"model": "LinearAE",
                    "recon_mse":    lin_metrics["recon_loss"],
                    "kl":           0.0,
                    "freq_cost":    0.0,
                    "linear_probe": lin_acc})
    print(f"  LinAE-> recon={lin_metrics['recon_loss']:.4f}  "
          f"                    probe={lin_acc:.4f}")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"benchmark_{args.dataset}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'Model':<15} {'Recon MSE':>12} {'KL':>10} {'Freq Cost':>12} {'LinProbe Acc':>14}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<15} {r['recon_mse']:>12.4f} {r['kl']:>10.4f} "
              f"{r['freq_cost']:>12.4f} {r['linear_probe']:>14.4f}")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
