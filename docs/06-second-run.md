This run is healthy in terms of ELBO convergence but is stuck in a persistent **mode explosion / mode collapse** pathology. Here is a full diagnostic and a set of concrete fixes to commit.

***

## Diagnostic Summary

### What is working

The ELBO converges cleanly and fast. By epoch ~17 the train/val curves are essentially flat at ≈ −2461 / −2451, with no overfitting gap (train and val track within ~10 nats throughout). Reconstruction loss dominates and stabilises at ≈ −2453 nats. There is no numerical instability.

### The core pathology: mode explosion that never resolves

Every single epoch from epoch 5 to 50 fires `spectral_kl_health_check: mode explosion -- 8.0/8 modes active (>90%)`. `N_active` is frozen at `8.0` throughout the entire run — the spectral selector **never prunes a single mode**. This means the model is treating all `q=8` vibrational modes as equally active, which defeats the purpose of the sparse spectral prior. The concurrent `mode_collapse` KL health warning refers to `kl_z` being below the expected threshold for a 16-dim Gaussian posterior — `kl_z` decays from 6.4 → 1.46, far below the ~8 nats you would expect for `latent_dim=16`.

### Root cause: under-penalised spectral selection

Looking at the config :

| Parameter | Current value | Problem |
|---|---|---|
| `lam_s` | 0.1 | Too weak — `kl_S` decays freely from ~20 → 0.5 with no resistance |
| `nu` | 1.0 | Active-mode penalty is ineffective at this scale |
| `a_min` | 0.1 | Floor permits all modes to stay near-uniform |
| `mass_clip` | 1000.0 | Conditioning ratio 999.2 ≈ clip limit — MassMatrix is saturating the clip, introducing poor eigenvalue scaling |

The `kl_S` (spectral basis KL) collapses from 19.9 at epoch 1 down to ~0.45 at epoch 50, indicating the posterior over spectral modes has collapsed toward a near-uniform, uninformative distribution. The `kl_tau` (mode frequency KL) decays to ~0.002 by epoch 30 — essentially zero — meaning the diffusion time-scale posterior is also degenerate.

### Secondary issue: MassMatrix conditioning

The runtime warning `MassMatrix conditioning ratio 999.2 > 100` with `mass_clip=1000.0` means the clip is almost exactly at the conditioning boundary . The eigenvalue scaling is effectively saturated, making the mass-weighted spectral basis poorly conditioned and preventing differential mode selection.

***

## [Nemotron 3] Recommended Fixes

Three targeted changes to `configs/mps.yaml`:

**1. Raise `lam_s` from 0.1 → 1.0** — the spectral KL is too cheap; the model escapes it entirely. A 10× increase will put meaningful pressure on mode selection without destabilising reconstruction.

**2. Raise `nu` from 1.0 → 5.0** — the active-mode penalty must cost more than the reconstruction gain of keeping an extra mode alive.

**3. Lower `mass_clip` from 1000.0 → 100.0** — this directly addresses the conditioning warning (ratio 999 ≈ clip), giving the MassMatrix meaningful dynamic range to differentiate eigenvalues and break the uniform-mode symmetry.

Optionally raise `a_min` from 0.1 → 0.2 to narrow the prior and force sharper selection pressure — but start with the three above.

***

## [Sonnet 4.6] Recommended Fixes
```yaml
# configs/mps.yaml changes
lam_s: 2.0        # was 0.1
nu: 0.5           # active-mode penalty (add if not present)
mass_clip: 100    # was 1000  
dt_init: 0.01     # was 0.001
kl_z_warmup: 10   # epochs to ramp lam_z from 0 -> 1
```

