# First run

## What is working

- Pre-flight passes cleanly with `dt_init=0.001`, `mass_clip=1e3`, `q=8`. 
- Loss is moving in the right direction: `train` goes from `179.3` → `-66.5`, `val` from `-4.6` → `-128.6`.
- `kl_z` is dropping (16.2 → 9.9), meaning the VAE posterior is converging toward the prior.
- `kl_tau` is very small and dropping (1.4 → 0.6), meaning mode-frequency KL is well-behaved.

## The one real warning: `mode_explosion: True`

Every epoch reports:

```
{'kl_z_ok': True, 'kl_S_ok': True, 'kl_tau_ok': True,
 'mode_collapse': False, 'mode_explosion': True}
```

`mode_explosion` being `True` means `kl_S` is very large relative to the expected range for `q=8` modes. Looking at the numbers:

- `kl_S = 197.97` (epoch 1), `195.15` (epoch 2) — essentially flat and enormous.
- `recon` is large and negative (`-220`, `-334`), which combined with a huge `kl_S` suggests the spectral basis KL term is not being regularised effectively yet.

`kl_S` represents the KL divergence on the spectral loading matrix **S** — how far the learned spectral basis deviates from the prior. A value of ~195 at epoch 2 is a strong signal that either:

1. **`lam_s` is too small** — the weight on `kl_S` in the ELBO is not penalising spectral explosion enough. Current value in `mps.yaml` is `0.01`, which is very loose.
2. **The spectral loading decoder is initialising with very large weights**, and two epochs is not enough for the spectral basis KL to start pulling them back toward the prior.
3. **The learning rate is too high for the spectral path**, causing it to overfit spectrally in early epochs.

## What the loss decomposition means

The ELBO at epoch 2 is approximately:

\[ \mathcal{L} = \underbrace{-334.2}_{\text{recon}} + \underbrace{9.9}_{kl_z} + \underbrace{195.1}_{kl_S} + \underbrace{0.6}_{kl_\tau} = -128.6 \]

The reconstruction term is dominating and driving the total loss negative, while `kl_S` is large but not large enough to override it. This is the classic ELBO trade-off in early training: the decoder learns to reconstruct aggressively before the KL terms have tightened. The spectral basis is "exploding" because there is very little penalty on it relative to the reconstruction reward.

## Recommended next steps

### 1. Increase `lam_s` to penalise spectral explosion

In `configs/mps.yaml`, raise `lam_s` from `0.01` to something that makes `kl_S` cost matter:

```yaml
model:
  lam_s: 0.1    # was 0.01 -- 10x stronger spectral regularisation
```

If `kl_S ~195` and `lam_s=0.01`, the effective contribution is `0.01 * 195 = 1.95` — almost invisible in the ELBO. With `lam_s=0.1`, it becomes `19.5`, which is comparable to `kl_z`.

### 2. Run for more epochs before drawing conclusions

Two epochs at `~700s/epoch` (~23 minutes total) is a good smoke test but not enough to see whether `kl_S` stabilises or keeps growing. Run 10–20 epochs to see the trajectory of `kl_S` over time before changing architecture.

### 3. Watch `kl_S` trend, not just the absolute value

The important question is: is `kl_S` decreasing, flat, or increasing epoch over epoch? `195.1` at epoch 2 vs `197.9` at epoch 1 is a tiny drop. If it keeps decreasing gradually, the spectral path is simply slow to regularise. If it grows, that is a genuine instability.

### 4. Consider KL annealing for `kl_S`

A common technique for spectral or structured KL terms that start large is to anneal the weight from 0 to its target over the first N epochs, rather than applying full `lam_s` from step 1. That gives the reconstruction path time to warm up before the spectral prior pulls back.

## Summary of current state

| Signal | Status | Action |
|---|---|---|
| Training runs end-to-end | Good | None |
| Pre-flight passes | Good | None |
| `kl_z` decreasing | Good | Monitor |
| `kl_tau` small and decreasing | Good | Monitor |
| `kl_S ~195`, flat | Concerning | Increase `lam_s` to 0.1 |
| `mode_explosion: True` | Warning | Consequence of low `lam_s` |
| 255M params for this model size | Large | Consider reducing `hidden_dim` further for faster iteration |

The code is working. The next step is tuning `lam_s` and running longer to get a cleaner picture of the spectral dynamics.
