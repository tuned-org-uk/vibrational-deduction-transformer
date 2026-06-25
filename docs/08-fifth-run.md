# Run 5 -- Cora, 50 epochs, Option D active (issue #82)

## Configuration

- Config: `configs/mps.yaml` (at time of run: q=2, lam_s=0.4, tau=0.5, nu=0.3, nu_entropy absent)
- Dataset: Cora (N=2708, D=1433, 7 classes)
- Device: MPS (Apple Silicon)
- Trainable params: 129,829,075
- Best val loss: -1840.9564 (epoch 50)

## Observations

### ELBO trajectory

The train and validation ELBO descend steeply during epochs 1-10 (warmup)
and plateau near -1840 after epoch 42.  The gap between val and train
reconstruction widens after epoch 42, indicating late-epoch encoder
overfitting once KL regularisation has fully collapsed.

### KL term collapse

All three KL terms decay monotonically to zero:

| Term    | Epoch 2 | Epoch 50 |
|---------|---------|----------|
| kl_z    | 0.584   | 0.013    |
| kl_S    | 0.878   | ~0 (eps) |
| kl_tau  | 0.644   | 0.004    |

kl_S reaches machine epsilon by epoch 28.  After that point the gradient
through SpectralLoadingDecoder is effectively zero -- the wiring decoder is
frozen while reconstruction and kl_z continue to interact.

### Uniform-mode attractor

The spectral_kl_health_check fires `mode_explosion` (N_active = 2.0/2 = 100%)
every epoch from epoch 5 onward.  Both vibrational modes remain maximally
active throughout -- no spectral selection ever occurs.

Root cause: with q=2 the mode-weight simplex is one-dimensional.  The
uniform attractor pi=[0.5, 0.5] is the unique fixed point of
softmax(log_a - log_b) when log_a ~= log_b.  At this fixed point the
gradient of Shannon entropy H = -sum pi_k log(pi_k) with respect to
log_a and log_b is identically zero.  The Option D entropy ceiling penalty
nu_entropy * H therefore contributes no gradient signal for breaking
symmetry when q=2, regardless of the value of nu_entropy.

Additionally, nu_entropy was absent from configs/mps.yaml; from_config()
silently defaulted to 0.5, which is insufficient to dominate the
reconstruction scale (~-1840 nats) even if q were large enough.

### float32 underflow on kl_S (MPS, epochs 35, 43, 47-50)

kl_S is reported as -0.0000 in several late epochs.  This is a float32
underflow artefact on MPS: the true value is a small positive number that
rounds to -epsilon in the logged CSV.  It is not a sign error in the KL.

## Recommended config changes (applied in commit for this run)

See `configs/mps.yaml` commit history.  Summary:

```
model.q:          2  -> 8   minimum DoF to escape uniform attractor
model.tau_modes:  2  -> 8   kept == q
model.hidden_dim: 32 -> 64  absorb extra mode capacity
model.lam_s:      0.4 -> 1.0  hold kl_S > 0
model.tau:        0.5 -> 1.0  resist kl_tau collapse
training.nu_entropy: added 2.0  (must be explicit; 0.5 default too weak)
training.nu:      0.3 -> 1.0
training.q_min:   2   -> 4
training.epochs:  50  -> 100  mode selection needs room to emerge
```

## What to look for in run 6

- kl_S should remain above 0.1 past epoch 40 with lam_s=1.0 and q=8.
- N_active should drop below q/2 (i.e. < 4) at some point, indicating
  genuine spectral selection.
- entropy_S should decrease from its initial value (~log(8) ~ 2.08 nats)
  toward 0.5-1.0 nats as sparse mode selection emerges.
- If kl_S still collapses, increase lam_s to 2.0 and check q_min pressure.
- If N_active stays at 8.0/8, the active-mode floor penalty is providing
  upward pressure but the ceiling is not providing downward pressure.
  Consider increasing nu_entropy to 5.0 or switching to a Dirichlet prior
  on the mode weights (Option E).
