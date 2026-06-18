# 04 — Stability of the Vibrational Deduction Transformer

> **Context.** This document covers the stability analysis for the VDT
> Spectral-PPCA architecture. All base stability conditions (CFL, damping,
> density matrix PSD, preconditioned GD convergence) apply unchanged and are
> extended with two VDT-specific diagnostics: mode weight collapse and spectral
> eigenvalue floor.

> **Design note.** The ArrowSpace index `I` enters the ELBO solely through the
> pre-computed eigenpair `(U_{1:q}, Lambda_{1:q})` of `L(I)`. These are frozen
> constants at training time — no Laplacian is evaluated or inverted at runtime.
> This eliminates the sample-Laplacian feedback loop present in earlier drafts
> and reduces the stability hierarchy from 7 levels to 6.

---

## Notation Quick-Reference

| Symbol | Meaning |
|---|---|
| `omega_k` | Mode weight for Laplacian mode `k`; `omega_k > 0` |
| `a_k, b_k` | Gamma shape and rate parametrising `q(omega_k) = Gamma(a_k, b_k)` |
| `tau` | Temperature in tau-mode prior `p(omega_k) = Exp(tau*lambda_k)` |
| `q_min` | Minimum number of active modes: `|{k : E[omega_k] > delta}| >= q_min` |
| `Lambda_{1:q}` | Frozen eigenvalues from pre-computed eigendecomposition of `L(I)` |

All other symbols follow the notation established in the VDT paper (Moriondo, 2026).

---

## 1. Base Stability Hierarchy (Retained)

All five base stability levels apply unchanged:

```
Level 1 -- Graph geometry: lambda_max(L_f), CFL condition
Level 2 -- Wave update per mode: gamma_k > 0, underdamped/overdamped classification
Level 3 -- Recurrent VDT dynamics: E_t monotone, H_spectral stable
Level 4 -- Density matrix: rho+_t PSD, ||rho_t||_F bounded
Level 5 -- Optimiser: kappa(P_{sigma,M}) << kappa(Lf), eta < 2/L_{sigma,M}, linear convergence
```

Fix lower-level instabilities before diagnosing higher-level ones.

---

## 2. Mode Weight Collapse

### 2.1 The collapse risk

The Gamma prior \(p(\omega_k) = \mathrm{Exp}(\tau\lambda_k)\) has rate \(\tau\lambda_k\).
For large \(\tau\lambda_k\) (high-frequency modes), the posterior \(q(\omega_k)\) can collapse
to a near-zero point mass. This is correct behaviour (mode selection), but if it
extends to low-frequency modes, the loading matrix \(W\) degenerates.

### 2.2 Stabilisation: shape parameter floor

Apply a minimum floor on the Gamma shape parameter \(a_k\):

```python
a_k = torch.clamp(a_k, min=a_min)   # e.g. a_min = 0.1
```

This prevents the distribution from collapsing to a point mass for any mode.

### 2.3 Stabilisation: active mode count constraint

Track the number of active modes \(N_{\text{active}} = |\{k : \mathbb{E}[\omega_k] > \delta\}|\)
and add a soft Lagrange multiplier penalising \(N_{\text{active}} < q_{\min}\):

```python
active = (omega_mean > delta).float().sum()
mode_floor_penalty = torch.relu(q_min - active)
loss = elbo + nu * mode_floor_penalty
```

### 2.4 Diagnostic

| Metric | Normal range | Warning |
|---|---|---|
| `N_active` (active mode count) | `>= q_min` throughout | Drops below `q_min` |
| `min_k E[omega_k]` for low-freq modes | `> delta = 0.01` | Near zero: low-freq collapse |
| `tau-mode KL` | Stable decrease | Sudden spike: tau too large for data |
| `||W_hat||_F` (artefact loading norm) | Stable | Near zero: W degenerate |

---

## 3. Spectral Eigenvalue Floor

The spectral-basis KL has precision proportional to \(\Lambda_{1:q}\). If
\(\lambda_1 \approx 0\) (Fiedler vector near zero for a nearly disconnected graph),
the prior on the first mode becomes nearly flat, causing gradient variance.

Apply a small floor in the KL computation only (do not modify the actual eigenvalues):

```python
eigvals_for_kl = eigvals_q.clamp(min=1e-3)
```

This does not affect the spectral geometry but prevents numerical instability in
the KL gradient. Both `spectral_basis_kl` and `tau_mode_kl` should apply this floor.

---

## 4. Full Stability Checklist

### Pre-training

- [ ] Base checks: `lambda_max(Lf)`, `kappa(Lf)`, graph connectivity, mass matrix range.
- [ ] Set `dt_init <= sqrt(2/lambda_max(Lf))`.
- [ ] Verify frozen `Lambda_{1:q}` from eigendecomposition of `L(I)` is non-degenerate.
- [ ] Set `a_min >= 0.1` in `ModeWeightHead` config.
- [ ] Set `q_min >= max(2, q // 4)` in config.
- [ ] Apply eigenvalue floor `clamp(min=1e-3)` in all KL computations.
- [ ] If using a normalised symmetric Laplacian with spectral density near
  `lambda = 1` (regular graphs, k-NN graphs on uniform point clouds),
  construct `MassMatrix` with `mass_clip=1e3` to avoid spurious
  conditioning warnings and preconditioner instability (see Section 7).

### During training (per-epoch diagnostics)

| Metric | Normal range | Warning |
|---|---|---|
| All base metrics | See base stability hierarchy | See base |
| `KL_S` (spectral basis) | Decreasing | Spike or oscillation |
| `tau-mode KL` | Stable decrease | Sudden spike |
| `N_active` | >= `q_min` | Drops below `q_min` |
| `log_var` (encoder) | In (-10, 4) without saturation | RuntimeWarning from WiringEncoder |

### Post-training

- [ ] Base checks: `E_t` vs depth, `occ_{t,k}` settling, `H_spectral` not collapsing.
- [ ] Plot `E[omega_k]` vs mode index `k`: should show low-freq concentration with
  some high-freq activity (not all zero, not all equal).
- [ ] Compute `||W_hat||_F` and `||S_memory||_F`: both should be non-degenerate.
- [ ] Test associative memory retrieval: for each spectral key `w_hat_k`, verify
  `argmax softmax(S_I^T w_hat_k)` returns the correct value `d_theta(w_hat_k)`.
- [ ] Compute ELBO Bayes factor if comparing ArrowSpace index candidates.

---

## 5. Stability Hierarchy (Full)

```
Level 1 -- Graph geometry
  +-- lambda_max(Lf) finite; CFL condition
       +-- Level 2 -- Wave update per mode
            +-- gamma_k > 0; underdamped/overdamped classification
                 +-- Level 3 -- Recurrent VDT dynamics
                      +-- E_t monotone; H_spectral stable
                           +-- Level 4 -- Density matrix
                                +-- rho+_t PSD; ||rho_t||_F bounded
                                     +-- Level 5 -- Optimiser
                                          +-- kappa(P_{sigma,M}) << kappa(Lf); linear convergence
                                               +-- Level 6 -- Mode weights
                                                    +-- N_active >= q_min
                                                         +-- ||W_hat||_F non-degenerate
```

---

## 6. log_var Clamping in WiringEncoder

The posterior log-variance produced by `WiringEncoder` is clamped to `[-10, 4]`
before the reparameterisation step. The physical meaning of the bounds is:

| Bound | log_var value | sigma = exp(0.5 * log_var) | Interpretation |
|---|---|---|---|
| Lower | -10 | ~0.007 | Very narrow posterior; near-deterministic encoder |
| Upper |  4  | ~7.4   | Very wide posterior; near-uninformative encoder |

**Lower bound -10** prevents numerical underflow in `exp(0.5 * log_var)` and
stops the posterior from collapsing to a point mass (which would make the KL
term diverge to `-inf`).

**Upper bound 4** prevents KL explosion. The bound is deliberately generous
to allow the encoder to maintain high uncertainty during early training. If
`kl_z` consistently exceeds `100 * latent_dim` during training, tighten
this bound to `2.0` (sigma ~2.7).

`WiringEncoder.forward()` emits a `RuntimeWarning` whenever the raw
`log_var_head` output hits either bound before clamping. Persistent
saturation warnings indicate the head may need a learning-rate adjustment
or gradient clipping.

---

## 7. MassMatrix Singularity at lambda = 1

For a **normalised symmetric Laplacian** `L_sym = I - D^{-1/2} A D^{-1/2}`,
eigenvalues lie in `[0, 2]`. The Rayleigh-damping mass:

```
M_ii = 1 / (1 - lambda_i^tau)
```

diverges at `lambda = 1` for all `tau`, because `lambda = 1` is the mode at
its natural frequency where kinetic and potential energy are in equipartition.

In practice, `eps` (default `1e-6`) prevents exact division by zero but
leaves `M_ii ~ 1/eps = 10^6` near `lambda = 1`. This causes:

- The conditioning ratio in `pre_training_checks` level 3 to exceed 100 and
  emit a spurious warning for any graph with significant spectral density
  near `lambda = 1`. This is common for **regular graphs** and **k-NN graphs
  on uniform point clouds**.
- Numerical instability in the Tikhonov preconditioner
  `H_prec = sigma * M_diag * I + L_f` inside `log_preconditioner_stability`.
- The time step `dt` to be dominated by the mass spike rather than the true
  spectral structure of the graph.

### Recommended fix: use mass_clip

Construct `MassMatrix` with a finite `mass_clip` to clamp the singularity:

```python
# Moderate graphs (k-NN, citation networks)
mass = MassMatrix(eigvals, tau=0.5, mass_clip=1e3)

# Sparse graphs where the singularity is rarely excited
mass = MassMatrix(eigvals, tau=0.5, mass_clip=1e4)
```

This clamps `M_diag` entries to at most `mass_clip`, preventing spurious
conditioning warnings and preconditioner instability without materially
affecting modes far from `lambda = 1`.

### How to tell if the warning is genuine

| Scenario | Action |
|---|---|
| Graph is regular or k-NN on uniform data | Set `mass_clip=1e3`; warning is caused by the singularity |
| Graph is sparse with irregular degree distribution | Investigate the Laplacian; genuine ill-conditioning |
| Conditioning ratio >> 1e4 even after setting mass_clip=1e3 | Genuine issue; check kNN construction |

### Interaction with pre_training_checks level 3

`pre_training_checks` checks the conditioning ratio of the `M_diag` passed
by the caller. If `MassMatrix` was constructed without `mass_clip` and the
graph has density near `lambda = 1`, the ratio will exceed 100 even for
well-behaved graphs. The warning message now explicitly notes this and
recommends `mass_clip=1e3`.
