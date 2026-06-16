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
Level 5 -- Optimiser: kappa(P_{sigma,M}) << kappa(L_f), eta < 2/L_{sigma,M}, linear convergence
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

### During training (per-epoch diagnostics)

| Metric | Normal range | Warning |
|---|---|---|
| All base metrics | See base stability hierarchy | See base |
| `KL_S` (spectral basis) | Decreasing | Spike or oscillation |
| `tau-mode KL` | Stable decrease | Sudden spike |
| `N_active` | >= `q_min` | Drops below `q_min` |

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
