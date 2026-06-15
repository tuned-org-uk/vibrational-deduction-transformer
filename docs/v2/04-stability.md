# 04 — Stability of WAE v2

> **Context.** This document extends the v1 stability analysis for the WAE v2
> Spectral-PPCA architecture. All v1 stability conditions (CFL, damping, density
> matrix PSD, preconditioned GD convergence) apply unchanged. Two new stability
> considerations arise from the v2 upgrades: the on-the-fly sample Laplacian
> feedback loop and mode weight collapse.

For the full v1 stability hierarchy (CFL condition, per-mode damping, energy
monotonicity, signed density matrix PSD, preconditioned GD convergence), refer
to the corresponding sections of the v1 `04-stability.md`. This document records
only the v2 additions.

---

## Notation Quick-Reference (v2 additions)

| Symbol | Meaning |
|---|---|
| \(L_s\) | Sample-graph Laplacian built from batch of latent means \(\{\mu_{x,n}\}\) |
| \(\beta\) | Diffusion strength in Laplacian-precision prior \((I + \beta L_s)^{-1}\) |
| \(\omega_k\) | Mode weight for Laplacian mode \(k\); \(\omega_k > 0\) |
| \(a_k, b_k\) | Gamma shape and rate parametrising \(q(\omega_k) = \mathrm{Gamma}(a_k, b_k)\) |
| \(\tau\) | Temperature in τ-mode prior \(p(\omega_k) = \mathrm{Exp}(\tau\lambda_k)\) |
| \(q_{\min}\) | Minimum number of active modes: \(|\{k : \mathbb{E}[\omega_k] > \delta\}| \ge q_{\min}\) |

All other symbols follow the v1 notation table in `04-stability.md`.

---

## 1. v1 Stability Hierarchy (Retained)

All five levels of the v1 stability hierarchy apply unchanged:

```
Level 1 — Graph geometry: λ_max(L_f), CFL condition
Level 2 — Wave update per mode: γ_k > 0, underdamped/overdamped classification
Level 3 — Recurrent VDT dynamics: E_t monotone, H_spectral stable
Level 4 — Density matrix: ϱ±_t PSD, ‖ϱ_t‖_F bounded
Level 5 — Optimiser: κ(P_{σ,M}) ≪ κ(L_f), η < 2/L_{σ,M}, linear convergence
```

Fix lower-level instabilities before diagnosing higher-level ones.

---

## 2. v2 Addition: Sample Laplacian Feedback Loop

### 2.1 The feedback structure

In v2, \(L_s\) is built on-the-fly from the batch of latent means \(\{\mu_{x,n}\}\),
which are themselves outputs of the encoder trained under the KL that uses \(L_s\).
This creates a feedback loop:

```
encoder parameters θ  →  latent means μ  →  L_s(μ)  →  KL(q(z) ∥ N(0,(I+βLs)⁻¹))  →  θ
```

### 2.2 Stabilisation: stop-gradient

The primary stabilisation is a **stop-gradient** on the \(L_s\) construction:

```python
Ls = build_knn_laplacian(mu_z.detach())   # no gradient through Ls
```

This breaks the feedback loop: gradients flow through the KL formula
but not through the Laplacian construction itself. The encoder is updated
to make latent codes smooth under the current graph, but the graph is not
differentiated with respect to the encoder.

### 2.3 Alternative: frozen base Laplacian

For additional stability, use a **frozen** sample Laplacian built from base
encoder embeddings (pre-WAE) as a fixed structural prior, updated only every
\(T_{\text{refresh}}\) epochs. This reduces the feedback to a slow outer loop.

Recommended for early training: use frozen \(L_s\) for epochs 1–\(T_{\text{warmup}}\),
then switch to stop-gradient on-the-fly construction.

### 2.4 Diagnostic

Monitor the **Laplacian KL per epoch**: it should decrease monotonically after warmup.
A sudden spike indicates the feedback loop is destabilising the sample graph.

| Metric | Normal range | Warning |
|---|---|---|
| `KL_Lap` per epoch | Decreasing after warmup | Spike > 2× previous epoch |
| `‖Ls‖_F` across batches | Stable (< 20% variation) | High variance → graph ill-conditioned |
| `λ_max(Ls)` | Bounded (< `λ_max(Lf) × 10`) | Exceeds bound → kNN radius too large |

---

## 3. v2 Addition: Mode Weight Collapse

### 3.1 The collapse risk

The Gamma prior \(p(\omega_k) = \mathrm{Exp}(\tau\lambda_k)\) has rate \(\tau\lambda_k\).
For large \(\tau\lambda_k\) (high-frequency modes), the posterior \(q(\omega_k)\) can collapse
to a near-zero point mass. This is correct behaviour (mode selection), but if it
extends to low-frequency modes, the loading matrix \(W\) degenerates.

### 3.2 Stabilisation: shape parameter floor

Apply a minimum floor on the Gamma shape parameter \(a_k\):

```python
a_k = torch.clamp(a_k, min=a_min)   # e.g. a_min = 0.1
```

This prevents the distribution from collapsing to a point mass for any mode.

### 3.3 Stabilisation: active mode count constraint

Track the number of active modes \(N_{\text{active}} = |\{k : \mathbb{E}[\omega_k] > \delta\}|\)
and add a soft Lagrange multiplier penalising \(N_{\text{active}} < q_{\min}\):

```python
active = (omega_mean > delta).float().sum()
mode_floor_penalty = torch.relu(q_min - active)
loss = elbo + nu * mode_floor_penalty
```

### 3.4 Diagnostic

| Metric | Normal range | Warning |
|---|---|---|
| `N_active` (active mode count) | \(\ge q_{\min}\) throughout | Drops below \(q_{\min}\) |
| `min_k E[ωk]` for low-freq modes | \(> \delta = 0.01\) | Near zero → low-freq collapse |
| `τ-mode KL` | Stable decrease | Sudden spike → τ too large for data |
| `‖Ŵ‖_F` (artefact loading norm) | Stable | Near zero → W degenerate |

---

## 4. v2 Addition: Spectral Eigenvalue Floor

The spectral-basis KL has precision proportional to \(\Lambda_{1:q}\). If
\(\lambda_1 \approx 0\) (Fiedler vector near zero for a nearly disconnected graph),
the prior on the first mode becomes nearly flat, causing gradient variance.

Apply a small floor in the KL computation only (do not modify the actual eigenvalues):

```python
eigvals_for_kl = eigvals_q.clamp(min=1e-3)
```

This does not affect the spectral geometry but prevents numerical instability in
the KL gradient.

---

## 5. Updated Full Stability Checklist

### Pre-training

- [ ] v1 checks: `λ_max(Lf)`, `κ(Lf)`, graph connectivity, mass matrix range.
- [ ] Set `Δt_init ≤ √(2/λ_max(Lf))`.
- [ ] Build initial frozen `Ls` from base embeddings. Verify `λ_max(Ls) < 10 × λ_max(Lf)`.
- [ ] Set `a_min ≥ 0.1` in `ModeWeightHead` config.
- [ ] Set `q_min ≥ max(2, q // 4)` in config.
- [ ] Apply eigenvalue floor `clamp(min=1e-3)` in all KL computations.

### During training (per-epoch diagnostics)

| Metric | Normal range | Warning |
|---|---|---|
| All v1 metrics | See v1 `04-stability.md` | See v1 |
| `KL_Lap` | Decreasing after warmup | Spike > 2× previous epoch |
| `KL_S` (spectral basis) | Decreasing | Spike or oscillation |
| `τ-mode KL` | Stable decrease | Sudden spike |
| `N_active` | ≥ `q_min` | Drops below `q_min` |
| `‖Ls‖_F` | Stable | High batch-to-batch variance |

### Post-training

- [ ] v1 checks: `E_t` vs depth, `occ_{t,k}` settling, `H_spectral` not collapsing.
- [ ] Plot `E[ωk]` vs mode index `k`: should show low-freq concentration with
  some high-freq activity (not all zero, not all equal).
- [ ] Compute `‖Ŵ‖_F` and `‖S_memory‖_F`: both should be non-degenerate.
- [ ] Test associative memory retrieval: for each spectral key `ŵk`, verify
  `argmax softmax(S_I⊤ ŵk)` returns the correct value `d_θ(ŵk)`.
- [ ] Compute ELBO Bayes factor if comparing ArrowSpace index candidates.

---

## 6. Stability Hierarchy (v2 extended)

```
Level 1 — Graph geometry
  └─ λ_max(Lf) finite; CFL condition
       └─ Level 2 — Wave update per mode
            └─ γk > 0; underdamped/overdamped classification
                 └─ Level 3 — Recurrent VDT dynamics
                      └─ E_t monotone; H_spectral stable
                           └─ Level 4 — Density matrix
                                └─ ϱ±_t PSD; ‖ϱ_t‖_F bounded
                                     └─ Level 5 — Optimiser
                                          └─ κ(P_{σ,M}) ≪ κ(Lf); linear convergence
                                               └─ Level 6 (v2) — Sample Laplacian
                                                    └─ Stop-gradient on Ls
                                                         └─ KL_Lap monotone after warmup
                                                              └─ Level 7 (v2) — Mode weights
                                                                   └─ N_active ≥ q_min
                                                                        └─ ‖Ŵ‖_F non-degenerate
```
