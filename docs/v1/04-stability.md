# 04 — Stability of Vibrational Energy-Based Learning Models

> **Context.** This document extends the dual-space learning framework of
> [`dual-space-learning/v2`](https://github.com/tuned-org-uk/dual-space-learning/tree/main/v2)
> into a vibrational, energy-based learning architecture grounded in Rayleigh's Theory of Sound.
> All notation follows the Vibrational Deduction Transformer (VDT) paper unless stated otherwise.

---

## 1. Why Stability Matters Here

The VDT and its energy-based extensions are unusual among neural architectures in that
their intermediate states are governed by a physically motivated dynamical system — a
discrete damped wave equation on a feature-space graph Laplacian `L_f`.  This is both
a strength (interpretable spectral geometry, principled inductive bias) and a liability:
unlike a standard transformer whose intermediate activations are unconstrained, the wave
update can blow up, oscillate without damping, or collapse to a trivial fixed point if
the governing parameters are badly chosen.

Stability therefore has two distinct meanings in this project and must be tracked at
two levels:

| Level | Question |
|---|---|
| **Numerical / wave stability** | Does the discrete wave update `Φ_L` remain bounded over recurrent depth K? |
| **Learning / optimisation stability** | Does gradient descent on the full VDT loss `L` converge, or does it diverge/oscillate? |

Both are analysed below, together with the interaction between them.

---

## 2. Notation Quick-Reference

| Symbol | Meaning |
|---|---|
| $$L_f$$ | Combinatorial graph Laplacian on feature space, $$L_f = D_f − W_f ⪰ 0$$ |
| $$M$$ | Positive-definite diagonal mass matrix (degree mass or taumode mass) |
| $$λ_k$$ | k-th eigenvalue of the generalised eigenproblem $$L_f u_k = λ_k M u_k$$; equals squared natural frequency $$ω_k²$$ |
| $$λ_max$$ | Largest eigenvalue of $$L_f$$ (or of the generalised problem w.r.t. M) |
| $$Δt$$ | Learnable time-step scalar (or diagonal) in the wave update |
| $$Γ$$ | Learnable damping tensor $$Γ ∈ ℝ^{n×d}$$, element-wise applied |
| $$Q_t$$ | Vibrational state at recurrent depth t, shape $$(n, d)$$ |
| $$S_{σ,M}$$ | Implicit mass-aware resolvent preconditioner $$(M + σ L_f)^{-1} M$$ |
| $$H_{σ,M}$$ | Preconditioned Hessian $$P_{σ,M} H$$, where $$H = (1/m) A^⊤ A$$ |
| $$κ(·)$$ | Condition number of a matrix |
| $$ϱ_t$$ | Signed density matrix $$ϱ_t = ϱ_t^+ − ϱ_t^-$$, each component PSD |

---

## 3. Numerical Stability of the Wave Update

### 3.1 Discrete wave equation

The vibrational state block implements (paper eq. 23):

$$
Q_{t+1} = 2 Q_t − Q_{t−1} − Δt² Q_t L_f^⊤ − Γ ⊙ (Q_t − Q_{t−1}) + Δt² B_t
$$

In modal coordinates $$Q̂_{t,k} = Q_t u_k$$ this decouples mode-by-mode into
independent damped harmonic oscillators (paper eq. 24):

$$
Q̂_{t+1,k} = 2 Q̂_{t,k} − Q̂_{t−1,k} − Δt² λ_k Q̂_{t,k} − Γ_k (Q̂_{t,k} − Q̂_{t−1,k}) + Δt² B̂_{t,k}
$$

with natural frequency $$ω_k = √λ_k$$.

### 3.2 CFL-type stability condition (explicit scheme)

For the undamped mode ($$Γ_k = 0$$, $$B̂_{t,k} = 0$$) the characteristic equation of the
recurrence is:

$$
r² − (2 − Δt² λ_k) r + 1 = 0
$$

The roots are $$r = [(2 − Δt² λ_k) ± √((2 − Δt² λ_k)² − 4)] / 2$$.  Both roots lie on
the unit circle (oscillatory but bounded) if and only if:

$$
|2 − Δt² λ_k| ≤ 2   ⟺   Δt² λ_k ≤ 4
$$

For a conservative bound covering all modes simultaneously:

$$
Δt ≤ √(2 / λ_max(L_f))          [CFL condition]
$$

This is the Courant–Friedrichs–Lewy condition for the graph wave equation, and it is
exactly the clipping rule implemented in `VibrationalStateBlock.forward`.

### 3.3 Effect of damping

With per-mode damping $$γ_k = Γ_k > 0$$, the characteristic roots are:

$$r = 1 − γ_k/2 ± i √(Δt² λ_k − (γ_k/2)²)$$
[underdamped: Δt² λ_k > (γ_k/2)²]

$$r = 1 − γ_k/2 ± √((γ_k/2)² − Δt² λ_k)$$
[overdamped:  Δt² λ_k < (γ_k/2)²]

For the underdamped case $$|r|² = (1 − γ_k/2)²$$; requiring $$|r| < 1$$ gives:

$$0 < γ_k < 2$$
[mode-k amplitude decays to zero]

For the overdamped case both roots are real and positive; the larger root is
$$1 − γ_k/2 + √(...)$$, which is less than 1 only when $$γ_k > Δt √λ_k$$.

**Rayleigh's critical damping** for mode k occurs at $$γ_k = 2 Δt √λ_k$$, giving the
fastest non-oscillatory decay.  Learning should adjust $$Γ_k$$ toward this value for
modes that need to settle quickly.

### 3.4 Implicit alternative (unconditionally stable)

The implicit resolvent from Part I of the paper provides an unconditionally stable
alternative:

$$
Q_{t+1} = (M + σ L_f)^{-1} M  Q_t + Δt² B_t
$$

This is positive-definite for all `σ ≥ 0` (Proposition 5.1 in the paper) and removes
the CFL constraint entirely, at the cost of solving a linear system per step.

---

## 4. Stability of the Preconditioned Gradient Descent

### 4.1 Convergence result (Proposition 5.1)

For the quadratic base model `J(x) = (1/2m) ‖Ax − b‖²`, preconditioned by
`P_{σ,M} = (M + σ L_f)^{-1} M`, the iterates satisfy (paper Appendix A):

$$
‖x_{t+1} − x*‖_{P⁻¹} ≤ (1 − η μ_{σ,M}) ‖x_t − x*‖_{P⁻¹}
$$

and, via norm equivalence:

$$
‖x_t − x*‖_2 ≤ √κ(P_{σ,M}) · (1 − η μ_{σ,M})^t · ‖x_0 − x*‖_2
$$

where `μ_{σ,M} = λ_min(H_{σ,M})` and the step size must satisfy `η < 2 / L_{σ,M}`.

### 4.2 Key stability parameters to monitor

The following scalar quantities fully characterise the convergence regime and should be
logged during training:

| Quantity | How to compute | What it tells you |
|---|---|---|
| $$λ_max(L_f)$$ | Largest eigenvalue of the Laplacian | Sets CFL bound on $$Δt$$; also denominator of Rayleigh quotient |
| $$κ(L_f)$$ | $$λ_max / λ_min_nonzero(L_f)$$ | Condition number of the graph geometry; large κ → slow convergence |
| $$κ(P_{σ,M})$$ | $$λ_max(P) / λ_min(P)$$ | Effective condition number after preconditioning; should be ≪ κ(L_f) |
| $$μ_{σ,M}$$ | $$λ_min(H_{σ,M})$$ | Lower bound on contraction; smaller → slower convergence |
| $$L_{σ,M}$$ | $$λ_max(H_{σ,M})$$ | Upper bound on step size $$η < 2/L_{σ,M}$$ |
| $$η · μ_{σ,M}$$ | product | Convergence rate per step; should be in (0, 1) |

### 4.3 Effect of σ on stability

Increasing `σ` (diffusion strength in the preconditioner):

- **Increases** mixing between features, making the gradient smoother.
- **Reduces** the effective condition number `κ(H_{σ,M})` when the graph smoothing
  aligns the level sets of J with the graph geometry.
- **Can slow** convergence if it over-smooths and drives `μ_{σ,M} → 0`.

Empirical check: plot `κ(H_{σ,M})` vs `σ` on a log-log scale for a sweep of values;
the optimal `σ*` minimises the condition number.

---

## 5. Stability of the Vibrational Recurrent Architecture (VDT)

The full recurrent system couples the transformer mixing block with the wave update.
Stability here is harder to prove in general but can be evaluated through a combination
of theoretical conditions and empirical diagnostics.

### 5.1 Energy monotonicity

Define the modal energy at depth t as:

$$
E_t = (1/d) Σ_k λ_k ‖Q̂_{t,k}‖²_F
$$

In a properly damped system with no external forcing (`B_t = 0`), `E_t` should decrease
monotonically.  With forcing, it should remain bounded.

**Diagnostic:** Log `E_t` at each recurrent depth during training.  If `E_t / E_{t-1} > 1`
consistently for many batches, the wave update is amplifying modes — a sign that `Δt`
is too large or `Γ` is insufficiently positive.

### 5.2 Signed density matrix stability

The signed density matrix state `ϱ_t = ϱ_t^+ − ϱ_t^-` introduces additional
dynamics.  For stability:

1. **PSD preservation of each component.** Verify during training that the smallest
   eigenvalue of `ϱ_t^+` and `ϱ_t^-` remains non-negative.  If it drops below zero,
   the softplus + symmetrisation in the update is failing to maintain PSD.

2. **Frobenius norm growth.** The depth penalty `L_depth = (1/K) Σ_t ‖ϱ_t‖²_F`
   in the VDT training objective (paper eq. 35) directly regularises this.  Monitor
   `‖ϱ_t‖_F` vs depth; it should saturate or decrease after a few steps.

3. **Signed occupancy bounds.** Track `occ_{t,k} = ϱ^+_{t,kk} − ϱ^-_{t,kk}` per mode.
   Stable inference should show these settling toward definite signs (positive for
   confirmed hypotheses, negative for refuted ones), not oscillating.

### 5.3 Representation collapse at large K

At large recurrent depth K, a plain recurrent transformer without spectral constraints
can collapse to a single attractor regardless of input ("rank collapse").  The Laplacian
constraint prevents this: the modal energy distribution `{λ_k ‖Q̂_{K,k}‖²}` should
remain spread across the spectrum rather than collapsing to a single mode.

**Diagnostic:** Compute the spectral entropy of the energy distribution:

$$
H_spectral = −Σ_k p_k log p_k,    p_k = λ_k ‖Q̂_{K,k}‖² / E_K
$$

A healthy model maintains high `H_spectral` across recurrent depths.  A collapsing
model will show `H_spectral → 0` as K grows.

---

## 6. Stability for Energy-Based Extensions

When the VDT is used as a backbone for energy-based learning (Option 2 from the
architecture design notes), additional stability criteria apply.

### 6.1 Energy landscape and minima

The Laplacian-regularised energy functional (paper eq. 9):

$$
J_λ(x) = (1/2m) ‖Ax − b‖² + (λ/2) x^⊤ L_f x
$$

is strictly convex when `(1/m) A^⊤A + λ L_f ≻ 0`.  This holds whenever:

- `A^⊤A ≻ 0` (full column rank of A), OR
- `A^⊤A` is rank-deficient but `λ > 0` and `L_f` covers the null space of A (i.e., the
  graph is connected and the only null-space vector of `L_f` is the constant vector,
  which is not in the null space of A if A has no all-equal columns).

**Check:** Verify `λ_min((1/m) A^⊤A + λ L_f) > 0` numerically; if it is near zero,
increase `λ` or add a small ridge `+ ε I` to ensure a unique minimum.

### 6.2 Wave relaxation as energy minimisation

The wave update can be seen as exploring the energy landscape via second-order dynamics.
For the vibrational energy-based model to converge to a minimum rather than oscillate
indefinitely, every mode k must satisfy the overdamped or critically damped condition
(see Section 3.3):

$$
γ_k ≥ Δt √λ_k          [mode-k is overdamped or critically damped]
$$

After learning, verify this holds for the majority of modes.  If many modes are
underdamped (`γ_k < Δt √λ_k`), the model is in an oscillatory regime: still
well-defined (bounded) as long as the CFL condition holds, but not converging to a
fixed point.

### 6.3 Fixed-point analysis

Assuming `B_t = B` (constant forcing), the wave update has a unique fixed point
`Q* = B · (L_f^⊤)^{-1}` (or more precisely the solution of `L_f^⊤ Q = B` in the
non-degenerate case). Stability of this fixed point requires all eigenvalues of the
linearised iteration matrix to lie strictly inside the unit circle, which is equivalent
to the per-mode conditions of Section 3.3.

---

## 7. Evaluation Protocol: Stability Metrics Checklist

Use the following checklist before and during every training run.

### Pre-training (graph and Laplacian construction)

- [ ] Compute `λ_max(L_f)` and record; set `Δt_init ≤ √(2/λ_max(L_f))`.
- [ ] Compute `κ(L_f) = λ_max / λ_min_nonzero`; if `κ > 10^4`, consider normalised
      Laplacian or graph re-weighting.
- [ ] Verify the graph is connected (zero eigenvalue of `L_f` has multiplicity 1);
      if disconnected, treat each component separately or add a small global edge weight.
- [ ] If using ArrowSpace taumode mass: verify `M_ii = (1 − λ^τ_i + ε)^{-1} > 0`
      for all i; check the range of mass values is not extreme (e.g. ratio
      `max(M_ii)/min(M_ii) < 100`).

### During training (per-epoch diagnostics)

| Metric | Normal range | Warning |
|---|---|---|
| $$Δt$$ (learnable) | $$(0, √(2/λ_max)]$$ | Clamp is triggering every step → Δt gradient is fighting the CFL bound |
| $$min_k γ_k$$ | $$> 0$$ | Any $$γ_k ≤ 0$$ means a mode is undamped or anti-damped |
| $$E_t / E_{t-1}$$ | $$≤ 1$$ for most steps | Consistently $$> 1$$ → wave is amplifying |
| $$‖ϱ_t‖_F$$ | Bounded, decreasing after early training | Unbounded growth → density matrix diverging |
| $$λ_min(ϱ_t^+)$$ | $$≥ 0$$ | Negative → PSD constraint violated |
| $$H_spectral$$ | Stable or increasing | Rapid decrease → representation collapse |
| $$η · μ_{σ,M}$$ | $$∈ (0, 1)$$ | $$> 1$$ → step size too large; $$≈ 0$$ → preconditioning degenerate |
| $$κ(P_{σ,M})$$ | $$< κ(L_f)$$ | Larger than $$κ(L_f)$$ → preconditioning is worsening conditioning |

### Post-training (convergence verification)

- [ ] Plot $$E_t$$ vs depth K for a held-out batch; confirm monotone decrease or bounded
      oscillation.
- [ ] Plot $$occ_{t,k}$$ for the top-m modes vs depth; confirm settling toward definite
      signs on the evaluation tasks.
- [ ] Compute the spectral entropy $$H_spectral$$ at K=2, 4, 8, 16; confirm it does not
      collapse to zero at large K.
- [ ] For the preconditioned GD baseline: plot $$‖x_t − x*‖_2$$ vs iteration; confirm
      linear convergence (straight line on log scale).

---

## 8. PyTorch Stability Diagnostics

```python
import torch

def stability_diagnostics(L_f, Q_states, rho_plus_list, rho_minus_list,
                           eigvals, dt, gamma):
    """
    L_f        : (d, d)   feature Laplacian
    Q_states   : list of (n, d) tensors, one per recurrent depth
    rho_plus_list, rho_minus_list : list of (m, m) tensors per depth
    eigvals    : (d,)   eigenvalues of L_f (ascending)
    dt         : scalar  current learnable time-step
    gamma      : (d,)   current damping per feature
    """
    diag = {}

    # 1. CFL condition
    lam_max = eigvals[-1].item()
    dt_max  = (2.0 / (lam_max + 1e-8)) ** 0.5
    diag["lambda_max"]  = lam_max
    diag["dt_max_CFL"]  = dt_max
    diag["dt_current"]  = dt.item() if hasattr(dt, 'item') else float(dt)
    diag["CFL_ok"]      = diag["dt_current"] <= dt_max

    # 2. Per-mode damping classification
    omega = eigvals.clamp(min=0).sqrt()          # natural frequencies
    crit_gamma = 2.0 * diag["dt_current"] * omega
    n_underdamped = (gamma.cpu() < crit_gamma.cpu()).sum().item()
    diag["n_underdamped_modes"] = n_underdamped
    diag["frac_underdamped"]    = n_underdamped / len(eigvals)

    # 3. Modal energy per depth
    energies = []
    for Q in Q_states:
        # project to modal coordinates: Q_hat = Q @ U (eigvecs not passed here,
        # but if available: Q_hat = Q @ eigvecs)
        # approximate: use Q itself as proxy (replace with Q @ eigvecs in practice)
        modal_energy = (Q.detach() ** 2 * eigvals.unsqueeze(0)).mean().item()
        energies.append(modal_energy)
    diag["modal_energy_per_depth"] = energies

    if len(energies) > 1:
        ratios = [energies[t] / (energies[t-1] + 1e-10) for t in range(1, len(energies))]
        diag["energy_amplified"] = any(r > 1.05 for r in ratios)

    # 4. Spectral entropy of energy distribution at final depth
    Q_K = Q_states[-1].detach()
    mode_energies = (Q_K ** 2).mean(0) * eigvals  # (d,)
    p = mode_energies / (mode_energies.sum() + 1e-10)
    H = -(p * (p + 1e-10).log()).sum().item()
    diag["spectral_entropy_K"] = H

    # 5. Density matrix PSD check
    min_eigs_plus  = []
    min_eigs_minus = []
    frob_norms     = []
    for rp, rm in zip(rho_plus_list, rho_minus_list):
        eig_p = torch.linalg.eigvalsh(rp.detach())
        eig_m = torch.linalg.eigvalsh(rm.detach())
        min_eigs_plus.append(eig_p.min().item())
        min_eigs_minus.append(eig_m.min().item())
        frob_norms.append((rp - rm).detach().norm('fro').item())
    diag["min_eig_rho_plus"]  = min(min_eigs_plus)
    diag["min_eig_rho_minus"] = min(min_eigs_minus)
    diag["max_frob_signed"]   = max(frob_norms)
    diag["rho_psd_ok"] = (diag["min_eig_rho_plus"] >= -1e-6 and
                          diag["min_eig_rho_minus"] >= -1e-6)

    return diag


def log_preconditioner_stability(A, L_f, M_diag, sigma, eta):
    """
    Quick check of preconditioned Hessian condition number and step-size validity.
    A      : (m, n) data matrix
    L_f    : (n, n) feature Laplacian
    M_diag : (n,)   diagonal mass
    sigma  : float  diffusion strength
    eta    : float  learning rate
    """
    n = L_f.shape[0]
    M = torch.diag(M_diag)
    A_sigma = M + sigma * L_f              # (n, n)
    P_inv   = torch.linalg.solve(A_sigma, M)  # P_{σ,M} = A_sigma^{-1} M

    H       = (A.T @ A) / A.shape[0]      # (n, n)
    H_prec  = P_inv @ H                   # preconditioned Hessian

    eigs    = torch.linalg.eigvalsh(H_prec)
    mu      = eigs[eigs > 1e-10].min().item()
    L_spec  = eigs.max().item()
    kappa   = L_spec / (mu + 1e-10)

    return {
        "mu_sigma_M":   mu,
        "L_sigma_M":    L_spec,
        "kappa_H_prec": kappa,
        "eta_ok":       eta < 2.0 / (L_spec + 1e-10),
        "convergence_rate": 1.0 - eta * mu,
    }
```

---

## 9. Summary: Stability Hierarchy

```
Level 1 — Graph geometry
  └─ λ_max(L_f) finite and well-conditioned
       └─ CFL condition Δt ≤ √(2/λ_max)
            └─ Level 2 — Wave update per mode k
                 └─ γ_k > 0  (damped)
                      └─ Per-mode: underdamped/critical/overdamped classification
                           └─ Level 3 — Recurrent VDT dynamics
                                └─ E_t monotone or bounded
                                     └─ H_spectral stable (no collapse)
                                          └─ Level 4 — Density matrix
                                               └─ ϱ^±_t PSD, ‖ϱ_t‖_F bounded
                                                    └─ occ_{t,k} settle to signs
                                                         └─ Level 5 — Optimiser
                                                              └─ κ(P_{σ,M}) ≪ κ(L_f)
                                                                   └─ η < 2/L_{σ,M}
                                                                        └─ linear convergence
```

Each level's stability conditions are **necessary but not sufficient** for the next.
Monitor the full hierarchy during development; fix lower levels first before diagnosing
higher-level instabilities.

---

## 10. References

- Rayleigh, Lord. *The Theory of Sound*, Vol. I, 2nd ed., Macmillan, 1894.
  — Classical damped vibration analysis; modal decomposition framework.
- Osher, S., Shi, W., and Zhu, W. "Laplacian smoothing gradient descent."
  *Research in the Mathematical Sciences*, 9(3):39, 2022.
  — Resolvent preconditioner stability and its connection to the heat equation.
- Merris, R. "Laplacian matrices of graphs: a survey."
  *Linear Algebra and its Applications*, 197–198:143–176, 1994.
  — Spectral properties of graph Laplacians used throughout this document.
- Moriondo, L. "Vibrational Deduction Transformers: From Mass–Spring Feature Geometry
  to Modal Dynamics for Recurrent Learning." Preprint, 2 June 2026.
  — Source paper for all notation and architecture details.
- Moriondo, L. "Graph Wiring and ArrowSpace: Spectral pathways for structure-aware
  learning." Technical report, 2026.
  — ArrowSpace taumode mass construction and spectral indexing.
