# Branching Paths: VDT Algorithm Tracks

This document describes the six algorithm tracks for the Vibrational Deduction
Transformer (VDT) Spectral-PPCA architecture. Each track retains its base
definition and is annotated with the specific VDT component that applies to it.

The shared foundation in all cases remains the feature-space graph Laplacian \(L_f\),
the mass matrix \(M\), the Rayleigh quotient \(R_M(z)\), and the discrete damped wave
operator \(\Phi_L\). All tracks additionally have access to the spectral eigenbasis
\(U_{1:q}\) and \(\Lambda_{1:q}\) — both frozen constants from the pre-computed
eigendecomposition of \(L(\mathcal{I})\) — and the spectral artefact.

---

## Option 1 — Deterministic Vibrational Autoencoder

### Base Architecture

**Encoder** — VDT recurrence for \(K\) steps, project to modal coordinates:

$$z = \mathrm{pool}(Q_K \, U_m) \in \mathbb{R}^m$$

**Decoder** — re-expand via \(U_m^\top\), reconstruct \(\hat{X}_0\).

### VDT Extension

- Replace decoder `Linear(m, d)` with `SpectralLoadingDecoder(q=m, d=d)` in
  deterministic mode (`omega` fixed to ones, no KL).
- Add `spectral_freq_cost` ablation: compare `J_freq` hard penalty vs the soft
  tau-mode weighting on reconstruction quality.
- Use `extract_spectral_artefact()` post-training to seed associative memory
  for a downstream classifier (Option 6).

### Training Objective

$$\mathcal{L} = \|X_0 - \hat{X}_0\|_F^2 + \alpha\, J_{\text{freq}}(L(z)) + \beta\, R_M(z)$$

### Density Matrix as Bottleneck

The signed density matrix \(\varrho_t = \varrho_t^+ - \varrho_t^-\) serves as a
structured bottleneck via low-rank factorisation \(\varrho = V V^\top - W W^\top\) with
trace penalty. \(\varrho_K\) can additionally initialise `SpectralAssociativeMemory`.

---

## Option 2 — Energy-Based Vibrational Model

### Base Architecture

Proposer produces \(Q_0 = f_\phi(X_0)\); energy relaxation via \(K\) preconditioned
steps; prediction head on \(Q_K\).

### VDT Extension

- Replace the unconstrained proposer linear layer with `SpectralLoadingDecoder`
  in energy-proposer mode: `W` defines a spectral Ansatz for the initial state \(Q_0\).
- The Laplacian energy \(E(Q) = \frac{\lambda}{2}\mathrm{tr}(Q^\top L_f Q)\) uses the same
  \(L_f\) that provides the eigenbasis \(U_{1:q}\), coupling the energy and the prior geometry.

### Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{task}}(\hat{y}, y) + \mu\, E(Q_K) + \nu\, \|Q_0 - Q_K\|_F^2$$

---

## Option 3 — Vibrational Latent Diffusion Model

### Base Architecture

Stage 1: vibrational AE (Option 1) producing \(z \in \mathbb{R}^m\).
Stage 2: denoiser \(\epsilon_\theta(z_\tau, \tau)\) over modal latent space.

### VDT Extension

- **Stage 1 prerequisite** is the three-term `WiringAutoencoder`
  (reconstruction + spectral-basis KL + tau-mode KL; no sample Laplacian).
  The modal latent \(z\) is shaped by the spectral loading prior and tau-mode prior.
- **Spectral noise schedule** uses the modal prior covariance directly:

$$p(z_\tau \mid z_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_\tau}\, z_0,\; (1-\bar\alpha_\tau)\,\Lambda_m^{-1}\right)$$

  High-frequency modes are corrupted earlier (smaller \(\bar\alpha_\tau\) for large \(\lambda_k\));
  low-frequency modes persist longer. \(\Lambda_m\) comes from the frozen eigendecomposition
  of \(L(\mathcal{I})\) — no runtime Laplacian computation required.
- The tau-mode distribution provides the **noise schedule**: set
  \(\bar\alpha_\tau^{(k)} = \exp(-\tau \lambda_k \tau_{\text{step}})\) per mode.

---

## Option 4 — Vibrational Bayesian Autoencoder (Variational Laplace)

### Base Architecture

Posterior mode via VDT energy minimisation; Laplace covariance from preconditioned Hessian.

### VDT Extension

Option 4 is the closest base option to the full VDT ELBO. The two representations
are related as follows:

| Option 4 (Laplace) | VDT (full MC ELBO) |
|---|---|
| Posterior mode `z_hat = argmin` | Posterior mean `mu_z` from encoder |
| Laplace covariance `(H + I)^-1` | Diagonal `Sigma_z` from encoder |
| Reconstruction Hessian `H_recon` | Isotropic latent KL term |
| Spectral-basis prior on `S_hat` | `spectral_basis_kl` |
| Tau-mode prior on `omega_hat` | `tau_mode_kl` |

When MC sampling is too expensive, use Option 4 as a deterministic approximation
to the full VDT ELBO. Both the spectral-basis KL and tau-mode KL are directly
applicable in the Laplace setting by plugging `S_hat` and `omega_hat` into the
KL formulas. All eigenvalues come from the frozen `Lambda_{1:q}`.

### Training Objective (Laplace ELBO)

$$\mathcal{L} = -\log p(X_0 \mid \hat{z}) + \tfrac{1}{2}\|\hat{z}\|^2 - \tfrac{1}{2}\log|H^{-1}|$$

---

## Option 5 — Vibrational Graph Forecasting (PDE-Inspired Predictor)

### Base Architecture

VDT as learned PDE solver; predicts future states of graph-structured systems.

### VDT Extension

- **Prior memory**: the spectral artefact \(S_{\mathcal{I}}\) from a pre-trained
  `WiringAutoencoder` can serve as a **boundary condition library** for the PDE
  predictor. At each step, the forcing term \(B_t\) is retrieved from \(S_{\mathcal{I}}\)
  via Hopfield retrieval on the current state \(Q_t\).
- The CFL condition and all stability diagnostics from `04-stability.md` are unchanged.

### Training Objective

$$\mathcal{L} = \|X^{(K)} - X^{(T)}\|_F^2 + \alpha\sum_{t=1}^K \mathrm{tr}(Q_t^\top L_f Q_t) + \beta\max(0, \Delta t^2 \lambda_{\max}(L_f) - 2)$$

---

## Option 6 — Spectrally Regularised Vibrational Classifier / Reasoner

### Base Architecture

VDT backbone + classification head + depth-supervised CE loss + spectral regularisation.

### VDT Extension

- **Spectral associative memory**: initialise the classification head's key matrix
  from \(S_{\mathcal{I}}\) extracted from a pre-trained VDT. This pre-populates the
  transformer's long-term memory with geometry-grounded spectral patterns.
- **Density matrix as artefact**: the signed density matrix \(\varrho_K\) at convergence
  is an alternative source for the spectral artefact. Set
  \(S_{\mathcal{I}} = \varrho_K^+ - \varrho_K^-\) after normalisation.
- **Evaluation ablation**: report Option 6 accuracy with and without spectral
  memory initialisation at recurrent depths \(K \in \{2, 4, 8, 16\}\).

### Training Objective (fully specified)

$$\mathcal{L} = \frac{1}{K}\sum_{t=1}^K \mathcal{L}_{\text{CE}}(\hat{y}_t, y) + \mu_1 \frac{1}{K}\sum_{t=1}^K \mathrm{tr}(Q_t^\top L_f Q_t) + \mu_2 \frac{1}{K}\sum_{t=1}^K \|\varrho_t\|_F^2$$

---

## All Six Tracks at a Glance

| Option | Probabilistic? | Objective | VDT new components | Complexity |
|---|---|---|---|---|
| 1 — Deterministic AE | No | Recon + spectral | `SpectralLoadingDecoder` (det. mode) | Low |
| 2 — Energy-based | No | Task + energy | Spectral Ansatz proposer | Medium |
| 3 — Latent diffusion | Yes (implicit) | Denoising score | Stage 1 AE + spectral noise schedule | High |
| 4 — Variational Laplace | Yes (Laplace) | Laplace ELBO | Spectral-basis + tau-mode KL (Laplace) | Medium |
| 5 — PDE forecasting | No | State prediction | Spectral artefact as boundary library | Low |
| 6 — Classifier/reasoner | No | Depth-supervised CE | Spectral memory init + ablation protocol | Low |

### Recommended Implementation Sequence

1. **Option 6** — closes the loop on Section 11 with spectral memory ablation. Minimal new code.
2. **Option 1** — adds reconstruction objective; validates `SpectralLoadingDecoder` deterministic mode.
3. **Option 4** — Laplace ELBO with spectral-basis + tau-mode KL. No MC sampling needed.
4. **Option 3** — full generative model; Stage 1 prerequisite is the three-term VDT ELBO.
5. **Option 2 / 5** — depending on classification (energy) or forecasting (PDE) application.
