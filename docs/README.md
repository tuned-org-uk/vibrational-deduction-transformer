# `docs/v2/` — Wiring Autoencoder v2: Spectral-PPCA Architecture

This directory contains the theoretical and architectural documentation for
**Wiring Autoencoder v2 (WAE v2)**, which upgrades the original WAE family with
three structural priors derived from **Spectral-PPCA** (probabilistic PCA in a
Laplacian eigenbasis). The result is a fully Bayesian generative model whose
latent space is shaped by the ArrowSpace index geometry, and whose post-training
**spectral artefact** initialises a transformer with pre-built associative memory.

All notation follows the VDT paper (Moriondo, 2026) and the ArrowSpace technical
report (Moriondo, 2026) unless stated otherwise.

---

## What is New in v2

| v1 component | v2 upgrade | File |
|---|---|---|
| Isotropic latent prior `N(0,I)` | Laplacian-precision prior `N(0,(I+βLs)⁻¹)` | `00-architecture.md` |
| Hard `J_freq` spectral penalty | Variational τ-mode KL over mode weights ω | `00-architecture.md` |
| Unconstrained MoE wiring decoder | Spectral-basis loading decoder `W = U_{1:q} diag(ω) S` | `00-architecture.md` |
| Single fixed ELBO | Four-term ELBO with three KL terms | `00-architecture.md` |
| No post-training export | Spectral artefact extraction + associative memory | `00-architecture.md` |
| Six option tracks (v1) | Six option tracks updated for v2 compatibility | `03-branching.md` |
| Stability hierarchy (v1) | Extended with two v2-specific diagnostics | `04-stability.md` |
| No code reference | Full module-level code for all v2 changes | `05-Code.md` |

---

## Conceptual Foundations

The modelling chain advances the v1 Spectral Laplacian analogue of PPCA into a
**fully Bayesian VAE** where the prior geometry is explicitly provided by the
ArrowSpace index `I`:

| Book concept | v1 WAE analogue | v2 WAE analogue |
|---|---|---|
| PCA | Spectral Laplacian `Lf = Df − Wf` | Same; now also the eigenbasis for `W` |
| Autoencoder | Wiring AE (`J_freq` loss) | Spectral-basis loading AE |
| PPCA | Probabilistic graph wiring (`p(z)=N(0,Λm⁻¹)`) | **Implemented**: Laplacian-precision KL replaces `N(0,I)` KL |
| VAE + ELBO | `recon + β·KL + α·J_freq` | `recon − KL_Lap(z) − KL_S − KL_τ` |
| Bayesian evidence | Not present in v1 | ELBO Bayes factor over ArrowSpace indices |
| Associative memory | Not present in v1 | Spectral artefact → pre-built Hopfield memory |
| Transformer memory | Random init | `SpectralAssociativeMemory` from artefact |

---

## v2 Concept Tree

```
                    ┌─────────────────────────────────────────┐
                    │   THE LITTLE BOOK FOUNDATIONS           │
                    │  PCA → Autoencoder → PPCA → VAE         │
                    └──────────────────┬──────────────────────┘
                                       │
                    SPECTRAL GRAPH WIRING ANALOGUE (v1)
                                       │
              ┌────────────────────────┴────────────────────────┐
              │  Graph Laplacian Lf                              │
              │  z⊤ Lf z = smoothness                           │
              └────────────────────────┬────────────────────────┘
                                       │
                    SPECTRAL-PPCA BAYESIAN UPGRADE (v2)
                                       │
         ┌─────────────────────────────┼──────────────────────────┐
         │                             │                          │
  ┌──────▼──────┐             ┌────────▼───────┐        ┌────────▼────────┐
  │  W = U_{1:q}│             │  p(z) =        │        │  p(ω|τ,Λ):      │
  │  diag(ω) S  │             │  N(0,(I+βLs)⁻¹)│        │  Exp(τλk) prior │
  │  eigenbasis │             │  Dirichlet KL  │        │  τ-mode KL      │
  │  loading    │             └────────┬───────┘        └────────┬────────┘
  └──────┬──────┘                      │                         │
         └──────────────────┬──────────┘─────────────────────────┘
                            │
               ┌────────────▼────────────────┐
               │   WAE v2 CORE               │
               │                             │
               │  ELBO =                     │
               │    recon                    │
               │  − KL_Lap(z)               │
               │  − KL_S (spectral basis)    │
               │  − KL_τ (mode weights)      │
               └────────────┬────────────────┘
                            │
               ┌────────────▼────────────────┐
               │   SPECTRAL ARTEFACT A(I)    │
               │  Ŵ, {ω̂k}, S_memory         │
               └────────────┬────────────────┘
                            │
               ┌────────────▼────────────────┐
               │  SpectralAssociativeMemory  │
               │  Initialises transformer    │
               │  FFN / cross-attn values    │
               │  Delta-rule online updates  │
               └─────────────────────────────┘
```

Here is a complete, structured summary of the v2 docs and open issues in [tuned-org-uk/wiring-autoencoder](https://github.com/tuned-org-uk/wiring-autoencoder).

***

## What is WAE v2?

The **Wiring Autoencoder v2 (WAE v2)** is a fully Bayesian generative model that upgrades WAE v1 with **Spectral-PPCA** — probabilistic PCA performed in the Laplacian eigenbasis of an ArrowSpace index graph.  Its latent space is shaped by graph spectral geometry, and its post-training output is a **spectral artefact** that seeds a transformer with pre-built associative memory. 

***

## Conceptual Progression

The architecture traces the classical generative modelling ladder, replacing each step with a graph-spectral analogue:

| Book concept | v1 WAE | v2 WAE |
|---|---|---|
| PCA | Spectral Laplacian `Lf` | Same + eigenbasis for `W` |
| Autoencoder | Wiring AE (`J_freq` loss) | `SpectralLoadingDecoder` |
| PPCA | Modal prior `N(0, Lm^-1)` | Implemented in KL terms |
| VAE + ELBO | `recon + beta*KL + alpha*J_freq` | Three-term spectral ELBO |
| Associative memory | Absent | `SpectralAssociativeMemory` |



***

## The Three-Term ELBO

The v2 training objective replaces v1's hard `J_freq` spectral penalty with three principled variational terms: 

```
L_WAEv2 = E_q[log p(x|z,W)]
         - KL( q(z)  || N(0,I)       )    # standard isotropic latent KL
         - KL( q(S)  || p(S|I)       )    # spectral-basis KL
         - KL( q(w)  || Exp(tau*lk)  )    # tau-mode frequency KL
```

- **Spectral-basis KL**: penalises the loading matrix `S` using eigenvalue-weighted Gaussian shrinkage — high-frequency eigenmodes (large `lk`) are penalised exponentially more, implementing a *spectral Occam's razor*. 
- **Tau-mode KL**: replaces the hard penalty by placing an Exponential prior over mode weights `w_k`, with heavy support on low-frequency modes and a closed-form Gamma/Exponential KL. 

***

## Key Modules

### `WiringEncoderV2` (`encoder.py`)
Wraps the `VDT` vibrational recurrence (discrete damped wave equation) and adds a `ModeWeightHead` — a linear layer outputting variational Gamma parameters `(log_a, log_b)` for each spectral mode.  It uses **standard isotropic KL** for the latent `z` (the Laplacian-precision KL path was removed in PR #35). 

### `SpectralLoadingDecoder` (`wiring_decoder.py`)
Replaces `WiringDecoder` as the v2 default. Maps `z (B, q)` to a loading matrix `W = U_q @ diag(w) @ S` in the Laplacian eigenbasis.  Edge weights are synthesised via `DifferentiableLaplacian.from_spectral_loading(W, L_base)`, fully differentiable back to `z`. 

### `WiringAutoencoderV2` (`model.py`)
Top-level assembly of encoder, spectral decoder, and diffusion decoder. Its `forward()` returns a 9-key dict: `{loss, recon, kl_z, kl_S, kl_tau, x_hat, z, mu, log_var}`.  After training, `extract_spectral_artefact()` packages the mean loading matrix `W_hat`, posterior mode weights `omega_hat`, and the associative memory matrix `S_I`. 

### `SpectralAssociativeMemory` (`spectral_memory.py`)
Wraps the spectral artefact into a pre-built Hopfield/outer-product memory: 

```
S_I = sum_k  E[w_k] * d_theta(w_hat_k) * w_hat_k^T
```

Keys `w_hat_k` are Laplacian eigenvector-aligned loading directions (approximately orthonormal for high retrieval SNR). Values `d_theta(w_hat_k)` are decoder responses per frequency band. Supports online delta-rule updates. 

***

## Two-Phase Architecture

The system separates offline spectral learning from online inference: 

- **Phase 1 (Offline):** One-time eigendecomposition of `L(I)` → train `WiringAutoencoderV2` via ELBO → extract artefact `A(I)` → build `S_I`. Frozen eigenpairs `(U_q, Lq)` are constants at runtime; no Laplacian is built during training.
- **Phase 2 (Online):** Transformer FFN / cross-attention initialised from `S_I`. Self-attention handles dynamic short-term associations; `S_I` supplies long-term spectral prior memory; delta-rule writes new associations online.

***

## Six Algorithm Tracks

The docs define six branching implementation options: 

- **Option 1** — Deterministic AE with `SpectralLoadingDecoder` (ablation: `J_freq` hard vs tau-mode soft penalty)
- **Option 3** — Latent diffusion with per-mode spectral noise schedule (`a_tau^(k) = exp(-tau * lk * tau_step)`)
- **Option 4** — Variational Laplace AE (no MC sampling; Laplace posterior + spectral-basis/tau-mode KLs)
- **Option 6** — Vibrational classifier/reasoner with `SpectralAssociativeMemory` key-matrix init and depth-supervised CE loss

Recommended implementation sequence: **6 → 1 → 4 → 3**. 

***

## Open Issues (19 total)

All issues are part of the v2 roadmap tracked under (https://github.com/tuned-org-uk/wiring-autoencoder/issues/34), organised into five phases:

| Phase | Issues | Focus |
|---|---|---|
| **0 — Foundations** | #16, #17, #19, #24 | `laplacian.py`, `vdt.py`, `stability.py`, two KL functions in `spectral.py` |
| **1 — Encoder/Decoder** | #25, #26 | `WiringEncoderV2` (isotropic KL + `ModeWeightHead`), `SpectralLoadingDecoder` |
| **2 — Model assembly** | #27 | `WiringAutoencoderV2`, three-term ELBO, `extract_spectral_artefact()` |
| **3 — Memory & Metrics** | #28, #32 | `SpectralAssociativeMemory`, 7-metric evaluation suite |
| **4 — App Tracks** | #18/#29, #20/#30, #21/#31, #33 | Options 6, 1, 4, 3 respectively |
| **5 — Benchmarks/Demo** | #9, #13 | Multi-seed results on Cora/PubMed, updated generation demo |

Key architectural decisions locked in by **PR #35**: the ELBO is three-term (Laplacian-precision latent KL removed); `kl_z` uses isotropic `N(0,I)`; `L_z` key dropped from `forward()` return dict; no runtime Laplacian construction during training. 

***

## Benchmark Metrics (v2)

Seven active metrics are tracked by `evaluate_v2()`: 

- `kl_S` — spectral basis KL
- `kl_tau` — tau-mode frequency KL
- `active_modes` — count of modes with `E[w_k] > 0.01`
- `memory_snr` — retrieval SNR via key orthogonality
- `elbo_bayes_factor` — `exp(L(I1) - L(I2))` for ArrowSpace index comparison
- `linear_probe_acc` — logistic regression on frozen `mu`
- `spectral_entropy H(L)` — diversity of generated wirings

---

## Document Map

| File | Content |
|---|---|
| `README.md` (this file) | Overview, concept tree, document map |
| `00-architecture.md` | Full v2 architecture reference: modules, ELBO, data flow |
| `01-references.md` | Bibliography and related work (updated for v2) |
| `03-branching.md` | Six algorithm tracks updated for v2 compatibility |
| `04-stability.md` | Stability hierarchy extended with two v2 diagnostics |
| `05-Code.md` | Complete module-level code for all v2 changes |

---

## Recommended Implementation Sequence

1. **Swap `kl_loss`** in `WiringEncoder` to use the modal prior `N(0, Λm⁻¹)` —
   one-line change, immediately makes the latent prior match the concept table.

2. **Replace `J_freq` hard penalty** with `tau_mode_kl` — soft variational KL
   over mode weights. Keep `α·J_freq` as ablation flag in config.

3. **Introduce `SpectralLoadingDecoder`** as a config-controlled drop-in for
   `WiringDecoder`. Validate reconstruction parity before making it default.

4. **Add sample-graph Laplacian KL** in the encoder forward pass (stop-gradient
   on `Ls` construction). Monitor latent smoothness KL convergence per epoch.

5. **Add `extract_spectral_artefact()`** and `SpectralAssociativeMemory`.
   Test retrieval SNR on a toy associative recall benchmark.

6. **Integrate `SpectralAssociativeMemory`** into VDT / transformer as FFN
   initialiser. Run Option 6 evaluation protocol with memory enabled vs disabled.

---

## Relationship to the VDT Paper

All six tracks remain grounded in the VDT paper backbone:

- **Part I (Foundations)**: `Lf`, `M`, `R_M`, preconditioned GD — underpin Options 1, 2, 4, 6.
  In v2, `Λm` eigenvalues now also parametrise the latent prior and spectral-basis KL.
- **Part II (Architecture)**: `Φ_L` wave update and `ϱt` density matrix — encoder backbone
  for all six options. In v2, `ϱt` is the source of a reasoning-grounded associative prior
  for Option 6.
- **Section 9 (Density Matrix)**: `ϱt = ϱt⁺ − ϱt⁻` is the starting point for the probabilistic
  reinterpretation in Options 3 and 4, now unified under the Spectral-PPCA ELBO.
- **Section 11 (Experiments)**: LDT-mirrored benchmarks now include a memory-enabled
  vs memory-disabled ablation for the associative memory component.
