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
