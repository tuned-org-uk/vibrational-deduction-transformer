# Wiring Autoencoder (WAE)

A **Spectral-PPCA Variational Autoencoder** whose generative path is mediated by a
learned graph wiring (Laplacian) constrained to the eigenbasis of an ArrowSpace index `I`.
Post-training, the model emits a **spectral artefact** that initialises a transformer
with pre-built associative memory.

This architecture is related to NVIB (Nonparametric Variational Information Bottleneck). It applies pre-built semantic spectral filters and inline memory to VAE to save learning steps by identifying the object of learning via spectral methods.

The feature-space Laplacian $$L_f$$ (as in Graph Wiring) replaces the basis of the latent space and the prior over mode weights, leaving reparameterization itself still diagonal and cheap .

The architecture follows the progression in *The Little Book of Generative AI Foundations*
(Chen, 2026) and is grounded in the VDT paper (Moriondo, 2026):

```
PCA → Autoencoder → PPCA → VAE
 ↕         ↕          ↕      ↕
Graph     Wiring    Prob.  Spectral-PPCA
Laplacian   AE     Wiring    WAE (this repo)
```

---

## Core Idea

The encoder produces a posterior `q(z|x)` enriched by a λ-fingerprint from `L(I)`.
The decoder maps `z` into a **spectral loading matrix**:

```
W = U_{1:q} diag(ω) S
```

where `U_{1:q}` are the `q` lowest-frequency eigenvectors of the ArrowSpace Laplacian
`L(I)`, `S` are learnable loadings in that eigenbasis, and `ω` are mode weights drawn
from a τ-mode prior. `W` then parametrises a differentiable Laplacian `L(z)`, over which
a tau-mode diffusion reconstructs `x̂` from the embedding table `E`.

---

## The Three-Term ELBO

```
ℒ_WAE = E_q[log p(x | z, W)]
      − KL( q(z)  ∥  N(0, I)         )   [isotropic latent KL]
      − KL( q(S)  ∥  p(S | I)        )   [spectral-basis KL — eigenvalue-weighted]
      − KL( q(ω)  ∥  p(ω | τ, Λ)    )   [τ-mode frequency KL — Gamma vs Exp(τλk)]
```

The ArrowSpace index `I` enters solely through the pre-computed frozen eigenpair
`(U_{1:q}, Λ_{1:q})` of `L(I)` — no Laplacian is evaluated or inverted at training time.
Index selection is Bayesian via the ELBO Bayes factor `exp(ℒ(I₁) − ℒ(I₂))`.

---

## Data Flow

```
Input x (B, D)                   Embedding table E (N, D)
    │                                    │
    ├── [λ-fingerprint from L(I)] ───────┤
    ▼
┌─────────────────────────────────┐
│  WiringEncoderV2                │
│  MLP + λ-fingerprint            │
│  → (z, μ, logσ, log_a, log_b)  │
└─────────────────────────────────┘
    │
  (z, μ, logσ)  ← reparameterise
  (log_a, log_b) → ModeWeightHead → q(ω)
    │
    ▼
┌──────────────────────────────────┐
│ SpectralLoadingDecoder           │
│  z, U_{1:q}  →  W, ω, S         │
│  W = U_{1:q} diag(ω) S          │
└──────────────────────────────────┘
    │
  W  →  DifferentiableLaplacian.from_spectral_loading(W, L_base)
    │
  L(z)  (B, N, N)
    │
    ▼
┌────────────────────┐      ┌────────────┐
│  DiffusionDecoder  │ ←─── │     E      │
│  TauModeDiffusion  │      └────────────┘
└────────────────────┘
    │
  x̂  (B, D)  ──►  WAE ELBO loss (recon + kl_z + kl_S + kl_tau)
```

Post-training, `extract_spectral_artefact()` builds:

```
A(I) = { Ŵ,  {ω̂_k},  S_memory }
```

`S_memory` is a pre-built outer-product Hopfield matrix keyed on Laplacian eigenvectors
(orthonormal by construction, maximising retrieval SNR) that initialises the transformer's
feed-forward / cross-attention value matrices.

---

## Architecture Modules

| Module | Role |
|--------|------|
| `wae/encoder.py` | `WiringEncoderV2` — amortised posterior with λ-fingerprint; `ModeWeightHead` outputs `(log_a, log_b)` for τ-mode prior |
| `wae/wiring_decoder.py` | `SpectralLoadingDecoder` — `z, U_q → W = U_q diag(ω) S → L(z)` |
| `wae/diffusion_decoder.py` | `L(z), E → x̂` via tau-mode diffusion + MLP refinement |
| `wae/model.py` | `WiringAutoencoderV2` — three-term ELBO + `extract_spectral_artefact()` |
| `wae/laplacian.py` | Differentiable Laplacian builder; `from_spectral_loading(W, L_base)` |
| `wae/spectral.py` | `spectral_basis_kl`, `tau_mode_kl`, `laplacian_precision_kl`, `build_knn_laplacian` |
| `wae/spectral_memory.py` | `SpectralAssociativeMemory` — Hopfield memory pre-built from `A(I)`; delta-rule online updates |
| `wae/stability.py` | Training diagnostics; `spectral_kl_health_check` (6-level hierarchy) |
| `wae/dataset.py` | Dataset helpers (MNIST, CORA, PubMed, custom CSV) |
| `train.py` | Training loop with W&B / CSV logging |
| `benchmark.py` | Evaluation suite — 8 metrics + ELBO Bayes factor |
| `configs/default.yaml` | Hyperparameters |

---

## Quickstart

```bash
pip install -e ".[dev]"
python train.py --config configs/default.yaml --dataset cora
python benchmark.py --dataset cora --output results/
```

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Reconstruction MSE | Quality of x̂ recovered through the wiring path |
| `kl_z` | Standard isotropic KL regularisation of latent `z` |
| `kl_S` | Spectral alignment of loadings with ArrowSpace index `I` |
| `kl_tau` | Effective frequency band selection via τ-mode prior |
| `active_modes` | Number of modes with `E[ωk] > 0.01` contributing to `W` |
| `memory_snr` | Retrieval quality of `SpectralAssociativeMemory` (key orthogonality) |
| `elbo_bayes_factor` | `exp(ℒ(I₁) − ℒ(I₂))` — comparison of competing ArrowSpace indices |
| `linear_probe_acc` | Discriminative quality of frozen latent `μ` |

---

## Flagship Demo — Spectral Graph Generation

Standard VAEs decode `z` into flat feature vectors. The WAE decodes `z` into a
*graph wiring* — a Laplacian — whose eigenvalues are vibrational modes of the system
(cf. Rayleigh's *Theory of Sound*). The latent space directly encodes spectral geometry,
enabling **entropy-controlled generation**: sample novel wirings whose Laplacian spectrum
matches a target entropy level.

```bash
# Train on synthetic spring-network graphs and run all evaluations
python demos/spectral_generation_demo.py --n-graphs 400 --epochs 60

# Interactive pluot + static visualisations
python demos/visualise_spectral_demo.py --results results/spectral_demo
```

Outputs written to `results/spectral_demo/`:

| File | Content |
|------|---------|
| `spectral_demo_results.csv` | Per-sample spectral entropy + Frobenius distance to nearest training Laplacian |
| `entropy_control_results.csv` | Entropy-targeting experiment: target vs best error vs match rate |
| `training_log.csv` | Epoch-level ELBO, reconstruction MSE, KL terms |
| `figures/training_curves.png` | Loss component curves |
| `figures/entropy_distribution.png` | Dataset vs generated spectral entropy histogram |
| `figures/spectral_distance.png` | Distribution of nearest-neighbour spectral distances |
| `figures/latent_entropy.png` | PCA-2D latent space coloured by spectral entropy |
| `figures/mode_shapes.png` | First 4 vibrational mode shapes of a sample spring network |
| `figures/entropy_target_error.png` | Entropy targeting precision across entropy range |
| `figures/pluot_manifest.json` | Load in [pluot](https://github.com/keller-mark/pluot) for interactive view |

---

## Connection to ArrowSpace

`wae/laplacian.py` mirrors `ArrowSpaceBuilder.build()` logic from
[pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace) as a differentiable
PyTorch layer so gradients flow through `L(z)`.

The ArrowSpace index `I` determines the frozen eigenpair `(U_{1:q}, Λ_{1:q})` that
parametrises both the loading-matrix prior and the τ-mode frequency prior.
Index selection is made Bayesian via the ELBO Bayes factor.

---

## Documentation

| File | Content |
|------|---------|
| [`docs/v2/README.md`](docs/v2/README.md) | Concept tree, document map, implementation sequence |
| [`docs/v2/00-architecture.md`](docs/v2/00-architecture.md) | Full architecture reference: modules, ELBO, data flow |
| [`docs/v2/01-references.md`](docs/v2/01-references.md) | Bibliography and related work |
| [`docs/v2/03-branching.md`](docs/v2/03-branching.md) | Six algorithm tracks and option compatibility |
| [`docs/v2/04-stability.md`](docs/v2/04-stability.md) | Stability hierarchy and diagnostics |

---

## References

- *The Little Book of Generative AI Foundations*, T. Chen, 2026
- VDT paper (Moriondo, 2026) — ArrowSpace / Graph Wiring
- ArrowSpace technical report (Moriondo, 2026) — see `docs/v2/01-references.md`
- Rayleigh, *Theory of Sound*, vol. 1 — vibrational mode decomposition
