# Wiring Autoencoder (WAE)

This architecture is related to NVIB (Nonparametric Variational Information Bottleneck). It applies pre-built semantic
 spectral filters and inline memory to VAE to save learning steps by identifying the object of learning via spectral methods.

A **Variational Autoencoder whose generative path is mediated by a graph wiring**
(i.e. a learned Laplacian) rather than a plain linear map.  
The progression mirrors the book *The Little Book of Generative AI Foundations*:

```
PCA → PPCA → VAE
 ↕      ↕      ↕
Graph Wiring → Probabilistic Graph Wiring → WAE
```

---

## v1 (stable) vs v2 (in development)

| | v1 WAE | v2 WAE (Spectral-PPCA) |
|---|---|---|
| Decoder | `WiringDecoder` — edge delta logits | `SpectralLoadingDecoder` — `W = U_{1:q} diag(ω) S` |
| ELBO | `recon + β·KL(z) + α·J_freq` | `recon + KL(z) + kl_S + kl_tau` (three-term) |
| Spectral prior | Hard `J_freq` penalty | Variational Gamma prior `KL(q(ω) ∥ Exp(τλk))` |
| Index `I` | Not Bayesian | Enters via frozen `(U_{1:q}, Λ_{1:q})` eigenpair of `L(I)` |
| Memory | None | `SpectralAssociativeMemory` — outer-product Hopfield, pre-built from spectral artefact |
| Index selection | — | ELBO Bayes factor `exp(ℒ(I1) − ℒ(I2))` |

See [`docs/v2/README.md`](docs/v2/README.md) and [`docs/v2/00-architecture.md`](docs/v2/00-architecture.md) for the full v2 spec.

---

## Core Idea

Instead of mapping a latent vector `z` directly to a reconstruction via `Wz + ε`,
the decoder:

1. Maps `z` to **wiring parameters** (edge-weight logits for a base kNN graph).
2. Builds a normalised Laplacian `L(z)` from those parameters.
3. Runs a **tau-mode diffusion** over the data embedding table using `L(z)`.
4. Predicts `x̂` from the diffused neighbourhood, with Gaussian likelihood.

### v1 ELBO

```
ℒ(θ,φ; x) = E_{q_φ(z|x)}[log p_θ(x|z)]  − β·KL(q_φ(z|x) ‖ p(z))
           − α·J_freq(L(z))               (spectral regulariser)
```

`J_freq` penalises high-frequency energy on the wiring, encouraging smooth,
interpretable spectral structures — the direct analogue of tau-mode truncation
in ArrowSpace.

### v2 Three-Term ELBO (Spectral-PPCA)

```
ℒ_WAEv2 = E_q[log p(x|z,W)]
         − KL( q(z)  ‖  N(0,I)  )      [standard isotropic latent KL]
         − KL( q(S)  ‖  p(S|I)  )      [spectral-basis KL — eigenvalue-weighted]
         − KL( q(ω)  ‖  p(ω|τ,Λ))     [τ-mode frequency KL — Gamma vs Exp]
```

The ArrowSpace index `I` enters solely through the pre-computed frozen eigenpair
`(U_{1:q}, Λ_{1:q})` of `L(I)` — no Laplacian is evaluated or inverted at training time.

---

## Architecture Modules

| Module | Role | Version |
|--------|------|---------|
| `wae/encoder.py` | Amortised posterior `q_φ(z|x)` — optional λ-fingerprint enrichment | v1 |
| `wae/encoder.py` | `WiringEncoderV2` — adds `ModeWeightHead` outputting `(log_a, log_b)` for τ-mode prior; isotropic KL for `z` | v2 |
| `wae/wiring_decoder.py` | `WiringDecoder` — `z → wiring logits → L(z)` | v1 |
| `wae/wiring_decoder.py` | `SpectralLoadingDecoder` — `z, U_q → W = U_q diag(ω) S → L(z)` | v2 |
| `wae/diffusion_decoder.py` | `L(z), E → x̂` via tau-mode diffusion | v1 + v2 |
| `wae/model.py` | `WiringAutoencoder` — v1 ELBO + spectral regulariser | v1 |
| `wae/model.py` | `WiringAutoencoderV2` — three-term ELBO + `extract_spectral_artefact()` | v2 |
| `wae/laplacian.py` | Differentiable Laplacian builder; v2 adds `from_spectral_loading(W, L_base)` | v1 + v2 |
| `wae/spectral.py` | `J_freq` cost, tau-mode diffusion, λ-fingerprint; v2 adds `spectral_basis_kl`, `tau_mode_kl` | v1 + v2 |
| `wae/spectral_memory.py` | `SpectralAssociativeMemory` — Hopfield memory pre-built from spectral artefact `A(I)` | v2 |
| `wae/stability.py` | Training diagnostics + `spectral_kl_health_check` (6-level hierarchy) | v1 + v2 |
| `wae/dataset.py` | Dataset helpers (MNIST, CORA, PubMed, custom CSV) | v1 + v2 |
| `train.py` | Training loop with W&B / CSV logging | v1 + v2 |
| `benchmark.py` | Comparative benchmarks vs. plain VAE and linear AE | v1 + v2 |
| `configs/default.yaml` | Hyperparameters | v1 + v2 |

---

## Quickstart

```bash
pip install -e ".[dev]"
python train.py --config configs/default.yaml --dataset cora
python benchmark.py --dataset cora --output results/
```

For v2:

```bash
python train.py --config configs/default.yaml --dataset cora --model_version 2
python benchmark.py --dataset cora --output results/ --model_version 2
```

---

## Flagship Demo — Molecular / Spectral Graph Generation

> **Why this is a distinctive WAE application:** Standard VAEs decode `z` into
> flat feature vectors.  The WAE decodes `z` into a *graph wiring* — a
> Laplacian — whose eigenvalues are the vibrational modes of the system
> (cf. Rayleigh's Theory of Sound).  This means the latent space directly
> encodes spectral geometry, enabling **entropy-controlled generation**:
> sample novel wirings whose Laplacian spectrum matches a target entropy level.

```bash
# Train on synthetic spring-network graphs and run all evaluations
python demos/spectral_generation_demo.py --n-graphs 400 --epochs 60

# Interactive pluot + static visualisations (reads the CSVs produced above)
python demos/visualise_spectral_demo.py --results results/spectral_demo
```

Outputs written to `results/spectral_demo/`:

| File | Content |
|---|---|
| `spectral_demo_results.csv` | Per-generated-sample spectral entropy + Frobenius distance to nearest training Laplacian |
| `entropy_control_results.csv` | Entropy-targeting experiment: target vs best error vs match rate |
| `training_log.csv` | Epoch-level ELBO, reconstruction MSE, KL, J_freq |
| `figures/training_curves.png` | Loss component curves |
| `figures/entropy_distribution.png` | Dataset vs generated spectral entropy histogram |
| `figures/spectral_distance.png` | Distribution of nearest-neighbour spectral distances |
| `figures/latent_entropy.png` | PCA-2D latent space coloured by spectral entropy |
| `figures/mode_shapes.png` | First 4 vibrational mode shapes of a sample spring network |
| `figures/entropy_target_error.png` | Entropy targeting precision across entropy range |
| `figures/pluot_manifest.json` | Load in [pluot](https://github.com/keller-mark/pluot) for interactive view |

### Quantitative Evaluation

#### v1 metrics

| Metric | What it measures |
|---|---|
| Reconstruction MSE | Quality of x̂ recovered through wiring path |
| KL divergence | Regularisation of latent z |
| J_freq (spectral cost) | Smoothness of learned wiring (low = low-frequency, ordered) |
| Spectral entropy H(Λ) | Shannon entropy of normalised eigenvalue spectrum of L(z) |
| Frobenius distance to NN | Novelty: how far are generated Laplacians from training set? |
| Entropy match rate | Fraction of samples within tol=0.05 of a target entropy |

#### v2 additional metrics

| Metric | What it measures |
|---|---|
| `kl_S` | Spectral alignment of loadings with ArrowSpace index `I` |
| `kl_tau` | Effective frequency band selection via τ-mode prior |
| `active_modes` | Number of modes with `E[ωk] > 0.01` contributing to `W` |
| `memory_snr` | Retrieval quality of `SpectralAssociativeMemory` (key orthogonality) |
| `elbo_bayes_factor` | `exp(ℒ(I1) − ℒ(I2))` — comparison of competing ArrowSpace indices |
| `linear_probe_acc` | Discriminative quality of frozen latent `mu` |
| Spectral entropy H(Λ) | As v1 |

---

## Benchmarks (CORA, 7-class node classification)

Run `benchmark.py` to reproduce. Reported metrics: reconstruction MSE,
latent KL, downstream accuracy (linear probe), and spectral entropy of `L(z)`.

---

## Connection to ArrowSpace

`wae/laplacian.py` mirrors `ArrowSpaceBuilder.build()` logic from
[pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace), but implemented
as a **differentiable PyTorch layer** so gradients flow through `L(z)`.

In v2, the ArrowSpace index `I` additionally determines the frozen eigenpair
`(U_{1:q}, Λ_{1:q})` that parametrises both the loading matrix prior and the
τ-mode frequency prior. Index selection is made Bayesian via the ELBO Bayes factor.

---

## References

- *The Little Book of Generative AI Foundations*, T. Chen, 2026
- ArrowSpace / Graph Wiring papers — see `docs/references.md`
- Scott Aaronson, *Quantum Computing Since Democritus*, ch. 9
- Rayleigh, *Theory of Sound*, vol. 1 — vibrational mode decomposition
