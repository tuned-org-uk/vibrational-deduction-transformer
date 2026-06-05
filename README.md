# Wiring Autoencoder (WAE)

A **Variational Autoencoder whose generative path is mediated by a graph wiring**
(i.e. a learned Laplacian) rather than a plain linear map.  
The progression mirrors the book *The Little Book of Generative AI Foundations*:

```
PCA → PPCA → VAE
 ↕      ↕      ↕
Graph Wiring → Probabilistic Graph Wiring → WAE
```

## Core Idea

Instead of mapping a latent vector `z` directly to a reconstruction via `Wz + ε`,
the decoder:

1. Maps `z` to **wiring parameters** (edge-weight logits for a base kNN graph).
2. Builds a normalised Laplacian `L(z)` from those parameters.
3. Runs a **tau-mode diffusion** over the data embedding table using `L(z)`.
4. Predicts `x̂` from the diffused neighbourhood, with Gaussian likelihood.

The ELBO is:

```
ℒ(θ,φ; x) = E_{q_φ(z|x)}[log p_θ(x|z)]  − β·KL(q_φ(z|x) ‖ p(z))
           − α·J_freq(L(z))               (spectral regulariser)
```

`J_freq` penalises high-frequency energy on the wiring, encouraging smooth,
interpretable spectral structures — the direct analogue of tau-mode truncation
in ArrowSpace.

## Architecture Modules

| Module | Role |
|--------|------|
| `wae/encoder.py` | Amortised posterior `q_φ(z|x)` — optional λ-fingerprint enrichment |
| `wae/wiring_decoder.py` | `z → wiring logits → Laplacian L(z)` |
| `wae/diffusion_decoder.py` | `L(z), E → x̂` via tau-mode diffusion |
| `wae/model.py` | Full WAE: ELBO + spectral regulariser, reparameterisation trick |
| `wae/laplacian.py` | Differentiable Laplacian builder from soft edge weights |
| `wae/spectral.py` | `J_freq` cost, tau-mode truncated diffusion, λ-fingerprint |
| `wae/dataset.py` | Dataset helpers (MNIST, CORA, PubMed, custom CSV) |
| `train.py` | Training loop with W&B / CSV logging |
| `benchmark.py` | Comparative benchmarks vs. plain VAE and linear AE |
| `configs/default.yaml` | Hyperparameters |

## Quickstart

```bash
pip install -e ".[dev]"
python train.py --config configs/default.yaml --dataset cora
python benchmark.py --dataset cora --output results/
```

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

| Metric | What it measures |
|---|---|
| Reconstruction MSE | Quality of x̂ recovered through wiring path |
| KL divergence | Regularisation of latent z |
| J_freq (spectral cost) | Smoothness of learned wiring (low = low-frequency, ordered) |
| Spectral entropy H(Λ) | Shannon entropy of normalised eigenvalue spectrum of L(z) |
| Frobenius distance to NN | Novelty: how far are generated Laplacians from training set? |
| Entropy match rate | Fraction of samples within tol=0.05 of a target entropy |

## Benchmarks (CORA, 7-class node classification)

Run `benchmark.py` to reproduce. Reported metrics: reconstruction MSE,
latent KL, downstream accuracy (linear probe), and spectral entropy of `L(z)`.

## Connection to ArrowSpace

`wae/laplacian.py` mirrors `ArrowSpaceBuilder.build()` logic from
[pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace), but implemented
as a **differentiable PyTorch layer** so gradients flow through `L(z)`.

## References

- *The Little Book of Generative AI Foundations*, T. Chen, 2026
- ArrowSpace / Graph Wiring papers — see `docs/references.md`
- Scott Aaronson, *Quantum Computing Since Democritus*, ch. 9
- Rayleigh, *Theory of Sound*, vol. 1 — vibrational mode decomposition
