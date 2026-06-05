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
