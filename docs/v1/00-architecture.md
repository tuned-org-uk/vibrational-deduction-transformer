# Wiring Autoencoder — Architecture Reference

## Conceptual Progression

The WAE follows the same step-by-step progression as the book
*The Little Book of Generative AI Foundations* (Chen, 2026),
but with graph wiring as the primitive object rather than linear maps:

```
Book progression:             WAE analogue:

PCA                    →     Spectral Laplacian (DifferentiableLaplacian)
  ↓                              ↓
Autoencoder            →     Wiring Autoencoder (deterministic, J_freq loss)
  ↓                              ↓
PPCA                   →     Probabilistic Graph Wiring (Gaussian z → L(z))
  ↓                              ↓
VAE + ELBO             →     WAE-ELBO = recon + β·KL + α·J_freq
  ↓                              ↓
Diffusion / Flows      →     (future) WAE-Diffusion: diffuse over wiring space
```

---

## Data Flow Diagram

```
 Input x (B, D)                   Embedding table E (N, D)
     │                                    │
     ├── [λ-fingerprint from base L] ─────┤
     │                                    │
     ▼                                    │
 ┌──────────────┐                          │
 │  WiringEncoder  │                          │
 │  (MLP + λ feat) │                          │
 └──────────────┘                          │
     │                                        │
   (z, μ, logσ)  ←── reparameterise ───────┘
     │
     ▼
 ┌──────────────┐
 │ WiringDecoder  │  z → edge deltas → DifferentiableLaplacian
 │ (MLP + n_heads) │  mixture-of-experts over edge templates
 └──────────────┘
     │
   L(z)  (B, N, N)   ←── differentiable, gradients flow back to z
     │
     ▼
 ┌────────────────────┐      ┌────────────┐
 │  DiffusionDecoder  │ ←─── │     E     │
 │  TauModeDiffusion  │      └────────────┘
 │  + MLP refinement  │
 └────────────────────┘
     │
   x̂  (B, D)    ───────────────────►  ELBO Loss
                                            │
                       recon  kl  j_freq ──┘
```

---

## Module Reference

### `wae/laplacian.py` — `DifferentiableLaplacian`

Mirrors `ArrowSpaceBuilder.build()` from pyarrowspace but as a PyTorch layer.

- Fixed topology: kNN edge index + base RBF weights (frozen buffers).
- Learnable: per-edge delta logits from `WiringDecoder`.
- Edge weight: `w_ij = base_w_ij * sigmoid(delta_ij)`.
- Outputs symmetric normalised Laplacian `L = I - D^{-1/2} A D^{-1/2}`.
- Fully differentiable: gradients from reconstruction loss flow through `L(z)` back to `z`.

### `wae/spectral.py` — `TauModeDiffusion`, `spectral_freq_cost`, `lambda_fingerprint`

**TauModeDiffusion:**
- Truncates to `k` lowest eigenvectors of `L(z)` (tau-mode approximation from ArrowSpace cost paper).
- Heat kernel: `K_tau = U · exp(-tΛ) · Uᵀ`.
- Learnable diffusion time `t`.
- Uses `torch.linalg.eigh` (differentiable, real eigenvalues guaranteed for symmetric `L`).

**spectral_freq_cost** (`J_freq`):
- Penalty on eigenvalues beyond the first `tau_modes`: `sum lambda_j for j > k`.
- Encourages smooth, low-frequency wiring — the direct spectral analogue of tau-mode truncation.
- Corresponds to the frequency cost `J_freq` in the ArrowSpace cost function paper.

**lambda_fingerprint:**
- ArrowSpace-style histogram of lowest eigenvalues, used as encoder enrichment.
- Non-differentiable (used in `torch.no_grad()` context in encoder forward pass).

### `wae/encoder.py` — `WiringEncoder`

- MLP with LayerNorm + GELU.
- Optional `lambda_fingerprint` concatenation (controlled by `use_lambda_features`).
- Outputs `(z, mu, log_var)` with reparameterisation trick.
- `kl_loss` static method computes `KL(q(z|x) || N(0,I))`.

### `wae/wiring_decoder.py` — `WiringDecoder`

- Maps `z → edge weight adjustments` via mixture-of-experts over `n_heads` edge templates.
- Each head is a `Linear(hidden_dim, E)` layer; outputs are mixed via softmax gates.
- Passes edge deltas to `DifferentiableLaplacian` to produce `L(z)`.

### `wae/diffusion_decoder.py` — `DiffusionDecoder`

- `L(z), E → x̂` via `TauModeDiffusion` + optional MLP refinement.
- Learnable `log_sigma` for Gaussian likelihood `log p(x|z)`.

### `wae/model.py` — `WiringAutoencoder`

- Assembles all modules.
- `forward()` returns `loss, recon_loss, kl_loss, freq_loss, x_hat, L, z, mu, log_var`.
- `generate()` samples `z ~ N(0,I)` and decodes to wiring + embeddings.
- `from_config()` factory reads YAML config.

---

## ELBO Derivation

The standard VAE ELBO is:

```
ℒ(θ,φ; x) = E_{q_φ(z|x)}[log p_θ(x|z)]  -  KL(q_φ(z|x) || p(z))
```

In the WAE, `log p_θ(x|z)` is computed via:

```
z  →  theta = f_θ(z)  (wiring parameters)
   →  L(z) = DifferentiableLaplacian(theta)
   →  K_tau(z) = U_k · exp(-tΛ_k) · U_k^T
   →  x̂_i = K_tau(z)[i, :] · E
   →  log p(x|z) = -||x - x̂||^2 / (2σ^2) - D log σ
```

With the additional spectral regulariser, the full objective is:

```
ℒ_WAE = E_q[log p(x|z)]  -  β·KL  -  α·J_freq(L(z))
```

where `J_freq = sum_{j>k} lambda_j(L(z))` penalises high-frequency wiring.

---

## Connection to ArrowSpace / Graph Wiring

| ArrowSpace concept | WAE equivalent |
|---|---|
| `ArrowSpaceBuilder.build(E, params)` | `DifferentiableLaplacian.from_embeddings(E)` |
| kNN affinity graph + RBF kernel | Base graph (frozen topology, frozen base weights) |
| Edge weight tuning (sigma, knn_k) | `WiringDecoder` edge delta logits (learned) |
| Lambda (λ) values | Eigenvalues of `L(z)`, used in `J_freq` and `lambda_fingerprint` |
| Tau-mode truncation | `TauModeDiffusion(tau_modes=k)` |
| `J_freq` cost | `spectral_freq_cost(L, tau_modes=k)` |
| λ-fingerprint (notebooks 01–05) | `lambda_fingerprint(L)` fed to encoder |

---

## Benchmark Metrics

| Metric | How computed | What it measures |
|---|---|---|
| Reconstruction MSE | `||x - x̂||^2` averaged over test set | Quality of reconstruction via wiring path |
| KL divergence | `KL(q(z|x) || N(0,I))` | Regularisation of latent space |
| J_freq (spectral cost) | `sum lambda_j for j > tau_modes` | Smoothness of learned wiring |
| Linear probe accuracy | Logistic regression on frozen `mu` | Discriminative quality of latent z |
| Spectral entropy H(Λ) | `H(normalised eigenvalues of L(z))` | Diversity of generated wirings |
