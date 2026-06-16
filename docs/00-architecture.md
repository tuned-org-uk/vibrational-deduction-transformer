# Wiring Autoencoder v2 — Architecture Reference

## Conceptual Progression

WAE v2 follows the same step-by-step progression as v1 and the book
*The Little Book of Generative AI Foundations* (Chen, 2026), but upgrades
the PPCA and VAE steps with two Spectral-PPCA structural priors:

```
Book progression:             v1 WAE analogue:             v2 WAE upgrade:

PCA                    →     Spectral Laplacian (Lf)   →  same; eigenbasis for W
  ↓                              ↓                           ↓
Autoencoder            →     Wiring AE (J_freq loss)   →  SpectralLoadingDecoder
  ↓                              ↓                           ↓
PPCA                   →     modal prior N(0,Λm⁻¹)     →  IMPLEMENTED in KL
  ↓                              ↓                           ↓
VAE + ELBO             →     recon+β·KL+α·J_freq        →  three-term ELBO
  ↓                              ↓                           ↓
Associative memory     →     (absent)                   →  SpectralAssociativeMemory
  ↓                              ↓                           ↓
Diffusion / Flows      →     (future) WAE-Diffusion     →  unchanged
```

---

## Data Flow Diagram

```
 Input x (B, D)                   Embedding table E (N, D)
     │                                    │
     ├── [λ-fingerprint from L(I)] ───────┤
     │                                    │
     ▼                                    │
 ┌─────────────────────────────────┐      │
 │  WiringEncoderV2                │      │
 │  (MLP + λ-fingerprint)          │      │
 │  → (z, μ, logσ, log_a, log_b)  │      │
 └─────────────────────────────────┘      │
     │                                    │
   (z, μ, logσ)  ←── reparameterise      │
   (log_a, log_b) ── mode weight params   │
     │
     ▼
 ┌──────────────────────────────────┐
 │ SpectralLoadingDecoder           │
 │  z, U_{1:q}  →  W, ω, S         │
 │  W = U_{1:q} diag(ω) S          │
 │  (spectral-basis edge synthesis) │
 └──────────────────────────────────┘
     │
   W (B, d, q)  →  DifferentiableLaplacian.from_spectral_loading(W, L_base)
     │
   L(z)  (B, N, N)
     │
     ▼
 ┌────────────────────┐      ┌────────────┐
 │  DiffusionDecoder  │ ←─── │     E      │
 │  TauModeDiffusion  │      └────────────┘
 │  + MLP refinement  │
 └────────────────────┘
     │
   x̂  (B, D)    ─────────────────────►  WAE v2 ELBO Loss
                                               │
            recon  kl_S  kl_tau  ──────────────┘
```

---

## The Three-Term ELBO

The v1 objective:

```
ℒ_WAE = E_q[log p(x|z)]  −  β·KL(q(z) ∥ N(0,I))  −  α·J_freq(L(z))
```

is replaced in v2 by the Spectral-PPCA ELBO:

```
ℒ_WAEv2 = E_q[log p(x|z,W)]
         − KL( q(S)    ∥  p(S|I)   )           [spectral basis KL]
         − KL( q(ω)    ∥  p(ω|τ,Λ))           [τ-mode frequency KL]
```

The latent `z` is regularised by the standard isotropic prior `N(0,I)`,
identical to v1. The ArrowSpace index `I` enters solely through the
pre-computed eigenpair `(U_{1:q}, Λ_{1:q})` of `L(I)`, which are frozen
constants at training time — no Laplacian is evaluated or inverted at runtime.

### Term 1 — Reconstruction

Unchanged from v1: Gaussian NLL through `TauModeDiffusion`.

```
log p(x|z) = −‖x − x̂‖² / (2σ²) − D log σ
```

### Term 2 — Spectral-Basis KL

The loading matrix `S` lives in the Laplacian eigenbasis. Prior:

```
p(S|I) ∝ exp(−λs/2 · tr(S⊤ Λ_{1:q} S))
```

Eigenvalue-weighted shrinkage: high-frequency components (large λk) are penalised
more strongly, implementing a spectral Occam's razor. Closed-form Gaussian KL.
`Λ_{1:q}` comes from the pre-computed eigendecomposition of `L(I)` — it is a
frozen constant throughout training.

### Term 3 — τ-Mode Frequency KL

Replaces the hard `J_freq` penalty. Mode weights `ω_k > 0` carry an exponential prior:

```
p(ωk | τ, λk) = Exponential(τ λk)
```

giving heavy support on low-frequency modes (small λk) and exponential decay
for high frequencies. `q(ωk) = Gamma(ak, bk)`, fully reparametrisable.
Closed-form KL between Gamma and Exponential:

```
KL(Gamma(a,b) ∥ Exp(r)) = log(b) − log(r) + lgamma(a)
                          + (1−a)·digamma(a) + a·b/r
```

`λk` values are taken from the same frozen `Λ_{1:q}` used in Term 2.

---

## Module Reference

### `wae/laplacian.py` — `DifferentiableLaplacian` (unchanged)

No changes from v1. The new entry point `from_spectral_loading(W, L_base)` is
added as a class method:

- Takes spectral loading `W (B, d, q)` and base Laplacian topology.
- Synthesises edge weights as `w_ij = base_w_ij * sigmoid(‖W_i − W_j‖²)`.
- Fully differentiable through `W` back to the spectral decoder.

### `wae/spectral.py` — updated

**`TauModeDiffusion`** — unchanged from v1.

**`spectral_freq_cost`** — retained as ablation; disabled by default in v2.

**`lambda_fingerprint`** — unchanged.

**NEW: `spectral_basis_kl(S, log_var_S, eigvals_q, lam_s)`**
- Computes KL(q(S) ∥ p(S|I)) under eigenvalue-weighted Gaussian prior.
- `S (B, q, q)`, `eigvals_q (q,)`.
- `eigvals_q` is a frozen constant from the pre-computed eigendecomposition of `L(I)`.

**NEW: `tau_mode_kl(log_a, log_b, eigvals_q, tau)`**
- Computes KL(Gamma(a,b) ∥ Exponential(τλk)) per mode, summed.
- Fully closed-form, no MC sampling.
- `eigvals_q` same frozen constant as above.

### `wae/encoder.py` — `WiringEncoderV2`

Changes from v1 `WiringEncoder`:

- **`kl_loss`**: standard isotropic `KL(q(z) ∥ N(0,I))`, identical to v1.
  No Laplacian-precision term; no runtime graph construction.
- **`ModeWeightHead` added**: small linear layer outputting `(log_a, log_b)`
  for each of the `q` mode weights. These parametrise `q(ω)`.
- **`lambda_fingerprint` concatenation**: unchanged. The fingerprint is read
  from the fixed `L(I)` — it is not rebuilt from data at runtime.
- Outputs `(z, mu, log_var, log_a, log_b)`.

### `wae/wiring_decoder.py` — `SpectralLoadingDecoder` (new, replaces `WiringDecoder`)

- Maps `z (B, q)` and `U_q (d, q)` to `(W, ω, S)`.
- `W = U_q @ diag(ω) @ S` — loading matrix in Laplacian eigenbasis.
- `S_net`: `Linear(q, q*q)` producing flattened `S` from `z`.
- `omega_net`: `Linear(q, q)` producing `log_ω`; `ω = exp(log_ω)`.
- Edge weights synthesised via `DifferentiableLaplacian.from_spectral_loading(W, L_base)`.

`WiringDecoder` from v1 is retained; `SpectralLoadingDecoder` is the default in v2.
Config flag: `decoder_type: spectral | mixture_of_experts`.

### `wae/diffusion_decoder.py` — `DiffusionDecoder` (unchanged)

No changes. `TauModeDiffusion` is already the Spectral-PPCA decoder.

### `wae/model.py` — `WiringAutoencoderV2`

- Assembles `WiringEncoderV2`, `SpectralLoadingDecoder`, `DiffusionDecoder`.
- `forward()` returns `(loss, recon, kl_z, kl_S, kl_tau, x_hat, L_z, z, mu, log_var)`.
  `kl_z` is the standard isotropic KL, included for monitoring but not weighted
  by the spectral priors.
- **NEW: `extract_spectral_artefact(U_q, eigvals_q)`** — packages
  `(W_hat, omega_hat, S_memory)` as the post-training spectral artefact.
- `generate()` unchanged.
- `from_config()` factory reads updated YAML config.

### `wae/spectral_memory.py` — `SpectralAssociativeMemory` (new module)

- Wraps a pre-built Hopfield/linear associative memory initialised from
  the spectral artefact `A(I)`.
- `forward(query)`: Hopfield retrieval via softmax-weighted spectral keys.
- `delta_update(key, value)`: online delta-rule write without corrupting
  spectral key structure.
- `from_wae(wae_v2, U_q, eigvals_q, d_model)`: class method for
  post-training construction from a trained `WiringAutoencoderV2`.

---

## ELBO Comparison: v1 vs v2

| Term | v1 WAE | v2 WAE |
|---|---|---|
| Reconstruction | `−‖x−x̂‖²/2σ²` | Same |
| Latent KL | `KL(q(z) ∥ N(0,I))` isotropic | Same — isotropic `N(0,I)` |
| Spectral basis KL | None | `KL(q(S) ∥ p(S|I))` eigenvalue-weighted Gaussian |
| τ-mode term | `α·Σ_{j>k} λj` hard penalty | `KL(q(ω) ∥ Exp(τλk))` variational Gamma prior |
| Density matrix | Stability diagnostic only | Optional: `μ2·‖ϱt‖²_F` occupancy penalty |

---

## Spectral Artefact and Associative Memory

After training, the spectral artefact `A(I)` is extracted:

```
A(I) = {
  W_hat    :  mean loading matrix  Ŵ = U_{1:q} diag(ω̂) Ŝ
  omega_hat:  posterior mode weights  {E[ωk]}_{k=1}^q
  S_memory :  associative memory matrix
               S_I = Σ_k E[ωk] · d_θ(ŵk) · ŵk⊤
}
```

Key properties of `S_I`:

- **Keys** `ŵk` are linear combinations of Laplacian eigenvectors: approximately
  orthonormal by construction, maximising retrieval SNR.
- **Values** `d_θ(ŵk)` are the decoder responses at each spectral direction:
  represent the data pattern explained by that frequency band.
- **Mode weights** `E[ωk]` down-weight high-frequency (noisy) components.

### Two-Phase Architecture

```
PHASE 1 — OFFLINE (WAE v2 training)
  ArrowSpace index I  →  L(I), U_q, Λq          [one-time eigendecomposition]
  WiringAutoencoderV2.train()  →  ELBO maximisation  [no runtime Laplacian ops]
  extract_spectral_artefact()  →  A(I)
  SpectralAssociativeMemory(A(I))  →  S_I
  Optional: ELBO Bayes factor over competing indices I1, I2, ...

PHASE 2 — ONLINE (Spectral Memory Transformer)
  Transformer FFN / cross-attention initialised from S_I
  Self-attention: dynamic short-term associations
  S_I: long-term spectral prior memory
  Delta-rule updates: write new associations online
```

---

## Benchmark Metrics (v2 additions)

| Metric | How computed | What it measures |
|---|---|---|
| Reconstruction MSE | `‖x − x̂‖²` averaged over test set | Quality of reconstruction via wiring path |
| KL spectral basis | `KL(q(S) ∥ p(S|I))` | Spectral alignment of loadings with index |
| τ-mode KL | `KL(q(ω) ∥ Exp(τΛ))` | Effective frequency band selection |
| Active mode count | `Σk 1[E[ωk] > δ]` | Modes contributing to W |
| Memory retrieval SNR | `dk / N_stored` via key orthogonality | Quality of associative memory |
| ELBO Bayes factor | `exp(ℒ(I1) − ℒ(I2))` | Comparison of ArrowSpace indices |
| Linear probe accuracy | Logistic regression on frozen `mu` | Discriminative quality of latent z |
| Spectral entropy H(Λ) | `H(normalised eigenvalues of L(z))` | Diversity of generated wirings |

---

## Connection to ArrowSpace / Graph Wiring

| ArrowSpace concept | v1 WAE equivalent | v2 WAE equivalent |
|---|---|---|
| `ArrowSpaceBuilder.build(E, params)` | `DifferentiableLaplacian.from_embeddings(E)` | Same + `from_spectral_loading(W, L_base)` |
| kNN affinity graph + RBF kernel | Base graph (frozen topology, frozen base weights) | Same |
| Edge weight tuning | `WiringDecoder` edge delta logits | `SpectralLoadingDecoder`: `W = U_{1:q} diag(ω) S` |
| Lambda (λ) values | Eigenvalues in `J_freq` and `lambda_fingerprint` | Same + eigenvalue-weighted KL priors (frozen) |
| Tau-mode truncation | `TauModeDiffusion(tau_modes=k)` | Same + `tau_mode_kl` variational prior |
| `J_freq` cost | `spectral_freq_cost(L, tau_modes=k)` | Replaced by `tau_mode_kl`; kept as ablation |
| λ-fingerprint | `lambda_fingerprint(L)` fed to encoder | Unchanged; read from fixed `L(I)` |
| Index selection | Not Bayesian | ELBO Bayes factor: `exp(ℒ(I1) − ℒ(I2))` |
