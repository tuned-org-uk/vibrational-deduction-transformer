# Vibrational Deduction Transformer — Architecture Reference

## Conceptual Progression

The VDT follows the same step-by-step progression as the book
*The Little Book of Generative AI Foundations* (Chen, 2026), upgrading
the PPCA and VAE steps with two Spectral-PPCA structural priors:

```
Book progression:             Wiring AE analogue:          VDT upgrade:

PCA                    ->    Spectral Laplacian (Lf)   ->  same; eigenbasis for W
  |                              |                           |
Autoencoder            ->    Wiring AE (J_freq loss)   ->  SpectralLoadingDecoder
  |                              |                           |
PPCA                   ->    modal prior N(0,Lm^-1)    ->  IMPLEMENTED in KL
  |                              |                           |
VAE + ELBO             ->    recon+beta*KL+alpha*J_fr.  ->  three-term ELBO
  |                              |                           |
Associative memory     ->    (absent)                   ->  SpectralAssociativeMemory
  |                              |                           |
Diffusion / Flows      ->    (future) VDT-Diffusion     ->  unchanged
```

---

## Data Flow Diagram

```
 Input x (B, D)                   Embedding table E (N, D)
     |                                    |
     +-- [lambda-fingerprint from L(I)] --+
     |                                    |
     v                                    |
 +-----------------------------------+    |
 |  WiringEncoder.                   |    |
 |  (MLP + lambda-fingerprint)       |    |
 |  -> (z, mu, logS, log_a, log_b)  |    |
 +-----------------------------------+    |
     |                                    |
   (z, mu, logS)  <-- reparameterise      |
   (log_a, log_b) -- mode weight params   |
     |
     v
 +----------------------------------+
 | SpectralLoadingDecoder           |
 |  z, U_{1:q}  ->  W, omega, S    |
 |  W = U_{1:q} diag(omega) S      |
 |  (spectral-basis edge synthesis) |
 +----------------------------------+
     |
   W (B, d, q)  ->  DifferentiableLaplacian.from_spectral_loading(W, L_base)
     |
   L(z)  (B, N, N)
     |
     v
 +--------------------+      +------------+
 |  DiffusionDecoder  | <--- |     E      |
 |  TauModeDiffusion  |      +------------+
 |  + MLP refinement  |
 +--------------------+
     |
   x_hat  (B, D)  ------------------>  VDT ELBO Loss
                                               |
            recon  kl_S  kl_tau  --------------+
```

---

## The Three-Term ELBO

The prior Wiring AE objective:

```
L_vdt = E_q[log p(x|z)]  -  beta*KL(q(z) || N(0,I))  -  alpha*J_freq(L(z))
```

is replaced in the VDT by the Spectral-PPCA ELBO:

```
L_VDT = E_q[log p(x|z,W)]
       - KL( q(S)    ||  p(S|I)   )           [spectral basis KL]
       - KL( q(omega)||  p(omega|tau,L))      [tau-mode frequency KL]
```

The latent `z` is regularised by the standard isotropic prior `N(0,I)`.
The ArrowSpace index `I` enters solely through the pre-computed eigenpair
`(U_{1:q}, L_{1:q})` of `L(I)`, which are frozen constants at training time—
no Laplacian is evaluated or inverted at runtime.

### Term 1 — Reconstruction

Unchanged from the base Wiring AE: Gaussian NLL through `TauModeDiffusion`.

```
log p(x|z) = -||x - x_hat||^2 / (2*sigma^2) - D*log(sigma)
```

### Term 2 — Spectral-Basis KL

The loading matrix `S` lives in the Laplacian eigenbasis. Prior:

```
p(S|I)  proportional to  exp(-lambda_s/2 * tr(S^T Lambda_{1:q} S))
```

Eigenvalue-weighted shrinkage: high-frequency components (large lambda_k) are
penalised more strongly, implementing a spectral Occam's razor. Closed-form
Gaussian KL. `Lambda_{1:q}` comes from the pre-computed eigendecomposition of
`L(I)` — it is a frozen constant throughout training.

### Term 3 — Tau-Mode Frequency KL

Replaces the hard `J_freq` penalty. Mode weights `omega_k > 0` carry an
exponential prior:

```
p(omega_k | tau, lambda_k) = Exponential(tau * lambda_k)
```

giving heavy support on low-frequency modes (small lambda_k) and exponential
decay for high frequencies. `q(omega_k) = Gamma(a_k, b_k)`, fully
reparametrisable. Closed-form KL between Gamma and Exponential:

```
KL(Gamma(a,b) || Exp(r)) = log(b) - log(r) + lgamma(a)
                           + (1-a)*digamma(a) + a*b/r
```

`lambda_k` values are taken from the same frozen `Lambda_{1:q}` used in Term 2.

---

## Module Reference

### `vdt/laplacian.py` — `DifferentiableLaplacian`

The new entry point `from_spectral_loading(W, L_base)` is added as a class method:

- Takes spectral loading `W (B, d, q)` and base Laplacian topology.
- Synthesises edge weights as `w_ij = base_w_ij * sigmoid(||W_i - W_j||^2)`.
- Fully differentiable through `W` back to the spectral decoder.

### `vdt/spectral.py` — updated

**`TauModeDiffusion`** — unchanged from base Wiring AE.

**`spectral_freq_cost`** — retained as ablation; disabled by default.

**`lambda_fingerprint`** — unchanged.

**NEW: `spectral_basis_kl(S, log_var_S, eigvals_q, lam_s)`**
- Computes KL(q(S) || p(S|I)) under eigenvalue-weighted Gaussian prior.
- `S (B, q, q)`, `eigvals_q (q,)`.
- `eigvals_q` is a frozen constant from the pre-computed eigendecomposition of `L(I)`.

**NEW: `tau_mode_kl(log_a, log_b, eigvals_q, tau)`**
- Computes KL(Gamma(a,b) || Exponential(tau*lambda_k)) per mode, summed.
- Fully closed-form, no MC sampling.
- `eigvals_q` same frozen constant as above.

### `vdt/encoder.py` — `WiringEncoder`

- **`kl_loss`**: standard isotropic `KL(q(z) || N(0,I))`. No Laplacian-precision
  term; no runtime graph construction.
- **`ModeWeightHead` added**: small linear layer outputting `(log_a, log_b)` for
  each of the `q` mode weights. These parametrise `q(omega)`.
- **`lambda_fingerprint` concatenation**: unchanged. The fingerprint is read from
  the fixed `L(I)` — it is not rebuilt from data at runtime.
- Outputs `(z, mu, log_var, log_a, log_b)`.

### `vdt/wiring_decoder.py` — `SpectralLoadingDecoder` (default)

- Maps `z (B, q)` and `U_q (d, q)` to `(W, omega, S)`.
- `W = U_q @ diag(omega) @ S` — loading matrix in Laplacian eigenbasis.
- `S_net`: `Linear(q, q*q)` producing flattened `S` from `z`.
- `omega_net`: `Linear(q, q)` producing `log_omega`; `omega = exp(log_omega)`.
- Edge weights synthesised via `DifferentiableLaplacian.from_spectral_loading(W, L_base)`.

`WiringDecoder` is retained for ablation; `SpectralLoadingDecoder` is the default.
Config flag: `decoder_type: spectral | mixture_of_experts`.

### `vdt/diffusion_decoder.py` — `DiffusionDecoder` (unchanged)

No changes. `TauModeDiffusion` is already the Spectral-PPCA decoder.

### `vdt/model.py` — `WiringAutoencoderV2`

- Assembles `WiringEncoder`, `SpectralLoadingDecoder`, `DiffusionDecoder`.
- `forward()` returns `(loss, recon, kl_z, kl_S, kl_tau, x_hat, L_z, z, mu, log_var)`.
  `kl_z` is the standard isotropic KL, included for monitoring but not weighted
  by the spectral priors.
- **`extract_spectral_artefact(U_q, eigvals_q)`** — packages
  `(W_hat, omega_hat, S_memory)` as the post-training spectral artefact.
- `generate()` unchanged.
- `from_config()` factory reads the YAML config.

### `vdt/spectral_memory.py` — `SpectralAssociativeMemory`

- Wraps a pre-built Hopfield/linear associative memory initialised from the
  spectral artefact `A(I)`.
- `forward(query)`: Hopfield retrieval via softmax-weighted spectral keys.
- `delta_update(key, value)`: online delta-rule write without corrupting spectral
  key structure.
- `from_vdt(vdt, U_q, eigvals_q, d_model)`: class method for post-training
  construction from a trained `WiringAutoencoderV2`.

---

## ELBO Structure

| Term | Wiring AE (base) | VDT |
|---|---|---|
| Reconstruction | `-||x-x_hat||^2/2*sigma^2` | Same |
| Latent KL | `KL(q(z) || N(0,I))` isotropic | Same — isotropic `N(0,I)` |
| Spectral basis KL | None | `KL(q(S) || p(S|I))` eigenvalue-weighted Gaussian |
| Tau-mode term | `alpha*sum_{j>k} lambda_j` hard penalty | `KL(q(omega) || Exp(tau*lambda_k))` variational Gamma prior |
| Density matrix | Stability diagnostic only | Optional: `mu2*||rho_t||^2_F` occupancy penalty |

---

## Spectral Artefact and Associative Memory

After training, the spectral artefact `A(I)` is extracted:

```
A(I) = {
  W_hat    :  mean loading matrix  W_hat = U_{1:q} diag(omega_hat) S_hat
  omega_hat:  posterior mode weights  {E[omega_k]}_{k=1}^q
  S_memory :  associative memory matrix
               S_I = sum_k E[omega_k] * d_theta(w_hat_k) * w_hat_k^T
}
```

Key properties of `S_I`:

- **Keys** `w_hat_k` are linear combinations of Laplacian eigenvectors:
  approximately orthonormal by construction, maximising retrieval SNR.
- **Values** `d_theta(w_hat_k)` are the decoder responses at each spectral
  direction: represent the data pattern explained by that frequency band.
- **Mode weights** `E[omega_k]` down-weight high-frequency (noisy) components.

### Two-Phase Architecture

```
PHASE 1 -- OFFLINE (VDT training)
  ArrowSpace index I  ->  L(I), U_q, Lq        [one-time eigendecomposition]
  WiringAutoencoderV2.train()  ->  ELBO max.    [no runtime Laplacian ops]
  extract_spectral_artefact()  ->  A(I)
  SpectralAssociativeMemory(A(I))  ->  S_I
  Optional: ELBO Bayes factor over competing indices I1, I2, ...

PHASE 2 -- ONLINE (Spectral Memory Transformer)
  Transformer FFN / cross-attention initialised from S_I
  Self-attention: dynamic short-term associations
  S_I: long-term spectral prior memory
  Delta-rule updates: write new associations online
```

---

## Benchmark Metrics

| Metric | How computed | What it measures |
|---|---|---|
| Reconstruction MSE | `||x - x_hat||^2` averaged over test set | Quality of reconstruction via wiring path |
| KL spectral basis | `KL(q(S) || p(S|I))` | Spectral alignment of loadings with index |
| Tau-mode KL | `KL(q(omega) || Exp(tau*L))` | Effective frequency band selection |
| Active mode count | `sum_k 1[E[omega_k] > delta]` | Modes contributing to W |
| Memory retrieval SNR | `dk / N_stored` via key orthogonality | Quality of associative memory |
| ELBO Bayes factor | `exp(L(I1) - L(I2))` | Comparison of ArrowSpace indices |
| Linear probe accuracy | Logistic regression on frozen `mu` | Discriminative quality of latent z |
| Spectral entropy H(L) | `H(normalised eigenvalues of L(z))` | Diversity of generated wirings |

---

## Connection to ArrowSpace / Graph Wiring

| ArrowSpace concept | Wiring AE equivalent | VDT equivalent |
|---|---|---|
| `ArrowSpaceBuilder.build(E, params)` | `DifferentiableLaplacian.from_embeddings(E)` | Same + `from_spectral_loading(W, L_base)` |
| kNN affinity graph + RBF kernel | Base graph (frozen topology, frozen base weights) | Same |
| Edge weight tuning | `WiringDecoder` edge delta logits | `SpectralLoadingDecoder`: `W = U_{1:q} diag(omega) S` |
| Lambda values | Eigenvalues in `J_freq` and `lambda_fingerprint` | Same + eigenvalue-weighted KL priors (frozen) |
| Tau-mode truncation | `TauModeDiffusion(tau_modes=k)` | Same + `tau_mode_kl` variational prior |
| `J_freq` cost | `spectral_freq_cost(L, tau_modes=k)` | Replaced by `tau_mode_kl`; kept as ablation |
| Lambda-fingerprint | `lambda_fingerprint(L)` fed to encoder | Unchanged; read from fixed `L(I)` |
| Index selection | Not Bayesian | ELBO Bayes factor: `exp(L(I1) - L(I2))` |
