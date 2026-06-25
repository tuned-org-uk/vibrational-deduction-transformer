# VDT Parameter Reference

This document describes every configuration parameter exposed in the YAML config files
(e.g. `configs/mps.yaml`, `configs/default.yaml`).  Each section links directly to the
relevant source files.

---

## Model Parameters  (`model:`)

These parameters control the architecture of `WiringAutoencoder` and are passed to
[`vdeductive/model.py`](vdeductive/model.py) via `from_config()`.

### `latent_dim`

Dimension of the isotropic VAE latent vector **z** produced by `WiringEncoder`.
`mu`, `log_var`, and the reparameterised sample `z` all have shape `(B, latent_dim)`.
This is independent of `q`: a linear projection `z_to_q` (shape `latent_dim -> q`)
bridges the two spaces.  Larger values give the encoder more capacity to represent
node-level variation but increase parameter count quadratically through the attention
layers.

- **YAML key:** `model.latent_dim`
- **Default:** `16`
- **Code:** [`vdeductive/model.py` -- `WiringAutoencoder.__init__`](vdeductive/model.py)

---

### `hidden_dim`

Per-node feature channel width (`feat_dim`) used inside `WiringEncoder` and as the
MLP width of `DiffusionDecoder`.  Controls the inner representation width of every
VDT block; increasing it raises both parameter count and per-epoch cost.

- **YAML key:** `model.hidden_dim`
- **Default:** `32`
- **Code:** [`vdeductive/model.py` -- `WiringAutoencoder.__init__`](vdeductive/model.py)

---

### `q`

Number of spectral modes retained from the graph Laplacian eigensystem.
`SpectralLoadingDecoder`, `ModeWeightHead` (which outputs `log_a`, `log_b` of shape
`(B, q)`), and both KL priors (`kl_S`, `kl_tau`) all operate over exactly `q` modes.
Must match `U_q.shape[1]` at runtime.  See [Modes explained](#modes-the-vibrational-basis)
below.

- **YAML key:** `model.q`
- **Default:** falls back to `tau_modes` when absent
- **Code:** [`vdeductive/spectral.py`](vdeductive/spectral.py), [`vdeductive/model.py`](vdeductive/model.py)

---

### `tau_modes`

Number of eigenvectors kept by `TauModeDiffusion` for the heat-kernel reconstruction.
The diffusion kernel is:

```
K_tau = U_k diag(exp(-t * lambda_k)) U_k^T
```

where `k = tau_modes`.  This is the direct analogue of Rayleigh's truncated modal
expansion: only the `tau_modes` lowest-frequency (smoothest) basis functions contribute
to the reconstructed signal.  When `tau_modes == q`, every retained mode participates
in diffusion; setting `tau_modes < q` means some modes inform the KL priors but not
the heat kernel.

- **YAML key:** `model.tau_modes`
- **Default:** `4`
- **Code:** [`vdeductive/spectral.py` -- `TauModeDiffusion`](vdeductive/spectral.py)

---

### `lam_s`

Precision multiplier for the spectral basis KL term **kl_S**.  Controls how tightly
the prior `p(S | I) = prod_{k,j} N(0, 1/(lam_s * lambda_k))` is enforced.  Higher
values shrink the prior variance for each mode proportionally to that mode's eigenvalue,
pushing the model to zero out high-frequency modes.  This is the primary dial for
inducing spectral sparsity -- if mode explosion (all modes remaining active throughout
training) is observed, `lam_s` should be increased.

Also serves as a minimum precision floor: the DC mode (`lambda_0 = 0`) is treated as
having at least `lam_s` precision, preventing a constant ~16 nats offset in `kl_S`
that could never be learned away.

- **YAML key:** `model.lam_s`
- **Default:** `0.2`
- **Recommended increase if mode explosion persists:** `0.8` or higher
- **Code:** [`vdeductive/spectral.py` -- `spectral_basis_kl`](vdeductive/spectral.py)

---

### `tau`

Diffusion time scale used in two places:

1. As the prior rate for `kl_tau`: `p(omega_k) = Exp(tau * lambda_k)`, which pushes
   high-frequency modes (large `lambda_k`) toward zero weight.
2. As the time argument passed to `MassMatrix` inside `build_L_f()`, weighting the
   eigenspectrum before constructing the feature-space Laplacian.

Larger `tau` amplifies the Exponential prior pressure and accelerates mode pruning
for high-frequency components.

- **YAML key:** `model.tau`
- **Default:** `0.5`
- **Code:** [`vdeductive/spectral.py` -- `tau_mode_kl`](vdeductive/spectral.py), [`vdeductive/model.py` -- `build_L_f`](vdeductive/model.py)

---

### `n_layers`

Number of VDT transformer blocks inside `WiringEncoder`.  Each block applies
multi-head spectral attention followed by a position-wise MLP.

- **YAML key:** `model.n_layers`
- **Default:** `4`
- **Code:** [`vdeductive/encoder.py`](vdeductive/encoder.py)

---

### `n_heads`

Number of attention heads per VDT block in `WiringEncoder`.  Must evenly divide
`hidden_dim`.

- **YAML key:** `model.n_heads`
- **Default:** `4`
- **Code:** [`vdeductive/encoder.py`](vdeductive/encoder.py)

---

### `dropout`

Dropout probability applied inside every VDT block.

- **YAML key:** `model.dropout`
- **Default:** `0.1`
- **Code:** [`vdeductive/encoder.py`](vdeductive/encoder.py)

---

### `eps`

Epsilon used for numerical conditioning inside `WiringEncoder` (e.g. layer-norm
stability, feature normalisation).  Separate from the eigensolver Tikhonov shift
`_EIGSOLVE_EPS = 1e-4` in `spectral.py`.

- **YAML key:** `model.eps`
- **Default:** `0.3`
- **Code:** [`vdeductive/encoder.py`](vdeductive/encoder.py)

---

### `mass_clip`

Maximum allowed value for any diagonal entry of `MassMatrix.M_diag`.  Applied during
`build_L_f()` before constructing the feature-space Laplacian
`L_f = U diag(M * lambda) U^T`.  Prevents modes near the `lambda = 1` singularity of
the normalised Laplacian from dominating the encoder attention.

- **YAML key:** `model.mass_clip`
- **Default:** `100.0` (tight; use `1e4` for very sparse graphs)
- **Code:** [`vdeductive/model.py` -- `build_L_f`, `WiringAutoencoder.__init__`](vdeductive/model.py), [`vdeductive/laplacian.py`](vdeductive/laplacian.py)

---

### `kl_z_warmup`

Number of epochs over which the `kl_z` term is linearly annealed from 0 to 1.
During warmup the isotropic KL exerts reduced pressure on the encoder, allowing
the reconstruction signal to establish stable gradients before the posterior is
regularised toward `N(0, I)`.  If mode explosion is established before epoch
`kl_z_warmup`, the warmup cannot help recover selection -- reducing this value can
add pressure earlier.

- **YAML key:** `model.kl_z_warmup`
- **Default:** `25`
- **Code:** [`train.py`](train.py)

---

## Training Parameters  (`training:`)

### `epochs`

Total number of training epochs.

- **YAML key:** `training.epochs` (also overridden by `--epochs` CLI flag)

---

### `batch_size` / `mps_batch_size`

Mini-batch size.  On Apple Silicon (MPS) the `mps_batch_size` override is used
automatically because MPS memory constraints are stricter than CUDA.

- **YAML keys:** `training.batch_size`, `training.mps_batch_size`
- **Code:** [`vdeductive/device.py`](vdeductive/device.py)

---

### `lr`

Base learning rate for the AdamW optimiser.

- **YAML key:** `training.lr`
- **Default:** `1e-4`

---

### `weight_decay`

L2 regularisation coefficient for AdamW.

- **YAML key:** `training.weight_decay`
- **Default:** `1e-5`

---

### `scheduler`

Learning-rate schedule.  Currently supported: `cosine` (cosine annealing with warmup).

- **YAML key:** `training.scheduler`

---

### `warmup_epochs`

Number of epochs for linear LR warmup before the cosine schedule begins.

- **YAML key:** `training.warmup_epochs`
- **Default:** `10`

---

### `grad_clip`

Maximum L2 norm for gradient clipping (`torch.nn.utils.clip_grad_norm_`).

- **YAML key:** `training.grad_clip`
- **Default:** `1.0`

---

### `seed`

Random seed for NumPy, Python `random`, and PyTorch.  Overridden by `--seed` CLI flag.

- **YAML key:** `training.seed`
- **Code:** [`train.py`](train.py)

---

### `density_loss_weight`

Scalar weight applied to the density regularisation loss component if active.

- **YAML key:** `training.density_loss_weight`
- **Default:** `1.0`
- **Code:** [`vdeductive/density.py`](vdeductive/density.py)

---

### `a_min`

Floor for the Gamma shape parameter `a = exp(log_a)` before `lgamma` and `digamma`
are evaluated in `tau_mode_kl`.  Prevents full Gamma collapse to a near-zero spike
(which would cause `lgamma(a) -> inf` and NaN gradients).  Gradients still flow
through `log_a`; only the forward value seen by the special functions is clipped.

- **YAML key:** `training.a_min`
- **Default:** `0.1`
- **Code:** [`vdeductive/spectral.py` -- `tau_mode_kl`](vdeductive/spectral.py)

---

### `q_min`

Minimum number of spectrally active modes required.  A mode `k` is active when its
expected value under the Gamma posterior exceeds `delta = 0.01`:

```
E[omega_k] = a_k / b_k > delta
```

When the mean batch count of active modes falls below `q_min`, the penalty
`nu * relu(q_min - N_active)` is added to the total loss.  Set to `0` to disable.

This parameter acts as a lower-bound safety net -- it prevents all modes from
collapsing simultaneously -- but does **not** prevent mode explosion (all modes
remaining active, `N_active == q`).  For mode explosion, `lam_s` is the correct dial.

- **YAML key:** `training.q_min`
- **Default:** `2`
- **Code:** [`vdeductive/spectral.py` -- `active_mode_penalty`, `count_active_modes`](vdeductive/spectral.py)

---

### `nu`

Lagrange multiplier weight for the active-mode penalty:

```
penalty = nu * relu(q_min - N_active)
```

This penalty fires only when too **few** modes are active, discouraging complete
collapse.  It is **not** effective against mode explosion (all modes active), for
which `lam_s` is the relevant parameter.  Set to `0.0` to disable.

- **YAML key:** `training.nu`
- **Default:** `0.05`
- **Recommended increase if mode explosion persists:** `0.2` or higher
- **Code:** [`vdeductive/spectral.py` -- `active_mode_penalty`](vdeductive/spectral.py), [`vdeductive/model.py`](vdeductive/model.py)

---

### `dt_init`

Initial diffusion time for `TauModeDiffusion.log_t`.  The parameter is stored as
`log_t = log(dt_init)` and exponentiated at each forward pass to keep `t > 0`.

- **YAML key:** `training.dt_init`
- **Default:** `0.001`
- **Code:** [`vdeductive/spectral.py` -- `TauModeDiffusion`](vdeductive/spectral.py)

---

## Graph Parameters  (`graph:`)

These parameters control construction of the kNN feature graph and its Laplacian
in [`vdeductive/laplacian.py`](vdeductive/laplacian.py).

### `knn_k`

Number of nearest neighbours per node for building the affinity graph.  Larger values
produce a denser graph with smoother Laplacian eigenvectors, which reduces the effective
frequency range available for mode selection.

- **YAML key:** `graph.knn_k`
- **Default:** `15`
- **Code:** [`vdeductive/laplacian.py` -- `DifferentiableLaplacian.from_embeddings`](vdeductive/laplacian.py)

---

### `sigma`

Gaussian kernel bandwidth for edge weights:
`w_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))`.  Smaller values sharpen the weight
distribution; larger values smooth it toward a uniform graph.

- **YAML key:** `graph.sigma`
- **Default:** `0.5`
- **Code:** [`vdeductive/laplacian.py`](vdeductive/laplacian.py)

---

### `normalised`

Whether to use the symmetric normalised Laplacian `L = I - D^{-1/2} A D^{-1/2}`
(eigenspectrum in [0, 2]) or the unnormalised combinatorial Laplacian.  The normalised
form is strongly recommended because `spectral_basis_kl` assumes the DC mode has
eigenvalue 0 and all others lie in (0, 2].

- **YAML key:** `graph.normalised`
- **Default:** `true`
- **Code:** [`vdeductive/laplacian.py`](vdeductive/laplacian.py)

---

### `sparse`

If `true`, the affinity matrix is stored as a sparse COO/CSR tensor rather than a
dense matrix.  Reduces memory for large graphs; requires MPS fallback or CPU for
sparse operations.

- **YAML key:** `graph.sparse`
- **Default:** `false`
- **Code:** [`vdeductive/laplacian.py`](vdeductive/laplacian.py)

---

## Modes -- the Vibrational Basis

Modes are the central physical concept of VDT.  Each mode is one eigenvector/eigenvalue
pair `(u_k, lambda_k)` of the graph Laplacian `L`.  They are the normal modes of the
graph in exactly the sense of Rayleigh's Theory of Sound: the `k`-th mode describes a
pattern of variation across nodes with spatial frequency proportional to `lambda_k`.

### How modes are used

**Reconstruction (TauModeDiffusion).**
The `tau_modes` lowest-frequency eigenvectors form a truncated heat kernel:

```
K_tau = U_k diag(exp(-t * lambda_k)) U_k^T
```

Applying `K_tau` to the node embedding table `E` reconstructs each node's
representation by spreading information along the graph's smoothest directions.  The
learnable scalar `t` (diffusion time, initialised to `dt_init`) controls how far
information propagates.  See [`vdeductive/spectral.py` -- `TauModeDiffusion`](vdeductive/spectral.py).

**Mode weights (Gamma variational posterior).**
Each mode `k` is assigned a stochastic weight `omega_k ~ Gamma(a_k, b_k)` learned by
`WiringEncoder` through `log_a` and `log_b` of shape `(B, q)`.  A mode is **active**
when `E[omega_k] = a_k / b_k > delta = 0.01`; otherwise it is dormant and contributes
nothing to the reconstruction.  The number of active modes is tracked as `N_active` in
every epoch output.  See
[`vdeductive/spectral.py` -- `count_active_modes`, `active_mode_penalty`](vdeductive/spectral.py).

**Spectral basis S (Gaussian variational posterior).**
The `q x q` matrix `S` is a learned reparametrisation of the spectral loading.  Its
prior variance per entry `(k, j)` is `1 / (lam_s * lambda_k)`, making the prior
progressively tighter for high-frequency modes.  This is the mechanism by which
`lam_s` induces sparsity.  See
[`vdeductive/spectral.py` -- `spectral_basis_kl`](vdeductive/spectral.py).

### KL terms that regulate modes

| Term | Form | Purpose | Key parameter |
|------|------|---------|---------------|
| `kl_S` | `KL(q(S) || p(S|I))` | Penalises spectral basis away from eigenvalue-weighted prior | `lam_s` |
| `kl_tau` | `KL(q(omega) || p(omega|tau, Lambda))` | Pushes high-frequency mode weights toward zero | `tau` |
| `penalty` | `nu * relu(q_min - N_active)` | Prevents too few modes from being active | `nu`, `q_min` |

### Mode collapse vs. mode explosion

| Symptom | Meaning | Fix |
|---------|---------|-----|
| `N_active < q_min` | Too many modes dormant (true collapse) | Reduce `lam_s`; lower `q_min`; check `nu` |
| `N_active == q` all training long, `kl_S` does not converge | Mode explosion -- no selection has occurred | **Increase `lam_s`** (primary); increase `nu`; reduce `kl_z_warmup` |

Runs 1-4 on Cora all exhibit mode explosion (`N_active = 4/4` throughout) with
`lam_s = 0.2` and `nu = 0.05`.  The recommended next experiment uses
`lam_s = 0.8`, `nu = 0.20`, and `kl_z_warmup = 10`.

---

## Quick-reference table

| YAML key | Section | Default | Role |
|----------|---------|---------|------|
| `model.latent_dim` | model | 16 | VAE latent dimension |
| `model.hidden_dim` | model | 32 | Encoder / decoder channel width |
| `model.q` | model | = tau_modes | Number of spectral modes |
| `model.tau_modes` | model | 4 | Eigenvectors in heat kernel |
| `model.lam_s` | model | 0.2 | Spectral basis KL weight -- **primary mode-sparsity dial** |
| `model.tau` | model | 0.5 | Diffusion time scale / mode prior rate |
| `model.n_layers` | model | 4 | VDT encoder depth |
| `model.n_heads` | model | 4 | Attention heads per block |
| `model.dropout` | model | 0.1 | Dropout probability |
| `model.eps` | model | 0.3 | Encoder numerical epsilon |
| `model.mass_clip` | model | 100.0 | MassMatrix diagonal clip |
| `model.kl_z_warmup` | model | 25 | Epochs to anneal kl_z from 0 to 1 |
| `training.lr` | training | 1e-4 | AdamW learning rate |
| `training.weight_decay` | training | 1e-5 | AdamW L2 regularisation |
| `training.grad_clip` | training | 1.0 | Gradient norm clip |
| `training.a_min` | training | 0.1 | Gamma shape floor (prevents NaN) |
| `training.q_min` | training | 2 | Min active modes before penalty fires |
| `training.nu` | training | 0.05 | Active-mode penalty weight |
| `training.dt_init` | training | 0.001 | Initial diffusion time |
| `graph.knn_k` | graph | 15 | Neighbours per node in kNN graph |
| `graph.sigma` | graph | 0.5 | Gaussian edge-weight bandwidth |
| `graph.normalised` | graph | true | Normalised vs combinatorial Laplacian |
| `graph.sparse` | graph | false | Sparse affinity storage |
