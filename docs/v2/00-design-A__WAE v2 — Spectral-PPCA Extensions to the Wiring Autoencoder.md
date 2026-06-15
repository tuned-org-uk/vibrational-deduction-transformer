# WAE v2 — Spectral-PPCA Extensions to the Wiring Autoencoder

## Overview

The existing Wiring Autoencoder (WAE) already shares much of its theoretical DNA with the Spectral-PPCA VAE framework derived from first principles in the prior conversation. This document maps the precise overlaps, identifies the gaps, and specifies a concrete **WAE v2** architecture that upgrades the existing design with the three Spectral-PPCA structural priors: (1) Laplacian eigenbasis parametrisation of loadings, (2) Laplacian-precision priors on latents, and (3) \(\tau\)-mode frequency prior on active spectral bands. It also specifies how the resulting spectral artefact feeds a downstream transformer with pre-built spectral associative memory.

***

## 1. Side-by-Side: Existing WAE vs Spectral-PPCA VAE

| Dimension | Existing WAE | Spectral-PPCA VAE | Gap / Upgrade |
|---|---|---|---|
| **Latent prior** \(p(z)\) | \(\mathcal{N}(0, I)\) — standard isotropic | \(\mathcal{N}(0, \Lambda_m^{-1})\) — modal prior via \(L_f\) eigenvalues | WAE README names the modal prior but does not enforce it in the KL; WAE v2 upgrades this |
| **Encoder output** | \((z, \mu, \log\sigma)\) from MLP + \(\lambda\)-fingerprint concatenation | \((z, \mu, \log\sigma)\) via reparametrisation under Laplacian-precision prior | Laplacian-precision KL replaces standard \(\mathrm{KL}(q(z)\|\mathcal{N}(0,I))\) |
| **Loading matrix** \(W\) | Implicit — decoder MLP maps \(z \to\) edge deltas; no explicit spectral basis | Explicit: \(W = U_{1:q} S\), \(S\) lives in Laplacian eigenbasis | Critical gap: WAE does not constrain \(W\) to the eigenbasis |
| **Wiring decoder** | Mixture-of-experts over \(n_\text{heads}\) edge templates | Replaced / augmented by spectral-basis edge synthesis | WAE v2 keeps MoE but gates heads by Laplacian mode index |
| **Diffusion decoder** | `TauModeDiffusion`: \(K_\tau = U_k \exp(-t\Lambda_k) U_k^\top\) | Heat kernel \(K_\tau\) with learnable \(t\) — **identical**; \(\tau\)-mode already present | No gap; existing TauModeDiffusion is already the spectral-PPCA decoder |
| **Spectral cost** \(J_\text{freq}\) | \(\sum_{j>k} \lambda_j(L(z))\) — penalises high-frequency wiring | \(\tau\)-mode prior KL: \(\mathrm{KL}(q(\omega)\|p(\omega\mid\tau,\Lambda))\) | \(J_\text{freq}\) is a hard penalty; WAE v2 replaces it with a proper Bayesian KL over mode weights \(\omega\) |
| **Latent smoothness** | None — no sample-graph Laplacian on \(z\) coordinates | \(p(X\mid\mathcal{I}) = \mathcal{N}(0, (I+\beta L_s)^{-1})\); Dirichlet energy in KL | New component in WAE v2 |
| **Laplacian as index** \(\mathcal{I}\) | Fixed topology (kNN + RBF); edge weights learned via `WiringDecoder` | \(\mathcal{I}\) is the full ArrowSpace index; ELBO scores competing indices | WAE v2 adds ELBO-based index selection loop |
| **ELBO objective** | \(\mathcal{L} = \mathbb{E}[\log p(x\mid z)] - \beta\cdot\mathrm{KL} - \alpha\cdot J_\text{freq}\) | \(\mathcal{L} = \text{recon} - \mathrm{KL}(X) - \mathrm{KL}(S) - \mathrm{KL}(\omega)\) | Three KL terms replace one KL + one hard penalty |
| **Signed density matrix** \(\varrho_t\) | Defined in VDT backbone; tracked as stability diagnostic | Not in Spectral-PPCA VAE (pure VAE model) | WAE v2 retains \(\varrho_t\) as interpretability layer; new: \(\varrho_t\) initialises associative memory |
| **Associative memory** | None | Pre-built \(S_\mathcal{I} = \sum_k \mathbb{E}[\omega_k] d_\theta(\hat{w}_k) \hat{w}_k^\top\) | Entirely new component in WAE v2 |
| **Transformer memory init** | Standard random init | Spectral artefact initialises FFN / cross-attention value matrices | New in WAE v2 |
| **Index selection** | Not Bayesian — single fixed graph | ELBO Bayes factor over candidate ArrowSpace indices | New in WAE v2 |
| **Stability framework** | Full CFL / damping / density-matrix hierarchy in `04-stability.md` | No dedicated stability analysis in Spectral-PPCA VAE derivation | WAE v2 retains the full stability hierarchy from `04-stability.md` |

***

## 2. What is Already Correct in WAE

The existing WAE is more aligned with the Spectral-PPCA framework than a surface reading suggests:

- The **`TauModeDiffusion` module** is exactly the heat kernel decoder \(K_\tau = U_k \exp(-t\Lambda_k) U_k^\top\) with differentiable eigendecomposition via `torch.linalg.eigh`. The Spectral-PPCA VAE's decoder is the same object; no change needed.
- The **`lambda_fingerprint`** fed to the encoder is the ArrowSpace spectral histogram — the same quantity that informs \(\mathcal{I}\) in the Spectral-PPCA framework.
- The **`spectral_freq_cost`** penalising \(\sum_{j>k}\lambda_j\) is the direct precursor to the \(\tau\)-mode KL term. The upgrade is to replace the hard penalty with a proper variational Gamma prior on mode weights \(\omega_k\).
- The **modal prior \(p(z) = \mathcal{N}(0, \Lambda_m^{-1})\)** is listed in the README concept table as the PPCA analogue, but the implementation (`WiringEncoder.kl_loss`) computes \(\mathrm{KL}(q(z)\|\mathcal{N}(0,I))\). The \(\Lambda_m\)-scaled prior is present conceptually but not yet wired into the KL.
- The **signed density matrix** \(\varrho_t = \varrho_t^+ - \varrho_t^-\) in the VDT backbone provides a natural low-rank factorisation of the latent covariance — exactly the kind of structured uncertainty representation that WAE v2 can expose as the spectral artefact.
- The **stability hierarchy** in `04-stability.md` (CFL condition, per-mode damping, spectral entropy, density-matrix PSD) is exactly the set of diagnostics needed to validate WAE v2 training.

***

## 3. WAE v2 Architecture Specification

### 3.1 Overview

WAE v2 retains the full VDT encoder backbone, `DifferentiableLaplacian`, `TauModeDiffusion`, and stability diagnostics from the existing codebase. The five targeted changes are:

1. **Spectral-basis loading reparametrisation** — \(W = U_{1:q} S\), replacing the unconstrained MLP decoder mapping.
2. **Laplacian-precision latent KL** — replacing the standard isotropic KL with the graph-smoothness-weighted KL.
3. **\(\tau\)-mode variational prior** — replacing the hard \(J_\text{freq}\) penalty with a learnable Gamma/Log-Normal KL over mode weights \(\omega_k\).
4. **Spectral artefact extraction** — a post-training export API that packages \(\hat{W}, \{\hat{\mu}_{x,n}\}, \{\mathbb{E}[\omega_k]\}\).
5. **Spectral associative memory init** — assembles \(S_\mathcal{I}\) from the artefact and injects it into transformer FFN / cross-attention layers.

### 3.2 Upgraded ELBO

The WAE v2 training objective replaces:

\[
\mathcal{L}_\text{WAE} = \mathbb{E}_q[\log p(x\mid z)] - \beta\cdot\mathrm{KL}(q(z)\|\mathcal{N}(0,I)) - \alpha\cdot J_\text{freq}(L(z))
\]

with the four-term Spectral-PPCA ELBO:

\[
\mathcal{L}_\text{WAE-v2} = \underbrace{\mathbb{E}_q[\log p(x\mid z, W)]}_\text{recon} - \underbrace{\mathrm{KL}(q(z)\|p_\text{Lap}(z))}_\text{latent smoothness} - \underbrace{\mathrm{KL}(q(S)\|p(S\mid\mathcal{I}))}_\text{spectral basis KL} - \underbrace{\mathrm{KL}(q(\omega)\|p(\omega\mid\tau,\Lambda))}_{\tau\text{-mode KL}}
\]

where:

- \(p_\text{Lap}(z) = \mathcal{N}(0, (I + \beta L_s)^{-1})\), \(L_s\) being the sample-graph Laplacian built on-the-fly from the current batch embeddings.
- \(p(S\mid\mathcal{I}) \propto \exp(-\frac{\lambda_s}{2}\operatorname{tr}(S^\top \Lambda_{1:q} S))\), i.e. a Gaussian in the spectral basis with eigenvalue-weighted shrinkage.
- \(p(\omega_k \mid \tau, \lambda_k)\) is Exponential(\(\tau\lambda_k\)) — heavier penalty on high-frequency modes.

The Laplacian-precision KL expands to:

\[
\mathrm{KL}(q(z)\|p_\text{Lap}(z)) = \frac{1}{2}\left[\operatorname{tr}(\Sigma_z(I+\beta L_s)) + \mu_z^\top(I+\beta L_s)\mu_z - d - \log\det\Sigma_z + \log\det(I+\beta L_s)^{-1}\right],
\]

where the cross term \(\mu_z^\top L_s \mu_z\) is the graph Dirichlet energy of the latent mean — penalising latent codes that jump sharply across neighbouring samples.

### 3.3 Module Changes

#### `wae/encoder.py` — `WiringEncoder` upgrade

- Replace `kl_loss` static method: instead of \(\mathrm{KL}(q\|\mathcal{N}(0,I))\), compute the Laplacian-precision KL above.
- Add `sample_graph_laplacian(z_batch)` utility: builds a sparse kNN Laplacian from the batch of latent means at each forward pass.
- Add `ModeWeightHead`: a small MLP or linear layer producing log-Gamma parameters \((a_k, b_k)\) for each mode weight \(\omega_k\); these parametrise \(q(\omega)\).

#### `wae/wiring_decoder.py` — `WiringDecoder` spectral-basis reparametrisation

Replace the current unconstrained MLP with an explicit spectral-basis loading decoder:

```python
class SpectralLoadingDecoder(nn.Module):
    """
    Replaces WiringDecoder for WAE v2.
    Parametrises W = U_{1:q} * diag(omega) * S,
    where U_{1:q} are the q lowest eigenvectors of L(I).
    """
    def __init__(self, q: int, d: int, n_heads: int = 4):
        super().__init__()
        # S lives in R^{q x q}; learnable in spectral basis
        self.S_net = nn.Linear(q, q * q)   # produces flattened S from z
        # omega (mode weights): positive, parametrised as log
        self.omega_net = nn.Linear(q, q)    # produces log_omega from z
        self.q = q; self.d = d

    def forward(self, z: torch.Tensor, U_q: torch.Tensor) -> tuple:
        """
        z    : (B, q)  latent code in modal coordinates
        U_q  : (d, q)  first q eigenvectors of L(I)
        Returns: W (B, d, q), omega (B, q), S (B, q, q)
        """
        B = z.size(0)
        S = self.S_net(z).view(B, self.q, self.q)
        omega = torch.exp(self.omega_net(z))           # (B, q), positive
        # W = U_q @ diag(omega) @ S  per sample
        W = U_q.unsqueeze(0) @ (omega.unsqueeze(-1) * S)  # (B, d, q)
        return W, omega, S
```

The output `W` feeds directly into `DifferentiableLaplacian` via a Cholesky-like edge weight synthesis: \(A_{ij} = (W_i - W_j)^\top (W_i - W_j)\), giving edge weights that are smooth functions of spectral mode activations.

#### `wae/spectral.py` — `TauModeKL` (new)

Replace the hard `spectral_freq_cost` with a proper variational KL:

```python
def tau_mode_kl(log_a: torch.Tensor, log_b: torch.Tensor,
                eigvals: torch.Tensor, tau: float) -> torch.Tensor:
    """
    KL( Gamma(a_k, b_k) || Exponential(tau * lambda_k) )
    Closed form: KL(Gamma(a,b) || Exponential(r)) =
        log(b) - log(r) + lgamma(a) + (1 - a) * digamma(a) + a * b / r
    eigvals : (q,)   Laplacian eigenvalues for active modes
    tau     : temperature controlling frequency penalty
    """
    a = torch.exp(log_a)                      # (B, q)
    b = torch.exp(log_b)                      # (B, q)
    r = tau * eigvals.unsqueeze(0).clamp(min=1e-6)   # (B, q)
    kl = (torch.lgamma(a) - torch.log(b) + torch.log(r)
          + (1 - a) * torch.digamma(a) + a * b / r)
    return kl.sum(-1).mean()
```

#### `wae/model.py` — `WiringAutoencoderV2`

```python
class WiringAutoencoderV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = WiringEncoderV2(cfg)          # Lap-precision KL
        self.spectral_decoder = SpectralLoadingDecoder(cfg.q, cfg.d, cfg.n_heads)
        self.diff_decoder = DiffusionDecoder(cfg)    # unchanged
        self.tau = nn.Parameter(torch.tensor(cfg.tau_init))  # learnable

    def forward(self, x, L_base, U_q, eigvals_q, E):
        z, mu, log_var = self.encoder(x, L_base)
        W, omega, S = self.spectral_decoder(z, U_q)
        L_z = DifferentiableLaplacian.from_spectral_loading(W, L_base)
        x_hat = self.diff_decoder(L_z, E)

        # --- losses ---
        recon = gaussian_nll(x, x_hat, self.diff_decoder.log_sigma)
        kl_z  = laplacian_precision_kl(mu, log_var, L_base, beta=cfg.beta_lap)
        kl_S  = spectral_basis_kl(S, eigvals_q, lam_s=cfg.lambda_s)
        kl_w  = tau_mode_kl(self.encoder.log_a, self.encoder.log_b,
                            eigvals_q, tau=self.tau)
        loss = recon + cfg.beta * kl_z + cfg.gamma * kl_S + cfg.alpha * kl_w
        return loss, recon, kl_z, kl_S, kl_w, x_hat, L_z, z, mu, log_var

    def extract_spectral_artefact(self, dataset_loader, U_q, eigvals_q):
        """Post-training: assemble the spectral artefact A(I)."""
        W_mean = self.spectral_decoder.S_net.weight @ U_q.T  # approx mean loading
        omega_mean = torch.exp(self.encoder.omega_net.bias)   # prior mode weights
        S_memory = sum(
            omega_mean[k] *
            torch.outer(self.diff_decoder.decode_mode(U_q[:, k]),
                        U_q[:, k])
            for k in range(U_q.size(1))
        )
        return {"W_hat": W_mean, "omega_hat": omega_mean,
                "S_memory": S_memory, "eigvals": eigvals_q}
```

#### New module: `wae/spectral_memory.py` — `SpectralAssociativeMemory`

```python
class SpectralAssociativeMemory(nn.Module):
    """
    Wraps a pre-built Hopfield/linear associative memory initialised from
    the spectral artefact A(I).  Can be used to initialise transformer
    FFN layers or cross-attention value matrices.
    """
    def __init__(self, artefact: dict, d_model: int, beta_H: float = 1.0):
        super().__init__()
        S_init = artefact["S_memory"]          # (d_v, d_k)
        self.S = nn.Parameter(S_init)
        self.beta_H = beta_H
        self.d_model = d_model

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Hopfield retrieval: softmax-weighted sum of spectral values.
        query : (B, T, d_k)
        """
        logits = self.beta_H * (query @ self.S.T)   # (B, T, d_v)
        weights = torch.softmax(logits, dim=-1)
        return weights @ self.S                      # (B, T, d_v)

    def delta_update(self, key: torch.Tensor, value: torch.Tensor):
        """
        Online delta-rule write: update S without overwriting spectral structure.
        key, value : (B, d_k), (B, d_v)
        """
        with torch.no_grad():
            residual = value - self.forward(key.unsqueeze(1)).squeeze(1)
            self.S.data += (residual.T @ key) / key.size(0)

    @classmethod
    def from_wae(cls, wae_v2: WiringAutoencoderV2,
                 loader, U_q, eigvals_q, d_model, **kw):
        artefact = wae_v2.extract_spectral_artefact(loader, U_q, eigvals_q)
        return cls(artefact, d_model, **kw)
```

### 3.4 Integration with Transformer (Spectral Memory Transformer)

```
┌──────────────────────────────────────────────────────┐
│  PHASE 1 — OFFLINE (WAE v2 training)                 │
│                                                      │
│  ArrowSpace index I  ──►  L(I), U_q, Λ_q            │
│      │                          │                    │
│      ▼                          ▼                    │
│  WiringAutoencoderV2.train()                         │
│      ELBO = recon - KL_Lap(z) - KL_S - KL_tau       │
│                                                      │
│  ──► spectral artefact A(I):                         │
│       W_hat = U_q diag(ω̂) Ŝ                        │
│       S_memory = Σ_k ω̂_k * v_k * u_k^T            │
└────────────────────────────┬─────────────────────────┘
                             │
                  SpectralAssociativeMemory(A(I))
                             │
┌────────────────────────────▼─────────────────────────┐
│  PHASE 2 — ONLINE (Spectral Memory Transformer)       │
│                                                      │
│  Transformer layer:                                  │
│    Self-attention  ──  dynamic, short-term           │
│    FFN / cross-attn ─  initialised from S_memory     │
│                        (long-term spectral prior)    │
│                                                      │
│  Optional delta-rule updates write new               │
│  associations without corrupting spectral structure  │
└──────────────────────────────────────────────────────┘
```

***

## 4. Extended ELBO Comparison

| Term | Existing WAE | WAE v2 |
|---|---|---|
| Reconstruction | \(-\|x - \hat{x}\|^2 / 2\sigma^2\) | Same; \(\sigma\) learnable via `log_sigma` |
| Latent KL | \(\mathrm{KL}(q(z)\|\mathcal{N}(0,I))\) — isotropic | \(\mathrm{KL}(q(z)\|p_\text{Lap}(z))\) — Laplacian-precision; adds Dirichlet energy |
| Spectral basis KL | None | \(\mathrm{KL}(q(S)\|p(S\mid\mathcal{I}))\) — eigenvalue-weighted Gaussian shrinkage |
| \(\tau\)-mode term | \(\alpha\cdot J_\text{freq} = \alpha\sum_{j>k}\lambda_j\) hard penalty | \(\mathrm{KL}(q(\omega)\|p(\omega\mid\tau,\Lambda))\) — soft variational frequency prior |
| Density matrix | Not in ELBO; in stability diagnostics | Option: add \(\mu_2\|\varrho_t\|_F^2\) occupancy penalty (from Option 6) |

***

## 5. Relationship to WAE Option Tracks

WAE v2 as specified corresponds most directly to **Option 4 (Variational Laplace AE)** from `03-branching.md`, with the key difference that WAE v2 uses full reparametrisation (Monte Carlo ELBO) rather than Laplace approximation. However, the Spectral-PPCA ELBO is also compatible with Option 4's Laplace covariance interpretation: the posterior precision \((H_\text{recon} + \Lambda_m)\) in Option 4 corresponds exactly to the combined Laplacian-precision plus spectral-shrinkage terms in the WAE v2 KL.

The associative memory component is new across all six options and is most naturally layered on top of **Option 1** (Deterministic AE) or **Option 4** (Variational Laplace AE) as a post-training export step. It also connects to **Option 6** (Spectrally Regularised Classifier / Reasoner): the signed density matrix \(\varrho_t\) produced during VDT inference can be used to initialise `S_memory` rather than the VAE loading posterior, providing a reasoning-grounded associative prior.

| Branching option | WAE v2 compatibility |
|---|---|
| Option 1 — Deterministic AE | WAE v2 adds probabilistic latent on top; backward-compatible as deterministic limit \(\beta\to 0\) |
| Option 2 — Energy-based | Spectral-basis loading can replace unconstrained proposer; Laplacian energy terms compatible |
| Option 3 — Latent diffusion | WAE v2 provides Stage 1 encoder; spectral noise schedule in Option 3 matches \(\Lambda_m^{-1}\) latent prior |
| Option 4 — Variational Laplace | WAE v2 is the full MC-ELBO version of Option 4; Laplace covariance is the deterministic approximation |
| Option 5 — Forecasting | Spectral artefact can be used as prior memory for the PDE-inspired predictor; CFL conditions unchanged |
| Option 6 — Classifier / Reasoner | \(\varrho_t\) as spectral artefact → associative memory init; depth-supervised CE loss unchanged |

***

## 6. Stability Considerations for WAE v2

The full stability hierarchy from `04-stability.md` applies unchanged. Two additional considerations for WAE v2:

### 6.1 Sample Laplacian \(L_s\) stability

\(L_s\) is built on-the-fly from the batch of latent means \(\{\mu_{x,n}\}\). This introduces a feedback loop: latent means are shaped by the KL that uses \(L_s\), and \(L_s\) is built from those means. To prevent degenerate attractors:

- Use a **stop-gradient** on the \(L_s\) construction: `L_s = build_knn_laplacian(mu_z.detach())`.
- Alternatively, use a **frozen** sample Laplacian built from the base encoder embeddings (pre-WAE) as a fixed structural prior, updated only every \(T_\text{refresh}\) epochs.

### 6.2 Spectral-basis KL conditioning

The spectral-basis KL has precision proportional to \(\Lambda_{1:q}\). If \(\lambda_1 \approx 0\) (Fiedler vector near zero), the prior on the first mode becomes nearly flat. This is correct behaviour (the DC / near-constant mode should not be penalised), but it can cause gradient variance. Apply a small floor: \(\lambda_k \leftarrow \max(\lambda_k, \epsilon)\) for \(\epsilon = 10^{-3}\) in the KL computation only.

### 6.3 Mode weight positivity and KL collapse

The Gamma/Log-Normal mode weight prior can collapse to a point mass at zero for high-frequency modes if \(\tau\lambda_k\) is large. This is the desired behaviour (mode selection), but monitor that at least \(q_\text{min}\) modes remain active (i.e., \(\mathbb{E}[\omega_k] > \delta\) for some minimum count). Add a soft constraint: \(\sum_k \mathbb{E}[\omega_k] \ge q_\text{min}\) via a Lagrange multiplier or a floor on the Gamma shape parameter \(a_k \ge a_\text{min}\).

***

## 7. New Metrics for WAE v2

In addition to the existing benchmark metrics, WAE v2 reports:

| Metric | How computed | What it measures |
|---|---|---|
| Laplacian KL | \(\mathrm{KL}(q(z)\|p_\text{Lap})\) per batch | Degree of latent smoothness w.r.t. sample graph |
| Spectral basis KL | \(\mathrm{KL}(q(S)\|p(S\mid\mathcal{I}))\) | Spectral alignment of loadings with index geometry |
| \(\tau\)-mode KL | \(\mathrm{KL}(q(\omega)\|p(\omega\mid\tau,\Lambda))\) | Effective frequency band selection |
| Active mode count | \(\sum_k \mathbf{1}[\mathbb{E}[\omega_k] > \delta]\) | Number of Laplacian modes contributing to \(W\) |
| Memory retrieval SNR | \(d_k / N_\text{stored}\) via spectral key orthogonality | Quality of pre-built associative memory |
| ELBO Bayes factor | \(\exp(\mathcal{L}(\mathcal{I}_1) - \mathcal{L}(\mathcal{I}_2))\) | Comparison of competing ArrowSpace indices |
| Spectral artefact norm | \(\|\hat{W}\|_F\) | Stability of post-training artefact export |

***

## 8. Implementation Sequence

Following the recommended order from `docs/README.md`, the WAE v2 upgrades should be applied in this sequence:

1. **Swap `kl_loss` in `WiringEncoder`** to use the modal prior \(\mathcal{N}(0, \Lambda_m^{-1})\) (already named in README concept table; minimal code change).
2. **Add `tau_mode_kl`** to replace hard \(J_\text{freq}\) penalty. Keep \(\alpha\cdot J_\text{freq}\) as an ablation flag.
3. **Introduce `SpectralLoadingDecoder`** as an optional drop-in for `WiringDecoder` (controlled by config flag `use_spectral_loading: bool`). Validate that reconstruction quality matches or exceeds the MoE decoder on a held-out set before making it default.
4. **Add sample-graph Laplacian construction** in the encoder forward pass (stop-gradient); enable Laplacian-precision KL. Monitor latent smoothness KL convergence.
5. **Add `extract_spectral_artefact()`** and `SpectralAssociativeMemory` as a post-training export utility. Test retrieval SNR on a toy associative recall task.
6. **Integrate `SpectralAssociativeMemory` into VDT / transformer** as FFN initialiser. Run Option 6 evaluation protocol with spectral memory enabled vs disabled.

This sequence ensures each upgrade is independently validated before the next is added, and is backward-compatible with the current six-option branching structure.