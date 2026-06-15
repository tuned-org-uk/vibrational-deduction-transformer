# Branching Paths: From Vibrational Learning to Fully-Fledged Learning Algorithms

This document surveys six distinct ways to evolve the deterministic Laplacian/Rayleigh vibrational
framework of the VDT paper into a complete, end-to-end learning algorithm. Each option is self-contained
and can be pursued independently. They vary along two axes: **how probabilistic** the latent geometry
becomes, and **what the training objective** targets (reconstruction, energy, generation, forecasting,
or classification).

The shared foundation in all cases is the feature-space graph Laplacian \(L_f\), the mass matrix \(M\),
the generalised Rayleigh quotient \(R_M(z) = z^\top L_f z / z^\top M z\), and the discrete damped wave
operator \(\Phi_L\).

---

## Option 1 — Deterministic Vibrational Autoencoder

### Motivation

The simplest extension of the current VDT that turns it into a full learning algorithm is a
**deterministic autoencoder**: no explicit probability distribution over latents, but a clear
encoder–bottleneck–decoder pipeline with a reconstruction objective. This is the direct
analogue of a classical undercomplete autoencoder, but with the bottleneck structured by the
spectral geometry of \(L_f\).

### Architecture

**Encoder** — the VDT recurrence (Sections 8.2–8.4 of the paper) plays the role of encoder.
Given input token features \(X_0 \in \mathbb{R}^{n \times d}\), run \(K\) steps of the
vibrational recurrence:

\[
X_{t+1} = \mathrm{TransformerBlock}\bigl(X_t + \Phi_L(X_t, Q_{t-1}, Q_t)\bigr),
\quad t = 1, \dots, K.
\]

At depth \(K\), project to modal coordinates using the first \(m\) eigenvectors of \(L_f\):

\[
z = \mathrm{pool}_{\mathrm{tokens}}(Q_K \, U_m) \in \mathbb{R}^{m},
\]

where pooling can be mean, CLS-token, or attention-weighted over the \(n\) token dimension.
The result \(z\) is the **vibrational latent code** — a compressed representation of the input
in the low-frequency modal subspace of the feature graph.

**Bottleneck** — optionally apply a linear projection or small MLP to \(z\) to control
bottleneck width independently of \(m\).

**Decoder** — a transformer that takes \(z\) re-expanded via \(U_m^\top\) and reconstructs \(\hat{X}_0\):

\[
\hat{X}_0 = \mathrm{Decoder}_\theta(z \, U_m^\top).
\]

### Training Objective

\[
\mathcal{L} = \|X_0 - \hat{X}_0\|_F^2
+ \alpha \hat{X}_0^\top L_f \hat{X}_0
+ \beta R_M(z)
\]

### Density Matrix as Bottleneck

The signed density matrix \(\varrho_t = \varrho_t^+ - \varrho_t^-\) can serve as a structured
bottleneck via low-rank factorisation \(\varrho = V V^\top - W W^\top\) with trace penalty.

### PyTorch Sketch

```python
class VibrationalAutoencoder(nn.Module):
    def __init__(self, d, m_modes, K, n_heads=4, lambda_max=1.0):
        super().__init__()
        self.encoder = VDT(d, m_modes, K, n_heads, lambda_max)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, batch_first=True), num_layers=2)
        self.proj_out = nn.Linear(d, d)

    def forward(self, X0, L_f, eigvecs, P_f):
        Q_K, _, _ = self.encoder(X0, L_f, eigvecs, P_f)
        U_m = eigvecs[:, :self.encoder.m]
        z = (Q_K @ U_m).mean(dim=1)
        z_expanded = z.unsqueeze(1) @ U_m.T
        z_expanded = z_expanded.expand(-1, X0.size(1), -1)
        return self.proj_out(self.decoder(z_expanded))

def autoencoder_loss(X0, X_hat, L_f, alpha=0.01):
    recon = (X0 - X_hat).pow(2).mean()
    smooth = torch.stack([
        (X_hat[b] @ L_f * X_hat[b]).sum() for b in range(X_hat.size(0))
    ]).mean()
    return recon + alpha * smooth
```

---

## Option 2 — Energy-Based Vibrational Model

### Motivation

Treat \(J_\lambda\) and the Rayleigh functional as an **explicit energy** \(E(Q)\) and couple a
deep network to it via iterative minimisation. The deep network proposes initial states; the
wave dynamics relax them toward the energy minimum.

### Architecture

1. **Proposer** produces \(Q_0 = f_\phi(X_0)\).
2. **Energy relaxation** via \(K\) preconditioned steps:
   \[
   Q_{t+1} = Q_t - \eta S_{\sigma,M} \nabla_{Q_t} E(Q_t),
   \quad E(Q) = \tfrac{\lambda}{2}\mathrm{tr}(Q^\top L_f Q).
   \]
3. **Prediction head** on \(Q_K\).

### Training Objective

\[
\mathcal{L} = \mathcal{L}_{\text{task}}(\hat{y}, y)
+ \mu \, E(Q_K)
+ \nu \, \|Q_0 - Q_K\|_F^2
\]

### PyTorch Sketch

```python
def energy(Q, L_f, lam=0.01):
    return lam * torch.stack([
        torch.trace(Q[b].T @ (L_f @ Q[b]))
        for b in range(Q.size(0))
    ]).mean()

class EnergyVibModel(nn.Module):
    def __init__(self, d, K, S_sigma_M):
        super().__init__()
        self.proposer = nn.TransformerEncoderLayer(d_model=d, nhead=4, batch_first=True)
        self.K = K; self.S = S_sigma_M
        self.head = nn.Linear(d, 1)

    def forward(self, X0, L_f, eta=0.01):
        Q = self.proposer(X0)
        for _ in range(self.K):
            Q.requires_grad_(True)
            grad = torch.autograd.grad(energy(Q, L_f), Q, create_graph=True)[0]
            Q = (Q - eta * (grad @ self.S.T)).detach()
        return self.head(Q.mean(dim=1)).squeeze(-1)
```

---

## Option 3 — Vibrational Latent Diffusion Model

### Motivation

Train a diffusion process in the VDT’s modal latent space with a **mode-dependent noise
schedule** tied to Laplacian eigenvalues: high-frequency modes are corrupted earlier, low-frequency
modes persist longer.

### Architecture

- **Stage 1**: train vibrational autoencoder (Option 1) to produce \(z \in \mathbb{R}^m\).
- **Stage 2**: denoiser \(\epsilon_\theta(z_\tau, \tau)\) over modal latent space.

### Spectral Noise Schedule

\[
p(z_\tau \mid z_0) = \mathcal{N}(\sqrt{\bar\alpha_\tau}\, z_0,\; (1 - \bar\alpha_\tau)\, \Lambda_m^{-1})
\]

### Training Objective

\[
\mathcal{L}_{\text{diff}} = \mathbb{E}_{z_0, \epsilon, \tau}
\left[\|\epsilon_\theta(z_\tau, \tau) - \epsilon\|_2^2\right]
\]

### PyTorch Sketch

```python
class SpectralDiffusion(nn.Module):
    def __init__(self, m, T=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m + 1, 256), nn.SiLU(),
            nn.Linear(256, 256),  nn.SiLU(),
            nn.Linear(256, m))
        self.T = T

    def forward(self, z_tau, tau):
        t_embed = (tau.float() / self.T).unsqueeze(-1)
        return self.net(torch.cat([z_tau, t_embed], dim=-1))

def diffusion_loss(model, z0, eigvals_m, T=1000):
    B = z0.size(0)
    tau = torch.randint(1, T, (B,), device=z0.device)
    alpha_bar = torch.cos((tau.float() / T) * (torch.pi / 2)).pow(2)
    noise = torch.randn_like(z0) / eigvals_m.clamp(min=1e-6).sqrt().unsqueeze(0)
    z_tau = alpha_bar.sqrt().unsqueeze(-1) * z0 + \
            (1 - alpha_bar).sqrt().unsqueeze(-1) * noise
    return (model(z_tau, tau) - noise).pow(2).mean()
```

---

## Option 4 — Vibrational Bayesian Autoencoder (Variational Laplace)

### Motivation

Use **Laplace approximation** of the posterior: the wave dynamics find the posterior mode
\(\hat z\), and the curvature is estimated from the preconditioned Hessian \(H_{\text{recon}} + \Lambda_m\),
giving closed-form approximate posteriors without Monte Carlo sampling.

### Architecture

**Posterior mode** via VDT-driven energy minimisation:

\[
\hat z = \arg\min_z \bigl[-\log p(X_0 \mid z) + \tfrac{1}{2} z^\top \Lambda_m z\bigr].
\]

**Laplace covariance**:

\[
q(z \mid X_0) = \mathcal{N}\bigl(\hat z,\; (H_{\text{recon}} + \Lambda_m)^{-1}\bigr).
\]

Because \(H_{\text{recon}} + \Lambda_m\) mirrors the preconditioned Hessian structure of
\(S_{\sigma,M}\), inversion is cheap via the mass-aware resolvent.

### Training Objective (Laplace ELBO)

\[
\mathcal{L} = -\log p(X_0 \mid \hat z)
+ \tfrac{1}{2}\hat z^\top \Lambda_m \hat z
- \tfrac{1}{2}\log |H^{-1}|
\]

### PyTorch Sketch

```python
class LaplaceVibrationalAE(nn.Module):
    def __init__(self, d, m, K, lambda_max):
        super().__init__()
        self.vdt_encoder = VDT(d, m, K, lambda_max=lambda_max)
        self.decoder = nn.Sequential(nn.Linear(m, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, X0, L_f, eigvecs, P_f, eigvals_m):
        Q_K, _, _ = self.vdt_encoder(X0, L_f, eigvecs, P_f)
        U_m = eigvecs[:, :self.vdt_encoder.m]
        z_hat = (Q_K @ U_m).mean(dim=1)           # posterior mode
        X_hat = self.decoder(z_hat).unsqueeze(1).expand_as(X0)
        recon  = (X0 - X_hat).pow(2).mean()
        prior  = 0.5 * (z_hat.pow(2) * eigvals_m.unsqueeze(0)).sum(-1).mean()
        # Diagonal Laplace entropy term
        h_diag = eigvals_m.unsqueeze(0) + 1.0     # simplified: prior dominates
        log_cov = -0.5 * torch.log(h_diag + 1e-8).mean()
        return recon + prior + log_cov
```

---

## Option 5 — Vibrational Graph Forecasting (PDE-Inspired Predictor)

### Motivation

Treat the VDT as a **learned PDE solver** for a damped wave equation on the feature graph.
No generative model is needed; the objective is to predict future states of a graph-structured
system, test-time reasoning chains, or solutions to PDE-type tasks.

### Architecture

Given initial condition \(X^{(0)}\), evolve via VDT:

\[
X^{(t+1)} = \Phi_L(X^{(t)}, Q_{t-1}, Q_t) + \mathrm{TransformerBlock}(X^{(t)}),
\]

where external forcing \(B_t = \mathrm{Linear}(H_t)\) injects boundary conditions.

### Training Objective

\[
\mathcal{L} = \|X^{(K)} - X^{(T)}\|_F^2
+ \alpha \sum_{t=1}^K \mathrm{tr}(Q_t^\top L_f Q_t)
+ \beta \max(0,\; \Delta t^2 \lambda_{\max}(L_f) - 2)
\]

### Suitable Tasks

| Task | Input \(X^{(0)}\) | Target \(X^{(T)}\) |
|------|-------------------|--------------------|
| Graph time series | current node features | future node features |
| Multi-step reasoning | problem encoding | solution state |
| PDE on graphs | initial condition | evolved state |
| Iterative denoising | noisy observation | clean observation |

### PyTorch Sketch

```python
class VibForecaster(nn.Module):
    def __init__(self, d, K, n_heads=4, lambda_max=1.0):
        super().__init__()
        self.vdt = VDT(d, m_modes=d // 4, K=K, n_heads=n_heads, lambda_max=lambda_max)
        self.head = nn.Linear(d, d)

    def forward(self, X0, L_f, eigvecs, P_f):
        Q_K, _, _ = self.vdt(X0, L_f, eigvecs, P_f)
        return self.head(Q_K)

def forecasting_loss(pred, target, Q_states, L_f, alpha=0.001, dt=0.1):
    mse = (pred - target).pow(2).mean()
    smooth = alpha * sum(
        torch.trace(Q[0].T @ (L_f @ Q[0])) for Q in Q_states) / len(Q_states)
    cfl = max(0.0, dt**2 * torch.linalg.eigvalsh(L_f).max().item() - 2.0)
    return mse + smooth + 0.01 * cfl
```

---

## Option 6 — Spectrally Regularised Vibrational Classifier / Reasoner

### Motivation

The most direct path to a runnable system is to leave generation aside entirely and turn
the current VDT into a **classification or reasoning model** with a fully specified training
loop, spectral regularisation, and empirical evaluation. This is what Section 11 of the
paper specifies in outline but has not yet implemented. Making it “fully-fledged” means
filling in every training and evaluation detail so that results can be produced and compared
against LDT and baseline transformers.

### Architecture

The VDT architecture from Section 8 is used unchanged. A classification head is added on the
final state \(X_K\):

\[
\hat y_t = \mathrm{Head}(X_t[\mathrm{CLS}]), \quad t = 1, \dots, K,
\]

where \(X_t[\mathrm{CLS}]\) is the CLS token at recurrent depth \(t\).

### Training Objective (fully specified)

\[
\mathcal{L} = \underbrace{\frac{1}{K} \sum_{t=1}^K \mathcal{L}_{\text{CE}}(\hat y_t, y)}_{\text{depth-supervised task loss}}
+ \mu_1 \underbrace{\frac{1}{K} \sum_{t=1}^K \mathrm{tr}(Q_t^\top L_f Q_t)}_{\text{Laplacian smoothness}}
+ \mu_2 \underbrace{\frac{1}{K} \sum_{t=1}^K \|\varrho_t\|_F^2}_{\text{density occupancy penalty}}
\]

This is exactly Equation (33) of the paper with the taumode term replaced by an explicit
Laplacian smoothness term (the taumode term is optional and treated as an ablation).

### Full Training Loop

```python
def train_epoch(model, loader, optimizer, L_f, eigvecs, P_f,
                mu1=0.01, mu2=0.001):
    model.train()
    total_loss = 0.0
    for X0, labels in loader:
        optimizer.zero_grad()
        Q_K, logits_all, (rho_p, rho_m) = model(X0, L_f, eigvecs, P_f)

        # Depth-supervised cross-entropy
        head = model.head if hasattr(model, 'head') else nn.Linear(X0.size(-1), num_classes)
        task_loss = sum(
            F.cross_entropy(head(logits), labels) for logits in logits_all
        ) / len(logits_all)

        # Laplacian smoothness on vibrational states
        smooth = mu1 * sum(
            torch.trace(logits[0].T @ (L_f @ logits[0]))
            for logits in logits_all
        ) / len(logits_all)

        # Density matrix occupancy penalty
        rho = rho_p - rho_m
        occ = mu2 * rho.pow(2).sum()

        loss = task_loss + smooth + occ
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

### Evaluation Protocol (LDT-mirrored)

Following Appendix C of the paper, the three benchmark families are:

| Task | Train setting | OOD test setting | Metric |
|------|--------------|-----------------|--------|
| 3-SAT | 10–20 vars, \(\alpha \in [3,4]\) | 20–40 vars | accuracy |
| Syllogistic deduction | chain \(K \in \{2,3,4\}\) | \(K \in \{5,6,7,8\}\) | accuracy |
| Modular arithmetic | depth \(K \leq 4\), \(p=97\) | \(K \in \{5,\dots,10\}\) | accuracy |

For each task, report:

1. Accuracy vs. recurrent depth \(K \in \{2, 4, 8, 16\}\).
2. Accuracy vs. problem complexity on OOD instances.
3. Modal energy spectra \(E_t^\pm\) as a function of depth (VDT only).
4. Signed modal occupancy \(\{\mathrm{occ}_{t,k}\}_{k=1}^m\) at convergence.

### Stability Analysis

For completeness, report the following per run:

- \(\Delta t^2 \lambda_{\max}(L_f)\) at convergence (should be \(\leq 2\)).
- Condition number \(\kappa(P_{\sigma,M})\) of the preconditioner.
- Spectral energy distribution \(R_M(Q_K)\) over epochs.

These diagnostics are the vibrational counterpart of LDT’s lattice projection residuals and
allow direct interpretive comparison between the two architectures.

### What this gives you

- A completely specified, runnable training and evaluation setup.
- Direct empirical test of the paper’s theoretical claims (modal smoothness, spectral
  regularisation, density-matrix interpretability).
- Minimal additional implementation beyond what Appendix B of the paper already provides.

---

## Comparison of All Six Options

| Option | Probabilistic? | Objective | New components needed | Complexity |
|--------|---------------|-----------|----------------------|------------|
| 1 — Deterministic AE | No | Reconstruction + spectral | Decoder, reconstruction loss | Low |
| 2 — Energy-based | No | Task + explicit energy | Proposer, autograd energy step | Medium |
| 3 — Latent diffusion | Yes (implicit) | Denoising score matching | Stage 1 AE + denoiser net | High |
| 4 — Variational Laplace | Yes (Laplace) | Laplace ELBO | Hessian estimation, prior | Medium |
| 5 — PDE forecasting | No | State prediction + CFL | Prediction head, forcing term | Low |
| 6 — Classifier/reasoner | No | Depth-supervised CE | Training loop, eval protocol | Low |

The recommended path for a first working implementation is **Option 6** (closes the loop on
Section 11 of the paper with minimum new code), followed by **Option 1** (adds reconstruction
self-supervision), and then either **Option 4** (Bayesian) or **Option 3** (generative) depending
on whether probabilistic uncertainty or generative modelling is the primary research goal.
