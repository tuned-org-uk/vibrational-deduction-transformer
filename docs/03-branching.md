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

**Decoder** — a transformer (possibly weight-shared with the encoder, as in masked autoencoders)
that takes \(z\) (re-expanded to \(\mathbb{R}^{n \times d}\) via \(U_m^\top\)) and reconstructs
\(\hat{X}_0\):

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
wave dynamics relax them toward the energy minimum. This is an **energy-based model (EBM)**
with analytically grounded energy.

### Architecture

1. **Proposer network** produces \(Q_0 = f_\phi(X_0)\).
2. **Energy relaxation** via \(K\) preconditioned gradient steps:
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
        self.K = K
        self.S = S_sigma_M
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

Train a diffusion process in the VDT’s modal latent space. The Laplacian eigenvalues provide
a principled, **mode-dependent noise schedule**: high-frequency modes (large \(\lambda_k\)) are
corrupted earlier in the forward process; low-frequency modes persist longer.

### Architecture

- **Stage 1**: train vibrational autoencoder (Option 1) to produce \(z \in \mathbb{R}^m\).
- **Stage 2**: train denoiser \(\epsilon_\theta(z_\tau, \tau)\) over the modal latent space.

### Spectral Noise Schedule

\[
p(z_\tau \mid z_0) = \mathcal{N}(\sqrt{\bar\alpha_\tau}\, z_0,\; (1 - \bar\alpha_\tau)\, \Lambda_m^{-1}),
\]

where \(\Lambda_m = \mathrm{diag}(\lambda_1, \dots, \lambda_m)\).

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

Instead of a standard VAE with an amortised posterior, one can use the **Laplace approximation**:
run the vibrational dynamics to find the *mode* of the posterior over latents, then approximate
the posterior as a Gaussian with covariance derived from the local curvature (the preconditioned
Hessian). This is the **Variational Laplace Autoencoder** strategy, but where the Hessian is
already structured by \(L_f\) and \(M\), giving a principled closed-form covariance estimate
without Monte Carlo sampling.

### Architecture

**Finding the posterior mode**: given input \(X_0\), treat the modal latent \(z\) as the
variable to be inferred. Define:

\[
E(z \mid X_0) = -\log p(X_0 \mid z) + \tfrac{1}{2} z^\top \Lambda_m z,
\]

where \(-\log p(X_0 \mid z)\) is the reconstruction loss given latent \(z\), and the second
term is the log-prior \(p(z) = \mathcal{N}(0, \Lambda_m^{-1})\) induced by the Laplacian
eigenvalues. The wave dynamics (or Laplacian-preconditioned GD) find the mode:

\[
\hat z = \arg\min_z E(z \mid X_0).
\]

**Laplace covariance**: at the mode, the Hessian of \(E\) is:

\[
H = \nabla^2_z E(\hat z \mid X_0) = H_{\text{recon}} + \Lambda_m,
\]

where \(H_{\text{recon}}\) is the Hessian of the reconstruction loss w.r.t. \(z\). The posterior
approximation is:

\[
q(z \mid X_0) = \mathcal{N}(\hat z, H^{-1}).
\]

Because \(H_{\text{recon}} + \Lambda_m\) is the same preconditioned Hessian structure as
\(S_{\sigma,M}\) from Section 5 of the paper, the inversion is cheap via the mass-aware
resolvent \(S_{\sigma,M} = (M + \sigma L_f)^{-1} M\).

### Training Objective (Laplace ELBO)

\[
\mathcal{L} = \underbrace{-\log p(X_0 \mid \hat z)}_{\text{reconstruction at mode}}
+ \underbrace{\tfrac{1}{2}\hat z^\top \Lambda_m \hat z}_{\text{prior energy}}
- \underbrace{\tfrac{1}{2}\log |H^{-1}|}_{\text{log Laplace covariance}}
\]

The third term penalises posterior collapse: a very sharp Hessian (overconfident posterior)
is penalised. In practice, with diagonal \(H \approx \mathrm{diag}(h_k)\), this simplifies to
\(-\frac{1}{2}\sum_k \log h_k^{-1}\).

### Connection to the VDT paper

The wave dynamics already seek the energy minimum (Section 7.1 frames learning as
\(M\ddot u + C\dot u + Hu = 0\)). In the Laplace-AE reading, this is exactly the optimisation
that finds \(\hat z\). The spectral modal decoupling (Section 8.5) means each mode \(k\)
has an independent scalar optimisation problem, so the Laplace approximation is exact if
\(H_{\text{recon}}\) is diagonal in the eigenbasis.

### PyTorch Sketch

```python
class LaplaceVibrationalAE(nn.Module):
    def __init__(self, d, m, K, lambda_max):
        super().__init__()
        self.vdt_encoder = VDT(d, m, K, lambda_max=lambda_max)
        self.decoder = nn.Sequential(nn.Linear(m, d), nn.SiLU(), nn.Linear(d, d))

    def encode_mode(self, X0, L_f, eigvecs, P_f):
        Q_K, _, _ = self.vdt_encoder(X0, L_f, eigvecs, P_f)
        U_m = eigvecs[:, :self.vdt_encoder.m]
        return (Q_K @ U_m).mean(dim=1)  # (B, m): posterior mode z_hat

    def forward(self, X0, L_f, eigvecs, P_f, eigvals_m):
        z_hat = self.encode_mode(X0, L_f, eigvecs, P_f)
        X_hat = self.decoder(z_hat).unsqueeze(1).expand_as(X0)
        recon = (X0 - X_hat).pow(2).mean()
        prior = 0.5 * (z_hat.pow(2) * eigvals_m.unsqueeze(0)).sum(-1).mean()
        # Diagonal Laplace covariance approximation via autograd
        recon.backward(retain_graph=True)
        h_diag = torch.zeros_like(z_hat)
        for k in range(z_hat.size(-1)):
            g = torch.autograd.grad(recon, z_hat, create_graph=False,
                                    retain_graph=True)[0][:, k]
            h_diag[:, k] = g  # first-order diagonal approx
        log_cov = -0.5 * torch.log(
            h_diag.detach() + eigvals_m.unsqueeze(0) + 1e-8).mean()
        return recon + prior + log_cov
```

### What this gives you

- A Bayesian autoencoder with **analytically structured uncertainty** tied to the Laplacian spectrum.
- No Monte Carlo sampling or reparameterisation: the posterior is computed via the wave dynamics
  and the Hessian inversion is cheap thanks to the preconditioner.
- Naturally extends to uncertainty quantification: at test time, \(H^{-1}\) gives per-mode
  confidence intervals.

---

## Option 5 — Vibrational Graph Forecasting (PDE-Inspired Predictor)

### Motivation

Without any generative ambitions, one can treat the VDT purely as a **learned PDE solver**:
the vibrational dynamics approximate the solution of a damped wave equation on the feature
graph, and the learning objective is to forecast future states or predict outputs of a
system governed by that PDE. This fits naturally into graph-based time series, physics simulation,
or multi-step reasoning without requiring a latent variable model.

### Architecture

Given an initial condition \(X^{(0)} \in \mathbb{R}^{n \times d}\) (e.g. the current state of a
graph-structured system), the VDT evolves it:

\[
X^{(t+1)} = \Phi_L\bigl(X^{(t)}, Q_{t-1}, Q_t\bigr) + \mathrm{TransformerBlock}(X^{(t)}),
\]

and the learning target is the true state \(X^{(T)}\) or a label derived from it.

**Boundary conditions / forcing**: external inputs (boundary conditions, forcing terms) are
injected as \(B_t = \mathrm{Linear}(H_t)\) exactly as in Equation (22) of the paper.
This corresponds to a driven wave equation:

\[
M\ddot Q + C\dot Q + L_f Q = B_t.
\]

### Training Objective

\[
\mathcal{L} = \|X^{(K)} - X^{(T)}\|_F^2
+ \alpha \sum_{t=1}^K \mathrm{tr}(Q_t^\top L_f Q_t)
+ \beta \, \Delta t^2 \lambda_{\max}(L_f)  \quad (\text{CFL stability penalty})
\]

The last term discourages the learnable time-step from violating the CFL stability condition
\(\Delta t^2 \lambda_{\max}(L_f) \leq 2\), turning the stability constraint into a soft
regulariser.

### Suitable tasks

| Task | Input \(X^{(0)}\) | Target \(X^{(T)}\) |
|------|-------------------|--------------------|
| Graph time series | current node features | future node features |
| Multi-step reasoning | problem encoding | solution state |
| PDE on graphs | initial condition | evolved state |
| Iterative refinement | noisy observation | denoised observation |

### PyTorch Sketch

```python
class VibForecaster(nn.Module):
    def __init__(self, d, K, n_heads=4, lambda_max=1.0):
        super().__init__()
        self.vdt = VDT(d, m_modes=d // 4, K=K, n_heads=n_heads, lambda_max=lambda_max)
        self.head = nn.Linear(d, d)

    def forward(self, X0, L_f, eigvecs, P_f):
        Q_K, _, _ = self.vdt(X0, L_f, eigvecs, P_f)
        return self.head(Q_K)  # (B, n, d) predicted future state

def forecasting_loss(pred, target, Q_states, L_f, alpha=0.001, dt=0.1):
    mse = (pred - target).pow(2).mean()
    smooth = alpha * sum(
        torch.trace(Q[0].T @ (L_f @ Q[0])) for Q in Q_states
    ) / len(Q_states)
    cfl = max(0, dt**2 * torch.linalg.eigvalsh(L_f).max().item() - 2.0)
    return mse + smooth + 0.01 * cfl
```

### What this gives you

- The clearest mapping from the VDT paper to a runnable task: no new probabilistic machinery
  needed, just a prediction head and an MSE loss.
- Directly tests hypothesis (ii) from Section 11.3 of the paper: *VDT generalises to longer
  chains better than a plain recurrent transformer*.
- Physical interpretability is maximal: training can be monitored via modal energy spectra
  \(E_t^+ = \mathrm{Tr}(\Lambda_m \varrho_t^+)\) as the dynamics evolve.
