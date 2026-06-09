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

The decoder can itself be vibrational: run a *reverse* wave update starting from \(z U_m^\top\)
and evolving toward a reconstruction, with the Laplacian acting as a smoothness constraint on
generated features.

### Training Objective

\[
\mathcal{L} = \underbrace{\|X_0 - \hat{X}_0\|_F^2}_{\text{reconstruction}}
+ \alpha \underbrace{\hat{X}_0^\top L_f \hat{X}_0}_{\text{Laplacian smoothness}}
+ \beta \underbrace{R_M(z)}_{\text{Rayleigh energy of latent}}
\]

The Laplacian smoothness term encourages the reconstruction to respect feature-graph structure.
The Rayleigh energy term penalises latent codes that live in high-frequency modes, implementing
a spectral bottleneck without any explicit probability.

For discrete-token tasks, replace the Frobenius reconstruction term with cross-entropy over the
vocabulary, and drop the Laplacian smoothness term on features (apply it instead to the decoder's
hidden states).

### Density Matrix as Bottleneck

The signed density matrix \(\varrho_t = \varrho_t^+ - \varrho_t^-\) from Section 9 of the paper
can serve as a **structured bottleneck** by:

- constraining its rank to \(r \ll m\) (low-rank factorisation \(\varrho = V V^\top - W W^\top\)),
- adding a trace penalty \(\lambda \mathrm{Tr}(|\varrho_K|)\) to the objective to encourage sparsity
  of modal occupancy.

This gives a structured, interpretable bottleneck without probabilistic machinery.

### PyTorch Sketch

```python
class VibrationalAutoencoder(nn.Module):
    def __init__(self, d, m_modes, K, n_heads=4, lambda_max=1.0):
        super().__init__()
        self.encoder = VDT(d, m_modes, K, n_heads, lambda_max)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, batch_first=True),
            num_layers=2
        )
        self.proj_out = nn.Linear(d, d)

    def forward(self, X0, L_f, eigvecs, P_f):
        Q_K, _, _ = self.encoder(X0, L_f, eigvecs, P_f)
        U_m = eigvecs[:, :self.encoder.m]
        z = (Q_K @ U_m).mean(dim=1)                # (B, m)
        z_expanded = z.unsqueeze(1) @ U_m.T        # (B, 1, d)
        z_expanded = z_expanded.expand(-1, X0.size(1), -1)
        X_hat = self.proj_out(self.decoder(z_expanded))
        return X_hat

def autoencoder_loss(X0, X_hat, L_f, alpha=0.01):
    recon = (X0 - X_hat).pow(2).mean()
    smooth = torch.stack([
        (X_hat[b] @ L_f * X_hat[b]).sum() for b in range(X_hat.size(0))
    ]).mean()
    return recon + alpha * smooth
```

### What this gives you

- A complete, runnable training loop with no probabilistic overhead.
- Modal latents that are interpretable via the Laplacian spectrum.
- A natural ablation baseline: remove the Laplacian smoothness term to recover a standard
  transformer autoencoder, and compare modal energy spectra.

---

## Option 2 — Energy-Based Vibrational Model

### Motivation

Instead of committing to a generative distribution, one can treat the Rayleigh functional and
the Laplacian-regularised cost \(J_\lambda\) as an **explicit energy** \(E(x)\) and couple a
deep network to that energy via iterative minimisation. This is an **energy-based model (EBM)**
where the energy is not learned from scratch but is derived analytically from \(L_f\) and \(M\).
The deep network’s role shifts: instead of directly predicting outputs, it proposes a good
initial condition for energy relaxation, and the wave dynamics perform the relaxation.

### Architecture

Given input \(X_0\), run:

1. **Proposer network** (e.g. one transformer layer) produces an initial vibrational state:
   \[
   Q_0 = f_\phi(X_0) \in \mathbb{R}^{n \times d}.
   \]

2. **Energy relaxation** via \(K\) steps of Laplacian-preconditioned gradient descent or the
   wave update \(\Phi_L\):
   \[
   Q_{t+1} = Q_t - \eta S_{\sigma,M} \nabla_{Q_t} E(Q_t),
   \]
   where \(E(Q) = \frac{1}{2n}\|A Q - B\|^2 + \frac{\lambda}{2}\mathrm{tr}(Q^\top L_f Q)\)
   and \(S_{\sigma,M} = (M + \sigma L_f)^{-1} M\) is the mass-aware resolvent.

3. **Prediction head** applied to the relaxed state \(Q_K\).

### Training Objective

The network is trained jointly on:

\[
\mathcal{L} = \mathcal{L}_{\text{task}}(\hat{y}, y)
+ \mu \, E(Q_K)
+ \nu \, \|Q_0 - Q_K\|_F^2,
\]

where \(\mathcal{L}_{\text{task}}\) is classification or regression loss, the second term encourages
the relaxed state to be near the energy minimum, and the third term (relaxation gap) regularises
how far the network’s proposal needs to travel.

### Connection to the VDT paper

Propositon 5.1 of the paper already gives a linear-convergence guarantee for the gradient step
under \(S_{\sigma,M}\), so the relaxation is provably well-behaved when the energy is quadratic.
For non-quadratic settings (deep networks, non-convex tasks), the energy descent is heuristic but
still benefits from the spectral geometry.

### PyTorch Sketch

```python
def energy(Q, L_f, lam=0.01):
    """Batch Laplacian energy: mean over batch."""
    # Q: (B, n, d)
    return lam * torch.stack([
        torch.trace(Q[b].T @ (L_f @ Q[b]))
        for b in range(Q.size(0))
    ]).mean()

class EnergyVibModel(nn.Module):
    def __init__(self, d, K, S_sigma_M):
        super().__init__()
        self.proposer = nn.TransformerEncoderLayer(d_model=d, nhead=4, batch_first=True)
        self.K = K
        self.S = S_sigma_M   # precomputed (d, d) preconditioner
        self.head = nn.Linear(d, 1)

    def forward(self, X0, L_f, eta=0.01):
        Q = self.proposer(X0)
        for _ in range(self.K):
            with torch.enable_grad():
                Q.requires_grad_(True)
                E = energy(Q, L_f)
                grad = torch.autograd.grad(E, Q, create_graph=True)[0]
            # Preconditioned gradient step: dQ per token
            Q = Q - eta * (grad @ self.S.T)
            Q = Q.detach().requires_grad_(False)
        return self.head(Q.mean(dim=1)).squeeze(-1)
```

### What this gives you

- An algorithm that is interpretable as **explicit energy minimisation** at inference time.
- Convergence can be analysed using the preconditioning theory of Section 5 in the paper.
- Suitable for semi-supervised and iterative-refinement settings.

---

## Option 3 — Vibrational Latent Diffusion Model

### Motivation

Diffusion models in latent spaces have become a dominant generative paradigm. Here, the VDT
encoder maps data into a spectrally structured latent space (the modal coordinates of \(L_f\)),
and a diffusion or flow-matching process is trained **within** that space. The Laplacian
eigenvectors serve as a fixed, meaningful basis for the noise and denoising process, biasing
generation toward spectrally smooth outputs.

### Architecture

**Stage 1 — Train vibrational autoencoder** (Option 1 above) to produce a well-behaved
modal latent space \(z \in \mathbb{R}^m\).

**Stage 2 — Train a latent diffusion model** over \(z\):

- **Forward process**: add Gaussian noise anisotropically, with mode-dependent variance:
  \[
  z_\tau = \sqrt{\bar\alpha_\tau}\, z_0 + \sqrt{1 - \bar\alpha_\tau}\, \Lambda_m^{-1/2}\, \epsilon,
  \quad \epsilon \sim \mathcal{N}(0, I),
  \]
  where \(\Lambda_m = \mathrm{diag}(\lambda_1, \dots, \lambda_m)\) are the Laplacian eigenvalues.
  This noise schedule adds *more* noise to high-frequency (high-\(\lambda\)) modes, exactly
  matching the spectral intuition of the Rayleigh quotient.

- **Denoiser \(\epsilon_\theta(z_\tau, \tau)\)**: a small transformer or MLP that takes
  noisy modal coordinates and timestep, and predicts the noise. This denoiser can be made
  Laplacian-aware by using \(L_f\) in its self-attention as a structural bias (analogous to
  graph-conditioned diffusion).

- **Decoder**: the Stage 1 decoder maps denoised \(\hat{z}_0\) back to feature/token space.

### Laplacian-structured noise schedule

The key distinction from standard latent diffusion is the **mode-dependent noise**:

\[
p(z_\tau \mid z_0) = \mathcal{N}(\sqrt{\bar\alpha_\tau}\, z_0,\; (1 - \bar\alpha_\tau)\, \Lambda_m^{-1}),
\]

so low-frequency modes (small \(\lambda_k\)) survive longer (less noise), while high-frequency
modes are corrupted earlier. Sampling from the prior at \(\tau = T\) gives
\(z_T \sim \mathcal{N}(0, \Lambda_m^{-1})\), a zero-mean Gaussian with modal variance inversely
proportional to Laplacian eigenvalues — a proper spectral prior.

### Training Objective

Standard denoising score matching on latents:

\[
\mathcal{L}_{\text{diff}} = \mathbb{E}_{z_0, \epsilon, \tau}
\left[\|\epsilon_\theta(z_\tau, \tau) - \epsilon\|_2^2\right],
\]

combined with the Stage 1 reconstruction loss.

### PyTorch Sketch

```python
class SpectralDiffusion(nn.Module):
    """Denoising network for Laplacian-structured latent diffusion."""
    def __init__(self, m, T=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m + 1, 256), nn.SiLU(),
            nn.Linear(256, 256),  nn.SiLU(),
            nn.Linear(256, m)
        )
        self.T = T

    def forward(self, z_tau, tau):
        # tau: (B,) integer timestep
        t_embed = (tau.float() / self.T).unsqueeze(-1)  # (B, 1)
        return self.net(torch.cat([z_tau, t_embed], dim=-1))

def diffusion_loss(model, z0, eigvals_m, T=1000, betas=None):
    B = z0.size(0)
    tau = torch.randint(1, T, (B,), device=z0.device)
    alpha_bar = (1 - betas[:tau]).prod()   # simplified; use cosine schedule
    noise = torch.randn_like(z0) / eigvals_m.sqrt().unsqueeze(0)  # modal noise
    z_tau = alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * noise
    eps_pred = model(z_tau, tau)
    return (eps_pred - noise).pow(2).mean()
```

### What this gives you

- A generative model with a principled spectral prior tied to the Laplacian geometry.
- Interpretable: low-frequency modes are generated first in the reverse diffusion process.
- Connects directly to graph generation literature on Laplacian-based spectral diffusion.
