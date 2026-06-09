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
        self.encoder = VDT(d, m_modes, K, n_heads, lambda_max)   # existing VDT
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, batch_first=True),
            num_layers=2
        )
        self.proj_out = nn.Linear(d, d)

    def forward(self, X0, L_f, eigvecs, P_f):
        # Encode
        Q_K, _, _ = self.encoder(X0, L_f, eigvecs, P_f)
        U_m = eigvecs[:, :self.encoder.m]          # (d, m)
        # Pool to latent
        z = (Q_K @ U_m).mean(dim=1)                # (B, m)
        # Decode: expand back to (B, n, d) and reconstruct
        z_expanded = z.unsqueeze(1) @ U_m.T        # (B, 1, d)
        z_expanded = z_expanded.expand(-1, X0.size(1), -1)
        X_hat = self.proj_out(self.decoder(z_expanded))
        return X_hat

def autoencoder_loss(X0, X_hat, L_f, alpha=0.01):
    recon = (X0 - X_hat).pow(2).mean()
    # Laplacian smoothness on reconstruction (batch mean)
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
