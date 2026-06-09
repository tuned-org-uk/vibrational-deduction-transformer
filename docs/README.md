# `docs/` — Wiring Autoencoder: Theory, Architecture, and Algorithm Tracks

This directory contains the theoretical and architectural documentation for the
**Wiring Autoencoder (WAE)** family of learning algorithms. These algorithms grow
directly out of the spectral graph-wiring primitives of
[ArrowSpace / Graph Wiring](https://github.com/tuned-org-uk) and are grounded in
the modelling progression of *The Little Book of Generative AI Foundations* (Chen, 2026).

The central object throughout is the **feature-space graph Laplacian** $$(L_f)$$, built
from a data matrix $$(A^\top)$$ via a similarity kernel. Equipping that Laplacian with a
positive diagonal mass matrix \(M\) turns it into a Rayleigh mass–spring system whose
eigenmodes define a spectral geometry for learning, compression, and generation.

---

## Documents in this directory

| File | Contents |
|------|----------|
| [`00-architecture.md`](00-architecture.md) | Module reference, data-flow diagram, and ELBO derivation for the core WAE |
| [`01-references.md`](01-references.md) | Annotated bibliography and connection to ArrowSpace / Graph Wiring papers |
| [`03-branching.md`](03-branching.md) | Six algorithm tracks: deterministic AE, EBM, latent diffusion, Variational Laplace, PDE forecasting, and spectrally regularised reasoner |

---

## Conceptual Foundations

The modelling chain starts from classical dimensionality reduction and moves toward
full generative and reasoning architectures. At every step, the standard linear/Gaussian
machinery of the book is replaced by its **spectral graph-wiring analogue**:

| Book concept | WAE analogue | Key object |
|---|---|---|
| PCA | Spectral Laplacian, $$L_f = D_f - W_f$$ | Graph smoothness $$z^\top L_f z$$ |
| Autoencoder | Wiring Autoencoder (deterministic, $$J_{\text{freq}}$$ loss) | $$L_f$$ as bottleneck geometry |
| PPCA | Probabilistic Graph Wiring (Gaussian $$z \to L(z)$$) | Modal prior $$p(z) = \mathcal{N}(0, \Lambda_m^{-1})$$ |
| VAE + ELBO | WAE-ELBO = recon + $$(\beta\)·KL + \(\alpha\)·\(J_{\text{freq}}$$ | Signed density matrix $$\varrho_t$$ |
| Diffusion / Flows | WAE-Diffusion over wiring space | Spectral noise schedule via $$\Lambda_m$$ |

---

## Concept Tree

The tree below maps the conceptual lineage from the *Little Book* primitives at the root
through the WAE family to the six algorithm tracks documented in `03-branching.md`.
Each leaf is a fully-specified learning algorithm; intermediate nodes are modelling
choices that define the branch.

```
                    ┌─────────────────────────────────────┐
                    │   THE LITTLE BOOK FOUNDATIONS        │
                    │                                     │
                    │  PCA → Autoencoder → PPCA → VAE     │
                    │  Diffusion → Flows → Reasoning       │
                    └──────────────────┬──────────────────┘
                                       │
                    SPECTRAL GRAPH WIRING ANALOGUE
                                       │
              ┌────────────────────────┴────────────────────────┐
              │                                                 │
   ┌──────────▼──────────┐                         ┌───────────▼───────────┐
   │  Graph Laplacian Lf  │                         │   Mass matrix M,      │
   │  (stiffness operator) │                         │   Rayleigh quotient   │
   │  z⊤ Lf z = smoothness │                         │   RM(z) = z⊤Lf z     │
   │                      │                         │           ───────     │
   │  Lf = Df − Wf        │                         │           z⊤ M z      │
   └──────────┬───────────┘                         └───────────┬───────────┘
              │                                                 │
              └─────────────────────┬───────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Laplacian eigenbasis U, Λ    │
                    │   Modal coordinates z = Q U_m  │
                    │   (natural frequencies ωk=√λk) │
                    └───────────────┬───────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
   ┌──────────▼──────────┐  ┌───────▼────────┐  ┌─────────▼─────────┐
   │  DETERMINISTIC       │  │  WAVE DYNAMICS │  │  PROBABILISTIC    │
   │  Laplacian-regularised│  │  Φ_L (discrete │  │  Modal prior      │
   │  cost Jλ(x)          │  │  damped wave)  │  │  p(z)=N(0,Λm⁻¹)  │
   │  Preconditioned GD   │  │  Qt+1 update   │  │  Density matrix   │
   │  Sσ,M convergence    │  │  (VDT paper)   │  │  ϱt = ϱt⁺ − ϱt⁻  │
   └──────────┬───────────┘  └───────┬────────┘  └─────────┬─────────┘
              │                      │                      │
   ┌──────────▼───────────────────────▼──────────────────────▼─────────┐
   │                    WIRING AUTOENCODER CORE                         │
   │                                                                    │
   │   Encoder: VDT recurrence  X0 → QK → z = pool(QK Um)             │
   │   Bottleneck: modal latent z ∈ ℝᵐ   (Laplacian eigenbasis)       │
   │   Decoder: z Um⊤ → X̂  (reconstruction or generation)            │
   │   Loss: recon + β·KL + α·Jfreq   (WAE-ELBO)                      │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────────┐
          │           ┌───────────┴────────┐                  │
          │           │                    │                  │
   NO PROBABILITY     │            PROBABILISTIC              │
   OVER LATENTS       │            LATENTS                    │
          │           │                    │                  │
    ┌─────┴──────┐  ┌─┴──────────┐  ┌─────┴──────┐  ┌───────┴──────┐
    │            │  │            │  │            │  │              │
    ▼            ▼  ▼            ▼  ▼            ▼  ▼              ▼
┌───────┐  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌──────┐  ┌─────────┐
│OPT. 1 │  │OPT. 2  │  │ OPT. 3   │  │ OPT. 4  │  │OPT.5 │  │ OPT. 6  │
│       │  │        │  │          │  │         │  │      │  │         │
│Deter- │  │Energy- │  │Vibrational│  │Variational│  │PDE / │  │Spectral │
│ministic│  │Based   │  │Latent    │  │Laplace  │  │Graph │  │Classifier│
│  AE   │  │Model   │  │Diffusion │  │   AE    │  │Fore- │  │Reasoner │
│       │  │(EBM)   │  │          │  │         │  │cast  │  │(VDT)    │
│recon  │  │task +  │  │denoising │  │Laplace  │  │state │  │depth-   │
│+ Lf   │  │E(Q)    │  │score     │  │ELBO     │  │pred  │  │supervis.│
│smooth │  │+ relax │  │matching  │  │+ Hessian│  │+ CFL │  │CE loss  │
│loss   │  │gap     │  │ modal    │  │covariance│  │penalty│  │+ modal  │
│       │  │        │  │ noise    │  │         │  │      │  │spectra  │
└───────┘  └────────┘  └──────────┘  └─────────┘  └──────┘  └─────────┘
   low        med         high           med         low        low
complexity  complexity  complexity    complexity complexity complexity
```

---

## Recommended Implementation Sequence

For a researcher starting from the current VDT/WAE codebase, the suggested order is:

1. **Option 6** — Spectrally regularised classifier/reasoner: closes the loop on
   Section 11 of the VDT paper with minimal new code. Validates that the wave dynamics
   and Laplacian constraints improve reasoning over depth.

2. **Option 1** — Deterministic vibrational AE: adds a reconstruction objective on top
   of the existing VDT encoder. Tests whether the modal latent code captures enough
   information to reconstruct inputs.

3. **Option 4** — Variational Laplace AE: upgrades the deterministic AE to a Bayesian
   model using the preconditioned Hessian structure already available in the codebase.
   No Monte Carlo sampling required.

4. **Option 2** or **Option 5** — depending on whether the application is classification
   (energy-based) or forecasting (PDE solver). Both require moderate additional work.

5. **Option 3** — Vibrational latent diffusion: the most ambitious generative extension;
   build on Stage 1 (Option 1 encoder/decoder) as prerequisite.

---

## Relationship to the VDT Paper

The *Vibrational Deduction Transformer* paper (`vibrational-deduction-transformer.pdf`)
provides the theoretical backbone for all six tracks:

- **Part I (Foundations)**: the Laplacian stiffness \(L_f\), mass matrix \(M\), generalised
  Rayleigh quotient \(R_M\), and linear-convergence guarantee for preconditioned GD underpin
  Options 1, 2, 4, and 6.

- **Part II (Architecture)**: the discrete wave update \(\Phi_L\) and the signed density
  matrix \(\varrho_t\) are the encoder backbone for all six options.

- **Section 11 (Experiments)**: the LDT-mirrored benchmark tasks (3-SAT, syllogisms,
  modular arithmetic) directly correspond to the evaluation protocol of Option 6.

- **Section 9 (Density Matrix)**: the signed state \(\varrho_t^+ - \varrho_t^-\) is the
  starting point for the probabilistic reinterpretation in Options 3 and 4.
