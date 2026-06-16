# References

This document collects the foundational references behind the Wiring Autoencoder (VDT).
References are grouped by conceptual pillar.

---

## 1. ArrowSpace / Graph Wiring

The `DifferentiableLaplacian`, `J_freq` cost, tau-mode truncation, and λ-fingerprint are
directly grounded in the ArrowSpace / Graph Wiring framework.

- **arrowspace** (pyarrowspace) computational library — Graph Wiring implementation:
  https://github.com/tuned-org-uk/arrowspace

- **Graph Wiring concept notes and notebooks** — tuned-org-uk internal notebooks 01–05
  covering λ-fingerprints, spectral cost functions, and tau-mode diffusion:
  https://github.com/tuned-org-uk

---

## 2. Rayleigh's Theory of Sound — Vibrational Framing

The VDT's core intuition treats graph wiring as a **vibrational system**:
the oscillating state of a string (low-entropy, low-frequency, standing wave) corresponds to
a smooth learned Laplacian; the rest/collapsed state (high-entropy) corresponds to wave-function
collapse or Brownian high-temperature disorder.

- **Rayleigh, J.W.S.** (1877/1894). *The Theory of Sound*, Vols I & II.
  Macmillan, London. (Public domain; digitised at
  https://archive.org/details/theoryofsound01raylgoog)

- **Relevant chapter**: §88–§101 (Vol. I) — transverse vibrations of strings,
  normal modes, and the Rayleigh quotient `R(v) = vᵀLv / vᵀv`.
  The Rayleigh quotient is the energy functional minimised by graph spectral embeddings;
  the VDT's `J_freq` is a discrete analogue of Rayleigh's frequency penalty.

- **Rayleigh quotient in graph Laplacians**:
  Chung, F.R.K. (1997). *Spectral Graph Theory*. AMS, CBMS Monographs.
  https://mathweb.ucsd.edu/~fan/research/revised.html
  (See Chapter 1 for the connection between graph eigenvalues and vibration modes.)

---

## 3. Quantum Probabilities Without Complex Numbers — Aaronson

The VDT sign-amplitude design (positive + negative amplitudes via the sign of graph edge
weights, without complex Hilbert space) is motivated by:

- **Aaronson, S.** (2013). *Quantum Computing Since Democritus*, Chapter 9:
  "Quantum" (probability amplitudes, interference, the Born rule).
  Cambridge University Press.
  https://scottaaronson.com/democritus/lec9.html

  Key result used: if negative probability amplitudes are allowed, the full interference
  structure of quantum mechanics is recovered without complex numbers, provided
  the update rule is a unitary (or in our case, heat-kernel) transformation.
  The VDT's `TauModeDiffusion` implements `K_τ = U exp(−tΛ) Uᵀ` as a real-valued
  analogue of a unitary quantum channel.

- **Renou, M.-O., et al.** (2021). Quantum theory based on real numbers can be
  experimentally falsified. *Nature*, 600, 625–629.
  https://doi.org/10.1038/s41586-021-04160-4
  (Experimental context for real vs. complex quantum theories.)

- **McKague, M., et al.** (2009). Simulating quantum systems using real Hilbert spaces.
  *Physical Review Letters*, 102, 020505.
  https://doi.org/10.1103/PhysRevLett.102.020505

---

## 4. Graph Laplacian and Spectral Learning

- **Chung, F.R.K.** (1997). *Spectral Graph Theory*. AMS.
  https://mathweb.ucsd.edu/~fan/research/revised.html

- **von Luxburg, U.** (2007). A tutorial on spectral clustering.
  *Statistics and Computing*, 17(4), 395–416.
  https://doi.org/10.1007/s11222-007-9033-z
  (Accessible derivation of normalised Laplacian `L = I − D^{−1/2} A D^{−1/2}`.)

- **Belkin, M. & Niyogi, P.** (2003). Laplacian Eigenmaps for Dimensionality
  Reduction and Data Representation. *Neural Computation*, 15(6), 1373–1396.
  https://doi.org/10.1162/089976603321780317
  (Direct predecessor of the `TauModeDiffusion` decoder.)

- **Coifman, R.R. & Lafon, S.** (2006). Diffusion maps.
  *Applied and Computational Harmonic Analysis*, 21(1), 5–30.
  https://doi.org/10.1016/j.acha.2006.04.006
  (Heat kernel `K_τ = exp(−tL)` and its relationship to τ-mode truncation.)

---

## 5. VAE / PPCA Lineage

The VDT inherits the probabilistic latent-variable structure from:

- **Kingma, D.P. & Welling, M.** (2013). Auto-Encoding Variational Bayes.
  *arXiv:1312.6114*. https://arxiv.org/abs/1312.6114
  (The core ELBO objective and reparameterisation trick.)

- **Tipping, M.E. & Bishop, C.M.** (1999). Probabilistic Principal Component Analysis.
  *Journal of the Royal Statistical Society B*, 61(3), 611–622.
  https://doi.org/10.1111/1467-9868.00196
  (PPCA as the linear limit of VAE; grounding for the LinearAE benchmark baseline.)

- **Higgins, I., et al.** (2017). β-VAE: Learning Basic Visual Concepts with a
  Constrained Variational Framework. *ICLR 2017*.
  https://openreview.net/forum?id=Sy2fchgIl
  (Motivation for the `β·KL` term in the VDT objective.)

- **Rezende, D.J. & Viola, F.** (2018). Taming VAEs. *arXiv:1810.00597*.
  https://arxiv.org/abs/1810.00597
  (Guidance on `β` annealing and avoiding posterior collapse, relevant to VDT training.)

---

## 6. Graph Neural Networks and Message Passing

- **Kipf, T.N. & Welling, M.** (2016). Semi-Supervised Classification with Graph
  Convolutional Networks. *arXiv:1609.02907*.
  https://arxiv.org/abs/1609.02907
  (Cora/PubMed benchmark datasets used in `benchmark.py`.)

- **Fey, M. & Lenssen, J.E.** (2019). Fast Graph Representation Learning with
  PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.
  https://arxiv.org/abs/1903.02428
  (PyG library used for dataset loading.)

---

## 7. Brownian Motion / Statistical Mechanics Analogy

The entropy framing (low-entropy oscillating state vs. high-entropy rest/collapse) maps onto:

- **Einstein, A.** (1905). Über die von der molekularkinetischen Theorie der Wärme
  geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen.
  *Annalen der Physik*, 322(8), 549–560.
  (The stationary pre-heated state as zero-entropy reference; Brownian diffusion as
  entropy increase — direct analogue of wiring collapse.)

- **Jaynes, E.T.** (1957). Information Theory and Statistical Mechanics.
  *Physical Review*, 106(4), 620–630.
  https://doi.org/10.1103/PhysRev.106.620
  (Maximum entropy principle as the equilibrium/high-entropy end-state; counterpart
  to `J_freq` which penalises approaching that state.)

---

## See Also

- `docs/architecture.md` — full data-flow diagram and module derivations
- `vdt/spectral.py` — `spectral_freq_cost`, `TauModeDiffusion`, `lambda_fingerprint`
- `vdt/laplacian.py` — `DifferentiableLaplacian` (ArrowSpace layer)
