# Spectral-PPCA VAE: A Laplacian-Prior Generative Framework for Spectral Associative Memory in Transformers

## Executive Summary

This document defines a unified generative framework — **Spectral-PPCA VAE** — that extends probabilistic PCA with three coupled inductive biases: (1) spectral constraint on loadings $$W$$ via Laplacian eigenbasis parametrisation, (2) Laplacian smoothing priors over latent coordinates and means, and (3) a $$\tau$$-mode distribution as a prior over active spectral frequencies. Learning proceeds via ELBO maximisation, giving a variational autoencoder whose latent space is explicitly shaped by index geometry provided by ArrowSpace (or any graph-wiring / Laplacian construction). The resulting **spectral artefact** — the posterior over loadings, latent codes, and mode weights — is then used to pre-build associative memory matrices in a downstream transformer, grounding key-value pairs in the spectral structure of the data manifold.

***

## 1. Background: PPCA as a Latent Variable Model

Standard PPCA defines a linear-Gaussian generative model[^1][^2]:

$$
x \sim \mathcal{N}(0, I_q), \quad t \mid x \sim \mathcal{N}(Wx + \mu, \sigma^2 I_d),
$$

producing a marginal distribution after integrating out the latent $$x$$:

$$
t \sim \mathcal{N}(\mu, C), \quad C = WW^\top + \sigma^2 I_d.
$$

In the fully Bayesian extension, one places priors over the parameters and defines the Bayesian evidence as[^3][^4]:

$$
p(D \mid \mathcal{M}) = \int p(D \mid X, \theta, \mathcal{M})\, p(X \mid \theta, \mathcal{M})\, p(\theta \mid \mathcal{M})\, dX\, d\theta,
$$

where $$\mathcal{M}$$ indexes the model class. The proposal here is to replace $$\mathcal{M}$$ with an index $$\mathcal{I}$$ encoding graph/spectral geometry — an ArrowSpace wiring graph, a feature Laplacian, or a $$\tau$$-mode distribution — so that evidence scores the compatibility of an entire index construction with the data[^5][^6][^7].

***

## 2. The Three Structural Priors

### 2.1 Spectral Constraint on Loadings $$W$$

Let $$L(\mathcal{I})$$ be the graph Laplacian associated with index $$\mathcal{I}$$, with eigendecomposition[^7][^8]:

$$
L(\mathcal{I}) = U \Lambda U^\top, \quad U \in \mathbb{R}^{d \times d},\ \Lambda = \mathrm{diag}(\lambda_1 \le \cdots \le \lambda_d).
$$

Parametrise the loading matrix as $$W = U_{1:q} S$$, where $$U_{1:q} \in \mathbb{R}^{d \times q}$$ are the $$q$$ lowest-frequency eigenvectors and $$S \in \mathbb{R}^{q \times q}$$ lives in the spectral basis[^9][^7]. This is the Laplacian eigenbasis reparametrisation.

Place a spectral shrinkage prior on $$S$$ weighted by eigenvalues[^10][^5][^7]:

$$
p(S \mid \mathcal{I}) \propto \exp\!\left(-\frac{\lambda_s}{2} \operatorname{tr}\!\left(S^\top \Lambda_{1:q} S\right)\right),
\quad \Lambda_{1:q} = \mathrm{diag}(\lambda_1, \dots, \lambda_q).
$$

This shrinks high-frequency components (large $$\lambda_k$$) more strongly, embodying a **spectral Occam's razor**: the prior favours loadings that lie along smooth, low-frequency directions of the index graph. The form is Gaussian in $$S$$, so it is conjugate and tractable[^5].

### 2.2 Laplacian Smoothing Priors on Latents and Means

Following the graph spectral regularisation framework[^10][^11], two Laplacian quadratic-form priors enforce smoothness of latent coordinates and decoder means relative to the index graph:

**Latent prior** (smooth over sample graph $$L_s$$):

$$
p(X \mid \mathcal{I}) \propto \exp\!\left(-\frac{\beta}{2} \operatorname{tr}\!\left(X L_s(\mathcal{I}) X^\top\right)\right).
$$

**Loading/mean prior** (smooth over feature graph $$L_f$$):

$$
p(W \mid \mathcal{I}) \propto \exp\!\left(-\frac{\alpha}{2} \operatorname{tr}\!\left(W^\top L_f(\mathcal{I}) W\right)\right).
$$

These priors generalise the standard isotropic Gaussian prior in vanilla PPCA by replacing the identity with a Laplacian-precision matrix[^5][^6][^12]. In practice, the sample Laplacian $$L_s$$ can be dynamically computed from the data (e.g. a $$k$$-NN graph in embedding space), while the feature Laplacian $$L_f$$ is provided by the ArrowSpace wiring graph — making $$\mathcal{I}$$ a first-class modelling choice rather than a hyperparameter[^7][^11].

### 2.3 $$\tau$$-Mode Distribution as a Prior over Active Spectral Frequencies

Let $$\pi \in \Delta^{q-1}$$ (the $$q$$-simplex) be a probability vector over the $$q$$ active eigenvectors. The $$\tau$$-mode distribution is a prior over $$\pi$$ that encodes which Laplacian modes carry signal[^7]:

$$
p(\pi \mid \tau) = \mathrm{Dir}(\tau \cdot \mathbf{1}_q), \quad \tau > 0.
$$

Alternatively, for a continuous mode-weighting interpretation, define modal weights $$\omega_k \ge 0$$ and a temperature-scaled prior:

$$
p(\omega \mid \tau, \Lambda) \propto \exp\!\left(-\tau \sum_k \lambda_k \omega_k\right), \quad \omega_k \ge 0,
$$

which is a product of exponential distributions with rates $$\tau \lambda_k$$. This gives heavy support on low-frequency modes (small $$\lambda_k$$) and exponential decay for high frequencies, controlled by the temperature $$\tau$$[^7]. The loading matrix becomes a **mode-weighted** linear combination: $$W = U_{1:q}\,\mathrm{diag}(\omega)\,S$$.

At the limit $$\tau \to 0$$, all modes are equally weighted (flat prior); at $$\tau \to \infty$$, only the Fiedler vector (the mode corresponding to $$\lambda_1 > 0$$) is active.

***

## 3. Spectral-PPCA VAE: Joint Generative Model

### 3.1 Full Joint Distribution

Combining all three structural priors, the joint model is:

$$
p(D, X, S, \omega, \sigma^2 \mid \mathcal{I}) = p(D \mid X, W(\mathcal{I}, S, \omega), \sigma^2)\, p(X \mid \mathcal{I})\, p(S \mid \mathcal{I})\, p(\omega \mid \tau, \Lambda)\, p(\sigma^2),
$$

where $$W(\mathcal{I}, S, \omega) = U_{1:q}\,\mathrm{diag}(\omega)\,S$$ and[^1][^9][^7]:

$$
p(D \mid X, W, \sigma^2) = \prod_{n=1}^N \mathcal{N}(t_n \mid Wx_n + \mu,\, \sigma^2 I_d).
$$

All components that are Gaussian in their argument admit closed-form KL divergences in the ELBO[^13][^14].

### 3.2 Variational Family

For scalable ELBO optimisation, adopt a mean-field amortised variational family with an encoder network $$e_\phi$$:

$$
q_\phi(X, S, \omega \mid D, \mathcal{I}) = q_\phi(X \mid D)\, q_\phi(S \mid D)\, q_\phi(\omega \mid D),
$$

each factor being a reparametrisable distribution[^13][^14]:

- $$q_\phi(x_n \mid t_n) = \mathcal{N}(\mu_{x,n}, \mathrm{diag}(\sigma^2_{x,n}))$$ — standard Gaussian encoder.
- $$q_\phi(S \mid D) = \mathcal{N}(M_S, \Sigma_S)$$ — Gaussian posterior in the spectral basis.
- $$q_\phi(\omega \mid D) = \prod_k \mathrm{Gamma}(a_k, b_k)$$ or Log-Normal (for positivity), reparametrisable via the implicit reparametrisation trick[^15][^16].

### 3.3 ELBO Decomposition

The evidence lower bound factorises into interpretable terms[^13][^14]:

$$
\mathcal{L}(\phi, \theta \mid \mathcal{I}) = \underbrace{\mathbb{E}_{q_\phi}[\log p(D \mid X, W, \sigma^2)]}_{\text{Reconstruction}} - \underbrace{\mathrm{KL}(q_\phi(X) \,\|\, p(X \mid \mathcal{I}))}_{\text{Latent smoothness}} - \underbrace{\mathrm{KL}(q_\phi(S) \,\|\, p(S \mid \mathcal{I}))}_{\text{Spectral basis regularity}} - \underbrace{\mathrm{KL}(q_\phi(\omega) \,\|\, p(\omega \mid \tau, \Lambda))}_{\tau\text{-mode frequency penalty}}.
$$

Each KL term has a specific geometric interpretation[^10][^5][^7]:

- **Latent smoothness KL**: penalises latent coordinates that vary sharply across neighbours in the sample graph — enforcing topological continuity of the embedding.
- **Spectral basis KL**: penalises high-frequency loading components weighted by Laplacian eigenvalues — pulling $$W$$ toward smooth directions of the feature graph.
- **$$\tau$$-mode KL**: penalises activation of high-frequency modes — acting as a learnable spectral band-limiter.

The KL between two Gaussians and between two exponential-family distributions (Gamma/exponential) are closed form, so the ELBO is fully differentiable and admits standard stochastic gradient ascent[^13][^14].

### 3.4 ELBO with Laplacian-Precision Prior

For the latent KL, substituting a Laplacian-precision prior $$p(X) = \mathcal{N}(0, (I + \beta L_s)^{-1})$$ gives a closed-form Gaussian KL[^10][^12]:

$$
\mathrm{KL}(q_\phi(X) \,\|\, p(X)) = \frac{1}{2} \sum_n \left[\operatorname{tr}\!\left(\Sigma_{x,n}(I + \beta L_s)\right) + \mu_{x,n}^\top (I + \beta L_s) \mu_{x,n} - d - \log \det \Sigma_{x,n} + \log \det (I + \beta L_s)^{-1}\right].
$$

The term $$\mu_{x,n}^\top L_s \mu_{x,n}$$ is exactly the graph Laplacian quadratic form (Dirichlet energy) of the latent means over the sample graph[^10][^11], providing a direct regularisation of embedding smoothness.

***

## 4. The Spectral Artefact

After training to convergence, the following quantities form the **spectral artefact** $$\mathcal{A}(\mathcal{I})$$:

| Component | Symbol | Meaning |
|-----------|--------|---------|
| Posterior loading basis | $$\hat{W} = U_{1:q}\,\mathrm{diag}(\mathbb{E}[\omega])\,\mathbb{E}[S]$$ | Mean loading matrix in Laplacian basis |
| Posterior latent codes | $$\{\hat{\mu}_{x,n}\}_{n=1}^N$$ | Smoothed latent representations of data |
| Spectral mode weights | $$\mathbb{E}[\omega_k]$$ for $$k=1,\dots,q$$ | Learned per-mode activity under $$\tau$$-mode prior |
| Mode posterior covariance | $$\Sigma_S$$ | Uncertainty over spectral decomposition |
| Residual noise | $$\hat{\sigma}^2$$ | Isotropic reconstruction residual |
| Bayesian evidence (approx.) | $$\mathcal{L}(\hat{\phi}, \hat{\theta} \mid \mathcal{I})$$ | ELBO score ranking index $$\mathcal{I}$$ |

The spectral artefact is geometrically interpretable: the columns of $$\hat{W}$$ are linear combinations of graph Laplacian eigenvectors, the latent codes are smooth over the sample graph, and the mode weights indicate which frequency bands of the index were necessary for reconstruction. By comparing $$\mathcal{L}$$ across different index constructions $$\{\mathcal{I}_1, \mathcal{I}_2, \dots\}$$, one can perform Bayesian model selection over ArrowSpace indices[^17][^18][^3].

***

## 5. Transformer with Pre-Built Spectral Associative Memory

### 5.1 Associative Memory Background

Feed-forward layers in transformers can be viewed as key-value associative memories: $$\mathrm{FFN}(x) = W_V^\top \mathrm{ReLU}(W_K x)$$, where $$W_K$$ stores keys and $$W_V$$ stores values[^19][^20]. The associative memory matrix $$S_t = \sum_i v_i k_i^\top$$ retrieves a value via $$S_t k_q \approx v_i$$ when keys are approximately orthonormal[^21][^22][^20]. The retrieval Signal-to-Noise Ratio (SNR) degrades as $$\mathrm{SNR}^{-1} \approx N/d_k$$, motivating orthogonal key bases with high capacity[^21][^20].

### 5.2 Spectral Keys and Values from the Artefact

The spectral artefact provides a natural, pre-built associative memory with three properties:

1. **Approximately orthogonal keys**: The columns of $$\hat{W}$$ are linear combinations of Laplacian eigenvectors, which are orthonormal by construction of the eigendecomposition[^7][^8]. When $$S \approx I$$, the keys are exactly orthonormal, maximising retrieval SNR[^21][^20].

2. **Spectrally grounded values**: Each key $$\hat{w}_k$$ (the $$k$$-th column of $$\hat{W}$$) corresponds to a frequency band of the ArrowSpace index. The associated value can be set to the posterior latent mean projected through the decoder, $$v_k = d_\theta(\hat{w}_k)$$, yielding a value that represents the data pattern explained by that frequency band.

3. **Mode-weighted initialisation**: The mode weights $$\mathbb{E}[\omega_k]$$ provide a natural initialisation of the value magnitudes, down-weighting high-frequency (noisy) components in the memory.

### 5.3 Pre-Building the Memory Matrix

Define the pre-built associative memory matrix $$S_\mathcal{I} \in \mathbb{R}^{d_v \times d_k}$$ as:

$$
S_\mathcal{I} = \sum_{k=1}^q \mathbb{E}[\omega_k]\, d_\theta(\hat{w}_k)\, \hat{w}_k^\top,
$$

where $$d_\theta(\hat{w}_k)$$ is the decoder output for the $$k$$-th spectral direction. This is an outer-product accumulation of spectral key-value pairs — exactly the Hopfield/linear-attention memory structure[^21][^22][^23], but with keys drawn from the Laplacian eigenbasis rather than randomly initialised weights. For a modern Hopfield network variant[^24], the energy function:

$$
E(x) = -\mathrm{lse}\left(\beta_H S_\mathcal{I}^\top x\right) + \frac{1}{2} x^\top x + \frac{1}{\beta_H} \log q
$$

uses $$S_\mathcal{I}$$ as the stored memory and $$x$$ as the query, retrieving the closest spectral pattern via softmax-weighted combination.

### 5.4 Integration into the Transformer Architecture

The full architecture operates in two phases:

**Phase 1 — Spectral Learning (offline)**:
1. Construct or load ArrowSpace index $$\mathcal{I}$$ (wiring graph, Laplacian).
2. Train Spectral-PPCA VAE on the dataset $$D$$ by maximising the ELBO.
3. Extract spectral artefact $$\mathcal{A}(\mathcal{I})$$: $$\hat{W},\ \{\hat{\mu}_{x,n}\},\ \{\mathbb{E}[\omega_k]\}$$.
4. Compute pre-built memory $$S_\mathcal{I}$$.
5. Optionally run ELBO-based model selection over candidate indices $$\{\mathcal{I}_1, \dots, \mathcal{I}_m\}$$, selecting the one with highest ELBO.

**Phase 2 — Transformer with Spectral Memory (online)**:
1. Initialise transformer feed-forward layers or cross-attention value matrices with $$S_\mathcal{I}$$.
2. At inference, queries $$q \in \mathbb{R}^{d_k}$$ retrieve from $$S_\mathcal{I}$$ via (soft) association: $$o = \mathrm{softmax}(S_\mathcal{I}^\top q / \sqrt{d_k})\, V_\mathcal{I}$$.
3. The transformer can optionally update $$S_\mathcal{I}$$ via delta-rule online learning, writing new associations without overwriting spectral structure[^25][^20].
4. The spectral artefact serves as a **frozen prior memory** (long-term associative storage), while attention handles dynamic short-term associations[^21][^19].

This architecture is related to NVIB (Nonparametric Variational Information Bottleneck) transformers[^26][^27], but with the key distinction that the prior is not isotropic Gaussian — it is spectral, shaped by $$\mathcal{I}$$.

***

## 6. Worked ELBO Gradient Flow

For implementation, the ELBO gradient with respect to encoder parameters $$\phi$$ decomposes as follows. Let $$\hat{x}_n = \mu_{x,n} + \epsilon_n \odot \sigma_{x,n}$$ (reparametrisation trick with $$\epsilon_n \sim \mathcal{N}(0,I)$$)[^13][^14]:

$$
\nabla_\phi \mathcal{L} = \underbrace{\nabla_\phi \mathbb{E}_q[\log p(t_n \mid \hat{x}_n, \hat{W})]}_{\text{recon. gradient}} - \nabla_\phi \underbrace{\left[\frac{\beta}{2}\,\hat{\mu}_{x,n}^\top L_s \hat{\mu}_{x,n} + \frac{1}{2}\operatorname{tr}(\Sigma_{x,n} L_s)\right]}_{\text{graph smoothness gradient}} - \nabla_\phi \underbrace{\left[\frac{\lambda_s}{2}\operatorname{tr}(M_S^\top \Lambda_{1:q} M_S)\right]}_{\text{spectral shrinkage gradient}}.
$$

The graph smoothness gradient is a **graph signal processing backpropagation** through the Laplacian quadratic form — equivalent to one step of graph heat diffusion applied to the latent means[^10][^11]. The spectral shrinkage gradient is a diagonal scaling in Laplacian eigenvalue space, which can be computed efficiently if $$L_f$$ is sparse[^7].

***

## 7. Complexity and Implementation Notes

| Step | Complexity | Bottleneck |
|------|-----------|------------|
| Laplacian eigendecomposition $$L(\mathcal{I})$$ | $$O(d^3)$$ or $$O(d \cdot k_\mathrm{eig})$$ with ARPACK/LOBPCG | Feature dimension $$d$$; do once offline |
| Encoder forward pass | $$O(N \cdot d \cdot q)$$ per batch | Standard VAE encoder |
| Graph smoothness KL | $$O(N \cdot \mathrm{nnz}(L_s))$$ | Sparse Laplacian; linear in edges |
| Spectral shrinkage KL | $$O(q^2)$$ | Cheap diagonal computation |
| $$\tau$$-mode KL (Gamma/exponential) | $$O(q)$$ per sample | Closed form |
| Spectral artefact extraction | $$O(q \cdot d)$$ | One forward pass post-training |
| Memory matrix assembly $$S_\mathcal{I}$$ | $$O(q \cdot d_k \cdot d_v)$$ | One-time offline computation |

For ArrowSpace specifically: the wiring graph is sparse by design, making $$\mathrm{nnz}(L) \ll d^2$$, and the eigendecomposition can be truncated to the low-lying $$q$$ modes using iterative solvers (e.g. LOBPCG in Rust via `ndarray-linalg` or SciPy's `eigsh`)[^7].

***

## 8. Connection to Bayesian Index Selection

Training the Spectral-PPCA VAE on multiple index candidates $$\{\mathcal{I}_1, \mathcal{I}_2\}$$ yields evidence estimates $$\{\mathcal{L}(\mathcal{I}_1), \mathcal{L}(\mathcal{I}_2)\}$$[^17][^18][^3]. The Bayes factor:

$$
\mathrm{BF}_{12} = \exp\!\left(\mathcal{L}(\mathcal{I}_1) - \mathcal{L}(\mathcal{I}_2)\right)
$$

provides a principled comparison between index constructions. This closes the loop: ArrowSpace does not only produce the Laplacian / wiring graph but also receives a generative-model score for how well its geometry explains the data covariance, making **index construction itself a Bayesian model selection problem**[^28][^17][^3].

***

## 9. Summary of Key Design Choices

| Modelling choice | Effect | Trade-off |
|------------------|--------|-----------|
| $$W = U_{1:q} S$$ (Laplacian basis) | Keys exactly orthonormal, max. retrieval SNR[^21] | Requires eigendecomposition of $$L(\mathcal{I})$$ offline |
| Laplacian-precision latent prior | Latent codes smooth over sample graph; interpretable manifold[^10][^11] | KL requires Laplacian eigenvalues; adds $$O(\mathrm{nnz}(L))$$ per step |
| $$\tau$$-mode Gamma prior on $$\omega$$ | Learns spectral band-limiting; auto-selects effective $$q$$[^7] | Requires implicit reparametrisation or LogNormal relaxation |
| ELBO as index score | Index selection is Bayesian, principled[^17][^3] | ELBO is a lower bound; may need importance-weighted correction (IWAE) |
| Pre-built $$S_\mathcal{I}$$ as transformer memory | Spectral prior memory replaces random FFN init; grounded in data geometry[^19][^29] | Memory frozen unless delta-rule updates are added[^25] |

---

## References

1. [Probabilistic PCA derivations](https://andrewcharlesjones.github.io/journal/ppca.html) - Probabilistic PCA generalizes traditional PCA into a probabilistic model whose maximum likelihood es...

2. [Probabilistic principal component analysis - Tagkopoulos Lab](http://tagkopouloslab.ucdavis.edu/uncategorized/2019/05/probabilistic-principal-component-analysis/) - In this blog, we will walk through a probabilistic formulation of the well-known technique of princi...

3. [Automatic Choice of Dimensionality for PCA](http://papers.neurips.cc/paper/1853-automatic-choice-of-dimensionality-for-pca.pdf) - This paper resolves the situation by deriving a method which is both accurate and fast. It is an app...

4. [Marginal likelihood](https://en.wikipedia.org/wiki/Marginal_likelihood) - A marginal likelihood is a likelihood function that has been integrated over the parameter space. In...

5. [Bayesian regularization via graph Laplacian - Scholars@Duke](https://scholars.duke.edu/publication/1032729) - Regularization plays a critical role in modern statistical research, especially in high-dimensional ...

6. [A graph Laplacian prior for Bayesian variable selection and grouping](https://research.ibm.com/publications/a-graph-laplacian-prior-for-bayesian-variable-selection-and-grouping) - In this paper, the focus is on cases in which some correlated predictors have similar effects on the...

7. [Graph Laplacian Spectral Prior Overview - Emergent Mind](https://www.emergentmind.com/topics/graph-laplacian-spectral-prior) - A graph Laplacian spectral prior is a structural or statistical constraint on the spectrum (the set ...

8. [Principal component analysis - Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)

9. [PCA Based on Graph Laplacian Regularization and P ...](https://pubmed.ncbi.nlm.nih.gov/28371780/) - In modern molecular biology, the hotspots and difficulties of this field are identifying characteris...

10. [Workshop on Representation Learning on Graphs and Manifold – ICLR 2019](https://rlgm.github.io/papers/53.pdf)

11. [[1807.11637] Deep Graph Laplacian Regularization - arXiv](https://arxiv.org/abs/1807.11637) - We propose to combine the robustness merit of model-based approaches and the learning power of data-...

12. [[PDF] PCA using Graph Total variation - Infoscience](https://infoscience.epfl.ch/server/api/core/bitstreams/6a346efb-d13f-4756-8f06-2b68532dee01/content)

13. [Deep Learning (BEV033DLE) Lecture 11 Variational ...](https://cw.fel.cvut.cz/b202/_media/courses/bev033dle/vae.pdf)

14. [[PDF] 1 Definitions](https://www.cs.cmu.edu/~10315/recitation/S25_10_315_Recitation_13.pdf) - Maximizing the ELBO encourages the model to reconstruct the data well while keeping the learned late...

15. [Location-scale Family of Laplacian Distributions and](https://assets.researchsquare.com/files/rs-3202724/v1_covered_47c35790-bfa7-49db-81d5-c4debc1a8cbe.pdf?c=1690957558)

16. [Published as a conference paper at ICLR 2024](https://openreview.net/pdf?id=RzNlECeoOB)

17. [Bayesian Model Selection, the Marginal Likelihood, and ...](https://proceedings.mlr.press/v162/lotfi22a/lotfi22a.pdf) - This probability of gen- erating a dataset from a prior model is called the marginal likelihood, or ...

18. [Paper Review: Bayesian Model Selection, the Marginal ...](http://blog.blackhc.net/2022/06/bayesian-model-selection-marginal-likehood-generalization/) - The paper examines the behavior of LML and CLML in various settings: density models, Fourier feature...

19. [[PDF] Transformer Feed-Forward Layers Are Key-Value Memories](https://www.semanticscholar.org/paper/Transformer-Feed-Forward-Layers-Are-Key-Value-Geva-Schuster/4a54d58a4b20e4f3af25cea3c188a12082a95e02) - This work shows that feed-forward layers in transformer-based language models operate as key-value m...

20. [[Papierüberprüfung] Understanding Transformer from the ...](https://www.themoonlight.io/de/review/understanding-transformer-from-the-perspective-of-associative-memory) - The paper interprets Transformer architectures through the lens of associative memory, a concept ins...

21. [Understanding Transformer from the Perspective of Associative ...](https://www.alphaxiv.org/overview/2505.19488v1) - View recent discussion. Abstract: In this paper, we share our reflections and insights on understand...

22. [[PDF] Understanding Transformer from the Perspective of Associative ...](https://idea.snu.ac.kr/wp-content/uploads/sites/6/2025/11/251102_%EA%B4%80%ED%98%B8Understanding_Transformer_from_the_Perspective_of_Associative_Memory.pdf)

23. [Universal Hopfield Networks: A General Framework for Single-Shot ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614148/) - A large number of neural network models of associative memory have been proposed in the literature. ...

24. [Modern Hopfield network - Wikipedia](https://en.wikipedia.org/wiki/Modern_Hopfield_network)

25. [Memory in Transformers (2): Associative Memory as Test-Time Regression](https://victorfiz.com/blog/2025/02/08/memory-in-transformers-2.html) - A blog showcasing various projects and posts.

26. [Attention-based Architectures as Latent Variable Models](https://infoscience.epfl.ch/entities/publication/dad86ab3-09eb-48bf-b6d4-8c3d903eb94d) - Transformers have achieved remarkable success across modalities including text, graphs, speech, and ...

27. [Published as a conference paper at ICLR 2023](https://openreview.net/pdf?id=6QkjC_cs03X)

28. [Bayes Factors and Marginal Likelihood - PyMC](https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Bayes_factor.html) - The “Bayesian way” to compare models is to compute the marginal likelihood of each model, ie the pro...

29. [Published as a conference paper at ICLR 2025](https://openreview.net/pdf?id=hwSmPOAmhk)

