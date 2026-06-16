"""
vdt/metrics.py  --  Benchmark metrics for WiringAutoencoder (issue #32)

All 7 active v2 benchmark metrics are implemented as pure functions.
Metrics that were removed per merged PR #35 (kl_lap) are not included.

Metric summary
--------------

  kl_S              KL( q(S) || p(S|I) )        -- via spectral.spectral_basis_kl
  kl_tau            KL( q(omega) || Exp(tau*L) ) -- via spectral.tau_mode_kl
  active_modes      sum_k 1[E[omega_k] > delta]  -- count of contributing modes
  memory_snr        d_k / N_stored               -- key-orthogonality SNR proxy
  elbo_bayes_factor exp( L(I1) - L(I2) )         -- relative evidence ratio
  linear_probe_acc  logistic regression on mu    -- representation quality
  spectral_entropy  H( normalised eigenvalues )   -- Laplacian mode diversity

High-level entry points
-----------------------
  evaluate       --  Run all metrics on a DataLoader and return a dict.
  compare_indices   --  ELBO Bayes-factor leaderboard over competing indices.

Design notes
------------
* All metrics are pure functions (no state, no nn.Module) to allow use
  outside the training loop and in offline evaluation scripts.
* evaluate and compare_indices depend only on the 9-key forward() dict
  from WiringAutoencoder, so they never touch model internals.
* linear_probe_acc requires only mu.detach() -- no model gradients.
* spectral_entropy delegates the eigensystem to DifferentiableLaplacian.

Ref: docs/v2/00-architecture.md -- Benchmark Metrics (v2 additions)
Ref: docs/v2/04-stability.md
Ref: issue #32
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from vdt.spectral import spectral_basis_kl, tau_mode_kl


# ---------------------------------------------------------------------------
# 1. kl_S  --  KL divergence for the spectral basis posterior
# ---------------------------------------------------------------------------

def compute_kl_S(
    S: Tensor,
    log_var_S: Tensor,
    eigvals_q: Tensor,
    lam_s: float = 1.0,
) -> Tensor:
    """
    KL( q(S) || p(S | I) ) for the spectral loading matrix posterior.

    Delegates to spectral.spectral_basis_kl; exposed here as a named
    metric function so evaluate can call it uniformly.

    Parameters
    ----------
    S : Tensor
        Posterior mean matrix.  Shape (B, q, q).
    log_var_S : Tensor
        Log-variance of the posterior.  Shape (B, q, q).
    eigvals_q : Tensor
        Leading q frozen Laplacian eigenvalues.  Shape (q,).
    lam_s : float
        Prior precision multiplier.

    Returns
    -------
    Tensor  scalar.
    """
    return spectral_basis_kl(S, log_var_S, eigvals_q, lam_s=lam_s)


# ---------------------------------------------------------------------------
# 2. kl_tau  --  KL divergence for the mode-weight posterior
# ---------------------------------------------------------------------------

def compute_kl_tau(
    log_a: Tensor,
    log_b: Tensor,
    eigvals_q: Tensor,
    tau: float = 1.0,
) -> Tensor:
    """
    KL( q(omega) || Exp(tau * lambda_k) ) for the mode-weight posterior.

    Delegates to spectral.tau_mode_kl; exposed here as a named metric
    function so evaluate can call it uniformly.

    Parameters
    ----------
    log_a : Tensor
        Log shape parameters.  Shape (B, q).
    log_b : Tensor
        Log rate parameters.  Shape (B, q).
    eigvals_q : Tensor
        Leading q frozen Laplacian eigenvalues.  Shape (q,).
    tau : float
        Diffusion time-scale multiplier for the prior rate.

    Returns
    -------
    Tensor  scalar.
    """
    return tau_mode_kl(log_a, log_b, eigvals_q, tau=tau)


# ---------------------------------------------------------------------------
# 3. active_modes  --  number of modes with non-negligible weight
# ---------------------------------------------------------------------------

def active_modes(
    omega_hat: Tensor,
    delta: float = 0.01,
) -> int:
    """
    Count the number of spectral modes whose expected weight exceeds delta.

    A mode is considered active when E[omega_k] > delta.  This provides an
    interpretable summary of how many Laplacian eigenvectors contribute
    meaningfully to the current spectral encoding.

    Parameters
    ----------
    omega_hat : Tensor
        Mode weights E[omega_k].  Shape (q,).  Values should be >= 0.
    delta : float
        Threshold below which a mode is considered inactive.  Default 0.01.

    Returns
    -------
    int  count of active modes in [0, q].
    """
    return int((omega_hat > delta).sum().item())


# ---------------------------------------------------------------------------
# 4. memory_snr  --  key-orthogonality signal-to-noise ratio
# ---------------------------------------------------------------------------

def memory_snr(
    keys: Tensor,
    n_stored: Optional[int] = None,
) -> float:
    """
    Associative memory signal-to-noise ratio proxy: d_k / N_stored.

    For a Hopfield memory seeded from q orthonormal loading directions,
    the retrieval SNR scales as d_k / N_stored where d_k is the effective
    key dimensionality and N_stored is the number of stored patterns.  This
    function computes the proxy using the effective rank of the key matrix
    as d_k.

    The effective rank of the key matrix K (shape q x d) is defined as:

        d_eff = exp( H( sigma / ||sigma||_1 ) )

    where sigma are the singular values of K and H is the Shannon entropy.
    This is the exponential of the spectral entropy of the singular values.

    Parameters
    ----------
    keys : Tensor
        Loading direction matrix.  Shape (q, d_model) or (d_model, q).
        Each row (or column) is one spectral key.
    n_stored : int or None
        Number of stored patterns.  Defaults to keys.shape[0] (one per key).

    Returns
    -------
    float  SNR proxy d_eff / n_stored.  >= 0.
    """
    if keys.ndim != 2:
        raise ValueError(f"keys must be 2-D, got shape {tuple(keys.shape)}")
    K = keys.float().detach()
    # Use shape (q, d) convention; transpose if q > d.
    if K.shape[0] > K.shape[1]:
        K = K.T

    singular_values = torch.linalg.svdvals(K)                    # (min(q,d),)
    sv_sum = singular_values.sum()
    if sv_sum < 1e-12:
        return 0.0

    p = singular_values / sv_sum                                  # normalised
    entropy = -(p * (p + 1e-12).log()).sum().item()               # H(sigma)
    d_eff = math.exp(entropy)                                     # effective rank

    n = n_stored if n_stored is not None else keys.shape[0]
    return d_eff / max(n, 1)


# ---------------------------------------------------------------------------
# 5. elbo_bayes_factor  --  relative evidence ratio between two indices
# ---------------------------------------------------------------------------

def elbo_bayes_factor(
    elbo_1: float,
    elbo_2: float,
) -> float:
    """
    Approximate Bayes factor between two ArrowSpace indices via their ELBOs.

    The ELBO is a lower bound on log p(data | index).  The Bayes factor
    approximation is:

        BF(I1, I2) ~= exp( ELBO(I1) - ELBO(I2) )

    A value > 1 indicates index I1 has higher marginal likelihood.
    Conventionally, BF > 3 is moderate evidence, BF > 10 strong evidence
    (Jeffreys scale).

    Parameters
    ----------
    elbo_1 : float
        Mean ELBO for index I1 over an evaluation dataset.
    elbo_2 : float
        Mean ELBO for index I2 over an evaluation dataset.

    Returns
    -------
    float  BF(I1, I2) = exp(elbo_1 - elbo_2).
    """
    return math.exp(elbo_1 - elbo_2)


# ---------------------------------------------------------------------------
# 6. linear_probe_acc  --  logistic regression on frozen mu
# ---------------------------------------------------------------------------

def linear_probe_acc(
    mu: Tensor,
    labels: Tensor,
    max_iter: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Accuracy of a linear probe (logistic regression) fitted on frozen mu.

    This is the standard representation-quality benchmark for VAE-style
    models.  Only mu.detach() is used -- no model gradients flow through
    this function.

    Uses sklearn.linear_model.LogisticRegression with L2 regularisation
    and a stratified 80/20 train/test split.

    Parameters
    ----------
    mu : Tensor
        Encoder mean vectors.  Shape (N, latent_dim).  Detached internally.
    labels : Tensor
        Integer class labels.  Shape (N,).
    max_iter : int
        Maximum iterations for the logistic regression solver.
    random_state : int
        Random seed for the train/test split.

    Returns
    -------
    float  Test-set classification accuracy in [0, 1].

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If fewer than 2 classes are present in labels.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for linear_probe_acc. "
            "Install with: pip install scikit-learn"
        ) from exc

    X = mu.detach().cpu().float().numpy()
    y = labels.detach().cpu().numpy()

    n_classes = len(set(y.tolist()))
    if n_classes < 2:
        raise ValueError(
            f"linear_probe_acc requires at least 2 classes, got {n_classes}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


# ---------------------------------------------------------------------------
# 7. spectral_entropy  --  Shannon entropy of the normalised eigenspectrum
# ---------------------------------------------------------------------------

def spectral_entropy(
    eigvals: Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Shannon entropy of the normalised eigenvalue distribution of L(z).

    Normalises the eigenvalue vector to a probability distribution by
    dividing by the sum, then computes Shannon entropy:

        p_k = lambda_k / sum_j lambda_j
        H   = -sum_k p_k * log(p_k)

    Boundary conditions:
      - If all mass is on one mode: H = 0 (deterministic spectrum).
      - If all q modes are equally weighted: H = log(q).

    Parameters
    ----------
    eigvals : Tensor
        Leading q Laplacian eigenvalues.  Shape (q,).  Must be >= 0.
        Negative values are clipped to 0 before normalisation.
    eps : float
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    float  Entropy in nats, in [0, log(q)].
    """
    ev = eigvals.detach().float().clamp(min=0.0)
    total = ev.sum()
    if total < eps:
        return 0.0
    p = ev / (total + eps)
    # Avoid log(0) by adding eps inside the log.
    h = -(p * (p + eps).log()).sum().item()
    return float(h)


# ---------------------------------------------------------------------------
# evaluate  --  run all 7 metrics on a DataLoader
# ---------------------------------------------------------------------------

def evaluate(
    model: "WiringAutoencoder",  # noqa: F821
    dataloader,
    U_q: Tensor,
    eigvals_q: Tensor,
    lam_s: float = 1.0,
    tau: float = 1.0,
    delta: float = 0.01,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute all active v2 benchmark metrics on a full DataLoader pass.

    Each batch is passed through model.forward(x, U_q, eigvals_q) and the
    9-key output dict is used to accumulate metric values.  No metric
    requires model internals beyond this dict.

    Metrics returned
    ----------------
    kl_S            Mean KL for the spectral basis posterior.
    kl_tau          Mean KL for the mode-weight posterior.
    active_modes    Mean active mode count across batches.
    memory_snr      SNR proxy from the spectral artefact W_hat.
    mean_elbo       Mean negative ELBO (lower = better fit).
    spectral_entropy Mean spectral entropy of eigvals_q.

    Note: linear_probe_acc and elbo_bayes_factor are not computed here
    because they require labels (probe) or a second index (Bayes factor).
    Call them directly via linear_probe_acc() and elbo_bayes_factor().

    Parameters
    ----------
    model : WiringAutoencoder
        Model in eval mode.
    dataloader : Iterable yielding (x, node_idx) pairs.
        x: (B, D) feature tensor.  node_idx: (B,) long tensor.
    U_q : Tensor
        Shape (N, q).  Leading eigenvectors of the frozen L(I).
    eigvals_q : Tensor
        Shape (q,).  Corresponding eigenvalues.
    lam_s : float
        Prior precision for kl_S.  Must match the value used during training.
    tau : float
        Diffusion time scale for kl_tau.
    delta : float
        Mode-activity threshold for active_modes.
    device : torch.device or None
        If given, moves x and node_idx to this device before each forward.

    Returns
    -------
    dict  {metric_name: float}
    """
    model.eval()
    accum: Dict[str, float] = {
        "kl_S": 0.0, "kl_tau": 0.0, "active_modes": 0.0,
        "memory_snr": 0.0, "mean_elbo": 0.0, "spectral_entropy": 0.0,
    }
    n_batches = 0

    # Spectral entropy depends only on eigvals_q, constant across batches.
    s_ent = spectral_entropy(eigvals_q)

    artefact = model.extract_spectral_artefact(U_q, eigvals_q)
    W_hat    = artefact.get("W_hat")       # (1, d_model, q) from prior mean
    omega_hat = artefact.get("omega_hat")  # (q,)

    snr = 0.0
    if W_hat is not None:
        # keys: (q, d_model)
        keys = W_hat.squeeze(0).T.detach()  # (q, d_model)
        snr  = memory_snr(keys, n_stored=keys.shape[0])

    with torch.no_grad():
        for batch in dataloader:
            x, node_idx = batch
            if device is not None:
                x        = x.to(device)
                node_idx = node_idx.to(device)
            out = model(x, U_q, eigvals_q, node_idx=node_idx)

            accum["kl_S"]         += out["kl_S"].item()
            accum["kl_tau"]        += out["kl_tau"].item()
            accum["mean_elbo"]     += out["loss"].item()

            # active_modes requires omega_hat from the current forward pass.
            # We re-extract per batch to capture the data-dependent mean.
            batch_artefact = model.extract_spectral_artefact(U_q, eigvals_q)
            batch_omega    = batch_artefact.get("omega_hat")
            if batch_omega is not None:
                accum["active_modes"] += active_modes(batch_omega, delta=delta)

            n_batches += 1

    if n_batches == 0:
        return accum

    for key in ("kl_S", "kl_tau", "active_modes", "mean_elbo"):
        accum[key] /= n_batches

    accum["memory_snr"]      = snr
    accum["spectral_entropy"] = s_ent
    return accum


# ---------------------------------------------------------------------------
# compare_indices  --  ELBO Bayes-factor leaderboard
# ---------------------------------------------------------------------------

def compare_indices(
    model: "WiringAutoencoder",  # noqa: F821
    dataloader,
    index_list: Sequence[Tuple[str, Tensor, Tensor]],
    device: Optional[torch.device] = None,
) -> List[Dict[str, object]]:
    """
    Compare competing ArrowSpace indices by ELBO and return a leaderboard.

    For each index (name, U_q, eigvals_q), runs a full evaluation pass
    over dataloader and records the mean ELBO.  Then computes pairwise
    Bayes factors relative to the best-scoring index.

    Parameters
    ----------
    model : WiringAutoencoder
        Model in eval mode.  The same model weights are used for all indices;
        only U_q and eigvals_q change between evaluations.
    dataloader : Iterable yielding (x, node_idx) pairs.
    index_list : sequence of (name, U_q, eigvals_q) triples.
        name      : str -- human-readable index identifier.
        U_q       : Tensor (N, q) -- leading eigenvectors.
        eigvals_q : Tensor (q,) -- corresponding eigenvalues.
    device : torch.device or None
        Forwarded to evaluate.

    Returns
    -------
    list of dicts, each containing:
        rank            int   -- 1-based rank (1 = best ELBO).
        name            str   -- index name.
        mean_elbo       float -- mean ELBO (lower = better fit).
        bayes_factor    float -- BF relative to rank-1 index (rank 1 = 1.0).

    Sorted ascending by mean_elbo.
    """
    results = []
    for name, U_q, eigvals_q in index_list:
        metrics = evaluate(
            model=model,
            dataloader=dataloader,
            U_q=U_q,
            eigvals_q=eigvals_q,
            device=device,
        )
        results.append({"name": name, "mean_elbo": metrics["mean_elbo"]})

    # Sort ascending: lower ELBO (less negative loss) = better.
    results.sort(key=lambda r: r["mean_elbo"])

    best_elbo = results[0]["mean_elbo"] if results else 0.0
    leaderboard = []
    for rank, entry in enumerate(results, start=1):
        bf = elbo_bayes_factor(
            elbo_1=best_elbo,
            elbo_2=entry["mean_elbo"],
        )
        leaderboard.append({
            "rank":         rank,
            "name":         entry["name"],
            "mean_elbo":    entry["mean_elbo"],
            "bayes_factor": bf,
        })
    return leaderboard
