"""
Wiring Autoencoder  --  model only.

This module provides the single top-level model class::

    WiringAutoencoder   -- four-term ELBO (recon + kl_z + kl_S + kl_tau)
                          with optional active-mode floor penalty and
                          mode-entropy ceiling penalty (Option D, issue #82).

architecture (four-term ELBO + entropy ceiling, issue #27 / #82)
----------------------------------------------------------------
Assembles WiringEncoder, SpectralLoadingDecoder, and DiffusionDecoder
under the five-term variational objective.  The ELBO is *maximised*,
which is equivalent to *minimising* the following total loss::

    loss  =  NLL_recon  +  KL_z  +  KL_S  +  KL_tau  [+  floor]  [+  ceiling]

where

    NLL_recon  =  -E_q[log p(x|z,W)] / N   (per-node NLL, minimised; issue #89)
    KL_z       =   KL( q(z)  || N(0,I)       )   -- isotropic
    KL_S       =   KL( q(S)  || p(S|I)       )   -- spectral basis
    KL_tau     =   KL( q(w)  || p(w|tau,L)   )   -- mode frequency
    floor      =   nu * relu(q_min - N_active)    -- active-mode floor (issue #68)
    ceiling    =   nu_entropy * H                 -- mode-entropy ceiling (Option D, issue #82)

All four KL terms are non-negative.  Minimising their sum is equivalent to
maximising the ELBO::

    ELBO  =  E_q[log p(x|z,W)] / N  -  KL_z  -  KL_S  -  KL_tau

ELBO normalisation (issue #89, Fix B)
--------------------------------------
DiffusionDecoder.recon_loss() returns the *sum* of per-node NLL averaged
over the batch, making it O(N) larger than the per-sample-mean KL terms.
Dividing by N (self._n_nodes) brings all four ELBO terms onto the same
per-node scale.  Without this normalisation the KL gradient signal is
suppressed by a factor of ~N relative to the reconstruction gradient,
causing kl_S and kl_tau to silently collapse to zero while the ELBO
continues to descend (observed in runs 4 and 5 on Cora, N=2708).

The reported ELBO value shifts by approximately -log(N) nats after this
fix, but the gradient balance between reconstruction and all KL terms is
now correct.

KL free-bits floor (issue #89, Fix A)
--------------------------------------
After per-dimension aggregation each KL term is clamped from below::

    kl_z   = kl_z.clamp(min=free_bits_z)
    kl_S   = kl_S.clamp(min=free_bits_s)
    kl_tau = kl_tau.clamp(min=free_bits_tau)

When a term is BELOW its floor the gradient of that term is zero
(standard torch.clamp behaviour), which is the intended free-bits
mechanic: the posterior is free to ignore the prior for that term
until the KL exceeds the floor.  When the term is ABOVE the floor the
gradient flows normally.

The floors are applied AFTER per-dimension aggregation and BEFORE the
ELBO sum, matching the Kingma et al. (2016) convention.  Three new
constructor parameters control the floors:

    free_bits_z   (default 0.5 nats) -- isotropic VAE KL
    free_bits_s   (default 0.5 nats) -- spectral basis KL
    free_bits_tau (default 0.1 nats) -- mode-frequency KL

The smaller default for free_bits_tau reflects the narrower range of
kl_tau (Gamma vs Gaussian prior) and prevents the floor from
dominating the gradient when q is large.

Option D -- mode-entropy ceiling penalty (issue #82)
------------------------------------------------------
The uniform-mode attractor observed in runs 4 and 5 is broken by adding an
entropy ceiling penalty to the ELBO.  The penalty penalises HIGH Shannon
entropy across the softmax-normalised mode-weight proxy pi_k::

    pi_k      = softmax(log_a - log_b)         normalised mode weight proxy
    H         = -sum_k pi_k * log(pi_k + eps)  Shannon entropy over modes
    ceiling   = nu_entropy * mean_batch(H)

The penalty is logged as 'entropy_S' in the per-step diagnostics dict and
in the per-epoch training log CSV.  Set nu_entropy=0.0 to disable.

Stability mitigations (issue #68)
-----------------------------------
Two mitigations prevent mode-weight collapse during training:

1.  Shape-parameter floor: tau_mode_kl receives a_min (default 0.1) and
    clamps exp(log_a) to min=a_min before lgamma/digamma.  Gradient still
    flows through log_a; only the forward value seen by the special
    functions is floored.

2.  Active-mode floor penalty: active_mode_penalty(log_a, log_b, q_min, nu,
    delta) returns nu * relu(q_min - N_active) where N_active is the
    mean batch count of modes with E[omega_k] > delta.  The penalty is
    added to the total loss after the four ELBO terms.  Set nu=0 or
    q_min=0 to disable.  The penalty is NOT returned as a separate key
    in the output dict -- it is absorbed into 'loss'.

3.  N_active diagnostic: count_active_modes(log_a, log_b) returns the
    same mean active-mode count as a plain Python int (no gradient graph
    involvement) and is stored in out['N_active'] (issue #77).  This
    feeds spectral_kl_health_check in train.py so mode-collapse is
    detected correctly at the end of every epoch.

MassMatrix and L_f construction (issue #74)
--------------------------------------------
The feature-space Laplacian L_f passed to WiringEncoder is built once
from the cached eigendecomposition and a MassMatrix with mass_clip applied
to prevent the lambda=1 singularity from dominating the preconditioner.

from_config() now reads model.mass_clip from the YAML config and stores it
as self.mass_clip.  The public helper build_L_f() encapsulates the
one-time construction so train.py can call it once after the spectral
pre-computation block::

    L_f = model.build_L_f(full_eigvals, full_eigvecs)

This avoids the per-step O(N^2 q) outer-product fallback inside forward()
and ensures mass_clip is applied before any forward pass.

Feature-space Laplacian convention (arrowspace / graph-wiring)
--------------------------------------------------------------
arrowspace and graph-wiring build the Laplacian in *feature* space, not in
node space.  from_config() therefore passes E.t().contiguous() (shape D x N)
to DifferentiableLaplacian.from_embeddings() so that:

  - D features become the graph nodes.
  - The resulting Laplacian is (D x D) in feature space.
  - The adjacency matrix is computed on the matrix transpose E^T, i.e.
    the Gram matrix G = E^T E (D x D) rather than E E^T (N x N).

Passing the raw E (shape N x D) would build a node-space Laplacian (N x N),
which is incorrect for the encoder path.

Node-space Laplacian for the wiring decoder (_lap_node_base)
------------------------------------------------------------
SpectralLoadingDecoder.forward() calls
DifferentiableLaplacian.from_spectral_loading(W, L_base) which requires
L_base to be (N x N) in node space (W has shape (B, N, q)).  The
feature-space self._laplacian.base_laplacian is (D x D) and must NOT be
passed here.

from_config() therefore builds a second DifferentiableLaplacian from the
raw E (shape N x D, node space) and stores its base_laplacian as the
registered buffer self._lap_node_base (shape N x N).  This buffer is
passed to self.wiring_decoder in forward(), extract_spectral_artefact(),
and generate() instead of self._laplacian.base_laplacian.

self._laplacian (feature-space, D x D) is used exclusively inside
WiringEncoder via the lap= kwarg.  self._lap_node_base (node-space, N x N)
is used exclusively inside SpectralLoadingDecoder.

A future tracked issue (Option C) will replace this two-buffer convention
with a principled dual-Laplacian architecture.  Until that work lands,
the explicit split below is the correct and minimal fix.

n_nodes vs laplacian.n_nodes (Option A fix, issue #83)
------------------------------------------------------
After the feature-space Laplacian change, lap.n_nodes == D (the feature
dimension), NOT N (the graph node count).  All downstream buffers that
must be sized N -- self.embedding (N, D), WiringEncoder n_nodes, and
SpectralLoadingDecoder d=n_nodes -- must therefore NOT be inferred from
lap.n_nodes.  from_config() passes n_nodes=E.shape[0] explicitly so that
the correct N is always used.  laplacian.n_nodes is now unambiguously the
feature-space node count D and must not be used to derive N elsewhere.

Data flow::

    x  (B, D),  U_q (N, q),  eigvals_q (q,),  L_f (B, N, N)
      --> WiringEncoder      z (B, latent_dim), mu, log_var, log_a (B, q), log_b (B, q)
      --> z_to_q projection  z_q (B, q)
      --> SpectralLoadingDecoder  W (B, d, q), omega (B, q), S (B, q, q),
                                  L_z (B, N, N), log_var_S (B, q, q)
                                  [log_var_S is independent of S -- fix #52]
                                  [L_base passed is self._lap_node_base (N x N)]
      --> DiffusionDecoder   uses self.embedding (N, D) as the node table
                             x_hat (B, D)
      --> total loss

latent_dim vs q
---------------
``latent_dim`` is the dimension of the reparameterised VAE latent z returned by
WiringEncoder (and exposed in the output dict).  ``q`` is the number of
spectral modes consumed by SpectralLoadingDecoder and ModeWeightHead
(log_a, log_b).  They may differ.
A linear projection ``self.z_to_q`` (shape latent_dim -> q) bridges them.
When latent_dim == q the projection is the identity in spirit but is still
present to keep the data flow uniform.

Embedding table
---------------
DiffusionDecoder.forward() requires an embedding table E of shape (N, D)
(the full per-node feature matrix) so that the heat-kernel row
k_row (B, N) can be multiplied against E_b (B, N, D).  The per-batch
query x (B, D) must NOT be used as E -- its leading dimension is B, not N.

WiringAutoencoder therefore stores ``self.embedding`` as an nn.Parameter
of shape (n_nodes, input_dim), initialised to zeros.  Callers should
populate it (e.g. with pre-trained node embeddings) before training::

    model.embedding.data.copy_(pretrained_E)

forward() passes self.embedding as E by default; an explicit
``embedding_table`` kwarg overrides it for inference on a new graph.

Config dispatch (from_config)
------------------------------
WiringAutoencoder is the single canonical model class.  There is no
separate v1 class.  The ``model.version`` key in the YAML config is used
only for backward compatibility:

    model.version: 2  (or absent)  -> WiringAutoencoder  (canonical path)
    model.version: 1               -> backward-compat shim; old v1 keys
                                      are silently ignored, missing q is
                                      derived from tau_modes.  The resulting
                                      instance is identical to the v2 path.

Ref: docs//00-architecture.md
Ref: docs//05-Code.md
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Any, Tuple

from .encoder import WiringEncoder
from .diffusion_decoder import DiffusionDecoder
from .laplacian import DifferentiableLaplacian, MassMatrix


# ---------------------------------------------------------------------------
# -specific imports  (SpectralLoadingDecoder + KL helpers)
# ---------------------------------------------------------------------------

from .wiring_decoder import SpectralLoadingDecoder
from .spectral import (
    spectral_basis_kl,
    tau_mode_kl,
    active_mode_penalty,
    count_active_modes,
    mode_entropy_penalty,
)


# ---------------------------------------------------------------------------
# WiringAutoencoder  ( -- four-term ELBO + entropy ceiling, issue #27 / #82)
# ---------------------------------------------------------------------------

class WiringAutoencoder(nn.Module):
    """
    Wiring Autoencoder  -- four-term ELBO with spectral mode priors,
    active-mode floor penalty (issue #68), mode-entropy ceiling
    penalty Option D (issue #82), and KL free-bits floor (issue #89 Fix A).

    Assembles WiringEncoder, SpectralLoadingDecoder, and DiffusionDecoder
    under the variational objective.  Training *minimises*::

        loss  =  NLL_recon  +  KL_z  +  KL_S  +  KL_tau  [+  floor]  [+  ceiling]

    which is equivalent to *maximising* the ELBO::

        ELBO  =  E_q[log p(x|z,W)] / N  -  KL_z  -  KL_S  -  KL_tau

    where

        NLL_recon  =  -E_q[log p(x|z,W)] / N   (per-node NLL; issue #89 Fix B)
        KL_z       =  KL( q(z)  || N(0,I)          )   -- isotropic
        KL_S       =  KL( q(S)  || p(S|I)          )   -- spectral basis
        KL_tau     =  KL( q(w)  || p(w|tau,Lambda) )   -- mode frequency
        floor      =  nu * relu(q_min - N_active)       -- active-mode floor
                      (absorbed into 'loss'; not a separate output key)
        ceiling    =  nu_entropy * H                    -- entropy ceiling (Option D)
                      returned as out['entropy_S']

    ELBO normalisation (issue #89, Fix B)
    ----------------------------------------
    DiffusionDecoder.recon_loss() returns the sum of per-node NLL averaged
    over the batch, making it O(N) times larger than the per-sample-mean KL
    terms (kl_z, kl_S, kl_tau).  Without normalisation the KL gradient is
    suppressed by ~N relative to the reconstruction gradient, causing kl_S
    and kl_tau to collapse silently to zero.  Dividing by self._n_nodes
    (the graph node count N) brings all four ELBO terms to the same
    per-node scale and restores correct gradient balance.

    KL free-bits floor (issue #89, Fix A)
    ----------------------------------------
    Each KL term is clamped from below after per-dimension aggregation and
    before the ELBO sum::

        kl_z   = kl_z.clamp(min=free_bits_z)
        kl_S   = kl_S.clamp(min=free_bits_s)
        kl_tau = kl_tau.clamp(min=free_bits_tau)

    When a KL term is below its floor the gradient of that term is zero
    (standard torch.clamp mechanic), preventing the spectral or isotropic
    prior from being silently zeroed out by the reconstruction gradient.
    When the term is above the floor the gradient flows normally.

    This is the standard VAE free-bits fix (Kingma et al., 2016) adapted
    to the three independent KL terms in this architecture.  The floors
    are configured via constructor parameters free_bits_z, free_bits_s,
    and free_bits_tau (see Parameters section below).

    Option D -- mode-entropy ceiling (issue #82)
    -----------------------------------------------
    Shannon entropy over the softmax-normalised mode-weight proxy::

        pi_k    = softmax(log_a - log_b)
        H       = -sum_k pi_k * log(pi_k + eps)
        ceiling = nu_entropy * mean_batch(H)

    Penalises UNIFORM mode activation (high H), pushing the posterior
    toward sparse, low-entropy mode selection.  nu_entropy is read from
    training.nu_entropy in the YAML config (default 0.5).  Set to 0.0 to
    disable.  The value is returned as out['entropy_S'] so train.py can
    log it alongside the existing KL diagnostics.

    Stability mitigations (issue #68)
    -----------------------------------
    Two mitigations prevent mode-weight collapse:

    1.  Shape-parameter floor -- tau_mode_kl clamps a = exp(log_a) to
        min=a_min before lgamma/digamma.  Configured via ``a_min``
        (default 0.1).

    2.  Active-mode floor penalty -- active_mode_penalty adds
        nu * relu(q_min - N_active) to the total loss, where N_active is
        the mean batch count of modes whose expected value E[omega_k]
        exceeds delta.  Configured via ``q_min`` (default 4) and
        ``nu`` (default 1.0).  Set nu=0 or q_min=0 to disable.
        The penalty is folded into 'loss' and is NOT returned as a
        separate key in the output dict.

    3.  N_active diagnostic (issue #77) -- count_active_modes(log_a, log_b)
        returns the same mean active-mode count as a plain Python int with
        no gradient graph involvement.  It is stored as out['N_active'] so
        train.py can pass it directly to spectral_kl_health_check without
        a None placeholder.

    MassMatrix and L_f (issue #74)
    --------------------------------
    ``mass_clip`` is stored on the model and forwarded to ``MassMatrix``
    whenever build_L_f() is called.  This prevents the lambda=1
    singularity from dominating the Tikhonov preconditioner.
    Call build_L_f() once in train.py after spectral pre-computation::

        L_f = model.build_L_f(full_eigvals, full_eigvecs)

    and pass the result to every train_one_epoch / eval_epoch call.
    This avoids the per-step fallback inside forward() (which reconstructs
    L_f from U_q and eigvals_q without mass_clip each time).

    Feature-space Laplacian convention (arrowspace / graph-wiring)
    ---------------------------------------------------------------
    from_config() passes E.t().contiguous() (shape D x N) to
    DifferentiableLaplacian.from_embeddings() so that the D features become
    the graph nodes and the resulting Laplacian is (D x D) in feature space.
    The adjacency matrix is therefore computed on E^T (the transpose of the
    node-feature matrix), yielding the Gram matrix G = E^T E (D x D) rather
    than E E^T (N x N).  This is the arrowspace convention.

    Node-space Laplacian for the wiring decoder (_lap_node_base)
    ------------------------------------------------------------
    SpectralLoadingDecoder requires L_base of shape (N x N) in node space.
    from_config() builds a second DifferentiableLaplacian from E (N x D)
    directly and stores its base_laplacian as the registered buffer
    self._lap_node_base (N x N).  All three wiring-decoder call sites
    (forward, extract_spectral_artefact, generate) pass this buffer as
    L_base instead of self._laplacian.base_laplacian (which is D x D).
    self._laplacian is used exclusively inside WiringEncoder.

    n_nodes vs laplacian.n_nodes (Option A, issue #83)
    ----------------------------------------------------
    Because the Laplacian operates in feature space, lap.n_nodes == D, not N.
    All downstream buffers sized N (self.embedding, WiringEncoder, and
    SpectralLoadingDecoder) must receive N = E.shape[0] from the caller.
    from_config() supplies n_nodes=E.shape[0] explicitly; __init__ no longer
    infers n_nodes from the Laplacian when a caller passes it directly.
    See the tracked Option C issue for the dual-Laplacian architecture that
    will make this split explicit and principled.

    Note: the Laplacian-precision latent KL (Term 2, kl_lap) has been
    removed per PR #35.  L_z is synthesised inside SpectralLoadingDecoder
    and is not returned from forward().

    latent_dim vs q
    ---------------
    ``latent_dim`` controls the dimension of the reparameterised VAE latent
    z (and hence mu, log_var).  ``q`` is the number of spectral modes used
    by SpectralLoadingDecoder, ModeWeightHead (log_a, log_b shape (B, q)),
    and the KL priors.  They are independent.
    A linear projection ``self.z_to_q`` maps z from latent_dim to q before
    the wiring decoder.

    Embedding table
    ---------------
    ``self.embedding`` is a learnable nn.Parameter of shape
    ``(n_nodes, input_dim)``.  It is the canonical node embedding table E
    passed to DiffusionDecoder at every forward call.  Initialised to zeros;
    callers should overwrite it with pre-trained embeddings before training::

        model.embedding.data.copy_(pretrained_E)

    The per-batch query ``x`` (shape ``(B, D)``) is the reconstruction
    target and encoder input; it is NOT the embedding table.

    Data flow::

        x  (B, D),  U_q (N, q),  eigvals_q (q,),  L_f (B, N, N)
          --> WiringEncoder
                z        (B, latent_dim)  -- reparameterised VAE latent
                mu       (B, latent_dim)
                log_var  (B, latent_dim)
                log_a    (B, q)           -- Gamma shape log-params
                log_b    (B, q)           -- Gamma rate  log-params
          --> z_to_q  (linear, latent_dim -> q)
                z_q  (B, q)
          --> SpectralLoadingDecoder  (L_base = self._lap_node_base, N x N)
                W          (B, feat_dim, q) -- spectral loading matrix
                omega      (B, q)           -- mode weights
                S          (B, q, q)        -- spectral coeff matrix (posterior mean)
                L_z        (B, N, N)        -- synthesised Laplacian
                log_var_S  (B, q, q)        -- independent posterior log-variance (fix #52)
          --> DiffusionDecoder
                uses self.embedding  (N, D) as the node embedding table
                x_hat    (B, D)
          --> total loss

    Parameters
    ----------
    input_dim : int
        D -- node embedding dimension.
    latent_dim : int
        Dimension of the isotropic VAE latent z, mu, and log_var returned
        by WiringEncoder and exposed in the output dict.
    hidden_dim : int
        Per-node feature channel width (feat_dim) used inside WiringEncoder
        and also the MLP width of DiffusionDecoder.
    q : int
        Number of spectral modes; must match U_q.shape[1] at runtime.
        SpectralLoadingDecoder, ModeWeightHead, and the KL priors all
        operate on q modes.  Independent of latent_dim.
    n_nodes : int or None
        Graph node count N.  Inferred from laplacian.n_nodes when None.
        WARNING: after the feature-space Laplacian change (issue #83),
        laplacian.n_nodes == D (feature dim), not N.  Callers that rely on
        this fallback will get the wrong value.  Always pass n_nodes
        explicitly when using a feature-space Laplacian.
    tau_modes : int
        Number of eigenvectors kept by DiffusionDecoder.
    lam_s : float
        Weight for kl_S (spectral basis KL).  Default 0.01.
    tau : float
        Diffusion time scale for kl_tau (mode frequency KL).  Default 0.5.
    laplacian : DifferentiableLaplacian
        Feature-space Laplacian module (D x D).  Passed to WiringEncoder
        via the lap= kwarg.  Its base_laplacian is (D x D) and must NOT
        be passed to SpectralLoadingDecoder -- use lap_node instead.
    lap_node : DifferentiableLaplacian or None
        Node-space Laplacian module (N x N).  Its base_laplacian is stored
        as the registered buffer self._lap_node_base and passed to
        SpectralLoadingDecoder.forward() as L_base at every step.
        When None, a RuntimeError is raised during the first wiring-decoder
        call (base_laplacian will be None).  from_config() always supplies
        this argument.
    n_layers : int
        Number of VDT blocks inside WiringEncoder.  Default 4.
    n_heads : int
        Attention heads per VDT block.  Default 4.
    dropout : float
        Dropout probability inside WiringEncoder.  Default 0.1.
    a_min : float
        Floor for the Gamma shape parameter a = exp(log_a) used inside
        tau_mode_kl.  Prevents full collapse to a near-zero spike.
        Default 0.1.  Set to 0.0 to disable.
    q_min : int
        Minimum number of active spectral modes required.  When the mean
        batch count of modes with E[omega_k] > delta falls below q_min,
        the active-mode floor penalty (nu * relu(q_min - N_active)) is added
        to the total loss.  Default 4.  Set to 0 to disable.
    nu : float
        Lagrange multiplier weight for the active-mode floor penalty.
        Default 1.0.  Set to 0.0 to disable.
    nu_entropy : float
        Weight for the mode-entropy ceiling penalty (Option D, issue #82).
        Penalises HIGH Shannon entropy across the mode-weight proxy pi_k.
        Default 0.5.  Set to 0.0 to disable.  Corresponds to config key
        training.nu_entropy.  The resulting penalty is logged as 'entropy_S'.
    free_bits_z : float
        Free-bits floor (nats) applied to kl_z after per-dimension
        aggregation (issue #89 Fix A).  kl_z = kl_z.clamp(min=free_bits_z).
        When kl_z is below the floor the gradient of kl_z is zero; the
        posterior is free to match the prior on those dimensions.  The floor
        prevents the isotropic KL from being dominated to zero by the
        reconstruction gradient.  Default 0.5.  Set to 0.0 to disable.
    free_bits_s : float
        Free-bits floor (nats) applied to kl_S (spectral basis KL) after
        per-dimension aggregation (issue #89 Fix A).
        kl_S = kl_S.clamp(min=free_bits_s).  Default 0.5.
        Set to 0.0 to disable.
    free_bits_tau : float
        Free-bits floor (nats) applied to kl_tau (mode-frequency KL) after
        per-dimension aggregation (issue #89 Fix A).
        kl_tau = kl_tau.clamp(min=free_bits_tau).  Default 0.1.
        Smaller default than free_bits_z / free_bits_s because the Gamma
        prior range for kl_tau is narrower.  Set to 0.0 to disable.
    mass_clip : float
        Maximum allowed value for any entry of MassMatrix.M_diag.  Passed
        to MassMatrix whenever build_L_f() is called (issue #74).  Prevents
        the lambda=1 singularity from dominating the preconditioner.
        Default 1e3.  Use 1e4 for sparse graphs.  Set to 1e6 to recover
        the pre-#74 (unclipped) behaviour.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        q: int,
        tau_modes: int,
        lam_s: float,
        tau: float,
        laplacian: DifferentiableLaplacian,
        n_nodes: Optional[int] = None,
        lap_node: Optional[DifferentiableLaplacian] = None,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        a_min: float = 0.1,
        q_min: int = 4,
        nu: float = 1.0,
        nu_entropy: float = 0.5,
        free_bits_z: float = 0.5,
        free_bits_s: float = 0.5,
        free_bits_tau: float = 0.1,
        mass_clip: float = 1e3,
    ) -> None:
        super().__init__()
        self.q = q
        self.latent_dim = latent_dim
        self.lam_s = lam_s
        self.tau = tau
        self.tau_modes = tau_modes
        self.a_min = a_min
        self.q_min = q_min
        self.nu = nu
        self.nu_entropy = nu_entropy
        self.free_bits_z = free_bits_z
        self.free_bits_s = free_bits_s
        self.free_bits_tau = free_bits_tau
        self.mass_clip = mass_clip

        # Resolve n_nodes from the laplacian when not supplied explicitly.
        # WARNING (issue #83): after the feature-space Laplacian change,
        # laplacian.n_nodes == D (feature dim), not N (graph node count).
        # from_config() always supplies n_nodes=E.shape[0] explicitly to
        # avoid this ambiguity.  Direct callers should do the same.
        if n_nodes is None:
            n_nodes = laplacian.n_nodes
        self._n_nodes = n_nodes

        # Canonical node embedding table E of shape (N, D).
        # DiffusionDecoder requires the full (N, D) table, not the per-batch
        # query x (B, D).  Initialised to zeros; overwrite with pre-trained
        # embeddings before training: model.embedding.data.copy_(E_pretrained)
        self.embedding = nn.Parameter(
            torch.zeros(n_nodes, input_dim), requires_grad=True
        )

        # WiringEncoder: latent_dim controls the VAE reparameterisation
        # space (z, mu, log_var).  q controls ModeWeightHead output shape
        # (log_a, log_b) -- passed explicitly to keep them independent.
        self.encoder = WiringEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            q=q,
            n_nodes=n_nodes,
            feat_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            use_isotropic_kl=True,
            dropout=dropout,
        )

        # Bridge from VAE latent space to spectral mode space.
        # Maps z (B, latent_dim) -> z_q (B, q) for SpectralLoadingDecoder.
        # When latent_dim == q this is still an explicit linear layer so the
        # data flow is always consistent.
        self.z_to_q = nn.Linear(latent_dim, q, bias=False)

        # SpectralLoadingDecoder takes (q, d) at init; L_base is supplied
        # at forward() time via self._lap_node_base (N x N node-space buffer).
        self.wiring_decoder = SpectralLoadingDecoder(
            q=q,
            d=n_nodes,
        )
        self.diffusion_decoder = DiffusionDecoder(
            embedding_dim=input_dim,
            hidden_dim=hidden_dim,
            tau_modes=tau_modes,
        )
        # Feature-space Laplacian (D x D) -- used inside WiringEncoder only.
        self._laplacian = laplacian

        # Node-space base Laplacian (N x N) -- used inside SpectralLoadingDecoder.
        # Stored as a non-trainable buffer so .to(device) moves it automatically.
        # Built from E (N x D) in from_config(); None when direct callers omit
        # lap_node (legacy path).
        node_lap_base = (
            lap_node.base_laplacian
            if lap_node is not None and lap_node.base_laplacian is not None
            else torch.zeros(n_nodes, n_nodes)
        )
        self.register_buffer("_lap_node_base", node_lap_base)

    @classmethod
    def from_config(cls, cfg: dict, E: torch.Tensor) -> "WiringAutoencoder":
        """
        Construct a WiringAutoencoder from the repository config dict.

        This classmethod preserves compatibility with call sites such as
        `WiringAutoencoder.from_config(cfg, E)` used by train.py, while
        delegating the actual version dispatch to the module-level
        `from_config(cfg, E)` function tested in tests/test_model.py.
        """
        return from_config(cfg, E)

    # ------------------------------------------------------------------
    # build_L_f  (issue #74)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_L_f(
        self,
        full_eigvals: torch.Tensor,
        full_eigvecs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the feature-space Laplacian L_f once from the full
        eigendecomposition, applying MassMatrix clipping before
        reconstruction.

        This should be called once in train.py after the spectral
        pre-computation block and the result passed to every
        train_one_epoch / eval_epoch call::

            L_f = model.build_L_f(full_eigvals, full_eigvecs)  # (N, N)

        Internally this builds MassMatrix(full_eigvals, tau=self.tau,
        mass_clip=self.mass_clip) to obtain clipped diagonal weights
        M_diag, then reconstructs::

            L_f = U diag(M_diag * eigvals) U^T

        so that the passed L_f already encodes the mass-weighted spectral
        structure.  Modes near the lambda=1 singularity are damped by
        mass_clip instead of dominating the encoder attention.

        Parameters
        ----------
        full_eigvals : torch.Tensor
            Shape (N,) -- full eigenvalue spectrum of the base Laplacian.
        full_eigvecs : torch.Tensor
            Shape (N, N) -- corresponding eigenvectors (columns).

        Returns
        -------
        torch.Tensor  shape (N, N)
            Mass-weighted feature-space Laplacian, ready to be passed
            as L_f to forward() / train_one_epoch / eval_epoch.
            The caller should move it to the training device after this
            call if needed:  L_f = model.build_L_f(...).to(device)
        """
        mass = MassMatrix(
            full_eigvals,
            tau=self.tau,
            mass_clip=self.mass_clip,
        )
        # Weight eigenvalues by the clipped mass diagonal, then reconstruct
        # L_f = U diag(M * lambda) U^T.  This is a (N, N) dense matrix;
        # it is computed once and cached by the caller.
        weighted_eigvals = mass.M_diag * full_eigvals   # (N,)
        L_f = full_eigvecs @ torch.diag(weighted_eigvals) @ full_eigvecs.t()   # (N, N)
        return L_f

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        U_q: torch.Tensor,
        eigvals_q: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
        spectral_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        L_f: Optional[torch.Tensor] = None,
        embedding_table: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """
        Single forward pass returning all ELBO components.

        Parameters
        ----------
        x : torch.Tensor
            Query node embeddings used as encoder input and reconstruction
            target.  Shape (B, D).  NOTE: this is NOT the embedding table
            passed to DiffusionDecoder -- see ``embedding_table`` below.
        U_q : torch.Tensor
            Leading q eigenvectors of the frozen index Laplacian L(I).
            Shape (N, q).  Passed to both WiringEncoder (as eigvecs) and
            SpectralLoadingDecoder.
        eigvals_q : torch.Tensor
            Corresponding q eigenvalues of L(I).  Shape (q,).
            Used by spectral_basis_kl and tau_mode_kl priors.
        node_idx : torch.Tensor or None
            Long tensor (B,) selecting one node per sample.  Required for
            per-node reconstruction loss; pass None only for diagnostics.
        spectral_cache : tuple(eigvals, eigvecs) or None
            Pre-computed full eigendecomposition of the base Laplacian for
            DiffusionDecoder.  Avoids an O(N^3) eigensolver at each step.
        L_f : torch.Tensor or None
            Feature-space Laplacian (B, N, N) or (N, N) for WiringEncoder.
            Strongly recommended: build once via model.build_L_f() (issue
            #74) so that MassMatrix clipping is applied before any forward
            pass.  When None, a uniform Laplacian is synthesised from U_q
            and eigvals_q on the fly (no mass_clip applied).
        embedding_table : torch.Tensor or None
            Full node embedding table of shape (N, D) to pass to
            DiffusionDecoder instead of the stored self.embedding buffer.
            Use this to override the buffer at inference time on a new graph.
            When None (default), self.embedding is used.

        Returns
        -------
        dict with exactly 11 keys:
            loss       -- total loss scalar (minimise this; equals negative ELBO
                          plus the active-mode penalty, up to constants).
                          Computed as recon + kl_z + kl_S + kl_tau + floor + ceiling,
                          where floor is NOT returned as a separate key.
                          All three KL terms have been clamped by their
                          respective free-bits floors before summation
                          (issue #89 Fix A).
            recon      -- per-node NLL reconstruction term (issue #89 Fix B):
                          DiffusionDecoder.recon_loss(x, x_hat) / N.
                          Normalised by N so all four ELBO terms share the
                          same per-node scale and the KL gradient is not
                          suppressed relative to the reconstruction gradient.
            kl_z       -- isotropic KL  KL(q(z) || N(0,I)), after free-bits
                          floor: kl_z.clamp(min=self.free_bits_z).
            kl_S       -- spectral basis KL  KL(q(S) || p(S|I)), after
                          free-bits floor: kl_S.clamp(min=self.free_bits_s).
            kl_tau     -- mode frequency KL  KL(q(w) || p(w|tau,L)), after
                          free-bits floor: kl_tau.clamp(min=self.free_bits_tau).
            entropy_S  -- mode-entropy ceiling penalty (Option D, issue #82):
                          nu_entropy * mean_batch(H) where H is the Shannon
                          entropy of softmax(log_a - log_b).  Zero when
                          nu_entropy=0.0.  Included in 'loss'.
            x_hat      -- (B, D) reconstructed embeddings
            z          -- (B, latent_dim) latent samples
            mu         -- (B, latent_dim) posterior means
            log_var    -- (B, latent_dim) posterior log-variances
            N_active   -- int, mean count of modes with E[omega_k] > delta
                          across the batch.  Computed under no_grad via
                          count_active_modes() (issue #77).  Feeds
                          spectral_kl_health_check in train.py.
        """
        B = x.shape[0]

        # Build a feature-space Laplacian from the spectral basis when the
        # caller does not supply one explicitly.
        # Preferred path: caller calls model.build_L_f(full_eigvals, full_eigvecs)
        # once before training (issue #74) and passes L_f here every step.
        # Fallback (no mass_clip applied):
        #   L_f = U_q diag(eigvals_q) U_q^T  expanded to (B, N, N).
        if L_f is None:
            L_f_base = U_q @ torch.diag(eigvals_q) @ U_q.t()   # (N, N)
            L_f = L_f_base.unsqueeze(0).expand(B, -1, -1)       # (B, N, N)
        elif L_f.ndim == 2:
            L_f = L_f.unsqueeze(0).expand(B, -1, -1)

        # --- Encode -------------------------------------------------------
        # WiringEncoder returns (z, mu, log_var, log_a, log_b).
        # z, mu, log_var are shape (B, latent_dim).
        # log_a, log_b are shape (B, q) from the mode-weight heads.
        z, mu, log_var, log_a, log_b = self.encoder(
            x,
            L_f=L_f,
            eigvecs=U_q,
            lap=self._laplacian,
        )

        # --- Project latent -> spectral modes ----------------------------
        # z is (B, latent_dim); SpectralLoadingDecoder expects (B, q).
        z_q = self.z_to_q(z)   # (B, q)

        # --- Spectral decode (wiring) -------------------------------------
        # SpectralLoadingDecoder.forward() requires L_base of shape (N x N)
        # in node space.  self._lap_node_base is the registered buffer built
        # from E (N x D) in from_config().  Do NOT use
        # self._laplacian.base_laplacian here -- that is (D x D) feature-space.
        # Returns 5 values (fix #52): W, omega, S, L_z, log_var_S.
        # log_var_S is an independent head output -- NOT derived from S.
        W, omega, S, L_z, log_var_S = self.wiring_decoder(
            z_q, U_q, self._lap_node_base
        )

        # Embedding table E must be (N, D) -- the full node feature matrix.
        # Using x (shape (B, D)) here would cause a shape mismatch inside
        # TauModeDiffusion when k_row (B, N) is bmm'd against E_b (B, N, D).
        E = embedding_table if embedding_table is not None else self.embedding

        # --- Diffusion decode ---------------------------------------------
        x_hat = self.diffusion_decoder(
            L_z, E, node_idx=node_idx, eig_cache=spectral_cache
        )  # (B, D)

        # --- ELBO terms ---------------------------------------------------
        # Term 1: per-node NLL reconstruction term (issue #89 Fix B).
        # recon_loss() returns the *sum* of per-node NLL averaged over the
        # batch, making it O(N) times larger than the per-sample-mean KL
        # terms.  Dividing by self._n_nodes normalises all four ELBO terms
        # to the same per-node scale and restores correct gradient balance
        # between reconstruction and the KL regularisers.
        recon = self.diffusion_decoder.recon_loss(x, x_hat) / self._n_nodes

        # Term 2: isotropic KL  KL(q(z) || N(0,I))  -- uses full latent_dim z
        kl_z = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        # Term 3: spectral basis KL  KL(q(S) || p(S|I))
        # log_var_S is the independent posterior log-variance from
        # SpectralLoadingDecoder.log_var_S_head -- not derived from S.
        # This is the fix for issue #52: the old proxy
        #   log_var_S = (S.pow(2) + 1e-6).log()
        # conflated the posterior mean and variance and produced an invalid
        # KL gradient.  log_var_S now comes from an independent linear head.
        kl_S = spectral_basis_kl(S, log_var_S, eigvals_q, lam_s=self.lam_s)

        # Term 4: mode frequency KL  KL(q(w) || p(w|tau,Lambda))
        # log_a, log_b are (B, q) -- matching eigvals_q shape (q,).
        # a_min clamps exp(log_a) >= a_min before lgamma/digamma (issue #68).
        kl_tau = tau_mode_kl(log_a, log_b, eigvals_q, tau=self.tau, a_min=self.a_min)

        # Free-bits floors (issue #89 Fix A).
        # Applied after per-dimension aggregation and before the ELBO sum.
        # When a KL term is below the floor its gradient is zero (torch.clamp
        # behaviour); the posterior is free to ignore the prior on those
        # dimensions.  When the term is above the floor the gradient flows
        # normally.  Set free_bits_* to 0.0 in the config to disable.
        kl_z   = kl_z.clamp(min=self.free_bits_z)
        kl_S   = kl_S.clamp(min=self.free_bits_s)
        kl_tau = kl_tau.clamp(min=self.free_bits_tau)

        # Active-mode floor penalty (issue #68): nu * relu(q_min - N_active).
        # Zero when nu=0 or q_min=0 (see active_mode_penalty docstring).
        # Folded into 'loss' only -- not returned as a separate output key.
        penalty = active_mode_penalty(log_a, log_b, q_min=self.q_min, nu=self.nu)

        # Mode-entropy ceiling penalty (Option D, issue #82).
        # Penalises HIGH Shannon entropy across softmax(log_a - log_b).
        # Zero when nu_entropy=0.0 (fast-path; no gradient allocated).
        # Returned as out['entropy_S'] for logging in train.py.
        entropy_ceil = mode_entropy_penalty(log_a, log_b, nu_entropy=self.nu_entropy)

        # N_active diagnostic (issue #77): plain Python int, no gradient graph.
        # count_active_modes() shares the E[omega_k] = a/b computation with
        # active_mode_penalty() but runs under no_grad and returns an int.
        # The value feeds spectral_kl_health_check in train.py each epoch.
        n_active = count_active_modes(log_a, log_b)

        # Total loss (all four KL terms are non-negative by construction).
        # Minimising this loss is equivalent to maximising the ELBO.
        loss = recon + kl_z + kl_S + kl_tau + penalty + entropy_ceil

        return {
            "loss": loss,
            "recon": recon,
            "kl_z": kl_z,
            "kl_S": kl_S,
            "kl_tau": kl_tau,
            "entropy_S": entropy_ceil,
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "N_active": n_active,
        }

    # ------------------------------------------------------------------
    # extract_spectral_artefact
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_spectral_artefact(
        self,
        U_q: torch.Tensor,
        eigvals_q: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Produce the spectral artefact dict used for post-training memory
        construction (consumed by SpectralAssociativeMemory, issue #28).

        The model is called at the posterior mode (z = mu = 0, the prior
        mean) to obtain the mean loading matrix W_hat and mode weights
        omega_hat.  The outer-product Hopfield memory matrix S_memory is
        then assembled as::

            S_memory = sum_k E[omega_k] * d_theta(w_hat_k) * w_hat_k^T

        where:
          - w_hat_k   is the k-th column of W_hat  (feat_dim,)
          - d_theta(.) is the decoder response at spectral direction w_hat_k
            (approximated as W_hat[:, k] for a linear decoder)
          - E[omega_k] = omega_hat[:, k].mean() for the scalar weight

        Shape:
          W_hat      (1, feat_dim, q)
          omega_hat  (q,)  -- averaged over the single-sample batch
          S_memory   (feat_dim, feat_dim)

        Parameters
        ----------
        U_q : torch.Tensor
            Leading q eigenvectors of L(I).  Shape (N, q).
        eigvals_q : torch.Tensor
            Leading q eigenvalues of L(I).  Shape (q,).

        Returns
        -------
        dict with keys: W_hat, omega_hat, S_memory
        """
        device = next(self.parameters()).device
        # Sample from the prior mean z = 0 to obtain the posterior-mode artefact.
        # z_prior is in latent_dim space; project to q before decoding.
        z_prior = torch.zeros(1, self.latent_dim, device=device)
        z_q = self.z_to_q(z_prior)  # (1, q)

        # Unpack 5 values; log_var_S is not needed for the artefact.
        # Pass self._lap_node_base (N x N) -- NOT self._laplacian.base_laplacian (D x D).
        W_hat, omega_raw, S, _L_z, _log_var_S = self.wiring_decoder(
            z_q, U_q, self._lap_node_base
        )
        # W_hat : (1, feat_dim, q)
        # omega_raw : (1, q)

        omega_hat = omega_raw.squeeze(0)   # (q,)
        W = W_hat.squeeze(0)               # (feat_dim, q)
        d_model = W.shape[0]

        S_memory = torch.zeros(d_model, d_model, device=device)
        for k in range(self.q):
            w_k = W[:, k]                  # (feat_dim,)
            S_memory += omega_hat[k] * torch.outer(w_k, w_k)

        return {
            "W_hat": W_hat,          # (1, feat_dim, q)
            "omega_hat": omega_hat,  # (q,)
            "S_memory": S_memory,    # (feat_dim, feat_dim)
        }

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        U_q: torch.Tensor,
        eigvals_q: torch.Tensor,
        E: torch.Tensor,
        n_samples: int = 1,
        node_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample z ~ N(0, I) and decode to reconstructed embeddings.

        Parameters
        ----------
        U_q : torch.Tensor
            Shape (N, q).
        eigvals_q : torch.Tensor
            Shape (q,).
        E : torch.Tensor
            Full embedding table (N, D).
        n_samples : int
            Number of samples to generate.
        node_idx : torch.Tensor or None
            Long tensor (n_samples,) selecting one node per sample.

        Returns
        -------
        torch.Tensor  (n_samples, D) or (n_samples, N, D)
        """
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        z_q = self.z_to_q(z)   # (n_samples, q)
        # Unpack 5 values; log_var_S is not needed for generation.
        # Pass self._lap_node_base (N x N) -- NOT self._laplacian.base_laplacian (D x D).
        W, omega, S, L_z, _log_var_S = self.wiring_decoder(
            z_q, U_q, self._lap_node_base
        )
        return self.diffusion_decoder(L_z, E, node_idx=node_idx)


# ---------------------------------------------------------------------------
# from_config  --  factory
# ---------------------------------------------------------------------------

def from_config(
    cfg: dict[str, Any],
    E: torch.Tensor,
) -> "WiringAutoencoder":
    """
    Build WiringAutoencoder from a parsed YAML config dict.

    WiringAutoencoder is the single canonical model class.  There is no
    separate v1 class.  The ``model.version`` key exists only for
    backward compatibility with old config files:

    version 2  (canonical)
        All keys should be present: latent_dim, hidden_dim, q, tau_modes,
        lam_s, tau.  Optional: n_layers, n_heads, dropout.

    version 1  (backward-compat shim)  or version absent
        Old v1 keys (n_wiring_heads, beta, alpha, use_lambda_features) are
        silently ignored.  Missing q is derived from tau_modes (or defaults
        to 8 when both are absent).  The resulting instance is identical to
        a version 2 build -- this path exists only so old config files do not
        break; it does not produce a different model class.

    mass_clip (issue #74)
        model.mass_clip is read and forwarded to WiringAutoencoder.__init__
        so that build_L_f() applies the correct clipping when called from
        train.py.  Defaults to 1e3 when absent, which is safe for
        normalised Laplacians.  Old configs without this key will behave
        identically to the new default (1e3), which is a tighter clip than
        the pre-#74 silent default of 1e6.

    nu_entropy (issue #82)
        training.nu_entropy is read and forwarded to
        WiringAutoencoder.__init__ as nu_entropy.  Defaults to 0.5 when
        absent.  Set to 0.0 in the config to disable the entropy ceiling
        penalty entirely (backward-compatible; zero triggers the fast-path
        in mode_entropy_penalty with no gradient allocation).

    free_bits (issue #89, Fix A)
        training.free_bits_z, training.free_bits_s, and
        training.free_bits_tau are read and forwarded to
        WiringAutoencoder.__init__ as free_bits_z, free_bits_s, and
        free_bits_tau respectively.  They default to 0.5 / 0.5 / 0.1
        when absent, which are safe conservative values for Cora-scale
        graphs.  Set all three to 0.0 in the config to recover the
        pre-fix behaviour for ablation studies.

    n_nodes vs laplacian.n_nodes (Option A fix, issue #83)
        After the feature-space Laplacian change, lap.n_nodes == D (feature
        dimension), not N (graph node count).  All downstream buffers sized
        by N -- self.embedding (N, D), WiringEncoder n_nodes, and
        SpectralLoadingDecoder d=n_nodes -- must NOT use lap.n_nodes.
        This function passes n_nodes=E.shape[0] explicitly so that the
        correct N is always used regardless of what the Laplacian reports.
        This is the single source of truth for N in from_config().

    Two-Laplacian construction (node-space L_base fix)
        SpectralLoadingDecoder.forward() requires L_base of shape (N x N)
        in node space.  A second DifferentiableLaplacian is therefore built
        from the raw E (N x D, node space) using the same knn_k / sigma /
        normalised / sparse config as the feature-space Laplacian.  Its
        base_laplacian (N x N) is stored as the registered buffer
        self._lap_node_base in WiringAutoencoder.__init__ and passed to
        self.wiring_decoder at every forward, extract_spectral_artefact,
        and generate call.

        self._laplacian (D x D, feature-space) continues to be used
        exclusively inside WiringEncoder.

        The dual-Laplacian architecture (Option C) tracked in the issue
        will make this separation explicit and principled.  Until that work
        lands, the two-Laplacian construction below is the correct minimal
        fix.

    Feature-space Laplacian convention (arrowspace / graph-wiring)
        DifferentiableLaplacian.from_embeddings receives E.t().contiguous()
        (shape D x N) so that the D features become the graph nodes and the
        resulting Laplacian is (D x D) in feature space.  The adjacency
        matrix is built on E^T, giving the feature-space Gram matrix
        G = E^T E (D x D).  Passing E directly (N x D) would build a
        node-space Laplacian (N x N) and is incorrect for the encoder path.

    ELBO normalisation (issue #89, Fix B)
        forward() divides the raw recon_loss() output by self._n_nodes so
        that the reconstruction term is on the same per-node scale as kl_z,
        kl_S, and kl_tau.  The YAML config does not require any new key for
        this fix -- it is always active.  The reported 'recon' value in the
        training log will be approximately N times smaller than in runs
        before this fix.

    YAML config example (v2)::

        model:
          version: 2
          latent_dim: 32
          hidden_dim: 256
          q: 16
          lam_s: 0.01
          tau: 0.5
          tau_modes: 16
          n_layers: 4
          n_heads: 4
          mass_clip: 1000.0    # issue #74
        training:
          a_min: 0.1       # Gamma shape floor (issue #68)
          q_min: 4         # min active modes (issue #68)
          nu: 1.0          # active-mode penalty weight (issue #68)
          nu_entropy: 0.5  # entropy ceiling weight (Option D, issue #82)
          free_bits_z:   0.5  # KL free-bits floor for kl_z (issue #89 Fix A)
          free_bits_s:   0.5  # KL free-bits floor for kl_S
          free_bits_tau: 0.1  # KL free-bits floor for kl_tau
        graph:
          knn_k: 15
          sigma: 0.5
          normalised: true
          sparse: false
        # NOTE: from_config passes n_nodes=E.shape[0] and
        #       E.t().contiguous() to from_embeddings (feature-space Laplacian)
        #       and E directly to from_embeddings (node-space Laplacian).
        #       Callers always supply the raw (N x D) matrix E.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.  Must contain a 'model' key.
    E : torch.Tensor
        Node-feature matrix (N, D).  The transpose E.t().contiguous() is
        passed to DifferentiableLaplacian.from_embeddings() to build the
        feature-space Laplacian (D x D).  E directly is passed to build
        the node-space Laplacian (N x N) for SpectralLoadingDecoder.
        E.shape[0] = N is passed as n_nodes so that self.embedding,
        WiringEncoder, and SpectralLoadingDecoder are all sized correctly.
        The raw E is also used to initialise the model's self.embedding
        buffer via model.embedding.data.copy_(E).

    Returns
    -------
    WiringAutoencoder
    """
    mc = cfg["model"]
    tc = cfg.get("training", {})
    # version key is accepted but has no dispatch effect -- both v1 and v2
    # configs build the same WiringAutoencoder instance.
    _ = int(mc.get("version", 1))

    gc = cfg.get("graph", {})

    # Feature-space Laplacian (arrowspace / graph-wiring convention):
    # Pass E.t().contiguous() (shape D x N) so that the D features become
    # the graph nodes.  The resulting Laplacian is (D x D) in feature space,
    # with adjacency built on the feature-transpose E^T.
    # IMPORTANT: lap.n_nodes == D after this call -- do NOT use it as N.
    lap = DifferentiableLaplacian.from_embeddings(
        E.t().contiguous(),
        knn_k=gc.get("knn_k", 15),
        sigma=gc.get("sigma", 0.5),
        normalised=gc.get("normalised", True),
        sparse=gc.get("sparse", False),
    )

    # Node-space Laplacian: built from E directly (N x D) so that the N
    # graph nodes become the Laplacian nodes.  The resulting base_laplacian
    # is (N x N) and is the correct L_base for SpectralLoadingDecoder.
    # Uses the same knn_k / sigma / normalised / sparse config.
    lap_node = DifferentiableLaplacian.from_embeddings(
        E,
        knn_k=gc.get("knn_k", 15),
        sigma=gc.get("sigma", 0.5),
        normalised=gc.get("normalised", True),
        sparse=gc.get("sparse", False),
    )

    # Resolve q: v2 configs supply it explicitly; v1/legacy configs may not.
    # Fall back to tau_modes, then to 8.
    tau_modes_default = mc.get("tau_modes", 8)
    q = mc.get("q", tau_modes_default)

    # Option A fix (issue #83): pass n_nodes=E.shape[0] explicitly.
    # lap.n_nodes == D (feature dimension) after the feature-space Laplacian
    # change; all downstream buffers sized by N must use E.shape[0] instead.
    # This is the single source of truth for N inside from_config().
    n_nodes = E.shape[0]  # N -- graph node count, NOT lap.n_nodes (which is D)

    model = WiringAutoencoder(
        input_dim=E.shape[1],
        latent_dim=mc.get("latent_dim", 16),
        hidden_dim=mc.get("hidden_dim", 256),
        q=q,
        tau_modes=mc.get("tau_modes", q),
        lam_s=mc.get("lam_s", 0.01),
        tau=mc.get("tau", 0.5),
        laplacian=lap,
        lap_node=lap_node,
        n_nodes=n_nodes,  # explicit N; prevents fallback to lap.n_nodes (== D)
        n_layers=mc.get("n_layers", 4),
        n_heads=mc.get("n_heads", 4),
        dropout=mc.get("dropout", 0.1),
        # Stability mitigations (issue #68)
        a_min=float(tc.get("a_min", 0.1)),
        q_min=int(tc.get("q_min", 4)),
        nu=float(tc.get("nu", 1.0)),
        # Entropy ceiling penalty (Option D, issue #82)
        nu_entropy=float(tc.get("nu_entropy", 0.5)),
        # KL free-bits floors (issue #89, Fix A)
        free_bits_z=float(tc.get("free_bits_z", 0.5)),
        free_bits_s=float(tc.get("free_bits_s", 0.5)),
        free_bits_tau=float(tc.get("free_bits_tau", 0.1)),
        # MassMatrix clipping (issue #74)
        mass_clip=float(mc.get("mass_clip", 1e3)),
    )
    # Initialise the embedding buffer with the supplied table.
    model.embedding.data.copy_(E)
    return model
