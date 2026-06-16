"""
vdt/classifier.py  --  Vibrational Classifier (Option 6, issue #18).

Provides a depth-supervised vibrational classifier built on VDT recurrent
steps.  At each recurrent depth t the hidden state Q_t (B, N, d_model) is
projected to class logits; all K depth predictions are averaged under a
depth-supervised cross-entropy objective.

Architecture
------------
A CLS token (learnable, (1, d_model)) is prepended to the node feature
matrix.  K recurrent VDT steps propagate information across the graph.
At every step the CLS position is extracted and passed to the
ClassificationHead which produces logits via a bilinear product with a
learnable key matrix K_mat::

    score_c = CLS_t @ K_mat_c               (B,)
    logits  = stack over C classes           (B, C)

Training objective (averaged over K depths)::

    L = (1/K) sum_t  CE(logits_t, y)
          + mu1 * (1/K) sum_t  tr(Q_t^T L_f Q_t) / (B*N)
          + mu2 * (1/K) sum_t  ||rho_t||_F^2 / (B*N^2)

where L_f is the frozen index Laplacian and rho_t is the signed density
matrix at step t (from SignedDensityMatrix).

Spectral memory initialisation (v2 upgrade)
--------------------------------------------
After loading a pre-trained WiringAutoencoderV2 checkpoint:

    memory = SpectralAssociativeMemory.from_vdt(vdt_v2, U_q, eigvals_q,
                                                d_model=hidden_dim)
    classifier.init_from_spectral_memory(memory, freeze=True)  # condition: spectral_memory
    classifier.init_from_spectral_memory(memory, freeze=False) # condition: spectral_memory_ft

Ref: docs/v2/03-branching.md -- Option 6
Depends on: vdt/vdt.py (#17), vdt/spectral_memory.py (#28)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .vdt import VDT
from .density import SignedDensityMatrix


# ---------------------------------------------------------------------------
# ClassificationHead
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Bilinear classification head with a learnable key matrix.

    The key matrix K_mat has shape (n_classes, d_model).  Each class score
    for a CLS token embedding h (d_model,) is::

        score_c = h @ K_mat[c]   -->  logits (B, n_classes)

    The key matrix can be pre-initialised from a SpectralAssociativeMemory
    artefact (S_memory, shape d_model x d_model) by taking the leading
    n_classes rows of U from the SVD of S_memory.

    Parameters
    ----------
    d_model : int
        Hidden dimension (must match VDT output dimension).
    n_classes : int
        Number of output classes.
    """

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        # key_matrix shape: (n_classes, d_model)
        self.key_matrix = nn.Parameter(
            torch.empty(n_classes, d_model)
        )
        nn.init.orthogonal_(self.key_matrix)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits from the CLS token.

        Parameters
        ----------
        cls_token : Tensor  shape (B, d_model)

        Returns
        -------
        logits : Tensor  shape (B, n_classes)
        """
        # (B, d_model) x (d_model, n_classes) -> (B, n_classes)
        return cls_token @ self.key_matrix.T

    def init_from_memory(
        self,
        S_memory: torch.Tensor,
        freeze: bool = False,
    ) -> None:
        """
        Initialise key_matrix rows from the leading singular vectors of
        S_memory (d_model, d_model).

        Takes the top n_classes rows of U from the full SVD so that each
        class direction aligns with a dominant spectral direction of the
        pre-trained memory.  If n_classes > d_model the tail rows are
        kept at their orthogonal-init values.

        Parameters
        ----------
        S_memory : Tensor  shape (d_model, d_model)
            Memory matrix from SpectralAssociativeMemory.S_memory.
        freeze : bool
            If True, key_matrix.requires_grad is set to False after init
            (spectral_memory condition).
            If False, key_matrix remains trainable (spectral_memory_ft).
        """
        with torch.no_grad():
            U, _S, _Vh = torch.linalg.svd(S_memory, full_matrices=False)
            # U shape: (d_model, min(d_model, d_model)) = (d_model, d_model)
            k = min(self.n_classes, U.shape[1])
            self.key_matrix[:k] = U[:k]    # top-k left singular vectors
        self.key_matrix.requires_grad_(not freeze)


# ---------------------------------------------------------------------------
# VibrationalClassifier
# ---------------------------------------------------------------------------

class VibrationalClassifier(nn.Module):
    """
    Vibrational classifier: K recurrent VDT steps + depth-supervised CE.

    A CLS token (learnable embedding) is prepended to the node feature
    matrix before the first VDT step.  At each of the K recurrent steps
    the CLS position is extracted and classified.  The training loss
    averages depth-specific predictions under a three-term objective::

        L = (1/K) sum_{t=1}^{K}  CE(logits_t, y)
              + mu1 * (1/K) sum_t  smoothness_t
              + mu2 * (1/K) sum_t  density_penalty_t

    smoothness_t   = tr(Q_t^T L_f Q_t) / (B * N * d_model)
    density_penalty = ||rho_t||_F^2 / (B * N^2)

    where Q_t (B, N, d_model) is the node-feature slice of the VDT state
    (i.e. all positions except the prepended CLS), and rho_t is the
    per-sample signed density matrix at step t.

    Parameters
    ----------
    input_dim : int
        D -- node embedding dimension.
    d_model : int
        Hidden width inside VDT and the classification head.
    n_classes : int
        Number of output classes.
    depth : int
        K -- number of recurrent VDT steps.
    n_nodes : int
        N -- number of graph nodes (excluding the CLS token).
    mu1 : float
        Laplacian-smoothness penalty weight.
    mu2 : float
        Density-matrix Frobenius penalty weight.
    vdt_kwargs : dict or None
        Additional keyword arguments forwarded to VDT.__init__.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_classes: int,
        depth: int,
        n_nodes: int,
        mu1: float = 0.01,
        mu2: float = 0.01,
        vdt_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model   = d_model
        self.n_classes = n_classes
        self.depth     = depth
        self.n_nodes   = n_nodes
        self.mu1       = mu1
        self.mu2       = mu2

        # Input projection: D -> d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable CLS token: shape (1, 1, d_model) for broadcasting
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # VDT recurrent block (shared weights across all K steps)
        vdt_kw = vdt_kwargs or {}
        self.vdt = VDT(d_model=d_model, **vdt_kw)

        # Classification head
        self.head = ClassificationHead(d_model=d_model, n_classes=n_classes)

        # Per-sample signed density matrices (one per sample in the batch)
        # Instantiated lazily at first forward() call since B is not known
        # at construction time.  _density is reset whenever batch size changes.
        self._density: Optional[SignedDensityMatrix] = None
        self._density_B: int = -1

    # ------------------------------------------------------------------
    # Spectral memory initialisation
    # ------------------------------------------------------------------

    def init_from_spectral_memory(
        self,
        memory,  # SpectralAssociativeMemory -- avoid circular import
        freeze: bool = False,
    ) -> None:
        """
        Pre-initialise the classification head key matrix from a
        SpectralAssociativeMemory artefact.

        Parameters
        ----------
        memory : SpectralAssociativeMemory
            Loaded from SpectralAssociativeMemory.from_vdt(...).
        freeze : bool
            True  -> spectral_memory condition (frozen key matrix).
            False -> spectral_memory_ft condition (fine-tuned key matrix).
        """
        S_memory = memory.S_memory.to(next(self.parameters()).device)
        self.head.init_from_memory(S_memory, freeze=freeze)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,        # (B, N, D)  node features
        L_f: torch.Tensor,      # (N, N)     frozen index Laplacian
    ) -> dict[str, torch.Tensor]:
        """
        K recurrent VDT forward pass with depth-supervised predictions.

        Parameters
        ----------
        x : Tensor  shape (B, N, D)
            Node feature matrix.  D must equal input_dim.
        L_f : Tensor  shape (N, N)
            Frozen Laplacian used for the smoothness penalty.  Not used
            inside VDT; provided externally for interpretability and the
            Rayleigh-quotient penalty term.

        Returns
        -------
        dict with keys:
            logits_per_depth : list of (B, n_classes) tensors, length K
            logits            : (B, n_classes)  final-depth logits
            loss              : scalar (requires y; None if y not supplied)
            smoothness        : scalar mean smoothness across K steps
            density_penalty   : scalar mean density penalty across K steps
        """
        B, N, D = x.shape
        assert N == self.n_nodes, (
            f"Expected n_nodes={self.n_nodes}, got N={N}"
        )

        # Project input to d_model
        h = self.input_proj(x)                   # (B, N, d_model)

        # Prepend CLS token: shape (B, N+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        h   = torch.cat([cls, h], dim=1)          # (B, N+1, d_model)

        # Lazy-init density matrix module
        if self._density is None or self._density_B != B:
            self._density   = SignedDensityMatrix(n=N).to(x.device)
            self._density_B = B

        logits_per_depth: List[torch.Tensor] = []
        smoothness_acc   = x.new_zeros(1)
        density_acc      = x.new_zeros(1)

        for _ in range(self.depth):
            # VDT step: h (B, N+1, d_model) -> h' (B, N+1, d_model)
            h = self.vdt(h)

            # CLS position: index 0
            cls_out = h[:, 0, :]                  # (B, d_model)

            # Node positions for penalties: indices 1..
            Q_t = h[:, 1:, :]                     # (B, N, d_model)

            # Depth prediction
            logits_t = self.head(cls_out)          # (B, n_classes)
            logits_per_depth.append(logits_t)

            # Smoothness penalty: tr(Q_t^T L_f Q_t) / (B*N*d)
            # L_f (N,N) @ Q_t (B,N,d) -> (B,N,d) via broadcasting
            LQ   = torch.einsum("nm,bmd->bnd", L_f, Q_t)  # (B,N,d)
            smth = (Q_t * LQ).sum() / (B * N * self.d_model)
            smoothness_acc = smoothness_acc + smth

            # Density penalty: ||rho_t||_F^2 / (B*N^2)
            rho   = self._density.rho               # (N, N)
            dp    = rho.pow(2).sum() / (B * N * N)
            density_acc = density_acc + dp

        smoothness_mean   = smoothness_acc   / self.depth
        density_mean      = density_acc      / self.depth

        return {
            "logits_per_depth": logits_per_depth,
            "logits":           logits_per_depth[-1],
            "smoothness":       smoothness_mean,
            "density_penalty":  density_mean,
        }

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        out: dict[str, torch.Tensor],
        y: torch.Tensor,             # (B,) long
    ) -> torch.Tensor:
        """
        Compute the full three-term depth-supervised loss.

        Parameters
        ----------
        out : dict
            Output dict from forward().
        y : Tensor  shape (B,)  long
            Ground-truth class labels.

        Returns
        -------
        loss : scalar Tensor
        """
        ce_sum = sum(
            F.cross_entropy(logits_t, y)
            for logits_t in out["logits_per_depth"]
        )
        ce_mean = ce_sum / self.depth

        return (
            ce_mean
            + self.mu1 * out["smoothness"]
            + self.mu2 * out["density_penalty"]
        )
