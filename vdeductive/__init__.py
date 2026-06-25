"""
vdeductive -- Vibrational Deduction Transformer.

Public API
----------
Core model::

    from vdeductive import WiringAutoencoder
    model = WiringAutoencoder.from_config(cfg, E)

Spectral utilities::

    from vdeductive import (
        build_spectral_cache,
        mode_entropy_penalty,
    )

Dataset helpers::

    from vdeductive import load_dataset, make_loaders

Stability / health checks::

    from vdeductive import pre_training_checks, spectral_kl_health_check

Device selection::

    from vdeductive import get_device

Version
-------
The installed package version is available as ``vdeductive.__version__``.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("vdeductive")
except PackageNotFoundError:  # editable / source install without metadata
    __version__ = "0.0.0.dev"

from .model import WiringAutoencoder
from .device import get_device
from .dataset import load_dataset, make_loaders, NodeEmbeddingDataset
from .laplacian import DifferentiableLaplacian, MassMatrix
from .stability import pre_training_checks, spectral_kl_health_check
from .spectral import (
    mode_entropy_penalty,
)

__all__ = [
    # version
    "__version__",
    # model
    "WiringAutoencoder",
    # device
    "get_device",
    # dataset
    "load_dataset",
    "make_loaders",
    "NodeEmbeddingDataset",
    # graph / spectral
    "DifferentiableLaplacian",
    "MassMatrix",
    "build_feature_laplacian",
    "mode_entropy_penalty",
    # stability
    "pre_training_checks",
    "spectral_kl_health_check",
]
