"""
Device initialisation for Wiring Autoencoder.

Priority: MPS (Apple Silicon) → CUDA → CPU.

When MPS is selected, automatically sets PYTORCH_ENABLE_MPS_FALLBACK=1
so ops unimplemented on MPS (e.g. aten::_linalg_eigh) silently fall
back to CPU without crashing.  Note that vdt/spectral.py also offloads
all eigensolver calls to CPU explicitly, so in practice the fallback env
var is belt-and-braces for any third-party code paths.
"""
from __future__ import annotations
import os
from typing import Literal, Optional
import torch


def get_device(
    force: Optional[str] = None,
    verbose: bool = True,
) -> torch.device:
    """
    Return the best available device, with MPS fallback env var set
    automatically.

    Priority (when force is None)
    -----------------------------
    1. MPS  — Apple Silicon GPU (sets PYTORCH_ENABLE_MPS_FALLBACK=1)
    2. CUDA — NVIDIA / AMD ROCm GPU
    3. CPU  — universal fallback

    Parameters
    ----------
    force : str or None
        Override auto-detection.  Accepts 'mps', 'cuda', 'cpu'.
        Useful for benchmarking or running on a specific device.
        MPS fallback env var is still set when force='mps'.
    verbose : bool
        Print the selected device and any relevant notes.

    Returns
    -------
    torch.device
    """
    if force is not None:
        d = torch.device(force)
        if force == "mps":
            _set_mps_fallback(verbose)
        if verbose:
            print(f"[VDT] Device forced to: {d}")
        return d

    if torch.backends.mps.is_available():
        _set_mps_fallback(verbose)
        return torch.device("mps")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            print(f"[VDT] CUDA selected: {name}")
        return device

    if verbose:
        print("[VDT] No GPU found, falling back to CPU.")
    return torch.device("cpu")


def _set_mps_fallback(verbose: bool) -> None:
    """Set PYTORCH_ENABLE_MPS_FALLBACK=1 if not already set."""
    if not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if verbose:
            print(
                "[VDT] MPS selected — set PYTORCH_ENABLE_MPS_FALLBACK=1.\n"
                "      All linalg.eigh calls are also offloaded to CPU "
                "explicitly in vdt/spectral.py."
            )
