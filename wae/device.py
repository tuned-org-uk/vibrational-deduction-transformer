"""
Device initialisation for Wiring Autoencoder.

Priority: MPS (Apple Silicon) → CUDA → CPU.

When MPS is selected, automatically sets PYTORCH_ENABLE_MPS_FALLBACK=1
so ops unimplemented on MPS (e.g. aten::_linalg_eigh) silently fall
back to CPU without crashing.
"""
from __future__ import annotations
import os
import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Return the best available device, with MPS fallback env var set
    automatically.

    Priority
    --------
    1. MPS  — Apple Silicon GPU (sets PYTORCH_ENABLE_MPS_FALLBACK=1)
    2. CUDA — NVIDIA / AMD ROCm GPU
    3. CPU  — universal fallback

    Parameters
    ----------
    verbose : bool
        Print the selected device and any relevant warnings.

    Returns
    -------
    torch.device
    """
    if torch.backends.mps.is_available():
        # Must be set before any MPS tensor is created.
        # os.environ is process-global so setting it here is sufficient
        # even if the import happens after torch is already loaded.
        if not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            if verbose:
                print(
                    "[WAE] MPS selected. Set PYTORCH_ENABLE_MPS_FALLBACK=1 "
                    "— ops unsupported on MPS (e.g. linalg.eigh) will "
                    "silently fall back to CPU."
                )
        return torch.device("mps")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            print(f"[WAE] CUDA selected: {name}")
        return device

    if verbose:
        print("[WAE] No GPU found, using CPU.")
    return torch.device("cpu")