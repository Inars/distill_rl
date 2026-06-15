"""Device selection. CUDA-only server (no MPS); falls back to CPU for tests."""

from __future__ import annotations

import torch


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the best available device.

    ``prefer`` is honored when possible; we deliberately do not use MPS (the
    target is the CUDA INRIA cluster), so anything non-CUDA collapses to CPU.
    """
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer if ":" in prefer else "cuda")
    return torch.device("cpu")
