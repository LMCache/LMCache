# SPDX-License-Identifier: Apache-2.0
"""Platform-aware CUDA utility functions.

These helpers encapsulate the ``if HAS_CUDA`` guard pattern so that
business-logic modules never need to import or check ``HAS_CUDA``
directly.
"""

# Third Party
import torch

# First Party
from lmcache.v1.platform.capabilities import HAS_CUDA


def current_device_id() -> int:
    """Return the current CUDA device index, or 0 on CPU-only."""
    if HAS_CUDA:
        return torch.cuda.current_device()
    return 0


def safe_device(requested: str) -> str:
    """Downgrade *requested* device to ``"cpu"`` when CUDA is absent.

    If the caller asks for a CUDA device but the platform has no GPU,
    this silently falls back to CPU.  Non-CUDA device strings (e.g.
    ``"cpu"``, ``"xpu:0"``) are returned unchanged.
    """
    if not HAS_CUDA and requested.startswith("cuda"):
        return "cpu"
    return requested


def synchronize() -> None:
    """Call ``torch.cuda.synchronize()`` if CUDA is available."""
    if HAS_CUDA:
        torch.cuda.synchronize()


def cuda_init() -> None:
    """Call ``torch.cuda.init()`` if CUDA is available."""
    if HAS_CUDA:
        torch.cuda.init()
