# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any
import importlib

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

try:
    # First Party
    from lmcache._version import __version__
except ImportError:
    __version__ = "unknown"

logger = init_logger(__name__)
__all__ = ["__version__"]


# --------------------------
# Dynamic backend selection
# --------------------------
def _get_backend() -> Any:
    """
    Try backends in order, first successful import wins.
    """
    module = importlib.import_module("lmcache.non_cuda_equivalents")

    backend_candidates = [
        (
            "lmcache.c_ops",
            "xpu_ops",
            lambda: torch.xpu.is_available(),
        ),
        (
            "lmcache.c_ops",
            "cuda_ops",
            lambda: torch.cuda.is_available(),
        ),
        # should extend to more HWs..
    ]

    for module_name, backend_name, predicate in backend_candidates:
        # 1 Check whether the backend is available before importing
        try:
            if not predicate():
                logger.info(
                    "Skipping backend %s: predicate returned False",
                    module_name,
                )
                continue
        except Exception as e:
            logger.warning(
                "Skipping backend %s: predicate raised error: %s",
                module_name,
                e,
            )
            continue
        # 2 Run availability check for the backend
        try:
            backend_module = importlib.import_module(module_name)
            for name in dir(backend_module):
                # If backend implements kernels, use them but not
                # the one in non_cuda_equivalents
                setattr(module, name, getattr(backend_module, name))
            logger.info("Using backend: %s", module_name)
            break
        except Exception as e:
            logger.warning("Failed to import backend %s: %s", module_name, e)

    return module


# --------------------------
# Backend instance
# --------------------------
_ops = _get_backend()
# override lmcache.c_ops with merged module,
# in which:
#     non_cuda_equivalents as base,
#     use backend implementation if exists
c_ops = _ops
