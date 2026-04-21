# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any
import importlib
import sys

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
    # Third Party
    import torch

    backend_candidates = [
        (
            "lmcache.c_ops",
            "cuda_ops",
            lambda: torch.cuda.is_available(),
        ),
        # should extend to more HWs..
    ]

    imported = False
    module = None
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
            module = importlib.import_module(module_name)
            logger.info("Using backend: %s", module_name)
            imported = True
            break
        except Exception as e:
            logger.warning("Failed to import backend %s: %s", module_name, e)

    if not imported:
        try:
            logger.warning("Fallback to python backend lmcache.non_cuda_equivalents")
            module = importlib.import_module("lmcache.non_cuda_equivalents")
            logger.info("Using backend: lmcache.non_cuda_equivalents")
        except ImportError as e:
            raise ImportError("No backend could be imported for lmcache.") from e
    return module


# --------------------------
# Backend instance
# --------------------------
try:
    _ops = _get_backend()
    sys.modules["lmcache.c_ops"] = _ops
except (ImportError, ModuleNotFoundError):
    logger.debug("No compute backend loaded; CLI-only mode (torch/numba not installed)")
