# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any
import logging
import sys

# Third Party
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# --------------------------
# Dynamic backend selection
# --------------------------
def _get_backend() -> Any:
    """
    Try backends in order, first successful import wins.
    """
    backend_candidates = [
        (
            "lmcache.c_ops",
            "cuda_ops",
            lambda: torch.cuda.is_available(),
        ),
        (
            "lmcache.non_cuda_equivalents",
            "python_ops",
            lambda: True,
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
            module = __import__(module_name, fromlist=["*"])
            logger.info("Using backend: %s", module_name)
            return module
        except ImportError as e:
            logger.warning("Failed to import backend %s: %s", module_name, e)
    raise ImportError("No backend could be imported for lmcache.")


# --------------------------
# Backend instance
# --------------------------
_ops = _get_backend()

sys.modules["lmcache.c_ops"] = _ops
