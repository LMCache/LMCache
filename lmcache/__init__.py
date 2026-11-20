# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# First Party
from lmcache.logging import init_logger

# --------------------------
# Backend instance & Device detection
# --------------------------
from lmcache.v1.platform import torch_dev as torch_dev
from lmcache.v1.platform import torch_device_type as torch_device_type

try:
    # First Party
    from lmcache._version import __version__
except ImportError:
    __version__ = "unknown"

logger = init_logger(__name__)

__all__ = ["__version__", "get_include", "torch_dev", "torch_device_type"]


def get_include() -> str:
    """Return the directory of LMCache's native extension API headers.

    Add it to a CUDA extension's ``include_dirs`` to reuse LMCache's transfer
    kernels (``#include "cuda/mp_mem_kernels_detail.cuh"``) and call the
    ``lmcache.cuda_ops._ext_api`` table (``#include "cuda/lmcache_ext_api.h"``).
    Wheels ship the headers as ``lmcache/include``; a source checkout (e.g. an
    editable install) uses ``csrc`` directly.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    shipped = os.path.join(package_dir, "include")
    if os.path.isdir(shipped):
        return shipped
    return os.path.join(os.path.dirname(package_dir), "csrc")


try:
    # First Party
    from lmcache.v1.platform import resolve_device_ops

    device_ops = resolve_device_ops(torch_device_type)
    __all__.append("device_ops")
except Exception as exc:
    logger.warning(
        "No compute backend loaded; CLI-only mode (torch/numba not installed). "
        "Reason: %s",
        exc,
    )
