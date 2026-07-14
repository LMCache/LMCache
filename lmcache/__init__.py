# SPDX-License-Identifier: Apache-2.0

# Standard
import sys

# First Party
from lmcache.logging import init_logger

# --------------------------
# Backend instance & Device detection
# --------------------------
from lmcache.v1.platform import torch_dev as torch_dev
from lmcache.v1.platform import torch_device_type as torch_device_type
from lmcache.v1.platform.c_ops_shim import build_c_ops_shim

try:
    # First Party
    from lmcache._version import __version__
except ImportError:
    __version__ = "unknown"

logger = init_logger(__name__)

__all__ = ["__version__", "torch_dev", "torch_device_type"]

# --------------------------
# Backward-compat ``lmcache.c_ops`` shim
# --------------------------
# The shim body lives in ``lmcache.v1.platform.c_ops_shim``; registration
# is done here because ``lmcache.c_ops`` is part of the top-level package's
# public contract and the parent-attribute mount (``globals()["c_ops"]``)
# must be set from ``lmcache/__init__.py``'s own globals so that
# ``from lmcache import c_ops`` (IMPORT_FROM bytecode) resolves correctly.
try:
    _shim = build_c_ops_shim(torch_device_type)
    sys.modules["lmcache.c_ops"] = _shim
    globals()["c_ops"] = _shim  # parent attr for IMPORT_FROM bytecode
except Exception as exc:
    logger.warning(
        "No compute backend loaded; CLI-only mode (torch/numba not installed). "
        "Reason: %s",
        exc,
    )
