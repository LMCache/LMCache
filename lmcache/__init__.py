# SPDX-License-Identifier: Apache-2.0

# Standard
import sys
import types

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

__all__ = ["__version__", "torch_dev", "torch_device_type"]


# --------------------------
# Backward-compat ``lmcache.c_ops`` shim
# --------------------------
def _install_c_ops_shim() -> None:
    """Register ``lmcache.c_ops`` via :meth:`DeviceOps.populate_module`.

    Creates a synthetic module and asks the resolved :class:`DeviceOps`
    subclass to populate it with all ops + shared types, so existing
    ``import lmcache.c_ops`` call sites keep working without change.
    """
    # First Party
    from lmcache.v1.platform import resolve_device_ops_cls

    ops_cls = resolve_device_ops_cls(torch_device_type)

    shim = types.ModuleType("lmcache.c_ops")
    # NOTE: Do NOT register the shim in sys.modules before populate_module.
    # CudaDeviceOps._load_native() uses importlib.import_module("lmcache.c_ops")
    # to find the compiled .so; if the shim is already in sys.modules, Python
    # returns it instead of loading the real extension (circular reference).
    ops_cls.populate_module(shim)
    sys.modules["lmcache.c_ops"] = shim
    globals()["c_ops"] = shim  # parent attr for IMPORT_FROM bytecode


try:
    _install_c_ops_shim()
except Exception as exc:
    logger.warning(
        "No compute backend loaded; CLI-only mode (torch/numba not installed). "
        "Reason: %s",
        exc,
    )
