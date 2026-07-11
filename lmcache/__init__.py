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
    """Register ``lmcache.c_ops`` as the resolved :class:`DeviceOps` class.

    Calls :meth:`DeviceOps.ensure_native` to lazily bind the compiled
    extension, then registers the class itself as the ``lmcache.c_ops``
    module.  Since all ops are staticmethods and types are class attrs,
    ``c_ops.multi_layer_kv_transfer(...)`` resolves with zero overhead.

    The installed ``shim`` uses the PEP 562 module-level ``__getattr__``
    hook: Python only calls it when an attribute is NOT already present
    on the module's own namespace, and it forwards the lookup live to
    ``ops_cls``. This gives ``lmcache.c_ops.foo`` MRO-based resolution
    with no dict snapshot, so later ``setattr(ops_cls, ...)`` patches
    (e.g. from tests) remain visible.
    """
    # First Party
    from lmcache.v1.platform import resolve_device_ops_cls

    ops_cls = resolve_device_ops_cls(torch_device_type)
    ops_cls.ensure_native()

    shim = types.ModuleType("lmcache.c_ops")
    shim.__getattr__ = lambda name: getattr(ops_cls, name)  # type: ignore[method-assign]
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
