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
    """Register ``lmcache.c_ops`` from the resolved :class:`DeviceOps` class.

    Calls :meth:`DeviceOps._ensure_native` to lazily bind the compiled
    extension, then copies all ops + types from the class onto a
    synthetic module so ``import lmcache.c_ops`` call sites keep working.
    """
    # First Party
    from lmcache.v1.platform import resolve_device_ops_cls
    from lmcache.v1.platform.base_device_ops import OPS

    ops_cls = resolve_device_ops_cls(torch_device_type)
    ops_cls._ensure_native()

    shim = types.ModuleType("lmcache.c_ops")
    # Copy all ops from the class (resolved via MRO after native binding).
    for name in OPS:
        setattr(shim, name, getattr(ops_cls, name))
    # Copy shared types.
    shim.TransferDirection = ops_cls.TransferDirection  # type: ignore[attr-defined]
    shim.EngineKVFormat = ops_cls.EngineKVFormat  # type: ignore[attr-defined]
    shim.GPUKVFormat = ops_cls.GPUKVFormat  # type: ignore[attr-defined]
    shim.PageBufferShapeDesc = ops_cls.PageBufferShapeDesc  # type: ignore[attr-defined]
    shim.set_shape_desc_dtype = ops_cls.set_shape_desc_dtype  # type: ignore[attr-defined]
    shim.StagingCopy = ops_cls.StagingCopy  # type: ignore[attr-defined]
    shim.LaunchVar = ops_cls.LaunchVar  # type: ignore[attr-defined]
    shim.BatchStep = ops_cls.BatchStep  # type: ignore[attr-defined]
    shim.KernelGroupSpec = ops_cls.KernelGroupSpec  # type: ignore[attr-defined]
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
