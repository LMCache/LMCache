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
    """Register ``lmcache.c_ops`` from the resolved :class:`DeviceOps` instance.

    Builds a module exposing every op in :data:`OPS` (native-or-torch, already
    resolved on the instance) plus the shared types, so existing
    ``import lmcache.c_ops`` call sites keep working without change.

    The backing :class:`DeviceOps` subclass is resolved directly from the
    ops registry keyed by ``torch_device_type``.
    """
    # First Party
    from lmcache.v1.platform._device_ops_registry import get_device_ops_cls
    from lmcache.v1.platform.base_device_ops import OPS

    ops_cls = get_device_ops_cls(torch_device_type)

    ops = ops_cls()

    shim = types.ModuleType("lmcache.c_ops")
    sys.modules["lmcache.c_ops"] = shim  # claim slot early
    globals()["c_ops"] = shim  # parent attr for IMPORT_FROM bytecode
    for name in OPS:
        setattr(shim, name, getattr(ops, name))
    # Resolve types from the ops instance: native overrides (CUDA pybind, etc.)
    # take precedence over the Python fallbacks in ops_types via MRO.
    shim.TransferDirection = ops.TransferDirection  # type: ignore[attr-defined]
    shim.EngineKVFormat = ops.EngineKVFormat  # type: ignore[attr-defined]
    shim.GPUKVFormat = ops.GPUKVFormat  # type: ignore[attr-defined]
    shim.PageBufferShapeDesc = ops.PageBufferShapeDesc  # type: ignore[attr-defined]
    shim.set_shape_desc_dtype = ops.set_shape_desc_dtype  # type: ignore[attr-defined]
    shim.StagingCopy = ops.StagingCopy  # type: ignore[attr-defined]
    shim.LaunchVar = ops.LaunchVar  # type: ignore[attr-defined]
    shim.BatchStep = ops.BatchStep  # type: ignore[attr-defined]
    shim.KernelGroupSpec = ops.KernelGroupSpec  # type: ignore[attr-defined]


try:
    _install_c_ops_shim()
except Exception as exc:
    logger.warning(
        "No compute backend loaded; CLI-only mode (torch/numba not installed). "
        "Reason: %s",
        exc,
    )
