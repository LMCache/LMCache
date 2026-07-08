# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.c_ops`` module.

:class:`CudaDeviceOps` overrides :meth:`populate_module` to layer the whole
compiled native module on top of the torch baseline, so all native ops shadow
the baseline. The shared types are rebound from the native pybind enums so
``int(direction)`` comparisons stay identical across the native boundary.

The native import is deferred to :meth:`populate_module` so that subclass
*discovery* remains side-effect-free on machines without the compiled extension.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar
import importlib

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_device_ops import DeviceOps

logger = init_logger(__name__)


def _load_native():
    """Import the compiled CUDA extension (``lmcache.c_ops``)."""
    try:
        return importlib.import_module("lmcache.c_ops")
    except ImportError:
        raise RuntimeError(
            "CudaDeviceOps requires the compiled 'lmcache.c_ops' extension "
            "but it could not be imported. Ensure the package was built with "
            "CUDA support (BUILD_WITH_CUDA=1 or the cuda build profile)."
        ) from None


class CudaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def populate_module(cls, target: object) -> None:
        native = _load_native()
        # Torch baseline first, then native shadows everything.
        super().populate_module(target)
        cls._bind_native(target, native)
        # Rebind types from the native pybind enums.
        target.TransferDirection = native.TransferDirection  # type: ignore[attr-defined]
        target.EngineKVFormat = native.EngineKVFormat  # type: ignore[attr-defined]
        target.GPUKVFormat = native.EngineKVFormat  # type: ignore[attr-defined]
        target.PageBufferShapeDesc = native.PageBufferShapeDesc  # type: ignore[attr-defined]
        target.StagingCopy = native.StagingCopy  # type: ignore[attr-defined]
        target.LaunchVar = native.LaunchVar  # type: ignore[attr-defined]
        target.BatchStep = native.BatchStep  # type: ignore[attr-defined]
        target.KernelGroupSpec = native.KernelGroupSpec  # type: ignore[attr-defined]
