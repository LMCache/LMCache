# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.c_ops`` module.

:class:`CudaDeviceOps` subclasses :class:`DeviceOps` and binds the whole
compiled module via :meth:`DeviceOps._bind_native`, so all native ops shadow
the torch baseline. The shared types are rebound from the native pybind enums so
``int(direction)`` comparisons stay identical across the native boundary.

The native import is deferred to ``__init__`` so that subclass *discovery*
remains side-effect-free on machines without the compiled extension.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_device_ops import DeviceOps

logger = init_logger(__name__)


class CudaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cuda"

    def __init__(self) -> None:
        try:
            # First Party
            import lmcache.c_ops as native  # compiled CUDA extension
        except ImportError:
            raise RuntimeError(
                "CudaDeviceOps requires the compiled 'lmcache.c_ops' extension "
                "but it could not be imported. Ensure the package was built with "
                "CUDA support (BUILD_WITH_CUDA=1 or the cuda build profile)."
            ) from None
        self._bind_native(native)
        # Rebind types from the native pybind enums so int() comparisons
        # stay identical across the native boundary.
        type(self).TransferDirection = native.TransferDirection
        type(self).EngineKVFormat = native.EngineKVFormat
        type(self).GPUKVFormat = native.EngineKVFormat
        type(self).PageBufferShapeDesc = native.PageBufferShapeDesc
        type(self).StagingCopy = native.StagingCopy
        type(self).LaunchVar = native.LaunchVar
        type(self).BatchStep = native.BatchStep
        type(self).KernelGroupSpec = native.KernelGroupSpec
