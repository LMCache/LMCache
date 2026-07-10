# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.c_ops`` extension.

:class:`CudaDeviceOps` calls :func:`~..base_device_ops.bind_native` in
:meth:`_ensure_native` to layer the compiled CUDA extension on top of the
torch baseline.  If the extension is missing, a warning is logged and the
class stays on the torch fallback (soft-fail, same as XPU).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_device_ops import DeviceOps, bind_native

logger = init_logger(__name__)


class CudaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cuda"
    _native_bound: ClassVar[bool] = False

    @classmethod
    def _ensure_native(cls) -> None:
        if cls._native_bound:
            return
        cls._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.c_ops as native
        except ImportError:
            logger.warning(
                "lmcache.c_ops compiled extension not found; "
                "CudaDeviceOps stays on the torch baseline for all ops."
            )
            return
        bind_native(native)(cls)
