# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` calls :func:`~..base_device_ops.bind_native` in
:meth:`_ensure_native` to layer the SYCL extension on top of the torch
baseline.  If the extension is not built, a warning is logged and the class
stays on the torch fallback.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_device_ops import DeviceOps, bind_native

logger = init_logger(__name__)


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"
    _native_bound: ClassVar[bool] = False

    @classmethod
    def _ensure_native(cls) -> None:
        if cls._native_bound:
            return
        cls._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.xpu_ops as sycl  # noqa: F401
        except ImportError:
            logger.warning(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the "
                "torch baseline for all ops."
            )
            return
        bind_native(sycl)(cls)
