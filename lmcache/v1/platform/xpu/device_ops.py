# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` binds the existing ``lmcache.xpu_ops`` SYCL module via
:meth:`DeviceOps._bind_native`; the remaining ops inherit the torch baseline --
identical to today's ``__dict__.update`` merge, just resolved through the
registry. If the SYCL extension is not built, the device stays on the torch
baseline with no degradation (``xpu_ops`` is optional today too).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_device_ops import DeviceOps

logger = init_logger(__name__)


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"

    def __init__(self) -> None:
        try:
            # First Party
            import lmcache.xpu_ops as sycl
        except ImportError:
            logger.info(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the torch "
                "baseline for all ops."
            )
            return
        self._bind_native(sycl)  # SYCL ops shadow base; the rest inherit
