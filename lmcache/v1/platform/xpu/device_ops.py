# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` overrides :meth:`populate_module` to layer the existing
``lmcache.xpu_ops`` SYCL module on top of the torch baseline; the remaining ops
keep the torch implementation -- identical to today's merge, just resolved
through the registry. If the SYCL extension is not built, the device stays on
the torch baseline with no degradation.
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

    @classmethod
    def populate_module(cls, target: object) -> None:
        super().populate_module(target)  # torch baseline
        try:
            # First Party
            import lmcache.xpu_ops as sycl
        except ImportError:
            logger.info(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the torch "
                "baseline for all ops."
            )
            return
        cls._bind_native(target, sycl)  # SYCL ops shadow base; the rest inherit
