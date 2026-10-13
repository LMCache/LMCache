# SPDX-License-Identifier: Apache-2.0
"""XPU (Intel SYCL) platform helpers."""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.devices.xpu.device_ops import XpuDeviceOps

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class XpuDeviceSpec(DeviceSpec):
    """XPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "xpu"

    @property
    def torch_module_name(self) -> str:
        return "xpu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        return XpuDeviceOps

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        """Return the SYCL host-memory backend when it can initialize."""
        try:
            # First Party
            from lmcache.v1.platform.devices.xpu.pin_memory import (
                XpuPinMemoryBackend,
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("XPU pin-memory backend is unavailable: %s", exc)
            return None

        if XpuPinMemoryBackend.is_available():
            return XpuPinMemoryBackend
        return None

    def is_available(self) -> bool:
        """Check XPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False
