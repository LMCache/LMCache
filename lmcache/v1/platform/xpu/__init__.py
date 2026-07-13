# SPDX-License-Identifier: Apache-2.0
"""XPU (Intel SYCL) platform helpers."""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec

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
    def ops_module(self) -> str | None:
        return "lmcache.xpu_ops"

    def is_available(self) -> bool:
        """Check XPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False

    def is_handle_transfer_available(self) -> bool:
        """Return whether XPU IPC handle transfer APIs are available."""
        try:
            # Third Party
            from dpctl.memory import IPCMemoryHandle, MemoryUSMDevice  # noqa: F401
        except ImportError:
            return False

        return True
