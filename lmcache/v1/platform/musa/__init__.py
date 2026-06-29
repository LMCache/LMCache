# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific platform primitives.
"""

# First Party
from lmcache.v1.platform.base_device_info import DeviceInfo

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class MusaDeviceInfo(DeviceInfo):
    """MUSA device information for the detection registry."""

    @property
    def device_type(self) -> str:
        return "musa"

    @property
    def torch_module_name(self) -> str:
        return "musa"

    @property
    def ops_module(self) -> str | None:
        return "lmcache.v1.platform.musa.ops"

    def is_available(self) -> bool:
        """Check MUSA availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "musa") and torch.musa.is_available()  # type: ignore[attr-defined]
        except Exception:
            return False

    @property
    def priority(self) -> int:
        return 0
