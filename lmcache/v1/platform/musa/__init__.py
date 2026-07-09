# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific platform primitives."""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class MusaDeviceSpec(DeviceSpec):
    """MUSA device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "musa"

    @property
    def torch_module_name(self) -> str:
        return "musa"

    @property
    def ops_cls(self) -> "type[DeviceOps]":
        # First Party
        from lmcache.v1.platform.musa.device_ops import MusaDeviceOps

        return MusaDeviceOps

    def is_available(self) -> bool:
        """Check MUSA availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "musa") and torch.musa.is_available()  # type: ignore[attr-defined]
        except Exception:
            return False

    def is_handle_transfer_available(self) -> bool:
        # First Party
        from lmcache.v1.platform.musa.ipc import is_musa_handle_transfer_available

        return is_musa_handle_transfer_available()
