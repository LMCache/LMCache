# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific platform primitives."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper

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
    def ops_module(self) -> str | None:
        return "lmcache.v1.platform.musa.ops"

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        # First Party
        from lmcache.v1.platform.musa.ipc_wrapper import MusaIPCWrapper

        return MusaIPCWrapper

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
        from lmcache.v1.platform.musa.ipc_wrapper import (
            is_musa_handle_transfer_available,
        )

        return is_musa_handle_transfer_available()
