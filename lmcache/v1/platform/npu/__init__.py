# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU platform primitives.

Registers :class:`NpuDeviceSpec` with the device-detection registry so
LMCache resolves ``torch.npu`` as an accelerator instead of falling back
to the CPU stub.  ``torch.npu`` is contributed by the ``torch_npu``
package, so it is visible on a bare ``import torch``.

Scope: the compiled ops backend (``lmcache_ascend.c_ops``) and all
NPU-specific transfer optimisation live in the external ``lmcache_ascend``
plugin; this package only wires device detection and backend selection,
so ``import lmcache`` loads the plugin's ``c_ops`` on an NPU host.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.npu.pin_memory import NpuPinMemoryBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_ops import DeviceOps

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class NpuDeviceSpec(DeviceSpec):
    """Ascend NPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "npu"

    @property
    def torch_module_name(self) -> str:
        return "npu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.npu.device_ops import NpuDeviceOps

        return NpuDeviceOps

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return NpuPinMemoryBackend

    def is_available(self) -> bool:
        """Check NPU availability without importing ``lmcache.__init__``.

        ``torch.npu`` is absent on non-Ascend hosts, and availability probing
        can raise when the CANN runtime is misconfigured, so the exception is
        swallowed and reported as "unavailable".

        Returns:
            bool: ``True`` when ``torch.npu`` is present and reports at
            least one usable device, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch

            return hasattr(torch, "npu") and torch.npu.is_available()
        except Exception:
            return False
