# SPDX-License-Identifier: Apache-2.0
"""Platform device extension namespace.

This module defines :class:`DeviceExt`, which is attached to the torch
device module as ``torch_dev.ext``.  It exposes platform-specific
capabilities (e.g. memory pinning) that do not exist on the original
torch device module.
"""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend


class DeviceExt:
    """Extension namespace attached as ``torch_dev.ext``.

    Holds platform-specific capabilities that do not exist on the original
    torch device module.  New capabilities can be added as methods or
    properties here without changing call-sites.

    Intended usage::

        torch_dev.ext.pin_memory(ptr, size)
        torch_dev.ext.pin_memory(ptr, size, flags)
        torch_dev.ext.unpin_memory(ptr)
        if not torch_dev.ext.is_pin_supported:
            raise RuntimeError(...)
    """

    def __init__(self, device_spec: DeviceSpec | None) -> None:
        if device_spec is not None and device_spec.pin_memory_backend is not None:
            backend_cls = device_spec.pin_memory_backend
        else:
            backend_cls = PinMemoryBackend
        self._pin: PinMemoryBackend = backend_cls()

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Pin a host memory region for DMA access.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.
            flags: Platform-specific registration flags (e.g.
                ``cudaHostRegisterDefault = 0``).

        Returns:
            True if pinning succeeded, False otherwise.
        """
        return self._pin.pin_memory(ptr, size, flags)

    def unpin_memory(self, ptr: int) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if unpinning succeeded, False otherwise.
        """
        return self._pin.unpin_memory(ptr)

    @property
    def is_pin_supported(self) -> bool:
        """Whether the current platform supports memory pinning."""
        return self._pin.is_pin_supported()
