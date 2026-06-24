# SPDX-License-Identifier: Apache-2.0
"""Platform device extension namespace.

This module defines :class:`DeviceExt`, which is attached to the torch
device module as ``torch_dev.ext``.  It exposes platform-specific
capabilities (e.g. memory pinning) that do not exist on the original
torch device module.
"""


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

    def __init__(self, device_type: str) -> None:
        # First Party
        from lmcache.v1.platform.base_pin_memory import PinMemoryBackend

        if device_type == "cuda":
            # First Party
            from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend

            self._pin: PinMemoryBackend = CudaPinMemoryBackend()
        else:
            self._pin = PinMemoryBackend()

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
