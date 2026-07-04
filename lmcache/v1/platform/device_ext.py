# SPDX-License-Identifier: Apache-2.0
"""Platform device extension namespace.

This module defines :class:`DeviceExt`, which is attached to the torch
device module as ``torch_dev.ext``.  It exposes platform-specific
capabilities (e.g. memory pinning) that do not exist on the original
torch device module.
"""

# First Party
from lmcache.v1.platform._registry import get_impl
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend


def _get_pin_memory_backend(device_type: str) -> type[PinMemoryBackend]:
    """Resolve the pin-memory backend class for *device_type*.

    Looks up a concrete backend in the universal registry and falls back
    to the no-op base class.

    Args:
        device_type: The ``torch.device.type`` string.

    Returns:
        A :class:`PinMemoryBackend` subclass (or the base class itself as
        a no-op fallback).
    """
    try:
        return get_impl(PinMemoryBackend, device_type)  # type: ignore[return-value]
    except ValueError:
        return PinMemoryBackend


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
        backend_cls = _get_pin_memory_backend(device_type)
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
