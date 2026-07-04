# SPDX-License-Identifier: Apache-2.0
"""Platform-abstraction base class for host-memory pinning.

:class:`PinMemoryBackend` is the registry base class for all
device-specific pin-memory backends.  It subclasses :class:`abc.ABC`
so the universal registry (``_registry.py``) discovers it automatically
when it scans the modules under ``lmcache/v1/platform/base/``.

The class itself is also a **no-op fallback**: because pin memory is an
optional capability, devices that do not provide a concrete
implementation can still call ``pin_memory`` / ``unpin_memory`` /
``is_pin_supported`` and get safe default responses.
"""

# Standard
import abc


class PinMemoryBackend(abc.ABC):  # noqa: B024
    """Base class for host-memory pinning per platform.

    The default implementation is a no-op that always returns ``False``,
    so platforms that do not support pinning do not need to subclass this.
    Call :func:`~lmcache.v1.platform._registry.resolve_impl` with this
    class to obtain the best available backend for a given device type;
    it will return the no-op base class itself when no concrete backend
    is registered for that device.
    """

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
        return False

    def unpin_memory(self, ptr: int) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if unpinning succeeded, False otherwise.
        """
        return False

    def is_pin_supported(self) -> bool:
        """Whether the current platform supports memory pinning.

        Returns:
            True if pinning is supported, False otherwise.
        """
        return False
