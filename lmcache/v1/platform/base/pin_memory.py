# SPDX-License-Identifier: Apache-2.0
"""Platform-abstraction base class for host-memory pinning.

Concrete subclasses live in the device sub-packages (e.g.
:class:`~lmcache.v1.platform.cuda.pin_memory.CudaPinMemoryBackend`).
The universal registry discovers them automatically by scanning for
subclasses that define a ``device_type`` ClassVar.
"""

# First Party
from lmcache.v1.platform.base._base import PlatformBase


class PinMemoryBackend(PlatformBase):
    """Base class for host-memory pinning per platform.

    The default implementation is a no-op that always returns ``False``,
    so platforms that do not support pinning do not need to subclass this.

    Concrete subclasses MUST set a ``device_type`` ClassVar to the
    ``torch.device.type`` string they handle (``"cuda"``, ...).  The
    universal platform registry uses that attribute to discover and
    index them automatically.
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
