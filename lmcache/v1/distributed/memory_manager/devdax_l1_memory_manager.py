# SPDX-License-Identifier: Apache-2.0
"""Device-DAX L1 memory manager."""

# Standard
from typing import cast

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1ReconfigureError
from lmcache.v1.distributed.internal_api import CxlArenaDescriptor, L1MemoryDesc
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaStatus,
    DevDaxMemoryAllocator,
    DevDaxNotMappedError,
    DevDaxRemoveMode,
)
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class DevDaxL1MemoryManager(L1MemoryManager):
    """L1 memory manager for Device-DAX-backed L1 memory.

    This is a peer of
    :class:`~lmcache.v1.distributed.memory_manager.l1_memory_manager.L1MemoryManager`
    and owns the Device-DAX allocator path. The configured device range is
    the primary arena; more arenas can be added at runtime.
    """

    def __init__(self, config: L1MemoryManagerConfig) -> None:
        """Create a Device-DAX L1 memory manager.

        Args:
            config: L1 memory configuration with ``devdax_path`` set.

        Raises:
            ValueError: If ``devdax_path`` is not configured.
        """
        if not config.devdax_path:
            raise ValueError("DevDaxL1MemoryManager requires devdax_path")

        logger.debug(
            "use devdax memory allocator, devdax path is %s, devdax size is %d "
            "bytes, align bytes is %d bytes",
            config.devdax_path,
            config.size_in_bytes,
            config.align_bytes,
        )
        self._allocator = DevDaxMemoryAllocator(
            config.size_in_bytes,
            config.devdax_path,
            align_bytes=config.align_bytes,
            pool_id=config.cxl_pool_id,
            offset=config.devdax_offset_in_bytes,
        )
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

    @property
    def cxl_arena(self) -> CxlArenaDescriptor | None:
        """Return the shared-pool descriptor, or None for ordinary Device-DAX."""
        return cast(DevDaxMemoryAllocator, self._allocator).cxl_arena

    def get_cxl_offset(self, obj: MemoryObj) -> int | None:
        """Locate an owned object in the shared slab.

        Args:
            obj: Read-reserved object to locate.

        Returns:
            Payload-relative byte offset, or None for a different backing arena.
        """
        if self.cxl_arena is None:
            return None
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        offset = obj.data_ptr - allocator.devdax_buffer.data_ptr()
        if offset < 0 or offset + obj.get_size() > self.cxl_arena.size:
            return None
        return offset

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Return a descriptor for the primary arena.

        Returns:
            The descriptor of the primary (configured) arena; arenas added at
            runtime are not part of it.
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        buffer = allocator.buffer
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
        )

    def owns_range(
        self, device_path: str, offset_in_bytes: int, size_in_bytes: int
    ) -> bool:
        """Return whether a mapped arena overlaps a byte range of a device.

        Args:
            device_path: Candidate device path or alias.
            offset_in_bytes: Start of the candidate range on that device.
            size_in_bytes: Length of the candidate range.

        Returns:
            ``True`` while an overlapping arena remains mapped.
        """
        return cast(DevDaxMemoryAllocator, self._allocator).owns_range(
            device_path, offset_in_bytes, size_in_bytes
        )

    def add_device(
        self, device_path: str, size_in_bytes: int, offset_in_bytes: int = 0
    ) -> DevDaxArenaStatus:
        """Map an additional Device-DAX range into the L1 arena pool at runtime.

        The new range becomes extra capacity available to subsequent
        allocations. Existing allocations are untouched.

        Args:
            device_path: Path to a readable and writable Device-DAX device.
            size_in_bytes: Number of bytes to map from the device.
            offset_in_bytes: Byte offset into the device.

        Returns:
            The status of the newly added arena.

        Raises:
            L1ReconfigureError: 409 when the request conflicts with the device
                or pool state (range overlap, capacity exceeded, allocator
                closed, device not openable, invalid offset).
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        try:
            return allocator.add_device(device_path, size_in_bytes, offset_in_bytes)
        except (ValueError, RuntimeError, OSError) as exc:
            raise L1ReconfigureError(409, str(exc)) from exc

    def remove_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
        offset_in_bytes: int = 0,
    ) -> DevDaxArenaStatus:
        """Retire a Device-DAX arena from the L1 arena pool at runtime.

        The arena stops accepting new allocations and is unmapped once its
        already-issued allocations are all freed. The primary arena (the one
        backing :meth:`get_l1_memory_desc`) may not be removed.

        Args:
            device_path: Path of the arena to remove.
            mode: Removal strategy. Only :attr:`DevDaxRemoveMode.DRAIN` is
                supported.
            offset_in_bytes: Device offset the arena was added with.

        Returns:
            The arena status after the request: ``REMOVED`` if it was unmapped
            immediately, otherwise ``DRAINING`` with the live allocation count.

        Raises:
            L1ReconfigureError: 404 when no arena matches; 409 when ``mode`` is
                unsupported or the arena is primary and non-removable.
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        try:
            return allocator.remove_device(device_path, mode, offset_in_bytes)
        except DevDaxNotMappedError as exc:
            raise L1ReconfigureError(404, str(exc)) from exc
        except ValueError as exc:
            raise L1ReconfigureError(409, str(exc)) from exc

    def get_arena_status(
        self, device_path: str, offset_in_bytes: int = 0
    ) -> DevDaxArenaStatus:
        """Return the status of the Device-DAX arena at ``(device_path, offset)``.

        Args:
            device_path: The exact path used when the arena was added.
            offset_in_bytes: Device offset the arena was added with.

        Returns:
            The arena's current status.

        Raises:
            L1ReconfigureError: 404 when no arena matches.
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        try:
            return allocator.arena_status(device_path, offset_in_bytes)
        except DevDaxNotMappedError as exc:
            raise L1ReconfigureError(404, str(exc)) from exc

    def memory_region_count(self) -> int:
        """Return the number of memory regions backing L1.

        Returns:
            One per mapped Device-DAX arena, active or draining.
        """
        return cast(DevDaxMemoryAllocator, self._allocator).memory_region_count()

    def get_arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return a status snapshot of every Device-DAX arena in the L1 pool.

        Returns:
            One :class:`DevDaxArenaStatus` per arena currently mapped, in pool
            order (the primary arena first).
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        return allocator.arena_statuses()
