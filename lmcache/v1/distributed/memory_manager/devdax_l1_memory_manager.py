# SPDX-License-Identifier: Apache-2.0
"""Device-DAX L1 memory manager."""

# Standard
from typing import cast

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaStatus,
    DevDaxMemoryAllocator,
    DevDaxRemoveMode,
)

logger = init_logger(__name__)


class DevDaxL1MemoryManager(L1MemoryManager):
    """L1 memory manager backed by Device-DAX. Pure Device-DAX maps the device
    as the full L1 arena; hybrid uses DRAM first and spills into Device-DAX.
    """

    def __init__(self, config: L1MemoryManagerConfig) -> None:
        if not config.devdax_path:
            raise ValueError("DevDaxL1MemoryManager requires devdax_path")

        devdax_size = config.devdax_size_in_bytes or config.size_in_bytes
        local_size = config.size_in_bytes if config.devdax_size_in_bytes else 0
        logger.debug(
            "use devdax memory allocator, dram size is %d bytes, "
            "devdax path is %s, devdax size is %d bytes, align bytes is %d bytes",
            local_size,
            config.devdax_path,
            devdax_size,
            config.align_bytes,
        )
        self._allocator = DevDaxMemoryAllocator(
            devdax_size,
            config.devdax_path,
            local_size=local_size,
            shm_name=config.shm_name or None,
            align_bytes=config.align_bytes,
        )
        self._config = config
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Return a descriptor for the primary L1 buffer."""
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        buffer = allocator.buffer
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
        )

    def add_device(self, device_path: str, size_in_bytes: int) -> DevDaxArenaStatus:
        """Map an additional Device-DAX device into the pool at runtime; it
        serves overflow immediately."""
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        return allocator.add_device(device_path, size_in_bytes)

    def remove_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
    ) -> DevDaxArenaStatus:
        """Drain-remove a device at runtime: no new allocations, unmapped once
        idle. Returns ``REMOVED`` or ``DRAINING``; the primary arena is
        rejected."""
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        return allocator.remove_device(device_path, mode)

    def get_arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return a status snapshot of every arena, in pool order."""
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        return allocator.arena_statuses()
