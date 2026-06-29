# SPDX-License-Identifier: Apache-2.0
"""Device-DAX L1 memory manager."""

# Standard
from typing import cast

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_management import DevDaxMemoryAllocator

logger = init_logger(__name__)


class DevDaxL1MemoryManager(L1MemoryManager):
    """L1 memory manager for Device-DAX-backed L1 memory.

    This is a peer of
    :class:`~lmcache.v1.distributed.memory_manager.l1_memory_manager.L1MemoryManager`
    and owns the Device-DAX allocator path that used to live inside the generic
    CPU L1 manager. A pure Device-DAX configuration maps the DAX device as the
    full L1 arena. A hybrid configuration uses DRAM first and spills overflow
    allocations into Device-DAX.
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
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Return a descriptor for the primary L1 buffer.

        Existing callers expect Device-DAX L1 to expose the mapped buffer here.
        Hybrid DRAM + Device-DAX L1 is still rejected by transfer paths that
        require one registerable L1 region via ``l1_exposes_single_memory_region``.

        Returns:
            The descriptor for the primary L1 buffer.
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        buffer = allocator.buffer
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
        )
