# SPDX-License-Identifier: Apache-2.0
"""Device-DAX L1 memory manager."""

# Standard
from typing import cast

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import L1BackendType
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.distributed.memory_manager.reconfiguration import (
    L1ReconfigureError,
)
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

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Return the storage medium backing ``memory_obj``.

        Args:
            memory_obj: An object allocated by this manager.

        Returns:
            ``L1BackendType.DEVDAX`` for objects in the Device-DAX arena,
            ``L1BackendType.DRAM`` for objects in the local DRAM pool of a
            hybrid configuration.
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        if allocator.is_devdax_obj(memory_obj):
            return L1BackendType.DEVDAX
        return L1BackendType.DRAM

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

    def add_device(self, device_path: str, size_in_bytes: int) -> DevDaxArenaStatus:
        """Map an additional Device-DAX device into the L1 arena pool at runtime.

        The new device becomes overflow capacity available to subsequent
        allocations. Existing allocations are untouched.

        Args:
            device_path: Path to a readable and writable Device-DAX device that
                is not already mapped.
            size_in_bytes: Number of bytes to map from the device.

        Returns:
            The status of the newly added arena.

        Raises:
            L1ReconfigureError: 409 when the request conflicts with the device
                or pool state (already mapped, alignment violation, capacity
                exceeded, allocator closed, device not openable).
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        try:
            return allocator.add_device(device_path, size_in_bytes)
        except (ValueError, RuntimeError, OSError) as exc:
            raise L1ReconfigureError(409, str(exc)) from exc

    def remove_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
    ) -> DevDaxArenaStatus:
        """Retire a Device-DAX device from the L1 arena pool at runtime.

        The device stops accepting new allocations and is unmapped once its
        already-issued allocations are all freed. The primary arena (the one
        backing :meth:`get_l1_memory_desc` in a pure Device-DAX configuration)
        may not be removed.

        Args:
            device_path: Path of the device to remove.
            mode: Removal strategy. Only :attr:`DevDaxRemoveMode.DRAIN` is
                supported.

        Returns:
            The arena status after the request: ``REMOVED`` if it was unmapped
            immediately, otherwise ``DRAINING`` with the live allocation count.

        Raises:
            L1ReconfigureError: 404 when no arena is mapped at ``device_path``;
                409 when ``mode`` is unsupported or the arena is primary and
                non-removable.
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        try:
            return allocator.remove_device(device_path, mode)
        except DevDaxNotMappedError as exc:
            raise L1ReconfigureError(404, str(exc)) from exc
        except ValueError as exc:
            raise L1ReconfigureError(409, str(exc)) from exc

    def get_arena_status(self, device_path: str) -> DevDaxArenaStatus:
        """Return the status of the Device-DAX arena mapped at ``device_path``.

        Args:
            device_path: Path (or any alias) of the mapped device.

        Returns:
            The arena's current status.

        Raises:
            L1ReconfigureError: 404 when no arena is mapped at ``device_path``.
        """
        allocator = cast(DevDaxMemoryAllocator, self._allocator)
        try:
            return allocator.arena_status(device_path)
        except DevDaxNotMappedError as exc:
            raise L1ReconfigureError(404, str(exc)) from exc

    def memory_region_count(self) -> int:
        """Return the number of memory regions backing L1.

        Returns:
            One for the DRAM tier of a hybrid configuration, if present, plus
            one per mapped Device-DAX arena, active or draining.
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
