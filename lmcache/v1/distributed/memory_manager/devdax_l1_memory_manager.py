# SPDX-License-Identifier: Apache-2.0
"""Device-DAX L1 memory manager."""

# Standard
from dataclasses import dataclass
from typing import Any, cast
import mmap
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import get_device_identity, get_size_bytes
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaStatus,
    DevDaxMemoryAllocator,
    DevDaxNotMappedError,
    DevDaxRemoveMode,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.memory_orchestrator.api import RegionContract
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)


@dataclass
class _SharedRegionMapping:
    """The mapping of a region whose extents a memory orchestrator grants."""

    path: str
    offset: int
    capacity: int
    alignment: int
    fd: int
    mmap_obj: mmap.mmap
    mmap_buffer: Any
    buffer: torch.Tensor
    registered: bool = False

    def close(self) -> None:
        """Unregister and unmap the region once; live views keep it mapped."""
        if self.fd < 0:
            return
        if self.registered:
            current_device_spec.unpin_memory(self.buffer.data_ptr())
            self.registered = False
        self.buffer = torch.empty(0, dtype=torch.uint8)
        self.mmap_buffer = None
        try:
            self.mmap_obj.close()
        except BufferError:
            logger.warning(
                "Shared Device-DAX region %s still has live views; leaving it mapped",
                self.path,
            )
        os.close(self.fd)
        self.fd = -1


class DevDaxL1MemoryManager(L1MemoryManager):
    """L1 memory manager for Device-DAX-backed L1 memory.

    This is a peer of
    :class:`~lmcache.v1.distributed.memory_manager.l1_memory_manager.L1MemoryManager`
    and owns the Device-DAX allocator path. The configured device range is
    the primary arena; more arenas can be added at runtime.

    A shared region (``shared_region`` set) has no arenas and no allocator:
    the manager maps and GPU-registers the memory orchestrator's region and
    builds bounds-checked views of the extents it grants.
    """

    def __init__(
        self,
        config: L1MemoryManagerConfig,
        shared_region: RegionContract | None = None,
    ) -> None:
        """Create a Device-DAX L1 memory manager.

        Args:
            config: L1 memory configuration with ``devdax_path`` set.
            shared_region: Contract of the memory orchestrator's region when
                the range is shared. The manager then maps the region's
                capacity at ``devdax_offset_in_bytes`` and GPU-registers it.

        Raises:
            ValueError: If ``devdax_path`` is not configured.
            RuntimeError: If a shared region leaves the device, breaks its
                alignment, or cannot be GPU-registered.
            OSError: If a shared region cannot be opened or mapped.
        """
        if not config.devdax_path:
            raise ValueError("DevDaxL1MemoryManager requires devdax_path")

        self._shared: _SharedRegionMapping | None = None
        if shared_region is not None:
            self._shared = self._map_shared_region(
                config.devdax_path, config.devdax_offset_in_bytes, shared_region
            )
            return

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
            offset=config.devdax_offset_in_bytes,
        )
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Return a descriptor for the primary arena.

        Returns:
            The descriptor of the primary (configured) arena; arenas added at
            runtime are not part of it.

        Raises:
            L1ReconfigureError: 409 for a shared region, which has no arena.
        """
        buffer = self._arenas().buffer
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
            ``True`` while an overlapping arena, or the overlapping shared
            region, remains mapped.
        """
        shared = self._shared
        if shared is not None:
            identity = get_device_identity(device_path)
            return (
                identity is not None
                and identity == get_device_identity(shared.fd)
                and offset_in_bytes < shared.offset + shared.capacity
                and shared.offset < offset_in_bytes + size_in_bytes
            )
        return self._arenas().owns_range(device_path, offset_in_bytes, size_in_bytes)

    def view(
        self, offset: int, length: int, layout: MemoryLayoutDesc
    ) -> TensorMemoryObj:
        """Return a parentless view of an extent the orchestrator granted.

        Dropping the view never frees or unmaps the orchestrator's bytes.

        Args:
            offset: Byte offset of the extent in the region.
            length: Aligned extent length in bytes.
            layout: Shapes and dtypes of the object's tensors.

        Returns:
            A memory object over ``[offset, offset + length)`` of the region.

        Raises:
            ValueError: The extent is negative, misaligned, outside the region,
                or smaller than the layout's payload.
        """
        shared = self._shared
        assert shared is not None, "view() requires a shared Device-DAX region"
        payload = get_size_bytes(layout.shapes, layout.dtypes)
        if (
            offset < 0
            or offset % shared.alignment
            or length < payload
            or offset + length > shared.capacity
        ):
            raise ValueError(
                f"extent [{offset}, {offset + length}) for a {payload}-byte "
                f"object does not fit the {shared.capacity}-byte region"
            )
        return TensorMemoryObj(
            raw_data=shared.buffer[offset : offset + length],
            metadata=MemoryObjMetadata(
                layout.shapes[0],
                layout.dtypes[0],
                offset,
                length,
                1,
                0,
                MemoryFormat.KV_2LTD,
                shapes=list(layout.shapes),
                dtypes=list(layout.dtypes),
            ),
            parent_allocator=None,
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
                closed, device not openable, invalid offset), or the region
                is shared.
        """
        allocator = self._arenas()
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
                unsupported, the arena is primary and non-removable, or the
                region is shared.
        """
        allocator = self._arenas()
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
            L1ReconfigureError: 404 when no arena matches; 409 for a shared
                region.
        """
        allocator = self._arenas()
        try:
            return allocator.arena_status(device_path, offset_in_bytes)
        except DevDaxNotMappedError as exc:
            raise L1ReconfigureError(404, str(exc)) from exc

    def memory_region_count(self) -> int:
        """Return the number of memory regions backing L1.

        Returns:
            One per mapped Device-DAX arena, active or draining; one for a
            shared region.
        """
        if self._shared is not None:
            return 1
        return self._arenas().memory_region_count()

    def get_arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return a status snapshot of every Device-DAX arena in the L1 pool.

        Returns:
            One :class:`DevDaxArenaStatus` per arena currently mapped, in pool
            order (the primary arena first).

        Raises:
            L1ReconfigureError: 409 for a shared region, which has no arenas.
        """
        return self._arenas().arena_statuses()

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Free objects; a shared region's views are never freed (see ``view``).

        Args:
            mem_objs: Objects to free; empty for a shared region.

        Returns:
            ``L1Error.SUCCESS``.
        """
        if self._shared is None:
            return super().free(mem_objs)
        assert not mem_objs, "A shared Device-DAX region frees nothing locally"
        return L1Error.SUCCESS

    def close(self) -> None:
        """Release all resources; close a shared region after its copies drain."""
        if self._shared is None:
            super().close()
            return
        self._shared.close()

    def _arenas(self) -> DevDaxMemoryAllocator:
        """Return the arena pool; a shared region has none."""
        if self._shared is not None:
            raise L1ReconfigureError(
                409,
                "A shared Device-DAX region has no arenas: its memory "
                "orchestrator owns the mapping, so hot-plug is unsupported",
            )
        return cast(DevDaxMemoryAllocator, self._allocator)

    def _map_shared_region(
        self, path: str, offset: int, contract: RegionContract
    ) -> _SharedRegionMapping:
        """Map the region's capacity at ``offset`` of ``path``; GPU-register it."""
        mapping = _SharedRegionMapping(
            path,
            offset,
            contract.capacity_bytes,
            contract.alignment_bytes,
            *DevDaxMemoryAllocator.open_devdax_mapping(
                path, contract.capacity_bytes, offset
            ),
        )
        try:
            # Shared direct transfers require registered memory; unlike the
            # private path there is no pageable fallback.
            mapping.registered = current_device_spec.pin_memory(
                mapping.buffer.data_ptr(), mapping.capacity
            )
            if not mapping.registered:
                raise RuntimeError(
                    f"A shared Device-DAX region must be GPU-registered: {path}"
                )
        except BaseException:
            mapping.close()
            raise
        return mapping
