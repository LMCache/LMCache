# SPDX-License-Identifier: Apache-2.0
"""Device-DAX L1 manager with runtime arena hot-plug."""

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.internal_api import CxlArenaDescriptor, L1MemoryDesc
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.memory_manager import DevDaxL1MemoryManager
from lmcache.v1.distributed.transfer_channel.api import MemoryRegionAddress
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxArenaStatus,
    DevDaxRemoveMode,
)
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class DevDaxL1Manager(BaseL1Manager):
    """L1 manager backed by Device-DAX arenas.

    The configured device range is the primary arena. Arenas can be added
    and drained at runtime, which is why this medium owns the hot-plug
    surface instead of the generic
    :class:`~lmcache.v1.distributed.internal_api.L1ManagerInterface`.
    """

    def __init__(self, config: L1ManagerConfig) -> None:
        """Map the arena pool described by ``config.memory_config``.

        Args:
            config: L1 settings with ``memory_config.devdax_path`` set.

        Raises:
            ValueError: If ``devdax_path`` is not configured.
        """
        memory_config = config.memory_config
        memory_manager = DevDaxL1MemoryManager(memory_config)
        super().__init__(config, memory_manager)
        # Typed reference for the arena calls the base class does not know.
        self._devdax_memory_manager = memory_manager
        if memory_config.use_hugepages:
            logger.warning(
                "l1-use-hugepages is a no-op for a Device-DAX L1 arena (no DRAM pool)."
            )
        logger.info("DevDaxL1Manager: Device-DAX L1 tier enabled; CPU-only L1 disabled")

    @property
    def backend_type(self) -> L1BackendType:
        """Return ``L1BackendType.DEVDAX``."""
        return L1BackendType.DEVDAX

    def get_memory_tags(self) -> MemoryTags:
        """Return ``(CXL, LOCAL)``."""
        return MemoryTags(segment=MemorySegment.CXL, residency=MemoryResidency.LOCAL)

    @property
    def cxl_arena(self) -> CxlArenaDescriptor | None:
        """Return the owned shared-pool slab, or None without a CXL pool id."""
        return self._devdax_memory_manager.cxl_arena

    def get_cxl_address(self, obj: MemoryObj) -> MemoryRegionAddress | None:
        """Return the location of a read-reserved owned CXL object.

        Args:
            obj: Object held under an owner read reservation.

        Returns:
            A CXL address, or None for objects outside the owned slab.
        """
        arena = self.cxl_arena
        if arena is None:
            return None
        offset = self._devdax_memory_manager.get_cxl_offset(obj)
        if offset is None:
            return None
        return MemoryRegionAddress(
            offset, obj.get_size(), arena, self.config.read_ttl_seconds
        )

    def get_capacity_bytes(self) -> int:
        """Return the bytes of every ACTIVE arena.

        Draining arenas stop accepting allocations and are excluded
        immediately; arenas added at runtime count as soon as they are active.

        Returns:
            The summed size in bytes of the active arenas.

        Note:
            Arena state is snapshotted under the allocator's pool lock. A
            separate usage query may observe an adjacent topology if
            reconfiguration is concurrent.
        """
        return sum(
            status.size_in_bytes
            for status in self._devdax_memory_manager.get_arena_statuses()
            if status.state is DevDaxArenaState.ACTIVE
        )

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Return the descriptor of the primary arena only.

        When extra arenas were added at runtime they are not part of the
        descriptor, so an L2 adapter or transfer channel that registers it
        covers only the primary arena; a warning says so.

        Returns:
            The primary region's descriptor.
        """
        region_count = self._devdax_memory_manager.memory_region_count()
        if region_count > 1:
            logger.warning(
                "Device-DAX L1 %s spans %d memory regions; only the primary "
                "region is exposed for registration, transfers cover only it",
                self.config.tag,
                region_count,
            )
        return self._devdax_memory_manager.get_l1_memory_desc()

    def get_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return runtime status for every Device-DAX arena.

        Returns:
            One status per mapped arena, in pool order.
        """
        return self._devdax_memory_manager.get_arena_statuses()

    def get_devdax_arena_status(
        self, device_path: str, offset_in_bytes: int = 0
    ) -> DevDaxArenaStatus:
        """Return the status of the Device-DAX arena at ``(device_path, offset)``.

        Args:
            device_path: The exact path used when the arena was added.
            offset_in_bytes: Device offset the arena was added with.

        Returns:
            The arena's current status.

        Raises:
            L1ReconfigureError: If no arena matches (404).
        """
        return self._devdax_memory_manager.get_arena_status(
            device_path, offset_in_bytes
        )

    def memory_region_count(self) -> int:
        """Return the number of mapped arenas, active or draining.

        An L2 adapter or transfer channel that registers a single L1 region
        covers only the primary arena when this is more than one.

        Returns:
            The arena count, including draining arenas.
        """
        return self._devdax_memory_manager.memory_region_count()

    def owns_range(
        self, device_path: str, offset_in_bytes: int, size_in_bytes: int
    ) -> bool:
        """Return whether this L1 maps a range overlapping one on a device.

        Args:
            device_path: Candidate device path or alias.
            offset_in_bytes: Start of the candidate range on that device.
            size_in_bytes: Length of the candidate range.

        Returns:
            ``True`` while an overlapping arena is mapped.
        """
        return self._devdax_memory_manager.owns_range(
            device_path, offset_in_bytes, size_in_bytes
        )

    def add_devdax_device(
        self,
        device_path: str,
        size_in_bytes: int,
        offset_in_bytes: int = 0,
    ) -> DevDaxArenaStatus:
        """Add a Device-DAX range to the L1 arena pool.

        Args:
            device_path: Path of the Device-DAX device to map.
            size_in_bytes: Number of bytes to map.
            offset_in_bytes: Byte offset into the device.

        Returns:
            Status of the newly added arena.

        Raises:
            L1ReconfigureError: If the request cannot be applied (409).
        """
        return self._devdax_memory_manager.add_device(
            device_path, size_in_bytes, offset_in_bytes
        )

    def remove_devdax_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
        offset_in_bytes: int = 0,
    ) -> DevDaxArenaStatus:
        """Remove a Device-DAX arena from the L1 arena pool.

        Args:
            device_path: Path of the mapped Device-DAX device.
            mode: Removal strategy. Only drain mode is currently supported.
            offset_in_bytes: Device offset the arena was added with.

        Returns:
            Status of the arena after the removal request.

        Raises:
            L1ReconfigureError: If no arena matches (404) or the request
                cannot be applied (409).
        """
        return self._devdax_memory_manager.remove_device(
            device_path, mode, offset_in_bytes
        )
