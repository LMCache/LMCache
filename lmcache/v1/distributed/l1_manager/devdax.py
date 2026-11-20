# SPDX-License-Identifier: Apache-2.0
"""Device-DAX L1 manager: runtime arena hot-plug, or one shared region."""

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    L1ManagerConfig,
    validate_shared_devdax_l1,
)
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L1MemoryDesc,
    L1OperationResult,
)
from lmcache.v1.distributed.l1_manager.base import (
    BaseL1Manager,
    L1ObjectState,
    validate_read_locks,
)
from lmcache.v1.distributed.memory_manager import DevDaxL1MemoryManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxArenaStatus,
    DevDaxRemoveMode,
)
from lmcache.v1.mp_observability.event import Event, EventType

logger = init_logger(__name__)


class DevDaxL1Manager(BaseL1Manager):
    """L1 manager backed by Device-DAX arenas.

    The configured device range is the primary arena. Arenas can be added
    and drained at runtime, which is why this medium owns the hot-plug
    surface instead of the generic
    :class:`~lmcache.v1.distributed.internal_api.L1ManagerInterface`.

    With a ``shared`` object in its config the range is one region that
    several MP servers map: a memory orchestrator owns its allocation and
    object state, :class:`OrchestratedObjectState` holds this server's write
    grants and read leases, and the lifecycle methods below route to it
    instead of the base state machine, outside ``_lock``, because they wait
    on the orchestrator. Such a region has no arenas (hot-plug returns 409),
    is never evicted, offers no memory descriptor to L2 adapters, and reports
    ``(CXL, REMOTE)`` tags.
    """

    def __init__(self, config: L1ManagerConfig) -> None:
        """Open private arenas or join the shared region for this configuration.

        Args:
            config: L1 settings with ``memory_config.devdax_path`` set.

        Raises:
            ValueError: If the device or shared region configuration is invalid.
        """
        memory_config = config.memory_config
        if isinstance(config, DevDaxL1ManagerConfig) and config.shared is not None:
            # Private L1s do not require the gRPC client or generated bindings.
            # First Party
            from lmcache.v1.distributed.shared_l1.object_state import (
                OrchestratedObjectState,
            )

            validate_shared_devdax_l1(config)
            self._shared: OrchestratedObjectState | None = OrchestratedObjectState(
                config.shared, memory_config
            )
            try:
                self._init_reporting(config)
            except BaseException:
                self._shared.close()
                raise
            return
        self._shared = None
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
    def _arenas(self) -> DevDaxL1MemoryManager:
        if self._shared is not None:
            raise L1ReconfigureError(409, "A shared Device-DAX region has no arenas")
        return self._devdax_memory_manager

    @property
    def backend_type(self) -> L1BackendType:
        """Return ``L1BackendType.DEVDAX``."""
        return L1BackendType.DEVDAX

    def get_memory_tags(self) -> MemoryTags:
        """A shared region has REMOTE residency."""
        if self._shared is not None:
            return MemoryTags(MemorySegment.CXL, MemoryResidency.REMOTE)
        return MemoryTags(segment=MemorySegment.CXL, residency=MemoryResidency.LOCAL)

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
        if self._shared is not None:
            return self._shared.capacity_bytes
        return sum(
            status.size_in_bytes
            for status in self._devdax_memory_manager.get_arena_statuses()
            if status.state is DevDaxArenaState.ACTIVE
        )

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return the descriptor of the primary arena only.

        When extra arenas were added at runtime they are not part of the
        descriptor, so an L2 adapter or transfer channel that registers it
        covers only the primary arena; a warning says so.

        Returns:
            The primary region's descriptor.
        """
        if self._shared is not None:
            return None
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
        return self._arenas.get_arena_statuses()

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
        return self._arenas.get_arena_status(device_path, offset_in_bytes)

    def memory_region_count(self) -> int:
        """Return the number of mapped arenas, active or draining.

        An L2 adapter or transfer channel that registers a single L1 region
        covers only the primary arena when this is more than one.

        Returns:
            The arena count, including draining arenas.
        """
        if self._shared is not None:
            return 1
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
        if self._shared is not None:
            return self._shared.region.owns_range(
                device_path, offset_in_bytes, size_in_bytes
            )
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
        return self._arenas.add_device(device_path, size_in_bytes, offset_in_bytes)

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
        return self._arenas.remove_device(device_path, mode, offset_in_bytes)

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve reads; a shared region leases through the orchestrator."""
        if self._shared is None:
            return super().reserve_read(keys, read_locks)
        ret = self._shared.reserve_read(keys, validate_read_locks(read_locks))
        leased = self._own(ret)
        for listener in self._listeners():
            listener.on_l1_keys_reserved_read(leased)
        self._publish_shared(EventType.L1_READ_RESERVED, {"keys": leased})
        return ret

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Read without locking; a shared region returns only leased views."""
        if self._shared is None:
            return super().unsafe_read(keys)
        # The retrieve path reads prefetched, so leased, keys this way.
        return self._shared.unsafe_read(keys)

    def finish_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1Error]:
        """Release reads; a shared region releases its leases."""
        if self._shared is None:
            return super().finish_read(keys, read_locks)
        ret = self._shared.finish_read(keys, validate_read_locks(read_locks))
        done = [key for key, err in ret.items() if err == L1Error.SUCCESS]
        for listener in self._listeners():
            listener.on_l1_keys_read_finished(done)
        self._publish_shared(EventType.L1_READ_FINISHED, {"keys": done})
        return ret

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
        *,
        allow_resident: bool = False,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve writes; a shared region takes extents from the orchestrator."""
        if self._shared is None:
            return super().reserve_write(
                keys, is_temporary, layout_desc, tag, allow_resident=allow_resident
            )
        ret = self._shared.reserve_write(keys, is_temporary, layout_desc, tag)
        granted = self._own(ret)
        for listener in self._listeners():
            listener.on_l1_keys_reserved_write(granted)
        self._publish_shared(EventType.L1_WRITE_RESERVED, {"keys": granted, "tag": tag})
        return ret

    def finish_write(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Admit writes; a shared region publishes and commits its extents."""
        if self._shared is None:
            return super().finish_write(keys, tag)
        ret = self._shared.finish_writes(keys, tag, publish=True)
        done = [key for key, err in ret.items() if err == L1Error.SUCCESS]
        self._report_shared_write_finished(done)
        return ret

    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1, tag: str = ""
    ) -> dict[ObjectKey, L1OperationResult]:
        """Admit writes and reserve reads; a shared region commits, then leases."""
        if self._shared is None:
            return super().finish_write_and_reserve_read(keys, read_locks, tag)
        count = validate_read_locks(read_locks)
        errors = self._shared.finish_writes(keys, tag, publish=True)
        done = [key for key, err in errors.items() if err == L1Error.SUCCESS]
        ret: dict[ObjectKey, L1OperationResult] = {
            key: (err, None) for key, err in errors.items() if err != L1Error.SUCCESS
        }
        ret.update(self._shared.reserve_read(done, count))
        # Report only what succeeded: a commit whose lease failed is written,
        # not read-reserved.
        leased = self._own(ret)
        unleased = [key for key in done if ret[key][0] != L1Error.SUCCESS]
        for listener in self._listeners():
            listener.on_l1_keys_finish_write_and_reserve_read(leased)
        self._publish_shared(
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, {"keys": leased}
        )
        self._report_shared_write_finished(unleased)
        return ret

    def delete(
        self, keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        """Delete objects; a shared region hides them from every server.

        The bytes of a shared object stay consumed until a region reset, even
        if ``force`` is set; ``KEY_IS_LOCKED`` means the object is still
        being written or the orchestrator cannot be used.
        """
        if self._shared is None:
            return super().delete(keys, force)
        ret = self._shared.delete(keys)
        deleted = [key for key, err in ret.items() if err == L1Error.SUCCESS]
        if deleted:
            for listener in self._listeners():
                listener.on_l1_keys_deleted_by_manager(deleted)
            self._publish_shared(EventType.L1_KEYS_EVICTED, {"keys": deleted})
        return ret

    def finish_write_and_delete(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Drop staged writes; a shared region aborts its grants."""
        if self._shared is None:
            return super().finish_write_and_delete(keys, tag)
        ret = self._shared.finish_writes(keys, tag, publish=False)
        self._report_staging_gone(
            [key for key, err in ret.items() if err != L1Error.KEY_NOT_EXIST]
        )
        return ret

    def clear(self, force: bool = False) -> None:
        """Clear objects; a shared region deletes only what this server wrote."""
        if self._shared is None:
            return super().clear(force)
        deleted = self._shared.delete_published()
        if deleted is None:
            logger.warning(
                "DevDaxL1Manager: clear() could not reach the memory orchestrator; "
                "the objects of shared Device-DAX L1 %r stay readable",
                self.config.tag,
            )
            return
        logger.info(
            "DevDaxL1Manager: cleared %d objects this server wrote to shared "
            "Device-DAX L1 %r",
            deleted,
            self.config.tag,
        )

    def get_memory_usage(self) -> tuple[int, int]:
        """Return (used, total); a shared region reports (staging, capacity)."""
        if self._shared is None:
            return super().get_memory_usage()
        return self._shared.get_staging_memory_usage(), self._shared.capacity_bytes

    def get_staging_memory_usage(self) -> int:
        """Return the bytes of uncommitted writes, local or granted."""
        if self._shared is None:
            return super().get_staging_memory_usage()
        return self._shared.get_staging_memory_usage()

    def report_status(self) -> dict:
        """Return the status fields; a shared region adds ``shared.state``."""
        if self._shared is None:
            return super().report_status()
        return self._shared.report_status()

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Only a private arena can evict a local object."""
        return self._shared is None and super().is_key_evictable(key)

    def get_object_state(self, key: ObjectKey) -> L1ObjectState | None:
        """Return the local object state; a shared region keeps none."""
        if self._shared is None:
            return super().get_object_state(key)
        return None

    def memcheck(self) -> bool:
        """Check the memory manager; a shared region checks the orchestrator."""
        if self._shared is None:
            return super().memcheck()
        return self._shared.is_healthy()

    def close(self) -> None:
        """Close the manager; retire a shared region's grants and leases first.

        Close a shared region only after copies through it have drained.
        """
        if self._shared is None:
            return super().close()
        self._shared.close()
        self._unregister_gauges()

    def _own(self, ret: dict[ObjectKey, L1OperationResult]) -> list[ObjectKey]:
        """Stamp this manager on the views in ``ret``; return their keys."""
        keys: list[ObjectKey] = []
        for key, (err, obj) in ret.items():
            if err == L1Error.SUCCESS and obj is not None:
                obj.set_l1_manager(self.l1_manager_id)
                keys.append(key)
        return keys

    def _listeners(self) -> list[L1ManagerListener]:
        with self._lock:
            return list(self._registered_listeners)

    def _report_shared_write_finished(self, keys: list[ObjectKey]) -> None:
        """Notify listeners and the event bus that shared ``keys`` committed."""
        if not keys:
            return
        for listener in self._listeners():
            listener.on_l1_keys_write_finished(keys)
        self._publish_shared(EventType.L1_WRITE_FINISHED, {"keys": keys})

    def _publish_shared(
        self, event_type: EventType, metadata: dict[str, object]
    ) -> None:
        # ``shared`` keeps region-owned placements out of per-server cache
        # events, which would otherwise report one object once per server.
        self._event_bus.publish(
            Event(
                event_type=event_type,
                metadata={"l1_tag": self.config.tag, "shared": True, **metadata},
            )
        )
