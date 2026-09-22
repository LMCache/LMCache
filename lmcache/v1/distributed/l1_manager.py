# SPDX-License-Identifier: Apache-2.0
"""
Managing objects and memory for L1 cache
"""

# Standard
from dataclasses import dataclass
import threading

# First Party
from lmcache.lmcache_native import TTLLock
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, get_configured_capacity_bytes
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener, L1ObjectMeta
from lmcache.v1.distributed.memory_manager import (
    GDSL1MemoryManager,
    L1ManagerProtocol,
    L1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.distributed.shared_l1.backend import SharedDevDaxL1Backend
from lmcache.v1.memory_coordinator.api import OutOfSpaceError
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge

logger = init_logger(__name__)


# Internal classes and helper functions
@dataclass
class L1ObjectState:
    """
    The internal state of an object in L1 cache
    """

    memory_obj: MemoryObj
    """ The memory object stored in L1 cache. """

    write_lock: TTLLock
    """ The writer's reservation; held while the object is staged. """

    read_lock: TTLLock
    """ The read lock with TTL for the object. """

    is_temporary: bool
    """ Whether the object is temporary (need to be deleted after read). """


def l1_mgr_synchronized(func):
    """
    Decorator to mark L1Manager methods as thread-safe
    """

    def wrapper(self: "L1Manager", *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


L1OperationResult = tuple[L1Error, MemoryObj | None]

# Upper bound for the count parameter in reserve_read / finish_read
# to prevent a single call from holding the global lock for too long.
MAX_READ_LOCK_COUNT = 128


def _validate_read_locks(read_locks: int) -> int:
    """Validate and clamp a per-key read-lock count.

    Args:
        read_locks: Total read locks to take or release per key.

    Returns:
        Clamped value in [1, MAX_READ_LOCK_COUNT].
    """
    if read_locks < 1:
        logger.warning(
            "L1Manager: read_locks=%d is invalid, clamping to 1",
            read_locks,
        )
        return 1
    if read_locks > MAX_READ_LOCK_COUNT:
        logger.warning(
            "L1Manager: read_locks=%d exceeds limit=%d, clamping",
            read_locks,
            MAX_READ_LOCK_COUNT,
        )
        return MAX_READ_LOCK_COUNT
    return read_locks


def _l1_usage_ratio_or_zero(target: "L1Manager | None") -> float:
    """Return ``target.get_memory_usage()`` as a 0.0-1.0 ratio.

    Returns 0.0 when ``target`` is None or ``total_bytes`` is zero so the
    observable-gauge callback never raises during scrape.
    """
    if target is None:
        return 0.0
    used, total = target.get_memory_usage()
    if total <= 0:
        return 0.0
    return used / total


def _l1_staging_bytes_or_zero(target: "L1Manager | None") -> int:
    """Return ``target.get_staging_memory_usage()``, or 0 without a target."""
    if target is None:
        return 0
    return target.get_staging_memory_usage()


# Main classes


class L1Manager:
    """
    Object lifecycle state machine for L1 cache

    A write creates a *staging object* owned by the writer's ``tag``. Staging
    objects are kept apart from the resident objects: readers of the key do
    not see them. In private L1, writers with different tags may stage the same
    key at the same time. Shared L1 keeps one writer per key and never reclaims
    extents; the reclamation paths in the diagram below are private-L1-only.

          +--------+
          |  None  | <---------------------------------------+
          +--------+                                         |
            |   ^                                            |
            |   | (write lock expired: evictable)            | delete()
            |   |                                            |
    reserve |   |                                            |
    write() |   |                                            |
    (key,   v   |                                            |
     tag) +--------------+           +-----------+           |
          | write_locked |  finish_  |           |-----------+
          | staging      |---------->|   ready   |
          | (key, tag)   |  write()  |           |---------------+
          +--------------+ (admit or +-----------+               |
            |               discard)       |                     |
            |                              | reserve_read()      | finish_read()
            | finish_write_and_            |                     | (if count becomes 0)
            | delete() -> None             v                     |
            |                      +-----------------+           |
            |                      |   read_locked   |-----------+
            |                      |   (count = 1)   |
            |                      +-----------------+
            |                            |     ^
            |             reserve_read() |     | finish_read()
            |                            v     |
            |                      +-----------------+
            |                      |   read_locked   |
            |                      |   (count = 2)   |
            |                      +-----------------+
            |                            |     ^
            |             reserve_read() |     | finish_read()
            |                            v     |
            v                          (...)  (...)
          (freed)                  (Higher Counts)

    For every operation on list of keys, the operation is atomic
    """

    # Singleton dispatch for ``lmcache_mp.l1_memory_usage_bytes``: tests may
    # construct multiple L1Managers but the OTel SDK only honors the first
    # gauge registration, so the callback reads from the most recently built
    # instance via ``_gauge_target``.
    _gauge_registered: bool = False
    _gauge_target: "L1Manager | None" = None

    def __init__(self, config: L1ManagerConfig):
        self._lock = threading.Lock()

        # Resident objects: readable, never write-locked.
        self._objects: dict[ObjectKey, L1ObjectState] = {}
        # Staging objects: key -> writer tag -> write-locked object that is
        # invisible to readers until it is admitted by finish_write.
        self._staging: dict[ObjectKey, dict[str, L1ObjectState]] = {}
        # Bytes held by staging objects (kept in sync with ``_staging``).
        self._staging_bytes: int = 0

        # GDS, Device-DAX, shared Device-DAX, and CPU L1 are mutually
        # exclusive tiers. The shared tier is not a local allocator at all:
        # it goes through the narrow L1 coordination seam instead of
        # L1ManagerProtocol, because the Memory Coordinator owns offsets and
        # object lifetimes.
        self._memory_manager: L1ManagerProtocol | None
        self._shared_backend: SharedDevDaxL1Backend | None = None
        if config.shared_l1_config is not None:
            self._memory_manager = None
            shared = config.shared_l1_config
            memory_config = config.memory_config
            self._shared_backend = SharedDevDaxL1Backend(
                devdax_path=memory_config.devdax_path or "",
                capacity_bytes=memory_config.size_in_bytes,
                alignment_bytes=memory_config.align_bytes,
                region_id=shared.region_id,
                layout_id=shared.layout_id,
                mapping_offset_bytes=shared.mapping_offset_bytes,
                coordinator_endpoint=shared.coordinator_endpoint,
                coordinator_token_file=shared.coordinator_token_file,
                visibility_library_path=shared.visibility_library_path,
            )
            logger.info("L1Manager: coordinator-owned shared Device-DAX L1 enabled")
        elif config.gds_l1_config is not None:
            self._memory_manager = GDSL1MemoryManager(config.gds_l1_config)
            logger.info("L1Manager: GDS L1 tier enabled; CPU pinned-DRAM L1 disabled")
        elif config.memory_config.devdax_path:
            self._memory_manager = DevDaxL1MemoryManager(config.memory_config)
            logger.info("L1Manager: Device-DAX L1 tier enabled; CPU-only L1 disabled")
        else:
            self._memory_manager = L1MemoryManager(config.memory_config)

        # Precomputed: it derives from config alone and never changes, and
        # report_status runs under the global L1 lock on a hot polling path.
        self._configured_capacity_bytes = sum(
            get_configured_capacity_bytes(config).values()
        )
        self._write_ttl_seconds = config.write_ttl_seconds
        self._read_ttl_seconds = config.read_ttl_seconds

        self._registered_listeners: list[L1ManagerListener] = []

        self._event_bus = get_event_bus()

        L1Manager._gauge_target = self
        if not L1Manager._gauge_registered:
            L1Manager._gauge_registered = True
            register_gauge(
                "lmcache.l1_manager",
                "lmcache_mp.l1_memory_usage_bytes",
                "Bytes currently held in L1 cache",
                lambda: (
                    L1Manager._gauge_target.get_memory_usage()[0]
                    if L1Manager._gauge_target is not None
                    else 0
                ),
            )
            register_gauge(
                "lmcache.l1_manager",
                "lmcache_mp.l1_usage_ratio",
                "L1 used/total ratio (0.0–1.0)",
                lambda: _l1_usage_ratio_or_zero(L1Manager._gauge_target),
            )
            register_gauge(
                "lmcache.l1_manager",
                "lmcache_mp.l1_staging_bytes",
                "Bytes held by L1 staging objects (write-reserved, not admitted)",
                lambda: _l1_staging_bytes_or_zero(L1Manager._gauge_target),
            )

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for L1Manager events.

        Args:
            listener: The listener to register.
        """
        with self._lock:
            self._registered_listeners.append(listener)

    @property
    def uses_shared_l1(self) -> bool:
        """Return whether this manager uses coordinator-owned shared L1."""
        return self._shared_backend is not None

    @l1_mgr_synchronized
    def reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve read access for the given keys.

        Args:
            keys: The list of object keys to reserve
                read access for.
            read_locks: Total read locks acquired per key --
                one per worker that consumes a read lock
                for the same key (e.g. MLA models with
                TP > 1).

        Returns:
            A dictionary mapping each object key to a tuple
            of (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.

        Note:
            Staging objects are never readable; a key that is only
            being written is reported as ``KEY_NOT_EXIST``.
        """
        if self._shared_backend is not None:
            return self._shared_reserve_read(keys, read_locks)

        total = _validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            # TODO(perf): support a count argument in
            # TTLLock.lock() to avoid Python for-loop
            # overhead (TTLLock is C++ std::atomic).
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        self._report_read_reserved(successful_keys)
        return ret

    def _shared_reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Acquire a whole key list and attach shared tensor views."""
        backend = self._shared_backend
        assert backend is not None
        if read_locks != 1:
            raise ValueError("shared L1 currently supports TP=1 reads only")
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        try:
            memory_objects = backend.reserve_read(keys)
            for key, memory_obj in zip(keys, memory_objects, strict=True):
                if memory_obj is None:
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                    continue
                entry = self._objects.get(key)
                if entry is None:
                    entry = L1ObjectState(
                        memory_obj=memory_obj,
                        write_lock=TTLLock(self._write_ttl_seconds),
                        read_lock=TTLLock(self._read_ttl_seconds),
                        is_temporary=False,
                    )
                    self._objects[key] = entry
                else:
                    entry.memory_obj = memory_obj
                entry.read_lock.lock()
                successful_keys.append(key)
                ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            for listener in self._registered_listeners:
                listener.on_l1_keys_reserved_read(successful_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_READ_RESERVED,
                    metadata={"keys": successful_keys},
                )
            )
        except BaseException:
            for key in successful_keys:
                self._objects[key].read_lock.unlock()
            raise

        return ret

    @l1_mgr_synchronized
    def unsafe_read(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1OperationResult]:
        """Unsafe read the read-locked objects without adding new read locks.

        This method does not acquire read locks. Therefore, the caller need
        to make sure the `unsafe_read` is called between `reserve_read` and
        `finish_read` calls.

        Args:
            keys: The list of object keys to read.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_NOT_READABLE: The key is not readable (in this case, not read-locked).
        """
        ret: dict[ObjectKey, L1OperationResult] = {}

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.read_lock.is_locked():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                continue

            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        return ret

    @l1_mgr_synchronized
    def finish_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
    ) -> dict[ObjectKey, L1Error]:
        """Finish read access for the given keys.

        Will delete the object if it is temporary and read
        count reaches zero.

        Args:
            keys: The list of object keys to finish read
                access for.
            read_locks: Read locks to release per key.  A caller
                releasing only its own read lock passes 1 (the
                default); the reservation owner releasing the
                whole reservation passes the ``reserve_read``
                total.

        Returns:
            A dictionary mapping each object key to an
            L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is not read-locked, which
                means the reader may read inconsistent data.
        """
        shared = self._shared_backend is not None
        if shared and read_locks != 1:
            raise ValueError("shared L1 currently supports TP=1 reads only")

        total = _validate_read_locks(read_locks)
        need_to_free: list[MemoryObj] = []
        need_to_free_keys: list[ObjectKey] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                logger.warning(
                    "L1Manager: finish read on non-existing key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if not entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish read on non-read-locked key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if not shared:
                # TODO(perf): support a count argument in
                # TTLLock.unlock() to avoid Python for-loop
                # overhead (TTLLock is C++ std::atomic).
                for _ in range(total):
                    entry.read_lock.unlock()
                if entry.is_temporary and not entry.read_lock.is_locked():
                    # NOTE: temporary objects shouldn't have write-locks
                    need_to_free.append(entry.memory_obj)
                    need_to_free_keys.append(key)
                    del self._objects[key]

            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        freed_meta = [self._object_meta(obj) for obj in need_to_free]
        if shared:
            # Validate the whole shared batch before releasing local read locks.
            for key in successful_keys:
                self._objects[key].read_lock.unlock()
        else:
            assert self._memory_manager is not None
            self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(successful_keys)
            if not shared:
                listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"keys": successful_keys},
            )
        )
        if not shared:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": need_to_free_keys, "meta": freed_meta},
                )
            )

        return ret

    @l1_mgr_synchronized
    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve a staging object for each of the given keys.

        Args:
            keys: The list of object keys to reserve write access for.
            is_temporary: The list of booleans indicating whether each key is
                temporary.
            layout_desc: The memory layout description for the objects to be
                allocated.
            tag: The writer's identity; the same tag must be passed to the
                ``finish_write`` variant that completes the write.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Raises:
            ValueError: If ``keys`` and ``is_temporary`` differ in length.

        Errors:
            KEY_NOT_WRITABLE: The key already has a resident object, or
                ``tag`` already stages the key.
            OUT_OF_MEMORY: Not enough memory to allocate for the object.

        Note:
            A staging object is invisible to readers and to other tags until
            it is admitted. Different tags may stage the same key at the same
            time in private L1; shared L1 keeps one writer per key.
        """
        if len(keys) != len(is_temporary):
            raise ValueError(
                f"L1Manager.reserve_write: {len(keys)} keys but "
                f"{len(is_temporary)} is_temporary flags"
            )
        if self._shared_backend is not None:
            return self._shared_reserve_write(keys, is_temporary, layout_desc, tag)

        need_to_allocate: list[tuple[ObjectKey, bool]] = []
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key, is_temp in zip(keys, is_temporary, strict=True):
            if key in self._objects:
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue

            staged = self._get_staging(key, tag)
            if staged is not None:
                if staged.write_lock.is_locked():
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                # The previous reservation expired: hand the buffer over.
                logger.warning(
                    "L1Manager: write reservation on key %s (tag %r) expired; "
                    "handing the buffer to a new writer",
                    key,
                    tag,
                )
                staged.write_lock.lock()
                staged.is_temporary = is_temp
                ret[key] = (L1Error.SUCCESS, staged.memory_obj)
                successful_keys.append(key)
                continue

            need_to_allocate.append((key, is_temp))

        # Early return if no allocation is needed
        if len(need_to_allocate) == 0:
            return ret

        assert self._memory_manager is not None
        err, allocated_objs = self._memory_manager.allocate(
            layout_desc, len(need_to_allocate)
        )

        if err != L1Error.SUCCESS:
            for key, _ in need_to_allocate:
                ret[key] = (L1Error.OUT_OF_MEMORY, None)

            # Free the memory if partial allocation succeeded
            if allocated_objs:
                self._memory_manager.free(allocated_objs)

        else:
            for (key, is_temp), mem_obj in zip(
                need_to_allocate, allocated_objs, strict=True
            ):
                entry = L1ObjectState(
                    memory_obj=mem_obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=is_temp,
                )
                entry.write_lock.lock()
                self._put_staging(key, tag, entry)
                ret[key] = (L1Error.SUCCESS, mem_obj)
                successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_write(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_RESERVED,
                metadata={"keys": successful_keys, "tag": tag},
            )
        )
        return ret

    def _shared_reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve immutable placements with one coordinator batch."""
        backend = self._shared_backend
        assert backend is not None
        if any(is_temporary):
            raise ValueError("shared L1 does not support temporary objects")
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        pending: list[ObjectKey] = []
        for key, _ in zip(keys, is_temporary, strict=True):
            if key in self._objects or key in self._staging:
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue
            pending.append(key)

        try:
            memory_objects = backend.reserve_write(pending, layout_desc)
        except OutOfSpaceError:
            for key in pending:
                ret[key] = (L1Error.OUT_OF_MEMORY, None)
            return ret

        granted_keys = [
            key
            for key, memory_obj in zip(pending, memory_objects, strict=True)
            if memory_obj is not None
        ]
        try:
            for key, memory_obj in zip(pending, memory_objects, strict=True):
                if memory_obj is None:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                entry = L1ObjectState(
                    memory_obj=memory_obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=False,
                )
                entry.write_lock.lock()
                self._put_staging(key, tag, entry)
                successful_keys.append(key)
                ret[key] = (L1Error.SUCCESS, memory_obj)
            for listener in self._registered_listeners:
                listener.on_l1_keys_reserved_write(successful_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_WRITE_RESERVED,
                    metadata={"keys": successful_keys, "tag": tag},
                )
            )
        except BaseException:
            try:
                backend.abort_write(granted_keys)
            finally:
                for key in successful_keys:
                    self._pop_staging(key, tag)
            raise
        return ret

    @l1_mgr_synchronized
    def finish_write(
        self,
        keys: list[ObjectKey],
        tag: str = "",
    ) -> dict[ObjectKey, L1Error]:
        """Finish write access for the given keys.

        Admits ``tag``'s staging objects as the resident objects of their
        keys.

        Temporary objects are unlocked normally but do not emit write-finished
        notifications because they are internal staging buffers that must not
        be routed to L2 storage.

        Args:
            keys: The list of object keys to finish write access for.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: ``tag`` stages nothing for the key.
            KEY_IN_WRONG_STATE: The staging object is not write-locked (its
                reservation expired), which means the writer may have
                caused inconsistent data.

        Note:
            If the key became resident before admission, the staging object
            is discarded, the resident object is kept and ``SUCCESS`` is
            still reported: the data is in L1 either way.
        """
        backend = self._shared_backend
        if backend is not None and len(keys) != len(set(keys)):
            raise ValueError("shared-L1 finish_write keys must be unique")

        ret: dict[ObjectKey, L1Error] = {}
        notification_keys: list[ObjectKey] = []
        notification_keys_meta: list[L1ObjectMeta] = []
        discarded: list[MemoryObj] = []

        for key in keys:
            err, entry = self._take_staging(key, tag, "finish write")
            ret[key] = err
            if err != L1Error.SUCCESS or entry is None:
                continue
            if backend is None:
                if key in self._objects:
                    logger.debug(
                        "L1Manager: discarding staging object for key %s (tag %r): "
                        "the key is already resident",
                        key,
                        tag,
                    )
                    discarded.append(entry.memory_obj)
                    continue
                self._objects[key] = entry
            if not entry.is_temporary:
                notification_keys.append(key)
                notification_keys_meta.append(self._object_meta(entry.memory_obj))

        if backend is not None:
            if any(result != L1Error.SUCCESS for result in ret.values()):
                for key in ret:
                    if ret[key] == L1Error.SUCCESS:
                        ret[key] = L1Error.KEY_IN_WRONG_STATE
                return ret
            # Publish the whole payload batch before unlocking or notifying readers.
            backend.finish_write(keys)
            for key in keys:
                entry = self._pop_staging(key, tag)
                entry.write_lock.unlock()
                self._objects[key] = entry
        else:
            assert self._memory_manager is not None
            self._memory_manager.free(discarded)

        if notification_keys or backend is not None:
            for listener in self._registered_listeners:
                listener.on_l1_keys_write_finished(notification_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_WRITE_FINISHED,
                    metadata={
                        "keys": notification_keys,
                        "meta": notification_keys_meta,
                    },
                )
            )
        return ret

    @l1_mgr_synchronized
    def abort_write(
        self,
        keys: list[ObjectKey],
        tag: str = "",
    ) -> dict[ObjectKey, L1Error]:
        """Abort shared-L1 write reservations without reclaiming their extents.

        Args:
            keys: Object keys whose writes are being aborted.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Each key maps to SUCCESS, KEY_NOT_EXIST, or KEY_IN_WRONG_STATE
            (not write-locked or has readers).

        Raises:
            RuntimeError: This manager is not using shared L1.
        """
        backend = self._shared_backend
        if backend is None:
            raise RuntimeError("abort_write is only supported by shared L1")

        ret: dict[ObjectKey, L1Error] = {}
        aborted_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._get_staging(key, tag)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue
            if not entry.write_lock.is_locked() or entry.read_lock.is_locked():
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue
            ret[key] = L1Error.SUCCESS
            aborted_keys.append(key)

        backend.abort_write(aborted_keys)
        for key in aborted_keys:
            self._pop_staging(key, tag).memory_obj.invalidate()
        return ret

    @l1_mgr_synchronized
    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomically finish write and acquire read lock for the given keys.

        This is used by the prefetch controller after successfully loading
        data from L2 into write-reserved L1 buffers. It transitions the
        object from write-locked to read-locked in a single atomic step,
        preventing a race window where eviction could interfere.

        Args:
            keys: Keys to transition from write-locked to read-locked.
            read_locks: Total read locks acquired per key -- one per TP
                worker that consumes a read lock for the same key
                (e.g. MLA models with TP > 1).
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]); the memory object is the one that
            is now resident and read-locked.

        Errors:
            KEY_NOT_EXIST: ``tag`` stages nothing for the key.
            KEY_IN_WRONG_STATE: The staging object is not write-locked (its
                reservation expired).

        Note:
            Admission follows :meth:`finish_write`. If the key became
            resident before admission, the staging object is discarded and
            the read locks are taken on the resident object, so the caller
            always holds the object that readers see.
        """
        if self._shared_backend is not None:
            raise RuntimeError("shared L1 does not support the L2 prefetch transition")

        total = _validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        successful_keys_meta: list[L1ObjectMeta] = []
        resident_keys: list[ObjectKey] = []
        discarded: list[MemoryObj] = []

        for key in keys:
            err, entry = self._take_staging(key, tag, "finish_write_and_reserve_read")
            if err != L1Error.SUCCESS or entry is None:
                ret[key] = (err, None)
                continue
            resident = self._objects.get(key, None)
            if resident is None:
                self._objects[key] = entry
                successful_keys.append(key)
                successful_keys_meta.append(self._object_meta(entry.memory_obj))
            else:
                logger.debug(
                    "L1Manager: discarding staging object for key %s (tag %r): "
                    "the key is already resident; read-locking the resident one",
                    key,
                    tag,
                )
                discarded.append(entry.memory_obj)
                resident_keys.append(key)
                entry = resident
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        assert self._memory_manager is not None
        self._memory_manager.free(discarded)
        if resident_keys:
            self._report_read_reserved(resident_keys)

        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                metadata={"keys": successful_keys, "meta": successful_keys_meta},
            )
        )
        return ret

    @l1_mgr_synchronized
    def delete(
        self, keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        """Delete the given keys from L1 cache.

        Deletes the resident object and reclaims the key's staging objects
        whose write lock expired (all of them when ``force`` is True).

        Args:
            keys: The list of object keys to delete.
            force: When True, delete even a read-locked key and discard its
                live staging objects. This may free memory a concurrent
                store/read still uses (same hazard as :meth:`clear` with
                ``force=True``); use with care.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IS_LOCKED: The key is read-locked, or a live staging object
                exists for it, so it cannot be deleted. Never returned when
                ``force`` is True.
        """
        if self._shared_backend is not None:
            # M0 never reuses extents, so a local MP must not pretend that
            # deleting its view reclaimed the global object.
            return {
                key: (
                    L1Error.KEY_IS_LOCKED
                    if key in self._objects or key in self._staging
                    else L1Error.KEY_NOT_EXIST
                )
                for key in keys
            }

        need_to_free: list[MemoryObj] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []
        gone_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None and key not in self._staging:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            self._reclaim_staging(key, force)
            if key in self._staging:
                # A live reservation still pins the key.
                ret[key] = L1Error.KEY_IS_LOCKED
                continue
            if entry is None:
                # Only expired reservations existed; they are gone now.
                ret[key] = L1Error.SUCCESS
                gone_keys.append(key)
                continue

            locked = entry.read_lock.is_locked()
            if locked and not force:
                ret[key] = L1Error.KEY_IS_LOCKED
                continue
            if locked:
                logger.warning("L1Manager: force-deleting locked key %s", key)

            need_to_free.append(entry.memory_obj)
            del self._objects[key]
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        self._free_and_report_deleted(successful_keys, need_to_free)
        self._report_staging_gone(gone_keys)
        return ret

    @l1_mgr_synchronized
    def finish_write_and_delete(
        self,
        keys: list[ObjectKey],
        tag: str = "",
    ) -> dict[ObjectKey, L1Error]:
        """Atomically finish write access and discard the staging objects.

        Unlock and discard happen in one critical section, so no other
        component can observe or lock the object in between. The staging
        objects never become resident, and no write-finished notification is
        emitted.

        Args:
            keys: The list of object keys whose staging objects to discard.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: ``tag`` stages nothing for the key.
            KEY_IN_WRONG_STATE: The staging object is not write-locked (its
                reservation expired).

        Raises:
            RuntimeError: Shared L1 does not support write-and-delete.
        """
        if self._shared_backend is not None:
            raise RuntimeError("shared L1 does not support write-and-delete")

        ret: dict[ObjectKey, L1Error] = {}
        discarded: list[MemoryObj] = []
        gone_keys: list[ObjectKey] = []

        for key in keys:
            err, entry = self._take_staging(key, tag, "finish_write_and_delete")
            ret[key] = err
            if err != L1Error.SUCCESS or entry is None:
                continue
            logger.debug(
                "L1Manager: discarding staging object for key %s (tag %r)",
                key,
                tag,
            )
            discarded.append(entry.memory_obj)
            if key not in self._objects and key not in self._staging:
                gone_keys.append(key)

        assert self._memory_manager is not None
        self._memory_manager.free(discarded)
        self._report_staging_gone(gone_keys)
        return ret

    def touch_keys(self, keys: list[ObjectKey]):
        """Touch the given keys, marking the keys as accessed(retrieved or stored).

        Args:
            keys: The list of object keys to touch.
        """
        for listener in self._registered_listeners:
            listener.on_l1_keys_accessed(keys)
        if self._event_bus.has_subscribers(EventType.L1_KEYS_ACCESSED):
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_ACCESSED,
                    metadata={"keys": keys},
                )
            )

    @l1_mgr_synchronized
    def clear(self, force: bool = False) -> None:
        """Clear objects from L1 cache.

        Args:
            force: If True, clear ALL objects including read-locked ones and
                every staging object. This may corrupt in-flight
                store/prefetch operations. If False (default), only clear
                unlocked resident objects and staging objects whose write
                lock expired, keeping read-locked objects and live staging
                objects intact.
        """
        if self._shared_backend is not None:
            logger.info("L1Manager: shared-L1 M0 does not reclaim objects")
            return

        if force:
            staging_count = sum(len(per_tag) for per_tag in self._staging.values())
            logger.warning(
                "L1Manager: force-clearing all %d objects and %d staging objects "
                "(including locked ones). This may corrupt in-flight "
                "store/prefetch operations — use with caution.",
                len(self._objects),
                staging_count,
            )
            all_keys = list(self._objects.keys())
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            all_meta = [self._object_meta(obj) for obj in all_memory_objs]
            assert self._memory_manager is not None
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(all_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": all_keys, "meta": all_meta},
                )
            )
            cleared = set(all_keys)
            staging_keys = [k for k in self._staging if k not in cleared]
            for key in list(self._staging.keys()):
                self._reclaim_staging(key, force=True)
            self._report_staging_gone(staging_keys)
            logger.info(
                "L1Manager: cleared %d objects and %d staging objects, 0 remaining.",
                len(all_keys),
                staging_count,
            )
            return

        keys_to_clear: list[ObjectKey] = []
        objs_to_free: list[MemoryObj] = []
        locked_count = 0

        for key, entry in list(self._objects.items()):
            if entry.read_lock.is_locked():
                locked_count += 1
                continue
            keys_to_clear.append(key)
            objs_to_free.append(entry.memory_obj)

        for key in keys_to_clear:
            del self._objects[key]

        if keys_to_clear:
            self._free_and_report_deleted(keys_to_clear, objs_to_free)

        reclaimed_count = 0
        gone_keys: list[ObjectKey] = []
        for key in list(self._staging.keys()):
            reclaimed = self._reclaim_staging(key, force=False)
            reclaimed_count += reclaimed
            if reclaimed and key not in self._staging and key not in self._objects:
                gone_keys.append(key)
        self._report_staging_gone(gone_keys)
        staging_count = sum(len(per_tag) for per_tag in self._staging.values())

        logger.info(
            "L1Manager: cleared %d objects and %d expired staging objects, "
            "%d locked objects and %d staging objects remaining.",
            len(keys_to_clear),
            reclaimed_count,
            locked_count,
            staging_count,
        )

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Check if a key is eligible for eviction (not locked).

        This method does NOT acquire the global L1Manager lock.
        L1Manager.delete() will check again and safely reject a key
        that became locked between the check and the actual deletion.

        Args:
            key: The object key to check.

        Returns:
            True if the key has a resident object that is not read-locked,
            or a staging object whose write lock expired; False otherwise.
        """
        entry = self._objects.get(key, None)
        if entry is not None and not entry.read_lock.is_locked():
            return True
        per_tag = self._staging.get(key, None)
        if per_tag is None:
            return False
        # Snapshot: this runs without the manager lock.
        return any(
            not staged.write_lock.is_locked() for staged in list(per_tag.values())
        )

    def get_memory_usage(self) -> tuple[int, int]:
        """Get the current memory usage of L1 cache.

        Returns:
            A tuple of (used_memory_bytes, total_memory_bytes).

        Note:
            In the future, we many want to make a "callback" based mechanism
            via "L1ManagerListener" to notify the memory usage changes.
        """
        if self._shared_backend is not None:
            return self._shared_backend.get_memory_usage()
        assert self._memory_manager is not None
        return self._memory_manager.get_memory_usage()

    @l1_mgr_synchronized
    def get_staging_memory_usage(self) -> int:
        """Get the bytes currently held by staging objects.

        Returns:
            The total size in bytes of all write-reserved objects that have
            not been admitted yet.

        Note:
            The value is part of :meth:`get_memory_usage`'s used bytes, not
            in addition to it.
        """
        return self._staging_bytes

    def get_l1_memory_desc(self):
        """Return an L1MemoryDesc describing the underlying L1 memory buffer."""
        if self._shared_backend is not None:
            return self._shared_backend.get_l1_memory_desc()
        assert self._memory_manager is not None
        return self._memory_manager.get_l1_memory_desc()

    def close(self) -> None:
        """Close the L1Manager and free all resources."""
        if self._shared_backend is not None:
            with self._lock:
                self._objects.clear()
                self._staging.clear()
                self._staging_bytes = 0
            self._shared_backend.close()
            return
        with self._lock:
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            for per_tag in self._staging.values():
                all_memory_objs.extend(staged.memory_obj for staged in per_tag.values())
            assert self._memory_manager is not None
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()
            self._staging.clear()
            self._staging_bytes = 0

        assert self._memory_manager is not None
        self._memory_manager.close()

    # Status reporting
    @l1_mgr_synchronized
    def report_status(self) -> dict:
        """Return a status dict describing L1 cache state.

        ``total_object_count`` covers resident and staging objects;
        ``staging_object_count`` / ``staging_bytes`` report the staging
        subset and ``write_locked_count`` the live reservations among them.
        """
        read_locked = 0
        temporary = 0
        for entry in self._objects.values():
            if entry.read_lock.is_locked():
                read_locked += 1
            if entry.is_temporary:
                temporary += 1
        staging = 0
        write_locked = 0
        for per_tag in self._staging.values():
            staging += len(per_tag)
            for staged in per_tag.values():
                if staged.write_lock.is_locked():
                    write_locked += 1
                if staged.is_temporary:
                    temporary += 1
        if self._shared_backend is not None:
            used, total = self._shared_backend.get_memory_usage()
            healthy = self._shared_backend.memcheck()
        else:
            assert self._memory_manager is not None
            used, total = self._memory_manager.get_memory_usage()
            healthy = self._memory_manager.memcheck()
        # ``memory_total_bytes`` is what the allocator currently backs (the
        # grown heap on the lazy tier); this is the declared size. Summed to
        # fit this dict's flat shape; ``0`` means undeclared.
        return {
            "is_healthy": healthy,
            "shared_l1": self._shared_backend is not None,
            "total_object_count": len(self._objects) + staging,
            "write_locked_count": write_locked,
            "read_locked_count": read_locked,
            "temporary_count": temporary,
            "staging_object_count": staging,
            "staging_bytes": self._staging_bytes,
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_configured_bytes": self._configured_capacity_bytes,
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._write_ttl_seconds,
            "read_ttl_seconds": self._read_ttl_seconds,
        }

    # Debugging APIs
    @l1_mgr_synchronized
    def get_object_state(self, key: ObjectKey) -> L1ObjectState | None:
        """Get the internal state of the resident object with the given key.

        Staging objects are not reported here.

        Args:
            key: The object key.

        Returns:
            The L1ObjectState if the object exists, None otherwise.
        """
        return self._objects.get(key, None)

    @l1_mgr_synchronized
    def memcheck(self) -> bool:
        """Perform memory check for L1 cache."""
        if self._shared_backend is not None:
            mem_check_result = self._shared_backend.memcheck()
        else:
            assert self._memory_manager is not None
            mem_check_result = self._memory_manager.memcheck()

        # Log the locked objects for debugging
        num_read_locked = sum(
            1 for entry in self._objects.values() if entry.read_lock.is_locked()
        )
        num_staging = sum(len(per_tag) for per_tag in self._staging.values())

        logger.info(
            "L1Manager memcheck: total objects = %d, read-locked = %d, "
            "staging = %d (%d bytes)",
            len(self._objects),
            num_read_locked,
            num_staging,
            self._staging_bytes,
        )
        return mem_check_result

    # Private helpers

    def _get_staging(self, key: ObjectKey, tag: str) -> L1ObjectState | None:
        """Return ``tag``'s staging object for ``key``, or None."""
        per_tag = self._staging.get(key, None)
        if per_tag is None:
            return None
        return per_tag.get(tag, None)

    def _put_staging(self, key: ObjectKey, tag: str, entry: L1ObjectState) -> None:
        """Store ``entry`` as ``tag``'s staging object for ``key``."""
        self._staging.setdefault(key, {})[tag] = entry
        self._staging_bytes += entry.memory_obj.get_size()

    def _pop_staging(self, key: ObjectKey, tag: str) -> L1ObjectState:
        """Remove and return ``tag``'s staging object for ``key``.

        The caller must have checked that it exists.
        """
        per_tag = self._staging[key]
        entry = per_tag.pop(tag)
        if not per_tag:
            del self._staging[key]
        self._staging_bytes -= entry.memory_obj.get_size()
        return entry

    def _take_staging(
        self,
        key: ObjectKey,
        tag: str,
        op: str,
    ) -> tuple[L1Error, "L1ObjectState | None"]:
        """Validate ``tag``'s staging object; consume it immediately for private L1.

        Args:
            key: The object key.
            tag: The writer's tag.
            op: Operation name used in the wrong-state warning logs.

        Returns:
            (SUCCESS, entry), kept staged for shared-L1 batch publication,
            otherwise removed from the staging table;
            (KEY_NOT_EXIST, None) when ``tag`` stages nothing for ``key``;
            (KEY_IN_WRONG_STATE, None) when the reservation's write lock
            expired (the object stays staged for eviction to reclaim).
        """
        staged = self._get_staging(key, tag)
        if staged is None:
            return L1Error.KEY_NOT_EXIST, None

        if not staged.write_lock.is_locked():
            logger.warning(
                "L1Manager: %s on key %s (tag %r) whose write reservation "
                "expired, potential inconsistent data might be written",
                op,
                key,
                tag,
            )
            return L1Error.KEY_IN_WRONG_STATE, None

        if self._shared_backend is not None:
            return L1Error.SUCCESS, staged
        staged.write_lock.unlock()
        return L1Error.SUCCESS, self._pop_staging(key, tag)

    def _reclaim_staging(self, key: ObjectKey, force: bool) -> int:
        """Free ``key``'s staging objects whose write lock expired.

        Args:
            key: The object key.
            force: When True, free live (still write-locked) staging objects
                too, with a warning.

        Returns:
            The number of staging objects freed.
        """
        per_tag = self._staging.get(key, None)
        if per_tag is None:
            return 0
        freed: list[MemoryObj] = []
        for tag, staged in list(per_tag.items()):
            if staged.write_lock.is_locked():
                if not force:
                    continue
                logger.warning(
                    "L1Manager: force-discarding live staging object %s (tag %r)",
                    key,
                    tag,
                )
            else:
                logger.debug(
                    "L1Manager: reclaiming expired staging object %s (tag %r)",
                    key,
                    tag,
                )
            freed.append(self._pop_staging(key, tag).memory_obj)
        assert self._memory_manager is not None
        self._memory_manager.free(freed)
        return len(freed)

    def _report_read_reserved(self, keys: list[ObjectKey]) -> None:
        """Notify listeners and the event bus that ``keys`` got read locks."""
        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_read(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_RESERVED,
                metadata={"keys": keys},
            )
        )

    def _report_staging_gone(self, keys: list[ObjectKey]) -> None:
        """Tell listeners that ``keys`` left L1 without ever becoming resident.

        Note:
            No event is published: nothing readable was evicted.
        """
        if not keys:
            return
        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(keys)

    def _free_and_report_deleted(
        self,
        keys: list[ObjectKey],
        objs: list[MemoryObj],
    ) -> None:
        """Free ``objs`` and report ``keys`` as deleted to listeners and
        the event bus."""
        freed_meta = [self._object_meta(obj) for obj in objs]
        assert self._memory_manager is not None
        self._memory_manager.free(objs)

        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": keys, "meta": freed_meta},
            )
        )

    def _object_meta(self, memory_obj: MemoryObj) -> L1ObjectMeta:
        """Build the listener-facing metadata for one resident object."""
        if self._shared_backend is not None:
            return L1ObjectMeta(
                size_bytes=memory_obj.get_size(),
                backend=L1BackendType.DEVDAX,
                shared=True,
            )
        assert self._memory_manager is not None
        return L1ObjectMeta(
            size_bytes=memory_obj.get_size(),
            backend=self._memory_manager.get_backend_type(memory_obj),
        )
