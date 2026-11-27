# SPDX-License-Identifier: Apache-2.0
"""Object lifecycle state machine shared by every L1 manager medium."""

# Standard
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
import threading
import time
import weakref

# First Party
from lmcache.lmcache_native import TTLLock
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import L1ManagerConfig, get_configured_capacity_bytes
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerInterface,
    L1ManagerListener,
    L1MemoryDesc,
    L1ObjectMeta,
    L1OperationResult,
)
from lmcache.v1.distributed.memory_manager import L1MemoryManagerProtocol
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge

logger = init_logger(__name__)
_l1_manager_ids = count()

# Upper bound for the count parameter in reserve_read / finish_read
# to prevent a single call from holding the global lock for too long.
MAX_READ_LOCK_COUNT = 128


def next_l1_manager_id() -> int:
    """Return a fresh process-local L1 identity.

    Every L1 binding draws from this counter, so the identity doubles as
    the owner tag stamped on memory objects and stays unique across bindings.
    """
    return next(_l1_manager_ids)


def validate_read_locks(read_locks: int) -> int:
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


@dataclass
class _ReadLease:
    """Owner-scoped TTL read reservation for a single lookup attempt."""

    remaining: int
    expires_at: float
    consumers: set[int] = field(default_factory=set)


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
    """ The legacy anonymous read lock with TTL for the object. """

    is_temporary: bool
    """ Whether the object is temporary (need to be deleted after read). """

    staging_charge: int = 0
    """ Bytes added to the staging counter while the object is staged.

    Recorded at ``_put_staging`` and subtracted unchanged at
    ``_pop_staging``, so a buffer whose logical size is narrowed in
    between (``MemoryObj.set_used_size`` after serialization) does not
    leave phantom staging bytes behind. """

    read_leases: dict[str, _ReadLease] = field(default_factory=dict)
    """ Named lookup reservations keyed by ``read_owner``. """

    max_lease_expiry: float = 0.0
    """ Highest ``expires_at`` across ``read_leases`` for lock-free O(1) checks. """

    active_reads: int = 0
    """ Untimed in-flight read handles pinning this object against eviction. """

    def is_read_locked(self, now: float | None = None) -> bool:
        """Return True if any active read, legacy lock, or live lease holds this key."""
        if self.active_reads > 0 or self.read_lock.is_locked():
            return True
        if self.max_lease_expiry <= 0.0:
            return False
        ts = time.monotonic() if now is None else now
        return self.max_lease_expiry > ts


def l1_mgr_synchronized(func):
    """
    Decorator to mark L1Manager methods as thread-safe
    """

    def wrapper(self: "BaseL1Manager", *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


def _l1_usage_ratio_or_zero(target: "BaseL1Manager | None") -> float:
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


def _l1_staging_bytes_or_zero(target: "BaseL1Manager | None") -> int:
    """Return ``target.get_staging_memory_usage()``, or 0 without a target."""
    if target is None:
        return 0
    return target.get_staging_memory_usage()


# Main classes


class BaseL1Manager(L1ManagerInterface):
    """
    Object lifecycle state machine for L1 cache

    Subclasses bind one memory medium (``DramL1Manager``, ``DevDaxL1Manager``,
    ``GdsL1Manager``): they build the memory manager that allocates the
    buffers and declare the medium's memory tags and gauge label. Everything
    about keys, locks, staging and notifications is shared here.

    A write creates a *staging object* owned by the writer's ``tag``. Staging
    objects are kept apart from the resident objects: readers of the key do
    not see them, and writers with different tags may stage the same key at
    the same time.

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

    # OTel registers each gauge once; its callback snapshots all live managers.
    _gauge_registered: bool = False
    _gauge_lock = threading.Lock()
    _gauge_targets: weakref.WeakValueDictionary[int, "BaseL1Manager"] = (
        weakref.WeakValueDictionary()
    )

    def __init__(
        self, config: L1ManagerConfig, memory_manager: L1MemoryManagerProtocol
    ):
        """Bind the state machine to the medium's memory manager.

        Args:
            config: Lifetime and reporting settings for this L1.
            memory_manager: Allocator facade for the medium; this manager owns
                it from now on and closes it in :meth:`close`.
        """
        self._config = config
        self._l1_manager_id = next_l1_manager_id()
        self._lock = threading.Lock()

        # Resident objects: readable, never write-locked.
        self._objects: dict[ObjectKey, L1ObjectState] = {}
        # Staging objects: key -> writer tag -> write-locked object that is
        # invisible to readers until it is admitted by finish_write.
        self._staging: dict[ObjectKey, dict[str, L1ObjectState]] = {}
        # Bytes held by staging objects (kept in sync with ``_staging``).
        self._staging_bytes: int = 0
        # Objects a prefetch read-locked for a request that has not consumed
        # them yet.
        self._prefetch_locked: dict[ObjectKey, L1ObjectState] = {}
        # Untimed active-read handles: handle_id -> (key, entry).
        self._active_reads: dict[int, tuple[ObjectKey, L1ObjectState]] = {}
        self._active_read_ids = count(1)
        self._closing: bool = False

        self._memory_manager = memory_manager

        # CPU and GDS capacity is fixed at boot. Device-DAX overrides
        # get_capacity_bytes with its live arena pool.
        self._boot_capacity_bytes = get_configured_capacity_bytes(config)
        self._write_ttl_seconds = config.write_ttl_seconds
        self._read_ttl_seconds = config.read_ttl_seconds

        self._registered_listeners: list[L1ManagerListener] = []

        self._event_bus = get_event_bus()

        # Class-level state is addressed through BaseL1Manager on purpose: one
        # registration per process must observe every subclass instance.
        with BaseL1Manager._gauge_lock:
            BaseL1Manager._gauge_targets[self._l1_manager_id] = self
            if not BaseL1Manager._gauge_registered:
                BaseL1Manager._gauge_registered = True
                register_gauge(
                    "lmcache.l1_manager",
                    "lmcache_mp.l1_memory_usage_bytes",
                    "Bytes currently held in L1 cache",
                    lambda: BaseL1Manager._observations(
                        lambda m: m.get_memory_usage()[0]
                    ),
                )
                register_gauge(
                    "lmcache.l1_manager",
                    "lmcache_mp.l1_usage_ratio",
                    "L1 used/total ratio (0.0–1.0)",
                    lambda: BaseL1Manager._observations(_l1_usage_ratio_or_zero),
                )
                register_gauge(
                    "lmcache.l1_manager",
                    "lmcache_mp.l1_staging_bytes",
                    "Bytes held by L1 staging objects (write-reserved, not admitted)",
                    lambda: BaseL1Manager._observations(_l1_staging_bytes_or_zero),
                )

    @property
    def l1_manager_id(self) -> int:
        """Return this manager's stable process-local memory-object owner tag."""
        return self._l1_manager_id

    @property
    @abstractmethod
    def backend_type(self) -> L1BackendType:
        """Return the medium backing this L1.

        Note:
            Fixed at construction. Read by the gauge callback as soon as the
            base constructor registers the manager, so subclasses must be able
            to answer before calling ``super().__init__``.
        """

    @abstractmethod
    def get_memory_tags(self) -> MemoryTags:
        """Return the memory tags of this L1's primary pool.

        Note:
            Tags are fixed at construction.
        """

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for L1Manager events.

        Args:
            listener: The listener to register.
        """
        with self._lock:
            self._registered_listeners.append(listener)

    @staticmethod
    def _sync_lease_expiry(entry: L1ObjectState) -> None:
        """Recompute ``entry.max_lease_expiry`` after mutating ``entry.read_leases``."""
        entry.max_lease_expiry = max(
            (lease.expires_at for lease in entry.read_leases.values()),
            default=0.0,
        )

    def _reserve_lease_locked(
        self,
        entry: L1ObjectState,
        read_owner: str,
        total: int,
        now: float,
    ) -> None:
        """Add ``total`` named read shares for ``read_owner`` and prune expired ones."""
        if entry.read_leases:
            expired = [
                owner
                for owner, lease in entry.read_leases.items()
                if lease.expires_at <= now
            ]
            for owner in expired:
                del entry.read_leases[owner]
        expires_at = now + self._read_ttl_seconds
        existing = entry.read_leases.get(read_owner)
        if existing is not None:
            existing.remaining += total
            existing.expires_at = max(existing.expires_at, expires_at)
        else:
            entry.read_leases[read_owner] = _ReadLease(
                remaining=total, expires_at=expires_at
            )
        self._sync_lease_expiry(entry)

    def _release_lease_locked(
        self, entry: L1ObjectState, read_owner: str, now: float
    ) -> bool:
        """Release all unclaimed named shares for ``read_owner``.

        Returns True iff ``read_owner`` held a live (unexpired) lease.
        """
        lease = entry.read_leases.pop(read_owner, None)
        if lease is None:
            return False
        self._sync_lease_expiry(entry)
        return lease.expires_at > now and lease.remaining > 0

    @l1_mgr_synchronized
    def reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        read_owner: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve read access for the given keys."""
        total = validate_read_locks(read_locks)
        if self._closing:
            return {key: (L1Error.KEY_NOT_READABLE, None) for key in keys}
        now = time.monotonic() if read_owner else 0.0
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if read_owner:
                self._reserve_lease_locked(entry, read_owner, total, now)
            else:
                # TODO(perf): support a count argument in
                # TTLLock.lock() to avoid Python for-loop
                # overhead (TTLLock is C++ std::atomic).
                for _ in range(total):
                    entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        self._report_read_reserved(successful_keys)
        return ret

    @l1_mgr_synchronized
    def unsafe_read(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1OperationResult]:
        """Unsafe read the read-locked objects without adding new read locks."""
        now = time.monotonic()
        ret: dict[ObjectKey, L1OperationResult] = {}

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.is_read_locked(now):
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                continue

            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        return ret

    @l1_mgr_synchronized
    def acquire_read(
        self,
        keys: list[ObjectKey],
        read_owner: str,
        consumer_id: int | None = None,
    ) -> tuple[dict[ObjectKey, L1OperationResult], list[int]]:
        """Atomically promote ``read_owner``'s named leases into untimed active reads.

        All-or-nothing across ``keys``: if any key is missing, invalid, or lacks
        an unclaimed live lease for ``read_owner``, no leases are consumed and
        no active-read handles are created.
        """
        unique_keys = list(dict.fromkeys(keys))
        if self._closing or not read_owner:
            return {k: (L1Error.KEY_NOT_READABLE, None) for k in unique_keys}, []

        now = time.monotonic()
        ret: dict[ObjectKey, L1OperationResult] = {}
        staged: list[tuple[ObjectKey, L1ObjectState, _ReadLease]] = []
        all_ok = True
        for key in unique_keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                all_ok = False
                continue
            if not entry.memory_obj.is_valid():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                all_ok = False
                continue
            lease = entry.read_leases.get(read_owner)
            if (
                lease is None
                or lease.expires_at <= now
                or lease.remaining <= 0
                or (consumer_id is not None and consumer_id in lease.consumers)
            ):
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                all_ok = False
                continue
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            staged.append((key, entry, lease))

        if not all_ok:
            return ret, []

        handles: list[int] = []
        for key, entry, lease in staged:
            lease.remaining -= 1
            if consumer_id is not None:
                lease.consumers.add(consumer_id)
            if lease.remaining <= 0:
                entry.read_leases.pop(read_owner, None)
                self._sync_lease_expiry(entry)
            entry.active_reads += 1
            hid = next(self._active_read_ids)
            self._active_reads[hid] = (key, entry)
            handles.append(hid)
        return ret, handles

    @l1_mgr_synchronized
    def finish_active_reads(self, handles: list[int]) -> None:
        """Release untimed active-read handles returned by :meth:`acquire_read`."""
        if not handles:
            return
        now = time.monotonic()
        need_to_free: list[MemoryObj] = []
        need_to_free_keys: list[ObjectKey] = []
        finished_keys: list[ObjectKey] = []

        for hid in handles:
            record = self._active_reads.pop(hid, None)
            if record is None:
                continue
            key, entry = record
            if entry.active_reads > 0:
                entry.active_reads -= 1
            finished_keys.append(key)
            if not entry.is_read_locked(now):
                if self._prefetch_locked.get(key) is entry:
                    self._prefetch_locked.pop(key, None)
                if entry.is_temporary and self._objects.get(key) is entry:
                    need_to_free.append(entry.memory_obj)
                    need_to_free_keys.append(key)
                    del self._objects[key]

        if not finished_keys:
            return
        freed_meta = [self._object_meta(obj) for obj in need_to_free]
        self._memory_manager.free(need_to_free)
        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(finished_keys)
            listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"l1_tag": self._config.tag, "keys": finished_keys},
            )
        )
        if need_to_free_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={
                        "l1_tag": self._config.tag,
                        "keys": need_to_free_keys,
                        "meta": freed_meta,
                    },
                )
            )

    @l1_mgr_synchronized
    def finish_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        read_owner: str = "",
    ) -> dict[ObjectKey, L1Error]:
        """Finish read access for the given keys."""
        total = validate_read_locks(read_locks)
        now = time.monotonic()
        need_to_free: list[MemoryObj] = []
        need_to_free_keys: list[ObjectKey] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                if not read_owner:
                    logger.warning(
                        "L1Manager: finish read on non-existing key %s, "
                        "potential inconsistent data might be read",
                        key,
                    )
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if read_owner:
                released = self._release_lease_locked(entry, read_owner, now)
                if not released:
                    ret[key] = L1Error.KEY_IN_WRONG_STATE
                    if not entry.is_read_locked(now):
                        self._prefetch_locked.pop(key, None)
                    continue
            else:
                if not entry.read_lock.is_locked():
                    logger.warning(
                        "L1Manager: finish read on non-read-locked key %s, "
                        "potential inconsistent data might be read",
                        key,
                    )
                    ret[key] = L1Error.KEY_IN_WRONG_STATE
                    if not entry.is_read_locked(now):
                        self._prefetch_locked.pop(key, None)
                    continue

                # TODO(perf): support a count argument in
                # TTLLock.unlock() to avoid Python for-loop
                # overhead (TTLLock is C++ std::atomic).
                for _ in range(total):
                    entry.read_lock.unlock()

            if not entry.is_read_locked(now):
                self._prefetch_locked.pop(key, None)
            if entry.is_temporary and not entry.is_read_locked(now):
                need_to_free.append(entry.memory_obj)
                need_to_free_keys.append(key)
                del self._objects[key]

            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        freed_meta = [self._object_meta(obj) for obj in need_to_free]
        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(successful_keys)
            listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"l1_tag": self._config.tag, "keys": successful_keys},
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={
                    "l1_tag": self._config.tag,
                    "keys": need_to_free_keys,
                    "meta": freed_meta,
                },
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
        *,
        allow_resident: bool = False,
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
            allow_resident: Stage a key even when it is already resident.
                Admission then discards the staging object and keeps the
                resident one (see :meth:`finish_write_and_reserve_read`), so a
                writer that raced another writer still ends up holding the
                resident object, at the cost of a redundant write.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Raises:
            ValueError: If ``keys`` and ``is_temporary`` differ in length.

        Errors:
            KEY_NOT_WRITABLE: The key already has a resident object (unless
                ``allow_resident``), or ``tag`` already stages the key.
            OUT_OF_MEMORY: Not enough memory to allocate for the object.

        Note:
            A staging object is invisible to readers and to other tags until
            it is admitted. Different tags may stage the same key at the same
            time.
        """
        if len(keys) != len(is_temporary):
            raise ValueError(
                f"L1Manager.reserve_write: {len(keys)} keys but "
                f"{len(is_temporary)} is_temporary flags"
            )

        need_to_allocate: list[tuple[ObjectKey, bool]] = []
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key, is_temp in zip(keys, is_temporary, strict=True):
            if key in self._objects and not allow_resident:
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
                mem_obj.set_l1_manager(self._l1_manager_id)
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
                metadata={
                    "l1_tag": self._config.tag,
                    "keys": successful_keys,
                    "tag": tag,
                },
            )
        )
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
        ret: dict[ObjectKey, L1Error] = {}
        notification_keys: list[ObjectKey] = []
        notification_keys_meta: list[L1ObjectMeta] = []
        discarded: list[MemoryObj] = []

        for key in keys:
            err, entry = self._take_staging(key, tag, "finish write")
            ret[key] = err
            if err != L1Error.SUCCESS or entry is None:
                continue
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

        self._memory_manager.free(discarded)

        if notification_keys:
            for listener in self._registered_listeners:
                listener.on_l1_keys_write_finished(notification_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_WRITE_FINISHED,
                    metadata={
                        "l1_tag": self._config.tag,
                        "keys": notification_keys,
                        "meta": notification_keys_meta,
                    },
                )
            )
        return ret

    @l1_mgr_synchronized
    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        tag: str = "",
        read_owner: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomically finish write and acquire read lock for the given keys."""
        total = validate_read_locks(read_locks)
        now = time.monotonic() if read_owner else 0.0
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
            if read_owner:
                self._reserve_lease_locked(entry, read_owner, total, now)
            else:
                for _ in range(total):
                    entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        self._memory_manager.free(discarded)
        if resident_keys:
            self._report_read_reserved(resident_keys)

        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                metadata={
                    "l1_tag": self._config.tag,
                    "keys": successful_keys,
                    "meta": successful_keys_meta,
                },
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

            if entry.active_reads > 0:
                logger.warning(
                    "L1Manager: refusing to delete key %s with %d active read(s)",
                    key,
                    entry.active_reads,
                )
                ret[key] = L1Error.KEY_IS_LOCKED
                continue

            locked = entry.is_read_locked()
            if locked and not force:
                ret[key] = L1Error.KEY_IS_LOCKED
                continue
            if locked:
                logger.warning("L1Manager: force-deleting locked key %s", key)

            need_to_free.append(entry.memory_obj)
            del self._objects[key]
            self._prefetch_locked.pop(key, None)
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
        """
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
                    metadata={"l1_tag": self._config.tag, "keys": keys},
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
        if force and self._active_reads:
            logger.warning(
                "L1Manager: force-clear downgraded to safe clear because %d "
                "active read(s) are still in flight.",
                len(self._active_reads),
            )
            force = False

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
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()
            self._prefetch_locked.clear()
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(all_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={
                        "l1_tag": self._config.tag,
                        "keys": all_keys,
                        "meta": all_meta,
                    },
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

        now = time.monotonic()
        for key, entry in list(self._objects.items()):
            if entry.is_read_locked(now):
                locked_count += 1
                continue
            keys_to_clear.append(key)
            objs_to_free.append(entry.memory_obj)

        for key in keys_to_clear:
            del self._objects[key]
            self._prefetch_locked.pop(key, None)

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
        ``delete()`` will check again and safely reject a key
        that became locked between the check and the actual deletion.

        Args:
            key: The object key to check.

        Returns:
            True if the key has a resident object that is not read-locked,
            or a staging object whose write lock expired; False otherwise.
        """
        entry = self._objects.get(key, None)
        if entry is not None and not entry.is_read_locked():
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

    @l1_mgr_synchronized
    def track_prefetch_locked(self, keys: list[ObjectKey]) -> None:
        """Track resident, read-locked objects a prefetch holds for a request.

        The prefetch controller tracks the L1 hits it locks and the objects
        it loads from L2 for a request that has not consumed them yet.

        An object stays tracked while it is resident and read-locked: until
        the last reader finishes, the read TTL expires, or it leaves L1. Each
        object is counted once, however many requests lock it.

        Args:
            keys: Keys of the locked objects. Keys that are not resident or
                not read-locked are skipped.
        """
        now = time.monotonic()
        for key in keys:
            entry = self._objects.get(key, None)
            if entry is not None and entry.is_read_locked(now):
                self._prefetch_locked[key] = entry

    @l1_mgr_synchronized
    def get_prefetch_locked_bytes(self) -> int:
        """Get the bytes prefetch holds read-locked for unconsumed requests.

        Returns:
            The total size in bytes of the objects tracked by
            :meth:`track_prefetch_locked` that are still resident and
            read-locked. Part of :meth:`get_memory_usage`'s used bytes, and
            disjoint from :meth:`get_staging_memory_usage`.
        """
        now = time.monotonic()
        self._prefetch_locked = {
            key: entry
            for key, entry in self._prefetch_locked.items()
            if self._objects.get(key, None) is entry and entry.is_read_locked(now)
        }
        return sum(
            entry.memory_obj.get_size() for entry in self._prefetch_locked.values()
        )

    def get_capacity_bytes(self) -> int:
        """Return the current declared L1 capacity in bytes.

        The base reports the boot-configured capacity; a medium whose
        capacity changes at runtime overrides this.

        Returns:
            The usable capacity in bytes.
        """
        return self._boot_capacity_bytes

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Describe the L1 memory buffer that L2 adapters and channels register.

        Returns:
            The descriptor of the underlying buffer, or None when the medium
            has no registerable buffer.
        """
        return self._memory_manager.get_l1_memory_desc()

    @property
    def config(self) -> L1ManagerConfig:
        """Return this manager's backend and lifetime configuration."""
        return self._config

    def _begin_close(self) -> None:
        """Latch closing state and refuse teardown while active reads remain."""
        with self._lock:
            if self._active_reads:
                raise RuntimeError(
                    f"Cannot close L1Manager '{self._config.tag}': "
                    f"{len(self._active_reads)} active read(s) still in flight"
                )
            self._closing = True

    def close(self) -> None:
        """Free every object, close the memory manager and drop the gauge target."""
        self._begin_close()
        with self._lock:
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            for per_tag in self._staging.values():
                all_memory_objs.extend(staged.memory_obj for staged in per_tag.values())
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()
            self._staging.clear()
            self._staging_bytes = 0
            self._prefetch_locked.clear()

        self._memory_manager.close()
        with BaseL1Manager._gauge_lock:
            BaseL1Manager._gauge_targets.pop(self._l1_manager_id, None)

    # Status reporting
    @l1_mgr_synchronized
    def report_status(self) -> dict:
        """Return a status dict describing L1 cache state.

        ``total_object_count`` covers resident and staging objects;
        ``staging_object_count`` / ``staging_bytes`` report the staging
        subset and ``write_locked_count`` the live reservations among them.
        """
        now = time.monotonic()
        read_locked = 0
        temporary = 0
        for entry in self._objects.values():
            if entry.is_read_locked(now):
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
        used, total = self._memory_manager.get_memory_usage()
        # ``memory_total_bytes`` is what the allocator currently backs (the
        # grown heap on the lazy tier). ``memory_configured_bytes`` is the
        # current declared capacity.
        return {
            "is_healthy": self._memory_manager.memcheck(),
            "total_object_count": len(self._objects) + staging,
            "write_locked_count": write_locked,
            "read_locked_count": read_locked,
            "temporary_count": temporary,
            "staging_object_count": staging,
            "staging_bytes": self._staging_bytes,
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_configured_bytes": self.get_capacity_bytes(),
            "backend": self.backend_type.value,
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
        mem_check_result = self._memory_manager.memcheck()

        # Log the locked objects for debugging
        now = time.monotonic()
        num_read_locked = sum(
            1 for entry in self._objects.values() if entry.is_read_locked(now)
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
        entry.staging_charge = entry.memory_obj.get_size()
        self._staging_bytes += entry.staging_charge

    def _pop_staging(self, key: ObjectKey, tag: str) -> L1ObjectState:
        """Remove and return ``tag``'s staging object for ``key``.

        The caller must have checked that it exists.
        """
        per_tag = self._staging[key]
        entry = per_tag.pop(tag)
        if not per_tag:
            del self._staging[key]
        self._staging_bytes -= entry.staging_charge
        entry.staging_charge = 0
        return entry

    def _take_staging(
        self,
        key: ObjectKey,
        tag: str,
        op: str,
    ) -> tuple[L1Error, "L1ObjectState | None"]:
        """Unlock and remove ``tag``'s staging object for ``key``.

        Args:
            key: The object key.
            tag: The writer's tag.
            op: Operation name used in the wrong-state warning logs.

        Returns:
            (SUCCESS, entry) with the entry removed from the staging table;
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
        self._memory_manager.free(freed)
        return len(freed)

    def _report_read_reserved(self, keys: list[ObjectKey]) -> None:
        """Notify listeners and the event bus that ``keys`` got read locks."""
        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_read(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_RESERVED,
                metadata={"l1_tag": self._config.tag, "keys": keys},
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
        self._memory_manager.free(objs)

        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"l1_tag": self._config.tag, "keys": keys, "meta": freed_meta},
            )
        )

    def _object_meta(self, memory_obj: MemoryObj) -> L1ObjectMeta:
        """Build the listener-facing metadata for one resident object."""
        return L1ObjectMeta(size_bytes=memory_obj.get_size(), backend=self.backend_type)

    @classmethod
    def _observations(
        cls, value: Callable[["BaseL1Manager"], int | float]
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Snapshot ``value`` for every live manager, tagged by L1 tag and medium."""
        with BaseL1Manager._gauge_lock:
            managers = list(BaseL1Manager._gauge_targets.values())
        return [
            (value(m), {"l1_tag": m.config.tag, "backend": m.backend_type.value})
            for m in managers
        ]
