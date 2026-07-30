# SPDX-License-Identifier: Apache-2.0
"""
Managing objects and memory for L1 cache
"""

# Standard
from dataclasses import dataclass
from typing import Literal
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import TTLLock
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.memory_manager import (
    GDSL1MemoryManager,
    L1ManagerProtocol,
    L1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.distributed.shared_l1.client import SharedL1Client
from lmcache.v1.distributed.shared_l1.pool import OutOfSpaceError
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
    """ Whether the object is write-locked. """

    read_lock: TTLLock
    """ The read lock with TTL for the object. """

    is_temporary: bool
    """ Whether the object is temporary (need to be deleted after read). """

    def available_for_read(self) -> bool:
        """Check if the object is available for read.

        Returns:
            True if the object is not write-locked, False otherwise.
        """
        return not self.write_lock.is_locked()

    def available_for_write(self) -> bool:
        """Check if the object is available for write.

        Returns:
            True if the object is not write-locked and has no read locks
            and is not a temporary object, False otherwise.
        """

        return (
            not self.write_lock.is_locked()
            and not self.read_lock.is_locked()
            and not self.is_temporary
        )


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


def _validate_extra_count(extra_count: int) -> int:
    """Validate and clamp extra_count.

    Args:
        extra_count: Extra lock count on top of the
            default 1 lock.

    Returns:
        Clamped value in [0, MAX_READ_LOCK_COUNT - 1].
    """
    if extra_count < 0:
        logger.warning(
            "L1Manager: extra_count=%d is invalid, clamping to 0",
            extra_count,
        )
        return 0
    upper = MAX_READ_LOCK_COUNT - 1
    if extra_count > upper:
        logger.warning(
            "L1Manager: extra_count=%d exceeds limit=%d, clamping",
            extra_count,
            upper,
        )
        return upper
    return extra_count


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


# Main classes


class L1Manager:
    """
    Object lifecycle state machine for L1 cache

          +--------+
          |  None  | <---------------------------------------+
          +--------+                                         |
            |   ^                                            |
            |   | (write lock expired)                       | delete()
            |   |                                            |
    reserve |   +----------------------+                     |
    write() |                          |                     |
            v                          |                     |
      +--------------+           +-----------+               |
      | write_locked |           |           |---------------+
      |              |---------->|   ready   |
      |              | finish_   |           |---------------+
      +--------------+ write()   +-----------+               |
            ^                          |                     |
            |                          | reserve_read()      | finish_read()
            +--------------------------+                     | (if count becomes 0)
                 reserve_write()       |                     |
                                       v                     |
                               +-----------------+           |
                               |   read_locked   |-----------+
                               |   (count = 1)   |
                               +-----------------+
                                     |     ^
                      reserve_read() |     | finish_read()
                                     v     |
                               +-----------------+
                               |   read_locked   |
                               |   (count = 2)   |
                               +-----------------+
                                     |     ^
                      reserve_read() |     | finish_read()
                                     v     |
                                   (...)  (...)
                               (Higher Counts)

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

        self._objects: dict[ObjectKey, L1ObjectState] = {}

        # GDS, Device-DAX, and CPU L1 are mutually exclusive tiers. Each tier
        # owns its backing allocator instead of branching inside the CPU path.
        self._memory_manager: L1ManagerProtocol | None
        self._shared_client: SharedL1Client | None = None
        if config.shared_l1_config is not None:
            self._memory_manager = None
            self._shared_client = SharedL1Client(
                config.shared_l1_config,
                config.memory_config,
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
        return self._shared_client is not None

    @l1_mgr_synchronized
    def reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve read access for the given keys.

        Args:
            keys: The list of object keys to reserve
                read access for.
            extra_count: Extra read locks on top of the
                default 1 lock.  Total locks acquired per
                key = 1 + extra_count.  Useful when multiple
                workers each consume one read lock for the
                same key (e.g. MLA models with TP > 1).
        Returns:
            A dictionary mapping each object key to a tuple
            of (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_NOT_READABLE: The key exists but is not
                readable.
        """
        if self._shared_client is not None:
            return self._shared_reserve_read(keys, extra_count)

        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.available_for_read():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                continue

            # TODO(perf): support a count argument in
            # TTLLock.lock() to avoid Python for-loop
            # overhead (TTLLock is C++ std::atomic).
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    def _shared_reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Acquire a whole key list and attach remote tensor views."""
        client = self._shared_client
        assert client is not None
        extra_count = _validate_extra_count(extra_count)
        if extra_count:
            raise ValueError("shared L1 currently supports TP=1 reads only")
        ret: dict[ObjectKey, L1OperationResult] = {}
        reserved_keys: list[ObjectKey] = []
        released_keys: set[ObjectKey] = set()
        successful_keys: list[ObjectKey] = []

        try:
            memory_objects = client.reserve_reads(keys)
            reserved_keys = [
                key
                for key, memory_obj in zip(keys, memory_objects, strict=True)
                if memory_obj is not None
            ]
            for key, memory_obj in zip(keys, memory_objects, strict=True):
                if memory_obj is None:
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                    continue
                entry = self._objects.get(key)
                if entry is not None and not entry.available_for_read():
                    client.abort_reads([key])
                    released_keys.add(key)
                    ret[key] = (L1Error.KEY_NOT_READABLE, None)
                    continue
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
            try:
                client.abort_reads(
                    [key for key in reserved_keys if key not in released_keys]
                )
            finally:
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
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1Error]:
        """Finish read access for the given keys.

        Will delete the object if it is temporary and read
        count reaches zero.

        Args:
            keys: The list of object keys to finish read
                access for.
            extra_count: Extra read locks to release on top
                of the default 1.  Must match the
                ``extra_count`` used in the corresponding
                ``reserve_read`` call.

        Returns:
            A dictionary mapping each object key to an
            L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is write-locked or
                non-read-locked, which means the reader may
                read inconsistent data.
        """
        if self._shared_client is not None:
            return self._shared_finish_read(keys, extra_count)

        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
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

            if entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish read on write-locked key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if not entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish read on non-read-locked key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

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

        assert self._memory_manager is not None
        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(successful_keys)
            listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"keys": successful_keys},
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": need_to_free_keys},
            )
        )

        return ret

    def _shared_finish_read(
        self,
        keys: list[ObjectKey],
        extra_count: int,
    ) -> dict[ObjectKey, L1Error]:
        """Release one coordinator batch before dropping local read locks."""
        client = self._shared_client
        assert client is not None
        extra_count = _validate_extra_count(extra_count)
        if extra_count:
            raise ValueError("shared L1 currently supports TP=1 reads only")
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key)
            if (
                entry is None
                or entry.write_lock.is_locked()
                or not entry.read_lock.is_locked()
            ):
                ret[key] = (
                    L1Error.KEY_NOT_EXIST
                    if entry is None
                    else L1Error.KEY_IN_WRONG_STATE
                )
                continue
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        client.finish_reads(successful_keys)
        for key in successful_keys:
            self._objects[key].read_lock.unlock()
        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"] = "all",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve write access for the given keys.

        Args:
            keys: The list of object keys to reserve write access for.
            is_temporary: The list of booleans indicating whether each key is
                temporary.
            shape_spec: The memory layout description for the objects to be
                allocated.
            mode (Literal["new", "update", "all"]): Reservation mode.
            - "new": Reserve only new objects that do not exist.
            - "update": Reserve only existing objects for update.
            - "all": Reserve all writable objects regardless of existence.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_WRITABLE: The key exists but is not writable.
            OUT_OF_MEMORY: Not enough memory to allocate for the object.
        """
        if self._shared_client is not None:
            return self._shared_reserve_write(
                keys,
                is_temporary,
                layout_desc,
                mode,
            )

        need_to_allocate: list[tuple[ObjectKey, bool]] = []
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key, is_temp in zip(keys, is_temporary, strict=False):
            entry = self._objects.get(key, None)
            if entry is None:
                need_to_allocate.append((key, is_temp))
                continue

            if mode == "new":
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue

            if not entry.available_for_write():
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue

            entry.write_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        # Early return if no allocation is needed
        if len(need_to_allocate) == 0:
            return ret

        # Don't allow allocation in "update" mode
        if mode == "update":
            for key, _ in need_to_allocate:
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
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
                need_to_allocate, allocated_objs, strict=False
            ):
                self._objects[key] = L1ObjectState(
                    memory_obj=mem_obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=is_temp,
                )
                self._objects[key].write_lock.lock()
                ret[key] = (L1Error.SUCCESS, mem_obj)
                successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_write(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    def _shared_reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"],
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve immutable placements with one coordinator batch."""
        client = self._shared_client
        assert client is not None
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        pending: list[tuple[ObjectKey, bool]] = []
        for key, is_temp in zip(keys, is_temporary, strict=False):
            if mode == "update" or key in self._objects:
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue
            pending.append((key, is_temp))

        try:
            memory_objects = client.reserve_writes(
                [key for key, _ in pending],
                layout_desc,
            )
        except OutOfSpaceError:
            for key, _ in pending:
                ret[key] = (L1Error.OUT_OF_MEMORY, None)
            return ret

        granted_keys = [
            key
            for (key, _), memory_obj in zip(pending, memory_objects, strict=True)
            if memory_obj is not None
        ]
        try:
            for (key, is_temp), memory_obj in zip(
                pending,
                memory_objects,
                strict=True,
            ):
                if memory_obj is None:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                entry = L1ObjectState(
                    memory_obj=memory_obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=is_temp,
                )
                entry.write_lock.lock()
                self._objects[key] = entry
                successful_keys.append(key)
                ret[key] = (L1Error.SUCCESS, memory_obj)
            for listener in self._registered_listeners:
                listener.on_l1_keys_reserved_write(successful_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_WRITE_RESERVED,
                    metadata={"keys": successful_keys},
                )
            )
        except BaseException:
            try:
                client.abort_writes(granted_keys)
            finally:
                for key in successful_keys:
                    del self._objects[key]
            raise
        return ret

    @l1_mgr_synchronized
    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1Error]:
        """Finish write access for the given keys.

        Args:
            keys: The list of object keys to finish write access for.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is not write-locked, or it's read-locked,
                which means the writer may have caused inconsistent data.
        """
        if self._shared_client is not None:
            return self._shared_finish_write(keys)

        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if not entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish write on non-write-locked key %s, "
                    "potential inconsistent data might be written",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish write on read-locked key %s, "
                    "potential inconsistent data might be written",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            entry.write_lock.unlock()
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_write_finished(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    def _shared_finish_write(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1Error]:
        """Publish the complete payload batch before one metadata commit."""
        client = self._shared_client
        assert client is not None
        ret: dict[ObjectKey, L1Error] = {}
        if len(keys) != len(set(keys)):
            raise ValueError("shared-L1 finish_write keys must be unique")

        failed = False
        for key in keys:
            entry = self._objects.get(key)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                failed = True
                continue
            if not entry.write_lock.is_locked() or entry.read_lock.is_locked():
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                failed = True
                continue
            ret[key] = L1Error.SUCCESS

        if failed:
            # No object in a request batch may become globally readable unless
            # every local reservation is ready to commit.
            for key, result in list(ret.items()):
                if result == L1Error.SUCCESS:
                    ret[key] = L1Error.KEY_IN_WRONG_STATE
            return ret

        client.finish_writes(keys)
        for key in keys:
            entry = self._objects[key]
            entry.write_lock.unlock()

        for listener in self._registered_listeners:
            listener.on_l1_keys_write_finished(keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED,
                metadata={"keys": keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def abort_write(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1Error]:
        """Abort coordinator-owned writes that never became readable.

        The existing local allocators do not expose an abort operation. This
        method is therefore intentionally limited to the shared-L1 path, where
        leaving a failed transfer in ``WRITING`` would block that object key for
        every MP server connected to the coordinator.
        """
        client = self._shared_client
        if client is None:
            raise RuntimeError("abort_write is only supported by shared L1")

        ret: dict[ObjectKey, L1Error] = {}
        aborted_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue
            if not entry.write_lock.is_locked() or entry.read_lock.is_locked():
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue
            ret[key] = L1Error.SUCCESS
            aborted_keys.append(key)

        client.abort_writes(aborted_keys)
        for key in aborted_keys:
            self._objects[key].memory_obj.invalidate()
            del self._objects[key]
        if aborted_keys:
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(aborted_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": aborted_keys},
                )
            )
        return ret

    @l1_mgr_synchronized
    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomically finish write and acquire read lock for the given keys.

        This is used by the prefetch controller after successfully loading
        data from L2 into write-reserved L1 buffers. It transitions the
        object from write-locked to read-locked in a single atomic step,
        preventing a race window where eviction could interfere.

        Args:
            keys: Keys to transition from write-locked to read-locked.
            extra_count: Extra read locks on top of the default 1 lock.
                Total locks acquired per key = 1 + extra_count.  Useful
                when multiple TP workers each consume one read lock for
                the same key (e.g. MLA models with TP > 1).

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is not write-locked, or it already
                has read locks.
        """
        if self._shared_client is not None:
            raise RuntimeError("shared L1 does not support the L2 prefetch transition")

        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish_write_and_reserve_read on "
                    "non-write-locked key %s",
                    key,
                )
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue

            if entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish_write_and_reserve_read on read-locked key %s",
                    key,
                )
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue

            entry.write_lock.unlock()
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def delete(
        self, keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        """Delete the given keys from L1 cache.

        Args:
            keys: The list of object keys to delete.
            force: When True, delete even a read/write-locked key. This may free
                memory a concurrent store/read still uses (same hazard as
                :meth:`clear` with ``force=True``); use with care.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IS_LOCKED: The key is write-locked or read-locked and cannot be
                deleted. Never returned when ``force`` is True.
        """
        if self._shared_client is not None:
            # M0 never reuses extents, so a local MP must not pretend that
            # deleting its view reclaimed the global object.
            return {
                key: (
                    L1Error.KEY_IS_LOCKED
                    if key in self._objects
                    else L1Error.KEY_NOT_EXIST
                )
                for key in keys
            }

        need_to_free: list[MemoryObj] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            locked = entry.read_lock.is_locked() or entry.write_lock.is_locked()
            if locked and not force:
                ret[key] = L1Error.KEY_IS_LOCKED
                continue
            if locked:
                logger.warning("L1Manager: force-deleting locked key %s", key)

            need_to_free.append(entry.memory_obj)
            del self._objects[key]
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        assert self._memory_manager is not None
        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    def touch_keys(self, keys: list[ObjectKey]):
        """Touch the given keys, marking the keys as accessed(retrieved or stored).

        Args:
            keys: The list of object keys to touch.
        """
        for listener in self._registered_listeners:
            listener.on_l1_keys_accessed(keys)

    @l1_mgr_synchronized
    def clear(self, force: bool = False) -> None:
        """Clear objects from L1 cache.

        Args:
            force: If True, clear ALL objects including locked ones.
                This may corrupt in-flight store/prefetch operations.
                If False (default), only clear unlocked objects, keeping
                write-locked and read-locked objects intact.
        """
        if self._shared_client is not None:
            logger.info("L1Manager: shared-L1 M0 does not reclaim objects")
            return

        if force:
            logger.warning(
                "L1Manager: force-clearing all %d objects "
                "(including locked ones). This may corrupt in-flight "
                "store/prefetch operations — use with caution.",
                len(self._objects),
            )
            all_keys = list(self._objects.keys())
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            assert self._memory_manager is not None
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(all_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": all_keys},
                )
            )
            logger.info(
                "L1Manager: cleared %d objects, 0 remaining.",
                len(all_keys),
            )
            return

        keys_to_clear: list[ObjectKey] = []
        objs_to_free: list[MemoryObj] = []
        locked_count = 0

        for key, entry in list(self._objects.items()):
            if entry.write_lock.is_locked() or entry.read_lock.is_locked():
                locked_count += 1
                continue
            keys_to_clear.append(key)
            objs_to_free.append(entry.memory_obj)

        for key in keys_to_clear:
            del self._objects[key]

        assert self._memory_manager is not None
        self._memory_manager.free(objs_to_free)

        if keys_to_clear:
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(keys_to_clear)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": keys_to_clear},
                )
            )

        logger.info(
            "L1Manager: cleared %d objects, %d locked objects remaining.",
            len(keys_to_clear),
            locked_count,
        )

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Check if a key is eligible for eviction (not locked).

        This method does NOT acquire the global L1Manager lock.
        L1Manager.delete() will check again and safely reject a key
        that became locked between the check and the actual deletion.

        Args:
            key: The object key to check.

        Returns:
            True if the key exists and is not locked (neither read-locked
            nor write-locked), False otherwise.
        """
        entry = self._objects.get(key, None)
        if entry is None:
            return False
        return not entry.read_lock.is_locked() and not entry.write_lock.is_locked()

    def get_memory_usage(self) -> tuple[int, int]:
        """Get the current memory usage of L1 cache.

        Returns:
            A tuple of (used_memory_bytes, total_memory_bytes).

        Note:
            In the future, we many want to make a "callback" based mechanism
            via "L1ManagerListener" to notify the memory usage changes.
        """
        if self._shared_client is not None:
            return self._shared_client.get_memory_usage()
        assert self._memory_manager is not None
        return self._memory_manager.get_memory_usage()

    def get_l1_memory_desc(self):
        """Return an L1MemoryDesc describing the underlying L1 memory buffer."""
        if self._shared_client is not None:
            return self._shared_client.get_l1_memory_desc()
        assert self._memory_manager is not None
        return self._memory_manager.get_l1_memory_desc()

    def close(self) -> None:
        """Close the L1Manager and free all resources."""
        if self._shared_client is not None:
            with self._lock:
                self._objects.clear()
            self._shared_client.close()
            return
        with self._lock:
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            assert self._memory_manager is not None
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()

        assert self._memory_manager is not None
        self._memory_manager.close()

    # Status reporting
    @l1_mgr_synchronized
    def report_status(self) -> dict:
        """Return a status dict describing L1 cache state."""
        write_locked = 0
        read_locked = 0
        temporary = 0
        for entry in self._objects.values():
            if entry.write_lock.is_locked():
                write_locked += 1
            if entry.read_lock.is_locked():
                read_locked += 1
            if entry.is_temporary:
                temporary += 1
        if self._shared_client is not None:
            used, total = self._shared_client.get_memory_usage()
            healthy = self._shared_client.memcheck()
        else:
            assert self._memory_manager is not None
            used, total = self._memory_manager.get_memory_usage()
            healthy = self._memory_manager.memcheck()
        return {
            "is_healthy": healthy,
            "shared_l1": self._shared_client is not None,
            "total_object_count": len(self._objects),
            "write_locked_count": write_locked,
            "read_locked_count": read_locked,
            "temporary_count": temporary,
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._write_ttl_seconds,
            "read_ttl_seconds": self._read_ttl_seconds,
        }

    # Debugging APIs
    @l1_mgr_synchronized
    def get_object_state(self, key: ObjectKey) -> L1ObjectState | None:
        """Get the internal state of the object with the given key.

        Args:
            key: The object key.

        Returns:
            The L1ObjectState if the object exists, None otherwise.
        """
        return self._objects.get(key, None)

    @l1_mgr_synchronized
    def memcheck(self) -> bool:
        """Perform memory check for L1 cache."""
        if self._shared_client is not None:
            mem_check_result = self._shared_client.memcheck()
        else:
            assert self._memory_manager is not None
            mem_check_result = self._memory_manager.memcheck()

        # Log the locked objects for debugging
        num_write_locked = 0
        num_read_locked = 0
        for key, entry in self._objects.items():
            if entry.write_lock.is_locked():
                num_write_locked += 1
            if entry.read_lock.is_locked():
                num_read_locked += 1

        logger.info(
            "L1Manager memcheck: total objects = %d, write-locked = %d, "
            "read-locked = %d",
            len(self._objects),
            num_write_locked,
            num_read_locked,
        )
        return mem_check_result
