# SPDX-License-Identifier: Apache-2.0

"""CXL-backed sibling of ``L1Manager`` for MP mode (maru).

``MaruL1Manager`` implements the ``L1ManagerInterface`` control surface for a
cross-instance shared L1 tier: membership and read protection live in the
MaruServer directory (``pin_count``), not in a local object table, so the
stock ``L1Manager`` state machine cannot be reused. Locally it keeps only
in-flight staging: ``_pending_write`` (reserved-but-unregistered pages) and
``_pending_read`` (pinned reads; the refcount balances N reserves = N pins =
N unpins).

Provenance convention: ``PARITY(L1Manager.X)`` marks behavior mirrored from
the stock manager (keep in sync with ``l1_manager.py``); ``MARU:`` marks
maru-specific logic.

A background sweeper reclaims staging whose TTL elapses (an abandoned client's
orphan write pages / read pins). Known gaps: (1) a pin whose RPC reply is lost
leaks server-side; reconciliation (per-instance pin ledger) is a maru-side
design item. (2) the prefetch
controller's load-failure cleanup calls ``finish_write`` then ``delete`` on the
failed keys -- ``finish_write`` publishes the page to the shared directory, so a
peer that pins it in the window before ``delete`` makes ``delete`` refuse
(KEY_IS_LOCKED, which the caller ignores) and the key lingers. Rare (L2-load
failure + concurrent same-key lookup); a caller-side batch-abort is the clean
future fix.
"""

# Standard
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Literal,
    ParamSpec,
    TypeVar,
)
import functools
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, MaruL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener, L1MemoryDesc
from lmcache.v1.distributed.l1_manager import (
    MAX_READ_LOCK_COUNT,
    L1ObjectState,
    L1OperationResult,
)
from lmcache.v1.distributed.memory_manager.maru_memory_allocator import (
    MaruMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

if TYPE_CHECKING:
    # Third Party
    from maru import MaruHandler
    from maru_handler.memory import MemoryInfo

logger = init_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _maru_l1_synchronized(
    func: Callable[Concatenate["MaruL1Manager", P], R],
) -> Callable[Concatenate["MaruL1Manager", P], R]:
    """Serialize the method under the manager's non-reentrant lock."""

    @functools.wraps(func)
    def wrapper(self: "MaruL1Manager", *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


def object_key_to_string(key: ObjectKey) -> str:
    """Encode an ObjectKey as the flat string key MaruServer uses.

    Format ``<model>@<kv_rank:08x>@<chunk_hash_hex>@<group>[@<salt>]`` --
    follows the MP L2 adapter key convention (every ObjectKey field encoded).

    Args:
        key: The object key to encode.

    Returns:
        The flat string key.
    """
    base = (
        f"{key.model_name}@{key.kv_rank:08x}"
        f"@{key.chunk_hash.hex()}@{key.object_group_id}"
    )
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


def _clamp_extra_count(extra_count: int) -> int:
    # PARITY(L1Manager._validate_extra_count): warn and clamp to [0, MAX-1].
    if extra_count < 0:
        logger.warning(
            "MaruL1Manager: extra_count=%d is invalid, clamping to 0", extra_count
        )
        return 0
    upper = MAX_READ_LOCK_COUNT - 1
    if extra_count > upper:
        logger.warning(
            "MaruL1Manager: extra_count=%d exceeds limit=%d, clamping",
            extra_count,
            upper,
        )
        return upper
    return extra_count


@dataclass
class _PendingRead:
    """A pinned read staged between reserve_read and the last finish_read.

    ``deadline`` is the monotonic time after which the sweeper treats the read
    as orphaned; it defaults to never (the real reserve path sets a finite
    value and refreshes it on overlapping reserves).
    """

    mem_obj: MemoryObj
    refcount: int
    is_temporary: bool = False
    deadline: float = float("inf")


@dataclass
class _PendingWrite:
    """A reserved page staged between reserve_write and finish_write.

    ``deadline`` is the monotonic time after which the sweeper reclaims the
    orphaned page (defaults to never; the reserve path sets a finite value).
    """

    mem_obj: MemoryObj
    is_temporary: bool
    deadline: float = float("inf")


class MaruL1Manager:
    """L1 control manager over the maru shared CXL pool.

    Structurally satisfies ``L1ManagerInterface`` (no nominal base). All
    tiering decisions stay with the stock controllers; this class executes
    them against the MaruServer directory and the CXL allocator.
    """

    def __init__(self, config: L1ManagerConfig) -> None:
        maru_config = config.memory_config.maru_config
        if maru_config is None:
            raise ValueError("MaruL1Manager requires memory_config.maru_config")
        self._config: MaruL1Config = maru_config
        self._write_ttl_seconds = config.write_ttl_seconds
        self._read_ttl_seconds = config.read_ttl_seconds
        self._lock = threading.Lock()
        self._allocator = MaruMemoryAllocator(maru_config)
        self._pending_read: dict[ObjectKey, _PendingRead] = {}
        self._pending_write: dict[ObjectKey, _PendingWrite] = {}
        self._registered_listeners: list[L1ManagerListener] = []
        # PARITY(L1Manager): observability events go to the shared event bus
        # alongside the listener notifications.
        self._event_bus = get_event_bus()
        # MARU: W1+R1 orphan sweeper -- a crashed/abandoned client leaves write
        # pages or read pins behind; reclaim them once their TTL elapses.
        self._sweep_interval = max(
            1.0, min(self._write_ttl_seconds, self._read_ttl_seconds) / 4
        )
        self._stop_event = threading.Event()
        self._sweeper = threading.Thread(
            target=self._sweep_loop, name="maru-l1-sweeper", daemon=True
        )
        self._sweeper.start()

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for ``on_l1_keys_*`` events.

        Args:
            listener: The listener to register.
        """
        # PARITY(L1Manager.register_listener): inline lock, append only.
        with self._lock:
            self._registered_listeners.append(listener)

    def _publish(self, event_type: EventType, keys: list[ObjectKey]) -> None:
        """Publish an L1 observability event (PARITY with stock L1Manager).

        Args:
            event_type: The L1 event type to publish.
            keys: The affected object keys (event metadata).
        """
        self._event_bus.publish(Event(event_type=event_type, metadata={"keys": keys}))

    def _safe_unpin(self, handler: "MaruHandler", key_strs: list[str]) -> None:
        """Release server pins, logging (not raising) on RPC failure."""
        if not key_strs:
            return
        try:
            results = handler.batch_unpin(key_strs)
        except Exception:
            logger.exception(
                "MaruL1Manager: batch_unpin failed for %d pins", len(key_strs)
            )
            return
        failed = sum(1 for ok in results if not ok)
        if failed:
            # A refused unpin means the server held no pin: a balance bug.
            logger.warning(
                "MaruL1Manager: %d/%d unpins had no pin to release",
                failed,
                len(key_strs),
            )

    def _pin_retrieve_stage(
        self,
        keys: list[ObjectKey],
        total: int,
        ret: dict[ObjectKey, L1OperationResult],
        successful_keys: list[ObjectKey],
    ) -> None:
        """Pin ``total`` units per key, resolve the page, and stage a read.

        Shared by ``reserve_read`` and the retained-promote re-resolve. On a
        hit sets ``ret[k] = (SUCCESS, obj)`` and appends ``k`` to
        ``successful_keys``; overlapping reads accumulate the refcount. Partial
        or unresolvable pins are rolled back; non-hit keys keep whatever value
        the caller pre-set in ``ret``.

        Args:
            keys: Keys to pin and stage (all pinned; misses roll back).
            total: Protection units (pins) to take per key.
            ret: Result map, mutated in place for hits.
            successful_keys: List extended in place with each staged key.
        """
        if not keys:
            return
        handler = self._allocator.handler
        key_strs = [object_key_to_string(k) for k in keys]

        # MARU: one RPC takes `total` pins per key (repeat-encoding); a key
        # is a hit only if all its pins landed.
        pin_list = [ks for ks in key_strs for _ in range(total)]
        try:
            pin_results = handler.batch_pin(pin_list)
        except Exception:
            logger.exception("MaruL1Manager: batch_pin failed for %d keys", len(keys))
            return
        if len(pin_results) != len(pin_list):
            # Malformed reply: release whatever was reported taken.
            logger.error(
                "MaruL1Manager: batch_pin returned %d/%d results; rolling back",
                len(pin_results),
                len(pin_list),
            )
            self._safe_unpin(
                handler,
                [ks for ks, ok in zip(pin_list, pin_results, strict=False) if ok],
            )
            return

        hits: list[int] = []
        rollback: list[str] = []  # partial pins to release
        for i, ks in enumerate(key_strs):
            got = sum(1 for ok in pin_results[i * total : (i + 1) * total] if ok)
            if got == total:
                hits.append(i)
            elif got:
                rollback.extend([ks] * got)

        mem_infos: list["MemoryInfo | None"] = []
        if hits:
            try:
                mem_infos = handler.batch_retrieve([key_strs[i] for i in hits])
            except Exception:
                logger.exception(
                    "MaruL1Manager: batch_retrieve failed for %d keys", len(hits)
                )
                for i in hits:
                    rollback.extend([key_strs[i]] * total)
                self._safe_unpin(handler, rollback)
                return
            # Normalize a malformed reply; missing tails roll back below.
            mem_infos = list(mem_infos[: len(hits)])
            mem_infos += [None] * (len(hits) - len(mem_infos))

        for i, mi in zip(hits, mem_infos, strict=False):
            mem_obj = (
                self._allocator.get_by_location(
                    region_id=mi.region_id,
                    page_index=mi.page_index,
                    actual_size=len(mi.view),
                )
                if mi is not None
                else None
            )
            if mem_obj is None:
                # MARU: pinned but unresolvable (raced delete / pool miss).
                rollback.extend([key_strs[i]] * total)
                continue
            k = keys[i]
            deadline = time.monotonic() + self._read_ttl_seconds
            staged = self._pending_read.get(k)
            if staged is not None:
                # Overlapping reserve: same CXL page, one staged object.
                # Refresh the TTL (mirrors a stock re-lock extending it).
                staged.refcount += total
                staged.deadline = deadline
                ret[k] = (L1Error.SUCCESS, staged.mem_obj)
            else:
                self._pending_read[k] = _PendingRead(
                    mem_obj=mem_obj, refcount=total, deadline=deadline
                )
                ret[k] = (L1Error.SUCCESS, mem_obj)
            successful_keys.append(k)

        self._safe_unpin(handler, rollback)

    def _store_staged(
        self, staged: list[tuple[ObjectKey, _PendingWrite]]
    ) -> tuple[list[ObjectKey], dict[ObjectKey, L1Error]]:
        """Register staged write pages in the directory and classify the result.

        Shared by ``finish_write`` and the retained-promote path.

        Args:
            staged: (key, pending write) pairs whose pages to register.

        Returns:
            ``(registered, errors)``: ``registered`` are keys the server stored
            or dup-skipped (the directory page is authoritative); ``errors``
            maps every other key to KEY_IN_WRONG_STATE. A handle-build failure
            reclaims the pages; a store RPC failure leaves them (unknown server
            state must never be recycled).
        """
        errors: dict[ObjectKey, L1Error] = {}
        if not staged:
            return [], errors
        try:
            handles = [
                self._allocator.create_store_handle(e.mem_obj) for _, e in staged
            ]
        except Exception:
            # Pages never reached the server -- safe to reclaim now.
            logger.exception(
                "MaruL1Manager: create_store_handle failed for %d keys", len(staged)
            )
            for k, entry in staged:
                self._allocator.abort_alloc(entry.mem_obj)
                errors[k] = L1Error.KEY_IN_WRONG_STATE
            return [], errors

        try:
            results = self._allocator.handler.batch_store(
                [object_key_to_string(k) for k, _ in staged], handles
            )
        except Exception:
            # MARU: server state unknown -- the pages must NOT be recycled
            # (a registered page must never return to the free list).
            logger.exception(
                "MaruL1Manager: batch_store failed for %d keys", len(staged)
            )
            for k, _ in staged:
                errors[k] = L1Error.KEY_IN_WRONG_STATE
            return [], errors

        registered: list[ObjectKey] = []
        for i, (k, entry) in enumerate(staged):
            ok = results[i] if i < len(results) else None
            if ok:
                # True covers newly-registered and dup-skip (page auto-freed).
                registered.append(k)
            elif ok is None:
                # Missing reply entry: state unknown -- do not recycle.
                errors[k] = L1Error.KEY_IN_WRONG_STATE
            else:
                # Definitively not registered -- reclaim the page.
                self._allocator.abort_alloc(entry.mem_obj)
                errors[k] = L1Error.KEY_IN_WRONG_STATE
        return registered, errors

    def _sweep_loop(self) -> None:
        """Daemon loop: sweep expired staging until ``close`` stops it."""
        while not self._stop_event.wait(self._sweep_interval):
            try:
                self._sweep_once()
            except Exception:
                logger.exception("MaruL1Manager: sweep iteration failed")

    def _sweep_once(self) -> None:
        """Reclaim staging whose TTL elapsed (orphan write pages / read pins).

        MARU: abandonment can only be judged by time -- a refcount says how
        many holds exist, not whether they will ever be released. Mirrors the
        stock write_lock/read_lock TTL: an expired write page is returned to
        the owner (abort_alloc) and an expired read releases its pins (a
        temporary read reclaims its private page instead). No listener fires --
        a late finish_read/unsafe_read then sees KEY_NOT_EXIST and recomputes
        (the same failure path as a stock TTL expiry), and firing across the
        daemon thread would be a novel hazard for the stock listeners.
        """
        now = time.monotonic()
        with self._lock:
            expired_writes = [
                k for k, e in self._pending_write.items() if e.deadline <= now
            ]
            expired_reads = [
                k for k, e in self._pending_read.items() if e.deadline <= now
            ]
            if not expired_writes and not expired_reads:
                return
            for k in expired_writes:
                write_entry = self._pending_write.pop(k)
                try:
                    self._allocator.abort_alloc(write_entry.mem_obj)
                except Exception:
                    logger.exception(
                        "MaruL1Manager: sweep failed to reclaim write page %s", k
                    )
            to_unpin: list[str] = []
            for k in expired_reads:
                read_entry = self._pending_read.pop(k)
                if read_entry.is_temporary:
                    try:
                        self._allocator.abort_alloc(read_entry.mem_obj)
                    except Exception:
                        logger.exception(
                            "MaruL1Manager: sweep failed to reclaim read page %s", k
                        )
                else:
                    to_unpin.extend([object_key_to_string(k)] * read_entry.refcount)
            if to_unpin:
                self._safe_unpin(self._allocator.handler, to_unpin)
            logger.warning(
                "MaruL1Manager: swept %d orphan write(s) / %d orphan read(s)",
                len(expired_writes),
                len(expired_reads),
            )

    @_maru_l1_synchronized
    def reserve_read(
        self, keys: list[ObjectKey], extra_count: int = 0
    ) -> dict[ObjectKey, L1OperationResult]:
        """Pin keys on MaruServer and stage zero-copy views for reading.

        PARITY(L1Manager.reserve_read): per-key independent results; takes
        ``1 + extra_count`` protection units per key. MARU: protection is the
        cross-node server ``pin_count``; the local refcount balances the pins
        so N finish_read calls release them all.

        Args:
            keys: The list of object keys to reserve read access for.
            extra_count: Extra protection units on top of the default 1.

        Returns:
            A dictionary mapping each key to (L1Error, MemoryObj | None).

        Errors:
            KEY_NOT_EXIST: Absent from the directory, unresolvable, or the
                pin RPC failed.
            KEY_NOT_READABLE: The key is mid-write on this instance.
        """
        total = 1 + _clamp_extra_count(extra_count)
        ret: dict[ObjectKey, L1OperationResult] = {
            k: (L1Error.KEY_NOT_EXIST, None) for k in keys
        }
        for k in keys:
            # PARITY(L1Manager.reserve_read): a mid-write key is not readable
            # (distinct from a plain miss). It is unregistered, so the pin
            # below misses and this pre-marked result stands.
            if k in self._pending_write:
                ret[k] = (L1Error.KEY_NOT_READABLE, None)
        if not keys:
            return ret
        successful_keys: list[ObjectKey] = []
        # MARU: pin + retrieve + resolve + stage (shared with retained promote).
        self._pin_retrieve_stage(keys, total, ret, successful_keys)
        # PARITY(L1Manager.reserve_read): notify listeners of the new read
        # holds (feeds the eviction LRU and the store controller).
        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_read(successful_keys)
        self._publish(EventType.L1_READ_RESERVED, successful_keys)
        return ret

    @_maru_l1_synchronized
    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Return staged read objects without taking new pins.

        Must be called between ``reserve_read`` and ``finish_read``.

        Args:
            keys: The list of object keys to read.

        Returns:
            A dictionary mapping each key to (L1Error, MemoryObj | None).

        Errors:
            KEY_NOT_EXIST: The key has no staged read.
            KEY_NOT_READABLE: The key is mid-write on this instance.
        """
        ret: dict[ObjectKey, L1OperationResult] = {}
        for k in keys:
            entry = self._pending_read.get(k)
            if entry is not None:
                ret[k] = (L1Error.SUCCESS, entry.mem_obj)
            elif k in self._pending_write:
                ret[k] = (L1Error.KEY_NOT_READABLE, None)
            else:
                ret[k] = (L1Error.KEY_NOT_EXIST, None)
        return ret

    @_maru_l1_synchronized
    def finish_read(
        self, keys: list[ObjectKey], extra_count: int = 0
    ) -> dict[ObjectKey, L1Error]:
        """Release the protection taken by ``reserve_read``.

        Releases ``1 + extra_count`` units per key: the local refcount drops
        and the same number of server pins are released; the staged entry is
        dropped at refcount zero.

        Args:
            keys: The list of object keys to finish read access for.
            extra_count: Extra units to release on top of the default 1.

        Returns:
            A dictionary mapping each key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key has no staged read.
        """
        total = 1 + _clamp_extra_count(extra_count)
        ret: dict[ObjectKey, L1Error] = {}
        to_unpin: list[str] = []
        need_to_free: list[MemoryObj] = []
        need_to_free_keys: list[ObjectKey] = []
        successful_keys: list[ObjectKey] = []
        for k in keys:
            entry = self._pending_read.get(k)
            if entry is None:
                logger.warning("MaruL1Manager: finish read on unstaged key %s", k)
                ret[k] = L1Error.KEY_NOT_EXIST
                continue
            # MARU: never release more than we hold (over-release would
            # corrupt the server pin_count).
            released = min(total, entry.refcount)
            if released < total:
                logger.warning(
                    "MaruL1Manager: finish read released %d/%d holds for key %s",
                    released,
                    total,
                    k,
                )
            entry.refcount -= released
            # MARU: temporary reads are local staging (never directory-pinned),
            # so at refcount zero the page is reclaimed, not unpinned. Normal
            # reads release the server pins taken by reserve_read.
            if entry.is_temporary:
                if entry.refcount <= 0:
                    need_to_free.append(entry.mem_obj)
                    need_to_free_keys.append(k)
                    del self._pending_read[k]
            else:
                to_unpin.extend([object_key_to_string(k)] * released)
                if entry.refcount <= 0:
                    del self._pending_read[k]
            ret[k] = L1Error.SUCCESS
            successful_keys.append(k)
        if to_unpin:
            self._safe_unpin(self._allocator.handler, to_unpin)
        for obj in need_to_free:
            # MARU: unregistered local page -> discard through the allocator.
            self._allocator.abort_alloc(obj)
        # PARITY(L1Manager.finish_read): read_finished for every release;
        # deleted_by_manager for temporary pages dropped at refcount zero.
        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(successful_keys)
            listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._publish(EventType.L1_READ_FINISHED, successful_keys)
        self._publish(EventType.L1_KEYS_EVICTED, need_to_free_keys)
        return ret

    @_maru_l1_synchronized
    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"] = "all",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Allocate CXL pages and stage them for writing.

        PARITY(L1Manager.reserve_write): in "new" mode existing keys return
        KEY_NOT_WRITABLE; allocation is all-or-nothing. MARU: "existing"
        covers locally staged keys and directory-registered keys (the latter
        check is the cross-instance dedup -- another instance stored it, so
        the D2H copy is skipped). In-place update of a registered shared page
        is not possible, so only ``mode="new"`` (the only mode MP callers
        use) is supported.

        Args:
            keys: The list of object keys to reserve write access for.
            is_temporary: Per-key flag; temporary objects are dropped after
                their read completes.
            layout_desc: The memory layout for the allocation.
            mode: Reservation mode; must be ``"new"``.

        Returns:
            A dictionary mapping each key to (L1Error, MemoryObj | None).

        Raises:
            ValueError: If ``mode`` is not ``"new"``.

        Errors:
            KEY_NOT_WRITABLE: The key is staged locally or already registered.
            OUT_OF_MEMORY: The CXL pool could not fit the batch.
        """
        if mode != "new":
            raise ValueError(f"MaruL1Manager supports mode='new' only, got {mode!r}")
        ret: dict[ObjectKey, L1OperationResult] = {}
        candidates: list[tuple[ObjectKey, bool]] = []
        for k, is_temp in zip(keys, is_temporary, strict=False):
            if k in self._pending_write or k in self._pending_read:
                ret[k] = (L1Error.KEY_NOT_WRITABLE, None)
            else:
                candidates.append((k, is_temp))
        if not candidates:
            return ret

        try:
            exists = self._allocator.handler.batch_exists(
                [object_key_to_string(k) for k, _ in candidates]
            )
        except Exception:
            # MARU: existence unknown -- proceed; batch_store dup-skips later.
            logger.exception(
                "MaruL1Manager: batch_exists failed for %d keys", len(candidates)
            )
            exists = [False] * len(candidates)
        # Normalize a malformed reply; unknown tails allocate (dup-skip later).
        exists = list(exists[: len(candidates)])
        exists += [False] * (len(candidates) - len(exists))
        need_allocate: list[tuple[ObjectKey, bool]] = []
        for (k, is_temp), ex in zip(candidates, exists, strict=False):
            if ex:
                ret[k] = (L1Error.KEY_NOT_WRITABLE, None)
            else:
                need_allocate.append((k, is_temp))
        if not need_allocate:
            return ret

        objs = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, len(need_allocate)
        )
        if objs is None:
            # PARITY(L1Manager.reserve_write): allocation failure marks the
            # whole batch OUT_OF_MEMORY (batched_allocate is all-or-nothing).
            for k, _ in need_allocate:
                ret[k] = (L1Error.OUT_OF_MEMORY, None)
            return ret
        successful_keys: list[ObjectKey] = []
        deadline = time.monotonic() + self._write_ttl_seconds
        for (k, is_temp), obj in zip(need_allocate, objs, strict=False):
            self._pending_write[k] = _PendingWrite(
                mem_obj=obj, is_temporary=is_temp, deadline=deadline
            )
            ret[k] = (L1Error.SUCCESS, obj)
            successful_keys.append(k)
        # PARITY(L1Manager.reserve_write): notify listeners of the new write
        # holds (the eviction LRU treats them as unevictable).
        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_write(successful_keys)
        self._publish(EventType.L1_WRITE_RESERVED, successful_keys)
        return ret

    @_maru_l1_synchronized
    def finish_write(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Register staged pages in the MaruServer directory.

        Args:
            keys: The list of object keys to finish write access for.

        Returns:
            A dictionary mapping each key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key was never reserved (or already finished).
            KEY_IN_WRONG_STATE: Registration failed.
        """
        ret: dict[ObjectKey, L1Error] = {}
        staged: list[tuple[ObjectKey, _PendingWrite]] = []
        for k in keys:
            entry = self._pending_write.pop(k, None)
            if entry is None:
                logger.warning("MaruL1Manager: finish write on unstaged key %s", k)
                ret[k] = L1Error.KEY_NOT_EXIST
            else:
                staged.append((k, entry))
        registered, errors = self._store_staged(staged)
        ret.update(errors)
        for k in registered:
            ret[k] = L1Error.SUCCESS
        # PARITY(L1Manager.finish_write): notify listeners of registered pages
        # (the store controller stops re-storing them; must NOT be the promote
        # event -- that is on_l1_keys_finish_write_and_reserve_read).
        for listener in self._registered_listeners:
            listener.on_l1_keys_write_finished(registered)
        self._publish(EventType.L1_WRITE_FINISHED, registered)
        return ret

    @_maru_l1_synchronized
    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey], extra_count: int = 0
    ) -> dict[ObjectKey, L1OperationResult]:
        """Finish a write and take read holds in one step (L2->L1 promote).

        Called by the prefetch controller after loading L2 bytes into a
        write-reserved page. Branches on the staged ``is_temporary`` flag
        (Decision A):

        - temporary (the default prefetch policy): the loaded page is private
          staging -- moved straight to read staging without touching the shared
          directory; finish_read reclaims it at refcount zero.
        - retained (``prefetch_policy: retain``): the page is registered in the
          directory (batch_store) and the authoritative page is re-resolved
          with pins, so a dup-skip that auto-freed our page still yields the
          winning shared page.

        Fires ``on_l1_keys_finish_write_and_reserve_read`` -- never
        ``on_l1_keys_write_finished`` (that would make the store controller
        re-store the promoted key to L2).

        Args:
            keys: Keys to transition from write-staged to read-staged.
            extra_count: Extra read holds on top of the default 1 (one per TP
                worker for MLA models with TP > 1).

        Returns:
            A dictionary mapping each key to (L1Error, MemoryObj | None).

        Errors:
            KEY_NOT_EXIST: The key is not write-staged on this instance.
            KEY_IN_WRONG_STATE: The key is already read-staged, or registration
                or re-resolve failed.
        """
        total = 1 + _clamp_extra_count(extra_count)
        ret: dict[ObjectKey, L1OperationResult] = {
            k: (L1Error.KEY_NOT_EXIST, None) for k in keys
        }
        temp_staged: list[tuple[ObjectKey, _PendingWrite]] = []
        retain_staged: list[tuple[ObjectKey, _PendingWrite]] = []
        for k in keys:
            entry = self._pending_write.get(k)
            if entry is None:
                # PARITY(L1Manager): a key not write-held cannot be promoted.
                logger.warning("MaruL1Manager: promote on non-write-staged key %s", k)
                continue
            if k in self._pending_read:
                # PARITY(L1Manager): a key already read-held is in wrong state.
                ret[k] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue
            if entry.is_temporary:
                temp_staged.append((k, entry))
            else:
                retain_staged.append((k, entry))

        successful_keys: list[ObjectKey] = []
        # MARU temporary promote: private staging -- no batch_store, no pin.
        # The loaded local page is authoritative; move it to read staging.
        for k, entry in temp_staged:
            del self._pending_write[k]
            self._pending_read[k] = _PendingRead(
                mem_obj=entry.mem_obj,
                refcount=total,
                is_temporary=True,
                deadline=time.monotonic() + self._read_ttl_seconds,
            )
            ret[k] = (L1Error.SUCCESS, entry.mem_obj)
            successful_keys.append(k)
        # MARU retained promote: register then re-resolve the authoritative page.
        if retain_staged:
            self._promote_retained(retain_staged, total, ret, successful_keys)

        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(successful_keys)
        self._publish(EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, successful_keys)
        return ret

    def _promote_retained(
        self,
        retain_staged: list[tuple[ObjectKey, _PendingWrite]],
        total: int,
        ret: dict[ObjectKey, L1OperationResult],
        successful_keys: list[ObjectKey],
    ) -> None:
        """Register retained-promote pages and stage authoritative reads.

        Pops the write staging, registers via ``_store_staged``, then pins and
        re-resolves the directory page (a dup-skip auto-freed our own page, so
        the pinned+retrieved page is the winning one). Keys that fail to
        register or re-resolve are left at KEY_IN_WRONG_STATE.

        Args:
            retain_staged: (key, pending write) pairs to register and stage.
            total: Read holds (pins) to take per key.
            ret: Result map, mutated in place.
            successful_keys: List extended in place with each staged key.
        """
        for k, _ in retain_staged:
            self._pending_write.pop(k, None)
        registered, errors = self._store_staged(retain_staged)
        for k, err in errors.items():
            ret[k] = (err, None)
        if not registered:
            return
        # A store that cannot be re-resolved to a read view is a wrong-state
        # promote (the page is registered but this instance holds no read).
        for k in registered:
            ret[k] = (L1Error.KEY_IN_WRONG_STATE, None)
        self._pin_retrieve_stage(registered, total, ret, successful_keys)

    @_maru_l1_synchronized
    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Delete keys from the shared directory.

        PARITY(L1Manager.delete): a key held by any reader or writer refuses
        with KEY_IS_LOCKED; the eviction policy keeps it and retries later.

        Args:
            keys: The list of object keys to delete.

        Returns:
            A dictionary mapping each key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key is not in the directory.
            KEY_IS_LOCKED: The key is staged locally, pinned on the server,
                or the delete RPC failed (retryable).
        """
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []
        handler = self._allocator.handler
        for k in keys:
            # MARU: locally staged keys are pinned/write-held by construction.
            if k in self._pending_read or k in self._pending_write:
                ret[k] = L1Error.KEY_IS_LOCKED
                continue
            ks = object_key_to_string(k)
            try:
                if handler.delete(ks):
                    ret[k] = L1Error.SUCCESS
                    successful_keys.append(k)
                    continue
                # MARU: handler.delete conflates pinned and missing; one
                # exists() round-trip disambiguates so pinned keys retry.
                # TODO(maru): a tri-state delete RPC would remove this.
                ret[k] = (
                    L1Error.KEY_IS_LOCKED
                    if handler.exists(ks)
                    else L1Error.KEY_NOT_EXIST
                )
            except Exception:
                logger.exception("MaruL1Manager: delete failed for key %s", ks)
                ret[k] = L1Error.KEY_IS_LOCKED
        # PARITY(L1Manager.delete): notify listeners of the keys actually
        # removed from the directory (the eviction LRU drops them).
        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(successful_keys)
        self._publish(EventType.L1_KEYS_EVICTED, successful_keys)
        return ret

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Mark keys as accessed, feeding the eviction LRU recency.

        PARITY(L1Manager.touch_keys): fires ``on_l1_keys_accessed`` without the
        manager lock (matching stock); recency lives in the eviction policy.

        Args:
            keys: The list of object keys touched.
        """
        for listener in self._registered_listeners:
            listener.on_l1_keys_accessed(keys)

    @_maru_l1_synchronized
    def clear(self, force: bool = False) -> None:
        """Release this instance's staging (unpin reads, reclaim write pages).

        PARITY(L1Manager.clear): ``force=False`` keeps locked entries -- all
        maru staging is locked by construction, so it only logs. MARU: shared
        directory data is never deleted (other instances may hold it).

        Args:
            force: If True, drain in-flight staging too (unsafe, like stock).
        """
        if not force:
            if self._pending_read or self._pending_write:
                logger.info(
                    "MaruL1Manager: clear kept %d staged reads / %d staged writes",
                    len(self._pending_read),
                    len(self._pending_write),
                )
            return
        dropped = self._drain_staging(force=True)
        # PARITY(L1Manager.clear): a force-clear notifies listeners of the drops.
        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(dropped)
        self._publish(EventType.L1_KEYS_EVICTED, dropped)

    def _drain_staging(self, force: bool) -> list[ObjectKey]:
        """Unpin staged reads (refcount times) and abort staged writes.

        Args:
            force: If True, log the dropped in-flight staging as a warning.

        Returns:
            The keys dropped from staging (staged reads then staged writes).
        """
        to_unpin: list[str] = []
        for k, entry in self._pending_read.items():
            # MARU: temporary reads hold no server pin -- reclaim the page.
            if entry.is_temporary:
                self._allocator.abort_alloc(entry.mem_obj)
            else:
                to_unpin.extend([object_key_to_string(k)] * entry.refcount)
        if force and (self._pending_read or self._pending_write):
            logger.warning(
                "MaruL1Manager: force-clear drops %d staged reads "
                "(%d pins) and %d staged writes",
                len(self._pending_read),
                len(to_unpin),
                len(self._pending_write),
            )
        if to_unpin:
            self._safe_unpin(self._allocator.handler, to_unpin)
        for write_entry in self._pending_write.values():
            self._allocator.abort_alloc(write_entry.mem_obj)
        dropped = list(self._pending_read.keys()) + list(self._pending_write.keys())
        self._pending_read.clear()
        self._pending_write.clear()
        return dropped

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Return whether ``key`` has no local hold (lock-free view).

        PARITY(L1Manager.is_key_evictable): deliberately lock-free; delete()
        re-checks authoritatively (the server refuses pinned keys). MARU:
        directory existence is not checked (remote); delete() drops stale
        candidates with KEY_NOT_EXIST.

        Args:
            key: The object key to check.

        Returns:
            False if the key is staged for read or write locally.
        """
        return key not in self._pending_read and key not in self._pending_write

    def get_memory_usage(self) -> tuple[int, int]:
        """Return (used, total) bytes of this instance's owned CXL regions.

        MARU: usage is the owned-region allocation pressure -- the basis the
        stock eviction watermark expects. Before the pool is up, reports the
        configured capacity so watermark math stays sane.

        Returns:
            A tuple of (used_bytes, total_bytes).
        """
        if not self._allocator.is_initialized:
            return 0, self._config.pool_size_bytes
        try:
            handler = self._allocator.handler
            regions = handler.get_stats().get("store_regions")
            if not regions:
                return 0, self._config.pool_size_bytes
            used = regions["total_allocated_pages"] * handler.get_chunk_size()
            return used, regions["total_pool_size"]
        except Exception:
            logger.exception("MaruL1Manager: get_stats failed")
            return 0, self._config.pool_size_bytes

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return None: the shared pool has no single registerable region."""
        return None

    def close(self) -> None:
        """Stop the sweeper, release staged state, and tear down the allocator."""
        self._stop_event.set()
        self._sweeper.join(timeout=self._sweep_interval + 5.0)
        with self._lock:
            self._drain_staging(force=False)
        # PARITY(L1Manager.close): backing teardown outside the lock.
        self._allocator.close()

    @_maru_l1_synchronized
    def report_status(self) -> dict[str, Any]:
        """Return a status dict describing the maru L1 state.

        Returns:
            A dict with the stock L1 status keys plus ``backend``.
        """
        used, total = self.get_memory_usage()
        return {
            "is_healthy": self._allocator.memcheck(),
            "backend": "maru",
            "total_object_count": len(self._pending_read) + len(self._pending_write),
            "write_locked_count": len(self._pending_write),
            "read_locked_count": len(self._pending_read),
            "temporary_count": sum(
                1 for e in self._pending_read.values() if e.is_temporary
            )
            + sum(1 for e in self._pending_write.values() if e.is_temporary),
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._write_ttl_seconds,
            "read_ttl_seconds": self._read_ttl_seconds,
        }

    def get_object_state(self, key: ObjectKey) -> L1ObjectState | None:
        """Return None: membership lives in the MaruServer directory."""
        return None

    def memcheck(self) -> bool:
        """Delegate to the allocator's consistency check."""
        return self._allocator.memcheck()

    @_maru_l1_synchronized
    def register_kv_layout(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
        chunk_size_in_tokens: int,
    ) -> None:
        """Bind the KV layout, bringing up the CXL pool (idempotent per layout).

        Args:
            shapes: Per-group tensor shapes of one chunk.
            dtypes: Per-group dtypes of one chunk.
            fmt: The KV memory format.
            chunk_size_in_tokens: Tokens per chunk.
        """
        self._allocator.init_layout(shapes, dtypes, fmt, chunk_size_in_tokens)
