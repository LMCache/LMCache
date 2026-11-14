# SPDX-License-Identifier: Apache-2.0
"""Shadow L1 manager: indexes borrowed objects whose bytes another party owns."""

# Standard
from typing import cast
import threading

# First Party
from lmcache import torch_dev
from lmcache.lmcache_native import TTLLock
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import ShadowL1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L1ObjectMeta,
    L1OperationResult,
)
from lmcache.v1.distributed.l1_manager.base import (
    BaseL1Manager,
    L1ObjectState,
    l1_mgr_synchronized,
)
from lmcache.v1.distributed.memory_manager import ShadowL1MemoryManager
from lmcache.v1.memory_management import BorrowedMemoryObj, MemoryObj
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig

# The L1 medium a borrowed segment is reported as; other segments cannot be
# borrowed.
_BACKEND_BY_SEGMENT: dict[MemorySegment, L1BackendType] = {
    MemorySegment.DRAM: L1BackendType.DRAM,
    MemorySegment.CXL: L1BackendType.DEVDAX,
    MemorySegment.GDS: L1BackendType.GDS,
}


class ShadowL1Manager(BaseL1Manager):
    """L1 manager for borrowed objects: a peer owns the bytes, we hold a lease.

    Configured like any L1 (``--l1-manager '{"type": "SHADOW", ...}'``), it
    is where borrowing L2 adapters of its segment land their objects. It
    speaks the ordinary L1 primitives, so the prefetch controller needs no
    special path:

    - ``reserve_write`` stages a :class:`BorrowedMemoryObj` placeholder per
      key, with no capacity check; the borrower's load binds it to a lease.
    - ``finish_write_and_reserve_read`` admits it, resident and read-locked.
      A key already resident adopts the fresh lease of the same bytes.
    - ``reserve_read`` always misses: every request borrows afresh, which
      renews the owner's lock.
    - ``unsafe_read`` answers ``KEY_NOT_READABLE`` once the lease ended, so a
      read must finish within the owner's TTL.
    - Freeing an object ends its lease; the owner evicts the bytes. Every
      view is temporary, and views whose readers' TTL lapsed are swept at the
      next ``reserve_write``.

    It owns no memory, publishes no L1 event and calls no listener, so it
    costs nothing while idle and its store and eviction controllers are inert.
    """

    def __init__(self, config: ShadowL1ManagerConfig) -> None:
        """Create a shadow L1 for views of ``config.segment``, tagged ``SHADOW``
        and reported as the L1 medium that backs the segment.

        Raises:
            ValueError: If no L1 medium backs the segment.
        """
        backend_type = _BACKEND_BY_SEGMENT.get(config.segment)
        if backend_type is None:
            raise ValueError(
                f"No L1 medium backs borrowed {config.segment.value} views"
            )
        self._backend_type = backend_type
        self._memory_tags = MemoryTags(
            segment=config.segment, residency=MemoryResidency.SHADOW
        )
        super().__init__(config, ShadowL1MemoryManager())
        # The overrides below extend a base primitive under one critical
        # section, so the base's lock must let the same thread re-enter it.
        self._lock = threading.RLock()  # type: ignore[assignment]
        # L1 events announce this node's residency per backend, and a
        # shadow's keys are the peer's residency: an eviction event from a
        # DRAM shadow would be recorded against this node's own DRAM copy.
        self._event_bus = EventBus(EventBusConfig(enabled=False))

    @property
    def backend_type(self) -> L1BackendType:
        """Return the L1 medium the borrowed views are reported as."""
        return self._backend_type

    def get_memory_tags(self) -> MemoryTags:
        """Return the borrowed segment with SHADOW residency."""
        return self._memory_tags

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Ignore ``listener``: views are nothing to store and nothing to
        evict, so the controllers of a shadow hear no key event."""

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1OperationResult]:
        """Miss every key: a shadow is never a lookup hit source. Its views
        belong to the requests that borrowed them; a new request borrows
        afresh through the borrower, which renews the owner's lock."""
        return {key: (L1Error.KEY_NOT_EXIST, None) for key in keys}

    @l1_mgr_synchronized
    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
        allow_resident: bool = False,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Stage an unbound placeholder per key, with no capacity check.

        Every placeholder is temporary, whatever ``is_temporary`` says: a
        view cannot outlive its lease. A key another request still reads is
        staged as well, so this request's lease can renew the resident view
        at admission. Entries no reader holds any more, and placeholders
        whose write reservation expired, are swept first. ``allow_resident``
        changes nothing here: resident keys are always staged.

        Returns:
            Per key: ``(SUCCESS, placeholder)``, or ``(KEY_NOT_WRITABLE,
            None)`` when ``tag`` already stages the key.
        """
        self._sweep_lapsed_locked()
        resident = [key for key in keys if key in self._objects]
        resident_set = set(resident)
        fresh = [key for key in keys if key not in resident_set]
        ret: dict[ObjectKey, L1OperationResult] = {}
        if fresh:
            ret.update(
                super().reserve_write(fresh, [True] * len(fresh), layout_desc, tag)
            )
        for key in resident:
            if self._get_staging(key, tag) is not None:
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue
            _, (obj,) = self._memory_manager.allocate(layout_desc, 1)
            obj.set_l1_manager(self._l1_manager_id)
            entry = L1ObjectState(
                memory_obj=obj,
                write_lock=TTLLock(self._write_ttl_seconds),
                read_lock=TTLLock(self._read_ttl_seconds),
                is_temporary=True,
            )
            entry.write_lock.lock()
            self._put_staging(key, tag, entry)
            ret[key] = (L1Error.SUCCESS, obj)
        return ret

    @l1_mgr_synchronized
    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Make bound placeholders resident and read-locked.

        A key already resident keeps its view, read-locked for this request
        too; the view adopts the placeholder's lease when both are views of
        the same bytes, and the other lease goes back with the discarded
        placeholder. A lease that ends from here on makes the view
        ``KEY_NOT_READABLE`` to ``unsafe_read``.
        """
        for key in keys:
            staged = self._get_staging(key, tag)
            entry = self._objects.get(key)
            if staged is None or entry is None:
                continue
            obj = cast(BorrowedMemoryObj, staged.memory_obj)
            resident = cast(BorrowedMemoryObj, entry.memory_obj)
            if obj.is_bound and resident.same_bytes(obj):
                older = resident.lease
                resident.replace_lease(obj.lease)
                obj.replace_lease(older)
        return super().finish_write_and_reserve_read(keys, read_locks, tag)

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Read held views; one whose lease ended is ``KEY_NOT_READABLE``."""
        ret = super().unsafe_read(keys)
        for key, (err, obj) in ret.items():
            if err is L1Error.SUCCESS and not cast(MemoryObj, obj).is_valid():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
        return ret

    def get_staging_memory_usage(self) -> int:
        """Return 0: placeholders hold no local bytes."""
        return 0

    def close(self) -> None:
        """Release every view, read-locked or not; idempotent. Outstanding
        GPU work is synchronized first so no read lands from a released
        view; a borrowed object is read into whichever device its reader
        chose, so every device is fenced."""
        with self._lock:
            has_views = bool(self._objects)
        if has_views and torch_dev.is_available():
            for index in range(torch_dev.device_count()):
                torch_dev.synchronize(index)
        super().close()

    def _object_meta(self, memory_obj: MemoryObj) -> L1ObjectMeta:
        """Report borrowed views as holding no local bytes."""
        return L1ObjectMeta(size_bytes=0, backend=self.backend_type)

    def _sweep_lapsed_locked(self) -> None:
        """Free the views no reader holds and the placeholders whose write
        reservation expired: their readers' or writer's TTL lapsed, so nothing
        will release them, and a shadow has no eviction to reclaim them."""
        lapsed = [
            key
            for key, entry in self._objects.items()
            if not entry.read_lock.is_locked()
        ]
        self._memory_manager.free([self._objects.pop(key).memory_obj for key in lapsed])
        for key in list(self._staging):
            self._reclaim_staging(key, force=False)
