# SPDX-License-Identifier: Apache-2.0
"""Shadow L1 manager: indexes borrowed views whose bytes another party owns."""

# Standard
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.lmcache_native import TTLLock
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ObjectMeta, L1OperationResult
from lmcache.v1.distributed.l1_manager.base import (
    BaseL1Manager,
    L1ObjectState,
    l1_mgr_synchronized,
    validate_read_locks,
)
from lmcache.v1.distributed.memory_manager import (
    BorrowedMemoryObj,
    ShadowL1MemoryManager,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig

# The L1 medium a borrowed segment is reported as; other segments cannot be
# borrowed.
_BACKEND_BY_SEGMENT: dict[MemorySegment, L1BackendType] = {
    MemorySegment.DRAM: L1BackendType.DRAM,
    MemorySegment.CXL: L1BackendType.DEVDAX,
    MemorySegment.GDS: L1BackendType.GDS,
}


class ShadowL1Manager(BaseL1Manager):
    """L1 manager whose objects are borrowed views: metadata only, no payload.

    An L2 adapter that lends views (e.g. of a peer's memory) produces
    :class:`BorrowedMemoryObj` objects; this manager indexes them by key
    so the ordinary read path (read locks, ``finish_read``, events) serves
    them. Freeing an object ends its lease. The manager has no capacity: write
    placement targets owned L1s, and the peer that owns the bytes evicts them.

    A view is admitted for one request's readers. Every request looks the
    lender up, so a key already resident adopts the newer lease while its
    holders keep reading, and the older reservation goes back. A view whose
    lease ended answers ``KEY_NOT_READABLE`` to ``unsafe_read``, the one way
    a shadow is read (prefetch reserves from owned L1s only); a holder keeps
    it until its last ``finish_read``, so a read must finish within the
    owner's TTL. An entry whose readers' TTL lapsed is swept at the next
    ``admit``. No L1 event is published for a shadow.
    """

    def __init__(
        self,
        config: L1ManagerConfig,
        segment: MemorySegment,
        backend_type: L1BackendType,
    ) -> None:
        """Create a shadow L1 for views of one memory segment.

        Args:
            config: Lifetime and reporting settings for the shadow L1.
            segment: The segment the borrowed bytes live in, reported as
                REMOTE.
            backend_type: The L1 medium the views are reported as.
        """
        self._backend_type = backend_type
        self._memory_tags = MemoryTags(
            segment=segment, residency=MemoryResidency.REMOTE
        )
        super().__init__(config, ShadowL1MemoryManager())
        # L1 events announce this node's residency per backend, and a
        # shadow's keys are the peer's residency: an eviction event from a
        # DRAM shadow would be recorded against this node's own DRAM copy.
        self._event_bus = EventBus(EventBusConfig(enabled=False))

    @classmethod
    def for_lender(
        cls, adapter_id: int, segment: MemorySegment, defaults: L1ManagerConfig
    ) -> "ShadowL1Manager":
        """Create the shadow L1 a lending L2 adapter brings with it.

        The adapter maps the peers' memory itself, so the shadow needs no size
        or device: it takes the lock TTLs from ``defaults`` and is tagged
        ``shadow.<segment>.<adapter_id>``.

        Args:
            adapter_id: Stable id of the lending adapter; makes the tag unique.
            segment: The segment the adapter borrows (its wanted L1 tags).
            defaults: The L1 settings the shadow inherits TTLs from.

        Returns:
            A new, open shadow L1 manager.

        Raises:
            ValueError: If no L1 medium backs ``segment``.
        """
        backend_type = _BACKEND_BY_SEGMENT.get(segment)
        if backend_type is None:
            raise ValueError(f"No L1 medium backs borrowed {segment.value} views")
        config = replace(
            defaults,
            tag=f"shadow.{segment.value}.{adapter_id}",
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=0, use_lazy=False, shm_name=""
            ),
            gds_l1_config=None,
            eviction=None,
        )
        return cls(config, segment, backend_type)

    @property
    def backend_type(self) -> L1BackendType:
        """Return the L1 medium the borrowed views are reported as."""
        return self._backend_type

    def get_memory_tags(self) -> MemoryTags:
        """Return the borrowed segment with REMOTE residency."""
        return self._memory_tags

    @l1_mgr_synchronized
    def admit(
        self,
        objects: Mapping[ObjectKey, BorrowedMemoryObj],
        read_locks: int = 1,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Make borrowed views resident and read-lock them, without allocating.

        The views' ownership passes to this manager: every view not kept
        resident is released before returning. A key already resident keeps
        its view, which adopts the given view's lease when both are views of
        the same bytes; the given view is released under the older lease,
        returning the older reservation. Entries no reader holds any more
        (their readers' TTL lapsed) are swept first. No L1 event is
        published: borrowed views are the peer's residency, not this node's.

        Args:
            objects: Borrowed view per key.
            read_locks: Read locks to take per admitted key, one per reader.

        Returns:
            Per key: ``(SUCCESS, obj)`` with the resident object now
            read-locked; ``(KEY_IN_WRONG_STATE, None)`` when the given view's
            lease ended before admission; ``(KEY_NOT_READABLE, None)`` when a
            resident view of other bytes is held, so the key is a miss for
            this request.

        Raises:
            ValueError: If ``read_locks`` is out of range.
        """
        total = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {}
        resident_keys: list[ObjectKey] = []
        discarded: list[MemoryObj] = []
        self._sweep_lapsed_locked()
        for key, obj in objects.items():
            if not obj.is_valid():
                discarded.append(obj)
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue
            entry = self._objects.get(key)
            if entry is not None:
                resident = cast(BorrowedMemoryObj, entry.memory_obj)
                if not resident.same_bytes(obj):
                    # The peer moved the key since the holders' lease: they
                    # keep their bytes, this request misses.
                    discarded.append(obj)
                    ret[key] = (L1Error.KEY_NOT_READABLE, None)
                    continue
                # Held by earlier readers: the fresh lease renews it, and the
                # older reservation goes back with the discarded view.
                older = resident.lease
                resident.replace_lease(*obj.lease)
                obj.replace_lease(*older)
                discarded.append(obj)
                resident_keys.append(key)
            else:
                entry = L1ObjectState(
                    memory_obj=obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=True,
                )
                obj.set_l1_manager(self.l1_manager_id)
                self._objects[key] = entry
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        self._memory_manager.free(discarded)
        if resident_keys:
            self._report_read_reserved(resident_keys)
        return ret

    @l1_mgr_synchronized
    def sweep_lapsed(self) -> None:
        """Release every view no reader holds any more.

        A temporary view leaves at its last ``finish_read``, so an unheld
        entry is one whose readers' TTL lapsed; its late ``finish_read`` is
        refused. ``admit`` sweeps too; a draining lender calls this so its
        borrow count can reach zero without a new admission.
        """
        self._sweep_lapsed_locked()

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Read held views; one whose lease ended is ``KEY_NOT_READABLE``."""
        ret = super().unsafe_read(keys)
        for key, (err, obj) in ret.items():
            if err is L1Error.SUCCESS and not cast(MemoryObj, obj).is_valid():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
        return ret

    def close(self) -> None:
        """Release every view, read-locked or not; idempotent.

        Call after stopping new controller and GPU submissions; outstanding
        work is synchronized first so no copy reads a released view: on the
        devices the views live on, or on every device when a host view may
        have been read by any GPU.
        """
        with self._lock:
            devices = {
                cast(BorrowedMemoryObj, entry.memory_obj).local_device()
                for entry in self._objects.values()
            }
        if torch_dev.is_available():
            if any(device is None or device.type == "cpu" for device in devices):
                indices = list(range(torch_dev.device_count()))
            else:
                indices = [cast(torch.device, device).index for device in devices]
            for index in indices:
                torch_dev.synchronize(index)
        super().close()

    def _object_meta(self, memory_obj: MemoryObj) -> L1ObjectMeta:
        """Report borrowed views as holding no local bytes."""
        return L1ObjectMeta(size_bytes=0, backend=self.backend_type)

    def _sweep_lapsed_locked(self) -> None:
        lapsed = [
            key
            for key, entry in self._objects.items()
            if not entry.read_lock.is_locked()
        ]
        self._memory_manager.free([self._objects.pop(key).memory_obj for key in lapsed])
