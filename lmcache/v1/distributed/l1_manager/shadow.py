# SPDX-License-Identifier: Apache-2.0
"""Shadow L1 manager: indexes borrowed objects whose bytes another party owns."""

# Standard
from dataclasses import replace
from typing import cast

# Third Party
import torch

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
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ObjectMeta, L1OperationResult
from lmcache.v1.distributed.l1_manager.base import (
    BaseL1Manager,
    L1ObjectState,
    l1_mgr_synchronized,
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
    """L1 manager whose objects are borrowed: metadata and a lease, no payload.

    It implements the L1 primitives so that a lending L2 adapter goes through
    the prefetch controller's ordinary load pipeline: ``reserve_write`` stages
    a :class:`BorrowedMemoryObj` placeholder per key without any capacity
    check, the lender's load binds the placeholder to the lease it holds and
    to the bytes, and ``finish_write_and_reserve_read`` makes it resident and
    read-locked like any loaded object. Freeing an object ends its lease. The
    manager has no capacity: write placement targets owned L1s, and the peer
    that owns the bytes evicts them.

    Every request looks the lender up: a shadow is never a lookup hit source
    (``reserve_read`` misses), and a key another request still reads is
    staged again; at admission the resident view adopts the fresh lease when
    both are views of the same bytes, and the older reservation goes back
    with the discarded placeholder. Every view is temporary: it leaves at its
    last ``finish_read``. A view whose lease ended answers
    ``KEY_NOT_READABLE`` to ``unsafe_read``, the one way a shadow is read; a
    holder keeps it until its last ``finish_read``, so a read must finish
    within the owner's TTL. An entry whose readers' TTL lapsed is swept at
    the next ``reserve_write`` or :meth:`sweep_lapsed`. No L1 event is
    published for a shadow.
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
            segment: The segment the borrowed bytes live in, reported with
                ``SHADOW`` residency.
            backend_type: The L1 medium the views are reported as.
        """
        self._backend_type = backend_type
        self._memory_tags = MemoryTags(
            segment=segment, residency=MemoryResidency.SHADOW
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
        """Return the borrowed segment with SHADOW residency."""
        return self._memory_tags

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1OperationResult]:
        """Miss every key: a shadow is never a lookup hit source.

        Its views belong to the requests that borrowed them; a new request
        borrows afresh through the lender, which renews the owner's lock.

        Args:
            keys: Keys looked up.
            read_locks: Ignored.

        Returns:
            ``(KEY_NOT_EXIST, None)`` for every key.
        """
        return {key: (L1Error.KEY_NOT_EXIST, None) for key in keys}

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Stage an unbound placeholder per key, with no capacity check.

        Every placeholder is temporary, whatever ``is_temporary`` says: a
        view cannot outlive its lease. A key another request still reads is
        staged as well, so that this request's lease can renew the resident
        view at admission. Entries no reader holds any more are swept first.

        Args:
            keys: Keys the lender's load will bind.
            is_temporary: Ignored; see above.
            layout_desc: Layout of the objects.
            tag: The writer's identity, passed to the ``finish_write``
                variant that completes the write.

        Returns:
            Per key: ``(SUCCESS, placeholder)``, or ``(KEY_NOT_WRITABLE,
            None)`` when ``tag`` already stages the key.

        Raises:
            ValueError: If ``keys`` and ``is_temporary`` differ in length.
        """
        if len(keys) != len(is_temporary):
            raise ValueError(
                f"L1Manager.reserve_write: {len(keys)} keys but "
                f"{len(is_temporary)} is_temporary flags"
            )
        with self._lock:
            self._sweep_lapsed_locked()
            resident = [key for key in keys if key in self._objects]
        resident_set = set(resident)
        fresh = [key for key in keys if key not in resident_set]
        ret: dict[ObjectKey, L1OperationResult] = {}
        if fresh:
            ret.update(
                super().reserve_write(fresh, [True] * len(fresh), layout_desc, tag)
            )
        if resident:
            ret.update(self._stage_beside_resident(resident, layout_desc, tag))
        return ret

    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Make bound placeholders resident and read-locked.

        A placeholder the load left unbound, or whose lease already ended, is
        discarded and reported ``KEY_IN_WRONG_STATE``. A key already resident
        keeps its view and read-locks it for this request; the view adopts
        the placeholder's lease when both are views of the same bytes, and
        the older reservation goes back with the discarded placeholder.
        Otherwise the resident view serves this request under its own lease
        and the placeholder's reservation goes back.

        Args:
            keys: Keys ``tag`` staged whose load succeeded.
            read_locks: Read locks to take per key, one per reader.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key: ``(SUCCESS, obj)`` with the object readers see, now
            read-locked; ``(KEY_IN_WRONG_STATE, None)`` for a placeholder
            with no live lease; ``(KEY_NOT_EXIST, None)`` when ``tag`` stages
            nothing for the key.
        """
        lapsed: list[ObjectKey] = []
        with self._lock:
            for key in keys:
                staged = self._get_staging(key, tag)
                if staged is None:
                    continue
                obj = cast(BorrowedMemoryObj, staged.memory_obj)
                if not obj.is_valid():
                    lapsed.append(key)
                    continue
                entry = self._objects.get(key)
                if entry is None:
                    continue
                resident = cast(BorrowedMemoryObj, entry.memory_obj)
                if resident.same_bytes(obj):
                    older = resident.lease
                    resident.replace_lease(obj.lease)
                    obj.replace_lease(older)
        ret: dict[ObjectKey, L1OperationResult] = {}
        if lapsed:
            for key, err in self.finish_write_and_delete(lapsed, tag=tag).items():
                ret[key] = (
                    L1Error.KEY_IN_WRONG_STATE if err is L1Error.SUCCESS else err,
                    None,
                )
        rest = [key for key in keys if key not in ret]
        if rest:
            ret.update(super().finish_write_and_reserve_read(rest, read_locks, tag))
        return ret

    @l1_mgr_synchronized
    def sweep_lapsed(self) -> None:
        """Release every view no reader holds any more.

        A temporary view leaves at its last ``finish_read``, so an unheld
        entry is one whose readers' TTL lapsed; its late ``finish_read`` is
        refused. ``reserve_write`` sweeps too; a draining lender's removal
        calls this so its borrow count can reach zero without a new request.
        """
        self._sweep_lapsed_locked()

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

    @l1_mgr_synchronized
    def _stage_beside_resident(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc, tag: str
    ) -> dict[ObjectKey, L1OperationResult]:
        """Stage a placeholder for keys the base refuses as already resident."""
        ret: dict[ObjectKey, L1OperationResult] = {}
        for key in keys:
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

    def _sweep_lapsed_locked(self) -> None:
        lapsed = [
            key
            for key, entry in self._objects.items()
            if not entry.read_lock.is_locked()
        ]
        self._memory_manager.free([self._objects.pop(key).memory_obj for key in lapsed])
