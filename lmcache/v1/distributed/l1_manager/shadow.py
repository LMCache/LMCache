# SPDX-License-Identifier: Apache-2.0
"""Shadow L1 manager: indexes borrowed views whose bytes another party owns."""

# Standard
from collections.abc import Mapping
from dataclasses import replace

# First Party
from lmcache import torch_dev
from lmcache.lmcache_native import TTLLock
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryResidency,
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
from lmcache.v1.distributed.memory_manager import ShadowL1MemoryManager
from lmcache.v1.memory_management import BorrowedTensorMemoryObj, MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType


class ShadowL1Manager(BaseL1Manager):
    """L1 manager whose objects are borrowed views: metadata only, no payload.

    An L2 adapter that lends views (e.g. of a peer's memory) produces
    :class:`BorrowedTensorMemoryObj` objects; this manager indexes them by key
    so the ordinary read path (read locks, ``finish_read``, eviction, events)
    serves them. Freeing an object ends its lease instead of returning memory,
    and the manager has no capacity, so write placement never targets it.

    A view whose lease ended is never handed to a new reader. A reader that
    already holds it keeps it until its last ``finish_read`` (views are
    temporary), so a copy in flight is never torn down.
    """

    def __init__(self, config: L1ManagerConfig, local: BaseL1Manager) -> None:
        """Create the shadow L1 that borrows peers' copies of ``local``'s medium.

        Args:
            config: Lifetime and reporting settings for the shadow L1.
            local: The local L1 whose medium the borrowed bytes share; this
                L1 reports the same backend and memory segment, as REMOTE.
        """
        self._backend_type = local.backend_type
        self._memory_tags = MemoryTags(
            segment=local.get_memory_tags().segment,
            residency=MemoryResidency.REMOTE,
        )
        super().__init__(config, ShadowL1MemoryManager())

    @classmethod
    def next_to(cls, local: BaseL1Manager) -> "ShadowL1Manager":
        """Create the shadow L1 paired with ``local``.

        It inherits ``local``'s lock TTLs and eviction settings, is tagged
        ``<local tag>.shadow``, and has no memory of its own.

        Args:
            local: The local L1 whose peers' copies are borrowed.

        Returns:
            A new, open shadow L1 manager.
        """
        config = replace(
            local.config,
            tag=f"{local.config.tag}.shadow",
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=0, use_lazy=False, shm_name=""
            ),
            gds_l1_config=None,
        )
        return cls(config, local)

    @property
    def backend_type(self) -> L1BackendType:
        """Return the paired local L1's backend."""
        return self._backend_type

    def get_memory_tags(self) -> MemoryTags:
        """Return the paired local L1's segment with REMOTE residency."""
        return self._memory_tags

    @l1_mgr_synchronized
    def admit(
        self,
        objects: Mapping[ObjectKey, BorrowedTensorMemoryObj],
        read_locks: int = 1,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Make borrowed views resident and read-lock them, without allocating.

        The view's ownership passes to this manager: every view not kept
        resident is released before returning.

        Args:
            objects: Borrowed view per key.
            read_locks: Read locks to take per admitted key, one per reader.

        Returns:
            Per key: ``(SUCCESS, obj)`` with the resident object now
            read-locked -- the given view, or an already-resident one, in
            which case the given view is released; ``(KEY_NOT_READABLE,
            None)`` when the resident view's lease ended while readers still
            hold it; ``(KEY_IN_WRONG_STATE, None)`` when the given view's
            lease ended before admission.

        Raises:
            ValueError: If ``read_locks`` is out of range.
        """
        total = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {}
        admitted_keys: list[ObjectKey] = []
        admitted_meta: list[L1ObjectMeta] = []
        resident_keys: list[ObjectKey] = []
        discarded: list[MemoryObj] = []
        for key, obj in objects.items():
            entry = self._objects.get(key)
            if entry is not None:
                discarded.append(obj)
                if not entry.memory_obj.is_valid():
                    ret[key] = (L1Error.KEY_NOT_READABLE, None)
                    continue
                resident_keys.append(key)
            elif not obj.is_valid():
                discarded.append(obj)
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue
            else:
                entry = L1ObjectState(
                    memory_obj=obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=True,
                )
                obj.set_l1_manager(self.l1_manager_id)
                self._objects[key] = entry
                admitted_keys.append(key)
                admitted_meta.append(self._object_meta(obj))
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        self._memory_manager.free(discarded)
        if resident_keys:
            self._report_read_reserved(resident_keys)
        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(admitted_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                metadata={
                    "l1_tag": self._config.tag,
                    "keys": admitted_keys,
                    "meta": admitted_meta,
                },
            )
        )
        return ret

    @l1_mgr_synchronized
    def reap_expired(self) -> None:
        """Release views whose lease ended and that no reader holds.

        A held, expired view stays (unreadable) until its last
        ``finish_read`` frees it, or its local read lock's TTL lapses.
        """
        expired = [
            key
            for key, entry in self._objects.items()
            if not entry.read_lock.is_locked() and not entry.memory_obj.is_valid()
        ]
        if expired:
            objs = [self._objects.pop(key).memory_obj for key in expired]
            self._free_and_report_deleted(expired, objs)

    def release_all(self) -> None:
        """Release every view, read-locked or not, before lenders shut down.

        Call after stopping new controller and GPU submissions; outstanding
        device work is synchronized first so no copy reads a released view.
        """
        if torch_dev.is_available():
            torch_dev.synchronize()
        self.clear(force=True)

    def _object_meta(self, memory_obj: MemoryObj) -> L1ObjectMeta:
        """Report borrowed views as holding no local bytes."""
        return L1ObjectMeta(size_bytes=0, backend=self.backend_type)
