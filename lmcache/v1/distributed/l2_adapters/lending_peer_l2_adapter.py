# SPDX-License-Identifier: Apache-2.0
"""Peer adapters that lend a peer's bytes instead of copying them.

A copying peer adapter (:class:`~.p2p_l2_adapter.P2PL2Adapter`) moves a
looked-up object into this host's L1 and then releases the owner's read lock.
A lending adapter keeps the owner's read lock for as long as readers here use
the object: the lookup reservation becomes the lease of a
:class:`~lmcache.v1.memory_management.BorrowedMemoryObj` that the shadow L1
indexes, and the unlock is sent when the last reader frees it.

:class:`LendingPeerL2AdapterBase` owns what every lender needs: the owner's
TTL bounded from this side, task-scoped reservations, the unlock worker that
batches releases, finalizers for views dropped without ``release()``, and the
drain on teardown. A subclass decides what a borrowed object is
(:meth:`LendingPeerL2AdapterBase._make_borrowed_object`) and when a lookup hit
is usable at all (:meth:`LendingPeerL2AdapterBase._accept_address`).
"""

# Standard
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, cast
import math
import queue
import threading
import time
import weakref

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    PeerL2AdapterBase,
    PeerL2AdapterConfigBase,
)
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
)

logger = init_logger(__name__)

# Stop trusting a borrowed object this long before the owner's read lock can
# expire: a GPU read that passed the last validity check must finish before
# the owner may reuse the bytes. Capped at a tenth of the TTL so short TTLs
# still leave a usable window.
_OWNER_TTL_MARGIN_S = 1.0


def layout_byte_size(layout: MemoryLayoutDesc) -> int:
    """Return the byte size of one object with ``layout``."""
    return sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
    )


@dataclass
class _PendingUnlock:
    """Owner reservations waiting for the unlock worker.

    ``view`` is the finalizer of the borrowed object that held them, or None
    for reservations returned by ``release_lookup`` without an object.
    """

    reservations: dict[ObjectKey, float]
    view: "weakref.finalize | None" = None


class LendingPeerL2AdapterBase(PeerL2AdapterBase):
    """Turn one peer's lookup reservations into leased, borrowed objects.

    Args:
        config: Peer RPC endpoint and lookup deadline.

    Borrow state is scoped by lookup task ID. Releases only enqueue work; a
    worker sends the existing unlock RPC outside L1 locks, batching
    everything queued since its last RPC so a slow or dead owner costs one
    timeout per batch, not per object.

    Thread model: lookups, borrows and ``release_lookup`` run on the prefetch
    thread; object finalizers run on whichever thread drops the last
    reference, including inside garbage collection, so they take no lock and
    only put on a reentrant ``queue.SimpleQueue``.
    """

    def __init__(self, config: PeerL2AdapterConfigBase) -> None:
        # Lending state is set up before the base constructor runs the
        # region hook, per the PeerL2AdapterBase constructor contract.
        self._borrowed_lookups: dict[
            L2TaskId, dict[ObjectKey, tuple[TransferChannelAddress, float]]
        ] = {}
        # Guards _views and _pending_lookup_unlocks. Never taken by finalizers.
        self._borrow_lock = threading.Lock()
        # Finalizers of objects whose unlock the worker has not yet sent.
        self._views: set[weakref.finalize] = set()
        self._pending_lookup_unlocks = 0
        self._unlocks: queue.SimpleQueue[_PendingUnlock | None] = queue.SimpleQueue()
        self._unlock_worker = threading.Thread(
            target=self._run_unlocks, name="peer-unlock", daemon=True
        )
        self._unlock_worker.start()
        try:
            super().__init__(config)
        except BaseException:
            # The base released the RPC resources; stop the worker too.
            self._unlocks.put(None)
            self._unlock_worker.join()
            raise

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        """Submit an owner lookup whose hits this adapter will lend.

        Args:
            keys: Object keys to look up, without duplicates.
            group_layout_descs: Expected layouts indexed by object-group ID.

        Returns:
            The local lookup task ID.

        Raises:
            ValueError: If keys contain duplicates, which cannot identify
                independent reservations within one result dictionary.
        """
        if len(set(keys)) != len(keys):
            raise ValueError("Lookup keys of a lending adapter must be unique")
        return super().submit_lookup_and_lock_task(keys, group_layout_descs)

    def take_borrowed_objects(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        layouts: dict[int, MemoryLayoutDesc],
    ) -> dict[ObjectKey, BorrowedMemoryObj]:
        """Move selected reservations into borrowed objects.

        Args:
            task_id: Completed lookup task.
            keys: Selected keys, at most once each.
            layouts: Expected layout by object-group ID.

        Returns:
            A borrowed object per key; its lease is the owner's reservation,
            ending at release or owner expiry (minus the margin). Keys whose
            layout or size does not match, or whose reservation already
            expired, are omitted and remain releasable through
            :meth:`release_lookup`. No payload capacity is used.
        """
        hits = self._borrowed_lookups.get(task_id, {})
        objects: dict[ObjectKey, BorrowedMemoryObj] = {}
        for key in keys:
            hit = hits.get(key)
            if hit is None:
                continue
            address, expires = hit
            layout = layouts.get(key.object_group_id)
            if layout is None or time.monotonic() >= expires:
                continue
            size = layout_byte_size(layout)
            if not layout.shapes or size != address.size:
                continue
            metadata = MemoryObjMetadata(
                shape=layout.shapes[0],
                dtype=layout.dtypes[0],
                address=address.offset,
                phy_size=size,
                ref_count=1,
                fmt=MemoryFormat.KV_2LTD,
                shapes=layout.shapes,
                dtypes=layout.dtypes,
            )
            pending = _PendingUnlock({key: expires})
            obj = self._make_borrowed_object(
                address,
                metadata,
                lease_is_valid=partial(self._lease_is_valid, pending, expires),
                on_release=partial(self._end_lease, pending),
            )
            if obj is None:
                continue
            # Also ends the lease of an object dropped without release().
            pending.view = weakref.finalize(
                obj, self._release_view, weakref.ref(obj), pending
            )
            with self._borrow_lock:
                self._views.add(pending.view)
            objects[key] = obj
            del hits[key]
        if not hits:
            self._borrowed_lookups.pop(task_id, None)
        return objects

    def release_lookup(self, task_id: L2TaskId, keys: list[ObjectKey]) -> None:
        """Return unused reservations from one lookup; repeated calls are inert.

        Args:
            task_id: Completed local lookup identity.
            keys: Keys to release; adopted objects own their own reservations.
        """
        hits = self._borrowed_lookups.get(task_id, {})
        release = {}
        for key in keys:
            hit = hits.pop(key, None)
            if hit is not None:
                release[key] = hit[1]
        if not hits:
            self._borrowed_lookups.pop(task_id, None)
        if release:
            self._queue_unlock(release)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Reject copying; this adapter lends through take_borrowed_objects.

        The prefetch controller admits every borrowed key into the shadow L1
        and releases the rest, so it never plans a load against this adapter.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(f"{type(self).__name__} lends; it never copy-loads")

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Reject load queries; see :meth:`submit_load_task`.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(f"{type(self).__name__} lends; it never copy-loads")

    def get_active_borrow_count(self) -> int:
        """Return borrowed objects and pending unlocks that must drain."""
        with self._borrow_lock:
            return len(self._views) + self._pending_lookup_unlocks

    def report_status(self) -> dict[str, Any]:
        """Return the peer endpoint and the outstanding borrow count."""
        return {
            **super().report_status(),
            "active_borrows": self.get_active_borrow_count(),
        }

    @abstractmethod
    def _accept_address(self, address: TransferChannelAddress) -> bool:
        """Return whether a valid lookup hit can be lent by this adapter."""

    @abstractmethod
    def _make_borrowed_object(
        self,
        address: TransferChannelAddress,
        metadata: MemoryObjMetadata,
        *,
        lease_is_valid: Callable[[], bool],
        on_release: Callable[[], None],
    ) -> BorrowedMemoryObj | None:
        """Build the borrowed object for one accepted hit, or None to skip it.

        Args:
            address: The owner's address of the object.
            metadata: Layout of the object; ``address`` and ``phy_size`` set.
            lease_is_valid: Lease check to hand to the object.
            on_release: Lease end to hand to the object.
        """

    def _drain_borrows_for_close(self) -> None:
        """Release every unadopted reservation and stop the unlock worker.

        Called by a subclass's ``_unregister_peer_region`` before it releases
        the peer's region.

        Raises:
            RuntimeError: If an L1 still indexes live borrowed objects; the
                caller must drain them first.
        """
        with self._borrow_lock:
            if any(view.alive for view in self._views):
                raise RuntimeError(
                    f"Cannot close {type(self).__name__} with live borrowed objects"
                )
        for task_id, hits in list(self._borrowed_lookups.items()):
            self.release_lookup(task_id, list(hits))
        # The worker sends everything queued before this sentinel, then exits.
        # A retry after a failed teardown finds it stopped with nothing queued:
        # the first attempt released every lookup and draining admits none.
        if self._unlock_worker.is_alive():
            self._unlocks.put(None)
            self._unlock_worker.join()

    def _process_lookup_result(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        addresses: list[TransferChannelAddress],
        started_at: float,
    ) -> Bitmap:
        """Retain task-scoped reservations with a locally bounded expiry."""
        result = Bitmap(len(keys))
        hits: dict[ObjectKey, tuple[TransferChannelAddress, float]] = {}
        for i, (key, address) in enumerate(zip(keys, addresses, strict=True)):
            if not address.is_valid():
                continue
            ttl = address.read_ttl_seconds
            # Without a reported TTL the reservation never expires locally, so
            # its unlock is always sent (a lender may still refuse the hit).
            expires = (
                started_at + ttl - min(_OWNER_TTL_MARGIN_S, ttl / 10)
                if ttl > 0
                else math.inf
            )
            if not self._accept_address(address) or time.monotonic() >= expires:
                self._queue_unlock({key: expires})
            else:
                result.set(i)
                hits[key] = (address, expires)
        if hits:
            self._borrowed_lookups[task_id] = hits
        return result

    def _lease_is_valid(self, pending: _PendingUnlock, expires: float) -> bool:
        """A lease ends at release or when the owner's lock may expire."""
        view = cast(weakref.finalize, pending.view)
        return view.alive and time.monotonic() < expires

    @staticmethod
    def _end_lease(pending: _PendingUnlock) -> None:
        # Calling the finalizer runs it at most once, also after GC ran it.
        cast(weakref.finalize, pending.view)()

    def _release_view(
        self,
        reference: weakref.ReferenceType[BorrowedMemoryObj],
        pending: _PendingUnlock,
    ) -> None:
        # Runs as a finalizer, possibly inside garbage collection: no locks.
        obj = reference()
        if obj is not None:
            obj.invalidate()
        self._unlocks.put(pending)

    def _queue_unlock(self, reservations: dict[ObjectKey, float]) -> None:
        with self._borrow_lock:
            self._pending_lookup_unlocks += 1
        self._unlocks.put(_PendingUnlock(reservations))

    def _run_unlocks(self) -> None:
        """Send queued unlocks in batches until the close sentinel arrives."""
        while True:
            batch = [self._unlocks.get()]
            while True:
                try:
                    batch.append(self._unlocks.get_nowait())
                except queue.Empty:
                    break
            pending = [item for item in batch if item is not None]
            self._send_unlock(pending)
            with self._borrow_lock:
                for item in pending:
                    if item.view is None:
                        self._pending_lookup_unlocks -= 1
                    else:
                        self._views.discard(item.view)
            if len(pending) < len(batch):
                return

    def _send_unlock(self, pending: list[_PendingUnlock]) -> None:
        # Expired reservations belong to the owner's TTL cleanup. A late
        # key-only unlock must not consume a subsequent borrow's count;
        # ``expires`` is early by the TTL margin, so skipping errs safe.
        now = time.monotonic()
        keys = [
            key
            for item in pending
            for key, expires in item.reservations.items()
            if now < expires
        ]
        if not keys:
            return
        try:
            self.submit_unlock(keys, wait=True)
        except Exception:
            logger.exception("Peer unlock failed; reservations follow the owner TTL")
