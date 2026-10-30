# SPDX-License-Identifier: Apache-2.0
"""P2P L2 adapter: reads KV objects from a single peer cache server.

Lookups and unlocks are sent to the peer's P2P controller over the MQ; the
located objects are pulled from the peer's L1 over the transfer channel. The
adapter never stores, evicts, or deletes -- a peer's cache is read-only here.

Because neither the lookup RPC nor the transfer-channel read exposes a
completion fd, the lookup and load event fds are pulsed by the
``PeriodicEventNotifier`` singleton so the prefetch controller re-polls
``query_*`` periodically.

Thread model: the lookup / load / unlock calls all come from the single
prefetch-controller loop thread, and the store calls come from the
store-controller loop thread. The two paths share no mutable state (each owns
its own task-id counter and bookkeeping dicts), so this class needs no locks.
"""

# Standard
from abc import abstractmethod
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from typing import Any, cast
import math
import queue
import threading
import time

# Third Party
import zmq

# First Party
from lmcache.lmcache_native import Bitmap, PeriodicEventNotifier
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.memory_manager import BorrowedMemoryObj, Lease
from lmcache.v1.distributed.transfer_channel import get_transfer_channel_context
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform import HAS_EVENTFD, create_event_notifier

logger = init_logger(__name__)

_LOOKUP_RPC_TIMEOUT_S = 3.0
_PERIODIC_NOTIFIER_INTERVAL_MS = 5


@dataclass
class _LookupTask:
    keys: list[ObjectKey]
    remote_task_id: int
    deadline: float
    started_at: float
    failed: bool = False


@dataclass
class _LoadTask:
    keys: list[ObjectKey]
    read_task_id: int
    deadline: float
    failed: bool = False


class PeerL2AdapterConfigBase(L2AdapterConfigBase):
    """Fields shared by every adapter that looks up and locks keys on one peer.

    Fields:
    - peer_mq_server_url: Peer request server URL (lookup/unlock RPCs).
    - lookup_timeout_s: deadline for a lookup result before it counts as a miss.
    """

    def __init__(self, peer_mq_server_url: str, lookup_timeout_s: float) -> None:
        self.peer_mq_server_url = peer_mq_server_url
        self.lookup_timeout_s = lookup_timeout_s


class P2PL2AdapterConfig(PeerL2AdapterConfigBase):
    """Config for the P2P L2 adapter.

    Fields:
    - peer_mq_server_url: Peer request server URL (lookup/unlock RPCs).
    - peer_transfer_channel_server_url: the peer's transfer-channel server url.
    - lookup_timeout_s: deadline for a lookup result before it counts as a miss.
    - load_timeout_s: deadline for a load before it counts as a failure.
    """

    def __init__(
        self,
        peer_mq_server_url: str,
        peer_transfer_channel_server_url: str,
        lookup_timeout_s: float = 10.0,
        load_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(peer_mq_server_url, lookup_timeout_s)
        self.peer_transfer_channel_server_url = peer_transfer_channel_server_url
        self.load_timeout_s = load_timeout_s

    @classmethod
    def from_dict(cls, d: dict) -> "P2PL2AdapterConfig":
        peer_mq_server_url = d.get("peer_mq_server_url")
        if not isinstance(peer_mq_server_url, str) or not peer_mq_server_url:
            raise ValueError("peer_mq_server_url must be a non-empty string")

        peer_tc_url = d.get("peer_transfer_channel_server_url")
        if not isinstance(peer_tc_url, str) or not peer_tc_url:
            raise ValueError(
                "peer_transfer_channel_server_url must be a non-empty string"
            )

        lookup_timeout_s = d.get("lookup_timeout_s", 10.0)
        load_timeout_s = d.get("load_timeout_s", 10.0)
        if not isinstance(lookup_timeout_s, (int, float)) or lookup_timeout_s <= 0:
            raise ValueError("lookup_timeout_s must be a positive number")
        if not isinstance(load_timeout_s, (int, float)) or load_timeout_s <= 0:
            raise ValueError("load_timeout_s must be a positive number")

        return cls(
            peer_mq_server_url=peer_mq_server_url,
            peer_transfer_channel_server_url=peer_tc_url,
            lookup_timeout_s=float(lookup_timeout_s),
            load_timeout_s=float(load_timeout_s),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "P2P L2 adapter config fields:\n"
            "- peer_mq_server_url (str): peer request server URL (required)\n"
            "- peer_transfer_channel_server_url (str): the peer's transfer channel "
            "server url (required)\n"
            "- lookup_timeout_s (float): lookup result deadline in seconds "
            "(optional, default 10)\n"
            "- load_timeout_s (float): load deadline in seconds "
            "(optional, default 10)"
        )


class PeerL2AdapterBase(L2AdapterInterface):
    """Shared control plane for adapters that read one peer's region.

    Owns the lookup/unlock RPC client, the event fds, and the no-op store
    path. Construction calls :meth:`_register_peer_region`; ``close`` calls
    :meth:`_unregister_peer_region` after the controllers drain reads.
    Subclasses decide how a locked lookup hit becomes readable data
    (:meth:`_process_lookup_result`) and implement the load interface: a
    copying adapter records each hit's address for a later load, while a
    borrowing adapter keeps the reservation and hands L1 a borrowed view of
    the peer's bytes instead of copying them.

    Constructor contract: the region hook runs inside this constructor,
    after the base state is set but before the subclass's own ``__init__``
    body, so a subclass sets every attribute its hook needs before calling
    ``super().__init__()`` and acquires nothing that needs releasing after
    it returns; a failure in the subclass after that point is not unwound
    here.

    Args:
        config: Peer RPC endpoint and lookup deadline.
    """

    def __init__(self, config: PeerL2AdapterConfigBase) -> None:
        super().__init__(max_capacity_bytes=0)
        self._config = config
        # Prefetch-loop-thread state (lookup / load).
        self._next_task_id: L2TaskId = 0
        self._lookup_tasks: dict[L2TaskId, _LookupTask] = {}

        # Store-loop-thread state (store no-op completions).
        self._next_store_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}

        self._closed = False

        with ExitStack() as resources:
            self._req_client: RequestClient = RequestClientFactory.create(
                config.peer_mq_server_url,
                context=zmq.Context.instance(),
            )
            resources.callback(self._req_client.close)
            self._store_efd = create_event_notifier()
            resources.callback(self._store_efd.close)
            self._lookup_efd = create_event_notifier()
            resources.callback(self._lookup_efd.close)
            self._load_efd = create_event_notifier()
            resources.callback(self._load_efd.close)

            PeriodicEventNotifier.create(
                interval_ms=_PERIODIC_NOTIFIER_INTERVAL_MS, use_eventfd=HAS_EVENTFD
            )
            notifier = PeriodicEventNotifier.get()
            if notifier is None:
                raise RuntimeError(
                    "PeriodicEventNotifier is unavailable after create()"
                )
            for event in (self._lookup_efd, self._load_efd):
                notifier.register_fd(event.fileno())
                resources.callback(notifier.unregister_fd, event.fileno())
            self._register_peer_region()
            self._resources = resources.pop_all()

    # --------------------
    # Event Fd Interface
    # --------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # --------------------
    # Store Interface (no-op: a peer's cache is read-only)
    # --------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Record a 0-byte success and signal the store fd immediately.

        The P2P adapter never writes to a peer, but the store controller still
        tracks every submitted task (and the L1 read locks it reserved) until
        the result is popped. Completing the task right away lets the
        controller finalize that bookkeeping instead of leaking it.
        """
        task_id = self._next_store_task_id
        self._next_store_task_id += 1
        self._completed_store_tasks[task_id] = L2StoreResult(True, 0)
        self._store_efd.notify()
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        completed = self._completed_store_tasks
        self._completed_store_tasks = {}
        return completed

    # --------------------
    # Lookup and Lock Interface
    # --------------------

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1

        if self._closed:
            self._lookup_tasks[task_id] = _LookupTask(
                keys=keys, remote_task_id=-1, deadline=0.0, started_at=0.0, failed=True
            )
            return task_id

        # The owner's read-lock TTL starts when it handles the lookup; this
        # earlier timestamp bounds it from the borrower's side.
        started_at = time.monotonic()
        future = self._req_client.p2p_lookup_and_lock(keys, group_layout_descs)
        failed = False
        remote_task_id = -1
        try:
            remote_task_id = future.result(timeout=_LOOKUP_RPC_TIMEOUT_S)
        except TimeoutError:
            logger.warning(
                "P2P lookup submit to %s timed out; treating as a miss",
                self._config.peer_mq_server_url,
            )
            failed = True

        self._lookup_tasks[task_id] = _LookupTask(
            keys=keys,
            remote_task_id=remote_task_id,
            deadline=started_at + self._config.lookup_timeout_s,
            started_at=started_at,
            failed=failed,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        task = self._lookup_tasks.get(task_id)
        if task is None:
            return None
        if task.failed:
            del self._lookup_tasks[task_id]
            return Bitmap(len(task.keys))
        if time.monotonic() > task.deadline:
            del self._lookup_tasks[task_id]
            logger.warning("P2P lookup task %d timed out; treating as a miss", task_id)
            return Bitmap(len(task.keys))

        future = self._req_client.p2p_query_lookup_results(task.remote_task_id)
        try:
            addresses = future.result(timeout=_LOOKUP_RPC_TIMEOUT_S)
        except TimeoutError:
            return None

        if addresses is None:
            return None

        bitmap = self._process_lookup_result(
            task_id, task.keys, addresses, task.started_at
        )
        del self._lookup_tasks[task_id]
        return bitmap

    def submit_unlock(self, keys: list[ObjectKey], *, wait: bool = False) -> None:
        """Release peer reservations through the unlock RPC.

        Args:
            keys: Object keys to unlock. Empty input is a no-op.
            wait: Wait for the peer's acknowledgment instead of fire-and-forget.

        Raises:
            TimeoutError: If ``wait`` is True and no acknowledgment arrives
                within the lookup RPC timeout.
        """
        if not keys:
            return
        future = self._req_client.p2p_unlock_objects(keys)
        if wait:
            future.result(timeout=_LOOKUP_RPC_TIMEOUT_S)

    # --------------------
    # Lifecycle / status
    # --------------------

    def close(self) -> None:
        """Unregister the drained peer region, then close shared RPC resources.

        Idempotent after success. If region teardown raises, the adapter
        stays open, so the caller keeps the unlock connection and can retry.
        Once the region is released the adapter counts as closed: closing the
        RPC and fd resources is attempted for every one of them, the first
        error propagates, and a retry is a no-op.
        """
        if self._closed:
            return
        self._unregister_peer_region()
        self._closed = True
        self._resources.close()

    def report_status(self) -> dict[str, Any]:
        """Return health, the peer endpoint and the in-flight lookup count."""
        return {
            "is_healthy": True,
            "peer_mq_server_url": self._config.peer_mq_server_url,
            "in_flight_lookups": len(self._lookup_tasks),
        }

    @abstractmethod
    def _register_peer_region(self) -> None:
        """Make the configured peer's region readable by this adapter.

        Lifecycle hook called once during construction, after the RPC client
        and event fds exist and before serving requests. The adapter owns the
        resulting region access until ``close`` calls
        :meth:`_unregister_peer_region`. If it raises, everything acquired
        before it is released and construction fails.

        Region access that is shared per peer (for example a transfer-channel
        client keyed by the peer's URL) must not be re-acquired for the same
        peer while an adapter whose close failed still holds it: that
        adapter's later successful close would release the shared access
        under the new adapter.
        """

    @abstractmethod
    def _unregister_peer_region(self) -> None:
        """Release the region access acquired by :meth:`_register_peer_region`.

        Lifecycle hook called by ``close`` after controllers stop submissions
        and drain reads. If it raises, ``close`` leaves the adapter open,
        with its RPC resources, so teardown can be retried.
        """

    @abstractmethod
    def _process_lookup_result(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        addresses: list[TransferChannelAddress],
        started_at: float,
    ) -> Bitmap:
        """Turn a completed lookup's addresses into a hit bitmap.

        Called once per completed lookup, before the task is forgotten. This
        is where the read strategy diverges: record the addresses for a copy
        load, or keep the reservations and hand out borrowed views.

        Args:
            task_id: The local lookup task.
            keys: Keys of the lookup, in request order.
            addresses: The peer's address per key; invalid for a miss.
            started_at: ``time.monotonic()`` just before the lookup was sent,
                a lower bound on when the owner's read locks started.

        Returns:
            A bitmap with bit ``i`` set when ``keys[i]`` is readable.
        """


# Stop trusting a borrowed object this long before the owner's read lock can
# expire: a GPU read that passed the last validity check must finish before
# the owner may reuse the bytes. Capped at a tenth of the TTL so short TTLs
# still leave a usable window.
_OWNER_TTL_MARGIN_S = 1.0


@dataclass(eq=False)
class _PendingUnlock:
    """Owner reservations waiting for the unlock worker.

    A lease is one reservation a borrowed object owns until its release; the
    others are reservations ``submit_unlock`` returns without an object.
    """

    reservations: dict[ObjectKey, float]
    is_lease: bool = False
    ended: bool = False


class BorrowingPeerL2AdapterBase(PeerL2AdapterBase):
    """A peer adapter that borrows the peer's bytes under the owner's read lock.

    A copying peer adapter (:class:`P2PL2Adapter`) moves a looked-up object
    into this host's L1 and then releases the owner's read lock. A borrower
    keeps that lock for as long as readers here use the object: the lookup
    reservation becomes the lease of a
    :class:`~lmcache.v1.distributed.memory_manager.BorrowedMemoryObj` that
    the shadow L1 indexes, and the unlock is sent when the last reader frees
    it. It wants a ``SHADOW`` L1, the configured shadow of the segment it
    borrows, and maps the peer's memory itself.

    It goes through the prefetch controller's ordinary load pipeline. Every
    lookup hit adds one reservation for its key, with the owner's address and
    a local expiry bounded by the owner's TTL. The controller stages a
    placeholder per planned key in the shadow and submits a load with them;
    :meth:`submit_load_task` binds each placeholder to one reservation of its
    key, which the object now owns and returns at its release.
    :meth:`submit_unlock` returns one reservation per key the caller did not
    borrow. Every request looks the borrower up; a view already resident in the
    shadow adopts the newer lease at admission, and the older reservation
    comes back through the discarded placeholder. Releases only enqueue work;
    a worker sends the unlock RPC outside L1 locks, batching everything queued
    since its last RPC so a slow or dead owner costs one timeout per batch,
    not per object.

    A subclass decides how a placeholder is bound to the bytes
    (:meth:`_bind_borrowed_object`) and when a lookup hit is usable at all
    (:meth:`_accept_address`).

    Thread model: lookups, loads and ``submit_unlock`` run on the prefetch
    thread; a lease ends on whichever thread frees its object, under the L1
    manager's lock, so ending it only flags the lease and enqueues its unlock.
    The shadow L1 frees every placeholder it staged, so every lease ends.
    """

    # How long a read that passed the lease check may still take: a lease is
    # valid only while that much of the owner's lock remains.
    _read_window_s: float = 0.0

    def __init__(self, config: PeerL2AdapterConfigBase) -> None:
        """Set up the lease bookkeeping and the peer control plane.

        The borrowing state is set up before the base constructor runs the
        region hook, per the :class:`PeerL2AdapterBase` constructor contract.

        Args:
            config: Peer RPC endpoint and lookup deadline.
        """
        self._reservations: dict[
            ObjectKey, deque[tuple[TransferChannelAddress, float]]
        ] = {}
        # Guards _leases and _pending_lookup_unlocks.
        self._borrow_lock = threading.Lock()
        # Leases whose unlock the worker has not yet sent.
        self._leases: set[_PendingUnlock] = set()
        self._pending_lookup_unlocks = 0
        self._load_results: dict[L2TaskId, Bitmap] = {}
        self._next_load_task = 0
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
        """Submit an owner lookup whose hits this adapter will borrow.

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
            raise ValueError("Lookup keys of a borrowing adapter must be unique")
        return super().submit_lookup_and_lock_task(keys, group_layout_descs)

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Bind each placeholder to one reservation of its key.

        Args:
            keys: Planned keys, at most once each.
            objects: The shadow L1's placeholder per key.

        Returns:
            The load task id; :meth:`query_load_result` has its result at
            once. A key with no reservation, a size mismatch or an expired
            reservation reports a miss and its placeholder stays unbound;
            the caller discards it and returns the reservation, if any,
            with :meth:`submit_unlock`. No payload capacity is used.

        Note:
            Never raises once it has bound an object: the shadow L1 releases
            every placeholder it staged, so a bound lease always ends.
        """
        bitmap = Bitmap(len(keys))
        for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
            held = self._reservations.get(key)
            if not held:
                continue
            address, expires = held[0]
            if not self._lease_open(expires) or obj.get_size() != address.size:
                continue
            borrowed = cast(BorrowedMemoryObj, obj)
            pending = _PendingUnlock({key: expires}, is_lease=True)
            lease: Lease = (
                partial(self._lease_is_valid, pending, expires),
                partial(self._end_lease, pending),
            )
            if not self._bind_borrowed_object(borrowed, address, lease):
                continue
            self._pop_reservation(key)
            with self._borrow_lock:
                self._leases.add(pending)
            bitmap.set(i)
        task_id = self._next_load_task
        self._next_load_task += 1
        self._load_results[task_id] = bitmap
        self._load_efd.notify()
        return task_id

    def submit_unlock(self, keys: list[ObjectKey], *, wait: bool = False) -> None:
        """Return one reservation per key to the owner, through the worker.

        Args:
            keys: Keys whose reservation to return; a key this adapter holds
                no reservation for (its hit was refused at lookup time, or
                became a borrowed object) is skipped.
            wait: Ignored; the worker always waits for the owner.
        """
        release: dict[ObjectKey, float] = {}
        for key in keys:
            held = self._reservations.get(key)
            if held:
                release[key] = self._pop_reservation(key)
        if release:
            self._queue_unlock(release)

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return which keys of a load task were bound, once.

        Args:
            task_id: The id :meth:`submit_load_task` returned.

        Returns:
            The bound bitmap over the task's keys; None for an unknown or
            already consumed task.
        """
        return self._load_results.pop(task_id, None)

    def get_local_borrow_count(self) -> int:
        """Return borrowed objects and pending unlocks that must drain."""
        with self._borrow_lock:
            return len(self._leases) + self._pending_lookup_unlocks

    def report_status(self) -> dict[str, Any]:
        """Return the peer endpoint and the outstanding borrow count."""
        return {
            **super().report_status(),
            "local_borrows": self.get_local_borrow_count(),
        }

    @abstractmethod
    def _accept_address(self, address: TransferChannelAddress) -> bool:
        """Return whether a valid lookup hit can be borrowed by this adapter."""

    @abstractmethod
    def _bind_borrowed_object(
        self, obj: BorrowedMemoryObj, address: TransferChannelAddress, lease: Lease
    ) -> bool:
        """Bind a placeholder to the bytes at one accepted hit.

        Args:
            obj: The shadow L1's unbound placeholder for the key.
            address: The owner's address of the object.
            lease: The reservation, to pass to ``obj.bind``.

        Returns:
            True once ``obj`` is bound; False to skip the key, leaving it
            unbound and its reservation held.
        """

    def _drain_borrows_for_close(self) -> None:
        """Return every unborrowed reservation and stop the unlock worker.

        Called by a subclass's ``_unregister_peer_region`` before it releases
        the peer's region.

        A lease the owner's lock may have expired under is drained: no read
        of it can be in flight, and the owner's own TTL reclaims the lock.

        Raises:
            RuntimeError: If an L1 still indexes live borrowed objects; the
                caller retries once their readers finish.
        """
        with self._borrow_lock:
            live = [
                lease
                for lease in self._leases
                if not lease.ended
                and any(self._lease_open(e) for e in lease.reservations.values())
            ]
            if live:
                raise RuntimeError(
                    f"Cannot close {type(self).__name__} with live borrowed objects"
                )
        for key, held in list(self._reservations.items()):
            for _, expires in held:
                self._queue_unlock({key: expires})
        self._reservations.clear()
        # The worker sends everything queued before this sentinel, then exits.
        # A retry after a failed teardown finds it stopped with nothing queued:
        # the first attempt returned every reservation and draining admits none.
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
        """Keep one reservation per hit with a locally bounded expiry."""
        result = Bitmap(len(keys))
        for i, (key, address) in enumerate(zip(keys, addresses, strict=True)):
            if not address.is_valid():
                continue
            ttl = address.read_ttl_seconds
            # Without a reported TTL the reservation never expires locally, so
            # its unlock is always sent (a borrower may still refuse the hit).
            expires = (
                started_at + ttl - min(_OWNER_TTL_MARGIN_S, ttl / 10)
                if ttl > 0
                else math.inf
            )
            if not self._accept_address(address) or not self._lease_open(expires):
                self._queue_unlock({key: expires})
            else:
                result.set(i)
                self._reservations.setdefault(key, deque()).append((address, expires))
        return result

    def _pop_reservation(self, key: ObjectKey) -> float:
        held = self._reservations[key]
        _, expires = held.popleft()
        if not held:
            del self._reservations[key]
        return expires

    def _lease_open(self, expires: float) -> bool:
        """Whether a read starting now finishes before the owner's lock may."""
        return time.monotonic() + self._read_window_s < expires

    def _lease_is_valid(self, pending: _PendingUnlock, expires: float) -> bool:
        """A lease ends at release or when the owner's lock may expire before
        a read started now could finish."""
        return not pending.ended and self._lease_open(expires)

    def _end_lease(self, pending: _PendingUnlock) -> None:
        # Runs under the L1 manager's lock on the releasing thread: flag and
        # enqueue only; the worker sends the unlock. Idempotent.
        with self._borrow_lock:
            if pending.ended:
                return
            pending.ended = True
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
            if keys:
                try:
                    PeerL2AdapterBase.submit_unlock(self, keys, wait=True)
                except Exception:
                    logger.exception(
                        "Peer unlock failed; reservations follow the owner TTL"
                    )
            with self._borrow_lock:
                for item in pending:
                    if item.is_lease:
                        self._leases.discard(item)
                    else:
                        self._pending_lookup_unlocks -= 1
            if len(pending) < len(batch):
                return


class P2PL2Adapter(PeerL2AdapterBase):
    """Copy objects from one peer's registered region over the transfer channel.

    Args:
        config: Peer RPC/transfer endpoints and lookup/load deadlines.
    """

    _config: P2PL2AdapterConfig

    def __init__(self, config: P2PL2AdapterConfig) -> None:
        # Prefetch-loop-thread state (load), set before the base constructor
        # runs the region hook so nothing is acquired after it returns.
        self._load_tasks: dict[L2TaskId, _LoadTask] = {}
        self._remote_addresses: dict[ObjectKey, TransferChannelAddress] = {}
        super().__init__(config)

    def _register_peer_region(self) -> None:
        """Get or create the transfer-channel client for the configured peer.

        The client performs the handshake that imports the peer's registered
        region, or reuses one the peer's own connection already created.
        """
        self._tc_context = get_transfer_channel_context()
        self._tc_client = self._tc_context.get_transfer_channel_client(
            self._config.peer_transfer_channel_server_url
        )

    def _unregister_peer_region(self) -> None:
        """Drop the peer's transfer-channel client through the context.

        Called by ``close`` after controllers stop submissions and drain reads.
        """
        self._tc_context.remove_transfer_channel_client(
            self._config.peer_transfer_channel_server_url
        )

    def submit_unlock(self, keys: list[ObjectKey], *, wait: bool = False) -> None:
        """Release peer reservations and discard their cached transfer addresses.

        Args:
            keys: Object keys to unlock. Empty input is a no-op.
            wait: Wait for the peer's acknowledgment.
        """
        super().submit_unlock(keys, wait=wait)
        for key in keys:
            self._remote_addresses.pop(key, None)

    # --------------------
    # Load Interface
    # --------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1

        remote_addresses: list[TransferChannelAddress] = []
        for key in keys:
            addr = self._remote_addresses.get(key)
            if addr is None or not addr.is_valid():
                logger.warning(
                    "P2P load task %d has a missing/invalid remote address; "
                    "treating as a failure",
                    task_id,
                )
                self._load_tasks[task_id] = _LoadTask(
                    keys=keys, read_task_id=-1, deadline=0.0, failed=True
                )
                return task_id
            remote_addresses.append(addr)

        local_addresses = self._tc_context.get_transfer_channel_address(
            [(obj.shm_offset, obj.shm_byte_length) for obj in objects]
        )
        read_task_id = self._tc_client.submit_read(
            local_addresses,
            remote_addresses,  # type: ignore
        )

        self._load_tasks[task_id] = _LoadTask(
            keys=keys,
            read_task_id=read_task_id,
            deadline=time.monotonic() + self._config.load_timeout_s,
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        task = self._load_tasks.get(task_id)
        if task is None:
            return None
        if task.failed:
            del self._load_tasks[task_id]
            return Bitmap(len(task.keys))
        if time.monotonic() > task.deadline:
            del self._load_tasks[task_id]
            logger.warning("P2P load task %d timed out; treating as a failure", task_id)
            return Bitmap(len(task.keys))

        result = self._tc_client.query_read_status(task.read_task_id)
        if not result.is_finished():
            return None

        bitmap = Bitmap(len(task.keys))
        for i, succeeded in enumerate(result.succeeded_mask):
            if succeeded:
                bitmap.set(i)
        del self._load_tasks[task_id]
        return bitmap

    # --------------------
    # Lifecycle / status
    # --------------------

    def get_memory_tags(self) -> MemoryTags:
        """Return the memory tags of this adapter: a peer's DRAM L1."""
        return MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE)

    def report_status(self) -> dict:
        return {
            **super().report_status(),
            "type": "P2PL2Adapter",
            "peer_transfer_channel_server_url": (
                self._config.peer_transfer_channel_server_url
            ),
            "in_flight_loads": len(self._load_tasks),
        }

    def _process_lookup_result(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        addresses: list[TransferChannelAddress],
        started_at: float,
    ) -> Bitmap:
        """Cache each hit's transfer address for the following load."""
        bitmap = Bitmap(len(keys))
        for i, (key, addr) in enumerate(zip(keys, addresses, strict=True)):
            if addr.is_valid():
                bitmap.set(i)
                self._remote_addresses[key] = addr
        return bitmap


# Self-register config type and adapter factory
register_l2_adapter_type("p2p", P2PL2AdapterConfig)


def _create_p2p_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    """Create a P2PL2Adapter from config (l1_memory_desc is unused)."""
    return P2PL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("p2p", _create_p2p_adapter)
