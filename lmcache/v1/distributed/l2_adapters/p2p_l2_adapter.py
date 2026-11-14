# SPDX-License-Identifier: Apache-2.0
"""P2P L2 adapter: reads KV objects from, and optionally writes them to, a
single peer cache server.

Lookups and unlocks are sent to the peer's P2P controller over the MQ; the
located objects are pulled from the peer's L1 over the transfer channel. With
``write_enabled`` a store reserves buffers in the peer's L1, pushes the
objects over the transfer channel and commits them. Without it a store is a
no-op. The adapter never evicts or deletes on the peer.

A lookup is two RPCs: ``p2p_lookup_and_lock`` returns the peer's task id and
``p2p_query_lookup_results`` returns the addresses. Both are chained from the
RPC completions: the lookup acknowledgement issues the query, and the query
result signals the lookup event fd, so a lookup completes without waiting for
a periodic pulse and the prefetch loop never blocks on a peer RPC. The
transfer-channel read exposes no completion fd, so the load event fd (and a
lookup whose result was not ready yet) is pulsed by the
``PeriodicEventNotifier`` singleton and re-polled through ``query_*``; so is
the store fd when writes are enabled. A store does not wait for a pulse to act
on an RPC answer: the answer's callback moves the store on.

Thread model: the lookup / load / unlock calls all come from the single
prefetch-controller loop thread, and the store calls come from the
store-controller loop thread. The two paths share no mutable state of this
class (each owns its own task-id counter and bookkeeping dicts). Lookup RPC
completions run on the request client's thread, so lookup task state is
guarded by a lock. With writes enabled, the answer to a store's reservation is
handled on the request client's receive thread, so the store bookkeeping is
guarded by a lock too. The request client and the transfer-channel client lock
internally.
"""

# Standard
from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any
import functools
import threading
import time

# Third Party
import zmq

# First Party
from lmcache.lmcache_native import Bitmap, PeriodicEventNotifier
from lmcache.logging import init_logger
from lmcache.v1.client.transport.errors import REQUEST_FAILURES
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.common.transfer_channel.api import TransferChannelAddress
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
from lmcache.v1.distributed.transfer_channel import get_transfer_channel_context
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform import HAS_EVENTFD, create_event_notifier
from lmcache.v1.transport.futures import MessagingFuture

logger = init_logger(__name__)

_LOOKUP_RPC_TIMEOUT_S = 3.0
_PERIODIC_NOTIFIER_INTERVAL_MS = 5


@dataclass
class _LookupTask:
    keys: list[ObjectKey]
    deadline: float
    ack_deadline: float
    remote_task_id: int | None = None
    addresses: list[TransferChannelAddress] | None = None
    # Monotonic send time of the in-flight result query, None if none.
    query_sent_at: float | None = None
    query_failures: int = 0
    failed: bool = False


@dataclass
class _LoadTask:
    keys: list[ObjectKey]
    read_task_id: int
    deadline: float
    failed: bool = False


@dataclass
class _StoreTask:
    """One store in flight; the fields set so far tell how far it got."""

    keys: list[ObjectKey]
    objects: list[MemoryObj]
    deadline: float
    reserve_future: MessagingFuture[list[TransferChannelAddress]]
    reserved_keys: list[ObjectKey] = field(default_factory=list)
    """ Keys the peer reserved a buffer for, in request order. """
    write_task_id: int | None = None
    """ Transfer-channel task writing ``reserved_keys``; set once submitted. """
    commit_future: MessagingFuture[list[bool]] | None = None
    """ Set once the peer was asked to commit a fully written task. """


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
    - write_enabled: store objects into the peer's L1; off makes stores no-ops.
    - store_timeout_s: deadline for a store before it counts as a failure.
    """

    def __init__(
        self,
        peer_mq_server_url: str,
        peer_transfer_channel_server_url: str,
        lookup_timeout_s: float = 10.0,
        load_timeout_s: float = 10.0,
        write_enabled: bool = False,
        store_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(peer_mq_server_url, lookup_timeout_s)
        self.peer_transfer_channel_server_url = peer_transfer_channel_server_url
        self.load_timeout_s = load_timeout_s
        self.write_enabled = write_enabled
        self.store_timeout_s = store_timeout_s

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

        write_enabled = d.get("write_enabled", False)
        if not isinstance(write_enabled, bool):
            raise ValueError("write_enabled must be a boolean")
        store_timeout_s = d.get("store_timeout_s", 10.0)
        if not isinstance(store_timeout_s, (int, float)) or store_timeout_s <= 0:
            raise ValueError("store_timeout_s must be a positive number")

        return cls(
            peer_mq_server_url=peer_mq_server_url,
            peer_transfer_channel_server_url=peer_tc_url,
            lookup_timeout_s=float(lookup_timeout_s),
            load_timeout_s=float(load_timeout_s),
            write_enabled=write_enabled,
            store_timeout_s=float(store_timeout_s),
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
            "(optional, default 10)\n"
            "- write_enabled (bool): store objects into the peer's L1 "
            "(optional, default false)\n"
            "- store_timeout_s (float): store deadline in seconds "
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
    lending adapter keeps the reservation and hands L1 a borrowed view of
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
        # Guards _lookup_tasks and _closed against lookup RPC completions,
        # which run on the request client's thread.
        self._lookup_lock = threading.Lock()

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
            polled_events = [self._lookup_efd, self._load_efd]
            if self._stores_need_polling():
                polled_events.append(self._store_efd)
            for event in polled_events:
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
    # Store Interface (no-op here; a subclass may write to the peer)
    # --------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Record a 0-byte success and signal the store fd immediately.

        This base never writes to a peer, but the store controller still
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
        now = time.monotonic()
        task = _LookupTask(
            keys=keys,
            deadline=now + self._config.lookup_timeout_s,
            ack_deadline=now + _LOOKUP_RPC_TIMEOUT_S,
        )
        with self._lookup_lock:
            self._lookup_tasks[task_id] = task
            if self._closed:
                task.failed = True
                return task_id

        future = self._req_client.p2p_lookup_and_lock(keys, group_layout_descs)
        future.add_done_callback(lambda fut: self._on_lookup_ack(task_id, fut))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lookup_lock:
            task = self._lookup_tasks.get(task_id)
            if task is None:
                return None
            now = time.monotonic()
            # A landed result wins over an expired deadline: the peer already
            # holds read locks for it, which the prefetch controller releases.
            done = task.failed or task.addresses is not None
            expired = now > task.deadline or (
                task.remote_task_id is None and now > task.ack_deadline
            )
            # A ZMQ reply lost with its peer never fails the future, so an
            # unanswered query is re-sent after the RPC timeout. The abandoned
            # future stays pending in the client; the lookup deadline bounds
            # how many a lookup can leave behind.
            query_pending = (
                task.query_sent_at is not None
                and now - task.query_sent_at < _LOOKUP_RPC_TIMEOUT_S
            )
            if done or expired:
                del self._lookup_tasks[task_id]
            elif task.remote_task_id is None or query_pending:
                return None
            else:
                task.query_sent_at = now
                remote_task_id = task.remote_task_id

        if task.addresses is not None:
            return self._process_lookup_result(task.keys, task.addresses)
        if task.failed:
            return Bitmap(len(task.keys))
        if expired:
            logger.warning("P2P lookup task %d timed out; treating as a miss", task_id)
            return Bitmap(len(task.keys))
        # The peer had no result yet: ask again; the answer signals the fd
        # (or is already here if the query completed synchronously).
        self._submit_query(task_id, remote_task_id)
        with self._lookup_lock:
            if task.addresses is None or self._lookup_tasks.get(task_id) is not task:
                return None
            del self._lookup_tasks[task_id]
        return self._process_lookup_result(task.keys, task.addresses)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        if not keys:
            return
        self._req_client.p2p_unlock_objects(keys)

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
        with self._lookup_lock:
            self._closed = True
        self._resources.close()

    def report_status(self) -> dict[str, Any]:
        """Return health, the peer endpoint and the in-flight lookup count."""
        return {
            "is_healthy": True,
            "peer_mq_server_url": self._config.peer_mq_server_url,
            "in_flight_lookups": len(self._lookup_tasks),
        }

    def _stores_need_polling(self) -> bool:
        """Whether store completions are found by polling.

        Called once during construction. When it returns True the store event
        fd is pulsed periodically, like the lookup and load fds, so the store
        controller keeps calling ``pop_completed_store_tasks`` while stores
        are in flight. The no-op store of this class signals the fd itself
        and needs no pulse.
        """
        return False

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

    def _on_lookup_ack(self, task_id: L2TaskId, future: MessagingFuture[int]) -> None:
        """Record the peer task id and chain the result query.

        Runs on the request client's thread when ``p2p_lookup_and_lock``
        completes.

        Args:
            task_id: The local lookup task id.
            future: The completed ``p2p_lookup_and_lock`` future.
        """
        try:
            remote_task_id = future.result()
        # Any transport or server error must become a miss now: an error that
        # escapes the callback is only logged, leaving the task pending until
        # its ack deadline.
        except Exception:
            if not self._closed:
                logger.exception(
                    "P2P lookup submit to %s failed; treating as a miss",
                    self._config.peer_mq_server_url,
                )
            self._finish_lookup(task_id, failed=True)
            return
        with self._lookup_lock:
            task = self._lookup_tasks.get(task_id)
            if task is None or self._closed:
                return
            task.remote_task_id = remote_task_id
            task.query_sent_at = time.monotonic()
        self._submit_query(task_id, remote_task_id)

    def _submit_query(self, task_id: L2TaskId, remote_task_id: int) -> None:
        """Send ``p2p_query_lookup_results`` and handle its answer when it lands.

        Args:
            task_id: The local lookup task id.
            remote_task_id: The peer's task id for the lookup.
        """
        future = self._req_client.p2p_query_lookup_results(remote_task_id)
        future.add_done_callback(lambda fut: self._on_query_result(task_id, fut))

    def _on_query_result(
        self,
        task_id: L2TaskId,
        future: MessagingFuture[list[TransferChannelAddress] | None],
    ) -> None:
        """Store the lookup addresses and signal the lookup event fd.

        Runs on the request client's thread. A query that finds no result
        yet leaves the task to be re-queried on the next poll; a first failed
        query is retried the same way, a second one makes the lookup a miss.

        Args:
            task_id: The local lookup task id.
            future: The completed ``p2p_query_lookup_results`` future.
        """
        error: Exception | None = None
        try:
            addresses = future.result()
        # Same as the ack: an error here must not escape the client thread.
        except Exception as exc:
            addresses, error = None, exc
        with self._lookup_lock:
            task = self._lookup_tasks.get(task_id)
            if task is None or self._closed:
                return
            task.query_sent_at = None
            if error is not None:
                task.query_failures += 1
                if task.query_failures < 2:
                    logger.warning(
                        "P2P lookup query to %s failed (%s); retrying once",
                        self._config.peer_mq_server_url,
                        error,
                    )
                    return
                logger.error(
                    "P2P lookup query to %s failed again (%s); treating as a miss",
                    self._config.peer_mq_server_url,
                    error,
                )
                task.failed = True
                self._lookup_efd.notify()
                return
            if addresses is None:
                return
            task.addresses = addresses
            self._lookup_efd.notify()

    def _finish_lookup(self, task_id: L2TaskId, failed: bool) -> None:
        """Mark a lookup finished and signal the lookup event fd.

        Args:
            task_id: The local lookup task id.
            failed: Whether the lookup failed (reported as a miss).
        """
        with self._lookup_lock:
            task = self._lookup_tasks.get(task_id)
            if task is None or self._closed:
                return
            task.failed = failed
            self._lookup_efd.notify()

    @abstractmethod
    def _process_lookup_result(
        self,
        keys: list[ObjectKey],
        addresses: list[TransferChannelAddress],
    ) -> Bitmap:
        """Turn a completed lookup's addresses into a hit bitmap.

        Called once per completed lookup, before the task is forgotten. This
        is where the read strategy diverges: record the addresses for a copy
        load, or keep the reservations to lend borrowed views.

        Args:
            keys: Keys of the lookup, in request order.
            addresses: The peer's address per key; invalid for a miss.

        Returns:
            A bitmap with bit ``i`` set when ``keys[i]`` is readable.
        """


class P2PL2Adapter(PeerL2AdapterBase):
    """Copy objects from, and optionally to, one peer's registered region.

    Args:
        config: Peer RPC/transfer endpoints, deadlines and the write switch.
    """

    _config: P2PL2AdapterConfig

    def __init__(self, config: P2PL2AdapterConfig) -> None:
        # Prefetch-loop-thread state (load), set before the base constructor
        # runs the region hook so nothing is acquired after it returns.
        self._load_tasks: dict[L2TaskId, _LoadTask] = {}
        self._remote_addresses: dict[ObjectKey, TransferChannelAddress] = {}
        # Stores in flight to the peer. With writes enabled this, the store
        # task-id counter and the completed stores are shared between the
        # store loop thread and the request client's receive thread.
        self._store_tasks: dict[L2TaskId, _StoreTask] = {}
        self._store_lock = threading.Lock()
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

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release peer reservations and discard their cached transfer addresses.

        Args:
            keys: Object keys to unlock. Empty input is a no-op.
        """
        super().submit_unlock(keys)
        for key in keys:
            self._remote_addresses.pop(key, None)

    # --------------------
    # Store Interface
    # --------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Start storing the objects into the peer's L1.

        Only asks the peer to reserve buffers. The transfer starts as soon
        as the peer answers, and the commit follows the transfer. With
        writes disabled the task completes at once as a 0-byte success.

        Args:
            keys: Keys to store.
            objects: The objects to store, parallel to ``keys``, all with
                the same memory layout.

        Returns:
            The id of the store task.

        Note:
            Never waits for the peer.
        """
        if not self._config.write_enabled:
            return super().submit_store_task(keys, objects)

        with self._store_lock:
            task_id = self._next_store_task_id
            self._next_store_task_id += 1
            if self._closed:
                self._completed_store_tasks[task_id] = L2StoreResult(False, 0)
                return task_id

            layout_desc = MemoryLayoutDesc(
                shapes=objects[0].get_shapes(), dtypes=objects[0].get_dtypes()
            )
            reserve_future = self._req_client.p2p_reserve_write(keys, layout_desc)
            self._store_tasks[task_id] = _StoreTask(
                keys=keys,
                objects=objects,
                deadline=time.monotonic() + self._config.store_timeout_s,
                reserve_future=reserve_future,
            )
        # Registered outside the lock: an already answered request runs the
        # callback right here, and the callback takes the lock.
        reserve_future.add_done_callback(
            functools.partial(self._on_reservation_answered, task_id)
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Advance the stores in flight and return the ones that finished.

        Returns:
            The result per finished task. A task succeeds only when every
            one of its keys is resident on the peer; any other outcome is a
            failure with 0 bytes.

        Note:
            Never waits: a store whose next step is not ready stays in
            flight until a later call.
        """
        with self._store_lock:
            completed = super().pop_completed_store_tasks()
            now = time.monotonic()
            for task_id, task in list(self._store_tasks.items()):
                result = self._advance_store_task(task, now)
                if result is not None:
                    del self._store_tasks[task_id]
                    completed[task_id] = result
        return completed

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

        result = self._tc_client.query_status(task.read_task_id)
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
        with self._store_lock:
            in_flight_stores = len(self._store_tasks)
        return {
            **super().report_status(),
            "type": "P2PL2Adapter",
            "peer_transfer_channel_server_url": (
                self._config.peer_transfer_channel_server_url
            ),
            "in_flight_loads": len(self._load_tasks),
            "write_enabled": self._config.write_enabled,
            "in_flight_stores": in_flight_stores,
        }

    def _stores_need_polling(self) -> bool:
        """Poll for store completions only when stores reach the peer."""
        return self._config.write_enabled

    def _on_reservation_answered(
        self,
        task_id: L2TaskId,
        reserve_future: MessagingFuture[list[TransferChannelAddress]],
    ) -> None:
        """Start a store's transfer as soon as the peer answered its reservation.

        Args:
            task_id: The store the reservation belongs to.
            reserve_future: The answered reservation request.

        Note:
            Runs on the thread that completed the request, normally the
            request client's receive thread, so the store does not wait for
            the next pulse of the store loop. A store that already failed
            (deadline, close) is left alone.
        """
        with self._store_lock:
            task = self._store_tasks.get(task_id)
            if task is None or self._closed:
                return
            result = self._start_transfer(task, reserve_future)
            if result is None:
                return
            del self._store_tasks[task_id]
            self._completed_store_tasks[task_id] = result
        self._store_efd.notify()

    def _on_commit_answered(self, commit_future: MessagingFuture[list[bool]]) -> None:
        """Wake the store loop to collect a store whose commit was answered.

        Args:
            commit_future: The answered commit request.
        """
        if not self._closed:
            self._store_efd.notify()

    def _advance_store_task(self, task: _StoreTask, now: float) -> L2StoreResult | None:
        """Take one store as far as it can go without waiting.

        Args:
            task: The store to advance.
            now: ``time.monotonic()`` of this poll.

        Returns:
            The store's result once it is known, else None.
        """
        if task.commit_future is not None:
            return self._check_commit(task, task.commit_future, now)
        if task.write_task_id is not None:
            return self._check_write(task, task.write_task_id, now)
        # Still waiting for the reservation; its callback starts the transfer.
        return self._fail_store_if_expired(task, now, "reservation")

    def _start_transfer(
        self,
        task: _StoreTask,
        reserve_future: MessagingFuture[list[TransferChannelAddress]],
    ) -> L2StoreResult | None:
        """Write the task's objects into the buffers the peer reserved.

        Args:
            task: The store whose reservation was answered.
            reserve_future: The answered reservation request.

        Returns:
            A failure when the request failed or the peer reserved nothing,
            else None: the transfer is in flight.
        """
        try:
            addresses = reserve_future.result()
        except REQUEST_FAILURES:
            logger.exception(
                "P2P store to %s failed: the reservation request failed",
                self._config.peer_mq_server_url,
            )
            return L2StoreResult(False, 0)

        local_extents: list[tuple[int, int]] = []
        remote_addresses: list[TransferChannelAddress] = []
        for key, obj, addr in zip(task.keys, task.objects, addresses, strict=True):
            if addr.is_valid():
                task.reserved_keys.append(key)
                local_extents.append((obj.shm_offset, obj.shm_byte_length))
                remote_addresses.append(addr)
        if not task.reserved_keys:
            logger.debug(
                "P2P store to %s failed: the peer reserved none of %d keys",
                self._config.peer_mq_server_url,
                len(task.keys),
            )
            return L2StoreResult(False, 0)

        task.write_task_id = self._tc_client.submit_write(
            self._tc_context.get_transfer_channel_address(local_extents),
            remote_addresses,
        )
        return None

    def _check_write(
        self, task: _StoreTask, write_task_id: int, now: float
    ) -> L2StoreResult | None:
        """Once the transfer finished, tell the peer what to commit."""
        result = self._tc_client.query_status(write_task_id)
        if not result.is_finished():
            expired = self._fail_store_if_expired(task, now, "transfer")
            if expired is not None:
                self._req_client.p2p_finish_write(
                    task.reserved_keys, [False] * len(task.reserved_keys)
                )
            return expired

        # The peer keeps the keys that were written and drops the rest, but
        # the task only succeeds if every one of its keys made it.
        future = self._req_client.p2p_finish_write(
            task.reserved_keys, result.succeeded_mask
        )
        if len(task.reserved_keys) < len(task.keys) or not all(result.succeeded_mask):
            logger.debug(
                "P2P store to %s failed: %d of %d keys were written",
                self._config.peer_mq_server_url,
                sum(result.succeeded_mask),
                len(task.keys),
            )
            return L2StoreResult(False, 0)
        task.commit_future = future
        future.add_done_callback(self._on_commit_answered)
        return None

    def _check_commit(
        self,
        task: _StoreTask,
        commit_future: MessagingFuture[list[bool]],
        now: float,
    ) -> L2StoreResult | None:
        """Once the peer answered the commit, report the store's result."""
        if not commit_future.query():
            return self._fail_store_if_expired(task, now, "commit")
        try:
            committed = commit_future.result()
        except REQUEST_FAILURES:
            logger.exception(
                "P2P store to %s failed: the commit request failed",
                self._config.peer_mq_server_url,
            )
            return L2StoreResult(False, 0)
        if not all(committed):
            logger.warning(
                "P2P store to %s failed: the peer committed %d of %d keys",
                self._config.peer_mq_server_url,
                sum(committed),
                len(task.keys),
            )
            return L2StoreResult(False, 0)
        return L2StoreResult(True, sum(obj.shm_byte_length for obj in task.objects))

    def _fail_store_if_expired(
        self, task: _StoreTask, now: float, step: str
    ) -> L2StoreResult | None:
        """Return a failure when the store's deadline passed, else None."""
        if now <= task.deadline:
            return None
        logger.warning(
            "P2P store to %s timed out waiting for the %s",
            self._config.peer_mq_server_url,
            step,
        )
        return L2StoreResult(False, 0)

    def _process_lookup_result(
        self,
        keys: list[ObjectKey],
        addresses: list[TransferChannelAddress],
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
