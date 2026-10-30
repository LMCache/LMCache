# SPDX-License-Identifier: Apache-2.0
"""P2P L2 adapter: reads KV objects from a single peer cache server.

Lookups and unlocks are sent to the peer's P2P controller over RPC by
:class:`PeerL2AdapterBase`; :class:`P2PL2Adapter` pulls the located objects
from the peer's L1 over the transfer channel. A peer adapter never stores,
evicts, or deletes -- a peer's cache is read-only here.

Because neither the lookup RPC nor the transfer-channel read exposes a
completion fd, the lookup and load event fds are pulsed by the
``PeriodicEventNotifier`` singleton so the prefetch controller re-polls
``query_*`` periodically.

Thread model: the lookup / load / unlock calls all come from the single
prefetch-controller loop thread, and the store calls come from the
store-controller loop thread. The two paths share no mutable state (each owns
its own task-id counter and bookkeeping dicts), so these classes need no locks.
"""

# Standard
from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
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
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelClient,
    TransferChannelContext,
    get_transfer_channel_context,
)
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
    (:meth:`_process_lookup_result`) and implement the load interface.

    Constructor contract: the region hook runs inside this constructor,
    before the subclass's own ``__init__`` body, so a subclass sets every
    attribute its hook needs before calling ``super().__init__()`` and
    acquires nothing that needs releasing after it returns; a failure in the
    subclass after that point is not unwound here.

    Args:
        config: Peer RPC endpoint and lookup deadline.
    """

    def __init__(self, config: PeerL2AdapterConfigBase) -> None:
        super().__init__(max_capacity_bytes=0)
        self._config = config

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

        # Prefetch-loop-thread state (lookup / load).
        self._next_task_id: L2TaskId = 0
        self._lookup_tasks: dict[L2TaskId, _LookupTask] = {}

        # Store-loop-thread state (store no-op completions).
        self._next_store_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}

        self._closed = False

    @abstractmethod
    def _register_peer_region(self) -> None:
        """Make the configured peer's region readable by this adapter.

        Lifecycle hook called once during construction, after the RPC client
        and event fds exist and before serving requests. The adapter owns the
        resulting region access until ``close`` calls
        :meth:`_unregister_peer_region`. If it raises, everything acquired
        before it is released and construction fails.
        """

    @abstractmethod
    def _unregister_peer_region(self) -> None:
        """Release the region access acquired by :meth:`_register_peer_region`.

        Lifecycle hook called by ``close`` after controllers stop submissions
        and drain reads. If it raises, ``close`` leaves the adapter open,
        with its RPC resources, so teardown can be retried.
        """

    # --------------------
    # Event Fd Interface
    # --------------------

    def get_store_event_fd(self) -> int:
        """Return the fd signaled when a (no-op) store task completes."""
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the fd the periodic notifier pulses to re-poll lookups."""
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the fd the periodic notifier pulses to re-poll loads."""
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
        """Return and clear the store tasks completed since the last call."""
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
        """Ask the peer to read-lock every resident key among ``keys``.

        Submits the lookup RPC and waits briefly for the peer's task id; a
        submit timeout, or a closed adapter, records a failed task that the
        next query reports as an all-miss.

        Args:
            keys: Object keys to look up, in request order.
            group_layout_descs: Expected layout per object group, so the peer
                can size the objects.

        Returns:
            The local task id to poll with ``query_lookup_and_lock_result``.
        """
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
        """Poll a lookup once.

        Args:
            task_id: Id returned by ``submit_lookup_and_lock_task``.

        Returns:
            None while the peer has no result yet; otherwise the hit bitmap
            from :meth:`_process_lookup_result`, all-zero for a failed or
            timed-out lookup. The task is forgotten once a bitmap is returned.
        """
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
        stays open, so callers keep the unlock connection and can retry. Once
        the region is released the adapter counts as closed: the RPC and fd
        resources are closed best effort and a failure there is not retried.
        """
        if self._closed:
            return
        self._unregister_peer_region()
        self._closed = True
        self._resources.close()

    def report_status(self) -> dict[str, Any]:
        """Return the peer endpoint and in-flight lookup count."""
        return {
            "peer_mq_server_url": self._config.peer_mq_server_url,
            "in_flight_lookups": len(self._lookup_tasks),
        }

    @abstractmethod
    def _process_lookup_result(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        addresses: list[TransferChannelAddress],
        started_at: float,
    ) -> Bitmap:
        """Turn a completed lookup's addresses into a hit bitmap.

        Args:
            task_id: The local lookup task.
            keys: The keys looked up, in request order.
            addresses: One address per key; invalid for misses.
            started_at: ``time.monotonic()`` just before the lookup was sent,
                a lower bound on when the owner's read locks started.
        """


class P2PL2Adapter(PeerL2AdapterBase):
    """Copy objects from one peer's registered region over the transfer channel.

    Args:
        config: Peer RPC/transfer endpoints and lookup/load deadlines.
    """

    _config: P2PL2AdapterConfig
    # Set by _register_peer_region during the base constructor.
    _tc_context: TransferChannelContext
    _tc_client: TransferChannelClient

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
        """Read the located objects from the peer into local L1 buffers.

        Pairs each key's cached peer address with the destination object's
        L1 offset and submits one transfer-channel read. A key without a
        valid cached address records a failed task instead.

        Args:
            keys: Keys found by a completed lookup.
            objects: Reserved L1 destination objects, aligned with ``keys``.

        Returns:
            The local task id to poll with ``query_load_result``.
        """
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
        """Poll a load once.

        Args:
            task_id: Id returned by ``submit_load_task``.

        Returns:
            None while the read is in flight; otherwise one bit per key, set
            for the objects the transfer delivered, all-zero for a failed or
            timed-out load. The task is forgotten once a bitmap is returned.
        """
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
    # Status
    # --------------------

    def get_memory_tags(self) -> MemoryTags:
        """Return the memory tags of this adapter: a peer's DRAM L1."""
        return MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE)

    def report_status(self) -> dict[str, Any]:
        """Return peer endpoints and in-flight lookup/load counts."""
        return {
            **super().report_status(),
            "is_healthy": True,
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
