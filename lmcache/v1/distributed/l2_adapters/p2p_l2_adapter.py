# SPDX-License-Identifier: Apache-2.0
"""P2P L2 adapter: reads KV objects from a single peer cache server.

Lookups and unlocks are sent to the peer's P2P controller over RPC; the
default implementation pulls objects over the transfer channel. CXL specializes
region registration and borrowing while sharing this control plane. The
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
from contextlib import ExitStack
from dataclasses import dataclass
import time

# Third Party
import zmq

# First Party
from lmcache.lmcache_native import Bitmap, PeriodicEventNotifier
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.transfer_channel import get_transfer_channel_context
from lmcache.v1.distributed.transfer_channel.api import MemoryRegionAddress
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


class P2PL2AdapterConfig(L2AdapterConfigBase):
    """Config for the P2P L2 adapter.

    Fields:
    - peer_rpc_url: Peer request server URL (lookup/unlock RPCs).
      The constructor and serialized config retain ``peer_mq_server_url``.
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
        self.peer_mq_server_url = peer_mq_server_url
        self.peer_transfer_channel_server_url = peer_transfer_channel_server_url
        self.lookup_timeout_s = lookup_timeout_s
        self.load_timeout_s = load_timeout_s

    @property
    def peer_rpc_url(self) -> str:
        """Return the lookup/unlock endpoint, independent of RPC transport."""
        return self.peer_mq_server_url

    @classmethod
    def from_dict(cls, d: dict) -> "P2PL2AdapterConfig":
        """Parse peer endpoints and lookup/load deadlines.

        Args:
            d: Config fields; ``peer_rpc_url`` takes precedence over the legacy
                ``peer_mq_server_url`` spelling when both are present.

        Returns:
            Validated peer adapter configuration.

        Raises:
            ValueError: If an endpoint is absent or a deadline is nonpositive.
        """
        peer_mq_server_url = d.get("peer_rpc_url", d.get("peer_mq_server_url"))
        if not isinstance(peer_mq_server_url, str) or not peer_mq_server_url:
            raise ValueError("peer_rpc_url must be a non-empty string")

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
        """Return supported peer configuration fields and deadline defaults."""
        return (
            "P2P L2 adapter config fields:\n"
            "- peer_rpc_url (str): peer request server URL (required; "
            "peer_mq_server_url is also accepted)\n"
            "- peer_transfer_channel_server_url (str): the peer's transfer channel "
            "server url (required)\n"
            "- lookup_timeout_s (float): lookup result deadline in seconds "
            "(optional, default 10)\n"
            "- load_timeout_s (float): load deadline in seconds "
            "(optional, default 10)"
        )


class P2PL2Adapter(L2AdapterInterface):
    """Read one peer region through shared registration and lookup lifecycles.

    Construction registers the region described by ``config``; the adapter
    itself scopes returned offsets to that region. ``close`` unregisters it
    after the controllers drain reads. Subclasses specialize registration and
    retrieval without adding a separate region manager.

    Args:
        config: Peer RPC/transfer endpoints and lookup/load deadlines.
    """

    def __init__(self, config: P2PL2AdapterConfig) -> None:
        super().__init__(max_capacity_bytes=0)
        self._config = config

        with ExitStack() as resources:
            self._req_client: RequestClient = RequestClientFactory.create(
                config.peer_rpc_url,
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
            self.register_peer_region()
            self._resources = resources.pop_all()

        # Prefetch-loop-thread state (lookup / load).
        self._next_task_id: L2TaskId = 0
        self._lookup_tasks: dict[L2TaskId, _LookupTask] = {}
        self._load_tasks: dict[L2TaskId, _LoadTask] = {}
        self._remote_addresses: dict[ObjectKey, MemoryRegionAddress] = {}

        # Store-loop-thread state (store no-op completions).
        self._next_store_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}

        self._closed = False

    def register_peer_region(self) -> None:
        """Connect to the configured peer and import its registered region.

        Lifecycle hook called once during construction, before serving requests.
        Uses the existing transfer-channel handshake; CXL overrides this with
        local mapping and GPU registration. The adapter owns the resulting
        region access until ``close`` calls ``unregister_peer_region``.

        Raises:
            RuntimeError: If the transfer context is unavailable or setup fails.
        """
        self._tc_context = get_transfer_channel_context()
        self._tc_client = self._tc_context.get_transfer_channel_client(
            self._config.peer_transfer_channel_server_url
        )

    def unregister_peer_region(self) -> None:
        """Release the configured peer's registration through its transfer context.

        Lifecycle hook called by ``close`` after controllers stop submissions
        and drain reads. Subclasses must reject teardown while borrowed views
        still need the region, leaving RPC resources available for release.
        """
        self._tc_context.remove_transfer_channel_client(
            self._config.peer_transfer_channel_server_url
        )

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
        started_at = time.monotonic()

        if self._closed:
            self._lookup_tasks[task_id] = _LookupTask(
                keys=keys,
                remote_task_id=-1,
                deadline=0.0,
                started_at=started_at,
                failed=True,
            )
            return task_id

        future = self._req_client.p2p_lookup_and_lock(keys, group_layout_descs)
        failed = False
        remote_task_id = -1
        try:
            remote_task_id = future.result(timeout=_LOOKUP_RPC_TIMEOUT_S)
        except TimeoutError:
            logger.warning(
                "P2P lookup submit to %s timed out; treating as a miss",
                self._config.peer_rpc_url,
            )
            failed = True

        self._lookup_tasks[task_id] = _LookupTask(
            keys=keys,
            remote_task_id=remote_task_id,
            deadline=time.monotonic() + self._config.lookup_timeout_s,
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
        """Release peer reservations and discard their cached transfer addresses.

        Args:
            keys: Object keys to unlock. Empty input is a no-op.
            wait: Wait for the RPC acknowledgment before returning. Borrowing
                adapters use this on their release worker so draining waits
                for queued unlocks; ordinary prefetch remains nonblocking.

        Returns:
            None.

        Raises:
            TimeoutError: If ``wait`` is True and the RPC acknowledgment times out.
            Exception: If the transport fails to submit or acknowledge the RPC.
        """
        if not keys:
            return
        future = self._req_client.p2p_unlock_objects(keys)
        for key in keys:
            self._remote_addresses.pop(key, None)
        if wait:
            future.result(timeout=_LOOKUP_RPC_TIMEOUT_S)

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

        remote_addresses: list[MemoryRegionAddress] = []
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

    def close(self) -> None:
        """Unregister the drained peer region, then close shared RPC resources.

        Idempotent after success. If region teardown fails, callers may finish
        outstanding reads and retry without losing the unlock connection.

        Raises:
            RuntimeError: If a borrowing adapter still has live views.
            BufferError: If a mapped region still has exported tensor buffers.
        """
        if self._closed:
            return
        self.unregister_peer_region()
        self._closed = True
        self._resources.close()

    def report_status(self) -> dict:
        return {
            "is_healthy": True,
            "type": "P2PL2Adapter",
            "peer_rpc_url": self._config.peer_rpc_url,
            "peer_mq_server_url": self._config.peer_rpc_url,
            "peer_transfer_channel_server_url": (
                self._config.peer_transfer_channel_server_url
            ),
            "in_flight_lookups": len(self._lookup_tasks),
            "in_flight_loads": len(self._load_tasks),
        }

    def _process_lookup_result(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        addresses: list[MemoryRegionAddress],
        started_at: float,
    ) -> Bitmap:
        """Retain copy addresses; borrowing subclasses retain reservations instead.

        ``started_at`` precedes lookup submission, allowing subclasses to bound
        remote TTLs without a second lookup timer or task dictionary.
        """
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
