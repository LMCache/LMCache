# SPDX-License-Identifier: Apache-2.0
"""P2P L2 adapter: reads KV objects from a single peer cache server.

Lookups and unlocks are sent to the peer's P2P controller over the MQ; the
located objects are pulled from the peer's L1 over the transfer channel. The
adapter never stores, evicts, or deletes -- a peer's cache is read-only here.

Because neither the lookup RPC nor the transfer-channel read exposes a
completion fd, the lookup and load event fds are pulsed by the
``PeriodicEventNotifier`` singleton so the prefetch controller re-polls
``query_*`` periodically.
"""

# Standard
from dataclasses import dataclass
import threading
import time

# Third Party
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap, PeriodicEventNotifier
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.transfer_channel import get_transfer_channel_context
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.platform import HAS_EVENTFD, create_event_notifier

logger = init_logger(__name__)

_LOOKUP_RPC_TIMEOUT_S = 3.0
_PERIODIC_NOTIFIER_INTERVAL_MS = 5
_NOOP_STORE_TASK_ID: L2TaskId = -1


@dataclass
class _LookupTask:
    keys: list[ObjectKey]
    remote_task_id: int
    deadline: float
    failed: bool = False


@dataclass
class _LoadTask:
    keys: list[ObjectKey]
    read_task_id: int
    deadline: float


class P2PL2AdapterConfig(L2AdapterConfigBase):
    """Config for the P2P L2 adapter.

    Fields:
    - peer_mq_server_url: ZMQ url of the peer's MQ server (lookup/unlock RPCs).
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
            "- peer_mq_server_url (str): ZMQ url of the peer's MQ server (required)\n"
            "- peer_transfer_channel_server_url (str): the peer's transfer channel "
            "server url (required)\n"
            "- lookup_timeout_s (float): lookup result deadline in seconds "
            "(optional, default 10)\n"
            "- load_timeout_s (float): load deadline in seconds "
            "(optional, default 10)"
        )


class P2PL2Adapter(L2AdapterInterface):
    """L2 adapter that reads KV objects from a single peer cache server."""

    def __init__(self, config: P2PL2AdapterConfig) -> None:
        super().__init__(max_capacity_bytes=0)
        self._config = config

        self._mq_client = MessageQueueClient(
            config.peer_mq_server_url, zmq.Context.instance()
        )
        self._tc_context = get_transfer_channel_context()
        self._tc_client = self._tc_context.get_transfer_channel_client(
            config.peer_transfer_channel_server_url
        )

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        PeriodicEventNotifier.create(
            interval_ms=_PERIODIC_NOTIFIER_INTERVAL_MS, use_eventfd=HAS_EVENTFD
        )
        notifier = PeriodicEventNotifier.get()
        if notifier is None:
            raise RuntimeError("PeriodicEventNotifier is unavailable after create()")
        self._notifier = notifier
        self._notifier.register_fd(self._lookup_efd.fileno())
        self._notifier.register_fd(self._load_efd.fileno())

        self._next_task_id: L2TaskId = 0
        self._lookup_tasks: dict[L2TaskId, _LookupTask] = {}
        self._load_tasks: dict[L2TaskId, _LoadTask] = {}
        self._remote_addresses: dict[ObjectKey, TransferChannelAddress] = {}
        self._closed = False
        self._lock = threading.Lock()

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
        """No-op. The store policy never routes store tasks to a P2P adapter."""
        return _NOOP_STORE_TASK_ID

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """No-op. The store event fd is never signaled."""
        return {}

    # --------------------
    # Lookup and Lock Interface
    # --------------------

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> L2TaskId:
        with self._lock:
            task_id = self._next_task_id_locked()
            closed = self._closed

        if closed:
            with self._lock:
                self._lookup_tasks[task_id] = _LookupTask(
                    keys=keys, remote_task_id=-1, deadline=0.0, failed=True
                )
            return task_id

        future = self._mq_client.submit_request(
            RequestType.P2P_LOOKUP_AND_LOCK,
            [keys, layout_desc],
            get_response_class(RequestType.P2P_LOOKUP_AND_LOCK),
        )
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

        with self._lock:
            self._lookup_tasks[task_id] = _LookupTask(
                keys=keys,
                remote_task_id=remote_task_id,
                deadline=time.monotonic() + self._config.lookup_timeout_s,
                failed=failed,
            )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            task = self._lookup_tasks.get(task_id)
            if task is None:
                return None
            if task.failed:
                del self._lookup_tasks[task_id]
                return Bitmap(len(task.keys))
            expired = time.monotonic() > task.deadline
            remote_task_id = task.remote_task_id
            keys = task.keys

        if expired:
            with self._lock:
                self._lookup_tasks.pop(task_id, None)
            logger.warning("P2P lookup task %d timed out; treating as a miss", task_id)
            return Bitmap(len(keys))

        future = self._mq_client.submit_request(
            RequestType.P2P_QUERY_LOOKUP_RESULTS,
            [remote_task_id],
            get_response_class(RequestType.P2P_QUERY_LOOKUP_RESULTS),
        )
        try:
            addresses = future.result(timeout=_LOOKUP_RPC_TIMEOUT_S)
        except TimeoutError:
            return None

        if addresses is None:
            return None

        bitmap = Bitmap(len(keys))
        with self._lock:
            for i, (key, addr) in enumerate(zip(keys, addresses, strict=True)):
                if addr.is_valid():
                    bitmap.set(i)
                    self._remote_addresses[key] = addr
            self._lookup_tasks.pop(task_id, None)
        return bitmap

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        if not keys:
            return
        self._mq_client.submit_request(
            RequestType.P2P_UNLOCK_OBJECTS,
            [keys],
            get_response_class(RequestType.P2P_UNLOCK_OBJECTS),
        )
        with self._lock:
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
        local_addresses = self._tc_context.get_transfer_channel_address(
            [(obj.shm_offset, obj.shm_byte_length) for obj in objects]
        )
        with self._lock:
            task_id = self._next_task_id_locked()
            remote_addresses = [self._remote_addresses[key] for key in keys]

        read_task_id = self._tc_client.submit_read(local_addresses, remote_addresses)

        with self._lock:
            self._load_tasks[task_id] = _LoadTask(
                keys=keys,
                read_task_id=read_task_id,
                deadline=time.monotonic() + self._config.load_timeout_s,
            )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            task = self._load_tasks.get(task_id)
            if task is None:
                return None
            expired = time.monotonic() > task.deadline
            read_task_id = task.read_task_id
            keys = task.keys

        if expired:
            with self._lock:
                self._load_tasks.pop(task_id, None)
            logger.warning("P2P load task %d timed out; treating as a failure", task_id)
            return Bitmap(len(keys))

        result = self._tc_client.query_read_status(read_task_id)
        if not result.is_finished():
            return None

        bitmap = Bitmap(len(keys))
        for i, succeeded in enumerate(result.succeeded_mask):
            if succeeded:
                bitmap.set(i)

        with self._lock:
            self._load_tasks.pop(task_id, None)
        return bitmap

    # --------------------
    # Lifecycle / status
    # --------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._notifier.unregister_fd(self._lookup_efd.fileno())
        self._notifier.unregister_fd(self._load_efd.fileno())
        self._mq_client.close()
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    def report_status(self) -> dict:
        with self._lock:
            return {
                "is_healthy": True,
                "type": "P2PL2Adapter",
                "peer_mq_server_url": self._config.peer_mq_server_url,
                "peer_transfer_channel_server_url": (
                    self._config.peer_transfer_channel_server_url
                ),
                "in_flight_lookups": len(self._lookup_tasks),
                "in_flight_loads": len(self._load_tasks),
            }

    # --------------------
    # Helpers
    # --------------------

    def _next_task_id_locked(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id


# Self-register config type and adapter factory
register_l2_adapter_type("p2p", P2PL2AdapterConfig)


def _create_p2p_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    """Create a P2PL2Adapter from config (l1_memory_desc is unused)."""
    return P2PL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("p2p", _create_p2p_adapter)
