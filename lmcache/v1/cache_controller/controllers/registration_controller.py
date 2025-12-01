# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import threading
import time

# Third Party
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (
    DeRegisterMsg,
    HealthMsg,
    HealthRetMsg,
    HeartbeatMsg,
    QueryInstMsg,
    QueryInstRetMsg,
    QueryWorkerInfoMsg,
    QueryWorkerInfoRetMsg,
    RegisterMsg,
)
from lmcache.v1.cache_controller.observability import PrometheusLogger
from lmcache.v1.cache_controller.utils import WorkerInfo
from lmcache.v1.rpc_utils import (
    close_zmq_socket,
    get_zmq_context,
    get_zmq_socket,
)

logger = init_logger(__name__)


class RegistrationController:
    def __init__(self):
        # Mapping from `instance_id` -> `worker_ids`
        self.worker_mapping: dict[str, list[int]] = {}

        # Mapping from `(instance_id, worker_id)` -> `peer_init_url`
        # NOTE(Jiayi): `peer_init_url` is used for actual KV cache transfer(p2p),
        # It's not the lmcache_worker_url.
        # if p2p is not used, peer_init_url is None and not registered.
        self.peer_init_url_mapping: dict[tuple[str, int], str] = {}

        # Mapping from `(instance_id, worker_id)` -> `socket`
        self.socket_mapping: dict[tuple[str, int], zmq.asyncio.Socket] = {}

        # Mapping from `ip` -> `instance_id`
        self.instance_mapping: dict[str, str] = {}

        # Mapping from `(instance_id, worker_id)` -> `WorkerInfo`
        self.worker_info_mapping: dict[tuple[str, int], WorkerInfo] = {}

        # Lock for thread-safe access to shared data structures
        self._lock = threading.Lock()

        self._setup_metrics()

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.registered_workers_count.set_function(
                lambda: len(self.worker_info_mapping)
            )

    def post_init(self, kv_controller, cluster_executor):
        """
        Post initialization of the Registration Controller.
        """
        self.kv_controller = kv_controller
        self.cluster_executor = cluster_executor

    def get_socket(
        self, instance_id: str, worker_id: int
    ) -> Optional[zmq.asyncio.Socket]:
        """
        Get the socket for a given instance and worker ID.
        """
        socket = self.socket_mapping.get((instance_id, worker_id))
        if socket is None:
            logger.warning(f"Instance-worker {(instance_id, worker_id)} not registered")
        return socket

    def get_peer_init_url(self, instance_id: str, worker_id: int) -> Optional[str]:
        """
        Get the URL for a given instance and worker ID.
        """
        url = self.peer_init_url_mapping.get((instance_id, worker_id))
        if url is None:
            logger.warning(
                "Instance-worker %s not registered or P2P is not used",
                (instance_id, worker_id),
            )
        return url

    def get_workers(self, instance_id: str) -> list[int]:
        """
        Get worker ids given an instance id.
        """
        return self.worker_mapping.get(instance_id, [])

    async def get_instance_id(self, msg: QueryInstMsg) -> QueryInstRetMsg:
        """
        Get the instance id given an ip address.
        """
        ip = msg.ip
        event_id = msg.event_id
        instance_id = self.instance_mapping.get(ip)
        if instance_id is None:
            logger.warning(f"Instance not registered for IP {ip}")
            return QueryInstRetMsg(instance_id=None, event_id=event_id)
        return QueryInstRetMsg(instance_id=instance_id, event_id=event_id)

    async def register(self, msg: RegisterMsg) -> None:
        """
        Register a new instance-worker connection mapping.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        key = (instance_id, worker_id)
        ip = msg.ip
        port = msg.port
        url = f"{ip}:{port}"
        peer_init_url = msg.peer_init_url

        # Create socket outside of lock to avoid holding lock during I/O
        context = get_zmq_context()
        socket = get_zmq_socket(
            context,
            url,
            protocol="tcp",
            role=zmq.REQ,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )

        socket_to_close = None
        with self._lock:
            # Re-check for duplicates that may have occurred during socket creation
            if key in self.socket_mapping and key in self.worker_info_mapping:
                logger.warning(
                    "Instance-worker %s already registered by another thread, "
                    "skip registration",
                    key,
                )
                # Mark socket for closing outside the lock
                socket_to_close = socket
            else:
                # register the instance-worker
                assert key not in self.peer_init_url_mapping
                assert key not in self.socket_mapping
                assert key not in self.worker_info_mapping

                if peer_init_url is not None:
                    self.peer_init_url_mapping[key] = peer_init_url
                else:
                    logger.info(
                        "peer init url of %s is None, only register when p2p is used.",
                        key,
                    )

                self.instance_mapping[ip] = instance_id

                self.socket_mapping[key] = socket
                self.worker_info_mapping[key] = WorkerInfo(
                    instance_id,
                    worker_id,
                    ip,
                    port,
                    peer_init_url,
                    time.time(),
                    time.time(),
                )
                if instance_id not in self.worker_mapping:
                    self.worker_mapping[instance_id] = []

                # TODO(Jiayi): Use more efficient data structures
                self.worker_mapping[instance_id].append(worker_id)
                self.worker_mapping[instance_id].sort()

        # Close the unused socket outside the lock
        if socket_to_close is not None:
            close_zmq_socket(socket_to_close)
            return

        logger.info("Registered instance-worker %s with URL %s", key, url)

    async def deregister(self, msg: DeRegisterMsg) -> None:
        """
        Deregister an instance-worker connection mapping.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        ip = msg.ip

        with self._lock:
            self.instance_mapping.pop(ip, None)

            if instance_id in self.worker_mapping:
                try:
                    self.worker_mapping[instance_id].remove(worker_id)
                    if not self.worker_mapping[instance_id]:
                        del self.worker_mapping[instance_id]
                except ValueError:
                    # Worker already removed, this is fine (idempotent operation)
                    logger.debug(
                        "Worker %s already removed from instance %s",
                        worker_id,
                        instance_id,
                    )
            else:
                logger.warning("Instance %s not registered", instance_id)

            self.peer_init_url_mapping.pop((instance_id, worker_id), None)

            socket = None
            if (instance_id, worker_id) in self.socket_mapping:
                socket = self.socket_mapping.pop((instance_id, worker_id))
                logger.info("Deregistered instance-worker %s", (instance_id, worker_id))
            else:
                logger.warning(
                    "Instance-worker %s not registered", (instance_id, worker_id)
                )

            if (instance_id, worker_id) in self.worker_info_mapping:
                self.worker_info_mapping.pop((instance_id, worker_id))
            else:
                logger.warning(
                    "Instance-worker %s not registered", (instance_id, worker_id)
                )

        # Close socket and call kv_controller outside the lock to avoid deadlock
        if socket is not None:
            close_zmq_socket(socket)
            await self.kv_controller.deregister(instance_id, worker_id)

    async def health(self, msg: HealthMsg) -> HealthRetMsg:
        """
        Check the health of the lmcache worker.
        """
        return await self.cluster_executor.execute(
            "health",
            msg,
        )

    # TODO: add more worker info in heartbeat
    async def heartbeat(self, msg: HeartbeatMsg) -> None:
        """
        Heartbeat from lmcache worker.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        worker_key = (instance_id, worker_id)

        # Fast path: check without lock first (common case)
        worker_info = self.worker_info_mapping.get(worker_key)
        if worker_info is not None:
            # Just update timestamp - no lock needed for single field update
            worker_info.last_heartbeat_time = time.time()
        else:
            # Slow path: need to register
            logger.warning(
                "%s has not been registered, re-register the worker.", worker_key
            )
            await self.register(msg)

    async def query_worker_info(self, msg: QueryWorkerInfoMsg) -> QueryWorkerInfoRetMsg:
        """
        Query worker info.
        """
        event_id = msg.event_id
        worker_infos = []

        with self._lock:
            if msg.instance_id not in self.worker_mapping:
                logger.warning(f"instance {msg.instance_id} not registered.")
            else:
                worker_ids = msg.worker_ids
                if worker_ids is None or len(worker_ids) == 0:
                    worker_ids = self.worker_mapping[msg.instance_id]
                for worker_id in worker_ids:
                    worker_key = (msg.instance_id, worker_id)
                    if worker_key in self.worker_info_mapping:
                        worker_infos.append(self.worker_info_mapping[worker_key])
                    else:
                        logger.warning(f"worker {worker_key} not registered.")

        return QueryWorkerInfoRetMsg(event_id=event_id, worker_infos=worker_infos)
