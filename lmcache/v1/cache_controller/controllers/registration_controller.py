# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
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
from lmcache.v1.cache_controller.utils import RegistryTree
from lmcache.v1.rpc_utils import (
    close_zmq_socket,
    get_zmq_context,
    get_zmq_socket,
)

logger = init_logger(__name__)


class RegistrationController:
    def __init__(self):
        # Central registry tree managing all instances and workers
        self.registry = RegistryTree()
        self._setup_metrics()

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.registered_workers_count.set_function(
                lambda: len(self.registry.get_all_worker_infos())
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
        worker_node = self.registry.get_worker(instance_id, worker_id)
        if worker_node is None:
            logger.warning(
                "Instance-worker %s not registered", (instance_id, worker_id)
            )
            return None
        return worker_node.socket

    def get_peer_init_url(self, instance_id: str, worker_id: int) -> Optional[str]:
        """
        Get the URL for a given instance and worker ID.
        """
        worker_node = self.registry.get_worker(instance_id, worker_id)
        if worker_node is None:
            logger.warning(
                "Instance-worker %s not registered or P2P is not used",
                (instance_id, worker_id),
            )
            return None
        return worker_node.peer_init_url

    def get_workers(self, instance_id: str) -> list[int]:
        """
        Get worker ids given an instance id.
        """
        return self.registry.get_worker_ids(instance_id)

    async def get_instance_id(self, msg: QueryInstMsg) -> QueryInstRetMsg:
        """
        Get the instance id given an ip address.
        """
        ip = msg.ip
        event_id = msg.event_id
        instance_node = self.registry.get_instance_by_ip(ip)
        if instance_node is None:
            logger.warning("Instance not registered for IP %s", ip)
            return QueryInstRetMsg(instance_id=None, event_id=event_id)
        return QueryInstRetMsg(instance_id=instance_node.instance_id, event_id=event_id)

    async def register(self, msg: RegisterMsg) -> None:
        """
        Register a new instance-worker connection mapping.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        ip = msg.ip
        port = msg.port
        url = f"{ip}:{port}"

        # prevent duplicate registration
        existing_worker = self.registry.get_worker(instance_id, worker_id)
        if existing_worker is not None:
            logger.warning(
                "Instance-worker %s already registered, skip registration",
                (instance_id, worker_id),
            )
            return

        peer_init_url = msg.peer_init_url
        if peer_init_url is None:
            logger.info(
                "peer init url of %s is None, only register when p2p is used.",
                (instance_id, worker_id),
            )

        context = get_zmq_context()
        socket = get_zmq_socket(
            context,
            url,
            protocol="tcp",
            role=zmq.REQ,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )

        # Register worker in the tree
        self.registry.register_worker(
            instance_id=instance_id,
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=peer_init_url,
            socket=socket,
            registration_time=time.time(),
        )

        logger.info(
            "Registered instance-worker %s with URL %s", (instance_id, worker_id), url
        )

    async def deregister(self, msg: DeRegisterMsg) -> None:
        """
        Deregister an instance-worker connection mapping.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id

        worker_node = self.registry.deregister_worker(instance_id, worker_id)
        if worker_node is None:
            logger.warning(
                "Instance-worker %s not registered", (instance_id, worker_id)
            )
            return

        # Close socket
        if worker_node.socket is not None:
            close_zmq_socket(worker_node.socket)

        await self.kv_controller.deregister(instance_id, worker_id)
        logger.info("Deregistered instance-worker %s", (instance_id, worker_id))

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
        success = self.registry.update_heartbeat(instance_id, worker_id, time.time())
        if not success:
            logger.warning(
                "%s has not been registered, re-register the worker.",
                (instance_id, worker_id),
            )
            # re-register the worker
            await self.register(msg)

    async def query_worker_info(self, msg: QueryWorkerInfoMsg) -> QueryWorkerInfoRetMsg:
        """
        Query worker info.
        """
        event_id = msg.event_id
        worker_infos = []
        instance_node = self.registry.get_instance(msg.instance_id)
        if instance_node is None:
            logger.warning("instance %s not registered.", msg.instance_id)
        else:
            worker_ids = msg.worker_ids
            if worker_ids is None or len(worker_ids) == 0:
                worker_ids = instance_node.get_worker_ids()
            for worker_id in worker_ids:
                worker_node = instance_node.get_worker(worker_id)
                if worker_node is not None:
                    worker_infos.append(worker_node.to_worker_info(msg.instance_id))
                else:
                    logger.warning(
                        "worker %s not registered.", (msg.instance_id, worker_id)
                    )

        return QueryWorkerInfoRetMsg(event_id=event_id, worker_infos=worker_infos)
