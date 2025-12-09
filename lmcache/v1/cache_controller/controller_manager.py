# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import asyncio
import json
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.controllers import KVController, RegistrationController
from lmcache.v1.cache_controller.executor import LMCacheClusterExecutor
from lmcache.v1.cache_controller.observability import (
    PrometheusLogger,
    SocketMetricsContext,
    SocketType,
)
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_socket,
)

from lmcache.v1.cache_controller.message import (  # isort: skip
    BatchedKVOperationMsg,
    BatchedP2PLookupMsg,
    CheckFinishMsg,
    ClearMsg,
    CompressMsg,
    DecompressMsg,
    DeRegisterMsg,
    ErrorMsg,
    HealthMsg,
    HeartbeatMsg,
    KVAdmitMsg,
    KVEvictMsg,
    LookupMsg,
    MoveMsg,
    Msg,
    MsgBase,
    OpType,
    OrchMsg,
    OrchRetMsg,
    PinMsg,
    QueryInstMsg,
    QueryWorkerInfoMsg,
    RegisterMsg,
    WorkerMsg,
    WorkerReqMsg,
    WorkerReqRetMsg,
)

logger = init_logger(__name__)

# TODO(Jiayi): Need to align the message types. For example,
# a controller should take in an control message and return
# a control message.


class LMCacheControllerManager:
    def __init__(
        self,
        controller_urls: dict[str, str],
        health_check_interval: int,
        lmcache_worker_timeout: int,
    ):
        # Initialize stats logger
        prometheus_labels = {
            "role": "controller",
        }
        PrometheusLogger.GetOrCreate(prometheus_labels)
        self.zmq_context = get_zmq_context()
        self.controller_urls = controller_urls
        # TODO(Jiayi): We might need multiple sockets if there are more
        # controllers. For now, we use a single socket to receive messages
        # for all controllers.
        # Similarly we might need more sockets to handle different control
        # messages. For now, we use one socket to handle all control messages.

        # TODO(Jiayi): Another thing is that we might need to decoupe the
        # interactions among `handle_worker_message`, `handle_control_message`
        # and `handle_orchestration_message`. For example, in
        # `handle_orchestration_message`, we might need to call
        # `issue_control_message`. This will make the system less concurrent.

        # Micro controllers
        self.controller_pull_socket = get_zmq_socket(
            self.zmq_context,
            self.controller_urls["pull"],
            protocol="tcp",
            role=zmq.PULL,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )

        if self.controller_urls["reply"] is not None:
            self.controller_reply_socket = get_zmq_socket(
                self.zmq_context,
                self.controller_urls["reply"],
                protocol="tcp",
                role=zmq.REP,  # type: ignore[attr-defined]
                bind_or_connect="bind",
            )
        self.kv_controller = KVController()
        self.reg_controller = RegistrationController()

        # Cluster executor
        self.cluster_executor = LMCacheClusterExecutor(
            reg_controller=self.reg_controller,
        )

        # post initialization of controllers
        self.kv_controller.post_init(
            reg_controller=self.reg_controller,
            cluster_executor=self.cluster_executor,
        )
        self.reg_controller.post_init(
            kv_controller=self.kv_controller,
            cluster_executor=self.cluster_executor,
        )
        self.health_check_interval = health_check_interval
        self.lmcache_worker_timeout = lmcache_worker_timeout

        if self.health_check_interval > 0:
            logger.info(
                "Start health check thread, interval: %s", self.health_check_interval
            )
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            self.thread.start()
            asyncio.run_coroutine_threadsafe(self.health_check(), self.loop)

        # Setup socket message count metrics
        self._setup_socket_metrics()

    async def handle_worker_message(self, msg: WorkerMsg) -> None:
        if isinstance(msg, HeartbeatMsg):
            await self.reg_controller.heartbeat(msg)
        elif isinstance(msg, RegisterMsg):
            await self.reg_controller.register(msg)
        elif isinstance(msg, DeRegisterMsg):
            await self.reg_controller.deregister(msg)
        elif isinstance(msg, BatchedKVOperationMsg):
            # Reconstruct full KV messages from lightweight operations
            instance_id = msg.instance_id
            worker_id = msg.worker_id
            location = msg.location
            for op in msg.operations:
                if op.op_type == OpType.ADMIT:
                    admit_msg = KVAdmitMsg(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        key=op.key,
                        location=location,
                        seq_num=op.seq_num,
                    )
                    self.kv_controller.check_sequence_number(admit_msg)
                    await self.kv_controller.admit(admit_msg)
                elif op.op_type == OpType.EVICT:
                    evict_msg = KVEvictMsg(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        key=op.key,
                        location=location,
                        seq_num=op.seq_num,
                    )
                    self.kv_controller.check_sequence_number(evict_msg)
                    await self.kv_controller.evict(evict_msg)
                else:
                    logger.error("Unknown operation type: %s", op.op_type)
        else:
            logger.error(f"Unknown worker message type: {msg}")

    async def handle_worker_req_message(self, msg: WorkerReqMsg) -> WorkerReqRetMsg:
        if isinstance(msg, BatchedP2PLookupMsg):
            ret_msg = await self.kv_controller.batched_p2p_lookup(msg)
        return ret_msg

    async def handle_orchestration_message(self, msg: OrchMsg) -> OrchRetMsg:
        if isinstance(msg, LookupMsg):
            return await self.kv_controller.lookup(msg)
        elif isinstance(msg, HealthMsg):
            return await self.reg_controller.health(msg)
        elif isinstance(msg, QueryInstMsg):
            return await self.reg_controller.get_instance_id(msg)
        elif isinstance(msg, ClearMsg):
            return await self.kv_controller.clear(msg)
        elif isinstance(msg, PinMsg):
            return await self.kv_controller.pin(msg)
        elif isinstance(msg, CompressMsg):
            return await self.kv_controller.compress(msg)
        elif isinstance(msg, DecompressMsg):
            return await self.kv_controller.decompress(msg)
        elif isinstance(msg, MoveMsg):
            return await self.kv_controller.move(msg)
        elif isinstance(msg, CheckFinishMsg):
            # FIXME(Jiayi): This `check_finish` thing
            # shouldn't be implemented in kv_controller.
            return await self.kv_controller.check_finish(msg)
        elif isinstance(msg, QueryWorkerInfoMsg):
            return await self.reg_controller.query_worker_info(msg)
        else:
            logger.error(f"Unknown orchestration message type: {msg}")
            raise RuntimeError(f"Unknown orchestration message type: {msg}")

    def _setup_socket_metrics(self):
        """Setup metrics for socket message counts."""
        # Initialize message counters for observability
        self.pull_socket_message_count = 0
        self.reply_socket_message_count = 0

        # Initialize active request counters
        self.pull_socket_active_requests = 0
        self.reply_socket_active_requests = 0

        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.pull_socket_message_count.set_function(
                lambda: self.pull_socket_message_count
            )
            prometheus_logger.reply_socket_message_count.set_function(
                lambda: self.reply_socket_message_count
            )

            # Socket queue/backlog metrics
            prometheus_logger.pull_socket_has_pending.set_function(
                lambda: self._check_socket_has_pending(self.controller_pull_socket)
            )
            if self.controller_urls["reply"] is not None:
                prometheus_logger.reply_socket_has_pending.set_function(
                    lambda: self._check_socket_has_pending(self.controller_reply_socket)
                )

            # Active request metrics
            prometheus_logger.pull_socket_active_requests.set_function(
                lambda: self.pull_socket_active_requests
            )
            prometheus_logger.reply_socket_active_requests.set_function(
                lambda: self.reply_socket_active_requests
            )

    def _check_socket_has_pending(self, socket) -> int:
        """Check if socket has pending messages.

        Returns:
            1 if socket has pending messages, 0 otherwise
        """
        try:
            events = socket.get(zmq.EVENTS)  # type: ignore[attr-defined]
            # Check if POLLIN flag is set (indicates readable/pending messages)
            has_pending = 1 if (events & zmq.POLLIN) else 0  # type: ignore[attr-defined]
            return has_pending
        except Exception as e:
            logger.error(f"Error checking socket pending status: {e}")
            return 0

    async def handle_batched_push_request(self, socket) -> Optional[MsgBase]:
        while True:
            parts = await socket.recv_multipart()
            part_count = len(parts)
            with SocketMetricsContext(self, SocketType.PULL, part_count):
                for part in parts:
                    # Parse message based on format
                    if part.startswith(b"{"):
                        # JSON format - typically from external systems
                        # like Mooncake
                        msg_dict = json.loads(part)
                        msg = msgspec.convert(msg_dict, type=Msg)
                    else:
                        # MessagePack format - internal LMCache communication
                        msg = msgspec.msgpack.decode(part, type=Msg)
                    if isinstance(msg, WorkerMsg):
                        await self.handle_worker_message(msg)

                    # FIXME(Jiayi): The abstraction of control messages
                    # might not be necessary.
                    # elif isinstance(msg, ControlMsg):
                    #    await self.issue_control_message(msg)
                    elif isinstance(msg, OrchMsg):
                        await self.handle_orchestration_message(msg)
                    else:
                        logger.error(f"Unknown message type: {type(msg)}")

    async def handle_batched_req_request(self, socket) -> Optional[MsgBase]:
        while True:
            part = await socket.recv()
            with SocketMetricsContext(self, SocketType.REPLY):
                # Parse message based on format
                if part.startswith(b"{"):
                    # JSON format - typically from external systems like Mooncake
                    msg_dict = json.loads(part)
                    msg = msgspec.convert(msg_dict, type=Msg)
                else:
                    # MessagePack format - internal LMCache communication
                    msg = msgspec.msgpack.decode(part, type=Msg)

                if isinstance(msg, WorkerReqMsg):
                    ret_msg = await self.handle_worker_req_message(msg)
                    await socket.send(msgspec.msgpack.encode(ret_msg))
                else:
                    logger.error(f"Unknown message type: {type(msg)}")
                    err_msg = ErrorMsg(error=f"Unknown message type: {type(msg)}")
                    await socket.send(msgspec.msgpack.encode(err_msg))

    async def health_check(self):
        while True:
            time.sleep(self.health_check_interval)
            worker_infos = self.reg_controller.registry.get_all_worker_infos()
            for worker_info in worker_infos:
                if (
                    time.time() - worker_info.last_heartbeat_time
                    > self.lmcache_worker_timeout
                ):
                    logger.warning(
                        "Worker %s_%s last heartbeat time: %s, "
                        "current time: %s, more than %s seconds",
                        worker_info.instance_id,
                        worker_info.worker_id,
                        worker_info.last_heartbeat_time,
                        time.time(),
                        self.lmcache_worker_timeout,
                    )
                    # Perform a full deregister to clean up all associated resources.
                    deregister_msg = DeRegisterMsg(
                        instance_id=worker_info.instance_id,
                        worker_id=worker_info.worker_id,
                        ip=worker_info.ip,
                        port=worker_info.port,
                    )
                    await self.reg_controller.deregister(deregister_msg)

    async def start_all(self):
        tasks = []
        if self.controller_urls["reply"] is not None:
            tasks.append(self.handle_batched_req_request(self.controller_reply_socket))
        tasks.append(self.handle_batched_push_request(self.controller_pull_socket))
        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
