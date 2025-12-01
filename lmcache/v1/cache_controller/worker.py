# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING
import asyncio
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    ClearWorkerMsg,
    ClearWorkerRetMsg,
    CompressWorkerMsg,
    CompressWorkerRetMsg,
    DecompressWorkerMsg,
    DecompressWorkerRetMsg,
    DeRegisterMsg,
    ErrorMsg,
    HealthWorkerMsg,
    HealthWorkerRetMsg,
    HeartbeatMsg,
    MoveWorkerMsg,
    MoveWorkerRetMsg,
    Msg,
    PinWorkerMsg,
    PinWorkerRetMsg,
    RegisterMsg,
    WorkerMsg,
    WorkerReqMsg,
    WorkerReqRetMsg,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.rpc_utils import (
    DEFAULT_SOCKET_RECV_TIMEOUT_MS,
    DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    close_zmq_socket,
    get_ip,
    get_zmq_context,
    get_zmq_socket,
    get_zmq_socket_with_timeout,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)


class LMCacheWorker:
    """
    LMCache Worker class to handle the execution of cache operations.
    This class is responsible for receiving requests from the executor and
    executing the corresponding operations on the LMCache engine.
    Each worker is associated with a specific LMCache instance and a worker id.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        lmcache_engine: "LMCacheEngine",
    ):
        # TODO (Jiayi): "instance_id" might not be needed anymore.
        # Please consider removing it.
        self.config = config
        self.lmcache_instance_id = config.lmcache_instance_id
        assert self.lmcache_instance_id is not None
        self.lmcache_engine = lmcache_engine
        self.worker_id = metadata.worker_id

        self.context = get_zmq_context()

        # Load timeout configurations from extra_config (in milliseconds)
        self.socket_recv_timeout_ms = config.get_extra_config_value(
            "worker_socket_recv_timeout_ms", DEFAULT_SOCKET_RECV_TIMEOUT_MS
        )
        self.socket_send_timeout_ms = config.get_extra_config_value(
            "worker_socket_send_timeout_ms", DEFAULT_SOCKET_SEND_TIMEOUT_MS
        )

        assert config.controller_pull_url is not None

        controller_pull_url = config.controller_pull_url
        self.push_socket = get_zmq_socket(
            self.context,
            controller_pull_url,
            protocol="tcp",
            role=zmq.PUSH,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )

        if config.controller_reply_url is not None:
            self.controller_rep_url = config.controller_reply_url
            self._create_req_socket()

        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        if not lmcache_worker_ids:
            # start lmcache worker on all ranks
            assert len(config.lmcache_worker_ports) == metadata.world_size
            lmcache_worker_port = config.lmcache_worker_ports[self.worker_id]
        else:
            # start lmcache worker on given worker ids
            assert len(lmcache_worker_ids) == len(config.lmcache_worker_ports)
            index = lmcache_worker_ids.index(self.worker_id)
            lmcache_worker_port = config.lmcache_worker_ports[index]

        self.lmcache_worker_internal_url = f"*:{lmcache_worker_port}"
        self.lmcache_worker_ip = get_ip()
        self.lmcache_worker_port = lmcache_worker_port

        self.p2p_init_url = None
        if config.enable_p2p:
            self.p2p_host = config.p2p_host
            self.p2p_init_port = config.p2p_init_ports[self.worker_id]
            self.p2p_init_url = f"{self.p2p_host}:{self.p2p_init_port}"

        self.reply_socket = get_zmq_socket(
            self.context,
            self.lmcache_worker_internal_url,
            protocol="tcp",
            role=zmq.REP,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )

        logger.info(f"Reply socket established at {self.lmcache_worker_internal_url}")

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start_all(), self.loop)

        self.msg_queue: asyncio.Queue[WorkerMsg] = asyncio.Queue()

        self.register()

    def register(self):
        """
        Register the lmcache worker with the controller.
        """
        assert self.lmcache_instance_id is not None
        logger.info(
            "Registering lmcache instance-worker: "
            f"{(self.lmcache_instance_id, self.worker_id)}"
        )
        self.put_msg(
            RegisterMsg(
                instance_id=self.lmcache_instance_id,
                worker_id=self.worker_id,
                ip=self.lmcache_worker_ip,
                port=self.lmcache_worker_port,
                peer_init_url=self.p2p_init_url,
            )
        )

    def deregister(self):
        """
        De-register the lmcache worker from the controller.
        """
        assert self.lmcache_instance_id is not None
        self.put_msg(
            DeRegisterMsg(
                instance_id=self.lmcache_instance_id,
                worker_id=self.worker_id,
                ip=self.lmcache_worker_ip,
                port=self.lmcache_worker_port,
            )
        )

    async def async_put_and_wait_msg(
        self,
        msg: WorkerReqMsg,
    ) -> WorkerReqRetMsg:
        """
        Send a message to the controller and wait for the response.
        """
        try:
            self.req_socket.send(msgspec.msgpack.encode(msg))
            serialized_ret_msg = await self.req_socket.recv()
            ret_msg = msgspec.msgpack.decode(serialized_ret_msg, type=Msg)
            return ret_msg
        except zmq.Again as e:
            logger.error("Timeout occurred, recreating socket. Error: %s", e)
            self._recreate_req_socket()
            return self._create_ret_msg(msg)
        except zmq.ZMQError as e:
            logger.error("ZMQ error occurred, recreating socket. Error: %s", e)
            self._recreate_req_socket()
            return self._create_ret_msg(msg)
        except Exception as e:
            logger.error("Error happens in lmcache worker req_socket. Error: %s", e)
            return self._create_ret_msg(msg)

    def _create_req_socket(self):
        self.req_socket = get_zmq_socket_with_timeout(
            self.context,
            self.controller_rep_url,
            "tcp",
            zmq.REQ,  # type: ignore[attr-defined]
            "connect",
            self.socket_recv_timeout_ms,
            self.socket_send_timeout_ms,
        )

    def _recreate_req_socket(self):
        try:
            self.req_socket.close(linger=0)
        except Exception as e:
            logger.error("Error closing req socket: %s", e)
        self._create_req_socket()

    def _create_ret_msg(self, msg: WorkerReqMsg) -> WorkerReqRetMsg:
        if isinstance(msg, BatchedP2PLookupMsg):
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")

    def put_msg(self, msg: WorkerMsg):
        """
        Put a message into the message queue.
        """
        # TODO(Jiayi): This might introduce ~0.05ms latency than
        # a normal function call.
        # Not sure how much overhead is blocking though.
        self.loop.call_soon_threadsafe(self.msg_queue.put_nowait, msg)

    async def batched_get_msg(self, max_bsz: int = 50) -> list[WorkerMsg]:
        """
        Get a batch of messages from the message queue.
        """
        batch = []

        # use blocking get for the first msg
        try:
            item = await self.msg_queue.get()
            batch.append(item)
        except asyncio.CancelledError:
            return batch  # shutdown path

        for _ in range(max_bsz - 1):
            try:
                item = self.msg_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break
        return batch

    async def heartbeat(self):
        enable_heartbeat = (
            self.config.lmcache_worker_heartbeat_time is not None
            and self.config.lmcache_worker_heartbeat_time > 0
        )
        if enable_heartbeat:
            logger.info(
                f"Start heartbeat in {self.lmcache_instance_id} : {self.worker_id}, "
                f"delay time: {self.config.lmcache_worker_heartbeat_delay_time}s, "
                f"heartbeat time: {self.config.lmcache_worker_heartbeat_time}s"
            )
            await asyncio.sleep(self.config.lmcache_worker_heartbeat_delay_time)
            while True:
                self.put_msg(
                    HeartbeatMsg(
                        instance_id=self.lmcache_instance_id,
                        worker_id=self.worker_id,
                        ip=self.lmcache_worker_ip,
                        port=self.lmcache_worker_port,
                        peer_init_url=self.p2p_init_url,
                    )
                )
                await asyncio.sleep(self.config.lmcache_worker_heartbeat_time)

    async def push(self):
        while True:
            try:
                msgs = await self.batched_get_msg()
                logger.debug(f"Sending {len(msgs)} messages")
                self.push_socket.send_multipart(
                    [msgspec.msgpack.encode(msg) for msg in msgs]
                )

            except Exception as e:
                logger.error(f"Push error: {e}")

    async def handle_request(self):
        """
        Handle incoming requests (control msgs) from the controller.
        """
        while True:
            try:
                serialized_request = await self.reply_socket.recv()
                request = msgspec.msgpack.decode(serialized_request, type=Msg)
                logger.debug(f"Received message: {request}")
                if isinstance(request, MoveWorkerMsg):
                    tokens = request.tokens
                    old_position = request.old_position
                    new_position = request.new_position
                    do_copy = request.copy
                    worker_event_id = request.worker_event_id

                    # Intra node move
                    if new_position[0] == self.lmcache_worker_internal_url:
                        # TODO(Jiayi): currently we only support moving from
                        # local disk to local cpu.
                        assert old_position[1] == "LocalDiskBackend"
                        assert new_position[1] == "LocalCPUBackend"
                        assert do_copy

                        # TODO(Jiayi): We need to align prefetch and move.
                        logger.debug("Executing prefetch operation.")
                        raise NotImplementedError(
                            "Prefetch from controller is not implemented yet."
                        )
                    else:
                        assert new_position[1] == "LocalCPUBackend", (
                            "Only support moving to cpu for now."
                        )
                        logger.debug("Executing cross-node move operation.")
                        num_tokens = self.lmcache_engine.move(
                            tokens=tokens,
                            old_position=old_position,
                            new_position=new_position,
                            event_id=worker_event_id,
                            do_copy=do_copy,
                        )

                    # TODO(Jiayi): LMCache needs to have an event tracking
                    # pool to enable more advanced control-plane optims.
                    # For now, we use a dummy `event_id`.
                    serialized_ret_msg = msgspec.msgpack.encode(
                        MoveWorkerRetMsg(num_tokens=num_tokens)
                    )
                elif isinstance(request, CompressWorkerMsg):
                    num_compressed_tokens = self.lmcache_engine.compress(
                        tokens=request.tokens,
                        method=request.method,
                        location=request.location,
                        event_id=request.worker_event_id,
                    )
                    serialized_ret_msg = msgspec.msgpack.encode(
                        CompressWorkerRetMsg(num_tokens=num_compressed_tokens)
                    )
                elif isinstance(request, DecompressWorkerMsg):
                    num_decompressed_tokens = self.lmcache_engine.decompress(
                        tokens=request.tokens,
                        method=request.method,
                        location=request.location,
                        event_id=request.worker_event_id,
                    )
                    serialized_ret_msg = msgspec.msgpack.encode(
                        DecompressWorkerRetMsg(num_tokens=num_decompressed_tokens)
                    )
                elif isinstance(request, PinWorkerMsg):
                    num_pinned_tokens = self.lmcache_engine.lookup(
                        tokens=request.tokens,
                        search_range=[request.location],
                        request_id=request.worker_event_id,
                        pin=True,
                    )
                    serialized_ret_msg = msgspec.msgpack.encode(
                        PinWorkerRetMsg(num_tokens=num_pinned_tokens)
                    )
                elif isinstance(request, ClearWorkerMsg):
                    num_cleared_tokens = self.lmcache_engine.clear(
                        locations=[request.location],
                    )
                    serialized_ret_msg = msgspec.msgpack.encode(
                        ClearWorkerRetMsg(num_tokens=num_cleared_tokens)
                    )
                elif isinstance(request, HealthWorkerMsg):
                    error_code = self.lmcache_engine.health()
                    serialized_ret_msg = msgspec.msgpack.encode(
                        HealthWorkerRetMsg(error_code=error_code)
                    )
                else:
                    logger.error(f"Unknown message: {request}")
                    serialized_ret_msg = msgspec.msgpack.encode(
                        ErrorMsg(error=f"Unknown message: {request}")
                    )

                await self.reply_socket.send(serialized_ret_msg)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                serialized_ret_msg = msgspec.msgpack.encode(
                    ErrorMsg(error=f"Worker error: {e}")
                )
                await self.reply_socket.send(serialized_ret_msg)

    async def start_all(self):
        try:
            logger.info(
                f"Starting lmcache worker {self.worker_id}"
                f"for instance {self.lmcache_instance_id}"
            )
            await asyncio.gather(
                self.push(),
                self.handle_request(),
                self.heartbeat(),
            )
        except Exception as e:
            logger.error(
                f"Instance {self.lmcache_instance_id}, "
                f"worker {self.worker_id} error: {e}"
            )

    def close(self):
        self.deregister()
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()
        close_zmq_socket(self.push_socket)
        close_zmq_socket(self.reply_socket)
