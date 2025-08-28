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
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.rpc_utils import (
    close_zmq_socket,
    get_ip,
    get_zmq_context,
    get_zmq_socket,
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

        assert config.controller_url is not None

        self.push_socket = get_zmq_socket(
            self.context,
            config.controller_url,
            protocol="tcp",
            role=zmq.PUSH,  # type: ignore[attr-defined]
            bind_or_connect="connect",
        )

        # TODO(Jiayi): Make this less hard-coded
        lmcache_worker_port = config.lmcache_worker_port
        assert lmcache_worker_port is not None
        # TODO(Jiayi): Make this port assignment smarter
        lmcache_worker_port += self.worker_id

        self.lmcache_worker_internal_url = f"*:{lmcache_worker_port}"
        self.lmcache_worker_ip = get_ip()
        self.lmcache_worker_port = lmcache_worker_port

        self.distributed_url = config.distributed_url

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
                distributed_url=self.distributed_url,
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

    def put_msg(self, msg: WorkerMsg):
        """
        Put a message into the message queue.
        """
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
                        distributed_url=self.distributed_url,
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
                        assert old_position[1] == "disk"
                        assert new_position[1] == "cpu"
                        assert do_copy

                        # TODO(Jiayi): We need to align prefetch and move.
                        logger.debug("Executing prefetch operation.")
                        self.lmcache_engine.prefetch(tokens)
                        num_tokens = 0
                    else:
                        assert self.lmcache_engine.distributed_server is not None
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
