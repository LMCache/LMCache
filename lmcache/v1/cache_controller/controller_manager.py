# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import asyncio
import json

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.controllers import KVController, RegistrationController
from lmcache.v1.cache_controller.executor import LMCacheClusterExecutor
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_socket,
)

from lmcache.v1.cache_controller.message import (  # isort: skip
    CheckFinishMsg,
    ClearMsg,
    CompressMsg,
    DecompressMsg,
    DeRegisterMsg,
    HealthMsg,
    HeartbeatMsg,
    KVAdmitMsg,
    KVEvictMsg,
    LookupMsg,
    MoveMsg,
    Msg,
    MsgBase,
    OrchMsg,
    OrchRetMsg,
    PinMsg,
    QueryInstMsg,
    RegisterMsg,
    WorkerMsg,
)

logger = init_logger(__name__)

# TODO(Jiayi): Need to align the message types. For example,
# a controller should take in an control message and return
# a control message.


class LMCacheControllerManager:
    def __init__(self, controller_url: str):
        self.zmq_context = get_zmq_context()
        self.controller_url = controller_url
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
        self.controller_socket = get_zmq_socket(
            self.zmq_context,
            self.controller_url,
            protocol="tcp",
            role=zmq.PULL,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )
        self.kv_controller = KVController()
        self.reg_controller = RegistrationController()

        # Cluster executor
        self.cluster_executor = LMCacheClusterExecutor(
            reg_controller=self.reg_controller,
        )

        # post initialization of controllers
        self.kv_controller.post_init(self.cluster_executor)
        self.reg_controller.post_init(
            kv_controller=self.kv_controller,
            cluster_executor=self.cluster_executor,
        )

        # self.loop = asyncio.new_event_loop()
        # self.thread = threading.Thread(target=self.loop.run_forever,
        #                               daemon=True)
        # self.thread.start()
        # asyncio.run_coroutine_threadsafe(self.start_all(), self.loop)

    async def handle_worker_message(self, msg: WorkerMsg) -> None:
        if isinstance(msg, HeartbeatMsg):
            await self.reg_controller.heartbeat(msg)
        elif isinstance(msg, RegisterMsg):
            await self.reg_controller.register(msg)
        elif isinstance(msg, DeRegisterMsg):
            await self.reg_controller.deregister(msg)
        elif isinstance(msg, KVAdmitMsg):
            await self.kv_controller.admit(msg)
        elif isinstance(msg, KVEvictMsg):
            await self.kv_controller.evict(msg)
        else:
            logger.error(f"Unknown worker message type: {msg}")

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
        else:
            logger.error(f"Unknown orchestration message type: {msg}")
            raise RuntimeError(f"Unknown orchestration message type: {msg}")

    async def handle_batched_request(self, socket) -> Optional[MsgBase]:
        while True:
            try:
                parts = await socket.recv_multipart()

                for part in parts:
                    # Parse message based on format
                    if part.startswith(b"{"):
                        # JSON format - typically from external systems like Mooncake
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
            except Exception as e:
                logger.error(f"Controller Manager error: {e}")

    async def start_all(self):
        await asyncio.gather(
            self.handle_batched_request(self.controller_socket),
        )
