# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import Optional, Union

import msgspec
import zmq

from lmcache.experimental.cache_controller.controllers import (
    KVController, RegistrationController)
from lmcache.experimental.cache_controller.executor import \
    LMCacheClusterExecutor
from lmcache.experimental.cache_controller.message import (  # noqa: E501
    ClearMsg, ClearRetMsg, ControlMsg, ControlRetMsg, DeRegisterMsg,
    KVAdmitMsg, KVEvictMsg, LookupMsg, Msg, MsgBase, OrchMsg, OrchRetMsg,
    RegisterMsg, WorkerMsg)
from lmcache.experimental.cache_controller.rpc_utils import (get_zmq_context,
                                                             get_zmq_socket)
from lmcache.logging import init_logger

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
        )
        self.kv_controller = KVController()
        self.reg_controller = RegistrationController()

        # Cluster executor
        self.cluster_executor = LMCacheClusterExecutor(
            reg_controller=self.reg_controller, )

        # post initialization of controllers
        self.kv_controller.post_init(self.cluster_executor)
        self.reg_controller.post_init(kv_controller=self.kv_controller,
                                      cluster_executor=self.cluster_executor)

        #self.loop = asyncio.new_event_loop()
        #self.thread = threading.Thread(target=self.loop.run_forever,
        #                               daemon=True)
        #self.thread.start()
        #asyncio.run_coroutine_threadsafe(self.start_all(), self.loop)

    # FIXME(Jiayi): the input and return type are weird
    async def issue_control_message(
        self, msg: Union[OrchMsg, ControlMsg]
    ) -> Optional[Union[OrchRetMsg, ControlRetMsg]]:
        if isinstance(msg, ClearMsg):
            return await self.kv_controller.clear(msg)
        else:
            logger.error("Unknown control or orchestration"
                         f"message type: {msg}")
            return None

    async def handle_worker_message(self, msg: WorkerMsg) -> None:
        if isinstance(msg, RegisterMsg):
            await self.reg_controller.register(msg)
        elif isinstance(msg, DeRegisterMsg):
            await self.reg_controller.deregister(msg)
        elif isinstance(msg, KVAdmitMsg):
            await self.kv_controller.admit(msg)
        elif isinstance(msg, KVEvictMsg):
            await self.kv_controller.evict(msg)
        else:
            logger.error(f"Unknown worker message type: {msg}")

    async def handle_orchestration_message(
            self, msg: OrchMsg) -> Optional[OrchRetMsg]:
        if isinstance(msg, LookupMsg):
            return await self.kv_controller.lookup(msg)
        elif isinstance(msg, ClearMsg):
            ret_msg = await self.issue_control_message(msg)
            assert isinstance(ret_msg, ClearRetMsg)
            return ret_msg
        else:
            logger.error(f"Unknown ochestration message type: {msg}")
            return None

    async def handle_batched_request(self, socket) -> Optional[MsgBase]:
        while True:
            try:
                parts = await socket.recv_multipart()

                for part in parts:
                    msg = msgspec.msgpack.decode(part, type=Msg)
                    logger.info(f"Received msg type: {type(msg)}")
                    if isinstance(msg, WorkerMsg):
                        await self.handle_worker_message(msg)
                    elif isinstance(msg, ControlMsg):
                        await self.issue_control_message(msg)
                    elif isinstance(msg, OrchMsg):
                        await self.handle_orchestration_message(msg)
                    else:
                        logger.error(f"Unknown message type: {type(msg)}")
            except Exception as e:
                logger.error(f"Controller Manager error: {e}")

    async def start_all(self):
        await asyncio.gather(
            self.handle_batched_request(self.controller_socket),
            #self.handle_batched_request(other socket),
        )
