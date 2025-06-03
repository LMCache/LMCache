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

# Standard
from typing import Union
import asyncio

# Third Party
import msgspec
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (  # noqa: E501
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    ClearWorkerMsg,
    CompressMsg,
    CompressRetMsg,
    ErrorMsg,
    HealthMsg,
    HealthRetMsg,
    MoveMsg,
    MoveRetMsg,
    Msg,
    MsgBase,
    PinMsg,
    PinRetMsg,
)

logger = init_logger(__name__)


# NOTE (Jiayi): `LMCacheClusterExecutor` might need to be in different processes
# in the future for the sake of performance.
# NOTE (Jiayi): Also, consider scaling up the number of cluster executors
# in the future.
# TODO (Jiayi): need better error handling
class LMCacheClusterExecutor:
    """
    LMCache Cluster Executor class to handle the execution of cache operations.
    """

    def __init__(self, reg_controller):
        """
        Initialize the LMCache Executor with a cache instance.

        :param lmcache_instance_id: lmcache_instance_id
        """
        self.reg_controller = reg_controller

    async def clear(self, msg: ClearMsg) -> Union[ClearRetMsg, ErrorMsg]:
        """
        Execute a cache operation with error handling.
        """
        instance_id = msg.instance_id
        tokens = msg.tokens

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None
        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)
            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registeredfor instance {instance_id}"
                    )
                )
            sockets.append(socket)
            serialized_msg = msgspec.msgpack.encode(
                ClearWorkerMsg(
                    tokens=tokens,
                )
            )
            serialized_msgs.append(serialized_msg)
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        success = True
        for i, serialized_result in enumerate(serialized_results):
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            if success:
                success = result.success
        return ClearRetMsg(success=success)

    async def pin(self, msg: PinMsg) -> Union[PinRetMsg, ErrorMsg]:
        raise NotImplementedError

    async def compress(self, msg: CompressMsg) -> Union[CompressRetMsg, ErrorMsg]:
        raise NotImplementedError

    async def move(self, msg: MoveMsg) -> Union[MoveRetMsg, ErrorMsg]:
        raise NotImplementedError

    async def health(self, msg: HealthMsg) -> Union[HealthRetMsg, ErrorMsg]:
        raise NotImplementedError

    async def check_finish(
        self, msg: CheckFinishMsg
    ) -> Union[CheckFinishRetMsg, ErrorMsg]:
        raise NotImplementedError

    # TODO(Jiayi): need to make the types more specific
    async def execute(self, operation: str, msg: MsgBase) -> MsgBase:
        """
        Execute a cache operation with error handling.

        :param operation: The operation to execute
        (e.g., 'clear').
        :param msg: The message containing the operation details.
        :return: The result of the operation or an error message.
        """
        try:
            method = getattr(self, operation)
            return await method(msg)
        except AttributeError:
            return ErrorMsg(error=f"Operation '{operation}' is not supported.")
        except Exception as e:
            return ErrorMsg(error=str(e))

    async def execute_workers(
        self,
        sockets: list[zmq.asyncio.Socket],
        serialized_msgs: list[bytes],
    ) -> list[bytes]:
        """
        Execute a list of serialized messages on the given sockets.
        :param sockets: The list of sockets to send the messages to.
        :param serialized_msgs: The list of serialized messages to send.
        :return: A list of serialized results received from the sockets.
        """
        tasks = []
        for socket, serialized_msg in zip(sockets, serialized_msgs, strict=False):

            async def send_and_receive(s, msg):
                await s.send(msg)
                return await s.recv()

            tasks.append(send_and_receive(socket, serialized_msg))

        serialized_results = await asyncio.gather(*tasks)
        return serialized_results
