# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import asyncio
import threading
import time

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)

logger = init_logger(__name__)


# Global shared data store for simulating cross-process data transfer
_GLOBAL_DATA_STORE: dict[str, dict[int, torch.Tensor]] = {}


class PySocketMsgBase(msgspec.Struct, tag=True):
    """Base class for all py-socket-related messages"""

    pass


class PySocketInitRequest(PySocketMsgBase):
    local_id: str


class PySocketMemRegRequest(PySocketMsgBase):
    remote_id: str
    local_id: str


class PySocketInitResponse(PySocketMsgBase):
    remote_id: str


class PySocketMemRegResponse(PySocketMsgBase):
    status: str


PySocketMsg = Union[
    PySocketInitRequest,
    PySocketInitResponse,
    PySocketMemRegRequest,
    PySocketMemRegResponse,
]


class PySocketChannel(BaseTransferChannel):
    """
    Python socket-based transfer channel for testing purposes.
    This channel simulates data transfer using CPU memory and simple sockets.
    """

    def __init__(
        self,
        async_mode: bool = False,
        **kwargs,
    ):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "align_bytes" in kwargs
        assert "tp_rank" in kwargs
        assert "peer_init_url" in kwargs

        self.role = kwargs["role"]
        self.buffer_ptr = kwargs["buffer_ptr"]
        self.buffer_size = kwargs["buffer_size"]
        self.align_bytes = kwargs["align_bytes"]
        self.tp_rank = kwargs["tp_rank"]

        # Used for P2P
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self.remote_connections: dict[str, dict] = {}

        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        self._init_side_channels()

    ############################################################
    # Initialization functions
    ############################################################
    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        raise NotImplementedError("Sync mode not supported in PySocketChannel")

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        # Initialize temporary socket for initialization
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        # Build and send init request
        py_socket_init_req = PySocketInitRequest(local_id=local_id)
        await init_tmp_socket.send(msgspec.msgpack.encode(py_socket_init_req))

        # Wait remote id and register remote connection
        py_socket_init_resp_bytes = await init_tmp_socket.recv()
        py_socket_init_resp = msgspec.msgpack.decode(
            py_socket_init_resp_bytes, type=PySocketMsg
        )
        remote_id = py_socket_init_resp.remote_id

        # Register remote memory
        py_socket_mem_reg_req = PySocketMemRegRequest(
            remote_id=remote_id,
            local_id=local_id,
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(py_socket_mem_reg_req))
        py_socket_mem_reg_resp_bytes = await init_tmp_socket.recv()
        _ = msgspec.msgpack.decode(py_socket_mem_reg_resp_bytes, type=PySocketMsg)

        # Store connection info
        self.remote_connections[peer_id] = {
            "remote_id": remote_id,
            "peer_init_url": peer_init_url,
        }

        # Initialize remote data store for this peer
        if peer_id not in _GLOBAL_DATA_STORE:
            _GLOBAL_DATA_STORE[peer_id] = {}

        # Send side message if any
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    def _init_side_channels(self):
        if self.peer_init_url is None:
            return

        if self.async_mode:
            # Start listening coroutine for initialization side channel
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            # Start listening thread for initialization side channel
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

    def _handle_init_msg(
        self, req: Union[PySocketMsg, InitSideMsgBase]
    ) -> Union[PySocketMsg, InitSideRetMsgBase]:
        resp: Union[PySocketMsg, InitSideRetMsgBase]
        if isinstance(req, PySocketInitRequest):
            _ = req.local_id

            resp = PySocketInitResponse(
                remote_id=self.peer_init_url,
            )

            logger.info("Replying initialization response")

        elif isinstance(req, PySocketMemRegRequest):
            # Store remote connection info
            self.remote_connections[req.local_id] = {
                "remote_id": req.remote_id,
            }

            # Initialize remote data store
            if req.local_id not in _GLOBAL_DATA_STORE:
                _GLOBAL_DATA_STORE[req.local_id] = {}

            resp = PySocketMemRegResponse(
                status="ok",
            )

            logger.info("Replying mem register response")
        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _init_loop(self):
        # Initialize initialization side channels
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)

        while self.running:
            try:
                req_bytes = self.init_side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(
                    req_bytes, type=Union[PySocketMsg, SideMsg]
                )

                resp = self._handle_init_msg(req)

                self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self):
        # Initialize initialization side channels
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)
        logger.info("Starting async initialization loop")

        while self.running:
            try:
                req_bytes = await self.init_side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(
                    req_bytes, type=Union[PySocketMsg, SideMsg]
                )

                resp = self._handle_init_msg(req)

                await self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    await asyncio.sleep(0.01)

    ############################################################
    # Utility functions
    ############################################################

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        local_indices = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                local_indices.append(mem_obj.meta.address)
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in PySocket channel"
            )
        return local_indices

    ############################################################
    # Send/Recv functions
    ############################################################

    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    ############################################################
    # Read/Write functions
    ############################################################

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError("Sync mode not supported in PySocketChannel")

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError("Sync mode not supported in PySocketChannel")

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the channel.
        """
        assert transfer_spec is not None

        receiver_id = transfer_spec["receiver_id"]
        remote_indexes = transfer_spec["remote_indexes"]

        # Simulate data transfer by copying tensors
        for obj, remote_idx in zip(objects, remote_indexes, strict=True):
            if isinstance(obj, MemoryObj) and obj.tensor is not None:
                # Store a copy of the tensor data
                if receiver_id not in _GLOBAL_DATA_STORE:
                    _GLOBAL_DATA_STORE[receiver_id] = {}
                _GLOBAL_DATA_STORE[receiver_id][remote_idx] = obj.tensor.clone()

        # Simulate transfer delay
        await asyncio.sleep(0.001)
        return len(objects)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the channel.
        """
        assert transfer_spec is not None

        sender_id = transfer_spec["sender_id"]
        remote_indexes = transfer_spec["remote_indexes"]

        # Simulate data transfer by copying from remote store
        for buf, remote_idx in zip(buffers, remote_indexes, strict=True):
            if isinstance(buf, MemoryObj) and buf.tensor is not None:
                if (
                    sender_id in _GLOBAL_DATA_STORE
                    and remote_idx in _GLOBAL_DATA_STORE[sender_id]
                ):
                    # Copy data from remote store
                    buf.tensor.copy_(_GLOBAL_DATA_STORE[sender_id][remote_idx])

        # Simulate transfer delay
        await asyncio.sleep(0.001)
        return len(buffers)

    ############################################################
    # Cleanup-related functions
    ############################################################

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
