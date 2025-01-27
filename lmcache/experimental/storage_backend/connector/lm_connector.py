import asyncio
import socket
import threading
from typing import List, Optional, no_type_check

import torch

from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryFormat, MemoryObj)
from lmcache.experimental.protocol import (ClientMetaMessage, Constants,
                                           ServerMetaMessage)
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

logger = init_logger(__name__)


# TODO: performance optimization for this class, consider using C/C++/Rust
# for communication + deserialization
class LMCServerConnector(RemoteConnector):

    def __init__(self, host, port, loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        # NOTE(Jiayi): According to Python documentation:
        # https://docs.python.org/3/library/asyncio-eventloop.html
        # In general, protocol implementations that use transport-based APIs
        # such as loop.create_connection() and loop.create_server() are faster
        # than implementations that work with sockets.
        # However, we use socket here as we need to use the socket.recv_into()
        # to reduce memory copy.

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        #loop.sock_recv_into(sock, buf)

        self.memory_allocator = memory_allocator
        self.loop = loop
        self.socket_lock = threading.Lock()

    # TODO(Jiayi): This should be an async function
    def receive_all(self, meta: ServerMetaMessage) -> Optional[MemoryObj]:
        received = 0
        shape = meta.shape
        dtype = meta.dtype
        n = meta.length

        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        #fmt = meta.fmt

        memory_obj = self.memory_allocator.allocate(
            shape,
            dtype,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        buffer = memory_obj.byte_array
        view = memoryview(buffer)

        while received < n:
            self.socket_lock.acquire()
            num_bytes = self.client_socket.recv_into(view[received:],
                                                     n - received)
            self.socket_lock.release()
            if num_bytes == 0:
                return None
            received += num_bytes

        return memory_obj

    def exists(self, key: CacheEngineKey) -> bool:
        logger.debug("Call to exists()!")

        self.socket_lock.acquire()
        self.client_socket.sendall(
            ClientMetaMessage(Constants.CLIENT_EXIST, key, 0,
                              MemoryFormat(1), torch.float16,
                              torch.Size([0, 0, 0])).serialize())
        response = self.client_socket.recv(ServerMetaMessage.packlength())
        self.socket_lock.release()

        return (ServerMetaMessage.deserialize(response).code ==
                Constants.SERVER_SUCCESS)

    async def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):

        logger.debug("Async call to set()!")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        await self.loop.sock_sendall(
            self.client_socket,
            ClientMetaMessage(Constants.CLIENT_PUT, key.to_string(),
                              len(kv_bytes), memory_format, kv_dtype,
                              kv_shape).serialize())

        await self.loop.sock_sendall(self.client_socket, kv_bytes)

        self.memory_allocator.ref_count_down(memory_obj)

    # TODO(Jiayi): This should be an async function
    @_lmcache_nvtx_annotate
    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:

        # TODO(Jiayi): the following send is a bit hacky
        # Please consider using another message type
        self.socket_lock.acquire()
        self.client_socket.sendall(
            ClientMetaMessage(Constants.CLIENT_GET, key, 0,
                              MemoryFormat(1), torch.float16,
                              torch.Size([0, 0, 0])).serialize())
        data = self.client_socket.recv(ServerMetaMessage.packlength())
        self.socket_lock.release()
        meta = ServerMetaMessage.deserialize(data)
        if meta.code != Constants.SERVER_SUCCESS:
            return None

        memory_obj = self.receive_all(meta)
        return memory_obj

    # TODO(Jiayi)
    @no_type_check
    def list(self) -> List[str]:
        pass

    def close(self):
        self.client_socket.close()
        logger.info("Closed the lmserver connection")
