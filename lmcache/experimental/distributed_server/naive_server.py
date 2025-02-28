import asyncio
import socket
import threading
import time
from typing import Optional

import torch

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.distributed_server.abstract_server import \
    DistributedServerInterface  # noqa: E501
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryFormat, MemoryObj)
from lmcache.experimental.protocol import (ClientMetaMessage, Constants,
                                           ServerMetaMessage)
from lmcache.experimental.storage_backend.storage_manager import StorageManager
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

# TODO(Jiayi): Logic related to "put" and "exists" is not implemented yet.
# Need to think when it's needed.

# TODO(Jiayi): Need to make `handle_get` async as blocking get from disk
# will affect the performance. Another simpler and cleaner option is to make
# `handle_get` always blocking but make disk loading always async.

# TODO(Jiayi): Need to find a way to make the code more concise.
# For example, consider reusing code from remote cache server?


class NaiveDistributedServer(DistributedServerInterface):

    def __init__(
        self,
        storage_manager: StorageManager,
        lookup_server: LookupServerInterface,
        memory_allocator: MemoryAllocatorInterface,
        loop: asyncio.AbstractEventLoop,
        config: LMCacheEngineConfig,
    ):

        self.storage_manager = storage_manager
        self.lookup_server = lookup_server
        self.memory_allocator = memory_allocator

        self.url = config.distributed_url
        assert self.url is not None
        host, port = self.url.split(":")
        self.host = host
        self.port = int(port)

        self.loop = loop
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start(), self.loop)

        self.async_socket_lock = asyncio.Lock()

    async def handle_get(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Handle get from the peer.
        This function is blocking for now but should be non-blocking.
        """
        memory_obj = self.storage_manager.get(key)
        return memory_obj

    def receive_all_client(
        self,
        meta: ServerMetaMessage,
        client_socket: socket.socket,
    ) -> Optional[MemoryObj]:
        received = 0
        n = meta.length

        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        memory_obj = self.memory_allocator.allocate(
            meta.shape,
            meta.dtype,
            meta.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        buffer = memory_obj.byte_array
        view = memoryview(buffer)

        while received < n:

            num_bytes = client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                return None
            received += num_bytes

        return memory_obj

    async def issue_get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Perform get from the peer.
        This function can be blocking for now.
        """
        # `url` has the format host:port
        host_and_port = self.lookup_server.lookup(key)
        if host_and_port is None:
            return None
        host, port = host_and_port

        # TODO(Jiayi): Cache the hot client sockets if possible.
        # For example, retrieving 100 chunks could create 100 the same
        # connection for 100 times.
        # However, too many live sockets could cause file descriptor exhaustion
        # (i.e., Too many open files).
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        logger.debug(f"Peer connection created at {host}:{port}")

        async with self.async_socket_lock:
            client_socket.sendall(
                ClientMetaMessage(Constants.CLIENT_GET, key, 0,
                                  MemoryFormat(1), torch.float16,
                                  torch.Size([0, 0, 0, 0])).serialize())

            data = client_socket.recv(ServerMetaMessage.packlength())

        meta = ServerMetaMessage.deserialize(data)
        if meta.code != Constants.SERVER_SUCCESS:
            return None

        async with self.async_socket_lock:
            memory_obj = self.receive_all_client(meta, client_socket)

        return memory_obj

    async def receive_all_server(self, reader, n):
        data = bytearray()
        while len(data) < n:
            packet = await reader.read(n - len(data))
            if not packet:
                return None  # Client disconnected
            data.extend(packet)
        return data

    async def handle_client(self, reader, writer):
        """
        Handle the client.
        """
        addr = writer.get_extra_info("peername")
        logger.info(f"Connected by {addr}")
        try:
            while True:
                header = await self.receive_all_server(
                    reader, ClientMetaMessage.packlength())
                if not header:
                    break
                meta = ClientMetaMessage.deserialize(header)

                match meta.command:

                    case Constants.CLIENT_GET:

                        t0 = time.perf_counter()

                        memory_obj = await self.handle_get(meta.key)

                        t1 = time.perf_counter()

                        if memory_obj is not None:
                            writer.write(
                                ServerMetaMessage(
                                    Constants.SERVER_SUCCESS,
                                    len(memory_obj.byte_array),
                                    memory_obj.get_memory_format(),
                                    memory_obj.get_dtype(),
                                    memory_obj.get_shape(),
                                ).serialize())
                            await writer.drain()

                            t2 = time.perf_counter()

                            writer.write(memory_obj.byte_array)
                            await writer.drain()
                            self.memory_allocator.ref_count_down(memory_obj)

                            t3 = time.perf_counter()
                            logger.info(f"Time to get data: {t1 - t0}, "
                                        f"time to send meta: {t2 - t1}, "
                                        f"time to send data: {t3 - t2}")
                        else:
                            writer.write(
                                ServerMetaMessage(Constants.SERVER_FAIL, 0,
                                                  MemoryFormat(1),
                                                  torch.float16,
                                                  torch.Size((0, 0, 0,
                                                              0))).serialize())
                            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self):
        """
        Start the server.
        """
        server = await asyncio.start_server(self.handle_client, self.host,
                                            self.port)
        addr = server.sockets[0].getsockname()
        logger.info(f"Server started at {addr}")

        async with server:
            await server.serve_forever()

    def close(self):
        """
        Close the server.
        """
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()
