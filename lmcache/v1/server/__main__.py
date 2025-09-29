# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import asyncio
import socket

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.server.storage_backend import CreateStorageBackend

logger = init_logger(__name__)


# we use event loop time-sharing instead of thread-base for less overhead
# still maintaining zero copy send/recv
# TODO: right now lots of python byte objects are created JIT, we can create
# a pool in the future to recycle
class LMCacheServer:
    def __init__(self, host, port, device, capacity):
        self.host = host
        self.port = port
        self.data_store = CreateStorageBackend(device, capacity)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen()
        self.server_socket.setblocking(False)

    async def recv_all(self, client_socket, nbytes: int):
        buf = bytearray(nbytes)
        # zero copy slicing
        view = memoryview(buf)
        cur_offset = 0
        while cur_offset < nbytes:
            bytes_received = await self.loop.sock_recv(
                client_socket, view[cur_offset:], nbytes - cur_offset
            )
            if bytes_received == 0:
                raise ConnectionError(
                    f"LMServer: client disconnected at {cur_offset} "
                    f"out of {nbytes} bytes"
                )
            cur_offset += bytes_received
        return buf

    async def handle_client(self, client_socket):
        try:
            while True:
                header = await self.recv_all(
                    client_socket, ClientMetaMessage.num_bytes()
                )
                if not header:
                    break
                meta = ClientMetaMessage.deserialize(header)
                match meta.command:
                    case ClientCommand.PUT:
                        data = await self.recv_all(client_socket, meta.length)
                        await self.data_store.put(meta, data)
                    case ClientCommand.GET:
                        lms_memory_obj = await self.data_store.get(meta.key)
                        if lms_memory_obj is not None:
                            await self.loop.sock_sendall(
                                client_socket,
                                ServerMetaMessage(
                                    ServerReturnCode.SUCCESS,
                                    lms_memory_obj.length,
                                    lms_memory_obj.fmt,
                                    lms_memory_obj.dtype,
                                    lms_memory_obj.shape,
                                ).serialize(),
                            )
                            await self.loop.sock_sendall(
                                client_socket, lms_memory_obj.data
                            )
                        else:
                            await self.loop.sock_sendall(
                                client_socket,
                                ServerMetaMessage(
                                    ServerReturnCode.FAIL,
                                    0,
                                    MemoryFormat(1),
                                    torch.float16,
                                    torch.Size((0, 0, 0, 0)),
                                ).serialize(),
                            )
                    case ClientCommand.EXIST | ClientCommand.EXISTS_PIN:
                        pin = meta.command == ClientCommand.EXISTS_PIN
                        code = (
                            ServerReturnCode.SUCCESS
                            if await self.data_store.contains(meta.key, pin=pin)
                            else ServerReturnCode.FAIL
                        )
                        await self.loop.sock_sendall(
                            client_socket,
                            ServerMetaMessage(
                                code,
                                0,
                                MemoryFormat(1),
                                torch.float16,
                                torch.Size((0, 0, 0, 0)),
                            ).serialize(),
                        )
                    case ClientCommand.HEALTH:
                        await self.loop.sock_sendall(
                            client_socket,
                            ServerMetaMessage(
                                ServerReturnCode.SUCCESS,
                                0,
                                MemoryFormat(1),
                                torch.float16,
                                torch.Size((0, 0, 0, 0)),
                            ).serialize(),
                        )
                    case ClientCommand.UNPIN:
                        data = await self.recv_all(client_socket, meta.length)
                        key_strs = data.decode().split("\n")
                        keys = [CacheEngineKey.from_string(key) for key in key_strs]
                        await self.data_store.batched_unpin(keys)
                    case ClientCommand.LIST:
                        keys = await self.data_store.list_keys()
                        key_strs = [key.to_string() for key in keys]
                        data = "\n".join(key_strs).encode()
                        await self.loop.sock_sendall(
                            client_socket,
                            ServerMetaMessage(
                                ServerReturnCode.SUCCESS,
                                len(data),
                                MemoryFormat(1),
                                torch.float16,
                                torch.Size((0, 0, 0, 0)),
                            ).serialize(),
                        )
                        await self.loop.sock_sendall(client_socket, data)
        finally:
            logger.info("Client disconnected")
            client_socket.close()

    async def run(self):
        logger.info(f"Server started at {self.host}:{self.port}")
        self.loop = asyncio.get_event_loop()
        try:
            while True:
                client_socket, addr = await self.loop.sock_accept(self.server_socket)
                logger.info(f"Connected by {addr}")
                self.loop.create_task(self.handle_client(client_socket))
        finally:
            self.server_socket.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=65432)
    # TODO: only cpu is supported for now, please don't pass in device
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--capacity", type=int, default=100)
    args = parser.parse_args()

    server = LMCacheServer(args.host, args.port, args.device, args.capacity)
    # asyncio.start_server
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
