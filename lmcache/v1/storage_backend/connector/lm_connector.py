# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from typing import List, Optional
import asyncio
import socket

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class SafeAsyncSocket:
    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop):
        self.lock = asyncio.Lock()
        # NOTE(Jiayi): According to Python documentation:
        # https://docs.python.org/3/library/asyncio-eventloop.html
        # In general, protocol implementations that use transport-based APIs
        # such as loop.create_connection() and loop.create_server() are faster
        # than implementations that work with sockets.
        # However, we use socket here as we need to use the socket.recv_into()
        # to reduce memory copy.
        self.loop = loop
        # we are still using socket API to keep zero copy
        # (1 copy from kernel to user)
        # but we will shift the blocking from thread-level to event-loop-level
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self.host = host
        self.port = port

    async def connect(self):
        await self.loop.sock_connect(self.sock, (self.host, self.port))

    async def send(self, data: bytes):
        async with self.lock:
            # loop.sock_sendall takes care of the writing loop for us
            await self.loop.sock_sendall(self.sock, data)

    async def recv_into(self, buf: bytearray, nbytes: int):
        async with self.lock:
            view = memoryview(buf)
            cur_offset = 0
            while cur_offset < nbytes:
                num_bytes_received = await self.loop.sock_recv_into(
                    self.sock, view[cur_offset:]
                )
                if num_bytes_received == 0:
                    raise ConnectionError(...)
                cur_offset += num_bytes_received

    async def recv(self, nbytes: int) -> bytes:
        # the loop being in the lock is ok since one socket one task at a time
        async with self.lock:
            buf = bytearray(nbytes)
            view = memoryview(buf)
            cur_offset = 0
            while cur_offset < nbytes:
                chunk = await self.loop.sock_recv(self.sock, nbytes - cur_offset)
                if not chunk:
                    raise ConnectionError(
                        f"LMClient: server disconnected at {cur_offset} "
                        f"out of {nbytes} bytes"
                    )
                view[cur_offset : cur_offset + len(chunk)] = chunk
                cur_offset += len(chunk)
            return buf

    def close(self):
        # there is no loop version of close
        self.sock.close()


class SocketAllocator:
    def __init__(self, num_sockets: int):
        self.socket_allocator = set(range(num_sockets))
        self.cv = asyncio.Condition()

    async def get_socket(self) -> int:
        async with self.cv:
            while len(self.socket_allocator) == 0:
                await self.cv.wait()
            return self.socket_allocator.pop()

    async def release_socket(self, socket_idx: int):
        async with self.cv:
            self.socket_allocator.add(socket_idx)
            self.cv.notify_all()


class SocketPool:
    """
    Round-robin socket pool.
    """

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        num_sockets: int = 125,
    ):
        self.loop = loop
        self.sockets = [SafeAsyncSocket(host, port, loop) for _ in range(num_sockets)]
        self.socket_allocator = SocketAllocator(num_sockets)

    async def get_socket(self) -> int:
        return await self.socket_allocator.get_socket()

    async def release_socket(self, socket_idx: int):
        await self.socket_allocator.release_socket(socket_idx)

    @asynccontextmanager
    # to help make sure we don't leak
    async def allocate(self):
        idx = await self.get_socket()
        try:
            yield idx
        finally:
            await self.release_socket(idx)

    async def connect(self):
        await asyncio.gather(*[sock.connect() for sock in self.sockets])

    async def recv_into(self, sock_idx: int, buf: bytearray, nbytes: int):
        sock = self.sockets[sock_idx]
        return await sock.recv_into(buf, nbytes)

    async def recv(self, sock_idx: int, nbytes: int) -> bytes:
        sock = self.sockets[sock_idx]
        return await sock.recv(nbytes)

    async def send(self, sock_idx: int, data: bytes):
        sock = self.sockets[sock_idx]
        await sock.send(data)

    async def close(self):
        # this code is synchronous
        for sock in self.sockets:
            sock.close()


# TODO: performance optimization for this class, consider using C/C++/Rust
# for communication + deserialization
class LMCServerConnector(RemoteConnector):
    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # with TP 8, this will create 1000 sockets
        self.socket_pool = SocketPool(host=host, port=port, num_sockets=125, loop=loop)
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop
        conc_fut = asyncio.run_coroutine_threadsafe(self.socket_pool.connect(), loop)
        # this will block the entire worker thread until socket pool is initialized
        conc_fut.result()

    async def exists(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # the LMServer supports the pin semantics to "unrace"
        # the gap between vllm lookup() and retrieve()
        client_command = ClientCommand.EXISTS_PIN if pin else ClientCommand.EXIST
        async with self.socket_pool.allocate() as sock_idx:
            await self.socket_pool.send(
                sock_idx,
                ClientMetaMessage(
                    client_command,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize(),
            )
            response = await self.socket_pool.recv(
                sock_idx, ServerMetaMessage.num_bytes()
            )
            return (
                ServerMetaMessage.deserialize(response).code == ServerReturnCode.SUCCESS
            )

    def exists_sync(self, key: CacheEngineKey, pin: bool = False) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key, pin), self.loop)
        try:
            res = future.result()
            return res
        except Exception as e:
            logger.warning(f"lm connector failed in exists: {e}")
            return False

    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        async with self.socket_pool.allocate() as sock_idx:
            await self.socket_pool.send(
                sock_idx,
                ClientMetaMessage(
                    ClientCommand.GET,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize(),
            )
            response = await self.socket_pool.recv(
                sock_idx, ServerMetaMessage.num_bytes()
            )

            meta = ServerMetaMessage.deserialize(response)
            if meta.code != ServerReturnCode.SUCCESS:
                return None

            memory_obj = self.local_cpu_backend.allocate(
                meta.shape,
                meta.dtype,
                meta.fmt,
            )
            if memory_obj is None:
                logger.warning("Failed to allocate memory during remote receive")
                return None
            buffer = memory_obj.byte_array

            await self.socket_pool.recv_into(sock_idx, buffer, meta.length)
            return memory_obj

    async def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        async with self.socket_pool.allocate() as sock_idx:
            await self.socket_pool.send(
                sock_idx,
                ClientMetaMessage(
                    ClientCommand.PUT,
                    key,
                    len(kv_bytes),
                    memory_format,
                    kv_dtype,
                    kv_shape,
                ).serialize(),
            )

            await self.socket_pool.send(sock_idx, kv_bytes)

    async def list(self) -> List[str]:
        async with self.socket_pool.allocate() as sock_idx:
            await self.socket_pool.send(
                sock_idx,
                ClientMetaMessage(
                    ClientCommand.LIST,
                    CacheEngineKey.make_dummy_key(),
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize(),
            )
            meta = await self.socket_pool.recv(sock_idx, ServerMetaMessage.num_bytes())
            if not ServerMetaMessage.deserialize(meta).code == ServerReturnCode.SUCCESS:
                return []
            data = await self.socket_pool.recv(
                sock_idx, ServerMetaMessage.deserialize(meta).length
            )
            return data.decode().split("\n")

    def support_batched_get(self) -> bool:
        return True

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        memory_objs = await asyncio.gather(*[self.get(key) for key in keys])
        return memory_objs

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            if not await self.exists(key, pin):
                return num_hit_counts
            num_hit_counts += 1
        return num_hit_counts

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        result = await self.batched_get(keys)
        return [r for r in result if r is not None]

    def support_batched_unpin(self) -> bool:
        return True

    async def batched_unpin(self, keys: List[CacheEngineKey]):
        if not keys:
            return
        async with self.socket_pool.allocate() as sock_idx:
            key_strs = [key.to_string() for key in keys]
            data = "\n".join(key_strs).encode()
            # in the unpin meta message, all fields are meaningless
            # except the length, which is the length of the string
            # (will be parsed on server side with the lowest priority

            async with self.socket_pool.allocate() as sock_idx:
                # everything is junk except the length
                await self.socket_pool.send(
                    sock_idx,
                    ClientMetaMessage(
                        ClientCommand.UNPIN,
                        keys[0],
                        len(data),
                        MemoryFormat(1),
                        torch.float16,
                        torch.Size([0, 0, 0, 0]),
                    ).serialize(),
                )

                await self.socket_pool.send(sock_idx, data)

    async def close(self):
        await self.socket_pool.close()
        logger.info("Closed the lmserver connection")
