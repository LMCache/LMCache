# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, no_type_check
import asyncio
import threading

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
from lmcache.v1.transfer_channel.py_socket_channel import (
    PySocketChannel as TransferChannel,
)

logger = init_logger(__name__)


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
        # NOTE(Jiayi): According to Python documentation:
        # https://docs.python.org/3/library/asyncio-eventloop.html
        # In general, protocol implementations that use transport-based APIs
        # such as loop.create_connection() and loop.create_server() are faster
        # than implementations that work with sockets.
        # However, we use socket here as we need to use the socket.recv_into()
        # to reduce memory copy.

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        # Create channel for this connection
        self.channel = TransferChannel(
            async_mode=True,
            role="sender",
            buffer_ptr=0,
            buffer_size=0,
            align_bytes=1,
            tp_rank=0,
            peer_init_url=None,
            event_loop=loop,
        )
        # Initialize data socket via base-class API using a tcp URL.
        init_url = f"tcp://{host}:{port}"
        if loop.is_running() and getattr(loop, "_thread_id", None) not in (
            None,
            threading.get_ident(),
        ):
            fut = asyncio.run_coroutine_threadsafe(
                self.channel.async_lazy_init_peer_connection(
                    local_id="client",
                    peer_id="server",
                    peer_init_url=init_url,
                ),
                loop,
            )
            fut.result()
        else:
            # Fallback for cases where we can't block on the event loop thread.
            self.channel.lazy_init_peer_connection(
                local_id="client",
                peer_id="server",
                peer_init_url=init_url,
            )

        self.async_socket_lock = asyncio.Lock()

    async def receive_all(self, meta: ServerMetaMessage) -> Optional[MemoryObj]:
        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        memory_obj = self.local_cpu_backend.allocate(
            meta.shape,
            meta.dtype,
            meta.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # Receive data using channel
        recv_count = await self.channel.async_batched_recv(
            [memory_obj],
            transfer_spec={"size": meta.length},
        )
        if recv_count == 0:
            return None

        return memory_obj

    async def exists(self, key: CacheEngineKey) -> bool:
        # logger.debug("Call to exists()!")

        async with self.async_socket_lock:
            request = ClientMetaMessage(
                ClientCommand.EXIST,
                key,
                0,
                MemoryFormat(1),
                torch.float16,
                torch.Size([0, 0, 0, 0]),
            )
            await self.channel.async_batched_send([request.serialize()])
            response = await self.channel.async_recv_exactly(
                ServerMetaMessage.packlength()
            )
            if response is None:
                return False

        return ServerMetaMessage.deserialize(response).code == ServerReturnCode.SUCCESS

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        try:
            res = future.result()
            return res
        except Exception as e:
            logger.warning(f"lm connector failed in exists: {e}")
            return False

    async def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        # logger.debug("Async call to put()!")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        async with self.async_socket_lock:
            request = ClientMetaMessage(
                ClientCommand.PUT,
                key,
                len(kv_bytes),
                memory_format,
                kv_dtype,
                kv_shape,
            )
            await self.channel.async_batched_send([request.serialize()])
            await self.channel.async_batched_send([kv_bytes])

    # TODO(Jiayi): This should be an async function
    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # IMPORTANT: Keep the socket lock held across the full request/response
        # (meta + payload). Otherwise concurrent GETs can interleave reads on
        # the same TCP stream and corrupt framing.
        async with self.async_socket_lock:
            request = ClientMetaMessage(
                ClientCommand.GET,
                key,
                0,
                MemoryFormat(1),
                torch.float16,
                torch.Size([0, 0, 0, 0]),
            )
            await self.channel.async_batched_send([request.serialize()])

            response = await self.channel.async_recv_exactly(
                ServerMetaMessage.packlength()
            )
            if response is None:
                return None

            meta = ServerMetaMessage.deserialize(response)
            if meta.code != ServerReturnCode.SUCCESS:
                return None

            return await self.receive_all(meta)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        async with self.async_socket_lock:
            self.channel.close()
        logger.info("Closed the lmserver connection")
