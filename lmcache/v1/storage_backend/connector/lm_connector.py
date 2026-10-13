# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Awaitable, Callable
from typing import List, TypeVar, no_type_check
import asyncio
import errno
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

_T = TypeVar("_T")
_MAX_RPC_ATTEMPTS = 3
_RECONNECT_TIMEOUT_SECONDS = 5.0
_RETRYABLE_ERRNOS = {
    errno.EPIPE,
    errno.ECONNRESET,
    errno.ECONNREFUSED,
    errno.ECONNABORTED,
    errno.ENOTCONN,
    errno.ETIMEDOUT,
}


# TODO: performance optimization for this class, consider using C/C++/Rust
# for communication + deserialization
class LMCServerConnector(RemoteConnector):
    """Use the LMC server protocol, recovering from connection failures.

    Each RPC gets at most three attempts, serialized with all other RPCs.
    PUT has no server acknowledgement; completion does not promise persistence.
    """

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Connect to ``host:port`` using ``loop`` and a local receive allocator.

        Args:
            host: Server host name or IPv4 address.
            port: Server TCP port.
            loop: Event loop used for socket operations.
            local_cpu_backend: Configuration, metadata and receive allocator.

        Raises:
            OSError: If the initial connection cannot be established.
        """
        # NOTE(Jiayi): According to Python documentation:
        # https://docs.python.org/3/library/asyncio-eventloop.html
        # In general, protocol implementations that use transport-based APIs
        # such as loop.create_connection() and loop.create_server() are faster
        # than implementations that work with sockets.
        # However, we use socket here as we need to use the socket.recv_into()
        # to reduce memory copy.

        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self._address = (host, port)
        self._closed = False
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect(self._address)
        except BaseException:
            self.client_socket.close()
            raise
        # loop.sock_recv_into(sock, buf)

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.async_socket_lock = asyncio.Lock()

    # TODO(Jiayi): This should be an async function
    def receive_all(self, meta: ServerMetaMessage) -> MemoryObj | None:
        """Receive the body described by ``meta`` while the RPC lock is held.

        Return a caller-owned object, or None if allocation fails. An allocation
        failure discards the unread stream. A receive error releases the object
        before propagating; EOF raises ConnectionResetError for RPC recovery.
        """
        received = 0
        n = meta.length

        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        memory_obj = self.local_cpu_backend.allocate(
            meta.shape,
            meta.dtype,
            meta.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            self.client_socket.close()
            return None

        try:
            view = memoryview(memory_obj.byte_array)
            while received < n:
                num_bytes = self.client_socket.recv_into(view[received:], n - received)
                if num_bytes == 0:
                    raise ConnectionResetError("LMC server closed during GET body")
                received += num_bytes
        except BaseException:
            memory_obj.ref_count_down()
            raise

        return memory_obj

    async def exists(self, key: CacheEngineKey) -> bool:
        """Return whether ``key`` exists, retrying connection failures.

        Raises OSError after the retry budget, RuntimeError after close, and
        propagates invalid protocol responses without replaying them.
        """

        async def request() -> bool:
            self.client_socket.sendall(
                ClientMetaMessage(
                    ClientCommand.EXIST,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize()
            )

            response = self._recv_exact(ServerMetaMessage.packlength())
            return (
                ServerMetaMessage.deserialize(response).code == ServerReturnCode.SUCCESS
            )

        return await self._run_with_reconnect("EXIST", request)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        try:
            res = future.result()
            return res
        except Exception as e:
            logger.warning("lm connector failed in exists: %s", e)
            return False

    async def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """Send ``memory_obj`` under ``key``, replaying after connection failures.

        The caller retains ownership of the source. Replays send the same key
        and bytes; the protocol has no acknowledgement or exactly-once guarantee.
        Raises OSError after three attempts, or RuntimeError after close.
        """

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        async def request() -> None:
            await self.loop.sock_sendall(
                self.client_socket,
                ClientMetaMessage(
                    ClientCommand.PUT,
                    key,
                    len(kv_bytes),
                    memory_format,
                    kv_dtype,
                    kv_shape,
                ).serialize(),
            )

            await self.loop.sock_sendall(self.client_socket, kv_bytes)

        await self._run_with_reconnect("PUT", request)

    # TODO(Jiayi): This should be an async function
    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> MemoryObj | None:
        """Fetch ``key``, returning a caller-owned object or None on a cache miss.

        Allocation failure also returns None, preserving the existing contract.
        Connection failures retry up to three attempts, then raise OSError.
        Closed connectors raise RuntimeError; invalid responses are not retried.
        """

        # NOTE(Jiayi): Not using any await in the following as
        # we don't want to yield control to other tasks which could
        # sacrifice the performance loading to trade the performance of
        # saving
        async def request() -> MemoryObj | None:
            self.client_socket.sendall(
                ClientMetaMessage(
                    ClientCommand.GET,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize()
            )

            data = self._recv_exact(ServerMetaMessage.packlength())
            meta = ServerMetaMessage.deserialize(data)
            if meta.code != ServerReturnCode.SUCCESS:
                return None
            return self.receive_all(meta)

        return await self._run_with_reconnect("GET", request)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self) -> None:
        """Close the socket after outstanding RPCs and reject future operations."""
        async with self.async_socket_lock:
            self._closed = True
            self.client_socket.close()
        logger.info("Closed the lmserver connection")

    async def _run_with_reconnect(
        self, operation: str, request: Callable[[], Awaitable[_T]]
    ) -> _T:
        """Serialize whole RPCs and discard incomplete streams before retrying."""
        async with self.async_socket_lock:
            if self._closed:
                raise RuntimeError("LMCServerConnector is closed")
            for attempt in range(_MAX_RPC_ATTEMPTS):
                try:
                    if attempt:
                        await asyncio.sleep(0.1 * (2 ** (attempt - 1)))
                    if self.client_socket.fileno() < 0:
                        await self._reconnect()
                    return await request()
                except OSError as exc:
                    self.client_socket.close()
                    retryable = isinstance(exc, (ConnectionError, TimeoutError)) or (
                        exc.errno in _RETRYABLE_ERRNOS
                    )
                    if not retryable or attempt == _MAX_RPC_ATTEMPTS - 1:
                        raise
                    logger.debug("LMC server %s failed; retrying: %s", operation, exc)
                except BaseException:
                    # Cancellation or an invalid frame can leave a partial RPC.
                    self.client_socket.close()
                    raise
        raise RuntimeError("LMC server retry budget exhausted")

    async def _reconnect(self) -> None:
        """Open a replacement without blocking the loop or leaking a failed FD."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setblocking(False)
            await asyncio.wait_for(
                self.loop.sock_connect(sock, self._address),
                timeout=_RECONNECT_TIMEOUT_SECONDS,
            )
            # Preserve the existing synchronous receive path.
            sock.setblocking(True)
        except BaseException:
            sock.close()
            raise
        self.client_socket = sock

    def _recv_exact(self, size: int) -> bytes:
        """Read one fixed-size header; early EOF is a retryable connection error."""
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self.client_socket.recv(remaining)
            if not chunk:
                raise ConnectionResetError("LMC server closed during response header")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
