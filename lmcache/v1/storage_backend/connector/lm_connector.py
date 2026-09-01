# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Awaitable, Callable, List, Optional, TypeVar, no_type_check
import asyncio
import errno
import random
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

# Total attempts (1 initial + retries) for a single RPC before giving up.
_MAX_RPC_ATTEMPTS = 3
# Exponential backoff base / cap between reconnect attempts.
_RECONNECT_BACKOFF_BASE_S = 0.1
_RECONNECT_BACKOFF_MAX_S = 2.0
# ``OSError.errno`` values that indicate the connection is dead and a
# reconnect should be attempted.
_RETRYABLE_ERRNOS = frozenset(
    {
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ECONNREFUSED,
        errno.ECONNABORTED,
        errno.ENOTCONN,
        errno.ESHUTDOWN,
        errno.EBADF,
    }
)


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

        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # Keep host/port so the socket can be re-established if the remote
        # lmcache-server restarts (see ``_reconnect``).
        self.host = host
        self.port = port
        self._closed = False

        self.client_socket = self._open_socket()
        # loop.sock_recv_into(sock, buf)

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.async_socket_lock = asyncio.Lock()

    def _open_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        return sock

    def _reconnect(self) -> None:
        """Tear down the dead socket and open a fresh connection.

        The caller must hold ``async_socket_lock``, so only one coroutine
        actually reconnects (single-flight); concurrent RPCs wait on the lock
        and then reuse the freshly opened socket.
        """
        if self._closed:
            return
        try:
            self.client_socket.close()
        except OSError:
            pass
        self.client_socket = self._open_socket()

    @staticmethod
    def _is_retryable(exc: OSError) -> bool:
        """A connection-level error worth reconnecting + retrying for."""
        return isinstance(exc, ConnectionError) or exc.errno in _RETRYABLE_ERRNOS

    async def _run_with_reconnect(
        self, op_name: str, op: Callable[[], Awaitable[_T]]
    ) -> _T:
        """Run a full send+recv RPC with reconnect-on-failure.

        ``op`` performs one complete RPC against ``self.client_socket`` and is
        re-driven from the start on each attempt (it reads
        ``self.client_socket`` afresh, so it picks up a reconnected socket). On
        a connection-level ``OSError`` — from either the RPC itself or the
        reconnect's ``connect()`` (e.g. ``ECONNREFUSED`` while the server pod
        is still coming back up) — the socket is reconnected and the RPC
        retried, up to ``_MAX_RPC_ATTEMPTS`` times with jittered exponential
        backoff. Non-connection errors propagate immediately. The caller must
        hold ``async_socket_lock``.
        """
        last_exc: Optional[OSError] = None
        for attempt in range(_MAX_RPC_ATTEMPTS):
            try:
                # Reconnect (with backoff) before every retry. Keeping the
                # reconnect inside the try means a failed ``connect()`` while
                # the server is still down is itself retried within the budget
                # instead of aborting the whole RPC.
                if attempt > 0:
                    backoff = min(
                        _RECONNECT_BACKOFF_BASE_S * (2 ** (attempt - 1)),
                        _RECONNECT_BACKOFF_MAX_S,
                    )
                    await asyncio.sleep(
                        backoff + random.uniform(0, _RECONNECT_BACKOFF_BASE_S)
                    )
                    self._reconnect()
                return await op()
            except OSError as exc:
                if not self._is_retryable(exc):
                    raise
                last_exc = exc
                logger.warning(
                    "lmserver connection error on %s (attempt %d/%d): %s; "
                    "will reconnect to %s:%d and retry",
                    op_name,
                    attempt + 1,
                    _MAX_RPC_ATTEMPTS,
                    exc,
                    self.host,
                    self.port,
                )
        assert last_exc is not None
        raise last_exc

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly ``n`` bytes from the socket.

        Raises ``ConnectionResetError`` if the peer closes the connection
        before ``n`` bytes arrive (a restarted server surfaces as an empty or
        short read), so the RPC layer treats it as a connection failure and
        reconnects + retries rather than feeding a truncated buffer to
        ``deserialize`` (which would raise a non-retryable ``struct.error``).
        """
        chunks: List[bytes] = []
        received = 0
        while received < n:
            chunk = self.client_socket.recv(n - received)
            if not chunk:
                raise ConnectionResetError("lmserver closed the connection during recv")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    # TODO(Jiayi): This should be an async function
    def receive_all(self, meta: ServerMetaMessage) -> Optional[MemoryObj]:
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
            return None

        buffer = memory_obj.byte_array
        view = memoryview(buffer)

        while received < n:
            num_bytes = self.client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                # Peer closed mid-body (e.g. server restart). Surface as a
                # connection error so the whole GET RPC reconnects + retries
                # instead of being silently reported as a cache miss.
                raise ConnectionResetError(
                    "lmserver closed the connection during body recv"
                )
            received += num_bytes

        return memory_obj

    async def exists(self, key: CacheEngineKey) -> bool:
        # logger.debug("Call to exists()!")

        async def op() -> bool:
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

        async with self.async_socket_lock:
            return await self._run_with_reconnect("exists", op)

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

        async def op() -> None:
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

        async with self.async_socket_lock:
            await self._run_with_reconnect("put", op)

    # TODO(Jiayi): This should be an async function
    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # NOTE(Jiayi): Not using any await in the following as
        # we don't want to yield control to other tasks which could
        # sacrifice the performance loading to trade the performance of
        # saving.
        # The send, meta recv, and body recv are kept under a single lock hold
        # so the whole RPC can be reconnected + retried atomically (the
        # original code released the lock between meta and body, which both
        # opened an interleave window and made a clean retry impossible).
        async def op() -> Optional[MemoryObj]:
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

        async with self.async_socket_lock:
            return await self._run_with_reconnect("get", op)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        async with self.async_socket_lock:
            self._closed = True
            self.client_socket.close()
        logger.info("Closed the lmserver connection")
