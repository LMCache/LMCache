# SPDX-License-Identifier: Apache-2.0

# TODO(baoloongmao): This module contains the control plane implementation
# for socket-based channels. Currently, it provides the base class with
# ZMQ-based handshake and initialization logic. In the future, this can
# be extended to a complete channel with data plane using native sockets.

# Standard
from typing import Optional, Union
import asyncio
import socket
import threading
import time

# Third Party
import msgspec
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


class PySocketMsgBase(msgspec.Struct, tag=True):
    """Base class for all py-socket-related messages"""

    pass


class PySocketInitRequest(PySocketMsgBase):
    """Initialization request, peer_init_url is used as peer identifier"""

    peer_init_url: str


class PySocketInitResponse(PySocketMsgBase):
    """Initialization response"""

    status: str


PySocketMsg = Union[
    PySocketInitRequest,
    PySocketInitResponse,
]


class PySocketChannel(BaseTransferChannel):
    """
    Base class for socket-based transfer channels.

    Control plane: Uses ZMQ sockets for handshake and initialization.
    Data plane: To be implemented by subclasses.

    Provides common control plane logic for different channel types.
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

        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self.remote_connections: dict[str, dict] = {}

        self.side_channel: Optional[zmq.Socket] = None
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        # Data plane: socket for data transfer
        self.data_socket: Optional[socket.socket] = None

        self._init_side_channels()

    ############################################################
    # Control plane: Initialization functions
    ############################################################

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        """Get memory indices from objects"""
        local_indices: list[int] = []
        if len(objects) == 0:
            return local_indices

        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                local_indices.append(mem_obj.meta.address)
        elif isinstance(objects[0], bytes):
            # For bytes, we don't have memory indices
            local_indices = [0] * len(objects)
        return local_indices

    def _is_data_socket_url(self, url: str) -> bool:
        # If it has a scheme, treat it as a data-plane socket URL.
        # ZMQ init URLs are typically like "host:port" without scheme.
        return "://" in url

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        # Data-plane init path: accept/connect socket via URL.
        if self._is_data_socket_url(peer_init_url):
            self.set_data_socket_from_url(peer_init_url)
            self.remote_connections[peer_id] = {"peer_init_url": peer_init_url}
            return None

        # Control-plane ZMQ handshake path is not supported in sync mode.
        raise NotImplementedError(
            "Sync ZMQ handshake not supported in PySocketChannel; "
            "pass a data socket URL like tcp://host:port or fd://<fd>."
        )

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """
        Initialize connection to a peer using ZMQ sockets for handshake.

        Note: peer_id is expected to be peer_init_url in this implementation.
        """
        # Data-plane init path: accept/connect socket via URL.
        if self._is_data_socket_url(peer_init_url):
            # Prefer non-blocking connect for tcp:// in async mode.
            if peer_init_url.startswith("tcp://"):
                hostport = peer_init_url[len("tcp://") :]

                def _parse_host_port(hp: str) -> tuple[str, int]:
                    if hp.startswith("["):
                        end = hp.find("]")
                        if end == -1 or end + 1 >= len(hp) or hp[end + 1] != ":":
                            raise ValueError(f"Invalid host:port: {hp}")
                        host = hp[1:end]
                        port_s = hp[end + 2 :]
                    else:
                        if ":" not in hp:
                            raise ValueError(f"Invalid host:port: {hp}")
                        host, port_s = hp.rsplit(":", 1)
                    return host, int(port_s)

                host, port = _parse_host_port(hostport)
                addrinfos = socket.getaddrinfo(
                    host,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                )
                if not addrinfos:
                    raise OSError(f"Failed to resolve {host}:{port}")
                family, socktype, proto, _, sockaddr = addrinfos[0]
                sock = socket.socket(family, socktype, proto)
                sock.setblocking(False)
                loop = asyncio.get_running_loop()
                await loop.sock_connect(sock, sockaddr)
                self.set_data_socket(sock)
            else:
                # fd:// or other supported URL: fall back to sync wrapper.
                self.set_data_socket_from_url(peer_init_url)

            self.remote_connections[peer_id] = {"peer_init_url": peer_init_url}
            return None

        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        init_req = PySocketInitRequest(peer_init_url=self.peer_init_url)
        await init_tmp_socket.send(msgspec.msgpack.encode(init_req))

        init_resp_bytes = await init_tmp_socket.recv()
        _ = msgspec.msgpack.decode(init_resp_bytes, type=PySocketMsg)

        self.remote_connections[peer_id] = {
            "peer_init_url": peer_init_url,
        }

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    def _init_side_channels(self):
        """Initialize side channel for handling incoming connections"""
        if self.peer_init_url is None:
            return

        self.side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )

        if self.async_mode:
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

    def _handle_init_msg(
        self, req: Union[PySocketMsg, InitSideMsgBase]
    ) -> Union[PySocketMsg, InitSideRetMsgBase]:
        """Handle initialization messages from peers"""
        resp: Union[PySocketMsg, InitSideRetMsgBase]
        if isinstance(req, PySocketInitRequest):
            peer_url = req.peer_init_url

            self.remote_connections[peer_url] = {
                "peer_init_url": peer_url,
            }

            self._on_peer_connected(peer_url)

            resp = PySocketInitResponse(status="ok")
            logger.info("Replying initialization response")

        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _on_peer_connected(self, peer_url: str):
        """Hook for subclasses to perform additional setup when a peer connects"""
        pass

    def _init_loop(self):
        """Synchronous initialization loop for handling incoming connections"""
        while self.running:
            try:
                req_bytes = self.side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(
                    req_bytes, type=Union[PySocketMsg, SideMsg]
                )

                resp = self._handle_init_msg(req)

                self.side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self):
        """Asynchronous initialization loop for handling incoming connections"""
        logger.info("Starting async initialization loop")

        while self.running:
            try:
                req_bytes = await self.side_channel.recv()

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(
                    req_bytes, type=Union[PySocketMsg, SideMsg]
                )

                resp = self._handle_init_msg(req)

                await self.side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    await asyncio.sleep(0.01)

    ############################################################
    # Data plane: Utility functions
    ############################################################

    def set_data_socket(self, sock: socket.socket):
        """Set the data socket for this channel"""
        self.data_socket = sock

    def get_data_socket(self) -> Optional[socket.socket]:
        """Get the data socket"""
        return self.data_socket

    def set_data_socket_from_url(self, url: str) -> socket.socket:
        """
        Create/wrap a TCP socket from a URL and set it as data socket.

        Supported URL formats:
        - ``tcp://host:port`` or ``host:port``: create a new TCP socket and connect.
        - ``fd://<int>``: take ownership of an existing file descriptor.
        """

        def _parse_host_port(hostport: str) -> tuple[str, int]:
            # Support bracketed IPv6 like "[::1]:1234"
            if hostport.startswith("["):
                end = hostport.find("]")
                if end == -1 or end + 1 >= len(hostport) or hostport[end + 1] != ":":
                    raise ValueError(f"Invalid host:port: {hostport}")
                host = hostport[1:end]
                port_s = hostport[end + 2 :]
            else:
                if ":" not in hostport:
                    raise ValueError(f"Invalid host:port: {hostport}")
                host, port_s = hostport.rsplit(":", 1)
            return host, int(port_s)

        if url.startswith("fd://"):
            fd_str = url[len("fd://") :]
            fd = int(fd_str)
            sock = socket.socket(fileno=fd)
            self.data_socket = sock
            return sock

        hostport = url[len("tcp://") :] if url.startswith("tcp://") else url
        host, port = _parse_host_port(hostport)

        addrinfos = socket.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
        if not addrinfos:
            raise OSError(f"Failed to resolve {host}:{port}")

        family, socktype, proto, _, sockaddr = addrinfos[0]
        sock = socket.socket(family, socktype, proto)
        sock.connect(sockaddr)
        self.data_socket = sock
        return sock

    def _receive_all(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket"""
        # NOTE: We intentionally avoid building a bytearray here to match the
        # connector-side pattern of working with immutable bytes for meta
        # messages. Payloads should use recv_into into a pre-allocated buffer.
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            packet = sock.recv(remaining)
            if not packet:
                return None
            chunks.append(packet)
            remaining -= len(packet)
        return b"".join(chunks)

    async def _async_receive_all(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Async receive exactly n bytes from socket"""
        chunks: list[bytes] = []
        remaining = n
        loop = asyncio.get_running_loop()
        while remaining > 0:
            packet = await loop.sock_recv(sock, remaining)
            if not packet:
                return None
            chunks.append(packet)
            remaining -= len(packet)
        return b"".join(chunks)

    def recv_exactly(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from the data socket (sync)."""
        if self.data_socket is None:
            raise RuntimeError("Data socket not initialized")
        return self._receive_all(self.data_socket, n)

    async def async_recv_exactly(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from the data socket (async)."""
        if self.data_socket is None:
            raise RuntimeError("Data socket not initialized")
        return await self._async_receive_all(self.data_socket, n)

    ############################################################
    # Data plane: Send/Recv functions
    ############################################################

    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Send a batch of data through the socket (sync)"""
        if self.data_socket is None:
            raise RuntimeError("Data socket not initialized")

        sent_count = 0
        for obj in objects:
            if isinstance(obj, (bytes, bytearray, memoryview)):
                self.data_socket.sendall(obj)
                sent_count += 1
            elif isinstance(obj, MemoryObj):
                # IMPORTANT: MemoryObj.byte_array can be a memoryview (e.g. pinned
                # memory). Do NOT convert to bytes, otherwise we'd copy payload.
                self.data_socket.sendall(obj.byte_array)
                sent_count += 1
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        return sent_count

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Receive a batch of data through the socket (sync)"""
        if self.data_socket is None:
            raise RuntimeError("Data socket not initialized")

        if transfer_spec is None:
            raise ValueError("transfer_spec is required for recv")

        recv_count = 0

        for buf in buffers:
            if isinstance(buf, bytearray):
                # For bytearray, receive directly into the buffer to avoid
                # allocating a temporary bytes object + copying.
                size = transfer_spec.get("size", len(buf))
                view = memoryview(buf)
                remaining = size
                offset = 0
                while remaining > 0:
                    chunk_size = min(remaining, 65536)
                    received = self.data_socket.recv_into(
                        view[offset : offset + chunk_size], chunk_size
                    )
                    if received == 0:
                        return recv_count
                    offset += received
                    remaining -= received
                recv_count += 1
            elif isinstance(buf, MemoryObj):
                # For MemoryObj, read into its byte_array
                size = buf.get_physical_size()
                byte_array = buf.byte_array

                # Use recv_into to directly read into the buffer
                if isinstance(byte_array, memoryview):
                    # Cast memoryview to 'B' (unsigned bytes) format for recv_into
                    view = byte_array.cast("B")
                    remaining = size
                    offset = 0
                    while remaining > 0:
                        chunk_size = min(remaining, 65536)  # Read in chunks
                        view_slice = view[offset : offset + chunk_size]
                        received = self.data_socket.recv_into(view_slice, chunk_size)
                        if received == 0:
                            return recv_count
                        offset += received
                        remaining -= received
                elif isinstance(byte_array, bytearray):
                    # Direct recv_into for bytearray
                    remaining = size
                    offset = 0
                    while remaining > 0:
                        chunk_size = min(remaining, 65536)
                        received = self.data_socket.recv_into(
                            memoryview(byte_array[offset : offset + chunk_size]),
                            chunk_size,
                        )
                        if received == 0:
                            return recv_count
                        offset += received
                        remaining -= received
                else:
                    # Fallback: read into temporary buffer and copy
                    data = self._receive_all(self.data_socket, size)
                    if data is None:
                        break
                    # For other types, we can't modify directly
                    raise ValueError(
                        f"Cannot receive into buffer type: {type(byte_array)}"
                    )
                recv_count += 1
            else:
                raise ValueError(f"Unsupported buffer type: {type(buf)}")

        return recv_count

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async send a batch of data through the socket"""
        if self.data_socket is None:
            raise RuntimeError("Data socket not initialized")

        loop = asyncio.get_event_loop()
        sent_count = 0

        for obj in objects:
            if isinstance(obj, (bytes, bytearray, memoryview)):
                await loop.sock_sendall(self.data_socket, obj)
                sent_count += 1
            elif isinstance(obj, MemoryObj):
                # IMPORTANT: avoid memoryview->bytes copy
                await loop.sock_sendall(self.data_socket, obj.byte_array)
                sent_count += 1
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        return sent_count

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async receive a batch of data through the socket"""
        if self.data_socket is None:
            raise RuntimeError("Data socket not initialized")

        if transfer_spec is None:
            raise ValueError("transfer_spec is required for recv")

        recv_count = 0
        loop = asyncio.get_event_loop()

        for buf in buffers:
            if isinstance(buf, bytearray):
                # For bytearray, receive directly into the buffer to avoid
                # allocating a temporary bytes object + copying.
                size = transfer_spec.get("size", len(buf))
                view = memoryview(buf)
                remaining = size
                offset = 0
                while remaining > 0:
                    chunk_size = min(remaining, 65536)
                    # asyncio BaseSelectorEventLoop.sock_recv_into(sock, buf)
                    # reads up to len(buf) bytes; it does NOT take a "nbytes"
                    # argument in Python 3.12.
                    received = await loop.sock_recv_into(
                        self.data_socket,
                        view[offset : offset + chunk_size],
                    )
                    if received == 0:
                        return recv_count
                    offset += received
                    remaining -= received
                recv_count += 1
            elif isinstance(buf, MemoryObj):
                # For MemoryObj, read into its byte_array
                size = buf.get_physical_size()
                byte_array = buf.byte_array

                # For async, we need to read in chunks and copy into the buffer
                if isinstance(byte_array, memoryview):
                    # Cast to 'B' format for byte operations
                    view = byte_array.cast("B")
                    remaining = size
                    offset = 0
                    while remaining > 0:
                        chunk_size = min(remaining, 65536)  # Read in chunks
                        view_slice = view[offset : offset + chunk_size]
                        received = await loop.sock_recv_into(
                            self.data_socket, view_slice
                        )
                        if received == 0:
                            return recv_count
                        offset += received
                        remaining -= received
                    if remaining > 0:
                        return recv_count  # Incomplete read
                elif isinstance(byte_array, bytearray):
                    # For bytearray, read directly
                    remaining = size
                    offset = 0
                    while remaining > 0:
                        chunk_size = min(remaining, 65536)
                        received = await loop.sock_recv_into(
                            self.data_socket,
                            memoryview(byte_array)[offset : offset + chunk_size],
                        )
                        if received == 0:
                            return recv_count
                        offset += received
                        remaining -= received
                    if remaining > 0:
                        return recv_count  # Incomplete read
                else:
                    # Fallback: read all and try to copy
                    data = await self._async_receive_all(self.data_socket, size)
                    if data is None:
                        break
                    raise ValueError(
                        f"Cannot receive into buffer type: {type(byte_array)}"
                    )
                recv_count += 1
            else:
                raise ValueError(f"Unsupported buffer type: {type(buf)}")

        return recv_count

    ############################################################
    # Data plane: Read/Write functions
    ############################################################

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        """Check if remote handler exists"""
        return self.data_socket is not None and self.data_socket.fileno() != -1

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Write a batch of data through the socket (sync)"""
        return self.batched_send(objects, transfer_spec)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Read a batch of data through the socket (sync)"""
        return self.batched_recv(buffers, transfer_spec)

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async write a batch of data through the socket"""
        return await self.async_batched_send(objects, transfer_spec)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async read a batch of data through the socket"""
        return await self.async_batched_recv(buffers, transfer_spec)

    ############################################################
    # Cleanup-related functions
    ############################################################

    def close(self):
        """Close all sockets and cleanup resources"""
        self.running = False
        for thread in self.running_threads:
            thread.join()

        if self.data_socket is not None:
            try:
                self.data_socket.close()
            except Exception as e:
                logger.warning(f"Error closing data socket: {e}")

        if self.side_channel is not None:
            try:
                self.side_channel.close()
            except Exception as e:
                logger.warning(f"Error closing side channel: {e}")

        # Note: We don't call zmq_context.term() because get_zmq_context()
        # returns a singleton instance (zmq.Context.instance()) that may be
        # shared with other parts of the application. Terminating it would
        # affect all users of the context.
