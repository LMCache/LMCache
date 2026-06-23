# SPDX-License-Identifier: Apache-2.0
"""TCP-backed implementation of the transfer channel abstraction.

This implementation uses plain TCP sockets for data transfer between peers.
It does not require RDMA hardware and is suitable for environments without
specialized networking (e.g. development, cross-datacenter, or commodity
hardware).

Architecture:
- Server: Listens for incoming TCP connections. Each connection is handled
  in a dedicated thread. Clients send read requests (remote offsets/sizes),
  and the server responds with the raw bytes from its L1 shared memory.
- Client: Maintains a persistent TCP connection to a peer's server. Read
  requests are submitted asynchronously and fulfilled by a background
  receiver thread.
- Context: Owns the server, manages client connections, and provides
  address translation.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from ctypes import c_char
from dataclasses import dataclass
from typing import Optional
import socket
import struct
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    register_transfer_channel_factory,
)

logger = init_logger(__name__)

# Protocol constants
_HANDSHAKE_MAGIC = b"LMTCP001"
_MSG_TYPE_READ_REQ = 1
_MSG_TYPE_READ_RESP = 2

# Default connection timeout (seconds)
_CONNECT_TIMEOUT_S = 30.0

# Maximum number of concurrent server handler threads
_MAX_SERVER_THREADS = 16

# Socket buffer sizes (4 MB for better throughput on large transfers)
_SOCKET_BUF_SIZE = 4 * 1024 * 1024


############################################################
# Helper functions
############################################################
def _parse_url(url: str) -> tuple[str, int]:
    """Parse a ``host:port`` (optionally ``tcp://host:port``) into (host, port)."""
    stripped = url.split("://", 1)[-1]
    host, _, port = stripped.rpartition(":")
    if not host or not port:
        raise ValueError(f"Invalid transfer channel url: {url!r} (expected host:port)")
    return host, int(port)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly ``n`` bytes from ``sock``, raising on premature close."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Connection closed while expecting {n} bytes "
                f"(received {len(buf)} so far)"
            )
        buf.extend(chunk)
    return bytes(buf)


def _send_all(sock: socket.socket, data: bytes | memoryview) -> None:
    """Send all bytes, handling partial sends."""
    sock.sendall(data)


def _set_socket_options(sock: socket.socket) -> None:
    """Apply common socket options for performance."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCKET_BUF_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCKET_BUF_SIZE)


############################################################
# Wire protocol structures
############################################################
@dataclass
class _ReadRequest:
    """A batch read request sent from client to server.

    Wire format:
        [4 bytes] msg_type = _MSG_TYPE_READ_REQ
        [4 bytes] request_id
        [4 bytes] num_entries
        For each entry:
            [8 bytes] offset (int64, little-endian)
            [8 bytes] size (int64, little-endian)
    """

    request_id: int
    entries: list[tuple[int, int]]  # (offset, size)

    def encode(self) -> bytes:
        """Serialize to wire format."""
        header = struct.pack(
            "<III", _MSG_TYPE_READ_REQ, self.request_id, len(self.entries)
        )
        body = b"".join(
            struct.pack("<qq", offset, size) for offset, size in self.entries
        )
        return header + body

    @staticmethod
    def decode_from_socket(
        sock: socket.socket, request_id: int, num_entries: int
    ) -> "_ReadRequest":
        """Decode entries from socket after header has been read."""
        entry_data = _recv_exact(sock, num_entries * 16)
        entries = []
        for i in range(num_entries):
            offset, size = struct.unpack_from("<qq", entry_data, i * 16)
            entries.append((offset, size))
        return _ReadRequest(request_id=request_id, entries=entries)


@dataclass
class _ReadResponse:
    """A batch read response sent from server to client.

    Wire format:
        [4 bytes] msg_type = _MSG_TYPE_READ_RESP
        [4 bytes] request_id
        [4 bytes] num_entries
        [4 bytes] success_mask_bytes (ceil(num_entries / 8))
        [N bytes] success_mask (packed bits)
        For each successful entry:
            [raw bytes] data of entry.size bytes
    """

    request_id: int
    success_mask: list[bool]
    # data is sent inline, not stored here for the server side


############################################################
# Server
############################################################
class TcpTransferChannelServer(TransferChannelServer):
    """TCP server that serves read requests from peer clients.

    Each connected client gets a dedicated handler thread. The server reads
    requests from the client, fetches the requested byte ranges from the
    local L1 shared memory, and sends the data back.
    """

    def __init__(
        self,
        listen_url: str,
        advertise_url: str,
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        self._listen_url = listen_url
        self._advertise_url = advertise_url
        self._l1_memory_desc = l1_memory_desc
        self._running = True

        # Track active client sockets so we can close them on shutdown
        self._client_sockets: set[socket.socket] = set()
        self._client_sockets_lock = threading.Lock()

        host, port = _parse_url(listen_url)
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((host, port))
        self._server_socket.listen(32)
        self._server_socket.settimeout(1.0)

        self._executor = ThreadPoolExecutor(
            max_workers=_MAX_SERVER_THREADS,
            thread_name_prefix="tc-tcp-handler",
        )
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="tc-tcp-accept", daemon=True
        )
        self._accept_thread.start()
        logger.info(
            "TCP transfer channel server listening on %s (advertise: %s)",
            listen_url,
            advertise_url,
        )

    def _accept_loop(self) -> None:
        """Accept incoming connections and dispatch to handler threads."""
        while self._running:
            try:
                # TODO(chunxiaozheng): use a more efficient method to accept
                #  and handle connections
                client_sock, addr = self._server_socket.accept()
                _set_socket_options(client_sock)
                with self._client_sockets_lock:
                    self._client_sockets.add(client_sock)
                self._executor.submit(self._handle_client, client_sock, addr)
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.exception("Error accepting TCP connection")
                break

    def _handle_client(self, client_sock: socket.socket, addr: tuple[str, int]) -> None:
        """Handle a single client connection, serving read requests until close."""
        try:
            # Perform handshake: expect magic bytes from client
            magic = _recv_exact(client_sock, len(_HANDSHAKE_MAGIC))
            if magic != _HANDSHAKE_MAGIC:
                logger.warning("Invalid handshake magic from %s: %r", addr, magic)
                return

            # Send magic back as acknowledgement
            _send_all(client_sock, _HANDSHAKE_MAGIC)

            # Serve read requests in a loop
            while self._running:
                try:
                    header_data = _recv_exact(client_sock, 12)
                except ConnectionError:
                    break

                msg_type, request_id, num_entries = struct.unpack("<III", header_data)
                if msg_type != _MSG_TYPE_READ_REQ:
                    logger.warning("Unexpected message type %d from %s", msg_type, addr)
                    break

                req = _ReadRequest.decode_from_socket(
                    client_sock, request_id, num_entries
                )
                self._serve_read_request(client_sock, req)

        except ConnectionError:
            logger.debug("Client %s disconnected", addr)
        except Exception:
            if self._running:
                logger.exception("Error handling client %s", addr)
        finally:
            with self._client_sockets_lock:
                self._client_sockets.discard(client_sock)
            client_sock.close()

    def _serve_read_request(
        self, client_sock: socket.socket, req: _ReadRequest
    ) -> None:
        """Read the requested byte ranges from L1 memory and send them back."""
        ptr = self._l1_memory_desc.ptr
        mem_size = self._l1_memory_desc.size
        num_entries = len(req.entries)

        # Determine which entries are valid
        success_mask = []
        for offset, size in req.entries:
            valid = offset >= 0 and size > 0 and offset + size <= mem_size
            success_mask.append(valid)

        # Build response header
        mask_byte_count = (num_entries + 7) // 8
        mask_bytes = bytearray(mask_byte_count)
        for i, ok in enumerate(success_mask):
            if ok:
                mask_bytes[i // 8] |= 1 << (i % 8)

        resp_header = struct.pack(
            "<IIII",
            _MSG_TYPE_READ_RESP,
            req.request_id,
            num_entries,
            mask_byte_count,
        )
        _send_all(client_sock, resp_header + bytes(mask_bytes))

        # Send data for each successful entry
        for i, (offset, size) in enumerate(req.entries):
            if not success_mask[i]:
                continue
            # Read directly from L1 shared memory
            src_addr = ptr + offset
            buf = (c_char * size).from_address(src_addr)
            _send_all(client_sock, memoryview(buf))

    def close(self) -> None:
        """Stop accepting connections and shut down handler threads."""
        self._running = False
        self._server_socket.close()

        # Close all active client sockets to unblock handler threads
        with self._client_sockets_lock:
            for sock in self._client_sockets:
                try:
                    sock.close()
                except OSError:
                    pass
            self._client_sockets.clear()

        if self._accept_thread.is_alive():
            self._accept_thread.join(timeout=5)
        self._executor.shutdown(wait=False)


############################################################
# Client
############################################################
class TcpTransferChannelClient(TransferChannelClient):
    """TCP client that reads data from a remote peer's L1 memory.

    Maintains a persistent TCP connection to the peer's server. Read requests
    are submitted and responses are received on the same connection. A
    background receiver thread processes incoming responses and marks tasks
    as complete.
    """

    def __init__(self, transfer_channel_server_url: str) -> None:
        self._server_url = transfer_channel_server_url
        self._closed = False

        self._task_counter = 0
        self._lock = threading.Lock()
        # task_id -> (num_entries, result or None if pending)
        self._tasks: dict[int, tuple[int, Optional[TransferChannelReadResult]]] = {}
        # task_id -> list of (local_offset, size) for writing received data
        self._task_local_addrs: dict[int, list[TransferChannelAddress]] = {}
        # L1 memory descriptor (set by context after creation)
        self._l1_ptr: int = 0
        self._l1_size: int = 0

        # Connect to the server
        host, port = _parse_url(transfer_channel_server_url)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(_CONNECT_TIMEOUT_S)
        self._socket.connect((host, port))
        _set_socket_options(self._socket)

        # Perform handshake
        _send_all(self._socket, _HANDSHAKE_MAGIC)
        ack = _recv_exact(self._socket, len(_HANDSHAKE_MAGIC))
        if ack != _HANDSHAKE_MAGIC:
            self._socket.close()
            raise ConnectionError(
                f"Handshake failed with {transfer_channel_server_url}: "
                f"expected {_HANDSHAKE_MAGIC!r}, got {ack!r}"
            )

        # Switch to non-blocking for the send path; the receiver thread
        # uses blocking recv.
        self._socket.settimeout(None)

        # Send lock to serialize write access to the socket
        self._send_lock = threading.Lock()

        # Start receiver thread
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name="tc-tcp-recv", daemon=True
        )
        self._recv_thread.start()

    def set_l1_memory(self, ptr: int, size: int) -> None:
        """Set the local L1 memory pointer and size for writing received data.

        Args:
            ptr: Base address of the local L1 shared memory.
            size: Total size of the L1 memory region.
        """
        self._l1_ptr = ptr
        self._l1_size = size

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit a read request to fetch data from the remote peer.

        Args:
            local_addresses: Where to write the received data in local L1.
            remote_addresses: Where to read from in the remote peer's L1.

        Returns:
            A task ID for tracking the read status.
        """
        if len(local_addresses) != len(remote_addresses):
            raise ValueError(
                "local_addresses and remote_addresses must have equal length "
                f"({len(local_addresses)} != {len(remote_addresses)})"
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("Client is closed")
            task_id = self._task_counter
            self._task_counter += 1
            self._tasks[task_id] = (len(remote_addresses), None)
            self._task_local_addrs[task_id] = list(local_addresses)

        # Build and send the read request
        entries = [(addr.offset, addr.size) for addr in remote_addresses]
        req = _ReadRequest(request_id=task_id, entries=entries)

        with self._send_lock:
            try:
                _send_all(self._socket, req.encode())
            except (OSError, ConnectionError) as e:
                # TODO(chunxiaozheng): should we add some recovery logic here?
                logger.warning("Failed to send read request: %s", e)
                # The socket is no longer usable after a send failure;
                # mark all pending tasks as failed and close.
                self._mark_all_failed()
                self._closed = True
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._socket.close()

        return task_id

    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        """Query the status of a previously submitted read.

        Args:
            task_id: The task ID returned by submit_read.

        Returns:
            The read result (finished=False if still in progress).
        """
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Unknown read task id: {task_id}")
            num_entries, result = self._tasks[task_id]
            if result is None:
                return TransferChannelReadResult(finished=False, succeeded_mask=[])
            # Remove completed task
            del self._tasks[task_id]
            self._task_local_addrs.pop(task_id, None)
            return result

    def _recv_loop(self) -> None:
        """Background thread that receives responses from the server."""
        while not self._closed:
            try:
                # Read response header: msg_type(4) + request_id(4) +
                # num_entries(4) + mask_byte_count(4)
                header_data = _recv_exact(self._socket, 16)
                msg_type, request_id, num_entries, mask_byte_count = struct.unpack(
                    "<IIII", header_data
                )

                if msg_type != _MSG_TYPE_READ_RESP:
                    logger.warning("Unexpected response message type: %d", msg_type)
                    break

                # Read success mask
                mask_bytes = _recv_exact(self._socket, mask_byte_count)
                success_mask = []
                for i in range(num_entries):
                    bit = (mask_bytes[i // 8] >> (i % 8)) & 1
                    success_mask.append(bool(bit))

                # Read data for successful entries and write to local L1
                with self._lock:
                    local_addrs = self._task_local_addrs.get(request_id, [])

                for i in range(num_entries):
                    if not success_mask[i]:
                        continue
                    # Determine size from local address
                    if i < len(local_addrs):
                        size = local_addrs[i].size
                        # Receive directly into L1 memory (zero intermediate copy)
                        dst_addr = self._l1_ptr + local_addrs[i].offset
                        dst_buf = (c_char * size).from_address(dst_addr)
                        view = memoryview(dst_buf)
                        remaining = size
                        while remaining > 0:
                            nbytes = self._socket.recv_into(view)
                            if nbytes == 0:
                                raise ConnectionError(
                                    f"Connection closed while expecting {size} bytes"
                                )
                            view = view[nbytes:]
                            remaining -= nbytes
                    else:
                        # Shouldn't happen, but handle gracefully
                        logger.warning(
                            "Response entry %d has no local address for task %d",
                            i,
                            request_id,
                        )
                        break

                # Mark task as complete
                with self._lock:
                    if request_id in self._tasks:
                        self._tasks[request_id] = (
                            num_entries,
                            TransferChannelReadResult(
                                finished=True, succeeded_mask=success_mask
                            ),
                        )

            except ConnectionError:
                if not self._closed:
                    logger.debug("TCP transfer channel connection closed")
                self._mark_all_failed()
                break
            except Exception:
                if not self._closed:
                    logger.exception("Error in TCP transfer channel recv loop")
                self._mark_all_failed()
                break

    def _mark_all_failed(self) -> None:
        """Mark all pending tasks as failed (connection lost)."""
        with self._lock:
            for task_id, (num_entries, result) in list(self._tasks.items()):
                if result is None:
                    self._tasks[task_id] = (
                        num_entries,
                        TransferChannelReadResult(
                            finished=True,
                            succeeded_mask=[False] * num_entries,
                        ),
                    )

    def close(self) -> None:
        """Close the connection and stop the receiver thread."""
        self._closed = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        if self._recv_thread.is_alive():
            self._recv_thread.join(timeout=5)


############################################################
# Context
############################################################
class TcpTransferChannelContext(TransferChannelContext):
    """Owns the TCP server and manages client connections to peers.

    Unlike NIXL which uses RDMA for zero-copy transfers, this implementation
    copies data over TCP sockets. It is simpler and works on any network but
    has higher latency and CPU overhead.
    """

    def __init__(
        self,
        l1_memory_desc: L1MemoryDesc,
        listen_url: str,
        advertise_url: str,
    ) -> None:
        """Create the TCP transfer channel context.

        Args:
            l1_memory_desc: Describes the local L1 memory region.
            listen_url: The ``host:port`` to bind the server on.
            advertise_url: The ``host:port`` advertised to peers.
        """
        self._l1_memory_desc = l1_memory_desc
        self._listen_url = listen_url
        self._advertise_url = advertise_url

        self._clients: dict[str, TcpTransferChannelClient] = {}
        self._lock = threading.Lock()

        # Start the server
        self._server = TcpTransferChannelServer(
            listen_url=listen_url,
            advertise_url=advertise_url,
            l1_memory_desc=l1_memory_desc,
        )

    ############################################################
    # Address translation
    ############################################################
    def get_transfer_channel_address(
        self,
        lmcache_addresses: list[tuple[int, int]],
    ) -> list[TransferChannelAddress]:
        """Translate (offset, size) L1 addresses into TransferChannelAddress.

        For TCP, the address is simply the offset and size within the L1
        memory region (no page-granular translation needed).

        Args:
            lmcache_addresses: List of (offset, size) tuples.

        Returns:
            List of TransferChannelAddress objects.
        """
        size = self._l1_memory_desc.size
        out = []
        for offset, obj_size in lmcache_addresses:
            if offset < 0 or offset + obj_size > size:
                raise ValueError(
                    f"Object [{offset:#x}, {offset + obj_size:#x}) is outside the "
                    f"registered L1 region [0x0, {size:#x})"
                )
            out.append(TransferChannelAddress(offset=offset, size=obj_size))
        return out

    ############################################################
    # Server / client management
    ############################################################
    def get_transfer_channel_server(self) -> TcpTransferChannelServer:
        """Return the TCP transfer channel server."""
        return self._server

    def get_transfer_channel_client(
        self,
        peer_advertise_url: str,
    ) -> TcpTransferChannelClient:
        """Get or create a client connected to the specified peer.

        Args:
            peer_advertise_url: The ``host:port`` of the peer's server.

        Returns:
            A connected TcpTransferChannelClient.
        """
        with self._lock:
            client = self._clients.get(peer_advertise_url)
            if client is not None:
                return client

        # Create a new client connection
        client = TcpTransferChannelClient(peer_advertise_url)
        client.set_l1_memory(self._l1_memory_desc.ptr, self._l1_memory_desc.size)

        with self._lock:
            # Check again in case another thread created it concurrently
            existing = self._clients.get(peer_advertise_url)
            if existing is not None:
                client.close()
                return existing
            self._clients[peer_advertise_url] = client

        logger.debug("Created TCP transfer channel client for %s", peer_advertise_url)
        return client

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Remove and close the client for the specified peer.

        Args:
            peer_advertise_url: The ``host:port`` of the peer.
        """
        with self._lock:
            client = self._clients.pop(peer_advertise_url, None)
        if client is None:
            return
        try:
            client.close()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error closing TCP transfer channel client for %s",
                peer_advertise_url,
            )

    def get_num_connected_clients(self) -> int:
        """Return the number of currently connected clients."""
        with self._lock:
            return len(self._clients)

    def close(self) -> None:
        """Shut down the server and all clients."""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()

        self._server.close()
        for client in clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


############################################################
# Factory registration
############################################################
def create_tcp_transfer_channel_context(
    l1_memory_desc: L1MemoryDesc,
    listen_url: str,
    advertise_url: str,
    **kwargs,
) -> TcpTransferChannelContext:
    """Create a ``TcpTransferChannelContext``.

    Args:
        l1_memory_desc: Describes the L1 memory region.
        listen_url: ``host:port`` this peer's server binds to.
        advertise_url: ``host:port`` this peer advertises as its identity.
        **kwargs: Unused; accepted for interface compatibility.

    Returns:
        A new ``TcpTransferChannelContext`` instance.
    """
    return TcpTransferChannelContext(
        l1_memory_desc=l1_memory_desc,
        listen_url=listen_url,
        advertise_url=advertise_url,
    )


register_transfer_channel_factory("tcp", create_tcp_transfer_channel_context)
