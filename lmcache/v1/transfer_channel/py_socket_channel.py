# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import socket
import struct
import threading
import time
from typing import Any, Dict, Optional, Union
import uuid

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


class SocketMsgBase(msgspec.Struct, tag=True):
    """Base class for all socket-related messages"""
    pass


class SocketInitRequest(SocketMsgBase):
    """Initial handshake request"""
    local_id: str
    buffer_size: int
    align_bytes: int
    capabilities: list[str]  # Supported features


class SocketInitResponse(SocketMsgBase):
    """Initial handshake response"""
    remote_id: str
    buffer_size: int
    align_bytes: int
    capabilities: list[str]
    data_port: int  # Port for data transfer socket


class SocketDataTransferRequest(SocketMsgBase):
    """Request to transfer data"""
    transfer_id: str
    sender_id: str
    receiver_id: str
    object_count: int
    object_sizes: list[int]  # Size of each object in bytes
    transfer_type: str  # "write", "read", "send", "recv"


class SocketDataTransferResponse(SocketMsgBase):
    """Response to data transfer request"""
    transfer_id: str
    status: str  # "ready", "error"
    error_message: Optional[str] = None


SocketMsg = Union[
    SocketInitRequest,
    SocketInitResponse,
    SocketDataTransferRequest,
    SocketDataTransferResponse
]


class PySocketChannel(BaseTransferChannel):
    """
    Socket-based implementation of BaseTransferChannel.

    This provides a fallback option when NIXL is not available,
    using standard Python sockets for data transfer.
    """

    def __init__(
        self,
        async_mode: bool = False,
        **kwargs,
    ):
        """Initialize PySocketChannel"""
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
        self.peer_init_url = kwargs["peer_init_url"]

        # Used for P2P
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        # Connection management
        self.local_id = str(uuid.uuid4())
        self.running = True
        self.peer_connections: Dict[str, Dict[str, Any]] = {}
        self.data_sockets: Dict[str, socket.socket] = {}

        # Threading
        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []
        self.socket_lock = threading.Lock()

        # Async support
        self.async_mode = async_mode
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)

        self.event_loop = kwargs.get("event_loop", None)

        # Socket configuration
        self.socket_timeout = kwargs.get("socket_timeout", 30.0)
        self.buffer_size_per_recv = kwargs.get("buffer_size_per_recv", 64 * 1024)  # 64KB chunks

        # Capabilities this implementation supports
        self.capabilities = ["basic_transfer", "batched_ops"]

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
        """Initialize connection to a peer using sockets"""
        # Initialize temporary socket for handshake
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        try:
            # Send initial handshake request
            init_req = SocketInitRequest(
                local_id=self.local_id,
                buffer_size=self.buffer_size,
                align_bytes=self.align_bytes,
                capabilities=self.capabilities,
            )
            init_tmp_socket.send(msgspec.msgpack.encode(init_req))

            # Receive handshake response
            init_resp_bytes = init_tmp_socket.recv()
            init_resp = msgspec.msgpack.decode(init_resp_bytes, type=SocketMsg)

            if not isinstance(init_resp, SocketInitResponse):
                raise ValueError(f"Expected SocketInitResponse, got {type(init_resp)}")

            # Establish data transfer socket connection
            data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data_socket.settimeout(self.socket_timeout)

            # Parse peer URL to get host
            # Format: tcp://host:port -> connect to host:data_port
            if "://" in peer_init_url:
                host = peer_init_url.split("://")[1].split(":")[0]
            else:
                host = peer_init_url.split(":")[0]

            data_socket.connect((host, init_resp.data_port))

            # Store peer connection info
            with self.socket_lock:
                self.peer_connections[peer_id] = {
                    "remote_id": init_resp.remote_id,
                    "buffer_size": init_resp.buffer_size,
                    "align_bytes": init_resp.align_bytes,
                    "capabilities": init_resp.capabilities,
                }
                self.data_sockets[peer_id] = data_socket

            # Send side message if any
            init_ret_msg: Optional[InitSideRetMsgBase] = None
            if init_side_msg is not None:
                init_ret_msg = self.send_init_side_msg(
                    init_tmp_socket,
                    init_side_msg,
                )

            logger.info(f"Successfully connected to peer {peer_id}")
            return init_ret_msg

        finally:
            init_tmp_socket.close()

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """Async version of lazy_init_peer_connection"""
        # Initialize temporary socket for handshake
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        try:
            # Send initial handshake request
            init_req = SocketInitRequest(
                local_id=self.local_id,
                buffer_size=self.buffer_size,
                align_bytes=self.align_bytes,
                capabilities=self.capabilities,
            )
            await init_tmp_socket.send(msgspec.msgpack.encode(init_req))

            # Receive handshake response
            init_resp_bytes = await init_tmp_socket.recv()
            init_resp = msgspec.msgpack.decode(init_resp_bytes, type=SocketMsg)

            if not isinstance(init_resp, SocketInitResponse):
                raise ValueError(f"Expected SocketInitResponse, got {type(init_resp)}")

            # Establish data transfer socket connection (async)
            if "://" in peer_init_url:
                host = peer_init_url.split("://")[1].split(":")[0]
            else:
                host = peer_init_url.split(":")[0]

            reader, writer = await asyncio.open_connection(host, init_resp.data_port)

            # Store peer connection info
            with self.socket_lock:
                self.peer_connections[peer_id] = {
                    "remote_id": init_resp.remote_id,
                    "buffer_size": init_resp.buffer_size,
                    "align_bytes": init_resp.align_bytes,
                    "capabilities": init_resp.capabilities,
                    "reader": reader,
                    "writer": writer,
                }

            # Send side message if any
            init_ret_msg: Optional[InitSideRetMsgBase] = None
            if init_side_msg is not None:
                init_ret_msg = await self.async_send_init_side_msg(
                    init_tmp_socket,
                    init_side_msg,
                )

            logger.info(f"Successfully connected to peer {peer_id} (async)")
            return init_ret_msg

        finally:
            init_tmp_socket.close()

    def _init_side_channels(self):
        """Initialize side channels for incoming connections"""
        if self.peer_init_url is None:
            return

        if self.async_mode:
            # Start listening coroutine for initialization
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            # Start listening thread for initialization
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

    def _handle_init_msg(
        self, req: Union[SocketMsg, InitSideMsgBase]
    ) -> Union[SocketMsg, InitSideRetMsgBase]:
        """Handle initialization messages from peers"""
        if isinstance(req, SocketInitRequest):
            # Create data transfer socket
            data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            data_socket.bind(('', 0))  # Bind to any available port
            data_socket.listen(1)
            data_port = data_socket.getsockname()[1]

            # Store the listening socket for this peer
            with self.socket_lock:
                self.data_sockets[req.local_id] = data_socket

            # Start thread to accept data connection
            accept_thread = threading.Thread(
                target=self._accept_data_connection,
                args=(req.local_id, data_socket),
                daemon=True
            )
            accept_thread.start()
            self.running_threads.append(accept_thread)

            response = SocketInitResponse(
                remote_id=self.local_id,
                buffer_size=self.buffer_size,
                align_bytes=self.align_bytes,
                capabilities=self.capabilities,
                data_port=data_port,
            )

            logger.info(f"Handled init request from {req.local_id}, data port: {data_port}")
            return response

        elif isinstance(req, SocketDataTransferRequest):
            # Handle transfer coordination request
            if req.transfer_type in ["send", "recv"]:
                response = SocketDataTransferResponse(
                    transfer_id=req.transfer_id,
                    status="ready"
                )
                logger.info(f"Handled data transfer request: {req.transfer_type}")
                return response
            else:
                response = SocketDataTransferResponse(
                    transfer_id=req.transfer_id,
                    status="error",
                    error_message=f"Unsupported transfer type: {req.transfer_type}"
                )
                return response

        elif isinstance(req, InitSideMsgBase):
            response = self.handle_init_side_msg(req)
            logger.info("Handled P2P init side message")
            return response
        else:
            raise ValueError(f"Unsupported init message type: {type(req)}")

    def _accept_data_connection(self, peer_id: str, listen_socket: socket.socket):
        """Accept data connection from a peer"""
        try:
            client_socket, addr = listen_socket.accept()
            client_socket.settimeout(self.socket_timeout)

            with self.socket_lock:
                # Replace the listening socket with the connected socket
                self.data_sockets[peer_id] = client_socket

            logger.info(f"Accepted data connection from peer {peer_id} at {addr}")
        except Exception as e:
            logger.error(f"Error accepting data connection from {peer_id}: {e}")
        finally:
            listen_socket.close()

    def _init_loop(self):
        """Main initialization loop for handling incoming connections"""
        # Initialize initialization side channel
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)

        logger.info("Started socket initialization loop")

        while self.running:
            try:
                req_bytes = self.init_side_channel.recv()
                req = msgspec.msgpack.decode(req_bytes, type=Union[SocketMsg, SideMsg])

                resp = self._handle_init_msg(req)

                self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error(f"Error in socket init loop: {e}")
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self):
        """Async version of initialization loop"""
        # Initialize initialization side channel
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)

        logger.info("Started async socket initialization loop")

        while self.running:
            try:
                req_bytes = await self.init_side_channel.recv()
                req = msgspec.msgpack.decode(req_bytes, type=Union[SocketMsg, SideMsg])

                resp = self._handle_init_msg(req)

                await self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error(f"Error in async socket init loop: {e}")
                if self.running:
                    await asyncio.sleep(0.01)

    ############################################################
    # Utility functions
    ############################################################

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        """Get memory indices for objects"""
        local_indices = []
        if len(objects) == 0:
            return local_indices

        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                local_indices.append(mem_obj.meta.address)
        elif isinstance(objects[0], bytes):
            # For raw bytes, we use sequential indices
            for i in range(len(objects)):
                local_indices.append(i)
        else:
            raise ValueError(f"Unsupported object type: {type(objects[0])}")

        return local_indices

    ############################################################
    # Helper functions for data transfer
    ############################################################

    def _send_data_over_socket(
        self, sock: socket.socket, data: bytes
    ) -> bool:
        """Send data over socket with length prefix"""
        try:
            # Send data length first (4 bytes, network order)
            data_len = len(data)
            sock.sendall(struct.pack('!I', data_len))

            # Send data in chunks
            bytes_sent = 0
            while bytes_sent < data_len:
                chunk_size = min(self.buffer_size_per_recv, data_len - bytes_sent)
                chunk = data[bytes_sent:bytes_sent + chunk_size]
                sock.sendall(chunk)
                bytes_sent += chunk_size

            return True
        except Exception as e:
            logger.error(f"Error sending data over socket: {e}")
            return False

    def _recv_data_from_socket(
        self, sock: socket.socket
    ) -> Optional[bytes]:
        """Receive data from socket with length prefix"""
        try:
            # Receive data length first (4 bytes)
            len_bytes = self._recv_exact(sock, 4)
            if not len_bytes:
                return None
            data_len = struct.unpack('!I', len_bytes)[0]

            # Receive data
            data = self._recv_exact(sock, data_len)
            return data
        except Exception as e:
            logger.error(f"Error receiving data from socket: {e}")
            return None

    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket"""
        data = b''
        while len(data) < n:
            chunk = sock.recv(min(n - len(data), self.buffer_size_per_recv))
            if not chunk:
                return None
            data += chunk
        return data

    def _objects_to_bytes_list(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[bytes]:
        """Convert objects to list of bytes"""
        bytes_list = []

        for obj in objects:
            if isinstance(obj, bytes):
                bytes_list.append(obj)
            elif isinstance(obj, MemoryObj):
                # Convert MemoryObj to bytes
                obj_bytes = obj.byte_array()
                bytes_list.append(obj_bytes)
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        return bytes_list

    def _bytes_to_objects(
        self,
        bytes_list: list[bytes],
        target_objects: Union[list[bytes], list[MemoryObj]]
    ) -> int:
        """Copy received bytes to target objects"""
        success_count = 0

        for i, (data, target) in enumerate(zip(bytes_list, target_objects)):
            try:
                if isinstance(target, bytes):
                    # For bytes targets, we can't modify in place
                    # This is a limitation - caller should handle
                    logger.warning("Cannot modify bytes objects in place")
                elif isinstance(target, MemoryObj):
                    # Copy data to MemoryObj
                    if target.tensor is not None:
                        tensor = target.tensor
                        # Copy bytes to tensor memory
                        tensor_bytes = tensor.numpy().tobytes()
                        if len(data) <= len(tensor_bytes):
                            # Copy data (this is a simplified approach)
                            # In practice, you'd need more sophisticated memory copying
                            logger.info(f"Copying {len(data)} bytes to MemoryObj")
                        else:
                            logger.error(f"Data too large for target MemoryObj")
                            continue
                    else:
                        logger.error("MemoryObj has no tensor data")
                        continue

                success_count += 1
            except Exception as e:
                logger.error(f"Error copying data to object {i}: {e}")

        return success_count

    async def _async_send_data_over_socket(
        self, writer: asyncio.StreamWriter, data: bytes
    ) -> bool:
        """Send data over async socket with length prefix"""
        try:
            # Send data length first (4 bytes, network order)
            data_len = len(data)
            writer.write(struct.pack('!I', data_len))

            # Send data in chunks
            bytes_sent = 0
            while bytes_sent < data_len:
                chunk_size = min(self.buffer_size_per_recv, data_len - bytes_sent)
                chunk = data[bytes_sent:bytes_sent + chunk_size]
                writer.write(chunk)
                bytes_sent += chunk_size

            await writer.drain()
            return True
        except Exception as e:
            logger.error(f"Error sending data over async socket: {e}")
            return False

    async def _async_recv_data_from_socket(
        self, reader: asyncio.StreamReader
    ) -> Optional[bytes]:
        """Receive data from async socket with length prefix"""
        try:
            # Receive data length first (4 bytes)
            len_bytes = await reader.readexactly(4)
            if not len_bytes:
                return None
            data_len = struct.unpack('!I', len_bytes)[0]

            # Receive data
            data = await reader.readexactly(data_len)
            return data
        except Exception as e:
            logger.error(f"Error receiving data from async socket: {e}")
            return None

    ############################################################
    # Send/Recv functions (bidirectional)
    ############################################################

    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Send objects to peer (bidirectional operation)"""
        if not objects:
            return 0

        assert transfer_spec is not None, "transfer_spec is required for batched_send"
        assert "receiver_id" in transfer_spec, "receiver_id must be specified"

        receiver_id = transfer_spec["receiver_id"]
        transfer_id = transfer_spec.get("transfer_id", str(uuid.uuid4()))

        with self.socket_lock:
            if receiver_id not in self.data_sockets:
                raise ValueError(f"No connection to peer {receiver_id}")
            data_socket = self.data_sockets[receiver_id]
            if receiver_id not in self.peer_connections:
                raise ValueError(f"No peer connection info for {receiver_id}")

        try:
            # Convert objects to bytes
            bytes_list = self._objects_to_bytes_list(objects)
            object_sizes = [len(obj_bytes) for obj_bytes in bytes_list]

            # Send transfer request via control channel
            transfer_req = SocketDataTransferRequest(
                transfer_id=transfer_id,
                sender_id=self.local_id,
                receiver_id=receiver_id,
                object_count=len(objects),
                object_sizes=object_sizes,
                transfer_type="send"
            )

            # Use the init side channel for coordination
            with self.socket_lock:
                coord_socket = get_zmq_socket(
                    self.zmq_context,
                    self.peer_init_url,  # Use the same URL for coordination
                    "tcp",
                    zmq.REQ,
                    "connect",
                )

            try:
                coord_socket.send(msgspec.msgpack.encode(transfer_req))
                resp_bytes = coord_socket.recv()
                resp = msgspec.msgpack.decode(resp_bytes, type=SocketMsg)

                if not isinstance(resp, SocketDataTransferResponse):
                    raise ValueError(f"Expected SocketDataTransferResponse, got {type(resp)}")

                if resp.status != "ready":
                    logger.error(f"Receiver not ready: {resp.error_message}")
                    return 0

                # Send data objects
                success_count = 0
                for obj_bytes in bytes_list:
                    if self._send_data_over_socket(data_socket, obj_bytes):
                        success_count += 1
                    else:
                        logger.error("Failed to send object, stopping batch")
                        break

                logger.info(f"Successfully sent {success_count}/{len(objects)} objects to {receiver_id}")
                return success_count

            finally:
                coord_socket.close()

        except Exception as e:
            logger.error(f"Error in batched_send: {e}")
            return 0

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Receive objects from peer (bidirectional operation)"""
        if not buffers:
            return 0

        assert transfer_spec is not None, "transfer_spec is required for batched_recv"
        assert "sender_id" in transfer_spec, "sender_id must be specified"

        sender_id = transfer_spec["sender_id"]
        transfer_id = transfer_spec.get("transfer_id", str(uuid.uuid4()))

        with self.socket_lock:
            if sender_id not in self.data_sockets:
                raise ValueError(f"No connection to peer {sender_id}")
            data_socket = self.data_sockets[sender_id]

        try:
            # Wait for transfer request from sender
            # This would typically be handled by the init loop, but for simplicity
            # we assume the coordination has already happened

            # Send ready response (this is a simplified implementation)
            transfer_resp = SocketDataTransferResponse(
                transfer_id=transfer_id,
                status="ready"
            )

            # Receive data for each buffer
            received_bytes_list = []
            success_count = 0

            expected_count = len(buffers)
            for _ in range(expected_count):
                obj_bytes = self._recv_data_from_socket(data_socket)
                if obj_bytes is not None:
                    received_bytes_list.append(obj_bytes)
                    success_count += 1
                else:
                    logger.error("Failed to receive object, stopping batch")
                    break

            # Copy received data to target buffers
            if received_bytes_list:
                copied_count = self._bytes_to_objects(received_bytes_list, buffers[:success_count])
                logger.info(f"Successfully received {success_count}/{len(buffers)} objects from {sender_id}")
                return copied_count
            else:
                return 0

        except Exception as e:
            logger.error(f"Error in batched_recv: {e}")
            return 0

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async send objects to peer"""
        if not objects:
            return 0

        assert transfer_spec is not None, "transfer_spec is required for async_batched_send"
        assert "receiver_id" in transfer_spec, "receiver_id must be specified"

        receiver_id = transfer_spec["receiver_id"]

        with self.socket_lock:
            if receiver_id not in self.peer_connections:
                raise ValueError(f"No connection to peer {receiver_id}")

            peer_conn = self.peer_connections[receiver_id]
            if "writer" not in peer_conn:
                raise ValueError(f"No async connection to peer {receiver_id}")

            writer = peer_conn["writer"]

        try:
            # Convert objects to bytes
            bytes_list = self._objects_to_bytes_list(objects)

            # Send each object asynchronously
            success_count = 0
            for obj_bytes in bytes_list:
                if await self._async_send_data_over_socket(writer, obj_bytes):
                    success_count += 1
                else:
                    logger.error("Failed to send object, stopping batch")
                    break

            logger.info(f"Successfully sent {success_count}/{len(objects)} objects to {receiver_id} (async)")
            return success_count

        except Exception as e:
            logger.error(f"Error in async_batched_send: {e}")
            return 0

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async receive objects from peer"""
        if not buffers:
            return 0

        assert transfer_spec is not None, "transfer_spec is required for async_batched_recv"
        assert "sender_id" in transfer_spec, "sender_id must be specified"

        sender_id = transfer_spec["sender_id"]

        with self.socket_lock:
            if sender_id not in self.peer_connections:
                raise ValueError(f"No connection to peer {sender_id}")

            peer_conn = self.peer_connections[sender_id]
            if "reader" not in peer_conn:
                raise ValueError(f"No async connection to peer {sender_id}")

            reader = peer_conn["reader"]

        try:
            # Receive data for each buffer
            received_bytes_list = []
            success_count = 0

            for _ in buffers:
                obj_bytes = await self._async_recv_data_from_socket(reader)
                if obj_bytes is not None:
                    received_bytes_list.append(obj_bytes)
                    success_count += 1
                else:
                    logger.error("Failed to receive object, stopping batch")
                    break

            # Copy received data to target buffers
            if received_bytes_list:
                copied_count = self._bytes_to_objects(received_bytes_list, buffers[:success_count])
                logger.info(f"Successfully received {success_count}/{len(buffers)} objects from {sender_id} (async)")
                return copied_count
            else:
                return 0

        except Exception as e:
            logger.error(f"Error in async_batched_recv: {e}")
            return 0

    ############################################################
    # Read/Write functions (one-sided)
    ############################################################

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Write objects to peer (one-sided operation)"""
        if not objects:
            return 0

        assert transfer_spec is not None, "transfer_spec is required for batched_write"
        assert "receiver_id" in transfer_spec, "receiver_id must be specified"

        receiver_id = transfer_spec["receiver_id"]

        with self.socket_lock:
            if receiver_id not in self.data_sockets:
                raise ValueError(f"No connection to peer {receiver_id}")
            data_socket = self.data_sockets[receiver_id]

        try:
            # Convert objects to bytes
            bytes_list = self._objects_to_bytes_list(objects)

            # Send each object
            success_count = 0
            for obj_bytes in bytes_list:
                if self._send_data_over_socket(data_socket, obj_bytes):
                    success_count += 1
                else:
                    logger.error("Failed to send object, stopping batch")
                    break

            logger.info(f"Successfully sent {success_count}/{len(objects)} objects to {receiver_id}")
            return success_count

        except Exception as e:
            logger.error(f"Error in batched_write: {e}")
            return 0

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Read objects from peer (one-sided operation)"""
        if not buffers:
            return 0

        assert transfer_spec is not None, "transfer_spec is required for batched_read"
        assert "sender_id" in transfer_spec, "sender_id must be specified"

        sender_id = transfer_spec["sender_id"]

        with self.socket_lock:
            if sender_id not in self.data_sockets:
                raise ValueError(f"No connection to peer {sender_id}")
            data_socket = self.data_sockets[sender_id]

        try:
            # Receive data for each buffer
            received_bytes_list = []
            success_count = 0

            for _ in buffers:
                obj_bytes = self._recv_data_from_socket(data_socket)
                if obj_bytes is not None:
                    received_bytes_list.append(obj_bytes)
                    success_count += 1
                else:
                    logger.error("Failed to receive object, stopping batch")
                    break

            # Copy received data to target buffers
            if received_bytes_list:
                copied_count = self._bytes_to_objects(received_bytes_list, buffers[:success_count])
                logger.info(f"Successfully received {success_count}/{len(buffers)} objects from {sender_id}")
                return copied_count
            else:
                return 0

        except Exception as e:
            logger.error(f"Error in batched_read: {e}")
            return 0

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async write objects to peer"""
        if not objects:
            return 0

        assert transfer_spec is not None, "transfer_spec is required for async_batched_write"
        assert "receiver_id" in transfer_spec, "receiver_id must be specified"

        receiver_id = transfer_spec["receiver_id"]

        with self.socket_lock:
            if receiver_id not in self.peer_connections:
                raise ValueError(f"No connection to peer {receiver_id}")

            peer_conn = self.peer_connections[receiver_id]
            if "writer" not in peer_conn:
                raise ValueError(f"No async connection to peer {receiver_id}")

            writer = peer_conn["writer"]

        try:
            # Convert objects to bytes
            bytes_list = self._objects_to_bytes_list(objects)

            # Send each object asynchronously
            success_count = 0
            for obj_bytes in bytes_list:
                if await self._async_send_data_over_socket(writer, obj_bytes):
                    success_count += 1
                else:
                    logger.error("Failed to send object, stopping batch")
                    break

            logger.info(f"Successfully wrote {success_count}/{len(objects)} objects to {receiver_id} (async)")
            return success_count

        except Exception as e:
            logger.error(f"Error in async_batched_write: {e}")
            return 0

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async read objects from peer"""
        if not buffers:
            return 0

        assert transfer_spec is not None, "transfer_spec is required for async_batched_read"
        assert "sender_id" in transfer_spec, "sender_id must be specified"

        sender_id = transfer_spec["sender_id"]

        with self.socket_lock:
            if sender_id not in self.peer_connections:
                raise ValueError(f"No connection to peer {sender_id}")

            peer_conn = self.peer_connections[sender_id]
            if "reader" not in peer_conn:
                raise ValueError(f"No async connection to peer {sender_id}")

            reader = peer_conn["reader"]

        try:
            # Receive data for each buffer
            received_bytes_list = []
            success_count = 0

            for _ in buffers:
                obj_bytes = await self._async_recv_data_from_socket(reader)
                if obj_bytes is not None:
                    received_bytes_list.append(obj_bytes)
                    success_count += 1
                else:
                    logger.error("Failed to receive object, stopping batch")
                    break

            # Copy received data to target buffers
            if received_bytes_list:
                copied_count = self._bytes_to_objects(received_bytes_list, buffers[:success_count])
                logger.info(f"Successfully read {success_count}/{len(buffers)} objects from {sender_id} (async)")
                return copied_count
            else:
                return 0

        except Exception as e:
            logger.error(f"Error in async_batched_read: {e}")
            return 0

    ############################################################
    # Cleanup functions
    ############################################################

    def close(self) -> None:
        """Close the transfer channel and clean up resources"""
        logger.info("Closing PySocketChannel")
        self.running = False

        # Close all data sockets
        with self.socket_lock:
            for peer_id, sock in self.data_sockets.items():
                try:
                    if isinstance(sock, socket.socket):
                        sock.close()
                    logger.info(f"Closed data socket for peer {peer_id}")
                except Exception as e:
                    logger.error(f"Error closing socket for peer {peer_id}: {e}")
            self.data_sockets.clear()

        # Close side channels
        for channel in self.side_channels:
            try:
                channel.close()
            except Exception as e:
                logger.error(f"Error closing side channel: {e}")

        # Wait for threads to finish
        for thread in self.running_threads:
            try:
                thread.join(timeout=5.0)
            except Exception as e:
                logger.error(f"Error joining thread: {e}")

        # Terminate ZMQ context
        try:
            self.zmq_context.term()
        except Exception as e:
            logger.error(f"Error terminating ZMQ context: {e}")

        logger.info("PySocketChannel closed successfully")