# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, Optional, Union
import asyncio
import os
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

try:
    # Mooncake Transfer Engine
    # Ref: https://kvcache-ai.github.io/Mooncake/python-api-reference/transfer-engine.html
    from mooncake.engine import TransferEngine
except ImportError as err:
    raise RuntimeError(
        "mooncake-transfer-engine is not installed. "
        "Please install it via `pip install mooncake-transfer-engine`."
    ) from err


class MooncakeMsgBase(msgspec.Struct, tag=True):
    """Base class for Mooncake-related initialization messages."""

    pass


class MooncakeInitRequest(MooncakeMsgBase):
    # ID of the local peer; used by the remote side as the key
    local_id: str

    # Local Mooncake session information (hostname:rpc_port)
    session_id: str

    # Local registered buffer information (the big pre-allocated buffer)
    buffer_base_addr: int
    buffer_len: int


class MooncakeInitResponse(MooncakeMsgBase):
    # Remote Mooncake session information
    session_id: str

    # Remote registered buffer information
    buffer_base_addr: int
    buffer_len: int


MooncakeMsg = Union[MooncakeInitRequest, MooncakeInitResponse]


@dataclass
class RemoteBufferInfo:
    session_id: str
    base_addr: int
    length: int


class MooncakeChannel(BaseTransferChannel):
    """
    Mooncake-based transfer channel.

    Design is similar to NixlChannel:
    - Each process pre-allocates a large contiguous buffer (already done by PD).
    - This buffer is registered with Mooncake TransferEngine.
    - During initialization, we use ZMQ to exchange:
        * remote session_id (hostname:rpc_port)
        * remote buffer (base_addr, length)
    - For PD, MemoryObj.meta.address is treated as a page index.
      Combined with align_bytes and buffer_ptr, we compute physical addresses
      on both local and remote sides and perform RDMA writes via Mooncake.

    Currently only batched_write is implemented, which is what PD needs.
    TODO(Yang): Support TP, async mode.
    """

    def __init__(
        self,
        async_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        # Required arguments
        # TODO(Yang): pass hostname, metadata_server, protocol, device_name from kwargs.
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "align_bytes" in kwargs
        assert "tp_rank" in kwargs
        assert "peer_init_url" in kwargs

        self.role: str = kwargs["role"]
        self.buffer_ptr: int = kwargs["buffer_ptr"]
        self.buffer_size: int = kwargs["buffer_size"]
        self.align_bytes: int = kwargs["align_bytes"]
        self.tp_rank: int = kwargs["tp_rank"]
        self.peer_init_url: Optional[str] = kwargs["peer_init_url"]

        self.async_mode: bool = async_mode
        self.running: bool = True

        # ZMQ context
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)

        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        # Initialize Mooncake TransferEngine via wrapper
        self.mooncake_wrapper = MooncakeEngineWrapper(
            buffer_ptr=self.buffer_ptr,
            buffer_size=self.buffer_size,
            hostname=kwargs.get("mooncake_hostname"),
            metadata_server=kwargs.get("mooncake_metadata_server"),
            protocol=kwargs.get("mooncake_protocol"),
            device_name=kwargs.get("mooncake_device_name"),
        )
        self.engine = self.mooncake_wrapper.engine
        self.session_id = self.mooncake_wrapper.session_id

        # Remote peers' buffer information
        # key: peer_id (e.g., receiver_id or sender_id)
        self.remote_buffers: dict[str, RemoteBufferInfo] = {}

        # Optional P2P lookup URL for controller-based peer discovery
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        # Event loop for async mode
        self.event_loop = kwargs.get("event_loop", None)

        # Start side-channel for initialization (REP side that receives REQ)
        self._init_side_channels()

    # ============================================================
    # Initialization-related methods
    # ============================================================
    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """
        Lazily initialize connection to a peer via ZMQ REQ/REP and Mooncake.

        Steps:
        1. Create a temporary ZMQ REQ socket to peer_init_url.
        2. Send MooncakeInitRequest with:
            - local_id
            - local session_id
            - local buffer base address and length
        3. Receive MooncakeInitResponse and store remote buffer info keyed by peer_id.
        4. Optionally send an InitSideMsgBase (e.g., P2PInitSideMsg) and get reply.
        """
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        init_req = MooncakeInitRequest(
            local_id=local_id,
            session_id=self.session_id,
            buffer_base_addr=self.buffer_ptr,
            buffer_len=self.buffer_size,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(init_req))

        resp_bytes = init_tmp_socket.recv()
        resp = msgspec.msgpack.decode(resp_bytes, type=MooncakeMsg)
        assert isinstance(resp, MooncakeInitResponse)

        # Store remote buffer info
        self.remote_buffers[peer_id] = RemoteBufferInfo(
            session_id=resp.session_id,
            base_addr=resp.buffer_base_addr,
            length=resp.buffer_len,
        )

        logger.info(
            "MooncakeChannel: initialized peer connection. "
            f"local_id={local_id}, peer_id={peer_id}, "
            f"peer_session={resp.session_id}, "
            f"peer_buffer=({resp.buffer_base_addr}, len={resp.buffer_len})"
        )

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = self.send_init_side_msg(init_tmp_socket, init_side_msg)

        init_tmp_socket.close()
        return init_ret_msg

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        """
        Async wrapper for lazy_init_peer_connection.

        For now, PD uses the synchronous path, but we keep this for API completeness.
        """
        return self.lazy_init_peer_connection(
            local_id=local_id,
            peer_id=peer_id,
            peer_init_url=peer_init_url,
            init_side_msg=init_side_msg,
        )

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        """
        For Mooncake, this means we have already finished the init handshake
        and recorded the remote buffer information for this peer.
        """
        return receiver_or_sender_id in self.remote_buffers

    def _init_side_channels(self) -> None:
        """
        Start the initialization side channel (REP side) if peer_init_url is set.
        """
        if self.peer_init_url is None:
            return

        if self.async_mode:
            assert self.event_loop is not None, (
                "async_mode=True requires an event_loop in MooncakeChannel."
            )
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

    def _handle_init_msg(
        self,
        req: Union[MooncakeMsg, InitSideMsgBase],
    ) -> Union[MooncakeMsg, InitSideRetMsgBase]:
        """
        Handle:
        - MooncakeInitRequest: record remote buffer info and reply with local info.
        - InitSideMsgBase: delegate to BaseTransferChannel.handle_init_side_msg.
        """
        if isinstance(req, MooncakeInitRequest):
            # Record remote buffer info keyed by req.local_id
            # (this will later be used as sender_id / receiver_id in PD)
            self.remote_buffers[req.local_id] = RemoteBufferInfo(
                session_id=req.session_id,
                base_addr=req.buffer_base_addr,
                length=req.buffer_len,
            )

            logger.info(
                "MooncakeChannel: received init request. "
                f"remote_id={req.local_id}, "
                f"session_id={req.session_id}, "
                f"buffer=({req.buffer_base_addr}, len={req.buffer_len})"
            )

            # Respond with local session and buffer info
            resp = MooncakeInitResponse(
                session_id=self.session_id,
                buffer_base_addr=self.buffer_ptr,
                buffer_len=self.buffer_size,
            )
            return resp

        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("MooncakeChannel: replying P2P init side response")
            return resp

        else:
            raise ValueError(f"Unsupported init msg type: {type(req)}")

    def _init_loop(self) -> None:
        """
        Blocking REP loop for handling initialization messages.
        """
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)
        logger.info(
            "MooncakeChannel: init loop listening on %s", self.peer_init_url
        )

        while self.running:
            try:
                req_bytes = self.init_side_channel.recv()
                logger.info("MooncakeChannel: received init request")
                req = msgspec.msgpack.decode(
                    req_bytes,
                    type=Union[MooncakeMsg, SideMsg],
                )

                resp = self._handle_init_msg(req)  # type: ignore[arg-type]
                self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("MooncakeChannel init loop error: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    async def _async_init_loop(self) -> None:
        """
        Async REP loop for handling initialization messages.
        """
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)
        logger.info(
            "MooncakeChannel: async init loop listening on %s", self.peer_init_url
        )

        while self.running:
            try:
                req_bytes = await self.init_side_channel.recv()
                logger.info("MooncakeChannel: received init request (async)")
                req = msgspec.msgpack.decode(
                    req_bytes,
                    type=Union[MooncakeMsg, SideMsg],
                )

                resp = self._handle_init_msg(req)  # type: ignore[arg-type]
                await self.init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("MooncakeChannel async init loop error: %s", str(e))
                if self.running:
                    await asyncio.sleep(0.01)

    # ============================================================
    # Utility methods
    # ============================================================
    def get_local_mem_indices(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
    ) -> list[int]:
        """
        For PD, MemoryObj.meta.address is treated as the page index.

        This function returns a list of indices corresponding to the objects.
        """
        if not objects:
            return []
        if isinstance(objects[0], MemoryObj):
            return [mem_obj.meta.address for mem_obj in objects]  # type: ignore[attr-defined]
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in MooncakeChannel"
            )
        else:
            raise TypeError(
                f"Unsupported object type in get_local_mem_indices: {type(objects[0])}"
            )

    # ============================================================
    # Send / Recv (not used by PD yet)
    # ============================================================
    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError(
            "batched_send is not implemented for MooncakeChannel yet"
        )

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError(
            "batched_recv is not implemented for MooncakeChannel yet"
        )

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError(
            "async_batched_send is not implemented for MooncakeChannel yet"
        )

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError(
            "async_batched_recv is not implemented for MooncakeChannel yet"
        )

    # ============================================================
    # Read / Write (PD uses batched_write)
    # ============================================================
    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of MemoryObjs to the remote peer via Mooncake.

        Expected transfer_spec fields:
        - 'receiver_id': remote peer ID (same as peer_id passed to lazy_init_peer_connection)
        - 'remote_indexes': list of remote page indices (one per object)

        For each MemoryObj:
        - local_index  = mem_obj.meta.address
        - remote_index = remote_indexes[i]
        - local_addr   = buffer_ptr + local_index  * align_bytes
        - remote_addr  = remote_base + remote_index * align_bytes
        - length       = align_bytes (full page transfer)

        Then call:
        engine.transfer_sync_write(
            target_session_id,
            local_addr,
            remote_addr,
            length,
        )
        """
        assert transfer_spec is not None, "transfer_spec is required for batched_write"

        receiver_id: str = transfer_spec["receiver_id"]
        remote_indexes: list[int] = transfer_spec["remote_indexes"]

        assert receiver_id in self.remote_buffers, (
            f"Receiver {receiver_id} is not initialized in MooncakeChannel. "
            "Did you call lazy_init_peer_connection first?"
        )

        remote_info = self.remote_buffers[receiver_id]
        target_session = remote_info.session_id
        remote_base = remote_info.base_addr

        if not objects:
            return 0

        if not isinstance(objects[0], MemoryObj):
            raise NotImplementedError(
                "MooncakeChannel.batched_write currently only supports MemoryObj"
            )

        mem_objs: list[MemoryObj] = objects  # type: ignore[assignment]
        assert len(mem_objs) == len(
            remote_indexes
        ), "objects and remote_indexes must have the same length"

        for idx, mem_obj in enumerate(mem_objs):
            local_page_index = mem_obj.meta.address
            remote_page_index = remote_indexes[idx]

            local_addr = self.buffer_ptr + local_page_index * self.align_bytes
            remote_addr = remote_base + remote_page_index * self.align_bytes
            length = self.align_bytes

            ret = self.engine.transfer_sync_write(
                target_session,
                local_addr,
                remote_addr,
                length,
            )
            if ret < 0:
                logger.error(
                    "MooncakeChannel.batched_write failed for object %d: "
                    "local_page_index=%d, remote_page_index=%d, ret=%d",
                    idx,
                    local_page_index,
                    remote_page_index,
                    ret,
                )
                raise RuntimeError(
                    f"Mooncake transfer_sync_write failed with code {ret}"
                )

        return len(mem_objs)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Placeholder for reading a batch of data from a remote peer.

        If future use cases require reading from a remote Mooncake buffer,
        this method can be implemented using TransferEngine.transfer_sync_read.
        """
        raise NotImplementedError(
            "batched_read is not implemented for MooncakeChannel yet"
        )

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async wrapper for batched_write.

        This simply calls the synchronous implementation for now.
        """
        return self.batched_write(objects, transfer_spec)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError(
            "async_batched_read is not implemented for MooncakeChannel yet"
        )

    # ============================================================
    # Cleanup
    # ============================================================
    def close(self) -> None:
        """
        Close the transfer channel and release resources:
        - stop threads;
        - close ZMQ sockets and context;
        - unregister the registered buffer from Mooncake.
        """
        self.running = False
        for thread in self.running_threads:
            thread.join()

        for socket in self.side_channels:
            socket.close()
        self.zmq_context.term()

        # Unregister memory via wrapper
        self.mooncake_wrapper.unregister_memory()


@dataclass
class MooncakeEngineWrapper:
    engine: TransferEngine
    session_id: str
    buffer_ptr: int
    buffer_size: int

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        hostname: Optional[str] = None,
        metadata_server: Optional[str] = None,
        protocol: Optional[str] = None,
        device_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the Mooncake TransferEngine.

        Args:
            buffer_ptr (int): The pointer to the pre-allocated buffer.
            buffer_size (int): The size of the buffer.
            hostname (str, optional): Local hostname. Defaults to env var or "localhost".
            metadata_server (str, optional): Metadata server. Defaults to env var or "P2PHANDSHAKE".
            protocol (str, optional): Protocol to use. Defaults to env var or "tcp".
            device_name (str, optional): Device name. Defaults to env var or "".
        """
        # Configuration: kwargs override environment variables
        hostname = hostname or os.getenv("MC_LOCAL_HOSTNAME", "localhost")
        metadata_server = metadata_server or os.getenv("MC_METADATA_SERVER", "P2PHANDSHAKE")
        protocol = protocol or os.getenv("MC_PROTOCOL", "tcp")
        device_name = device_name or os.getenv("MC_DEVICE_NAME", "")

        logger.info(
            "Initializing Mooncake TransferEngine: "
            f"hostname={hostname}, metadata_server={metadata_server}, "
            f"protocol={protocol}, device_name={device_name}"
        )

        # Create and initialize TransferEngine
        engine = TransferEngine()
        ret = engine.initialize(hostname, metadata_server, protocol, device_name)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake TransferEngine.initialize failed with code {ret}"
            )

        # Get RPC port and create session_id
        rpc_port = engine.get_rpc_port()
        session_id = f"{hostname}:{rpc_port}"

        # Register the buffer
        ret = engine.register_memory(buffer_ptr, buffer_size)
        if ret != 0:
            raise RuntimeError(
                "Mooncake register_memory failed with code "
                f"{ret} (buffer_ptr={buffer_ptr}, size={buffer_size})"
            )

        logger.info(
            "MooncakeEngineWrapper initialized: "
            f"session_id={session_id}, "
            f"buffer_ptr={buffer_ptr}, buffer_size={buffer_size}"
        )

        self.engine = engine
        self.session_id = session_id
        self.buffer_ptr = buffer_ptr
        self.buffer_size = buffer_size

    def unregister_memory(self) -> None:
        """Unregister the registered buffer from Mooncake."""
        try:
            ret = self.engine.unregister_memory(self.buffer_ptr)
            if ret != 0:
                logger.warning(
                    "MooncakeEngineWrapper.unregister_memory failed with code %d",
                    ret,
                )
        except Exception as e:
            logger.error(
                "MooncakeEngineWrapper.unregister_memory error: %s", e
            )

