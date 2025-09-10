# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Any, Optional, Union
import copy
import threading
import time
import uuid

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfigXpYd, NixlRole

if TYPE_CHECKING:
    # Third Party
    from nixl._api import NixlAgent

    # First Party
    from lmcache.v1.storage_backend.nixl_backend_v3 import NixlBackend

logger = init_logger(__name__)


class NixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all nixl-related messages"""

    pass


class NixlAllocRequest(NixlMsgBase):
    """Nixl allocation request message"""

    keys: list[str]  # len(keys) indicates num_chunks
    fmt: int
    shape: list[int]  # The shape of the memory objects
    dtype: str
    last_chunk_toks: int


class NixlAllocResponse(NixlMsgBase):
    """Nixl allocation response message"""

    # Indexes (local) of already sent memory objects
    already_sent_indexes: list[int]

    remote_indexes: list[int]


class NixlInitRequest(NixlMsgBase):
    sender_meta_bytes: bytes  # Metadata from the sender nixl agent


class NixlMemRegRequest(NixlMsgBase):
    pass


class NixlInitResponse(NixlMsgBase):
    receiver_meta_bytes: bytes  # Metadata from the receiver nixl agent


class NixlMemRegResponse(NixlMsgBase):
    receiver_xfer_dlist_bytes: bytes  # Serialized transfer descriptors for the receiver


class NixlProxyNotif(NixlMsgBase):
    req_id: str  # The request UUID to notify the proxy


NixlMsg = Union[
    NixlAllocRequest,
    NixlAllocResponse,
    NixlProxyNotif,
    NixlInitRequest,
    NixlInitResponse,
    NixlMemRegRequest,
    NixlMemRegResponse,
]


@dataclass
class NixlReceiverInfo:
    receiver_id: str
    receiver_host: Optional[str] = None
    receiver_init_port: Optional[int] = None
    receiver_alloc_port: Optional[int] = None


# no need to be msgspec
@dataclass
class NixlSenderTask:
    req_id: str
    receiver_info: NixlReceiverInfo
    keys: list[CacheEngineKey]  # The keys to send
    mem_objs: list[MemoryObj]  # The memory objects to send

    def get_alloc_request(self) -> NixlAllocRequest:
        """
        Get the allocation request for this sender task.

        Let's say there are N memory objects in total.
        We have the following assumptions:
        - The first N-1 memory objects are full chunks, each with
        `full_chunk_size` tokens.
        - The last memory object can be a partial chunk, which has
        `last_chunk_toks` tokens.
        """

        fmt = self.mem_objs[0].meta.fmt
        shape = self.mem_objs[0].meta.shape
        dtype = TORCH_DTYPE_TO_STR_DTYPE[self.mem_objs[0].meta.dtype]
        token_dim = fmt.token_dim()
        last_chunk_toks = self.mem_objs[-1].meta.shape[token_dim]

        # TODO(Jiayi): Reomove this for loop
        keys = [key.to_string() for key in self.keys]

        return NixlAllocRequest(
            keys=keys,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
        )

    # TODO (Jiayi): reduce for loop
    def get_local_indexes(
        self,
        already_sent_indexes: list[int],
    ) -> list[int]:
        """
        Get the page indexes of the memory objects.
        This is needed for nixl transfer.
        """
        local_indexes = []
        for idx, mem_obj in enumerate(self.mem_objs):
            if idx in already_sent_indexes:
                continue
            local_indexes.append(mem_obj.meta.address)
        return local_indexes

    def free_mem_objs(self):
        for mem_obj in self.mem_objs:
            mem_obj.ref_count_down()


class NixlSender:
    """Handles sending data through a NixlPipe."""

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: "NixlBackend",
        tp_rank: int,
    ):
        assert nixl_config.role == NixlRole.SENDER, (
            "NixlSender should only be initialized with NixlRole.SENDER"
        )

        self.device = nixl_config.buffer_device

        self.nixl_config = nixl_config

        self.memory_allocator = backend.memory_allocator

        self._sender_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.nixl_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.nixl_allocator.buffer_size,
            page_size=self.memory_allocator.nixl_allocator.align_bytes,
            tp_rank=tp_rank,
        )
        self._nixl_agent = self._sender_nixl_wrapper.agent

        # Initialize the ZeroMQ context
        self._context = zmq.Context()

        self._mem_alloc_sockets: dict[str, zmq.Socket] = {}

        self.req_queue: Queue[NixlSenderTask] = Queue()

        self._remote_xfer_handlers_dict: dict[
            str, NixlAgent.nixl_prepped_dlist_handle
        ] = {}

        # Start the seder thread
        self._running = True

        # self._sender_thread = threading.Thread(
        #     target=self._sender_loop, daemon=True
        # )
        # self._sender_thread.start()

        proxy_host = nixl_config.proxy_host
        proxy_port = nixl_config.proxy_port
        proxy_url = f"{proxy_host}:{proxy_port}"

        self._proxy_side_channel = self._context.socket(zmq.PUSH)
        self._proxy_side_channel.connect(get_zmq_path(proxy_url, protocol="tcp"))

        self.tp_rank = tp_rank

    def prepare_send(
        self,
        keys: list[CacheEngineKey],
        mem_objs: list[MemoryObj],
        transfer_spec=None,
    ):
        """
        Put the sender task into the request queue.
        """

        receiver_info = copy.deepcopy(transfer_spec.receiver_info)
        receiver_info.receiver_init_port = (
            transfer_spec.receiver_info.receiver_init_port[self.tp_rank]
        )
        receiver_info.receiver_alloc_port = (
            transfer_spec.receiver_info.receiver_alloc_port[self.tp_rank]
        )
        receiver_info.receiver_id = transfer_spec.receiver_info.receiver_host + str(
            receiver_info.receiver_init_port
        )

        sender_task = NixlSenderTask(
            req_id=transfer_spec.req_id,
            receiver_info=receiver_info,
            keys=keys,
            mem_objs=mem_objs,
        )

        logger.debug(
            "Preparing to send %s objs with request ID: %s to receiver: %s",
            len(sender_task.keys),
            sender_task.req_id,
            receiver_info,
        )

        # self.req_queue.put(sender_task)

        req_id = sender_task.req_id
        receiver_id = receiver_info.receiver_id

        # NOTE (Jiayi): Currently, a sender needs to connect to
        # 3 side channels:
        # (1) _init_side_channel (ad-hoc-established and destroyed
        # after nixl connection is established),
        # (2) _alloc_side_channel (ad-hoc-established),
        # (3) _proxy_side_channel (pre-established).
        # NOTE (Jiayi): In addition, a sender also needs to
        # initialize nixl connection.

        # NOTE (Jiayi): `_init_all_comm` checks and initializes
        # _alloc_side_channel and nixl connection.
        if not self._check_init(receiver_info):
            self._init_all_comm(receiver_info)

        # use remote alloc
        alloc_request = sender_task.get_alloc_request()

        alloc_response = self._remote_allocate(receiver_id, alloc_request)

        # send kv
        local_indexes = sender_task.get_local_indexes(
            alloc_response.already_sent_indexes
        )
        remote_indexes = alloc_response.remote_indexes

        # NOTE (vladnosiv): len(local_indexes) may be zero
        # if the requests in the batch have a large common prefix
        if not local_indexes:
            logger.debug(
                "Sending objs with request ID: %s is not required: "
                "all indexes already sent",
                sender_task.req_id,
            )
        else:
            self._blocking_send(req_id, receiver_id, local_indexes, remote_indexes)

        logger.debug(f"transfer spec: {transfer_spec}")
        if transfer_spec.is_last_prefill:
            # Notify the proxy that the transfer is done
            notif_msg = NixlProxyNotif(req_id=req_id)
            notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
            self._proxy_side_channel.send(notif_msg_bytes)

        # free local memory
        sender_task.free_mem_objs()

    def _remote_allocate(
        self, receiver_id: str, alloc_request: NixlAllocRequest
    ) -> NixlAllocResponse:
        """Send the allocation request to the remote peer and get the response."""

        logger.debug(
            "Sent allocation request to receiver %s with %s objs needed",
            receiver_id,
            len(alloc_request.keys),  # Use the first key as the request ID
        )

        side_channel = self._mem_alloc_sockets[receiver_id]

        side_channel.send(msgspec.msgpack.encode(alloc_request))
        msg = side_channel.recv()
        alloc_response = msgspec.msgpack.decode(msg, type=NixlMsg)

        assert isinstance(alloc_response, NixlAllocResponse), (
            "The response from the remote peer is not a NixlAllocResponse"
        )

        logger.debug("Received allocation response.")

        return alloc_response

    @_lmcache_nvtx_annotate
    def _blocking_send(
        self,
        req_id: str,
        receiver_id: str,
        local_indexes: list[int],
        remote_indexes: list[int],
    ):
        """
        Send the KV cache in a blocking manner.
        """
        logger.debug(
            "Blocking send %s objs to receiver %s with request ID: %s",
            len(local_indexes),
            receiver_id,
            req_id,
        )

        handle = self._nixl_agent.make_prepped_xfer(
            "WRITE",
            self._sender_nixl_wrapper.xfer_handler,
            local_indexes,
            self._remote_xfer_handlers_dict[receiver_id],
            remote_indexes,
            # notif_msg_bytes,
        )

        # NOTE (Jiayi): cannot make this transfer in another thread,
        # giving error: `UCX  ERROR cuCtxGetDevice(&key.cu_device)
        # failed: invalid device context`
        self._nixl_agent.transfer(handle)

        # TODO (Jiayi): offload the following to another thread
        # TODO (Jiayi) tune hyperparameters
        wait_time = 0.0007
        decay = 1.1
        while True:
            status = self._nixl_agent.check_xfer_state(handle)
            logger.debug(f"Transfer status: {status}")

            if status == "ERR":
                logger.error("Error in send operation")
                raise RuntimeError("Failed to send data to remote peer")
            elif status == "PROC":
                time.sleep(wait_time)  # Avoid busy waiting
                wait_time /= decay
                continue
            assert status == "DONE", f"Transfer status is {status}, expected DONE"
            # self._proxy_side_channel.send(notif_msg_bytes)
            break

    def _initialize_nixl_sender_connection(
        self,
        receiver_id: str,
        receiver_init_url: str,
    ) -> None:
        """
        Initialize the NIXL sender connection with the receiver.
        """

        # Exchange nixl metadata
        init_tmp_socket = self._context.socket(zmq.REQ)
        init_tmp_socket.connect(get_zmq_path(receiver_init_url, protocol="tcp"))

        nixl_init_req = NixlInitRequest(
            sender_meta_bytes=self._nixl_agent.get_agent_metadata(),
        )

        init_tmp_socket.send(msgspec.msgpack.encode(nixl_init_req))

        nixl_init_resp_bytes = init_tmp_socket.recv()

        nixl_init_resp = msgspec.msgpack.decode(nixl_init_resp_bytes, type=NixlMsg)

        remote_meta_bytes = nixl_init_resp.receiver_meta_bytes
        remote_agent_name = self._nixl_agent.add_remote_agent(remote_meta_bytes)

        # Register memory
        nixl_mem_reg_req = NixlMemRegRequest()
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_mem_reg_req))
        nixl_mem_reg_resp_bytes = init_tmp_socket.recv()
        nixl_mem_reg_resp = msgspec.msgpack.decode(
            nixl_mem_reg_resp_bytes, type=NixlMsg
        )
        remote_xfer_dlist_bytes = nixl_mem_reg_resp.receiver_xfer_dlist_bytes
        remote_xfer_dlist = self._nixl_agent.deserialize_descs(remote_xfer_dlist_bytes)
        remote_xfer_handlers = self._nixl_agent.prep_xfer_dlist(
            remote_agent_name, remote_xfer_dlist
        )
        self._remote_xfer_handlers_dict[receiver_id] = remote_xfer_handlers

        init_tmp_socket.close()

    def _initialize_mem_alloc_side_channel(
        self, receiver_id: str, receiver_mem_alloc_url: str
    ) -> None:
        """
        Initialize zmq connection for memory allocation.
        """
        mem_alloc_socket = self._context.socket(zmq.REQ)

        mem_alloc_socket.connect(get_zmq_path(receiver_mem_alloc_url, protocol="tcp"))

        self._mem_alloc_sockets[receiver_id] = mem_alloc_socket

    def _check_init(self, receiver_info: NixlReceiverInfo):
        receiver_id = receiver_info.receiver_id
        return (
            receiver_id in self._remote_xfer_handlers_dict
            and receiver_id in self._mem_alloc_sockets
        )

    def _init_all_comm(
        self,
        receiver_info: NixlReceiverInfo,
    ):
        """
        Initialize all communication channels with the receiver.
        """
        logger.debug(
            "Initializing all communication channels with receiver %s",
            receiver_info,
        )

        receiver_id = receiver_info.receiver_id
        receiver_host = receiver_info.receiver_host
        receiver_init_port = receiver_info.receiver_init_port
        receiver_alloc_port = receiver_info.receiver_alloc_port

        receiver_init_url = f"{receiver_host}:{receiver_init_port}"
        receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"

        # Initialize the nixl sender connection
        self._initialize_nixl_sender_connection(receiver_id, receiver_init_url)

        # Initialize the memory allocation side channel
        self._initialize_mem_alloc_side_channel(receiver_id, receiver_mem_alloc_url)

    def close(self):
        """Close the sender resources."""
        # Wait for the receiver thread to finish with timeout
        # self._sender_thread.join(timeout=3.0)  # 3 second timeout

        # self._running = False
        # if self._sender_thread.is_alive():
        #     logger.warning(
        #         "Sender thread did not shut down cleanly within timeout"
        #     )

        for s in self._mem_alloc_sockets.values():
            s.close()
        self._context.term()

        self._sender_nixl_wrapper.close(self._remote_xfer_handlers_dict)


class NixlReceiver:
    """Handles receiving data through a NixlPipe."""

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: "NixlBackend",
        tp_rank: int,
    ):
        assert nixl_config.role == NixlRole.RECEIVER, (
            "NixlReceiver should only be initialized with NixlRole.RECEIVER"
        )

        self._backend = backend
        self.memory_allocator = backend.memory_allocator

        self.device = nixl_config.buffer_device
        self._receiver_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.nixl_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.nixl_allocator.buffer_size,
            page_size=self.memory_allocator.nixl_allocator.align_bytes,
            tp_rank=tp_rank,
            backends=nixl_config.backends,
        )

        self._nixl_agent = self._receiver_nixl_wrapper.agent

        self.nixl_config = nixl_config

        receiver_host = nixl_config.peer_host
        receiver_init_port = nixl_config.peer_init_port[tp_rank]
        receiver_alloc_port = nixl_config.peer_alloc_port[tp_rank]

        receiver_init_url = f"{receiver_host}:{receiver_init_port}"
        receiver_alloc_url = f"{receiver_host}:{receiver_alloc_port}"

        self.full_chunk_size = config.chunk_size

        # TODO (Jiayi)" make it async?"
        # Initialize the ZeroMQ context and side channel
        self._context = zmq.Context()  # type: ignore

        self._side_channels = []

        # TODO (Jiayi): have a util func to do this
        # Create/listen initialization side channel
        self._init_side_channel = self._context.socket(zmq.REP)
        self._init_side_channel.bind(get_zmq_path(receiver_init_url, protocol="tcp"))
        self._side_channels.append(self._init_side_channel)

        # Create/listen allocation side channel
        self._alloc_side_channel = self._context.socket(zmq.REP)
        self._alloc_side_channel.bind(get_zmq_path(receiver_alloc_url, protocol="tcp"))
        self._side_channels.append(self._alloc_side_channel)

        # TODO: might be better to put them into one thread
        # and use asyncio to manage.
        # Start the receiver threads
        self._running = True
        self._running_threads = []

        self._mem_alloc_thread = threading.Thread(
            target=self._mem_alloc_loop, daemon=True
        )
        self._mem_alloc_thread.start()
        self._running_threads.append(self._mem_alloc_thread)

        self._init_thread = threading.Thread(target=self._init_loop, daemon=True)
        self._init_thread.start()
        self._running_threads.append(self._init_thread)

    def _allocate_and_put(self, alloc_request: NixlAllocRequest) -> NixlAllocResponse:
        total_allocs = len(alloc_request.keys)
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = alloc_request.shape

        alloc_indexes = []
        already_send_indexes = []

        for idx, key_str in enumerate(alloc_request.keys):
            key = CacheEngineKey.from_string(key_str)
            if self._backend.contains(key, pin=True):
                already_send_indexes.append(idx)
                continue

            if idx == total_allocs - 1:
                num_alloc_tokens = alloc_request.last_chunk_toks
                token_dim = fmt.token_dim()
                shape[token_dim] = num_alloc_tokens
            else:
                num_alloc_tokens = self.full_chunk_size

            mem_obj = self._backend.allocate(torch.Size(shape), dtype, fmt)

            # TODO(Jiayi): tune this hyperparameters
            wait_time = 0.01
            decay = 1.1
            while mem_obj is None:
                logger.warning(
                    "Failed to allocate memory object, retrying...",
                )
                time.sleep(wait_time)
                wait_time /= decay
                mem_obj = self._backend.allocate(torch.Size(shape), dtype, fmt)

            alloc_indexes.append(mem_obj.meta.address)

            self._backend.put(key, mem_obj)

        return NixlAllocResponse(
            already_sent_indexes=already_send_indexes, remote_indexes=alloc_indexes
        )

    # TODO: have a loop wrapper to wrap different loops
    def _mem_alloc_loop(self):
        """ """
        torch.cuda.set_device(self.device)
        # TODO: `self._running` might not be safe here
        while self._running:
            try:
                # NOTE: this is a req-reply zmq for now
                # receive alloc request
                alloc_req_bytes = self._alloc_side_channel.recv()
                alloc_req = msgspec.msgpack.decode(alloc_req_bytes, type=NixlMsg)
                assert isinstance(alloc_req, NixlAllocRequest), (
                    "The request from the remote peer is not a NixlAllocRequest"
                )

                logger.debug(
                    "Received allocation request for %s objs",
                    len(alloc_req.keys),
                )

                # NOTE: it's okay to put the memory objs into the storage backend
                # first because decode vllm will not be able to see the decode
                # request until proxy receives the ack.
                alloc_resp = self._allocate_and_put(alloc_req)

                logger.debug(
                    "Replying allocation response for %s objs",
                    len(alloc_resp.remote_indexes),
                )

                # send back response
                self._alloc_side_channel.send(msgspec.msgpack.encode(alloc_resp))

            except zmq.Again as e:  # type: ignore
                # Handle the timeout when waiting for a message
                logger.debug(
                    "Timeout waiting for a message on the side channel: %s",
                    str(e),
                )
                continue
            except Exception as e:
                logger.error("Failed to process mem alloc loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)

    def _init_loop(self):
        local_meta = self._nixl_agent.get_agent_metadata()

        # NOTE: Initialization has to be two stages:
        # (1) Exchanging the metadata.
        # (2) Registering the memory descriptors.
        # Otherwise, there's a chance that nixl got stuck
        # (handle always give "PROC" status) during the first request.
        while self._running:
            try:
                req_bytes = self._init_side_channel.recv()

                logger.debug("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=NixlMsg)

                if isinstance(req, NixlInitRequest):
                    self._nixl_agent.add_remote_agent(req.sender_meta_bytes)

                    resp = NixlInitResponse(
                        receiver_meta_bytes=local_meta,
                    )

                    logger.debug("Replying initialization response")

                elif isinstance(req, NixlMemRegRequest):
                    local_xfer_descs = self._nixl_agent.get_serialized_descs(
                        self._receiver_nixl_wrapper.xfer_descs
                    )

                    resp = NixlMemRegResponse(
                        receiver_xfer_dlist_bytes=local_xfer_descs,
                    )

                    logger.debug("Replying mem register response")

                self._init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)

    def close(self):
        """Close the receiver resources."""
        self._running = False

        for t in self._running_threads:
            # Wait for the receiver thread to finish with timeout
            t.join(timeout=3.0)  # 3 second timeout

            if t.is_alive():
                logger.warning(
                    "Receiver thread did not shut down cleanly within timeout"
                )
        for side_channel in self._side_channels:
            side_channel.close()
        self._context.term()

        self._receiver_nixl_wrapper.close()


class NixlChannel:
    """Provides the primitives to send the data and process the received data.
    It will have some internal threads to handle the data receiving.
    """

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: "NixlBackend",
    ):
        self.nixl_config = nixl_config
        self.role = nixl_config.role

        # Create sender or receiver based on role
        self._sender = None
        self._receiver = None

        self._backend = backend

        # Third Party
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        tp_rank = get_tensor_model_parallel_rank()

        if nixl_config.role == NixlRole.SENDER:
            self._sender = NixlSender(nixl_config, config, backend, tp_rank)
        else:
            self._receiver = NixlReceiver(nixl_config, config, backend, tp_rank)

    def _check_sender(self):
        """Check if this channel is configured as a sender."""
        if self._sender is None:
            raise RuntimeError(f"Cannot perform sender operation with role {self.role}")
        return self._sender

    def _check_receiver(self):
        """Check if this channel is configured as a receiver."""
        if self._receiver is None:
            raise RuntimeError(
                f"Cannot perform receiver operation with role {self.role}"
            )
        return self._receiver

    def prepare_send(
        self,
        keys: list[CacheEngineKey],
        mem_objs: list[MemoryObj],
        transfer_spec=None,
    ):
        """Prepare a send transaction by sending the request using
        the side channel.
        """
        sender = self._check_sender()
        sender.prepare_send(keys, mem_objs, transfer_spec)

    def close(self):
        """Close all resources."""
        if self._sender:
            self._sender.close()
        if self._receiver:
            self._receiver.close()


############################################################
# helper functions
############################################################


# TODO (Jiayi): support multiple protocols
def get_zmq_path(url: str, protocol: str = "tcp") -> str:
    """Get the ZeroMQ path for the given base path and suffix."""
    if protocol == "tcp":
        return f"tcp://{url}"
    raise ValueError(f"Unsupported protocol: {protocol}")


@dataclass
class NixlAgentWrapper:
    agent: "NixlAgent"
    reg_descs: Any
    xfer_descs: Any
    xfer_handler: Any

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        tp_rank: int,
        backends: Optional[list[str]] = None,
    ):
        """
        Initialize the NIXL agent.

        Args:
            buffer_size (int): The size of the buffer.
            buffer_ptr (int): The pointer to the buffer.
            page_size (int): The page size of NIXL and
                the lmcache memory allocator.
            tp_rank (int): The tensor parallel rank.

        Returns:
            NixlWrapper: The NIXL agent.
            reg_dlist: the registered memory descriptor list.
            xfer_dlist: the local transfer descriptor list.
            prepped_xfer_handler: the prepped transfer handler.
        """
        try:
            # Third Party
            from nixl._api import nixl_agent as NixlAgent
            from nixl._api import nixl_agent_config
        except ImportError as err:
            raise RuntimeError("NIXL is not available") from err

        # Handle None backends by setting default to ["UCX"]
        if backends is None:
            backends = ["UCX"]

        # Create a NIXL agent
        nixl_agent = NixlAgent(
            str(uuid.uuid4()),
            nixl_agent_config(backends=backends),
        )

        # Register the memory
        memory_desc = [(buffer_ptr, buffer_size, tp_rank, "")]
        # TODO(Jiayi): remove hardcode `mem_type`
        reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type="cuda")
        nixl_agent.register_memory(reg_descs)

        # Create xfer handlers
        xfer_desc = []
        for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
            xfer_desc.append((base_addr, page_size, tp_rank))

        xfer_descs = nixl_agent.get_xfer_descs(xfer_desc, mem_type="cuda")
        xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs, mem_type="cuda")

        self.agent = nixl_agent
        self.reg_descs = reg_descs
        self.xfer_descs = xfer_descs
        self.xfer_handler = xfer_handler

    def close(self, remote_xfer_handlers: Optional[dict[str, Any]] = None):
        self.agent.deregister_memory(self.reg_descs)

        self.agent.release_dlist_handle(self.xfer_handler)

        for remote_xfer_handler in self.agent._remote_xfer_handlers_dict.values():
            self.agent.release_dlist_handle(remote_xfer_handler)

        if remote_xfer_handlers is not None:
            for remote_xfer_handler in remote_xfer_handlers.values():
                self.agent.release_dlist_handle(remote_xfer_handler)
