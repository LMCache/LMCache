# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
import threading
import time

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.transfer_channel import CreateTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


class PDMsgBase(msgspec.Struct, tag=True):
    """Base class for all PD-related messages"""

    pass


class AllocRequest(PDMsgBase):
    """Allocation request message"""

    keys: list[str]  # len(keys) indicates num_chunks
    fmt: int
    shape: list[int]  # The shape of the memory objects
    dtype: str
    last_chunk_toks: int


class AllocResponse(PDMsgBase):
    """Allocation response message"""

    # Indexes (local) of already sent memory objects
    already_sent_indexes: list[int]

    # Indexes (remote) of allocated memory objects (to be written)
    remote_indexes: list[int]


class ProxyNotif(PDMsgBase):
    req_id: str  # The request UUID to notify the proxy


PDMsg = Union[AllocRequest, AllocResponse, ProxyNotif]


@dataclass
class PDConfig:
    role: str

    peer_host: str
    peer_init_port: int
    peer_alloc_port: int

    proxy_host: str
    proxy_port: int

    buffer_size: int
    buffer_device: str

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        tp_rank: int,
    ) -> "PDConfig":
        """Convert the LMCacheEngineConfig to PDConfig"""

        role = config.pd_role

        # TODO(Jiayi): Could be both if we want to do dynamic role switch.
        assert role in ["sender", "receiver"], (
            f"Invalid role: {config.pd_role}, must be either sender or receiver"
        )

        assert config.pd_buffer_size is not None
        assert config.pd_buffer_device is not None

        if role == "receiver":
            assert config.pd_peer_host is not None
            assert config.pd_peer_init_port is not None
            assert config.pd_peer_alloc_port is not None
        elif role == "sender":
            assert config.pd_proxy_host is not None
            assert config.pd_proxy_port is not None

        corrected_device = get_correct_device(
            config.pd_buffer_device, metadata.worker_id
        )

        if config.pd_peer_alloc_port is not None:
            pd_peer_alloc_port = config.pd_peer_alloc_port[tp_rank]
        else:
            pd_peer_alloc_port = None

        if config.pd_peer_init_port is not None:
            pd_peer_init_port = config.pd_peer_init_port[tp_rank]
        else:
            pd_peer_init_port = None

        return PDConfig(
            role=role,
            peer_host=config.pd_peer_host,
            peer_init_port=pd_peer_init_port,
            peer_alloc_port=pd_peer_alloc_port,
            proxy_host=config.pd_proxy_host,
            proxy_port=config.pd_proxy_port,
            buffer_size=config.pd_buffer_size,
            buffer_device=corrected_device,
        )


class PDBackend(AllocatorBackendInterface):
    """
    Implementation of the StorageBackendInterface for PD Disaggregation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ):
        self.running = True

        self.tp_rank = metadata.worker_id

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )

        # NOTE(Jiayi): sender/prefiller will not use this pool;
        # only receiver/decoder will.
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self.data_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        # TODO(Jiayi): add async zmq context if we want better asynchrony.
        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

        # Initialize transfer channel
        peer_init_url = None
        self.local_id = ""
        # TODO(Jiayi): both sender and receiver have to have
        # peer_init_url if they want to do instance flip.
        if self.pd_config.peer_init_port is not None:
            peer_init_url = (
                f"{self.pd_config.peer_host}:{self.pd_config.peer_init_port}"
            )
            self.local_id = self.pd_config.peer_host + str(
                self.pd_config.peer_init_port
            )

        self.transfer_channel = CreateTransferChannel(
            async_mode=False,
            channel_type=config.transfer_channel,
            role=self.pd_config.role,
            buffer_ptr=self.memory_allocator.gpu_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.gpu_allocator.buffer_size,
            align_bytes=self.memory_allocator.gpu_allocator.align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=peer_init_url,
            backends=config.nixl_backends,
        )

        if self.pd_config.role == "sender":
            self._init_sender()
            self.initialized_peers: set[str] = set()
            self.mem_alloc_sockets: dict[str, zmq.Socket] = {}
        elif self.pd_config.role == "receiver":
            self._init_receiver()
        else:
            raise ValueError("Invalid PD role.")

        self.full_chunk_size = config.chunk_size

    def __str__(self):
        return self.__class__.__name__

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        # First Party
        from lmcache.v1.transfer_channel.transfer_utils import (
            get_correct_device,
        )

        corrected_device = get_correct_device(
            config.pd_buffer_device,
            metadata.worker_id,
        )
        logger.info(f"Setting cuda device to {corrected_device} ")
        torch.cuda.set_device(corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()
        paged_mem_allocator.init_gpu_memory_allocator(
            config.pd_buffer_size,
            torch.Size(metadata.kv_shape),
            metadata.kv_dtype,
            MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
            corrected_device,
        )

        return paged_mem_allocator

    def get_memory_allocator(self) -> PagedCpuGpuMemoryAllocator:
        return self.memory_allocator

    def get_allocator_backend(self):
        return self

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        # NOTE: no eviction and busy_loop in PD
        return self.memory_allocator.allocate(
            shape=shape, dtype=dtype, fmt=fmt, allocator_type="gpu"
        )

    # TODO(Jiayi): Please implement batched allocate to reduce memory
    # allocation overhead.
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ):
        return self.memory_allocator.batched_allocate(
            shape, dtype, batch_size, fmt, allocator_type="gpu"
        )

    # NOTE(Jiayi): If two requests have overlapped keys, will
    # the later one cause any problems here?
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        assert isinstance(key, CacheEngineKey)
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if pin:
                    mem_obj.ref_count_up()
                return True
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    ############################################################
    # Prefiller functions
    ############################################################
    def _init_sender(self):
        proxy_url = f"{self.pd_config.proxy_host}:{self.pd_config.proxy_port}"
        self.proxy_side_channel = get_zmq_socket(
            self.zmq_context,
            proxy_url,
            "tcp",
            zmq.PUSH,
            "connect",
        )

    def _ensure_peer_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_init_port: int,
        receiver_alloc_port: int,
    ) -> None:
        if receiver_id in self.initialized_peers:
            return

        receiver_init_url = f"{receiver_host}:{receiver_init_port}"
        receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"

        # Establish the connection with the receiver/decoder
        self.transfer_channel.lazy_init_peer_connection(
            local_id=self.local_id, peer_id=receiver_id, peer_init_url=receiver_init_url
        )

        # Set up the memory allocation socket
        mem_alloc_socket = get_zmq_socket(
            self.zmq_context,
            receiver_mem_alloc_url,
            "tcp",
            zmq.REQ,
            "connect",
        )
        self.mem_alloc_sockets[receiver_id] = mem_alloc_socket

        self.initialized_peers.add(receiver_id)

    def _remote_allocate(
        self, receiver_id: str, alloc_request: AllocRequest
    ) -> AllocResponse:
        side_channel = self.mem_alloc_sockets[receiver_id]
        side_channel.send(msgspec.msgpack.encode(alloc_request))
        msg = side_channel.recv()
        alloc_response = msgspec.msgpack.decode(msg, type=PDMsg)

        return alloc_response

    def _get_remote_alloc_request(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> AllocRequest:
        """
        Get the allocation request given the keys and memory objects.

        Let's say there are N memory objects in total.
        We have the following assumptions:
        - The first N-1 memory objects are full chunks, each with
        `full_chunk_size` tokens.
        - The last memory object can be a partial chunk, which has
        `last_chunk_toks` tokens.
        """

        fmt = mem_objs[0].meta.fmt
        shape = mem_objs[0].meta.shape
        dtype = TORCH_DTYPE_TO_STR_DTYPE[mem_objs[0].meta.dtype]
        token_dim = fmt.token_dim()
        last_chunk_toks = mem_objs[-1].meta.shape[token_dim]

        str_keys = [key.to_string() for key in keys]

        return AllocRequest(
            keys=str_keys,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
        )

    # TODO(Jiayi): make this async in the future
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

        receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
        receiver_alloc_port = transfer_spec.receiver_alloc_port[self.tp_rank]
        receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
        receiver_host = transfer_spec.receiver_host

        self._ensure_peer_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_init_port=receiver_init_port,
            receiver_alloc_port=receiver_alloc_port,
        )

        # Allocate remote memory objects
        alloc_request = self._get_remote_alloc_request(keys, memory_objs)
        alloc_response = self._remote_allocate(receiver_id, alloc_request)
        already_sent_indexes = alloc_response.already_sent_indexes
        remote_indexes = alloc_response.remote_indexes

        # Filter out already sent memory objects and free them
        mem_objs_to_send = []
        for idx, mem_obj in enumerate(memory_objs):
            if idx in already_sent_indexes:
                mem_obj.ref_count_down()
            else:
                mem_objs_to_send.append(mem_obj)

        if mem_objs_to_send:
            # TODO(Jiayi): make this decoupled with transfer channel
            # Construct transfer spec
            channel_transfer_spec = {
                "receiver_id": receiver_id,
                "remote_indexes": remote_indexes,
            }

            # TODO(Jiayi): Consider making this real async
            # Perform the actual transfer
            self.transfer_channel.batched_write(
                objects=mem_objs_to_send,
                transfer_spec=channel_transfer_spec,
            )

            # TODO(Jiayi): consider moving this to the transfer channel
            # since we might want the transfer to be async.
            for mem_obj in mem_objs_to_send:
                mem_obj.ref_count_down()
        else:
            logger.debug(
                "All memory objects have been already sent to the remote peer."
                " Skipping transfer."
            )

        if transfer_spec.is_last_prefill:
            # Notify the proxy that the transfer is done
            notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
            notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
            self.proxy_side_channel.send(notif_msg_bytes)

    ############################################################
    # Prefiller functions end
    ############################################################

    ############################################################
    # Decoder functions
    ############################################################
    def _init_receiver(self):
        # Initialize initialization side channels
        receiver_alloc_url = (
            f"{self.pd_config.peer_host}:{self.pd_config.peer_alloc_port}"
        )
        self.alloc_side_channel = get_zmq_socket(
            self.zmq_context, receiver_alloc_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self.alloc_side_channel)

        # Start the memory allocation thread
        self.mem_alloc_thread = threading.Thread(
            target=self._mem_alloc_loop, daemon=True
        )
        self.mem_alloc_thread.start()
        self.running_threads.append(self.mem_alloc_thread)

    def _allocate_and_put(self, alloc_request: AllocRequest) -> AllocResponse:
        total_allocs = len(alloc_request.keys)
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = alloc_request.shape

        alloc_indexes = []
        already_send_indexes = []

        for idx, key_str in enumerate(alloc_request.keys):
            key = CacheEngineKey.from_string(key_str)
            if self.contains(key, pin=True):
                already_send_indexes.append(idx)
                continue

            if idx == total_allocs - 1:
                num_alloc_tokens = alloc_request.last_chunk_toks
                token_dim = fmt.token_dim()
                shape[token_dim] = num_alloc_tokens
            else:
                num_alloc_tokens = self.full_chunk_size

            mem_obj = self.allocate(torch.Size(shape), dtype, fmt)

            # TODO(Jiayi): make busy loop allocation part of
            # memory allocator instead of backend as both PD
            # and CPU offloading might need this.
            wait_time = 0.01
            while mem_obj is None:
                logger.warning(
                    "Failed to allocate memory object, retrying...",
                )
                time.sleep(wait_time)
                mem_obj = self.allocate(torch.Size(shape), dtype, fmt)

            alloc_indexes.append(mem_obj.meta.address)

            self.put(key, mem_obj)

        return AllocResponse(
            already_sent_indexes=already_send_indexes, remote_indexes=alloc_indexes
        )

    def _mem_alloc_loop(self):
        """
        Running the memory allocation loop.
        """
        while self.running:
            try:
                # receive alloc request
                alloc_req_bytes = self.alloc_side_channel.recv()
                alloc_req = msgspec.msgpack.decode(alloc_req_bytes, type=PDMsg)
                assert isinstance(alloc_req, AllocRequest), (
                    "The request from the remote peer is not a AllocRequest"
                )

                # NOTE: it's okay to put the memory objs into the storage backend
                # first because decode vllm will not be able to see the decode
                # request until proxy receives the ack.
                alloc_resp = self._allocate_and_put(alloc_req)

                # send back response
                self.alloc_side_channel.send(msgspec.msgpack.encode(alloc_resp))

            except Exception as e:
                logger.error("Failed to process mem alloc loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ):
        with self.data_lock:
            self.data[key] = mem_obj

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.data_lock:
            # NOTE(Jiayi): we assume that the key must be in local data
            # because we are using a push-based transfer
            mem_obj = self.data.get(key, None)
            assert mem_obj is not None, f"Key {key} not found in local data."
            return mem_obj

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """
        # TODO(Jiayi): The logic here is confusing. Ref count down
        # will be done after this function call in cache engine.
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if mem_obj.get_ref_count() == 1:
                    del self.data[key]
                return True
            return False

    ############################################################
    # Decoder functions end
    ############################################################

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.transfer_channel.close()
        self.zmq_context.term()

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True
