# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import concurrent.futures as cf
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
from lmcache.v1.transfer_channel import CreateTransferChannel, NixlChannel
from lmcache.v1.transfer_channel.nixl_channel import ShardingSpec, TPRankRecvInfo
from lmcache.v1.transfer_channel.transfer_utils import (
    TransferRole,
    get_correct_device,
)

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
    transpose: bool


class AllocResponse(PDMsgBase):
    """Allocation response message"""

    # Indexes (local) of already sent memory objects
    already_sent_indexes: list[int]

    # Indexes (remote) of allocated memory objects (to be written)
    remote_indexes: list[int]


class ProxyNotif(PDMsgBase):
    req_id: str  # The request UUID to notify the proxy


PDMsg = Union[AllocRequest, AllocResponse, ProxyNotif]


class PDReceiverInfo:
    def __init__(
        self,
        receiver_ids: list[str],
        receiver_host: str,
        receiver_init_ports: list[int],
        receiver_alloc_ports: list[int],
    ):
        self.receiver_ids = receiver_ids
        self.receiver_host = receiver_host
        self.receiver_init_ports = receiver_init_ports
        self.receiver_alloc_ports = receiver_alloc_ports

    def get_group_receivers(self) -> list[TPRankRecvInfo]:
        assert len(self.receiver_ids) == len(self.receiver_init_ports)
        assert len(self.receiver_ids) == len(self.receiver_alloc_ports)
        return [
            TPRankRecvInfo(
                group_tp_rank=tp_rank,
                receiver_id=self.receiver_ids[tp_rank],
                receiver_init_url=f"{self.receiver_host}:{self.receiver_init_ports[tp_rank]}",
                receiver_mem_alloc_url=f"{self.receiver_host}:{self.receiver_alloc_ports[tp_rank]}",
            )
            for tp_rank in range(len(self.receiver_ids))
        ]


@dataclass
class PDConfig:
    role: TransferRole

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

        role = TransferRole(config.pd_role)

        # TODO(Jiayi): Could be both if we want to do dynamic role switch.
        assert role in [TransferRole.SENDER, TransferRole.RECEIVER], (
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
        loop: asyncio.AbstractEventLoop,
    ):
        self.running = True

        self.tp_rank = metadata.worker_id
        # TODO(novahow): verify if this is needed, will metadata.world_size suffice?
        # Third Party
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )

        self.tp_world_size = get_tensor_model_parallel_world_size()

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )
        self.loop = loop

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

        self.transfer_channel: NixlChannel = CreateTransferChannel(
            async_mode=False,
            channel_type=config.transfer_channel,
            role=self.pd_config.role,
            allocator_meta=self.memory_allocator.gpu_allocator.metadata,
            tp_rank=self.tp_rank,
            tp_size=self.tp_world_size,
            peer_init_url=peer_init_url,
            backends=config.nixl_backends,
        )

        if self.pd_config.role == TransferRole.SENDER:
            self._init_sender()
            self.initialized_peers: set[str] = set()
            self.mem_alloc_sockets: dict[str, zmq.Socket] = {}
        elif self.pd_config.role == TransferRole.RECEIVER:
            self._init_receiver()
        else:
            raise ValueError(f"Invalid PD role: {self.pd_config.role}")

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
            [torch.Size(metadata.kv_shape)],
            [metadata.kv_dtype],
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
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        # NOTE: no eviction and busy_loop in PD
        return self.memory_allocator.allocate(
            shapes, dtypes, fmt=fmt, allocator_type="gpu"
        )

    # TODO(Jiayi): Please implement batched allocate to reduce memory
    # allocation overhead.
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ):
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        return self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type="gpu"
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
        receiver_info: PDReceiverInfo,
    ) -> None:
        for receiver in receiver_info.get_group_receivers():
            receiver_id = receiver.receiver_id
            receiver_init_url = receiver.receiver_init_url
            receiver_mem_alloc_url = receiver.receiver_mem_alloc_url
            if receiver_id in self.initialized_peers:
                continue

            # Establish the connection with the receiver/decoder
            self.transfer_channel.lazy_init_peer_connection(
                local_id=self.local_id,
                peer_id=receiver_id,
                peer_init_url=receiver_init_url,
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
        self,
        keys: Sequence[CacheEngineKey],
        mem_objs: List[MemoryObj],
        dp_ratio: int,
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
        shape = list(shape)
        assert (
            fmt == MemoryFormat.KV_DT2L or dp_ratio == 1
        ), """Asymmetric tensor parallelism is only supported
            for KV_DT2L format for now."""
        shape[fmt.hidden_dim()] //= dp_ratio
        return AllocRequest(
            keys=str_keys,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
            transpose=dp_ratio > 1,
        )

    # TODO(Jiayi): make this async in the future
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to transfer KV caches to peer.

        :param on_complete_callback: Optional callback invoked once per key
            after the transfer completes. Callback exceptions are caught and logged.
        """
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

        decoder_tp_size = len(transfer_spec.receiver_init_ports)
        assert (
            decoder_tp_size % self.tp_world_size == 0
        ), f"""Decoder TP size {decoder_tp_size} must be divisible 
            by sender TP size {self.tp_world_size}."""

        # TODO(novahow): is there better way to obtain receiver tp_size?
        # NOTE(novahow), dp_ratio implies how many decoder tp ranks
        # are mapped to one prefiller tp rank. Having larger tp size on decoder
        # side is beneficial for memory-bound workloads like decoding.
        # For example, assume dp_ratio = 2
        # rank 0 on P maps to rank [0,1] on D
        dp_ratio = decoder_tp_size // self.tp_world_size

        receiver_init_ports = transfer_spec.receiver_init_ports[
            self.tp_rank * dp_ratio : (self.tp_rank + 1) * dp_ratio
        ]
        receiver_alloc_ports = transfer_spec.receiver_alloc_ports[
            self.tp_rank * dp_ratio : (self.tp_rank + 1) * dp_ratio
        ]
        receiver_ids = transfer_spec.receiver_ids[
            self.tp_rank * dp_ratio : (self.tp_rank + 1) * dp_ratio
        ]
        receiver_info = PDReceiverInfo(
            receiver_ids=receiver_ids,
            receiver_host=transfer_spec.receiver_host,
            receiver_init_ports=receiver_init_ports,
            receiver_alloc_ports=receiver_alloc_ports,
        )

        self._ensure_peer_connection(
            receiver_info,
        )

        # Allocate remote memory objects
        alloc_request = self._get_remote_alloc_request(keys, memory_objs, dp_ratio)
        all_sent_indexes: list[set[int]] = [set() for _ in range(dp_ratio)]
        any_remote_indexes: set[int] = set()
        alloc_responses: list[AllocResponse] = []

        send_tasks: list[cf.Future] = []
        for receiver in receiver_info.get_group_receivers():
            # TODO(novahow): make socket async so that we can asyncio.gather responses
            alloc_response = self._remote_allocate(receiver.receiver_id, alloc_request)
            alloc_responses.append(alloc_response)

        for receiver in receiver_info.get_group_receivers():
            alloc_response = alloc_responses[receiver.group_tp_rank]
            mem_objs_to_send = []
            already_sent_indexes = set(alloc_response.already_sent_indexes)
            for idx, mem_obj in enumerate(memory_objs):
                if idx not in already_sent_indexes:
                    mem_objs_to_send.append(mem_obj)

            all_sent_indexes[receiver.group_tp_rank] = already_sent_indexes
            if mem_objs_to_send:
                # TODO(Jiayi): make this decoupled with transfer channel
                # Construct transfer spec

                sharding_spec = ShardingSpec(
                    shard_index=receiver.group_tp_rank,
                    num_shards=dp_ratio,
                )
                channel_transfer_spec = {
                    "receiver_id": receiver.receiver_id,
                    "remote_indexes": alloc_response.remote_indexes,
                    "sharding_spec": sharding_spec,
                }

                send_task = asyncio.run_coroutine_threadsafe(
                    self.transfer_channel.async_batched_write(
                        memory_objs,
                        channel_transfer_spec,
                    ),
                    self.loop,
                )
                send_tasks.append(send_task)
            else:
                logger.debug(
                    f"All memory objects have been already sent"
                    f" to the remote peer {receiver}."
                    " Skipping transfer."
                )

            for idx in range(len(memory_objs)):
                if idx not in already_sent_indexes:
                    any_remote_indexes.add(idx)

        # take intersection of all_sent_indexes
        already_sent = set.intersection(*all_sent_indexes)
        for idx, mem_obj in enumerate(memory_objs):
            if idx in already_sent:
                mem_obj.ref_count_down()

        # Wait for all send tasks to complete
        cf.wait(send_tasks)

        # TODO(Jiayi): consider moving this to the transfer channel
        # since we might want the transfer to be async.
        for idx in any_remote_indexes:
            memory_objs[idx].ref_count_down()

        if transfer_spec.is_last_prefill:
            # Notify the proxy that the transfer is done
            notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
            notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
            self.proxy_side_channel.send(notif_msg_bytes)

        # Call completion callback for all keys after transfer completes
        if on_complete_callback is not None:
            for key in keys:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(f"on_complete_callback failed for key {key}: {e}")

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
            # NOTE(novahow): `contains` checks attributes such as
            # world_size and worker_id, but the key is from the prefiller side,
            # which may have different tp_rank/world_size compared to decoder side,
            # therefore to let cache to work properly on
            # decoder in asymmetric TP setting, we need to update the rank info here.
            decoder_key = key.with_new_world_size(
                self.tp_world_size
            ).with_new_worker_id(self.tp_rank)
            # TODO(novahow): investigate why pin causes
            # failed to allocate in L40S in TP=(2,2)
            if self.contains(decoder_key, pin=True):
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

            self.put(decoder_key, mem_obj)

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
