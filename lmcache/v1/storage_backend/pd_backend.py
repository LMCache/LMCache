# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import math
import threading
import time

# Third Party
from mooncake.store import (
    MooncakeDistributedStore,
    ReplicateConfig,
    bind_to_numa_node,
)
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
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
)
from lmcache.v1.system_detection import NUMADetector

logger = init_logger(__name__)


class PDMsgBase(msgspec.Struct, tag=True):
    """Base class for all PD-related messages."""

    pass


class AllocRequest(PDMsgBase):
    """Allocation request message."""

    keys: list[str]
    fmt: int
    shape: list[int]
    dtype: str
    last_chunk_toks: int


class AllocResponse(PDMsgBase):
    """Allocation response message."""

    already_sent_indexes: list[int]
    remote_indexes: list[int]


class ProxyNotif(PDMsgBase):
    req_id: str


PDMsg = Union[AllocRequest, AllocResponse, ProxyNotif]


@dataclass
class PDConfig:
    role: str
    peer_host: Optional[str]
    peer_init_port: Optional[int]
    peer_alloc_port: Optional[int]
    proxy_host: Optional[str]
    proxy_port: Optional[int]
    buffer_size: int
    buffer_device: str

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        tp_rank: int,
    ) -> "PDConfig":
        role = config.pd_role
        # NOTE: pd_peer_* arrays are sharded by tp_rank when provided.
        pd_peer_alloc_port = None
        if config.pd_peer_alloc_port is not None:
            pd_peer_alloc_port = config.pd_peer_alloc_port[tp_rank]

        pd_peer_init_port = None
        if config.pd_peer_init_port is not None:
            pd_peer_init_port = config.pd_peer_init_port[tp_rank]

        return PDConfig(
            role=role,
            peer_host=config.pd_peer_host,
            peer_init_port=pd_peer_init_port,
            peer_alloc_port=pd_peer_alloc_port,
            proxy_host=config.pd_proxy_host,
            proxy_port=config.pd_proxy_port,
            buffer_size=config.pd_buffer_size,
            buffer_device=config.pd_buffer_device,
        )


@dataclass
class PendingChunk:
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    mem_obj: Optional[MemoryObj] = None
    ready: bool = False

    @property
    def num_bytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


class PDBackend(AllocatorBackendInterface):
    """PD backend that stages KV chunks via Mooncake distributed store."""

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
        self.data: Dict[CacheEngineKey, MemoryObj] = {}
        self.pending_chunks: Dict[CacheEngineKey, PendingChunk] = {}
        self.remote_keys: Dict[CacheEngineKey, str] = {}
        self.data_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.mooncake_config = MooncakeStoreConfig.load_from_lmcache_config(config)
        self.mooncake_store = MooncakeDistributedStore()
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = 1
        self.registered_gpu_ptr: Optional[int] = None

        self._setup_mooncake(metadata, config)

        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

        if self.pd_config.role == "sender":
            self._init_sender()
            self.initialized_peers: set[str] = set()
            self.mem_alloc_sockets: Dict[str, zmq.Socket] = {}
        elif self.pd_config.role == "receiver":
            self._init_receiver()
        else:
            raise ValueError("Invalid PD role.")

        self.full_chunk_size = config.chunk_size
        self.meta_shape = torch.Size(metadata.kv_shape)
        self.meta_dtype = metadata.kv_dtype
        self.meta_fmt = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )

    def __str__(self):
        return self.__class__.__name__

    def _to_local_key(self, remote_key: CacheEngineKey) -> CacheEngineKey:
        if remote_key.worker_id == self.tp_rank:
            return remote_key
        return CacheEngineKey(
            remote_key.fmt,
            remote_key.model_name,
            remote_key.world_size,
            self.tp_rank,
            remote_key.chunk_hash,
            remote_key.request_configs,
        )

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        # First Party
        from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

        corrected_device = get_correct_device(
            config.pd_buffer_device,
            metadata.worker_id,
        )
        logger.info(f"Setting cuda device to {corrected_device}")
        torch.cuda.set_device(corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()
        paged_mem_allocator.init_gpu_memory_allocator(
            config.pd_buffer_size,
            torch.Size(metadata.kv_shape),
            metadata.kv_dtype,
            MemoryFormat.KV_2LTD,
            corrected_device,
        )

        return paged_mem_allocator

    def _setup_mooncake(
        self,
        metadata: LMCacheEngineMetadata,
        config: LMCacheEngineConfig,
    ) -> None:
        if self.mooncake_config.prefer_local_alloc:
            try:
                numa_mapping = NUMADetector.get_numa_mapping(config)
                if numa_mapping:
                    current_device_id = torch.cuda.current_device()
                    gpu_to_numa = getattr(numa_mapping, "gpu_to_numa_mapping", {})
                    numa_id = gpu_to_numa.get(current_device_id)
                    if numa_id is not None:
                        bind_to_numa_node(numa_id)
                        logger.info(
                            "Mooncake bind_to_numa_node success for GPU %s -> NUMA %s",
                            current_device_id,
                            numa_id,
                        )
            except Exception as exc:
                logger.warning("Failed to bind Mooncake store to NUMA node: %s", exc)

        if (
            self.mooncake_config.storage_root_dir is not None
            and self.mooncake_config.storage_root_dir != ""
        ):
            # Standard
            import os

            os.environ["MOONCAKE_STORAGE_ROOT_DIR"] = (
                self.mooncake_config.storage_root_dir
            )

        setup_ret = self.mooncake_store.setup(
            self.mooncake_config.local_hostname,
            self.mooncake_config.metadata_server,
            self.mooncake_config.global_segment_size,
            self.mooncake_config.local_buffer_size,
            self.mooncake_config.protocol,
            self.mooncake_config.device_name,
            self.mooncake_config.master_server_address,
        )
        if setup_ret != 0:
            raise RuntimeError(
                "Mooncake store setup failed with error code %s" % setup_ret
            )
        logger.info(
            "Mooncake store setup succeeded with config %s", self.mooncake_config
        )

        if self.mooncake_config.prefer_local_alloc:
            self.replica_config.preferred_segment = self.mooncake_store.get_hostname()

        gpu_allocator = self.memory_allocator.gpu_allocator
        ptr = gpu_allocator.buffer_ptr
        size = gpu_allocator.buffer_size
        result = self.mooncake_store.register_buffer(ptr, size)
        if result != 0:
            raise RuntimeError(
                "Mooncake failed to register GPU buffer ptr=%s size=%s err=%s"
                % (hex(ptr), size, result)
            )
        self.registered_gpu_ptr = ptr
        logger.info("Mooncake registered GPU buffer: ptr=%s size=%s", hex(ptr), size)

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
        return self.memory_allocator.allocate(
            shape=shape, dtype=dtype, fmt=fmt, allocator_type="gpu"
        )

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

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self.pd_config.role == "sender":
            return self.mooncake_store.is_exist(key.to_string())
        elif self.pd_config.role == "receiver":
            with self.data_lock:
                mem_obj = self.data.get(key)
                if mem_obj is None:
                    return False
                if pin:
                    # Pin instead of increasing ref count.
                    # Unpinned state continues to control eviction.
                    mem_obj.pin()
                return True
        raise ValueError("Invalid PD role.")

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

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

        receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"
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
        socket = self.mem_alloc_sockets[receiver_id]
        socket.send(msgspec.msgpack.encode(alloc_request))
        msg = socket.recv()
        alloc_response = msgspec.msgpack.decode(msg, type=PDMsg)
        assert isinstance(alloc_response, AllocResponse)
        return alloc_response

    def _get_remote_alloc_request(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> AllocRequest:
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

    def _batch_put_to_mooncake(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        if not keys:
            return

        key_strs = [k.to_string() for k in keys]
        buffer_ptrs = [obj.data_ptr for obj in memory_objs]
        buffer_sizes = [obj.get_size() for obj in memory_objs]

        put_results = self.mooncake_store.batch_put_from(
            key_strs,
            buffer_ptrs,
            buffer_sizes,
            self.replica_config,
        )

        for idx, ret in enumerate(put_results):
            if ret != 0:
                raise RuntimeError(
                    "Mooncake batch_put_from failed for %s with code %s"
                    % (key_strs[idx], ret)
                )
        for obj in memory_objs:
            obj.ref_count_down()

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
        receiver_host = transfer_spec.receiver_host
        receiver_id = receiver_host + str(receiver_init_port)

        self._ensure_peer_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_init_port=receiver_init_port,
            receiver_alloc_port=receiver_alloc_port,
        )

        alloc_request = self._get_remote_alloc_request(keys, memory_objs)
        alloc_response = self._remote_allocate(receiver_id, alloc_request)
        already_sent_indexes = alloc_response.already_sent_indexes

        upload_keys: list[CacheEngineKey] = []
        upload_objs: list[MemoryObj] = []
        for idx, (key, mem_obj) in enumerate(zip(keys, memory_objs, strict=False)):
            if idx in already_sent_indexes:
                mem_obj.ref_count_down()
                continue
            upload_keys.append(key)
            upload_objs.append(mem_obj)

        if upload_keys:
            self._batch_put_to_mooncake(upload_keys, upload_objs)

        if transfer_spec.is_last_prefill:
            notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
            notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
            self.proxy_side_channel.send(notif_msg_bytes)

    ############################################################
    # Receiver helpers
    ############################################################
    def _init_receiver(self):
        assert self.pd_config.peer_host is not None
        assert self.pd_config.peer_alloc_port is not None
        receiver_alloc_url = (
            f"{self.pd_config.peer_host}:{self.pd_config.peer_alloc_port}"
        )
        self.alloc_side_channel = get_zmq_socket(
            self.zmq_context, receiver_alloc_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self.alloc_side_channel)

        self.mem_alloc_thread = threading.Thread(
            target=self._mem_alloc_loop, daemon=True
        )
        self.mem_alloc_thread.start()
        self.running_threads.append(self.mem_alloc_thread)

    def _mooncake_exists(self, key_strs: Sequence[str]) -> list[bool]:
        if isinstance(key_strs, str):  # guard against accidental str iteration
            key_list = [key_strs]
        else:
            key_list = list(key_strs)

        if not key_list:
            return []

        rets = self.mooncake_store.batch_is_exist(key_list)
        return [ret == 1 for ret in rets]

    def _record_pending_chunk(
        self,
        key: CacheEngineKey,
        alloc_request: AllocRequest,
        idx: int,
    ) -> PendingChunk:
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        base_shape = list(alloc_request.shape)
        token_dim = fmt.token_dim()
        if idx == len(alloc_request.keys) - 1:
            base_shape[token_dim] = alloc_request.last_chunk_toks
        else:
            base_shape[token_dim] = self.full_chunk_size
        chunk_shape = torch.Size(base_shape)
        pending = PendingChunk(chunk_shape, dtype, fmt)
        self.pending_chunks[key] = pending
        return pending

    def _allocate_and_put(self, alloc_request: AllocRequest) -> AllocResponse:
        already_sent: list[int] = []

        remote_exists_flags = self._mooncake_exists(alloc_request.keys)
        if len(remote_exists_flags) != len(alloc_request.keys):
            raise RuntimeError(
                "Mooncake batch_is_exist returned mismatched result length"
            )

        for idx, key_str in enumerate(alloc_request.keys):
            exists_remote = remote_exists_flags[idx]
            remote_key = CacheEngineKey.from_string(key_str)
            local_key = self._to_local_key(remote_key)

            with self.data_lock:
                if local_key in self.data:
                    already_sent.append(idx)
                    continue
                pending = self.pending_chunks.get(local_key)
            if pending is None:
                pending = self._record_pending_chunk(local_key, alloc_request, idx)
                with self.data_lock:
                    self.remote_keys[local_key] = key_str

            if exists_remote:
                already_sent.append(idx)

            if pending.mem_obj is None:
                mem_obj = self.allocate(
                    torch.Size(pending.shape), pending.dtype, pending.fmt
                )

                wait_time = 0.01
                while mem_obj is None:
                    logger.warning(
                        "Failed to allocate memory object, retrying...",
                    )
                    time.sleep(wait_time)
                    mem_obj = self.allocate(
                        torch.Size(pending.shape), pending.dtype, pending.fmt
                    )

                pending.mem_obj = mem_obj
                pending.ready = False

                with self.data_lock:
                    self.data[local_key] = mem_obj

        return AllocResponse(
            already_sent_indexes=already_sent,
            remote_indexes=[],
        )

    def _mem_alloc_loop(self):
        while self.running:
            try:
                alloc_req_bytes = self.alloc_side_channel.recv()
                alloc_req = msgspec.msgpack.decode(alloc_req_bytes, type=PDMsg)
                assert isinstance(alloc_req, AllocRequest)
                logger.info(
                    "PDBackend receiver got alloc request for keys %s", alloc_req.keys
                )
                alloc_resp = self._allocate_and_put(alloc_req)
                self.alloc_side_channel.send(msgspec.msgpack.encode(alloc_resp))
            except Exception as exc:
                logger.error("Failed to process mem alloc loop: %s", exc)
                if self.running:
                    time.sleep(0.01)

    def _fetch_from_mooncake(
        self, key_str: str, mem_obj: MemoryObj, pending: PendingChunk
    ) -> None:
        ptrs = [mem_obj.data_ptr]
        sizes = [pending.num_bytes]

        bytes_read = self.mooncake_store.batch_get_into([key_str], ptrs, sizes)

        if len(bytes_read) != len(ptrs):
            raise RuntimeError(
                "Mooncake batch_get_into returned %s entries for %s keys"
                % (len(bytes_read), len(ptrs))
            )

        for idx, num_bytes in enumerate(bytes_read):
            expected = sizes[idx]
            if num_bytes <= 0 or num_bytes != expected:
                raise RuntimeError(
                    (
                        f"Mooncake get_into failed for key {key_str} at index {idx}, "
                        f"expected {expected} but got {num_bytes}"
                    )
                )
            logger.info(
                "Mooncake batch_get_into fetched %s bytes for %s",
                num_bytes,
                key_str,
            )

    def reshape_partial_chunk(
        self,
        memory_obj: MemoryObj,
        bytes_read: int,
    ) -> MemoryObj:
        """Trim `memory_obj` to match the actual bytes read from remote storage."""

        dtype = memory_obj.meta.dtype
        if dtype is None:
            raise ValueError(
                "memory_obj meta dtype is required to reshape partial chunk"
            )

        fmt = memory_obj.meta.fmt
        shape_list = list(memory_obj.meta.shape)
        token_dim = fmt.token_dim()

        elements_per_token = 1
        for dim_idx, dim_size in enumerate(shape_list):
            if dim_idx == token_dim:
                continue
            elements_per_token *= dim_size

        dtype_size = torch.tensor([], dtype=dtype).element_size()
        single_token_size = elements_per_token * dtype_size
        full_chunk_size = single_token_size * shape_list[token_dim]

        if bytes_read % single_token_size != 0 or bytes_read > full_chunk_size:
            raise ValueError(
                f"bytes_read: {bytes_read} is illegal, "
                f"single_token_size: {single_token_size}, "
                f"full_chunk_size: {full_chunk_size}"
            )

        if bytes_read == full_chunk_size:
            return memory_obj

        actual_tokens = bytes_read // single_token_size
        shape_list[token_dim] = actual_tokens
        memory_obj.raw_data = memory_obj.raw_data[:bytes_read]
        memory_obj.meta.shape = torch.Size(shape_list)

        return memory_obj

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ):
        raise NotImplementedError("PDBackend put is not implemented")

    def _ensure_chunk_loaded(
        self,
        key: CacheEngineKey,
        *,
        add_ref: bool,
    ) -> MemoryObj:
        """Make sure the chunk backing ``key`` is allocated and hydrated."""
        remote_key_str = self.remote_keys.get(key)
        if remote_key_str is None:
            remote_key_str = key.to_string()
            logger.error(
                "PDBackend fallback to local key string for %s during fetch",
                remote_key_str,
            )

        with self.data_lock:
            mem_obj = self.data.get(key)
            pending = self.pending_chunks.get(key)

        if pending is None and mem_obj is None:
            raise KeyError(f"Key {key} not found in PD backend pending set")

        if pending is not None:
            if pending.mem_obj is None:
                logger.info("PDBackend allocating GPU buffer for %s", key.to_string())
                mem_obj = self.allocate(pending.shape, pending.dtype, pending.fmt)
                wait_time = 0.01
                while mem_obj is None:
                    logger.warning(
                        "Failed to allocate GPU memory for %s, retrying...", key
                    )
                    time.sleep(wait_time)
                    mem_obj = self.allocate(pending.shape, pending.dtype, pending.fmt)
                pending.mem_obj = mem_obj
                pending.ready = False
                with self.data_lock:
                    self.data[key] = mem_obj
            else:
                mem_obj = pending.mem_obj

            if not pending.ready:
                logger.info(
                    "PDBackend fetching chunk from Mooncake for %s",
                    key.to_string(),
                )
                self._fetch_from_mooncake(remote_key_str, pending.mem_obj, pending)
                pending.ready = True

        if mem_obj is None:
            with self.data_lock:
                mem_obj = self.data.get(key)
            if mem_obj is None:
                raise KeyError(f"Key {key} not found in PD backend data store")

        if add_ref:
            mem_obj.ref_count_up()

        return mem_obj

    def get_blocking_for_sender(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        mem_obj = self.allocate(self.meta_shape, self.meta_dtype, self.meta_fmt)
        if mem_obj is None:
            logger.error(
                "PDBackend sender failed to allocate buffer for key %s",
                key.to_string(),
            )
            return None

        bytes_read = self.mooncake_store.batch_get_into(
            [key.to_string()], [mem_obj.data_ptr], [mem_obj.get_size()]
        )
        if not bytes_read:
            logger.error(
                "Mooncake batch_get_into returned no data for key %s",
                key.to_string(),
            )
            return None

        self.reshape_partial_chunk(mem_obj, bytes_read[0])
        return mem_obj

    def batched_get_blocking_for_sender(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        mem_objs: list[Optional[MemoryObj]] = []
        valid_indices: list[int] = []
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []

        for idx, key in enumerate(keys):
            mem_obj = self.allocate(self.meta_shape, self.meta_dtype, self.meta_fmt)
            mem_objs.append(mem_obj)
            if mem_obj is None:
                logger.error(
                    "PDBackend sender failed to allocate buffer for key %s",
                    key.to_string(),
                )
                continue

            valid_indices.append(idx)
            buffer_ptrs.append(mem_obj.data_ptr)
            buffer_sizes.append(mem_obj.get_size())

        if valid_indices:
            key_strs = [keys[idx].to_string() for idx in valid_indices]
            bytes_read = self.mooncake_store.batch_get_into(
                key_strs, buffer_ptrs, buffer_sizes
            )
            if len(bytes_read) != len(valid_indices):
                raise RuntimeError(
                    "Mooncake batch_get_into returned %s entries for %s keys"
                    % (len(bytes_read), len(valid_indices))
                )

            for offset, num_bytes in enumerate(bytes_read):
                idx = valid_indices[offset]
                mem_obj = mem_objs[idx]
                if mem_obj is None:
                    logger.error(
                        "Unexpected None MemoryObj during reshape for key %s",
                        keys[idx].to_string(),
                    )
                    continue
                self.reshape_partial_chunk(mem_obj, num_bytes)

        return mem_objs

    def batched_get_blocking(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        if self.pd_config.role == "sender":
            return self.batched_get_blocking_for_sender(keys)
        elif self.pd_config.role == "receiver":
            mem_objs: list[Optional[MemoryObj]] = []
            for key in keys:
                mem_obj = self._ensure_chunk_loaded(key, add_ref=True)
                mem_objs.append(mem_obj)
            return mem_objs
        raise ValueError("Invalid PD role.")

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if self.pd_config.role == "sender":
            return self.get_blocking_for_sender(key)
        elif self.pd_config.role == "receiver":
            return self._ensure_chunk_loaded(key, add_ref=True)
        raise ValueError("Invalid PD role.")

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        raise NotImplementedError(
            "PDBackend batched_get_non_blocking is not implemented"
        )

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        removed = False
        with self.data_lock:
            popped_mem_obj = None
            if key in self.data:
                popped_mem_obj = self.data.pop(key)
                popped_mem_obj.ref_count_down()
                removed = True
                self.remote_keys.pop(key, None)
            if key in self.pending_chunks:
                pending = self.pending_chunks.pop(key)
                if (
                    pending.mem_obj is not None
                    and pending.mem_obj is not popped_mem_obj
                ):
                    pending.mem_obj.ref_count_down()
                removed = True
        return removed

    def close(self) -> None:
        self.running = False
        for thread in self.running_threads:
            thread.join()

        if self.registered_gpu_ptr is not None:
            result = self.mooncake_store.unregister_buffer(self.registered_gpu_ptr)
            if result != 0:
                logger.warning(
                    "Mooncake GPU buffer unregister failed: ptr=%s err=%s",
                    hex(self.registered_gpu_ptr),
                    result,
                )

        try:
            self.mooncake_store.close()
        except Exception as exc:
            logger.warning("Failed to close Mooncake store cleanly: %s", exc)

        self.zmq_context.term()

    def pin(self, key: CacheEngineKey) -> bool:
        with self.data_lock:
            mem_obj = self.data.get(key)
            if mem_obj is None:
                pending = self.pending_chunks.get(key)
                if pending is not None and pending.mem_obj is not None:
                    mem_obj = pending.mem_obj
            if mem_obj is None:
                return False
            mem_obj.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.data_lock:
            mem_obj = self.data.get(key)
            if mem_obj is None:
                pending = self.pending_chunks.get(key)
                if pending is not None and pending.mem_obj is not None:
                    mem_obj = pending.mem_obj
            if mem_obj is None:
                return False
            mem_obj.unpin()
            return True
