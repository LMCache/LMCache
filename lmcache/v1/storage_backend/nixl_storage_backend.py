# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Union, cast
from urllib.parse import quote as url_quote
import asyncio
import os
import threading
import time
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig
from nixl._api import nixl_prepped_dlist_handle as NixlDlistHandle
from nixl._api import nixl_xfer_handle as NixlXferHandle
from nixl._api import (
    nixlBind,
)
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    PagedTensorMemoryAllocator,
    _allocate_cpu_memory,
    _allocate_gpu_memory,
    _free_cpu_memory,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


@dataclass
class NixlStorageConfig:
    buffer_size: int
    pool_size: int
    buffer_device: str
    backend: str
    backend_params: dict[str, str]
    dynamic_storage: bool
    enable_presence_cache: bool
    enable_async_put: bool
    use_direct_io: bool
    path: str

    @staticmethod
    def validate_nixl_backend(dynamic_storage: bool, backend: str, device: str):
        if dynamic_storage:  # For now only supported for OBJ & cpu
            if backend in ("OBJ",):
                return device == "cpu"
            else:
                return False
        else:
            if backend in ("GDS", "GDS_MT"):
                return device == "cpu" or device == "cuda"
            elif backend in ("POSIX", "HF3FS", "OBJ"):
                return device == "cpu"
            else:
                return False

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ):
        assert config.nixl_buffer_size is not None
        assert config.nixl_buffer_device is not None

        extra_config = config.extra_config
        assert extra_config is not None
        assert extra_config.get("enable_nixl_storage")

        enable_presence_cache = extra_config.get("nixl_presence_cache", False)
        enable_async_put = extra_config.get("nixl_async_put", False)
        backend_params = extra_config.get("nixl_backend_params", {})
        use_direct_io = extra_config.get("use_direct_io", False)
        pool_size = extra_config.get("nixl_pool_size")
        backend = extra_config.get("nixl_backend")
        path = extra_config.get("nixl_path")

        assert pool_size is not None
        assert backend is not None
        assert use_direct_io in [False, True]

        dynamic_storage = pool_size == 0
        if dynamic_storage:
            assert not config.save_unfull_chunk, (
                "save_unfull_chunk should be set to False when using dynamic storage"
            )

        corrected_device = get_correct_device(
            config.nixl_buffer_device, metadata.worker_id
        )

        assert NixlStorageConfig.validate_nixl_backend(
            dynamic_storage, backend, config.nixl_buffer_device
        ), "Invalid NIXL backend & device combination"

        return NixlStorageConfig(
            buffer_size=config.nixl_buffer_size,
            pool_size=pool_size,
            buffer_device=corrected_device,
            backend=backend,
            backend_params=backend_params,
            dynamic_storage=dynamic_storage,
            enable_presence_cache=enable_presence_cache,
            enable_async_put=enable_async_put,
            use_direct_io=use_direct_io,
            path=path,
        )


class NixlDescPool(ABC):
    def __init__(self, size: int):
        self.lock = threading.Lock()
        self.size: int = size
        self.indices: List[int] = []
        self.indices.extend(reversed(range(size)))

    def get_num_available_descs(self) -> int:
        with self.lock:
            return len(self.indices)

    def pop(self) -> int:
        with self.lock:
            assert len(self.indices) > 0
            return self.indices.pop()

    def push(self, index: int):
        with self.lock:
            assert len(self.indices) < self.size
            self.indices.append(index)

    @abstractmethod
    def close(self):
        pass


class NixlFilePool(NixlDescPool):
    def __init__(self, size: int, path: str, use_direct_io: bool):
        super().__init__(size)
        self.fds: List[int] = []

        assert path is not None

        flags = os.O_CREAT | os.O_RDWR
        if use_direct_io:
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            else:
                logger.warning(
                    "use_direct_io is True, but O_DIRECT is not available on "
                    "this system. Falling back to buffered I/O."
                )
        for i in reversed(range(size)):
            filename = f"obj_{i}_{uuid.uuid4().hex[0:4]}.bin"
            tmp_path = os.path.join(path, filename)
            fd = os.open(tmp_path, flags)
            self.fds.append(fd)

    def close(self):
        # TODO: do we need to delete the files?
        with self.lock:
            assert len(self.fds) == self.size
            for fd in self.fds:
                os.close(fd)


class NixlObjectPool(NixlDescPool):
    def __init__(self, size: int):
        super().__init__(size)
        self.keys: List[str] = []

        for i in reversed(range(size)):
            key = f"obj_{i}_{uuid.uuid4().hex[0:4]}"
            self.keys.append(key)

    def close(self):
        pass


class SetPresenceCache:
    """Default presence cache using a thread-safe Python set."""

    def __init__(self) -> None:
        self._keys: set[int] = set()

    def add(self, key: int) -> None:
        self._keys.add(key)

    def discard(self, key: int) -> None:
        self._keys.discard(key)

    def contains(self, key: int) -> bool:
        return key in self._keys


PresenceCache = Union[SetPresenceCache]


@dataclass
class NixlDesc:
    device_id: int
    meta_info: str


@dataclass
class NixlKeyMetadata:
    index: int
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: Optional[MemoryFormat] = None
    pin_count: int = 0

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check if the related key can be evicted.
        """
        return not self.is_pinned


class NixlStorageAgent(ABC):
    agent_name: str
    nixl_agent: NixlAgent
    mem_reg_descs: nixlBind.nixlRegDList
    mem_xfer_handler: NixlDlistHandle

    def __init__(
        self,
        allocator: PagedTensorMemoryAllocator,
        device: str,
        backend: str,
        backend_params: dict[str, str],
    ):
        buffer_ptr = allocator.buffer_ptr
        buffer_size = allocator.buffer_size
        page_size = allocator.align_bytes

        self.backend = backend
        self.agent_name = "NixlAgent_" + str(uuid.uuid4())
        nixl_conf = NixlAgentConfig(backends=[])
        self.nixl_agent = NixlAgent(self.agent_name, nixl_conf)
        self.nixl_agent.create_backend(backend, backend_params)

        device_id = torch.cuda.current_device()
        self.init_mem_handlers(device, buffer_ptr, buffer_size, page_size, device_id)

    def init_mem_handlers(self, device, buffer_ptr, buffer_size, page_size, device_id):
        reg_list = [(buffer_ptr, buffer_size, device_id, "")]
        xfer_desc = [
            (base_addr, page_size, device_id)
            for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size)
        ]

        if device == "cpu":
            mem_type = "DRAM"
        else:
            mem_type = "VRAM"

        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type=mem_type)
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            "", xfer_descs, mem_type=mem_type
        )

        self.mem_reg_descs = reg_descs
        self.mem_xfer_handler = xfer_handler

    def get_mem_to_storage_handle(
        self, mem_indices, storage_xfer_handler, storage_indices
    ) -> NixlXferHandle:
        return self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.mem_xfer_handler,
            mem_indices,
            storage_xfer_handler,
            storage_indices,
        )

    def get_storage_to_mem_handle(
        self, mem_indices, storage_xfer_handler, storage_indices
    ) -> NixlXferHandle:
        return self.nixl_agent.make_prepped_xfer(
            "READ",
            self.mem_xfer_handler,
            mem_indices,
            storage_xfer_handler,
            storage_indices,
        )

    def post_blocking(self, handle: NixlXferHandle):
        state = self.nixl_agent.transfer(handle)

        while state != "DONE" and state != "ERR":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError:
                raise

        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")

    def release_handle(self, handle):
        self.nixl_agent.release_xfer_handle(handle)

    @abstractmethod
    def close(self):
        pass


class NixlStaticStorageAgent(NixlStorageAgent):
    pool: NixlDescPool
    storage_reg_descs: nixlBind.nixlRegDList
    storage_xfer_descs: nixlBind.nixlXferDList
    storage_xfer_handler: NixlDlistHandle

    def __init__(
        self,
        allocator: PagedTensorMemoryAllocator,
        pool: NixlDescPool,
        device: str,
        backend: str,
        backend_params: dict[str, str],
    ):
        super().__init__(allocator, device, backend, backend_params)

        page_size = allocator.align_bytes

        if isinstance(pool, NixlFilePool):
            self.init_storage_handlers_file(page_size, pool.fds)
        elif isinstance(pool, NixlObjectPool):
            self.init_storage_handlers_object(page_size, pool.keys)
        else:
            raise TypeError(f"Unsupported pool type: {type(pool).__name__}")

    def init_storage_handlers_file(self, page_size, fds):
        reg_list = []
        xfer_desc = []
        for fd in fds:
            reg_list.append((0, page_size, fd, ""))
            xfer_desc.append((0, page_size, fd))
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="FILE")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="FILE")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="FILE"
        )

        self.storage_reg_descs = reg_descs
        self.storage_xfer_descs = xfer_descs
        self.storage_xfer_handler = xfer_handler

    def init_storage_handlers_object(self, page_size, keys):
        reg_list = []
        xfer_desc = []
        for i, key in enumerate(keys):
            reg_list.append((0, page_size, i, key))
            xfer_desc.append((0, page_size, i))
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="OBJ"
        )

        self.storage_reg_descs = reg_descs
        self.storage_xfer_descs = xfer_descs
        self.storage_xfer_handler = xfer_handler

    def get_mem_to_storage_handle(self, mem_indices, storage_indices) -> NixlXferHandle:  # type: ignore[override]
        return super().get_mem_to_storage_handle(
            mem_indices, self.storage_xfer_handler, storage_indices
        )

    def get_storage_to_mem_handle(self, mem_indices, storage_indices) -> NixlXferHandle:  # type: ignore[override]
        return super().get_storage_to_mem_handle(
            mem_indices, self.storage_xfer_handler, storage_indices
        )

    def close(self):
        self.nixl_agent.release_dlist_handle(self.storage_xfer_handler)
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.storage_reg_descs)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)


class NixlDynamicStorageAgent(NixlStorageAgent):
    def __init__(
        self,
        allocator: PagedTensorMemoryAllocator,
        device: str,
        backend: str,
        backend_params: dict[str, str],
    ):
        super().__init__(allocator, device, backend, backend_params)

        if backend == "OBJ":
            self.mem_type = "OBJ"
        else:
            # Already validated in validate_nixl_backend
            raise ValueError(f"unexpected backend: {backend}")

    def create_batched_storage_handler(self, descs: list[NixlDesc], page_size: int):
        reg_list = []
        xfer_desc = []

        for i in range(len(descs)):
            reg_list.append((0, page_size, descs[i].device_id, descs[i].meta_info))
            xfer_desc.append((0, page_size, descs[i].device_id))

        reg_descs = self.nixl_agent.register_memory(reg_list, self.mem_type)
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, self.mem_type)
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type=self.mem_type
        )
        return reg_descs, xfer_handler

    def post_async(self, handle: NixlXferHandle):
        """Non-blocking async post for WRITE operations."""
        state = self.nixl_agent.transfer(handle)
        return state

    def release_storage_handler(self, reg_descs, xfer_handler):
        """Release storage handler resources"""
        self.nixl_agent.release_dlist_handle(xfer_handler)
        self.nixl_agent.deregister_memory(reg_descs)

    def nixl_desc_exists(self, meta_info: str) -> bool:
        reg_list = [(0, 0, 0, meta_info)]
        try:
            resp = self.nixl_agent.query_memory(
                reg_list, self.backend, mem_type=self.mem_type
            )
            # nixl api query_memory returns a list of nixlRegDesc
            if resp[0] is None:
                return False
            return True
        except Exception as exc:
            logger.warning(f"NIXL Desc {meta_info} query failed: {exc}")
            return False

    def close(self):
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)


class NixlStorageBackend(AllocatorBackendInterface, ABC):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the
    implementation.
    """

    def __init__(
        self,
        nixl_config: NixlStorageConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
    ):
        """
        Initialize the Nixl storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=nixl_config.buffer_device)

        self.loop = loop
        self.key_lock = threading.RLock()

        self.progress_lock = threading.RLock()
        self.progress_set: Set[CacheEngineKey] = set()

        self.memory_allocator = self.initialize_allocator(config, metadata)

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> PagedTensorMemoryAllocator:
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )
        assert enable_nixl_storage
        corrected_device = get_correct_device(
            config.nixl_buffer_device,
            metadata.worker_id,
        )

        if corrected_device == "cpu":
            self.buffer = _allocate_cpu_memory(config.nixl_buffer_size)
            self.free_pinned_buffer = True
        else:
            base_buffer, self.buffer = _allocate_gpu_memory(
                config.nixl_buffer_size, corrected_device
            )
            torch.cuda.set_device(corrected_device)
            self.base_buffer = base_buffer  # Prevents early GC of the aligned tensor.
            self.free_pinned_buffer = False

        return PagedTensorMemoryAllocator(
            self.buffer,
            [torch.Size(metadata.kv_shape)],
            [metadata.kv_dtype],
            MemoryFormat.KV_2LTD,
        )

    def get_memory_allocator(self):
        return self.memory_allocator

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if busy_loop:
            logger.warning("NixlStorageBackend does not support busy loop for now")

        return self.memory_allocator.allocate(shapes, dtypes, fmt)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if busy_loop:
            logger.warning("NixlStorageBackend does not support busy loop for now")

        return self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)

    def get_allocator_backend(self):
        return self

    @abstractmethod
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        pass

    @abstractmethod
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        pass

    @abstractmethod
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        pass

    @abstractmethod
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        pass

    @abstractmethod
    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        pass

    @abstractmethod
    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        pass

    @abstractmethod
    def pin(self, key: CacheEngineKey) -> bool:
        pass

    @abstractmethod
    def unpin(self, key: CacheEngineKey) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @staticmethod
    def CreateNixlStorageBackend(
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        metadata: LMCacheEngineMetadata,
    ):
        """
        Create a Nixl backend with the given configuration.

        :param nixl_config: The Nixl configuration.
        :param dst_device: The device where the data is stored.

        :return: A NixlBackend instance.
        """
        # Create the Nixl config
        nixl_config = NixlStorageConfig.from_cache_engine_config(config, metadata)
        # Create the Nixl backend
        if nixl_config.dynamic_storage:
            return NixlDynamicStorageBackend(nixl_config, config, metadata, loop)
        else:
            return NixlStaticStorageBackend(nixl_config, config, metadata, loop)


class NixlStaticStorageBackend(NixlStorageBackend):
    def __init__(
        self,
        nixl_config: NixlStorageConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__(nixl_config, config, metadata, loop)

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.key_dict = self.cache_policy.init_mutable_mapping()

        self.pool = self.createPool(
            nixl_config.backend,
            nixl_config.pool_size,
            nixl_config.path,
            nixl_config.use_direct_io,
        )
        assert self.pool is not None

        self.agent = NixlStaticStorageAgent(
            self.memory_allocator,
            self.pool,
            nixl_config.buffer_device,
            nixl_config.backend,
            nixl_config.backend_params,
        )

    @staticmethod
    def createPool(backend: str, size: int, path: str, use_direct_io: bool):
        if backend in ("GDS", "GDS_MT", "POSIX", "HF3FS"):
            return NixlFilePool(size, path, use_direct_io)
        elif backend in ("OBJ"):
            return NixlObjectPool(size)
        else:
            raise ValueError(f"Unsupported NIXL backend: {backend}")

    def add_key_to_dict(
        self, key: CacheEngineKey, obj: MemoryObjMetadata, index: int
    ) -> None:
        with self.key_lock:
            assert key not in self.key_dict
            self.key_dict[key] = NixlKeyMetadata(
                shape=obj.shape,
                dtype=obj.dtype,
                fmt=obj.fmt,
                index=index,
            )
            self.cache_policy.update_on_put(key)

    async def mem_to_storage(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> None:
        mem_indices = [mem_obj.meta.address for mem_obj in mem_objs]

        storage_indices = []
        for i in range(len(keys)):
            index = self.pool.pop()
            storage_indices.append(index)
            self.add_key_to_dict(keys[i], mem_objs[i].meta, index)

        handle = self.agent.get_mem_to_storage_handle(mem_indices, storage_indices)
        self.agent.post_blocking(handle)
        self.agent.release_handle(handle)

        for key in keys:
            with self.progress_lock:
                self.progress_set.discard(key)

    def _collect_metadata_with_lock(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[NixlKeyMetadata]]:
        """
        Fast metadata collection with lock.
        Returns metadata for each key, None if key doesn't exist.
        """
        metadata_list: list[Optional[NixlKeyMetadata]] = []
        with self.key_lock:
            for key in keys:
                metadata = self.key_dict.get(key)
                if metadata is not None:
                    self.cache_policy.update_on_hit(key, self.key_dict)
                metadata_list.append(metadata)
        return metadata_list

    async def _nixl_transfer_async(
        self, metadata_list: list[Optional[NixlKeyMetadata]]
    ) -> list[Optional[MemoryObj]]:
        """
        Memory allocation and NIXL transfer without locks.
        Can run async and in parallel with other transfers.
        """
        obj_list: list[Optional[MemoryObj]] = []
        mem_indices = []
        storage_indices = []

        # Memory allocation outside lock
        for metadata in metadata_list:
            if metadata is None:
                obj_list.append(None)
                continue

            dtype = metadata.dtype
            shape = metadata.shape
            fmt = metadata.fmt
            assert dtype is not None
            assert shape is not None
            assert fmt is not None

            obj = self.memory_allocator.allocate(shape, dtype, fmt)
            assert obj is not None

            obj_list.append(obj)

            mem_indices.append(obj.meta.address)
            storage_indices.append(metadata.index)

        if not mem_indices:
            return obj_list

        handle = self.agent.get_storage_to_mem_handle(mem_indices, storage_indices)
        self.agent.post_blocking(handle)
        self.agent.release_handle(handle)

        return obj_list

    async def storage_to_mem(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        """
        Combined method: collect metadata with lock, then do NIXL transfer.
        """
        metadata_list = self._collect_metadata_with_lock(keys)
        return await self._nixl_transfer_async(metadata_list)

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        :param key: The key to check
        :param pin: Whether to pin the object in the backend.

        :return: True if the key exists, False otherwise
        """

        with self.key_lock:
            if key in self.key_dict:
                if pin:
                    self.key_dict[key].pin()
                return True
            else:
                return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.

        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        with self.progress_lock:
            return key in self.progress_set

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        with self.key_lock:
            available_descs = self.pool.get_num_available_descs()
            num_evict = len(keys) - available_descs
            if num_evict > 0:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.key_dict, num_candidates=num_evict
                )

                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Backend under pressure."
                    )
                    return None

                self.batched_remove(evict_keys, force=False)

        with self.progress_lock:
            for key in keys:
                self.progress_set.add(key)

        asyncio.run_coroutine_threadsafe(
            self.mem_to_storage(keys, memory_objs), self.loop
        )

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """

        future = asyncio.run_coroutine_threadsafe(self.storage_to_mem([key]), self.loop)

        obj_list = future.result()
        return obj_list[0]

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.

        :return: a list of memory objects.
        """

        if not keys:
            return []

        future = asyncio.run_coroutine_threadsafe(self.storage_to_mem(keys), self.loop)

        obj_list = future.result()
        return obj_list

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        obj_list = await self.storage_to_mem(keys)
        assert None not in obj_list
        return cast(list[MemoryObj], obj_list)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """

        with self.key_lock:
            metadata = self.key_dict.pop(key, None)
            if metadata is None:
                return False
            if force:
                self.cache_policy.update_on_force_evict(key)

        self.pool.push(metadata.index)
        return True

    def pin(self, key: CacheEngineKey) -> bool:
        with self.key_lock:
            if key in self.key_dict:
                self.key_dict[key].pin()
                return True
            else:
                return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.key_lock:
            if key in self.key_dict:
                self.key_dict[key].unpin()
                return True
            else:
                return False

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.agent.close()
        self.pool.close()
        self.memory_allocator.close()

        if self.free_pinned_buffer:
            _free_cpu_memory(self.buffer)


class NixlDynamicStorageBackend(NixlStorageBackend):
    def __init__(
        self,
        nixl_config: NixlStorageConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        cache_policy: Optional[PresenceCache] = None,
    ):
        super().__init__(nixl_config, config, metadata, loop)

        self.async_mode = nixl_config.enable_async_put
        self.enable_presence_cache = nixl_config.enable_presence_cache

        # Presence cache to reduce remote contains checks
        self.hit_counter = 0
        self.total_counter = 0
        self.key_presence_cache: Optional[PresenceCache] = None
        if self.enable_presence_cache:
            self.key_presence_cache = (
                cache_policy if cache_policy is not None else SetPresenceCache()
            )

        # Initialize metadata from config
        self.meta_shape: Optional[torch.Size] = None
        self.meta_dtype: Optional[torch.dtype] = None
        self.meta_fmt: Optional[MemoryFormat] = None
        self.init_chunk_meta(metadata)

        self.agent = NixlDynamicStorageAgent(
            self.memory_allocator,
            nixl_config.buffer_device,
            nixl_config.backend,
            nixl_config.backend_params,
        )

    def set_presence_cache(self, cache: PresenceCache) -> None:
        """Configure a custom cache policy for key presence tracking."""
        if self.enable_presence_cache:
            self.key_presence_cache = cache

    def _cache_contains(self, chunk_hash: int) -> bool:
        if not self.enable_presence_cache or self.key_presence_cache is None:
            return False
        found = self.key_presence_cache.contains(chunk_hash)
        self.hit_counter += 1 if found else 0
        self.total_counter += 1
        if self.total_counter % 100 == 0:
            logger.debug(f"Cache hit: {self.hit_counter} vs {self.total_counter}")
        return found

    def _cache_add(self, chunk_hash: int) -> None:
        if not self.enable_presence_cache or self.key_presence_cache is None:
            return
        self.key_presence_cache.add(chunk_hash)

    def _cache_discard(self, chunk_hash: int) -> None:
        if not self.enable_presence_cache or self.key_presence_cache is None:
            return
        self.key_presence_cache.discard(chunk_hash)

    def init_chunk_meta(
        self,
        metadata: Optional[LMCacheEngineMetadata],
    ) -> None:
        """Initialize chunk metadata similar to base_connector.init_chunk_meta()"""
        if metadata is None:
            return

        self.meta_shape = torch.Size(
            [
                metadata.kv_shape[1],
                metadata.kv_shape[0],
                metadata.kv_shape[2],
                metadata.kv_shape[3] * metadata.kv_shape[4],
            ]
        )
        self.meta_dtype = metadata.kv_dtype
        self.meta_fmt = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )
        logger.info(
            f"Initialized nixl object backend metadata: "
            f"shape: {self.meta_shape}, "
            f"dtype: {self.meta_dtype}, "
            f"fmt: {self.meta_fmt}"
        )

    def _format_object_key(self, key: CacheEngineKey) -> str:
        """
        Generate object key name based on CacheEngineKey information.
        Similar to s3_connector._format_safe_path()
        """
        key_str = key.to_string()
        # Replace slashes with underscores to make it safe for object storage
        flat_key_str = key_str.replace("/", "_").replace("@", "_")
        # URL encode for safety
        return url_quote(flat_key_str, safe="")

    def key_exists(self, key: CacheEngineKey) -> bool:
        if self.agent.mem_type == "OBJ":
            meta_info = self._format_object_key(key)
        else:
            # Already validated in validate_nixl_backend
            raise ValueError(f"unexpected mem_type: {self.agent.mem_type}")

        return self.agent.nixl_desc_exists(meta_info)

    def storage_to_mem(
        self, keys: list[CacheEngineKey], pin: bool = False
    ) -> list[Optional[MemoryObj]]:
        obj_list: list[Optional[MemoryObj]] = []
        page_size = self.memory_allocator.align_bytes
        start_time = time.time()
        storage_indices = []
        mem_indices = []
        descs = []

        # Prepare mem and storage indices
        for idx in range(len(keys)):
            # Allocate memory for the object
            assert self.meta_shape is not None
            assert self.meta_dtype is not None
            assert self.meta_fmt is not None
            obj = self.memory_allocator.allocate(
                self.meta_shape, self.meta_dtype, self.meta_fmt
            )
            if obj is None:
                # free previous allocated objects
                logger.warning("Failed to allocate memory")
                for obj in obj_list:
                    self.memory_allocator.free(obj)
                return [None] * len(keys)

            obj_list.append(obj)
            mem_indices.append(obj.meta.address)
            storage_indices.append(idx)

        if self.agent.mem_type == "OBJ":
            for idx in range(len(keys)):
                object_key = self._format_object_key(keys[idx])
                descs.append(NixlDesc(device_id=idx, meta_info=object_key))
        else:
            # Already validated in validate_nixl_backend
            raise ValueError(f"unexpected mem_type: {self.agent.mem_type}")

        # Create batched storage handler
        storage_reg_descs, storage_xfer_handler = (
            self.agent.create_batched_storage_handler(descs, page_size)
        )
        # Create transfer handle
        handle = self.agent.get_storage_to_mem_handle(
            mem_indices, storage_xfer_handler, storage_indices
        )

        # Try to read the object
        try:
            self.agent.post_blocking(handle)
            xfer_state = True
        except nixlBind.nixlBackendError as exc:
            logger.warning(f"Batch Transfer failed: {exc}")
            # Treat "not found", timeout or other transfer failures as recoverable
            # Do not raise exception to avoid terminating the program
            xfer_state = False

        self.agent.release_handle(handle)
        self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)

        if xfer_state:
            for i in range(len(keys)):
                key = keys[i]
                self._cache_add(key.chunk_hash)
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"storage_to_mem for {len(keys)} objects size {page_size * len(keys)} "
                f"took {duration:.6f} seconds"
            )
            return obj_list
        else:
            # Free the allocated memory and discard cache if transfer failed
            for obj in obj_list:
                self.memory_allocator.free(obj)
            for key in keys:
                self._cache_discard(key.chunk_hash)
            return [None] * len(keys)

    async def _wait_for_transfer(
        self,
        handle: NixlXferHandle,
        initial_state: str,
        keys: Sequence[CacheEngineKey],
        storage_reg_descs: nixlBind.nixlRegDList,
        storage_xfer_handler: NixlDlistHandle,
        mem_objs: List[MemoryObj],
    ):
        """Asynchronously wait for transfer to complete without blocking."""
        state = ""
        try:
            state = initial_state
            while state != "DONE" and state != "ERR":
                state = self.agent.nixl_agent.check_xfer_state(handle)
                await asyncio.sleep(0.001)  # Avoid busy-waiting, yield to event loop
            if state == "ERR":
                raise RuntimeError("NIXL transfer failed")

        finally:
            # Release the handle after transfer completes (success or failure)
            self.agent.release_handle(handle)
            self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)

            if state == "DONE":
                for key in keys:
                    with self.progress_lock:
                        self.progress_set.discard(key)
                    self._cache_add(key.chunk_hash)

            for mem_obj in mem_objs:
                mem_obj.ref_count_down()

    async def mem_to_storage(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> None:
        start_time = time.time()
        if len(keys) == 0:
            return

        storage_indices = range(len(keys))
        mem_indices = [mem_obj.meta.address for mem_obj in mem_objs]
        page_size = self.memory_allocator.align_bytes
        descs = []

        if self.agent.mem_type == "OBJ":
            for idx in range(len(keys)):
                object_key = self._format_object_key(keys[idx])
                descs.append(NixlDesc(device_id=idx, meta_info=object_key))
        else:
            # Already validated in validate_nixl_backend
            raise ValueError(f"unexpected mem_type: {self.agent.mem_type}")

        storage_reg_descs, storage_xfer_handler = (
            self.agent.create_batched_storage_handler(descs, page_size)
        )

        handle = self.agent.get_mem_to_storage_handle(
            mem_indices, storage_xfer_handler, storage_indices
        )

        if self.async_mode:
            initial_state = self.agent.post_async(handle)
            # Submit the async wait to the event loop and return immediately
            asyncio.create_task(
                self._wait_for_transfer(
                    handle,
                    initial_state,
                    keys,
                    storage_reg_descs,
                    storage_xfer_handler,
                    mem_objs,
                )
            )
        else:
            self.agent.post_blocking(handle)
            self.agent.release_handle(handle)
            self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)

            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"mem_to_storage for {len(keys)} objects size {page_size * len(keys)} "
                f"took {duration:.3f} seconds"
            )

            for key in keys:
                with self.progress_lock:
                    self.progress_set.discard(key)
                self._cache_add(key.chunk_hash)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.

        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        with self.progress_lock:
            return key in self.progress_set

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        This method uses nixl querymem to check existence.
        If successful, it caches the name for later use.

        :param key: The key to check
        :param pin: Whether to pin the object in the backend
            (Not Implemented)

        :return: True if the key exists, False otherwise
        """
        # Check if already in progress
        if self.exists_in_put_tasks(key):
            logger.debug(f"Key {key.chunk_hash:x} is in put tasks")
            return False

        # Check presence cache before hitting remote storage if not prefetching
        if self._cache_contains(key.chunk_hash):
            return True

        xfer_state = self.key_exists(key)
        if xfer_state:
            self._cache_add(key.chunk_hash)

        return xfer_state

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if not keys:
            return 0
        """
        Nixl API query_memory also supports batched query. However when there
        are hundreds of keys to be queried and keys in the first few places
        don't exist, the batched query has to be failed fast.
        Therefore we implement batched contains() in a managed thread pool,
        which fails fast when a key doesn't exist.
        """
        n = len(keys)
        idx = 0
        batch_size = 16

        while idx < n:
            batch = keys[idx : idx + batch_size]
            tasks = [asyncio.to_thread(self.contains, k, pin) for k in batch]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            # Stop at the first miss
            for j, ok in enumerate(results):
                if not ok:
                    return idx + j
            idx += len(batch)

        return n

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        with self.progress_lock:
            for key in keys:
                self.progress_set.add(key)

        if self.async_mode:
            for mem_obj in memory_objs:
                mem_obj.ref_count_up()
            asyncio.run_coroutine_threadsafe(
                self.mem_to_storage(keys, memory_objs), self.loop
            )
        else:
            future = asyncio.run_coroutine_threadsafe(
                self.mem_to_storage(keys, memory_objs), self.loop
            )
            future.result()

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """
        obj_list = self.storage_to_mem([key], False)
        return obj_list[0]

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        A blocking function to get the kv cache from the storage backend.
        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.
        :return: a list of memory objects.
        """
        if not keys:
            return []

        obj_list = self.storage_to_mem(keys, False)
        return obj_list

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        Non blocking interface to get the kv cache from the storage backend.
        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.
        :return: a list of memory objects.
        """
        obj_list = self.storage_to_mem(keys, False)
        assert None not in obj_list
        return cast(list[MemoryObj], obj_list)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        :param force: Whether to force removal (not used in this implementation)
        """
        self._cache_discard(key.chunk_hash)
        return True

    def pin(self, key: CacheEngineKey) -> bool:
        """
        Not implemented yet
        """
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        """
        Not implemented yet
        """
        return False

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.agent.close()
        self.memory_allocator.close()

        if self.free_pinned_buffer:
            _free_cpu_memory(self.buffer)
