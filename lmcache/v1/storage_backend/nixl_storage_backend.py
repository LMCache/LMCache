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
from typing import Any, List, Optional, Sequence, Set, cast
import asyncio
import os
import threading
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
    path: str
    backend: str
    use_direct_io: bool
    backend_params: dict[str, str]

    @staticmethod
    def validate_nixl_backend(backend: str, device: str):
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

        pool_size = extra_config.get("nixl_pool_size")
        backend = extra_config.get("nixl_backend")
        path = extra_config.get("nixl_path")

        assert pool_size is not None
        assert backend is not None

        use_direct_io = extra_config.get("use_direct_io", False)
        assert use_direct_io in [False, True]

        assert NixlStorageConfig.validate_nixl_backend(
            backend, config.nixl_buffer_device
        ), "Invalid NIXL backend & device combination"

        backend_params = extra_config.get("nixl_backend_params")
        if backend_params is None:
            backend_params = {}

        corrected_device = get_correct_device(
            config.nixl_buffer_device, metadata.worker_id
        )

        return NixlStorageConfig(
            buffer_size=config.nixl_buffer_size,
            pool_size=pool_size,
            buffer_device=corrected_device,
            path=path,
            use_direct_io=use_direct_io,
            backend=backend,
            backend_params=backend_params,
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


class NixlStorageAgent:
    agent_name: str
    nixl_agent: NixlAgent
    pool: NixlDescPool
    mem_reg_descs: nixlBind.nixlRegDList
    storage_reg_descs: nixlBind.nixlRegDList
    mem_xfer_descs: nixlBind.nixlXferDList
    storage_xfer_descs: nixlBind.nixlXferDList
    mem_xfer_handler: NixlDlistHandle
    storage_xfer_handler: NixlDlistHandle

    def __init__(
        self,
        allocator: PagedTensorMemoryAllocator,
        pool: NixlDescPool,
        device: str,
        backend: str,
        backend_params: dict[str, str],
    ):
        buffer_ptr = allocator.buffer_ptr
        buffer_size = allocator.buffer_size
        page_size = allocator.align_bytes

        self.agent_name = "NixlAgent_" + str(uuid.uuid4())
        nixl_conf = NixlAgentConfig(backends=[])
        self.nixl_agent = NixlAgent(self.agent_name, nixl_conf)
        self.nixl_agent.create_backend(backend, backend_params)

        device_id = torch.cuda.current_device()
        self.init_mem_handlers(device, buffer_ptr, buffer_size, page_size, device_id)

        if isinstance(pool, NixlFilePool):
            self.init_storage_handlers_file(page_size, pool.fds)
        elif isinstance(pool, NixlObjectPool):
            self.init_storage_handlers_object(page_size, pool.keys)
        else:
            raise TypeError(f"Unsupported pool type: {type(pool).__name__}")

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
        self.mem_xfer_descs = xfer_descs
        self.mem_xfer_handler = xfer_handler

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

    def get_mem_to_storage_handle(self, mem_indices, storage_indices) -> NixlXferHandle:
        return self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.mem_xfer_handler,
            mem_indices,
            self.storage_xfer_handler,
            storage_indices,
        )

    def get_storage_to_mem_handle(self, mem_indices, storage_indices) -> NixlXferHandle:
        return self.nixl_agent.make_prepped_xfer(
            "READ",
            self.mem_xfer_handler,
            mem_indices,
            self.storage_xfer_handler,
            storage_indices,
        )

    def post_blocking(self, handle: NixlXferHandle):
        state = self.nixl_agent.transfer(handle)

        while state != "DONE" and state != "ERR":
            state = self.nixl_agent.check_xfer_state(handle)
        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")

    def release_handle(self, handle):
        self.nixl_agent.release_xfer_handle(handle)

    def close(self):
        self.nixl_agent.release_dlist_handle(self.storage_xfer_handler)
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.storage_reg_descs)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)


class NixlStorageBackend(AllocatorBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the
    implementation.
    """

    @staticmethod
    def createPool(backend: str, size: int, path: str, use_direct_io: bool):
        if backend in ("GDS", "GDS_MT", "POSIX", "HF3FS"):
            return NixlFilePool(size, path, use_direct_io)
        elif backend in ("OBJ"):
            return NixlObjectPool(size)
        else:
            raise ValueError(f"Unsupported NIXL backend: {backend}")

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
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.key_dict = self.cache_policy.init_mutable_mapping()

        self.progress_lock = threading.RLock()
        self.progress_set: Set[CacheEngineKey] = set()

        self.memory_allocator = self.initialize_allocator(config, metadata)

        self.pool = NixlStorageBackend.createPool(
            nixl_config.backend,
            nixl_config.pool_size,
            nixl_config.path,
            nixl_config.use_direct_io,
        )
        assert self.pool is not None

        self.agent = NixlStorageAgent(
            self.memory_allocator,
            self.pool,
            nixl_config.buffer_device,
            nixl_config.backend,
            nixl_config.backend_params,
        )

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

    async def storage_to_mem(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        obj_list: list[Optional[MemoryObj]] = []
        mem_indices = []
        storage_indices = []
        with self.key_lock:
            for key in keys:
                metadata = self.key_dict.get(key)
                if metadata is None:
                    obj_list.append(None)
                    continue

                self.cache_policy.update_on_hit(key, self.key_dict)

                dtype = metadata.dtype
                shape = metadata.shape
                fmt = metadata.fmt
                assert dtype is not None
                assert shape is not None
                assert fmt is not None

                obj = self.memory_allocator.allocate(shape, dtype, fmt)
                assert obj is not None

                obj_list.append(obj)

                mem_indices.append(obj.metadata.address)
                storage_indices.append(metadata.index)

        if not mem_indices:
            return obj_list

        handle = self.agent.get_storage_to_mem_handle(mem_indices, storage_indices)
        self.agent.post_blocking(handle)
        self.agent.release_handle(handle)

        return obj_list

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

        if future is None:
            return None

        obj_list = future.result()
        return obj_list[0]

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
            torch.Size(metadata.kv_shape),
            metadata.kv_dtype,
            MemoryFormat.KV_2LTD,
        )

    def get_memory_allocator(self):
        return self.memory_allocator

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if busy_loop:
            logger.warning("NixlStorageBackend does not support busy loop for now")

        return self.memory_allocator.allocate(shape, dtype, fmt)

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if busy_loop:
            logger.warning("NixlStorageBackend does not support busy loop for now")

        return self.memory_allocator.batched_allocate(shape, dtype, batch_size, fmt)

    def get_allocator_backend(self):
        return self

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
        backend = NixlStorageBackend(nixl_config, config, metadata, loop)
        return backend
