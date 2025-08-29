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
from concurrent.futures import Future
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set
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
    MemoryObj,
    MemoryObjMetadata,
    PagedTensorMemoryAllocator,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector.nixl_utils import get_correct_nixl_device

logger = init_logger(__name__)


@dataclass
class NixlStorageConfig:
    buffer_size: int
    file_pool_size: int
    buffer_device: str
    path: str
    backends: list[str]

    @staticmethod
    def validate_nixl_backend(backend: str, device: str):
        if backend in ("GDS", "GDS_MT"):
            return device == "cpu" or device == "cuda"
        elif backend in ("POSIX", "HF3FS"):
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
        assert extra_config.get("nixl_backend") is not None
        assert extra_config.get("nixl_path") is not None
        assert extra_config.get("nixl_file_pool_size") is not None

        assert NixlStorageConfig.validate_nixl_backend(
            extra_config.get("nixl_backend"), config.nixl_buffer_device
        ), "Invalid NIXL backend & device combination"

        corrected_device = get_correct_nixl_device(
            config.nixl_buffer_device, metadata.worker_id
        )

        return NixlStorageConfig(
            buffer_size=config.nixl_buffer_size,
            file_pool_size=extra_config.get("nixl_file_pool_size"),
            buffer_device=corrected_device,
            path=extra_config.get("nixl_path"),
            backends=[extra_config.get("nixl_backend")],
        )


class NixlFilePool:
    def __init__(self, path: str, size: int):
        self.lock = threading.Lock()
        self.size: int = size
        self.fds: List[int] = []
        self.indices: List[int] = []

        for i in reversed(range(size)):
            tmp_path = path + f"obj_{i}_{uuid.uuid4().hex[0:4]}.bin"
            fd = os.open(tmp_path, os.O_CREAT | os.O_RDWR)
            self.fds.append(fd)
            self.indices.append(i)

    def pop(self) -> int:
        with self.lock:
            assert len(self.indices) > 0
            return self.indices.pop()

    def push(self, index: int):
        with self.lock:
            assert len(self.indices) < self.size
            self.indices.append(index)

    def close(self):
        # TODO: do we need to delete the files?
        with self.lock:
            assert len(self.fds) == self.size
            for fd in self.fds:
                os.close(fd)


class NixlStorageAgent:
    agent_name: str
    nixl_agent: NixlAgent
    file_pool: NixlFilePool
    reg_descs: nixlBind.nixlRegDList
    file_reg_descs: nixlBind.nixlRegDList
    xfer_descs: nixlBind.nixlXferDList
    file_xfer_descs: nixlBind.nixlXferDList
    xfer_handler: NixlDlistHandle
    file_xfer_handler: NixlDlistHandle

    def __init__(
        self,
        allocator: PagedTensorMemoryAllocator,
        file_pool: NixlFilePool,
        device: str,
        backends: list[str],
    ):
        buffer_ptr = allocator.buffer_ptr
        buffer_size = allocator.buffer_size
        page_size = allocator.align_bytes

        self.agent_name = "NixlAgent_" + str(uuid.uuid4())
        nixl_conf = NixlAgentConfig(backends=backends)
        self.nixl_agent = NixlAgent(self.agent_name, nixl_conf)

        device_id = torch.cuda.current_device()
        self.init_handlers(device, buffer_ptr, buffer_size, page_size, device_id)

        self.init_file_handlers(page_size, file_pool.fds)

    def init_handlers(self, device, buffer_ptr, buffer_size, page_size, device_id):
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

        self.reg_descs = reg_descs
        self.xfer_descs = xfer_descs
        self.xfer_handler = xfer_handler

    def init_file_handlers(self, page_size, fds):
        reg_list = [(0, page_size, fd, "") for fd in fds]
        xfer_desc = [(0, page_size, fd) for fd in fds]
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="FILE")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="FILE")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="FILE"
        )

        self.file_reg_descs = reg_descs
        self.file_xfer_descs = xfer_descs
        self.file_xfer_handler = xfer_handler

    def get_gpu_to_file_handle(self, mem_indices, file_indices) -> NixlXferHandle:
        return self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.xfer_handler,
            mem_indices,
            self.file_xfer_handler,
            file_indices,
        )

    def get_file_to_gpu_handle(self, mem_indices, file_indices) -> NixlXferHandle:
        return self.nixl_agent.make_prepped_xfer(
            "READ", self.xfer_handler, mem_indices, self.file_xfer_handler, file_indices
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
        self.nixl_agent.release_dlist_handle(self.file_xfer_handler)
        self.nixl_agent.release_dlist_handle(self.xfer_handler)
        self.nixl_agent.deregister_memory(self.file_reg_descs)
        self.nixl_agent.deregister_memory(self.reg_descs)


class NixlStorageBackend(StorageBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the
    implementation.
    """

    def __init__(
        self,
        nixl_config: NixlStorageConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: PagedTensorMemoryAllocator,
    ):
        """
        Initialize the Nixl storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=nixl_config.buffer_device)

        self.loop = loop
        self.key_lock = threading.Lock()
        self.key_dict: dict[int, MemoryObjMetadata] = {}

        self.progress_lock = threading.Lock()
        self.progress_set: Set[int] = set()

        self.memory_allocator = memory_allocator

        self.file_pool = NixlFilePool(nixl_config.path, nixl_config.file_pool_size)
        self.agent = NixlStorageAgent(
            memory_allocator,
            self.file_pool,
            nixl_config.buffer_device,
            nixl_config.backends,
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        :param key: The key to check
        :param pin: Whether to pin the object in the backend.

        :return: True if the key exists, False otherwise
        """

        with self.key_lock:
            chunk_hash = key.chunk_hash
            if chunk_hash in self.key_dict and not self.exists_in_put_tasks(key):
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
            return key.chunk_hash in self.progress_set

    def add_key_to_dict(
        self, key: CacheEngineKey, obj: MemoryObjMetadata, index: int
    ) -> None:
        with self.key_lock:
            assert key.chunk_hash not in self.key_dict
            self.key_dict[key.chunk_hash] = MemoryObjMetadata(
                shape=obj.shape,
                dtype=obj.dtype,
                fmt=obj.fmt,
                phy_size=obj.phy_size,
                ref_count=1,
                address=index,
            )

    async def gpu_to_file(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> None:
        mem_indices = [mem_obj.meta.address for mem_obj in mem_objs]

        file_indices = []
        for i in range(len(keys)):
            index = self.file_pool.pop()
            file_indices.append(index)
            self.add_key_to_dict(keys[i], mem_objs[i].meta, index)

        handle = self.agent.get_gpu_to_file_handle(mem_indices, file_indices)
        self.agent.post_blocking(handle)
        self.agent.release_handle(handle)

        for key in keys:
            with self.progress_lock:
                self.progress_set.discard(key.chunk_hash)

    async def file_to_gpu(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.key_lock:
            metadata = self.key_dict.get(key.chunk_hash)
            if metadata is None:
                return None

        dtype = metadata.dtype
        shape = metadata.shape
        fmt = metadata.fmt
        assert dtype is not None
        assert shape is not None
        assert fmt is not None

        obj = self.memory_allocator.allocate(shape, dtype, fmt, allocator_type="nixl")
        if obj is None:
            return None

        handle = self.agent.get_file_to_gpu_handle(
            [obj.metadata.address], [metadata.address]
        )
        self.agent.post_blocking(handle)
        self.agent.release_handle(handle)

        return obj

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        with self.progress_lock:
            for key in keys:
                self.progress_set.add(key.chunk_hash)

        asyncio.run_coroutine_threadsafe(self.gpu_to_file(keys, memory_objs), self.loop)

    def submit_prefetch_task(self, key: CacheEngineKey) -> bool:
        """
        An async function to get the MemoryObj from the storage backend.

        :param key: The key of the MemoryObj.

        :return: True if prefetch succeeded, else False.
        """

        return False

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """

        future = self.get_non_blocking(key)

        if future is None:
            return None

        return future.result()

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        future = asyncio.run_coroutine_threadsafe(self.file_to_gpu(key), self.loop)

        return future

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """

        with self.key_lock:
            metadata = self.key_dict.pop(key.chunk_hash, None)
            if metadata is None:
                return False

        self.file_pool.push(metadata.address)
        return True

    def pin(self, key: CacheEngineKey) -> bool:
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        return False

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.agent.close()

        self.file_pool.close()

    @staticmethod
    def CreateNixlStorageBackend(
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        metadata: LMCacheEngineMetadata,
        memory_allocator: PagedTensorMemoryAllocator,
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
        backend = NixlStorageBackend(nixl_config, loop, memory_allocator)
        return backend
