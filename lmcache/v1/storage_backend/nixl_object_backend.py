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
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union, cast
from urllib.parse import quote as url_quote
import asyncio
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
    PagedTensorMemoryAllocator,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


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
class NixlObjectConfig:
    buffer_size: int
    buffer_device: str
    backend: str
    backend_params: dict[str, str]
    enable_presence_cache: bool
    enable_async_put: bool

    @staticmethod
    def validate_nixl_backend(backend: str, device: str):
        if backend in ("OBJ",):
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
        assert extra_config.get("enable_nixl_object")

        backend = extra_config.get("nixl_object_backend")

        assert backend is not None

        assert NixlObjectConfig.validate_nixl_backend(
            backend, config.nixl_buffer_device
        ), "Invalid NIXL object backend & device combination"

        backend_params = extra_config.get("nixl_object_backend_params")
        if backend_params is None:
            backend_params = {}

        corrected_device = get_correct_device(
            config.nixl_buffer_device, metadata.worker_id
        )

        enable_presence_cache = extra_config.get("nixl_object_presence_cache", False)
        enable_async_put = extra_config.get("nixl_object_async_put", False)

        return NixlObjectConfig(
            buffer_size=config.nixl_buffer_size,
            buffer_device=corrected_device,
            backend=backend,
            backend_params=backend_params,
            enable_presence_cache=enable_presence_cache,
            enable_async_put=enable_async_put,
        )


class NixlObjectAgent:
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

        self.agent_name = "NixlObjectAgent_" + str(uuid.uuid4())
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

    def create_batched_storage_handler(self, object_keys: list[str], page_size: int):
        reg_list = []
        xfer_desc = []
        for i in range(len(object_keys)):
            reg_list.append((0, page_size, i, object_keys[i]))
            xfer_desc.append((0, page_size, i))

        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type="OBJ"
        )
        return reg_descs, xfer_handler

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

    def post_blocking_async(self, handle: NixlXferHandle):
        """Non-blocking async post for WRITE operations."""
        state = self.nixl_agent.transfer(handle)
        return state

    def release_handle(self, handle):
        self.nixl_agent.release_xfer_handle(handle)

    def release_storage_handler(self, reg_descs, xfer_handler):
        """Release storage handler resources"""
        self.nixl_agent.release_dlist_handle(xfer_handler)
        self.nixl_agent.deregister_memory(reg_descs)

    def nixl_obj_exists(self, object_key: str) -> bool:
        reg_list = [(0, 0, 0, object_key)]
        try:
            resp = self.nixl_agent.query_memory(reg_list, "OBJ", mem_type="OBJ")
            # nixl api query_memory returns a list of nixlRegDesc
            if resp[0] is None:
                return False
            return True
        except Exception as exc:
            logger.warning(f"NIXL Object {object_key} query failed: {exc}")
            return False

    def close(self):
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)


class NixlObjectBackend(AllocatorBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl with OBJ plugin.

    This backend uses object storage naming based on CacheEngineKey information
    instead of pre-allocated object pools.
    """

    def __init__(
        self,
        nixl_config: NixlObjectConfig,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        cache_policy: Optional[PresenceCache] = None,
    ):
        """
        Initialize the Nixl object storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=nixl_config.buffer_device)

        self.loop = loop
        self.progress_lock = threading.Lock()
        self.progress_set: set[int] = set()

        self.async_mode = nixl_config.enable_async_put
        self.enable_presence_cache = nixl_config.enable_presence_cache
        self.memory_allocator = self.initialize_allocator(config, metadata)

        self.agent = NixlObjectAgent(
            self.memory_allocator,
            nixl_config.buffer_device,
            nixl_config.backend,
            nixl_config.backend_params,
        )

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
        object_key = self._format_object_key(key)
        return self.agent.nixl_obj_exists(object_key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.

        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        with self.progress_lock:
            return key.chunk_hash in self.progress_set

    def storage_to_mem(
        self, keys: list[CacheEngineKey], pin: bool = False
    ) -> list[Optional[MemoryObj]]:
        obj_list: list[Optional[MemoryObj]] = []
        object_keys = []
        mem_indices = []
        storage_indices = []
        page_size = self.memory_allocator.align_bytes
        start_time = time.time()

        # Prepare mem and storage indices
        for idx in range(len(keys)):
            object_key = self._format_object_key(keys[idx])
            object_keys.append(object_key)

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
            mem_indices.append(obj.metadata.address)
            storage_indices.append(idx)

        # Create batched storage handler
        storage_reg_descs, storage_xfer_handler = (
            self.agent.create_batched_storage_handler(object_keys, page_size)
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
        try:
            state = initial_state
            while state != "DONE" and state != "ERR":
                state = self.agent.nixl_agent.check_xfer_state(handle)
                await asyncio.sleep(0.001)  # Avoid busy-waiting, yield to event loop
            if state == "ERR":
                raise RuntimeError("NIXL transfer failed")

            for key in keys:
                with self.progress_lock:
                    self.progress_set.discard(key.chunk_hash)
                self._cache_add(key.chunk_hash)
        finally:
            # Release the handle after transfer completes (success or failure)
            self.agent.release_handle(handle)
            self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)
            for mem_obj in mem_objs:
                mem_obj.ref_count_down()

    async def mem_to_storage(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> None:
        start_time = time.time()
        if len(keys) == 0:
            return

        object_keys = []
        mem_indices = [mem_obj.meta.address for mem_obj in mem_objs]
        page_size = self.memory_allocator.align_bytes
        storage_indices = []

        for i in range(len(keys)):
            # Generate object key based on CacheEngineKey
            object_key = self._format_object_key(keys[i])
            object_keys.append(object_key)
            storage_indices.append(i)

        storage_reg_descs, storage_xfer_handler = (
            self.agent.create_batched_storage_handler(object_keys, page_size)
        )

        handle = self.agent.get_mem_to_storage_handle(
            mem_indices, storage_xfer_handler, storage_indices
        )

        if self.async_mode:
            for mem_obj in mem_objs:
                mem_obj.ref_count_up()
            initial_state = self.agent.post_blocking_async(handle)
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
                    self.progress_set.discard(key.chunk_hash)
                self._cache_add(key.chunk_hash)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        with self.progress_lock:
            for key in keys:
                self.progress_set.add(key.chunk_hash)

        if self.async_mode:
            asyncio.run_coroutine_threadsafe(
                self.mem_to_storage(keys, memory_objs), self.loop
            )
        else:
            asyncio.run(self.mem_to_storage(keys, memory_objs))

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

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> PagedTensorMemoryAllocator:
        extra_config = config.extra_config
        enable_nixl_object = extra_config is not None and extra_config.get(
            "enable_nixl_object"
        )
        assert enable_nixl_object
        corrected_device = get_correct_device(
            config.nixl_buffer_device,
            metadata.worker_id,
        )

        buffer = torch.empty(
            config.nixl_buffer_size,
            dtype=torch.uint8,
            device=corrected_device,
        )

        if corrected_device == "cpu":
            torch.cuda.cudart().cudaHostRegister(
                buffer.data_ptr(), config.nixl_buffer_size, 0
            )
        else:
            logger.info(f"Setting cuda device to {corrected_device} ")
            torch.cuda.set_device(corrected_device)

        return PagedTensorMemoryAllocator(
            buffer,
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
        """
        eviction and busy loop are not supported for now.
        """
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
        """
        eviction and busy loop are not supported for now.
        """
        return self.memory_allocator.batched_allocate(shape, dtype, batch_size, fmt)

    def get_allocator_backend(self):
        return self

    @staticmethod
    def CreateNixlObjectBackend(
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        metadata: LMCacheEngineMetadata,
    ):
        """
        Create a Nixl object backend with the given configuration.

        :param config: The LMCache engine configuration.
        :param loop: The asyncio event loop.
        :param metadata: The LMCache engine metadata.

        :return: A NixlObjectBackend instance.
        """
        # Create the Nixl object config
        nixl_config = NixlObjectConfig.from_cache_engine_config(config, metadata)
        # Create the Nixl object backend
        backend = NixlObjectBackend(nixl_config, config, metadata, loop)
        return backend
