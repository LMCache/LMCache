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
from typing import Any, List, Optional, Sequence, cast
from urllib.parse import quote as url_quote
import asyncio
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
import time

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
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


@dataclass
class NixlObjectConfig:
    buffer_size: int
    buffer_device: str
    backend: str
    backend_params: dict[str, str]
    lookup_and_fetch: bool

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

        lookup_and_fetch = extra_config.get("nixl_object_lookup_and_fetch", False)

        return NixlObjectConfig(
            buffer_size=config.nixl_buffer_size,
            buffer_device=corrected_device,
            backend=backend,
            backend_params=backend_params,
            lookup_and_fetch=lookup_and_fetch,
        )


@dataclass
class NixlObjectMapEntry:
    """Entry in the NixlObjectMap to track object information"""
    obj: Optional[MemoryObj]
    object_key: str


class NixlObjectMap:
    """Map to track CacheEngineKey to object information"""
    def __init__(self):
        self.lock = threading.Lock()
        self.map: dict[int, NixlObjectMapEntry] = {}

    def add(self, chunk_hash: int, obj: Optional[MemoryObj], object_key: str):
        with self.lock:
            self.map[chunk_hash] = NixlObjectMapEntry(obj, object_key)
        #logger.info(f"OOOO Added key {chunk_hash:x} now map size is {len(self.map)}")

    def get(self, chunk_hash: int) -> Optional[NixlObjectMapEntry]:
        with self.lock:
            return self.map.get(chunk_hash)

    def contains(self, chunk_hash: int) -> bool:
        with self.lock:
            return chunk_hash in self.map

    def pop(self, chunk_hash: int) -> Optional[NixlObjectMapEntry]:
        with self.lock:
            entry = self.map.pop(chunk_hash, None)
            #logger.info(f"OOOO Poped key {chunk_hash:x} now map size is {len(self.map)}")
            return entry

class NixlObjectAgent:
    agent_name: str
    nixl_agent: NixlAgent
    mem_reg_descs: nixlBind.nixlRegDList
    mem_xfer_descs: nixlBind.nixlXferDList
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

        # Initialize monotonic indices for put and get operations

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

    def create_single_storage_handler(self, object_key: str, page_size: int):
        """Create a storage handler for a single specific object key"""
        reg_list = [(0, page_size, 0, object_key)]
        xfer_desc = [(0, page_size, 0)]
        
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="OBJ"
        )
        
        return reg_descs, xfer_descs, xfer_handler

    def create_batched_storage_handler(self, object_keys: str, page_size: int):
        """Create a storage handler for a single specific object key"""
        reg_list = []
        xfer_desc = []
        for i in range(len(object_keys)):
            reg_list.append((0, page_size, i, object_keys[i]))
            xfer_desc.append((0, page_size, i))
            logger.debug(f"Initializing storage handler for {object_keys[i]} with index {i}")
        
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        # descs not used
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="OBJ"
        )       
        return reg_descs, xfer_descs, xfer_handler

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

    def post_blocking(self, handle: NixlXferHandle) -> bool:
        state = self.nixl_agent.transfer(handle)

        while state != "DONE" and state != "ERR" and state != "MISS":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError as exc:
                logger.info(f"NIXL transfer failed: {exc}")
                state = "MISS"
            except Exception as exc:
                logger.info(f"Other exception: {exc}")

        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")
        
        return state == "DONE"

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
            if resp[0] is None:
                return False
            return True
        except Exception as exc:
            logger.debug(f"NIXL Object {object_key} qeury failed: {exc}")
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

        self.memory_allocator = self.initialize_allocator(config, metadata)

        self.agent = NixlObjectAgent(
            self.memory_allocator,
            nixl_config.buffer_device,
            nixl_config.backend,
            nixl_config.backend_params,
        )

        self.lookup_and_fetch = nixl_config.lookup_and_fetch

        # Initialize object map to track keys and their objects
        self.object_map = NixlObjectMap()

        # Initialize metadata from config
        self.meta_shape: Optional[torch.Size] = None
        self.meta_dtype: Optional[torch.dtype] = None
        self.meta_fmt: Optional[MemoryFormat] = None
        self.init_chunk_meta(config, metadata)

    def init_chunk_meta(
        self,
        config: Optional[LMCacheEngineConfig],
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

    def get_one_object_to_mem(self, key: CacheEngineKey, pin: bool = False) -> Optional[MemoryObj]:
        object_key = self._format_object_key(key)
        page_size = self.memory_allocator.align_bytes

        # Create storage handler for this object
        storage_reg_descs, storage_xfer_descs, storage_xfer_handler = (
            self.agent.create_single_storage_handler(object_key, page_size)
        )

        # Allocate memory for the object
        assert self.meta_shape is not None
        assert self.meta_dtype is not None
        assert self.meta_fmt is not None

        obj = self.memory_allocator.allocate(
            self.meta_shape, self.meta_dtype, self.meta_fmt
        )
        if obj is None:
            logger.warning("Failed to allocate memory for contains check")
            self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)
            return None

        mem_indices = [obj.metadata.address]
        storage_indices = [0]

        # Create transfer handle
        handle = self.agent.get_storage_to_mem_handle(
            mem_indices, storage_xfer_handler, storage_indices
        )

        # Try to read the object
        try:
            xfer_state = self.agent.post_blocking(handle)
        except nixlBind.nixlBackendError as exc:
            logger.debug(f"NIXL Object {object_key} does not exist: {exc}")
            xfer_state = False

        self.agent.release_handle(handle)
        self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)

        if xfer_state:
            # Object exists, cache it in the object map
            if pin:
                self.object_map.add(key.chunk_hash, obj, object_key)
                logger.debug(f"PREFETCH Key {key.chunk_hash:x} and keep in map")

            logger.debug(f"GET Key {key.chunk_hash:x} shape is {obj.tensor.size()}")
            #logger.debug(f"OOOO object tensor is {obj.tensor}")
            return obj
        else:
            # Object doesn't exist, free the allocated memory
            self.memory_allocator.free(obj)
            logger.debug(f"PREFETCH Key {key.chunk_hash:x} failed")
            return None

    def get_objects_to_mem(
        self, keys: list[CacheEngineKey], pin: bool = False
    ) -> list[Optional[MemoryObj]]:
        obj_list: list[Optional[MemoryObj]] = []
        object_keys = []
        mem_indices = []
        storage_indices = []
        page_size = self.memory_allocator.align_bytes
        idx = 0
        start_time = time.time()

        # Prepare mem and storage indice
        for key in keys:
            object_key = self._format_object_key(key)
            object_keys.append(object_key)

            # Allocate memory for the object
            assert self.meta_shape is not None
            assert self.meta_dtype is not None
            assert self.meta_fmt is not None
            obj = self.memory_allocator.allocate(
                self.meta_shape, self.meta_dtype, self.meta_fmt
            )
            if obj is None:
                logger.warning("Failed to allocate memory for contains check")
                self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)
                return None

            obj_list.append(obj)
            mem_indices.append(obj.metadata.address)
            storage_indices.append(idx)
            idx += 1
        
        # Create batched storage handler
        storage_reg_descs, storage_xfer_descs, storage_xfer_handler = (
            self.agent.create_batched_storage_handler(object_keys, page_size)
        )
        # Create transfer handle
        handle = self.agent.get_storage_to_mem_handle(
            mem_indices, storage_xfer_handler, storage_indices
        )

        # Try to read the object
        try:
            xfer_state = self.agent.post_blocking(handle)
        except nixlBind.nixlBackendError as exc:
            logger.debug(f"Batch Transfer failed: {exc}")
            return [None] * len(keys)

        self.agent.release_handle(handle)
        self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)

        if xfer_state:
            for i in range(len(keys)):
                key = keys[i]
                obj = obj_list[i]
                object_key = object_keys[i]
                # Cache them in the object map
                if pin:
                    self.object_map.add(key.chunk_hash, obj, object_key)
                    logger.debug(f"PREFETCH Key {key.chunk_hash:x} in batch and keep in map")
                else:
                    logger.debug(f"GET Key {key.chunk_hash:x} in batch")
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"get_objects_to_mem for {len(keys)} objects size {page_size*len(keys)} took {duration:.6f} seconds")                
            return obj_list
        else:
            # Object doesn't exist, free the allocated memory
            for obj in obj_list:
                self.memory_allocator.free(obj)
            logger.debug(f"Get keys failed")
            return [None] * len(keys)

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        This method attempts to read the object from storage to verify existence.
        If successful, it caches the object in the object_map for later retrieval.

        :param key: The key to check
        :param pin: Whether to pin the object in the backend (not used in this implementation)

        :return: True if the key exists, False otherwise
        """
        # Check if already in progress
        if self.exists_in_put_tasks(key):
            logger.debug(f"Key {key.chunk_hash:x} is in put tasks")
            return False

        # Check if already in object map
        if self.object_map.contains(key.chunk_hash):
            logger.debug(f"LOOKUP: Key {key.chunk_hash:x} exists in object map")
            return True

        xfer_state = False # not found by default
        if (pin and self.lookup_and_fetch):
            """Only enable prefetch if lookup_and_fetch is enabled"""
            logger.debug(f"PREFETCH key {key.chunk_hash:x}")
            '''retrieve and pin in memory for later use'''
            xfer_state = (self.get_one_object_to_mem(key, pin) is not None)
        else:
            xfer_state = self.key_exists(key)
            if xfer_state:
                logger.debug(f"LOOKUP Key {key.chunk_hash:x} exists in storage")
            else:
                logger.debug(f"LOOKUP Key {key.chunk_hash:x} does not exist in storage")

        return xfer_state

    """
    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if not keys:
            return 0

        #sem = asyncio.Semaphore(4)

        async def async_contains(key: CacheEngineKey) -> bool:
            # Limit concurrent threads
            #async with sem:
            return await asyncio.to_thread(self.contains, key, pin)

        # Start all tasks concurrently (semaphore will throttle)
        tasks = [asyncio.create_task(async_contains(k)) for k in keys]

        try:
            for i, t in enumerate(tasks):
                ok = await t
                if not ok:
                    # Cancel remaining tasks and return
                    for rt in tasks[i+1:]:
                        rt.cancel()
                    await asyncio.gather(*tasks[i+1:], return_exceptions=True)
                    return i
            return len(keys)
        except Exception:
            # On any error: cancel all and treat as miss=0
            for rt in tasks:
                if not rt.done():
                    rt.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return 0
    """

    async def mem_to_storage_parallel(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> None:
        start_time = time.time()
        if len(keys) == 0:
            return
        
        mem_indices = [mem_obj.meta.address for mem_obj in mem_objs]
        page_size = self.memory_allocator.align_bytes

        storage_indices = []
        reg_list = []
        xfer_desc = []

        for i in range(len(keys)):
            # Generate object key based on CacheEngineKey
            object_key = self._format_object_key(keys[i])
            storage_indices.append(i)

            logger.info(f"Initializing storage handler for {object_key} with index {i}")
            reg_list.append((0, page_size, i, object_key))
            xfer_desc.append((0, page_size, i))

        #logger.info(f"Registering handlers for {len(keys)} objects, total {len(reg_list)} and {len(xfer_desc)}")

        storage_reg_descs = self.agent.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        storage_xfer_descs = self.agent.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        storage_xfer_handler = self.agent.nixl_agent.prep_xfer_dlist(
            self.agent.agent_name, storage_xfer_descs, mem_type="OBJ"
        )   

        handle = self.agent.get_mem_to_storage_handle(mem_indices, storage_xfer_handler, storage_indices)
        self.agent.post_blocking(handle)
        self.agent.release_handle(handle)
        self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"mem_to_storage_parallel for {len(keys)} objects size {page_size*len(keys)} took {duration:.6f} seconds")

        for key in keys:
            with self.progress_lock:
                self.progress_set.discard(key.chunk_hash)

    async def mem_to_storage_seq(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
    ) -> None:
        """
        Write memory objects to storage with dynamically generated object keys.

        Unlike mem_to_storage in nixl_storage_backend, this doesn't use a pre-allocated
        pool but generates object keys on-the-fly based on CacheEngineKey.
        """
        mem_indices = [mem_obj.meta.address for mem_obj in mem_objs]
        page_size = self.memory_allocator.align_bytes

        for i in range(len(keys)):
            key = keys[i]
            mem_obj = mem_objs[i]

            # Generate object key based on CacheEngineKey
            object_key = self._format_object_key(key)

            # Create storage handler for this specific object
            storage_reg_descs, storage_xfer_descs, storage_xfer_handler = (
                self.agent.create_storage_handler(object_key, page_size)
            )

            # Create transfer handle
            handle = self.agent.get_mem_to_storage_handle(
                [mem_indices[i]], storage_xfer_handler, [0]
            )

            # Perform the transfer
            self.agent.post_blocking(handle)

            # Release resources
            self.agent.release_handle(handle)
            self.agent.release_storage_handler(storage_reg_descs, storage_xfer_handler)

            logger.debug(f"Write key {key.chunk_hash:x} to object {object_key} with shape {mem_obj.tensor.size()}")

        # Remove from progress set
        for key in keys:
            with self.progress_lock:
                self.progress_set.discard(key.chunk_hash)

    async def get_from_mem(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        """
        Retrieve memory objects from storage using the object_map.

        This method uses objects that were cached during contains() calls.
        """
        obj_list: list[Optional[MemoryObj]] = []

        for key in keys:
            entry = self.object_map.pop(key.chunk_hash)
            if entry is None:
                logger.debug(f"Key {key.chunk_hash:x} not found in object map")
                obj_list.append(None)
            else:
                # Return the cached object
                obj_list.append(entry.obj)
                logger.debug(f"Retrieved key {key.chunk_hash:x} from object map")

        return obj_list

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        with self.progress_lock:
            for key in keys:
                self.progress_set.add(key.chunk_hash)

        #asyncio.run_coroutine_threadsafe(
        #    self.mem_to_storage(keys, memory_objs), self.loop
        #)
        # test sync
        asyncio.run(self.mem_to_storage_parallel(keys, memory_objs))

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """
        if self.lookup_and_fetch:
            obj_list = asyncio.run(self.get_from_mem([key]))
            return obj_list[0]
        else:
            #obj = self.get_one_object_to_mem(key, False)
            obj_list = self.get_objects_to_mem([key], False)
            return obj_list[0]
        return None

    """
    TODO implement this
    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
    """

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

        obj_list = self.get_objects_to_mem(keys, False)
        return obj_list

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        if self.lookup_and_fetch:
            obj_list = await self.get_from_mem(keys)
            assert None not in obj_list
            return cast(list[MemoryObj], obj_list)
        else:
            # TODO zirui
            return []

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        :param force: Whether to force removal (not used in this implementation)
        """
        entry = self.object_map.pop(key.chunk_hash)
        if entry is None:
            return False

        # Free the memory object if it exists
        if entry.obj is not None:
            self.memory_allocator.free(entry.obj)

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
        if eviction:
            logger.warning("NixlObjectBackend does not support eviction for now")
        if busy_loop:
            logger.warning("NixlObjectBackend does not support busy loop for now")
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
        if eviction:
            logger.warning("NixlObjectBackend does not support eviction for now")
        if busy_loop:
            logger.warning("NixlObjectBackend does not support busy loop for now")

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
