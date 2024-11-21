"""
High-level design

MemoryObj:  -- Done
    raw_array
    metadata

PinBuffer: -- Done
    - Allocate(shape) -> MemoryOb 
    - Free(MemoryObj)

GPUConnector: -- Should be in lmcache_vllm
    # MemoryObj is flat + shape as metadata
    # Target buffer is paged memory or something else
    - to_gpu(MemoryObj, **kwargs)
    - to_host(dst_MemoryObj, **kwargs) 

TokenDB: -- Done
    - process_tokens(tokens, mask) -> List[CacheEngineKey]
    - insert(tokens, mask) -> List[CacheEngineKey]

LMCacheEngine:
    - __init__() # pin buffer, gpu connector, token db, backend manager
    - store_from_paged_memory()
    - retrieve_to_paged_memory()
    - retrieve_layers()
    - prefetch

LMCBackendInterface:
    - put()
    - get()
    - prefetch()

LMCBackendConnector:
    - put_task()
    - get_task()

MemoryObjs is allocated in CacheEngine.store(), and StorageManager.get().
The allocated memory objects should be managed by the StorageManager, and 
When the allocation fails, StorageManager should know this and determine
how to evict the memory objects.

Current design: 
- When the allocation fails in CacheEngine, it will directly stop the storing 
process.
- When the allocation fails in the StorageManager, it will try to do some
internal evictions and retry the allocation.
"""

from typing import Dict, Optional

import torch

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.experimental.gpu_connector import GPUConnectorInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    PinMemoryAllocator)
from lmcache.experimental.storage_backend.storage_manager import StorageManager
from lmcache.experimental.token_database import (ChunkedTokenDatabase,
                                                 TokenDatabase)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class LMCacheEngine:
    """The main class for the cache engine. 

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the 
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends. 
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: MemoryAllocatorInterface,
        token_database: TokenDatabase,
        gpu_connector: GPUConnectorInterface,
    ):
        self.config = config
        self.metadata = metadata
        self.memory_allocator = memory_allocator
        self.token_database = token_database
        self.gpu_connector = gpu_connector

        self.storage_manager = StorageManager(config, metadata,
                                              self.memory_allocator)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(self,
              tokens: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              **kwargs) -> None:
        """Store the tokens and mask into the cache engine.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables). 

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            if self.storage_manager.contains(key):
                # TODO: update the hit information to the storage manager
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            memory_obj = self.memory_allocator.allocate(kv_shape, kv_dtype)
            if memory_obj is None:
                logger.warning("Failed to allocate memory for the KV cache.")
                # TODO: let StorageManager know the failure here
                return

            # Copy the KV from GPU to memory obj
            # TODO: remember to free the memory obj in the storage backend
            # after store is finished
            self.gpu_connector.from_gpu(memory_obj, start, end, **kwargs)

            # TODO: Store the memory object into the storage backend
            self.storage_manager.put(key, memory_obj)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(self,
                 tokens: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables). 

        :return: the boolean mask indicating which tokens are retrieved. The 
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            if not self.storage_manager.contains(key):
                break

            ret_mask[start:end] = True

            # Get the memory object from the storage backend
            memory_obj = self.storage_manager.get(key)

            # Move the memory object to the GPU
            if memory_obj is not None:
                self.gpu_connector.to_gpu(memory_obj, start, end, **kwargs)
            else:
                logger.warning("Failed to retrieve the KV cache "
                               "when storage backend contains the key.")
                break
        return ret_mask

    def prefetch(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        """Launch the prefetching process in the storage manager to load the 
        KV to the local CPU memory
        """
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            self.storage_manager.prefetch(key)

    def lookup(
        self,
        tokens: torch.Tensor,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]

        :return: An int indicating how many prefix tokens are cached.
        """

        for start, end, key in self.token_database.process_tokens(tokens):
            if not self.storage_manager.contains(key):
                return start
        return end

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        pass


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}

    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> MemoryAllocatorInterface:
        # TODO: change this based on the config in the future
        return PinMemoryAllocator(8 * 1024 * 1024 * 1024)

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> TokenDatabase:
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
            cls,
            instance_id: str,
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata,
            gpu_connector:
        GPUConnectorInterface,  # gpu connectors is from outside
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        if instance_id not in cls._instances:
            memory_allocator = cls._Create_memory_allocator(config, metadata)
            token_database = cls._Create_token_database(config, metadata)
            engine = LMCacheEngine(config, metadata, memory_allocator,
                                   token_database, gpu_connector)
            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            return engine
        else:
            if (cls._cfgs[instance_id] != config
                    or cls._metadatas[instance_id] != metadata):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata.")
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID, 
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)
