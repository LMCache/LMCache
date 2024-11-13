from typing import List, Optional

import torch

from lmcache.config import LMCacheMemPoolMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.mem_pool.base_pool import BasePool, KVObj

logger = init_logger(__name__)


class LocalPool(BasePool):

    def __init__(self, metadata: LMCacheMemPoolMetadata):
        self.chunk_size = metadata.kv_shape[2]
        self.max_chunk_num = 0
        self.mem_pool: List[torch.Tensor] = []

        self.free_pool = [i for i in range(self.max_chunk_num)]

    def allocate(self, kv_chunk: torch.Tensor) -> Optional[KVObj]:
        """
        Allocate a buffer memory pointer from the memory pool.
        
        Input:
            kv_chunk: the kv tensor to be stored
        
        Returns:
            KVObj with a memory pointer (torch tensor view).
            None if memory is full.
        
        Note:
            This does not perform the actual memory movement.
        """
        num_tok = kv_chunk.shape[2]
        assert num_tok <= self.chunk_size
        if not self.free_pool:
            logger.error("No free memory chunks. Evictor might be failing!")
            raise Exception("No free chunks in cpu memory. \
                Shouldn't happen in local cpu-only backend.")
        chunk_idx = self.free_pool.pop()
        return KVObj(chunk_idx, self.mem_pool[chunk_idx][:, :, 0:num_tok])

    def free(self, kv_obj: KVObj):
        """
        Free the corresponding memory chunk
        
        Input:
            the KVObj to be freed
        """
        self.free_pool.append(kv_obj.chunk_idx)


class LocalCPUPool(LocalPool):

    def __init__(self, metadata: LMCacheMemPoolMetadata, max_chunk_num=200):
        self.chunk_size = metadata.kv_shape[2]
        # TODO(Jiayi): the `max_chunk_num` should be computed
        # from `config.max_cache_size`
        self.max_chunk_num = max_chunk_num
        use_pinned_memory = True
        kv_dtype = metadata.kv_dtype

        logger.info(f"Initializing cpu mem, is_pinned: {use_pinned_memory}")
        with torch.inference_mode():
            self.mem_pool = [
                torch.empty(metadata.kv_shape,
                            dtype=kv_dtype,
                            device='cpu',
                            pin_memory=use_pinned_memory)
                for i in range(self.max_chunk_num)
            ]

        self.free_pool = [i for i in range(self.max_chunk_num)]


class LocalCPUBufferPool(LocalCPUPool):

    def allocate(self, kv_chunk: torch.Tensor) -> Optional[KVObj]:
        num_tok = kv_chunk.shape[2]
        assert num_tok <= self.chunk_size
        if not self.free_pool:
            logger.info("No free memory chunks. Waiting...")
            return None
        chunk_idx = self.free_pool.pop()
        return KVObj(chunk_idx, self.mem_pool[chunk_idx][:, :, 0:num_tok])


class LocalGPUPool(LocalPool):
    """ only for unit testing, might not be useful in production """
    """ incur double copy, but we can use this as the only gpu buffer"""

    def __init__(self, metadata: LMCacheMemPoolMetadata, max_chunk_num=100):
        self.chunk_size = metadata.kv_shape[2]
        # TODO(Jiayi): the `max_chunk_num` should be computed
        # from `config.max_cache_size`
        self.max_chunk_num = max_chunk_num
        kv_dtype = metadata.kv_dtype

        logger.info("Initializing gpu mem")
        with torch.inference_mode():
            self.mem_pool = [
                torch.empty(metadata.kv_shape, dtype=kv_dtype, device='cuda')
                for i in range(self.max_chunk_num)
            ]

        self.free_pool = [i for i in range(self.max_chunk_num)]
