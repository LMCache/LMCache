# SPDX-License-Identifier: Apache-2.0
# Standard
from math import prod
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheMemPoolMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.mem_pool.base_pool import BasePool, KVObj

logger = init_logger(__name__)


class LocalPool(BasePool):
    def __init__(self, metadata: LMCacheMemPoolMetadata):
        self.chunk_size = metadata.kv_shape[2]
        self.max_chunk_num = 200
        self.size_per_chunk = prod(metadata.kv_shape) * metadata.kv_dtype.itemsize
        self.mem_pool: List[torch.Tensor] = []

        self.free_pool = [i for i in range(self.max_chunk_num)]

    def init_max_chunk_num(self, metadata: LMCacheMemPoolMetadata) -> int:
        """
        Initialize the maximum number of chunks in the memory pool.
        """
        max_chunk_num = (
            int(metadata.max_local_cache_size * 1024**3) // self.size_per_chunk + 1
        )
        return int(max_chunk_num)

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
            logger.warning(
                "No free memory chunks. Shouldn't happen! Evictor might be failing!"
            )
            raise Exception("Mempool allocation failed")
        chunk_idx = self.free_pool.pop()
        return KVObj(
            chunk_idx,
            self.size_per_chunk,
            self.mem_pool[chunk_idx][:, :, 0:num_tok],
        )

    def free(self, kv_obj: KVObj) -> None:
        """
        Free the corresponding memory chunk

        Input:
            the KVObj to be freed
        """
        self.free_pool.append(kv_obj.chunk_idx)


class LocalCPUPool(LocalPool):
    def __init__(self, metadata: LMCacheMemPoolMetadata):
        self.chunk_size = metadata.kv_shape[2]
        self.size_per_chunk = prod(metadata.kv_shape) * metadata.kv_dtype.itemsize
        self.max_chunk_num = self.init_max_chunk_num(metadata)
        use_pinned_memory = True
        kv_dtype = metadata.kv_dtype

        logger.info(
            f"Initializing cpu mem, is_pinned: {use_pinned_memory}, "
            f"max_local_cache_size: {metadata.max_local_cache_size} GB, "
            f"max_chunk_num: {self.max_chunk_num}."
        )
        with torch.inference_mode():
            self.mem_pool = [
                torch.empty(
                    metadata.kv_shape,
                    dtype=kv_dtype,
                    device="cpu",
                    pin_memory=use_pinned_memory,
                )
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
        return KVObj(
            chunk_idx,
            self.size_per_chunk,
            self.mem_pool[chunk_idx][:, :, 0:num_tok],
        )


class LocalGPUPool(LocalPool):
    """only for unit testing, might not be useful in production"""

    """ incur double copy, but we can use this as the only gpu buffer"""

    def __init__(self, metadata: LMCacheMemPoolMetadata):
        self.chunk_size = metadata.kv_shape[2]
        self.size_per_chunk = prod(metadata.kv_shape) * metadata.kv_dtype.itemsize
        self.max_chunk_num = self.init_max_chunk_num(metadata)
        kv_dtype = metadata.kv_dtype

        logger.info("Initializing gpu mem")
        with torch.inference_mode():
            self.mem_pool = [
                torch.empty(metadata.kv_shape, dtype=kv_dtype, device="cuda")
                for i in range(self.max_chunk_num)
            ]

        self.free_pool = [i for i in range(self.max_chunk_num)]
