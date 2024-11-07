from typing import Optional
import torch

from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.mem_pool.base_pool import BasePool, KVObj
from lmcache.logging import init_logger

logger = init_logger(__name__)

class LocalCPUPool(BasePool):
    def __init__(self, config: LMCacheEngineConfig):  
        self.chunk_size = config.chunk_size
        # TODO(Jiayi): the `max_chunk_num` should be computed
        # from `config.max_cache_size`   
        max_chunk_num = 200
        use_pinned_memory = True
        max_num_tok = max_chunk_num * self.chunk_size
        kv_dtype = torch.bfloat16
        
        #mem_shape = (32, 2, max_num_tok, 8, 128)
        mem_shape = (32, 2, self.chunk_size, 8, 128)
        logger.info(f"Initializing cpu mem buffer, is_pinned: {use_pinned_memory}")
        with torch.inference_mode():
            #self.mem_pool = torch.empty(mem_shape, dtype=kv_dtype,
            #    device='cpu', pin_memory=use_pinned_memory)
            self.mem_pool = [torch.empty(mem_shape, dtype=kv_dtype,
                device='cpu', pin_memory=use_pinned_memory) for i in range(max_chunk_num)]
        
        self.free_pool = [i for i in range(max_chunk_num)]
            
    def allocate(self, kv_chunk: torch.Tensor) -> Optional[KVObj]:
        """
        Allocate a buffer memory pointer from the memory pool.
        
        Input:
            kv_chunk: the kv tensor to be stored
        
        Returns:
            A memory pointer (torch tensor view).
            None if memory is full.
        
        Note:
            This does not perform the actual memory movement.
        """
        num_tok = kv_chunk.shape[2]
        assert num_tok <= self.chunk_size
        if not self.free_pool:
            logger.warning("No free memory chunks. Evictor might be failing!")
        chunk_idx = self.free_pool.pop()
        return KVObj(chunk_idx, self.mem_pool[chunk_idx][:, :, 0:num_tok])
        
        
    def free(self, kv_obj: KVObj):
        """
        Free the corresponding memory chunk
        
        Input:
            the KVObj to be freed
        """
        self.free_pool.append(kv_obj.chunk_idx)