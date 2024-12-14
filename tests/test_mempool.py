import pytest
import torch

from lmcache.config import LMCacheMemPoolMetadata
from lmcache.storage_backend.mem_pool import (LocalCPUBufferPool, LocalCPUPool,
                                              LocalGPUPool)


def dumb_metadata(
        kv_shape=(32, 2, 256, 8, 128), kv_dtype=torch.bfloat16,
        max_cache_size=10):
    return LMCacheMemPoolMetadata(kv_shape, kv_dtype, max_cache_size)


@pytest.mark.parametrize("mem_pool_type",
                         [LocalCPUBufferPool, LocalCPUPool, LocalGPUPool])
def test_alloc_full(mem_pool_type):
    kv_shape = (32, 2, 256, 8, 128)
    kv_dtype = torch.bfloat16
    metadata = dumb_metadata(kv_shape, kv_dtype, max_cache_size=1)
    mem_pool = mem_pool_type(metadata)
    max_chunk_num = mem_pool.max_chunk_num
    kv_obj_list = []

    # allocate max_num chunks
    for i in range(max_chunk_num):
        kv_tensor = torch.rand(kv_shape, dtype=kv_dtype)
        kv_obj_list.append(mem_pool.allocate(kv_tensor))

    kv_tensor = torch.rand(kv_shape, dtype=kv_dtype)
    if mem_pool_type in [LocalCPUPool, LocalGPUPool]:
        expected_err_msg = "Mempool allocation failed"

        with pytest.raises(Exception, match=expected_err_msg):
            mem_pool.allocate(kv_tensor)
    else:
        kv_obj_none = mem_pool.allocate(kv_tensor)
        assert kv_obj_none is None


@pytest.mark.parametrize("mem_pool_type",
                         [LocalCPUBufferPool, LocalCPUPool, LocalGPUPool])
def test_alloc_partial(mem_pool_type):
    kv_shape = (32, 2, 256, 8, 128)
    kv_dtype = torch.bfloat16
    metadata = dumb_metadata(kv_shape, kv_dtype)
    mem_pool = mem_pool_type(metadata)
    max_chunk_num = 1
    mem_pool.max_chunk_num = max_chunk_num

    partial_tok_num = 128
    kv_shape_partial = (32, 2, partial_tok_num, 8, 128)
    kv_tensor = torch.rand(kv_shape_partial, dtype=kv_dtype)
    kv_obj = mem_pool.allocate(kv_tensor)
    assert kv_obj.data.shape[2] == partial_tok_num
