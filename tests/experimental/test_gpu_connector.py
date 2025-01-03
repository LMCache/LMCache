import pytest
from utils import check_kv_cache_equal, generate_kv_cache

from lmcache.experimental.gpu_connector import VLLMNestedTupleGPUConnector
from lmcache.experimental.memory_management import (HostMemoryAllocator,
                                                    MemoryFormat)


def test_vllm_nested_gpu_connector():
    num_layers = 32
    num_heads = 8
    head_size = 128
    hidden_dim = num_heads * head_size
    connector = VLLMNestedTupleGPUConnector(hidden_dim, num_layers)
    allocator = HostMemoryAllocator(1024 * 1024 * 1024)

    assert connector.get_shape(10) == (2, num_layers, 10, hidden_dim)

    num_tokens = 512
    gpu_kv_src = generate_kv_cache(num_tokens, "vllm", "cuda")
    gpu_kv_dst = generate_kv_cache(num_tokens, "vllm", "cuda")

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_kv_cache_equal(gpu_kv_src, gpu_kv_dst, 512, "vllm")

    slices = 4
    num_slice_tokens = num_tokens // slices
    for slice_id in range(slices):
        print("Here", slice_id)
        st, ed = slice_id * num_slice_tokens, (slice_id + 1) * num_slice_tokens
        shape = connector.get_shape(num_slice_tokens)
        memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
        connector.from_gpu(memory_obj, st, ed, kvcaches=gpu_kv_src)
        assert memory_obj.metadata.fmt == MemoryFormat.KV_BLOB
        connector.to_gpu(memory_obj, st, ed, kvcaches=gpu_kv_dst)
        allocator.free(memory_obj)
        assert allocator.memcheck()

    # Check gpu_kv becomes the same
    check_kv_cache_equal(gpu_kv_src, gpu_kv_dst, 512, "vllm")
