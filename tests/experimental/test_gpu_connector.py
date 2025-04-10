import random

import pytest
import torch
from utils import (check_kv_cache_equal, check_paged_kv_cache_equal,
                   generate_kv_cache, generate_kv_cache_paged_list_tensors)

from lmcache.experimental.gpu_connector import (VLLMNestedTupleGPUConnector,
                                                VLLMPagedMemGPUConnectorV2)
from lmcache.experimental.memory_management import (HostMemoryAllocator,
                                                    MemoryFormat,
                                                    PinMemoryAllocator)


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


def test_vllm_paged_connector_v2():
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device,
                                                      block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device,
                                                      block_size)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_kv_cache_equal(gpu_kv_src, gpu_kv_dst, num_tokens, "vllm")

    connector = VLLMPagedMemGPUConnectorV2(hidden_dim, num_layers)
    connector2 = VLLMPagedMemGPUConnectorV2(hidden_dim, num_layers)
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(end - start)
        memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
        connector.from_gpu(memory_obj,
                           start,
                           end,
                           kvcaches=gpu_kv_src,
                           slot_mapping=slot_mapping,
                           offset=0)
        assert memory_obj.metadata.fmt == MemoryFormat.KV_BLOB
        connector2.to_gpu(memory_obj,
                          start,
                          end,
                          kvcaches=gpu_kv_dst,
                          slot_mapping=slot_mapping,
                          offset=0)
        allocator.free(memory_obj)
        assert allocator.memcheck()

    check_paged_kv_cache_equal(gpu_kv_src, gpu_kv_dst, num_tokens,
                               slot_mapping, num_heads, head_size)


@pytest.mark.parametrize("use_gpu", [True, False])
def test_vllm_paged_connector_v2_with_gpu(use_gpu):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device,
                                                      block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device,
                                                      block_size)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_kv_cache_equal(gpu_kv_src, gpu_kv_dst, num_tokens, "vllm")

    connector = VLLMPagedMemGPUConnectorV2(hidden_dim,
                                           num_layers,
                                           use_gpu=use_gpu,
                                           chunk_size=chunk_size,
                                           dtype=gpu_kv_src[0].dtype,
                                           device=device)
    connector2 = VLLMPagedMemGPUConnectorV2(hidden_dim,
                                            num_layers,
                                            use_gpu=use_gpu,
                                            chunk_size=chunk_size,
                                            dtype=gpu_kv_src[0].dtype,
                                            device=device)
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(end - start)
        memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
        connector.from_gpu(memory_obj,
                           start,
                           end,
                           kvcaches=gpu_kv_src,
                           slot_mapping=slot_mapping,
                           offset=0)
        assert memory_obj.metadata.fmt == MemoryFormat.KV_BLOB
        connector2.to_gpu(memory_obj,
                          start,
                          end,
                          kvcaches=gpu_kv_dst,
                          slot_mapping=slot_mapping,
                          offset=0)
        allocator.free(memory_obj)
        assert allocator.memcheck()

    check_paged_kv_cache_equal(gpu_kv_src, gpu_kv_dst, num_tokens,
                               slot_mapping, num_heads, head_size)
