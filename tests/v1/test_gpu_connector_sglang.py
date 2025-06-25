# Standard
import random

# Third Party
from utils import (
    check_sglang_paged_kv_cache_equal,
    check_paged_kv_cache_equal_with_mla,
    generate_sglang_kv_cache_paged_list_tensors,
)
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector_sglang import SGLangPagedMemGPUConnector
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    HostMemoryAllocator,
    MemoryFormat,
    PinMemoryAllocator,
)


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
def test_sglang_paged_connector_v2_with_gpu_and_mla(use_gpu, use_mla):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 1 if use_mla else 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = num_blocks * block_size // 2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers, num_blocks=num_blocks, block_size=block_size,
        num_heads=num_heads, head_size=head_size, use_mla=use_mla,
        device=device,
    )
    gpu_kv_dst = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers, num_blocks=num_blocks, block_size=block_size,
        num_heads=num_heads, head_size=head_size, use_mla=use_mla,
        device=device,
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, head_size
            )
        else:
            check_sglang_paged_kv_cache_equal(
                gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
            )

    connector = SGLangPagedMemGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=gpu_kv_src[0].dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = SGLangPagedMemGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=gpu_kv_src[0].dtype,
        device=device,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(end - start)
        memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_src,
            slot_mapping=slot_mapping,
            offset=0,
        )
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_dst,
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    if use_mla:
        check_paged_kv_cache_equal_with_mla(
            gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, head_size
        )
    else:
        check_sglang_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
        )
