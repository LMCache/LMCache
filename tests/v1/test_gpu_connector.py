# Standard
import random

# Third Party
from utils import (
    check_kv_cache_equal,
    check_paged_kv_cache_equal,
    check_paged_kv_cache_equal_with_mla,
    generate_kv_cache,
    generate_kv_cache_paged_list_tensors,
)
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import (
    VLLMBufferLayerwiseGPUConnector,
    VLLMNestedTupleGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    HostMemoryAllocator,
    MemoryFormat,
    PinMemoryAllocator,
)


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
        assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector.to_gpu(memory_obj, st, ed, kvcaches=gpu_kv_dst)
        allocator.free(memory_obj)
        assert allocator.memcheck()

    # Check gpu_kv becomes the same
    check_kv_cache_equal(gpu_kv_src, gpu_kv_dst, 512, "vllm")


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
def test_vllm_paged_connector_v2_with_gpu_and_mla(use_gpu, use_mla):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 1 if use_mla else 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks, device=device, block_size=block_size, use_mla=use_mla
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks, device=device, block_size=block_size, use_mla=use_mla
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
            check_paged_kv_cache_equal(
                gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
            )

    connector = VLLMPagedMemGPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=gpu_kv_src[0].dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = VLLMPagedMemGPUConnectorV2(
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
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
        )


@pytest.mark.parametrize("use_gpu", [True])
def test_layerwise_vllm_paged_connector_with_gpu(use_gpu):
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

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
        )

    connector = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    )

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        sync=True,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
        sync=True,
    )
    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
    )


@pytest.mark.parametrize("use_gpu", [True])
def test_batched_layerwise_vllm_paged_connector_with_gpu(use_gpu):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens_1 = 800
    num_tokens_2 = 500
    num_tokens_total = num_tokens_1 + num_tokens_2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping_total = random.sample(
        range(0, num_blocks * block_size), num_tokens_total
    )
    slot_mapping_total = torch.tensor(
        slot_mapping_total, device=device, dtype=torch.int64
    )

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_kv_cache_equal(gpu_kv_src, gpu_kv_dst, num_tokens_total, "vllm")

    connector = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    )

    # from gpu to cpu
    starts_1 = []
    ends_1 = []
    memory_objs_1 = []

    for start in range(0, num_tokens_1, chunk_size):
        end = min(start + chunk_size, num_tokens_1)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts_1.append(start)
        ends_1.append(end)
        memory_objs_1.append(memory_objs_multi_layer)

    memory_objs_1 = [list(row) for row in zip(*memory_objs_1, strict=False)]

    starts_2 = []
    ends_2 = []
    memory_objs_2 = []
    for start in range(num_tokens_1, num_tokens_total, chunk_size):
        end = min(start + chunk_size, num_tokens_total)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts_2.append(start)
        ends_2.append(end)
        memory_objs_2.append(memory_objs_multi_layer)

    memory_objs_2 = [list(row) for row in zip(*memory_objs_2, strict=False)]

    mem_obj_generator_1 = connector.batched_from_gpu(
        memory_objs_1,
        starts_1,
        ends_1,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    mem_obj_generator_1 = connector.batched_from_gpu(
        memory_objs_1,
        starts_1,
        ends_1,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    mem_obj_generator_2 = connector.batched_from_gpu(
        memory_objs_2,
        starts_2,
        ends_2,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=False,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator_1)
        next(mem_obj_generator_2)

    # from cpu to gpu
    mem_obj_consumer_1 = connector.batched_to_gpu(
        starts_1,
        ends_1,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping_total,
        sync=False,
    )
    mem_obj_consumer_2 = connector.batched_to_gpu(
        starts_2,
        ends_2,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    next(mem_obj_consumer_1)
    next(mem_obj_consumer_2)
    for layer_id in range(num_layers):
        mem_obj_consumer_1.send(memory_objs_1[layer_id])
        mem_obj_consumer_2.send(memory_objs_2[layer_id])
    next(mem_obj_consumer_1)
    next(mem_obj_consumer_2)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs_1:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    for mem_obj_multi_layer in memory_objs_2:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src,
        gpu_kv_dst,
        num_tokens_total,
        slot_mapping_total,
        num_heads,
        head_size,
    )


@pytest.mark.skip(reason="This test is skipped due to vllm dependency")
@pytest.mark.parametrize("use_gpu", [True])
def test_layerwise_vllm_buffer_connector_with_gpu(use_gpu):
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

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
        )

    connector = VLLMBufferLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        dtype=dtype,
        device=device,
    )

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_2TD
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
    )
    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, num_tokens, slot_mapping, num_heads, head_size
    )


def test_vllm_paged_connector_v2_to_gpu_bench(benchmark):
    """
    VLLMPagedMemGPUConnectorV2.to_gpu() micro-benchmark.

    This test is to measure the performance of
    VLLMPagedMemGPUConnectorV2.to_gpu() when both KV caches and
    memobject are on GPU.

    """
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    chunk_size = 256

    allocator = GPUMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)

    slot_mapping = random.sample(range(0, num_blocks * block_size), chunk_size)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    connector = VLLMPagedMemGPUConnectorV2(hidden_dim, num_layers)
    shape = connector.get_shape(chunk_size)
    memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
    connector.from_gpu(
        memory_obj,
        0,
        chunk_size,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        offset=0,
    )
    assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
    benchmark.pedantic(
        connector.to_gpu,
        args=(memory_obj, 0, chunk_size),
        kwargs={
            "kvcaches": gpu_kv_dst,
            "slot_mapping": slot_mapping,
            "offset": 0,
        },
        rounds=100,
        iterations=1000,
        warmup_rounds=10,
    )
    allocator.free(memory_obj)
    assert allocator.memcheck()
