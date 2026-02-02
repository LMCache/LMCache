# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List
import random

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import PinMemoryAllocator

pytest.importorskip(
    "lmcache.c_ops",
    reason="TODO: require non CUDA implementations for CUDA enhanced functions",
)

# First Party
import lmcache.c_ops as lmc_ops

# Local
from .utils import (
    check_mem_obj_equal,
    check_paged_kv_cache_equal,
    generate_kv_cache_paged,
    generate_kv_cache_paged_list_tensors,
    generate_mla_kv_cache_paged_list_tensors,
)


def _tuple_kv_to_blob(
    kv_tensors,
) -> torch.Tensor:
    k_temp = []
    v_temp = []
    for kv_layer in kv_tensors:
        k_temp.append(kv_layer[0])
        v_temp.append(kv_layer[1])
    k_tensor_blob = torch.stack(k_temp)
    v_tensor_blob = torch.stack(v_temp)

    # kv_tensors: [num_layer, 2, num_tok, num_kv_head, head_size]
    kv_tensors_flatten = torch.stack((k_tensor_blob, v_tensor_blob))
    kv_tensors_flatten = kv_tensors_flatten.permute([1, 0, 2, 3, 4])

    return kv_tensors_flatten


def _slice_kv_at(
    start_idx: int,
    kv_tensors: torch.Tensor,
    chunk_size: int,
) -> List[torch.Tensor]:
    return [
        x.contiguous()
        for x in list(
            torch.split(
                kv_tensors[:, :, start_idx:, ...],
                chunk_size,
                dim=2,
            )
        )
    ]


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
def test_extract_and_load_back(num_tokens):
    device = "cuda"

    num_blocks = 1000
    block_size = 16
    num_heads = 8
    head_size = 128
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged(num_blocks, device, block_size, dtype)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # Old extract
    kv_tuple_list = []
    memory_obj_old_list = []
    chunk_size = 256
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for layer_id in range(32):
        key_cache = kv_cache[layer_id][0].reshape(-1, num_heads, head_size)
        value_cache = kv_cache[layer_id][1].reshape(-1, num_heads, head_size)
        kv_tuple_list.append((key_cache[slot_mapping], value_cache[slot_mapping]))
    kv_blob = _tuple_kv_to_blob(kv_tuple_list)
    kv_chunked = _slice_kv_at(0, kv_blob, chunk_size)
    for chunk_id, chunk in enumerate(kv_chunked):
        mem_obj_shape = torch.Size([2, 32, chunk.shape[2], num_heads * head_size])

        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)
        chunk = chunk.contiguous()
        for layer_id in range(32):
            memory_obj_old.tensor[0, layer_id].copy_(
                chunk[layer_id, 0].reshape(-1, 1024)
            )
            memory_obj_old.tensor[1, layer_id].copy_(
                chunk[layer_id, 1].reshape(-1, 1024)
            )
        memory_obj_old_list.append(memory_obj_old)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Old extract time: ", elapsed_time_ms / 1000)

    # New extract (zero-copy kernels)
    memory_obj_new_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, 32, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        for layer_id in range(32):
            lmc_ops.load_and_reshape_flash(
                memory_obj_new.tensor,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        memory_obj_new_list.append(memory_obj_new)
    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)
    check_mem_obj_equal(
        memory_obj_old_list,
        memory_obj_new_list,
    )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged(num_blocks, device, block_size, dtype)

    # New load back (zero-copy kernels)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        for layer_id in range(32):
            lmc_ops.reshape_and_cache_back_flash(
                memory_obj_new.tensor,
                kv_cache_new[layer_id][0],
                kv_cache_new[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
def test_multi_layer_kernel(num_tokens):
    device = "cuda"

    num_blocks = 1000
    block_size = 16
    num_heads = 8
    head_size = 128
    chunk_size = 256
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # layer by layer extract
    memory_obj_old_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, 32, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)
        for layer_id in range(32):
            lmc_ops.load_and_reshape_flash(
                memory_obj_old.tensor,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        memory_obj_old_list.append(memory_obj_old)
    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Old extract time: ", elapsed_time_ms / 1000)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(
        32, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(32):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()

    memory_obj_new_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, 32, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0].device,
            page_buffer_size,
            True,
            False,
        )
        memory_obj_new_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)

    check_mem_obj_equal(
        memory_obj_old_list,
        memory_obj_new_list,
    )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )

    kv_cache_pointers_new = torch.empty(
        32, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(32):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0].device,
            page_buffer_size,
            False,
            False,
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("head_size", [576, 66])  # Use 68 for dsv32 (132x int8)
def test_multi_layer_kernel_use_mla(num_tokens, head_size):
    device = "cuda"

    num_blocks = 1000
    block_size = 64
    chunk_size = 256
    dtype = torch.bfloat16
    num_layers = 32
    kv_cache = generate_mla_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype, num_layers, head_size
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # layer by layer extract
    memory_obj_old_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size([1, num_layers, len(slot_mapping_temp), head_size])
        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)

        for layer_id in range(num_layers):
            for token_idx, slot_idx in enumerate(slot_mapping_temp):
                slot_idx = slot_idx.item()

                block_idx = slot_idx // block_size
                block_offset = slot_idx % block_size

                memory_obj_old.tensor[0][layer_id][token_idx] = kv_cache[layer_id][
                    block_idx
                ][block_offset]

        memory_obj_old_list.append(memory_obj_old)
    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Old extract time: ", elapsed_time_ms / 1000)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()

    memory_obj_new_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size([1, num_layers, len(slot_mapping_temp), head_size])

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0].device,
            0,
            True,
            True,
        )
        memory_obj_new_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)

    for left_mem_obj, right_mem_obj in zip(
        memory_obj_old_list, memory_obj_new_list, strict=False
    ):
        left_kv, right_kv = left_mem_obj.tensor[0], right_mem_obj.tensor[0]
        right_kv = right_kv.to(left_kv.device)

        assert len(left_kv.shape) == 3
        assert len(right_kv.shape) == 3

        assert (left_kv[:, :, :] == right_kv[:, :, :]).all()

    # Generate new paged kv_cache
    kv_cache_new = generate_mla_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype, num_layers, head_size
    )

    kv_cache_pointers_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0].device,
            0,
            False,
            True,
        )

    for left_kv, right_kv in zip(kv_cache, kv_cache_new, strict=False):
        assert len(left_kv.shape) == 3
        assert len(right_kv.shape) == 3

        left_reshaped = left_kv.reshape(
            left_kv.shape[0] * left_kv.shape[1], left_kv.shape[2]
        )
        right_reshaped = right_kv.reshape(
            right_kv.shape[0] * right_kv.shape[1], right_kv.shape[2]
        )

        assert (left_reshaped[slot_mapping, :] == right_reshaped[slot_mapping, :]).all()

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("token_major", [True, False])
def test_single_layer_kernel(num_tokens, token_major):
    device = "cuda"

    num_layers = 32
    num_blocks = 1000
    block_size = 16
    num_heads = 8
    head_size = 128
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    if token_major:
        tmp_gpu_buffer = torch.empty(
            (num_tokens, 2, hidden_dim_size), dtype=dtype, device=device
        )
    else:
        tmp_gpu_buffer = torch.empty(
            (2, num_tokens, hidden_dim_size), dtype=dtype, device=device
        )

    for layer_id in range(num_layers):
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache[layer_id],
            slot_mapping,
            True,
            token_major,
            True,
            False,
        )
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_new[layer_id],
            slot_mapping,
            False,
            token_major,
            True,
            False,
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
    )


def test_lmcache_memcpy_async():
    # Register 4 memory regions and try to launch copy
    chunk_size = 1 << 26
    num_chunks = 4
    dtype = torch.bfloat16
    elements_per_chunk = chunk_size // dtype.itemsize

    big_cpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device="cpu"
    )
    big_gpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device="cuda"
    )

    def check_gpu_and_cpu_equal(
        gpu_tensor: torch.Tensor,
        cpu_tensor: torch.Tensor,
    ):
        cpu_tensor_from_gpu = gpu_tensor.to("cpu")
        assert torch.allclose(cpu_tensor_from_gpu, cpu_tensor, atol=1e-5)

    rt = torch.cuda.cudart()

    # Register the cpu memory
    ptr = big_cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostRegister(ptr + i * chunk_size, chunk_size, 0)

    # Launch default cuda copy
    with pytest.raises(RuntimeError):
        big_gpu_tensor.copy_(big_cpu_tensor, non_blocking=True)
        torch.cuda.synchronize()

    # Launc default cuda copy for a small page
    big_gpu_tensor[: elements_per_chunk // 2].copy_(
        big_cpu_tensor[: elements_per_chunk // 2], non_blocking=True
    )
    torch.cuda.synchronize()

    check_gpu_and_cpu_equal(
        big_gpu_tensor[: elements_per_chunk // 2],
        big_cpu_tensor[: elements_per_chunk // 2],
    )

    # Positions to copy
    # - 0.25 chunk -- 0.75 chunk
    # - 0.75 chunk -- 1.25 chunk
    # - 1.75 chunk -- 3.25 chunk
    # - whole 4 chunks
    starts = [chunk_size // 4, (3 * chunk_size) // 4, (7 * chunk_size) // 4, 0]
    ends = [
        (3 * chunk_size) // 4,
        (5 * chunk_size) // 4,
        (13 * chunk_size) // 4,
        chunk_size * num_chunks,
    ]

    # H2D copy
    for start, end in zip(starts, ends, strict=False):
        # should not be equal before copy
        with pytest.raises(AssertionError):
            check_gpu_and_cpu_equal(
                big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
                big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            )

        lmc_ops.lmcache_memcpy_async(
            big_gpu_tensor.data_ptr() + start,
            big_cpu_tensor.data_ptr() + start,
            end - start,
            lmc_ops.TransferDirection.H2D,
            start,
            chunk_size,
        )
        torch.cuda.synchronize()

        check_gpu_and_cpu_equal(
            big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
        )

    # Reset the data in gpu
    big_gpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device="cuda"
    )
    # D2H copy
    for start, end in zip(starts, ends, strict=False):
        # copy from gpu to cpu
        with pytest.raises(AssertionError):
            check_gpu_and_cpu_equal(
                big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
                big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            )

        lmc_ops.lmcache_memcpy_async(
            big_cpu_tensor.data_ptr() + start,
            big_gpu_tensor.data_ptr() + start,
            end - start,
            lmc_ops.TransferDirection.D2H,
            start,
            chunk_size,
        )
        torch.cuda.synchronize()

        check_gpu_and_cpu_equal(
            big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
        )

    # Unregister the cpu memory
    ptr = big_cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostUnregister(ptr + i * chunk_size)
