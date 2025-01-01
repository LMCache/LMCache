import random
from typing import List

import torch
from utils import (check_mem_obj_equal, check_paged_kv_cache_equal,
                   generate_kv_cache_paged)

import lmcache.c_ops as lmc_ops
from lmcache.experimental.memory_management import PinMemoryAllocator


def _tuple_kv_to_blob(kv_tensors, ) -> torch.Tensor:

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
        x.contiguous() for x in list(
            torch.split(
                kv_tensors[:, :, start_idx:, ...],
                chunk_size,
                dim=2,
            ))
    ]


def test_extract_and_load_back():
    device = "cuda"
    num_tokens = 500

    num_blocks = 500
    block_size = 16
    num_heads = 8
    head_size = 128
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged(num_blocks, device, block_size, dtype)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 2 * 1024 * 1024 * 1024  # 2GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # Old extract
    kv_tuple_list = []
    memory_obj_old_list = []
    chunk_size = 256
    for layer_id in range(32):
        key_cache = kv_cache[layer_id][0].reshape(-1, num_heads, head_size)
        value_cache = kv_cache[layer_id][1].reshape(-1, num_heads, head_size)
        kv_tuple_list.append(
            (key_cache[slot_mapping], value_cache[slot_mapping]))
    kv_blob = _tuple_kv_to_blob(kv_tuple_list)
    kv_chunked = _slice_kv_at(0, kv_blob, chunk_size)
    for chunk_id, chunk in enumerate(kv_chunked):

        mem_obj_shape = [2, 32, chunk.shape[2], num_heads * head_size]

        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)
        chunk = chunk.contiguous()
        for layer_id in range(32):
            memory_obj_old.tensor[0,
                                  layer_id].copy_(chunk[layer_id,
                                                        0].reshape(-1, 1024))
            memory_obj_old.tensor[1,
                                  layer_id].copy_(chunk[layer_id,
                                                        1].reshape(-1, 1024))
        memory_obj_old_list.append(memory_obj_old)

    # New extract (zero-copy kernels)
    memory_obj_new_list = []
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = [2, 32, len(slot_mapping_temp), num_heads * head_size]

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        for layer_id in range(32):
            lmc_ops.load_and_reshape_flash(memory_obj_new.tensor,
                                           kv_cache[layer_id][0],
                                           kv_cache[layer_id][1],
                                           slot_mapping_temp, layer_id)
        memory_obj_new_list.append(memory_obj_new)

    check_mem_obj_equal(
        memory_obj_old_list,
        memory_obj_new_list,
        num_tokens,
    )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged(num_blocks, device, block_size,
                                           dtype)

    # New load back (zero-copy kernels)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        for layer_id in range(32):
            lmc_ops.reshape_and_cache_back_flash(memory_obj_new.tensor,
                                                 kv_cache_new[layer_id][0],
                                                 kv_cache_new[layer_id][1],
                                                 slot_mapping_temp, layer_id)
    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        num_tokens,
        slot_mapping,
    )
