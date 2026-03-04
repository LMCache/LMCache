# SPDX-License-Identifier: Apache-2.0
#
# This file contains Python non-CUDA fallback implementations for
# CUDA-specific operations.
#
# Standard
from enum import Enum, IntEnum
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional
import ctypes
import subprocess

# Third Party
import numpy as np
import torch

# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: dict[int, torch.Tensor] = {}
_shm_registry: dict[int, shared_memory.SharedMemory] = {}
_buf_registry: dict[int, ctypes.Array] = {}

# Cached copy library for lmcache_memcpy_async (lazy-initialized)
_copy_lib: Optional[ctypes.CDLL] = None


def _get_copy_lib() -> Optional[ctypes.CDLL]:
    """Lazily load and cache the CUDA runtime library, or None for CPU fallback."""
    global _copy_lib
    if _copy_lib is None:
        try:
            _copy_lib = ctypes.CDLL("libcudart.so")
        except OSError:
            pass
    return _copy_lib


class TransferDirection(Enum):
    H2D = 0
    D2H = 1


class GPUKVFormat(IntEnum):
    # used by: vLLM CROSS_LAYER mode
    NB_NL_TWO_BS_NH_HS = 0

    # used by: vLLM non-MLA flash attention
    NL_X_TWO_NB_BS_NH_HS = 1

    # used by: vLLM non-MLA flash infer
    NL_X_NB_TWO_BS_NH_HS = 2

    # used by: vLLM MLA
    NL_X_NB_BS_HS = 3

    # used by: SGLang MHA (flash attention and flash infer)
    TWO_X_NL_X_NBBS_NH_HS = 4

    # used by: SGLang MLA
    NL_X_NBBS_ONE_HS = 5


def _alloc_cpu_ptr(size: int) -> int:
    """Allocate a zeroed CPU tensor and register it; return its data pointer."""
    # Create a zeroed 1D uint8 CPU tensor (uint8 == 1 byte)
    tensor = torch.zeros(size, dtype=torch.uint8, pin_memory=False)

    # Get a pointer to the start of the tensor object as this is what is
    # returned by the CUDA equivalent function
    ptr = tensor.data_ptr()

    # Store the tensor so it can be accessed outside this function scope
    _tensor_registry[ptr] = tensor

    return ptr


def alloc_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory with NUMA awareness.
    Note: NUMA and pinned memory are not supported on non-CUDA."""
    return _alloc_cpu_ptr(size)


def free_pinned_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory and returning pointer
    to it. Note: Pinned memory is not supported on non-CUDA."""
    return _alloc_cpu_ptr(size)


def free_pinned_ptr(ptr: int) -> None:
    """Non-CUDA equivalent of freeing a previously allocated pinned pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_shm_pinned_ptr(size: int, shm_name: str = "") -> int:
    """Non-CUDA equivalent of allocating shared memory pinned pointer.
    Uses multiprocessing.shared_memory for cross-platform POSIX shm."""

    # Strip leading '/' for SharedMemory name
    name = shm_name.lstrip("/") if shm_name else None

    # Clean up stale shm segment if it exists
    if name:
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
        except OSError:
            pass

    shm = shared_memory.SharedMemory(name=name, create=True, size=size)

    array_type = ctypes.c_uint8 * size
    buf = array_type.from_buffer(shm.buf)
    ptr = ctypes.addressof(buf)

    # Store references to keep them alive
    tensor = torch.frombuffer(buf, dtype=torch.uint8)
    _tensor_registry[ptr] = tensor
    _buf_registry[ptr] = buf
    _shm_registry[ptr] = shm
    return ptr


def free_shm_pinned_ptr(ptr: int, size: int = 0, shm_name: str = "") -> None:
    """Non-CUDA equivalent of freeing a shared memory
    pinned pointer."""

    # Release in order: tensor -> ctypes buf -> shm
    _tensor_registry.pop(ptr, None)
    _buf_registry.pop(ptr, None)
    shm = _shm_registry.pop(ptr, None)
    if shm is not None:
        shm.close()
        shm.unlink()


def alloc_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating numa memory and returning pointer
    to it. Note: Numa memory is not supported on non-CUDA."""
    return alloc_pinned_numa_ptr(size, numa_id)


def free_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""
    return free_pinned_numa_ptr(ptr, size)


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
    block_size: int,
):
    """
    Python fallback for multi_layer_kv_transfer (csrc/mem_kernels.cu L524-L548).

    Mirrors the CUDA kernel `load_and_reshape_multi_layer_kernel` which uses
    `page_buffer_offset<format>()` and `key_value_offset()` for addressing.

    key_value layout:
        - Standard: [2, num_layers, num_tokens, hidden_size]
        - MLA:      [1, num_layers, num_tokens, hidden_size]

    Each paged buffer (one per layer) layout depends on gpu_kv_format:
        - NB_NL_TWO_BS_NH_HS / NL_X_TWO_NB_BS_NH_HS:
              [2, page_buffer_size, hidden_size]
        - NL_X_NB_TWO_BS_NH_HS (flash infer):
              [num_blocks, 2, block_size, hidden_size]
        - NL_X_NB_BS_HS / NL_X_NBBS_ONE_HS (MLA):
              [page_buffer_size, hidden_size]

    direction:
        H2D  = LMCache  -> PagedBuffer
        D2H  = PagedBuffer -> LMCache
    """
    is_mla = gpu_kv_format in (
        GPUKVFormat.NL_X_NB_BS_HS,
        GPUKVFormat.NL_X_NBBS_ONE_HS,
    )

    num_layers = key_value.size(1)
    num_tokens = slot_mapping.size(0)
    hidden_size = key_value.size(3)
    element_size = key_value.element_size()
    token_bytes = hidden_size * element_size

    # k_or_v_size: MLA has 1 part (KV fused), standard has 2 (K and V)
    k_or_v_size = 1 if is_mla else 2

    # Base pointer of key_value tensor (contiguous [k_or_v_size, L, T, D])
    kv_base_ptr = key_value.data_ptr()

    # Raw pointers for each layer's paged buffer
    ptr_list = key_value_ptrs.cpu().numpy().tolist()
    slots = slot_mapping.cpu().numpy().tolist()

    for token_id in range(num_tokens):
        slot_idx = slots[token_id]
        if slot_idx < 0:
            continue

        for layer_id in range(num_layers):
            paged_buffer_ptr = int(ptr_list[layer_id])

            for k_or_v in range(k_or_v_size):
                # ── LMCache side offset ──
                # Mirrors key_value_offset() in mem_kernels.cu L229-L236:
                #   k_or_v * num_layers * num_tokens * scalars_per_token
                #   + layer_id * num_tokens * scalars_per_token
                #   + token_id * scalars_per_token
                lmc_offset = (
                    k_or_v * (num_layers * num_tokens * hidden_size)
                    + layer_id * (num_tokens * hidden_size)
                    + token_id * hidden_size
                ) * element_size

                # ── Paged buffer side offset ──
                # Mirrors page_buffer_offset<format>() in mem_kernels.cu L194-L222
                if gpu_kv_format in (
                    GPUKVFormat.NB_NL_TWO_BS_NH_HS,
                    GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
                ):
                    # Layout: [2, page_buffer_size, hidden_size]
                    paged_offset = (
                        k_or_v * page_buffer_size * hidden_size + slot_idx * hidden_size
                    ) * element_size

                elif gpu_kv_format == GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
                    # Layout: [num_blocks, 2, block_size, hidden_size]
                    blk_idx = slot_idx // block_size
                    blk_off = slot_idx % block_size
                    paged_offset = (
                        blk_idx * 2 * block_size * hidden_size
                        + k_or_v * block_size * hidden_size
                        + blk_off * hidden_size
                    ) * element_size

                elif gpu_kv_format in (
                    GPUKVFormat.NL_X_NB_BS_HS,
                    GPUKVFormat.NL_X_NBBS_ONE_HS,
                ):
                    # MLA: [page_buffer_size, hidden_size], no K/V split
                    paged_offset = slot_idx * hidden_size * element_size

                else:
                    raise ValueError(f"Unsupported GPUKVFormat: {gpu_kv_format}")

                # ── Absolute addresses ──
                lmc_addr = kv_base_ptr + lmc_offset
                paged_addr = paged_buffer_ptr + paged_offset

                # ── Copy direction ──
                if direction == TransferDirection.D2H:
                    # PagedBuffer -> LMCache
                    dst, src = lmc_addr, paged_addr
                else:
                    # LMCache -> PagedBuffer
                    dst, src = paged_addr, lmc_addr

                ctypes.memmove(
                    ctypes.c_void_p(dst),
                    ctypes.c_void_p(src),
                    token_bytes,
                )


def multi_layer_kv_transfer_unilateral(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
):
    """
    Python fallback for multi_layer_kv_transfer_unilateral (mem_kernels.cu L576-L628).

    Handles SGLang MHA format where K and V paged buffers are stored separately:
        ptrs = [K_layer0, K_layer1, ..., V_layer0, V_layer1, ...]
        each buffer shape: [page_buffer_size, hidden_size]

    For MLA, delegates to multi_layer_kv_transfer (same as C++ implementation).

    key_value layout:
        - Standard: [2, num_layers, num_tokens, hidden_size]
        - MLA:      [1, num_layers, num_tokens, hidden_size]

    direction:
        H2D = LMCache  -> PagedBuffer
        D2H = PagedBuffer -> LMCache
    """
    is_mla = gpu_kv_format in (
        GPUKVFormat.NL_X_NB_BS_HS,
        GPUKVFormat.NL_X_NBBS_ONE_HS,
    )

    # MLA case collapses back to multi_layer_kv_transfer
    # (vLLM and SGLang indexing are compatible)
    if is_mla:
        return multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            paged_memory_device,
            page_buffer_size,
            direction,
            gpu_kv_format,
            0,  # block_size unused for MLA formats
        )

    # ── Non-MLA path: unilateral (separate K/V buffers per layer) ──
    num_layers = key_value.size(1)
    num_tokens = slot_mapping.size(0)
    hidden_size = key_value.size(3)
    element_size = key_value.element_size()
    token_bytes = hidden_size * element_size

    kv_base_ptr = key_value.data_ptr()

    # ptrs layout: [K_layer0, K_layer1, ..., V_layer0, V_layer1, ...]
    ptr_list = key_value_ptrs.cpu().numpy().tolist()
    slots = slot_mapping.cpu().numpy().tolist()

    for token_id in range(num_tokens):
        slot_idx = slots[token_id]
        if slot_idx < 0:
            continue

        for layer_id in range(num_layers):
            for k_or_v in range(2):
                # ── LMCache side offset ──
                # key_value layout: [2, num_layers, num_tokens, hidden_size]
                # Mirrors key_value_offset() in mem_kernels.cu L229-L236
                lmc_offset = (
                    k_or_v * (num_layers * num_tokens * hidden_size)
                    + layer_id * (num_tokens * hidden_size)
                    + token_id * hidden_size
                ) * element_size

                # ── Paged buffer side offset ──
                # ptrs[layer_id]            = K buffer for this layer
                # ptrs[layer_id + num_layers] = V buffer for this layer
                # Each buffer: [page_buffer_size, hidden_size]
                # Mirrors page_buffer_offset_unilateral() in mem_kernels.cu L224-L227:
                #   slot_idx * scalars_per_token + scalar_offset
                if k_or_v == 0:
                    buffer_ptr = int(ptr_list[layer_id])
                else:
                    buffer_ptr = int(ptr_list[layer_id + num_layers])

                paged_offset = slot_idx * hidden_size * element_size

                # ── Absolute addresses ──
                lmc_addr = kv_base_ptr + lmc_offset
                paged_addr = buffer_ptr + paged_offset

                # ── Copy direction ──
                if direction == TransferDirection.D2H:
                    # PagedBuffer -> LMCache
                    dst, src = lmc_addr, paged_addr
                else:
                    # LMCache -> PagedBuffer
                    dst, src = paged_addr, lmc_addr

                ctypes.memmove(
                    ctypes.c_void_p(dst),
                    ctypes.c_void_p(src),
                    token_bytes,
                )


def single_layer_kv_transfer(
    lmc_key_value_cache: torch.Tensor,
    vllm_key_value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
    token_major: bool,
):
    """
    Python fallback for single_layer_kv_transfer (mem_kernels.cu L630-L749).

    Transfers KV data between LMCache buffer
    and a single vLLM paged KV cache layer.

    lmc_key_value_cache layout:
        - MLA:                    [num_tokens, aligned_head_size]
        - token_major=True:       [num_tokens, 2, num_heads * head_size]
        - token_major=False:      [2, num_tokens, num_heads * head_size]

    vllm_key_value_cache layout:
        - NL_X_TWO_NB_BS_NH_HS (flash attn):
            [2, num_blocks, block_size, num_heads, head_size]
        - NL_X_NB_TWO_BS_NH_HS (flash infer):
            [num_blocks, 2, block_size, num_heads, head_size]
        - NL_X_NB_BS_HS (vLLM MLA):
            [num_blocks, block_size, head_size]

    direction:
        H2D = LMCache  -> vLLM GPU
        D2H = vLLM GPU -> LMCache
    """
    is_mla = gpu_kv_format in (
        GPUKVFormat.NL_X_NB_BS_HS,
        GPUKVFormat.NL_X_NBBS_ONE_HS,
    )

    num_tokens = slot_mapping.size(0)
    slots = slot_mapping.cpu().tolist()

    if is_mla:
        # ── MLA format ──
        # vllm: [num_blocks, block_size, head_size]
        # lmc:  [num_tokens, aligned_head_size]
        block_size = vllm_key_value_cache.size(1)

        for token_idx in range(num_tokens):
            slot_idx = slots[token_idx]
            if slot_idx < 0:
                continue

            block_idx = slot_idx // block_size
            block_offset = slot_idx % block_size

            if direction == TransferDirection.D2H:
                # vLLM -> LMCache
                lmc_key_value_cache[token_idx] = vllm_key_value_cache[
                    block_idx, block_offset
                ]
            else:
                # LMCache -> vLLM
                vllm_key_value_cache[block_idx, block_offset] = lmc_key_value_cache[
                    token_idx
                ]

    else:
        # ── Non-MLA format ──
        # Determine vLLM layout and block_size
        is_two_major = gpu_kv_format == GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        # flash attn:
        #   [2, num_blocks, block_size, num_heads, head_size]
        #   -> dim2 = block_size
        # flash infer:
        #   [num_blocks, 2, block_size, num_heads, head_size]
        #   -> dim2 = block_size
        block_size = vllm_key_value_cache.size(2)
        num_heads = vllm_key_value_cache.size(3)
        head_size = vllm_key_value_cache.size(4)

        for token_idx in range(num_tokens):
            slot_idx = slots[token_idx]
            if slot_idx < 0:
                continue

            block_idx = slot_idx // block_size
            block_offset = slot_idx % block_size

            for kv in range(2):  # 0=Key, 1=Value
                # ── Read vLLM side: [num_heads, head_size] ──
                if is_two_major:
                    # [2, num_blocks, block_size, num_heads, head_size]
                    vllm_slice = vllm_key_value_cache[
                        kv, block_idx, block_offset
                    ]  # [num_heads, head_size]
                else:
                    # [num_blocks, 2, block_size, num_heads, head_size]
                    vllm_slice = vllm_key_value_cache[
                        block_idx, kv, block_offset
                    ]  # [num_heads, head_size]

                vllm_flat = vllm_slice.reshape(-1)  # [num_heads * head_size]

                # ── Read/write LMC side ──
                if token_major:
                    # [num_tokens, 2, num_heads * head_size]
                    lmc_flat = lmc_key_value_cache[token_idx, kv]
                else:
                    # [2, num_tokens, num_heads * head_size]
                    lmc_flat = lmc_key_value_cache[kv, token_idx]

                # ── Transfer ──
                if direction == TransferDirection.D2H:
                    # vLLM -> LMCache
                    lmc_flat.copy_(vllm_flat)
                else:
                    # LMCache -> vLLM
                    vllm_slice.copy_(lmc_flat.reshape(num_heads, head_size))


def single_layer_kv_transfer_sgl(
    lmc_key_value_cache: torch.Tensor,
    sgl_key_cache: torch.Tensor,
    sgl_value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    token_major: bool,
):
    """
    Python fallback implementation of single_layer_kv_transfer_sgl.

    Args:
        lmc_key_value_cache:
            [num_tokens, 2, num_heads*head_size] or
            [2, num_tokens, num_heads*head_size]
        sgl_key_cache: [num_blocks, block_size, num_heads, head_size]
        sgl_value_cache: [num_blocks, block_size, num_heads, head_size]
        slot_mapping: [num_tokens] - maps each token to a global slot index
        direction: False for LMCache -> SGLang, True for SGLang -> LMCache
        token_major: Boolean to determine the layout of lmc_key_value_cache
    """

    # 1. Get basic dimensions
    block_size = sgl_key_cache.size(1)
    num_heads = sgl_key_cache.size(2)
    head_size = sgl_key_cache.size(3)

    # 2. Calculate block indices and offsets within the blocks from slot_mapping
    # In SGLang/vLLM, slot_idx = block_idx * block_size + block_offset
    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    block_indices = valid_slots // block_size
    block_offsets = valid_slots % block_size

    # 3. Prepare LMCache views for K and V
    if token_major:
        # Layout: [num_tokens, 2, hidden_size]
        lmc_k = lmc_key_value_cache[:, 0, :]
        lmc_v = lmc_key_value_cache[:, 1, :]
    else:
        # Layout: [2, num_tokens, hidden_size]
        lmc_k = lmc_key_value_cache[0, :, :]
        lmc_v = lmc_key_value_cache[1, :, :]

    # 4. Perform the transfer
    if direction == TransferDirection.H2D:
        # --- Direction: LMCache to SGLang (Paged Buffer) ---
        # Reshape LMC flat tensors to match SGL [num_heads, head_size]
        src_k_reshaped = lmc_k[valid_mask].view(-1, num_heads, head_size)
        src_v_reshaped = lmc_v[valid_mask].view(-1, num_heads, head_size)

        # Advanced indexing: update specific slots in the paged cache
        sgl_key_cache[block_indices, block_offsets] = src_k_reshaped
        sgl_value_cache[block_indices, block_offsets] = src_v_reshaped

    else:
        # --- Direction: SGLang (Paged Buffer) to LMCache ---
        # Gather tensors from paged cache based on mapping
        sampled_k = sgl_key_cache[block_indices, block_offsets]
        sampled_v = sgl_value_cache[block_indices, block_offsets]

        # Flatten the head dimensions and copy into LMC tensors
        lmc_k[valid_mask] = sampled_k.reshape(-1, num_heads * head_size)
        lmc_v[valid_mask] = sampled_v.reshape(-1, num_heads * head_size)


def load_and_reshape_flash(
    key_value: torch.Tensor,
    # Destination (Dst): Pinned CPU Tensor [2, L, T, H]
    key_cache: torch.Tensor,
    # Source (Src): GPU Cache [Blocks, BlockSize, NumHeads, HeadSize]
    value_cache: torch.Tensor,  # Source (Src): GPU Cache
    slot_mapping: torch.Tensor,  # Mapping indices [num_tokens]
    layer_idx: int,
):
    """
    Python equivalent of load_and_reshape_flash.
    Note: In the context of 'test_extract_and_load_back', this function performs
    an EXTRACT operation (Reads from GPU Cache and writes to Pinned CPU memory).
    """
    # 1. Prepare indices on the target device
    # Mapping must be on the same GPU as the cache to perform indexing
    device = key_cache.device
    slot_mapping = slot_mapping.to(device=device, dtype=torch.long)

    block_size = key_cache.size(1)

    # Calculate physical locations within the paged cache
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = slot_mapping % block_size

    # 2. Extract data from Cache (Gather operation)
    # The result k_out/v_out will be on the GPU
    # Shape: [num_tokens, num_heads, head_size]
    k_out = key_cache[block_indices, block_offsets]
    v_out = value_cache[block_indices, block_offsets]

    # 3. Write to the destination tensor (CPU Copy)
    # Target shape: [2, num_layers, num_tokens, hidden_dim]

    # Flatten heads into the hidden dimension: [T, NumHeads, HeadSize] -> [T, HiddenDim]
    hidden_dim = k_out.shape[1] * k_out.shape[2]

    # Assignment automatically handles the Device-to-Host (D2H) transfer
    key_value[0, layer_idx] = k_out.view(-1, hidden_dim)
    key_value[1, layer_idx] = v_out.view(-1, hidden_dim)


def reshape_and_cache_back_flash(
    key_value: torch.Tensor,
    # Source: [2, num_layer, num_tokens, num_heads * head_size]
    # (Can be on CPU/Pinned Memory or GPU)
    key_cache: torch.Tensor,
    # Destination: [num_blocks, block_size, num_heads, head_size]
    # (Must be on GPU)
    value_cache: torch.Tensor,  # Destination: (Must be on GPU)
    slot_mapping: torch.Tensor,  # Indices: [num_tokens]
    layer_idx: int,
):
    """
    Python implementation of reshape_and_cache_back_flash.

    Operation:
        Flat Tensor (Source) -> Paged Attention Cache (Destination)

    Logic:
        1. Extract the specific layer's data from key_value.
        2. Move it to the GPU (if it's on CPU).
        3. Reshape it to match the cache's head structure.
        4. Scatter (write) it into the non-contiguous cache blocks using slot_mapping.
    """

    # 1. Setup Device & Dimensions
    # The cache is on the GPU, so all indices and source data must eventually be there.
    device = key_cache.device

    block_size = key_cache.size(1)
    num_heads = key_cache.size(2)
    head_size = key_cache.size(3)

    # 2. Prepare Indices
    # slot_mapping might be on CPU, must move to GPU for indexing.
    slot_mapping = slot_mapping.to(device=device, dtype=torch.long)

    # Calculate physical block indices and offsets
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = slot_mapping % block_size

    # 3. Process Source Data (Key)
    # Step A: Slice the specific layer from the source tensor
    # Source shape: [2, num_layers, num_tokens, hidden_dim] -> [num_tokens, hidden_dim]
    k_src_flat = key_value[0, layer_idx]
    v_src_flat = key_value[1, layer_idx]

    # Step B: Reshape & Move to GPU
    # .to(device) handles the CPU -> GPU transfer if key_value is in pinned memory.
    # View shape: [num_tokens, num_heads, head_size]
    k_src = k_src_flat.to(device).view(-1, num_heads, head_size)
    v_src = v_src_flat.to(device).view(-1, num_heads, head_size)

    # 4. Write to Cache (Scatter)
    # Using Advanced Indexing to write data into specific blocks/offsets
    key_cache[block_indices, block_offsets] = k_src
    value_cache[block_indices, block_offsets] = v_src


def lmcache_memcpy_async(
    dest: int,
    src: int,
    nbytes: int,
    direction: TransferDirection,
    host_buffer_offset: int,
    host_buffer_alignments: int,
):
    """
    Python fallback implementation that passes the UT by correctly
    handling GPU pointers via libcudart.
    """
    # 1. Power of two check (as in C++)
    if host_buffer_alignments <= 0 or (
        host_buffer_alignments & (host_buffer_alignments - 1) != 0
    ):
        raise ValueError("host_buffer_alignments must be power of two")

    # 2. Get direction value
    # H2D: 0 -> cudaMemcpyHostToDevice (1)
    # D2H: 1 -> cudaMemcpyDeviceToHost (2)
    if direction == TransferDirection.H2D:
        cuda_kind = 1
    elif direction == TransferDirection.D2H:
        cuda_kind = 2
    else:
        cuda_kind = 1 if direction == 0 else 2

    # 3. Load CUDA runtime library
    # We must use the C library to handle these raw pointers
    libcudart = _get_copy_lib()

    # 4. Pointer arithmetic and aligned copy loop
    offset = 0
    mask = host_buffer_alignments - 1

    while offset < nbytes:
        # Calculate chunks based on alignment (1:1 with C++ logic)
        aligned_area_end = (
            (offset + host_buffer_offset) & ~mask
        ) + host_buffer_alignments
        real_end = min(host_buffer_offset + nbytes, aligned_area_end)
        max_nbytes = real_end - offset - host_buffer_offset

        if max_nbytes <= 0:
            break

        current_dest = dest + offset
        current_src = src + offset

        # Use cudaMemcpy if available (supports GPU pointers)
        # Note: We use synchronous cudaMemcpy for the fallback to ensure completion
        if libcudart is not None and hasattr(libcudart, "cudaMemcpy"):
            ret = libcudart.cudaMemcpy(
                ctypes.c_void_p(current_dest),
                ctypes.c_void_p(current_src),
                ctypes.c_size_t(max_nbytes),
                ctypes.c_int(cuda_kind),
            )
            if ret != 0:
                # If CUDA call fails, we try ctypes.memmove as a last resort
                ctypes.memmove(
                    ctypes.c_void_p(current_dest),
                    ctypes.c_void_p(current_src),
                    int(max_nbytes),
                )
        else:
            # Fallback for CPU-only pointers
            ctypes.memmove(
                ctypes.c_void_p(current_dest),
                ctypes.c_void_p(current_src),
                int(max_nbytes),
            )

        offset += max_nbytes


def encode_fast_new(cdf, input_sym, output_buffer, output_lengths):
    """
    Python equivalent of C++ Arithmetic Encoder.
    FIXED:
    1. Renamed 'l' to 'layer_idx' to fix Ruff E741.
    2. Used default arguments in flush_bit to fix Ruff B023 (Late Binding).
    3. Strictly emulates 32-bit unsigned overflow for high/low.
    """
    # 💡 View as uint16 to treat bit-patterns correctly
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    sym_np = input_sym.cpu().numpy().astype(np.uint8)

    n_layers, n_tokens, n_channels = sym_np.shape
    lp = cdf_np.shape[2]
    max_symbol = lp - 2
    precision = 16
    MASK32 = 0xFFFFFFFF

    out_buf_np = np.zeros(output_buffer.shape, dtype=np.uint8)
    out_len_np = np.zeros(output_lengths.shape, dtype=np.int32)

    for layer_idx in range(n_layers):
        for channel_idx in range(n_channels):
            low, high = 0, MASK32
            pending_bits = 0
            output_reg, output_reg_len = 0, 0
            ptr = 0

            def flush_bit(bit, l_idx=layer_idx, c_idx=channel_idx):
                nonlocal output_reg, output_reg_len, ptr
                output_reg = (output_reg << 1) | (int(bit) & 1)
                output_reg_len += 1
                if output_reg_len == 8:
                    if ptr < out_buf_np.shape[2]:
                        out_buf_np[l_idx, c_idx, ptr] = output_reg & 0xFF
                        ptr += 1
                    output_reg, output_reg_len = 0, 0

            for token_idx in range(n_tokens):
                sym = sym_np[layer_idx, token_idx, channel_idx]
                c_low = int(cdf_np[layer_idx, channel_idx, sym])
                c_high = (
                    0x10000
                    if sym == max_symbol
                    else int(cdf_np[layer_idx, channel_idx, sym + 1])
                )

                # 💡 CRITICAL: Span must be uint64 equivalent in Python
                span = (high - low + 1) & MASK32
                if span == 0:
                    span = 0x100000000  # 2^32

                high = (low + ((span * c_high) >> precision) - 1) & MASK32
                low = (low + ((span * c_low) >> precision)) & MASK32

                # Renormalization loop (32-bit state machine)
                while True:
                    if (high & 0x80000000) == (low & 0x80000000):
                        bit = (high >> 31) & 1
                        flush_bit(bit)
                        while pending_bits > 0:
                            flush_bit(1 - bit)
                            pending_bits -= 1
                        low = (low << 1) & MASK32
                        high = ((high << 1) | 1) & MASK32
                    elif (low & 0x40000000) and not (high & 0x40000000):
                        pending_bits += 1
                        low = (low << 1) & 0x7FFFFFFF
                        high = ((high << 1) | 0x80000001) & MASK32
                    else:
                        break

            # Final flushing sequence
            pending_bits += 1
            bit = 1 if (low & 0x40000000) else 0
            flush_bit(bit)
            while pending_bits > 0:
                flush_bit(1 - bit)
                pending_bits -= 1

            if output_reg_len > 0:
                out_buf_np[layer_idx, channel_idx, ptr] = (
                    output_reg << (8 - output_reg_len)
                ) & 0xFF
                ptr += 1
            out_len_np[layer_idx, channel_idx] = ptr

    output_buffer.copy_(torch.from_numpy(out_buf_np))
    output_lengths.copy_(torch.from_numpy(out_len_np))


def uint32_val(val):
    return int(val & 0xFFFFFFFF)


def decode_fast_new(cdf, bytestreams, lengths, output):
    """
    Python implementation of Arithmetic Decoding.
    Strictly aligned with CUDA decode_with_accessor_kernel.
    """
    # Reinterpret raw bytes of cdf as uint16 bit-patterns, then widen to uint32
    # for arithmetic. This matches encode_fast_new and decode_fast_prefsum.
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    bs_np = bytestreams.cpu().numpy().astype(np.uint8)
    len_np = lengths.cpu().numpy().astype(np.int32)

    n_layers, n_tokens, n_channels = output.shape
    _, _, lp = cdf_np.shape
    max_symbol = lp - 2
    precision = 16

    out_np = np.zeros(output.shape, dtype=np.uint8)

    # Use layer_idx to avoid Ruff E741
    for layer_idx in range(n_layers):
        for c in range(n_channels):
            curr_len = int(len_np[layer_idx, c])
            channel_bs = bs_np[layer_idx, c]

            v_val = 0
            if curr_len >= 4:
                v_val = (
                    int(channel_bs[0]) << 24
                    | int(channel_bs[1]) << 16
                    | int(channel_bs[2]) << 8
                    | int(channel_bs[3])
                )
                v_val = uint32_val(v_val)

            byte_buffer_offset = 4
            byte_buffer = (
                int(channel_bs[byte_buffer_offset])
                if byte_buffer_offset < curr_len
                else 0
            )
            bit_idx = 1

            l_val = 0
            h_val = 0xFFFFFFFF

            current_cdf_slice = cdf_np[layer_idx, c]
            for i in range(n_tokens):
                # Calculate span and count for symbol search
                MASK32 = 0xFFFFFFFF
                span = (int(h_val) - int(l_val) + 1) & MASK32
                if span == 0:
                    span = 0x100000000  # 2^32
                v_minus_l = uint32_val(v_val - l_val)
                count = ((int(v_minus_l) + 1) * 65536 - 1) // span
                count = int(count & 0xFFFF)

                # Binary search for the symbol in CDF
                left, right = 0, max_symbol + 1
                while left + 1 < right:
                    m = (left + right) // 2
                    if int(current_cdf_slice[m]) < count:
                        left = m
                    elif int(current_cdf_slice[m]) > count:
                        right = m
                    else:
                        left = m
                        break

                sym_i = left
                out_np[layer_idx, i, c] = sym_i

                if i == n_tokens - 1:
                    break

                c_low = int(current_cdf_slice[sym_i])
                c_high = (
                    0x10000
                    if sym_i == max_symbol
                    else int(current_cdf_slice[sym_i + 1])
                )

                # Update range
                h_val = uint32_val((l_val - 1) + ((span * c_high) >> precision))
                l_val = uint32_val(l_val + ((span * c_low) >> precision))

                # Renormalization
                while True:
                    if (l_val >= 0x80000000) or (h_val < 0x80000000):
                        l_val = uint32_val(l_val << 1)
                        h_val = uint32_val((h_val << 1) | 1)

                        v_val = uint32_val(v_val << 1)
                        v_val |= (byte_buffer >> (8 - bit_idx)) & 1
                        bit_idx += 1

                    elif (l_val >= 0x40000000) and (h_val < 0xC0000000):
                        l_val = uint32_val((l_val << 1) & 0x7FFFFFFF)
                        h_val = uint32_val((h_val << 1) | 0x80000001)
                        v_val = uint32_val(v_val - 0x40000000)

                        v_val = uint32_val(v_val << 1)
                        v_val |= (byte_buffer >> (8 - bit_idx)) & 1
                        bit_idx += 1
                    else:
                        break

                    # Update bit index and byte buffer
                    if bit_idx == 9:
                        bit_idx = 1
                        byte_buffer_offset += 1
                        if byte_buffer_offset < curr_len:
                            byte_buffer = int(channel_bs[byte_buffer_offset])
                        else:
                            byte_buffer = 0

    # Mypy: Validate output is not None before copying
    if output is not None:
        output.copy_(torch.from_numpy(out_np))


def decode_fast_prefsum(cdf, bytestreams, lengths_prefsum, output):
    """
    Python equivalent of C++ decode_fast_prefsum.
    Fixed: Range calculation and ZeroDivisionError handling to match CUDA kernel.
    """
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    bs_np = bytestreams.cpu().numpy().astype(np.uint8)
    pref_np = lengths_prefsum.cpu().numpy().astype(np.int64).flatten()

    n_layers, n_tokens, n_channels = output.shape
    max_symbol = cdf_np.shape[2] - 2
    precision, c_count, MASK32 = 16, 0x10000, 0xFFFFFFFF
    out_np = np.zeros(output.shape, dtype=np.uint8)

    for layer_idx in range(n_layers):
        for c in range(n_channels):
            cid = layer_idx * n_channels + c
            start_off = 0 if cid == 0 else int(pref_np[cid - 1])

            v_val = 0
            if start_off + 4 <= bs_np.size:
                v_val = (
                    int(bs_np[start_off]) << 24
                    | int(bs_np[start_off + 1]) << 16
                    | int(bs_np[start_off + 2]) << 8
                    | int(bs_np[start_off + 3])
                ) & MASK32

            low, high = 0, MASK32
            byte_buffer_offset, bit_idx = start_off + 4, 1
            byte_buffer = (
                int(bs_np[byte_buffer_offset]) if byte_buffer_offset < bs_np.size else 0
            )

            for i in range(n_tokens):
                # 💡 FIX: Emulate 32-bit overflow for span.
                # In C++, 0xFFFFFFFF - 0 + 1 == 0.
                # But for division, we must treat it as 2^32.
                span = (high - low + 1) & MASK32
                if span == 0:
                    span = 0x100000000  # 2^32

                v_minus_l = (v_val - low) & MASK32
                count = ((v_minus_l + 1) * c_count - 1) // span
                count = int(count & 0xFFFF)

                left, right = 0, max_symbol + 1
                current_cdf = cdf_np[layer_idx, c]
                while left + 1 < right:
                    m = (left + right) // 2
                    if int(current_cdf[m]) < count:
                        left = m
                    else:
                        right = m

                sym_i = left
                out_np[layer_idx, i, c] = sym_i
                if i == n_tokens - 1:
                    break

                c_low = int(current_cdf[sym_i])
                c_high = 0x10000 if sym_i == max_symbol else int(current_cdf[sym_i + 1])

                # Update interval
                high = (low + ((span * c_high) >> precision) - 1) & MASK32
                low = (low + ((span * c_low) >> precision)) & MASK32

                # Renormalization
                while True:
                    if low >= 0x80000000 or high < 0x80000000:
                        v_val = (
                            (v_val << 1) | ((byte_buffer >> (8 - bit_idx)) & 1)
                        ) & MASK32
                        low, high = (low << 1) & MASK32, ((high << 1) | 1) & MASK32
                        bit_idx += 1
                    elif low >= 0x40000000 and high < 0xC0000000:
                        v_val = (v_val - 0x40000000) & MASK32
                        v_val = (
                            (v_val << 1) | ((byte_buffer >> (8 - bit_idx)) & 1)
                        ) & MASK32
                        low, high = (
                            (low << 1) & 0x7FFFFFFF,
                            ((high << 1) | 0x80000001) & MASK32,
                        )
                        bit_idx += 1
                    else:
                        break

                    if bit_idx == 9:
                        bit_idx, byte_buffer_offset = 1, byte_buffer_offset + 1
                        byte_buffer = (
                            int(bs_np[byte_buffer_offset])
                            if byte_buffer_offset < bs_np.size
                            else 0
                        )

    output.copy_(torch.from_numpy(out_np))


def calculate_cdf(input_tensor: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    Equivalent to CUDA calculate_cdf.
    Input: Expects a 3D tensor (e.g., [1, N, 1]).
    num_bins: Total number of bins (max_val + 1).
    Returns: Tensor of length (num_bins + 1) with CDF values.
    """
    # Force flattening to match bincount expectation
    flat_input = input_tensor.flatten().long()

    # Use num_bins directly to match the CUDA kernel's 'Alphabet Size'
    counts = torch.bincount(flat_input, minlength=num_bins)

    # Slice to ensure output length is exactly num_bins
    counts = counts[:num_bins]

    cdf = torch.cumsum(counts, dim=0).float()

    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    cdf = torch.cat([torch.tensor([0.0], device=cdf.device), cdf])

    return cdf


def rotary_embedding_k_fused(
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Apply fused rotary embedding undo/redo to key tensor in-place.

    Reverses the rotary embedding at old_positions and applies the rotary
    embedding at new_positions. head_size is unused but kept for API
    compatibility with the CUDA equivalent.

    Args:
        old_positions: Token positions whose rotary embedding to reverse.
        new_positions: Token positions whose rotary embedding to apply.
        key: Key tensor to update in-place.
        head_size: Head size (unused; kept for API compatibility).
        cos_sin_cache: Precomputed cosine/sine cache indexed by position.
        is_neox: If True, uses NeoX-style rotary (contiguous halves);
            otherwise uses GPT-J-style (interleaved).
    """
    rot_dim = cos_sin_cache.shape[1]
    half_rot = rot_dim // 2

    old_cs = cos_sin_cache[old_positions]
    new_cs = cos_sin_cache[new_positions]

    oc, os = old_cs[:, :half_rot].unsqueeze(1), old_cs[:, half_rot:].unsqueeze(1)
    nc, ns = new_cs[:, :half_rot].unsqueeze(1), new_cs[:, half_rot:].unsqueeze(1)

    if is_neox:
        x = key[..., :half_rot]
        y = key[..., half_rot:rot_dim]
    else:
        x = key[..., :rot_dim:2]
        y = key[..., 1:rot_dim:2]

    x_rev = x * oc + y * os
    y_rev = y * oc - x * os

    x_out = x_rev * nc - y_rev * ns
    y_out = y_rev * nc + x_rev * ns

    if is_neox:
        key[..., :half_rot] = x_out
        key[..., half_rot:rot_dim] = y_out
    else:
        key[..., :rot_dim:2] = x_out
        key[..., 1:rot_dim:2] = y_out


def get_gpu_pci_bus_id(device_id: int = 0, keyword: str = "NVIDIA") -> str | None:
    """
    Get the PCI bus ID of a GPU device on Linux in a stable order.

    Args:
        device_id (int): Index of the GPU among matching devices.
        keyword (str): Keyword to match in the PCI device description.

    Returns:
        str | None: PCI bus ID (e.g., "0000:29:00.0") or None if not found.
    """
    PCI_IDS_PATHS = ["/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids"]

    def parse_pci_ids(keyword: str) -> set[int]:
        """
        Parse pci.ids and return a set of device IDs (hex) that match the keyword.
        """
        ids = set()
        for path in PCI_IDS_PATHS:
            f = Path(path)
            if not f.exists():
                continue
            with f.open("r", encoding="utf-8", errors="ignore") as fd:
                for line in fd:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("\t"):
                        continue
                    if keyword.lower() in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                ids.add(int(parts[0], 16))
                            except ValueError:
                                pass
        return ids

    # Try /sys/bus/pci/devices
    pci_base = Path("/sys/bus/pci/devices")
    matching_devices = []
    target_device_ids = parse_pci_ids(keyword)

    if pci_base.exists():
        for dev in pci_base.iterdir():
            try:
                vendor_file = dev / "vendor"
                if not vendor_file.exists():
                    continue
                vendor_id_hex = int(vendor_file.read_text().strip(), 16)
                if not target_device_ids or vendor_id_hex in target_device_ids:
                    addr = dev.name
                    # Ensure full format "0000:BB:DD.F"
                    parts = addr.split(":")
                    if len(parts) == 2:  # short format
                        addr = "0000:" + addr
                    matching_devices.append(addr)
            except Exception:
                continue

    # Sort by bus number to guarantee stable device order
    def pci_key(addr: str) -> int:
        """
        Convert PCI address to a sortable integer by bus number.
        Assumes full format "0000:BB:DD.F".
        """
        try:
            # addr = "0000:29:00.0" -> bus = 29
            bus = addr.split(":")[1]
            return int(bus, 16)
        except Exception:
            return 0

    matching_devices.sort(key=pci_key)

    if device_id < len(matching_devices):
        addr = matching_devices[device_id]
        return addr.upper()

    # Fallback to lspci
    try:
        output = subprocess.check_output(["lspci"], text=True)
        lines = [
            line for line in output.splitlines() if keyword.lower() in line.lower()
        ]
        # Take first word (PCI address) and sort
        lspci_addrs = [line.split()[0] for line in lines]

        # Complete the domain to ensure full format "0000:BB:DD.F"
        for i, addr in enumerate(lspci_addrs):
            if len(addr.split(":")) == 2:
                lspci_addrs[i] = "0000:" + addr

        lspci_addrs.sort(key=pci_key)
        if device_id < len(lspci_addrs):
            addr = lspci_addrs[device_id]
            return addr.upper()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return None
