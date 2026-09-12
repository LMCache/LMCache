# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Tuple
import ctypes

# Third Party
import torch

# First Party
from lmcache.lmcache_native import (
    EngineKVFormat,
    TransferDirection,
)
from lmcache.v1.platform.torch_ops._kv_format import _format_spec, _is_two_major_format
from lmcache.v1.platform.torch_ops._tensor_from_ptr import (
    _copy_bytes_with_tensor,
    _get_copy_lib,
    _tensor_from_ptr,
)


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor | list[torch.Tensor],
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
    block_size: int = 0,
    head_size: int = 0,
    skip_prefix_n_tokens: int = 0,
    block_stride_elems: int = 0,
):
    """
    Fully vectorized Python fallback for multi_layer_kv_transfer.
    Eliminates ALL token- and KV-level Python loops.

    ``block_stride_elems`` mirrors the cuda_ops signature; pointer-based
    paged reconstruction rejects non-tight pools in _normalize_paged_layers.
    """
    if not isinstance(key_value_ptrs, (torch.Tensor, list)):
        raise TypeError(
            f"Expected torch.Tensor or list, but got {type(key_value_ptrs).__name__}"
        )

    # TODO: Implement head_size support for HND layouts (NL_X_TWO_NB_NH_BS_HS,
    # NL_X_NB_TWO_NH_BS_HS) as next step.
    if int(engine_kv_format) in (
        int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS),
        int(EngineKVFormat.NL_X_NB_TWO_NH_BS_HS),
    ):
        raise NotImplementedError(
            "HND layouts (NL_X_TWO_NB_NH_BS_HS, NL_X_NB_TWO_NH_BS_HS) "
            "are not supported in the non-CUDA fallback. "
            "head_size parameter is required but not implemented in this path."
        )
    # 1. Filter out invalid slots.
    #    valid_mask_kv:  on key_value.device, used to index key_value
    #    valid_slots:    on paged_memory_device, used to index paged_tensor
    kv_device = key_value.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0
    # Skip the first skip_prefix_n_tokens tokens from transfer.
    # This matches the CUDA kernel semantics where the grid starts at
    # token_id=0 but indexes key_value/slot_mapping at
    # kv_token_id = token_id + skip_prefix_n_tokens.
    # By masking them as invalid, the vectorized indexing via valid_mask_kv
    # naturally skips them while keeping key_value indices aligned.
    if skip_prefix_n_tokens > 0:
        valid_mask_kv[:skip_prefix_n_tokens] = False
    if not valid_mask_kv.any():
        return

    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)

    # 2. Determine architecture variant and tensor dimensions.
    is_mla = _format_spec(engine_kv_format).is_mla
    is_flash_infer = int(engine_kv_format) == int(EngineKVFormat.NL_X_NB_TWO_BS_NH_HS)

    num_layers = key_value.size(1)
    hidden_size = key_value.size(3)

    # For the flash_infer interleaved layout, pre-compute block-level indices.
    if is_flash_infer:
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

    # Determine the physical shape of the underlying paged tensor
    # (used when wrapping a raw pointer).
    layer_shape: Tuple[int, ...]

    if is_mla:
        layer_shape = (page_buffer_size, hidden_size)
    elif is_flash_infer:
        num_blocks = page_buffer_size // block_size
        layer_shape = (num_blocks, 2, block_size, hidden_size)
    else:
        layer_shape = (2, page_buffer_size, hidden_size)

    # 3. Iterate over layers — the only remaining Python-level loop.
    for layer_id in range(num_layers):
        # --- A. Obtain the physical device-memory view for this layer. ---
        if isinstance(key_value_ptrs, list):
            paged_tensor = key_value_ptrs[layer_id]
        else:
            ptr = int(key_value_ptrs[layer_id].item())
            # Convert a raw device pointer into a PyTorch tensor view.
            paged_tensor = _tensor_from_ptr(
                ptr, layer_shape, key_value.dtype, paged_memory_device
            )

        # --- B. Vectorized bulk data transfer. ---
        if is_mla:
            # Paged layout : [page_buffer_size, hidden_size]
            # key_value layout: [1, num_layers, num_tokens, hidden_size]
            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[0, layer_id, valid_mask_kv, :]
                paged_tensor.index_copy_(
                    0, valid_slots, lmc_valid.to(paged_tensor.device)
                )
            else:
                gathered = paged_tensor.index_select(0, valid_slots)
                key_value[0, layer_id, valid_mask_kv, :] = gathered.to(
                    kv_device, non_blocking=False
                )
        elif is_flash_infer:
            # Paged layout : [num_blocks, 2, block_size, hidden_size]
            # key_value layout: [2, num_layers, num_tokens, hidden_size]
            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[:, layer_id, valid_mask_kv, :]
                src_data = lmc_valid.transpose(0, 1).to(paged_memory_device)
                # src_data: [num_valid, 2, hidden_size]
                paged_tensor[block_indices, :, block_offsets, :] = src_data
            else:
                gathered = paged_tensor[block_indices, :, block_offsets, :]
                # gathered: [num_valid, 2, hidden_size]
                key_value[:, layer_id, valid_mask_kv, :] = gathered.to(
                    kv_device, non_blocking=False
                ).transpose(0, 1)
        else:
            # Paged layout : [2, page_buffer_size, hidden_size]
            # key_value layout: [2, num_layers, num_tokens, hidden_size]
            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[:, layer_id, valid_mask_kv, :]
                paged_tensor.index_copy_(
                    1, valid_slots, lmc_valid.to(paged_memory_device)
                )
            else:
                gathered = paged_tensor.index_select(1, valid_slots)
                key_value[:, layer_id, valid_mask_kv, :] = gathered.to(
                    kv_device, non_blocking=False
                )


def multi_layer_kv_transfer_unilateral(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor | list[torch.Tensor],
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
):
    """
    Python fallback for multi_layer_kv_transfer_unilateral

    Handles SGLang MHA format where K and V paged buffers are stored separately:
        ptrs = [K_layer0, K_layer1, ..., V_layer0, V_layer1, ...]
        each buffer shape: [page_buffer_size, hidden_size]

    For MLA, delegates to multi_layer_kv_transfer (same as C++ implementation).

    key_value_ptrs:
        - If torch.Tensor: int64 tensor containing raw memory pointers.
        - If list[torch.Tensor]: list of tensor objects.

    key_value layout:
        - Standard: [2, num_layers, num_tokens, hidden_size]
        - MLA:      [1, num_layers, num_tokens, hidden_size]

    direction:
        H2D = LMCache  -> PagedBuffer
        D2H = PagedBuffer -> LMCache
    """
    is_mla = _format_spec(engine_kv_format).is_mla

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
            engine_kv_format,
            0,  # block_size unused for MLA formats
        )
    # ── Non-MLA path: unilateral (separate K/V buffers per layer) ──
    num_layers = key_value.size(1)
    hidden_size = key_value.size(3)
    layer_shape = (page_buffer_size, hidden_size)

    kv_device = key_value.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0
    if not valid_mask_kv.any():
        return

    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)

    for layer_id in range(num_layers):
        for kv_idx in range(2):  # 0 = K, 1 = V
            buffer_idx = layer_id + kv_idx * num_layers
            if isinstance(key_value_ptrs, list):
                paged_tensor = key_value_ptrs[buffer_idx]
            else:
                ptr = int(key_value_ptrs[buffer_idx].item())
                paged_tensor = _tensor_from_ptr(
                    ptr, layer_shape, key_value.dtype, paged_memory_device
                )

            if int(direction) == int(TransferDirection.H2D):
                lmc_valid = key_value[kv_idx, layer_id, valid_mask_kv, :]
                paged_tensor.index_copy_(
                    0, valid_slots, lmc_valid.to(paged_memory_device)
                )
            else:
                gathered = paged_tensor.index_select(0, valid_slots)
                key_value[kv_idx, layer_id, valid_mask_kv, :] = gathered.to(kv_device)


def single_layer_kv_transfer(
    lmc_key_value_cache: torch.Tensor,
    vllm_key_value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
    token_major: bool = False,
):
    """
    Vectorized Python fallback for single_layer_kv_transfer
    (eliminates per-token loops).

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
    kv_device = lmc_key_value_cache.device
    paged_memory_device = vllm_key_value_cache.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0

    if not valid_mask_kv.any():
        return

    valid_token_indices = torch.nonzero(valid_mask_kv, as_tuple=True)[0]
    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)

    is_mla = _format_spec(engine_kv_format).is_mla

    if is_mla:
        # ── MLA format ──
        # vllm: [num_blocks, block_size, head_size]
        # lmc:  [num_tokens, aligned_head_size]
        block_size = vllm_key_value_cache.size(1)
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

        if int(direction) == int(TransferDirection.D2H):
            # vLLM -> LMCache
            lmc_key_value_cache[valid_token_indices] = vllm_key_value_cache[
                block_indices, block_offsets
            ].to(lmc_key_value_cache.device)
        else:
            # LMCache -> vLLM
            vllm_key_value_cache[block_indices, block_offsets] = lmc_key_value_cache[
                valid_token_indices
            ].to(paged_memory_device)

    else:
        # ── Non-MLA format ──
        # Determine vLLM layout and block_size
        is_two_major = _is_two_major_format(engine_kv_format)
        # flash attn:
        #   [2, num_blocks, block_size, num_heads, head_size]
        #   -> dim2 = block_size
        # flash infer:
        #   [num_blocks, 2, block_size, num_heads, head_size]
        #   -> dim2 = block_size
        block_size = vllm_key_value_cache.size(2)
        num_heads = vllm_key_value_cache.size(3)
        head_size = vllm_key_value_cache.size(4)
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

        for kv in range(2):
            if int(direction) == int(TransferDirection.D2H):
                if is_two_major:
                    gathered = vllm_key_value_cache[kv, block_indices, block_offsets]
                else:
                    gathered = vllm_key_value_cache[block_indices, kv, block_offsets]

                gathered_flat = gathered.reshape(-1, num_heads * head_size).to(
                    lmc_key_value_cache.device
                )
                if token_major:
                    lmc_key_value_cache[valid_token_indices, kv] = gathered_flat
                else:
                    lmc_key_value_cache[kv, valid_token_indices] = gathered_flat
            else:
                if token_major:
                    lmc_src = lmc_key_value_cache[valid_token_indices, kv]
                else:
                    lmc_src = lmc_key_value_cache[kv, valid_token_indices]
                lmc_reshaped = lmc_src.reshape(-1, num_heads, head_size).to(
                    vllm_key_value_cache.device
                )

                if is_two_major:
                    vllm_key_value_cache[kv, block_indices, block_offsets] = (
                        lmc_reshaped
                    )
                else:
                    vllm_key_value_cache[block_indices, kv, block_offsets] = (
                        lmc_reshaped
                    )


def single_layer_kv_transfer_sgl(
    lmc_key_value_cache: torch.Tensor,
    sgl_key_cache: torch.Tensor,
    sgl_value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    token_major: bool = False,
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
    kv_device = lmc_key_value_cache.device
    paged_memory_device = sgl_key_cache.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0
    if not valid_mask_kv.any():
        return

    # 1. Get basic dimensions
    block_size = sgl_key_cache.size(1)
    num_heads = sgl_key_cache.size(2)
    head_size = sgl_key_cache.size(3)

    # 2. Calculate block indices and offsets within the blocks from slot_mapping
    # In SGLang/vLLM, slot_idx = block_idx * block_size + block_offset
    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)
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
    if int(direction) == int(TransferDirection.H2D):
        # --- Direction: LMCache to SGLang (Paged Buffer) ---
        # Reshape LMC flat tensors to match SGL [num_heads, head_size]
        src_k_reshaped = (
            lmc_k[valid_mask_kv]
            .reshape(-1, num_heads, head_size)
            .to(paged_memory_device)
        )
        src_v_reshaped = (
            lmc_v[valid_mask_kv]
            .reshape(-1, num_heads, head_size)
            .to(paged_memory_device)
        )

        # Advanced indexing: update specific slots in the paged cache
        sgl_key_cache[block_indices, block_offsets] = src_k_reshaped
        sgl_value_cache[block_indices, block_offsets] = src_v_reshaped

    else:
        # --- Direction: SGLang (Paged Buffer) to LMCache ---
        # Gather tensors from paged cache based on mapping
        sampled_k = sgl_key_cache[block_indices, block_offsets].to(kv_device)
        sampled_v = sgl_value_cache[block_indices, block_offsets].to(kv_device)

        # Flatten the head dimensions and copy into LMC tensors
        lmc_k[valid_mask_kv] = sampled_k.reshape(-1, num_heads * head_size)
        lmc_v[valid_mask_kv] = sampled_v.reshape(-1, num_heads * head_size)


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
    dest: int | torch.Tensor,
    src: int | torch.Tensor,
    nbytes: int,
    direction: TransferDirection,
    host_buffer_offset: int,
    host_buffer_alignments: int,
):
    """
    Python fallback for lmcache_memcpy_async.

    - Tensor mode (non-CUDA devices like HPU): uses .to(device) + copy_()
    - Pointer mode with libcudart: uses synchronous cudaMemcpy (cudaMemcpyDefault)
    - Pointer mode without libcudart: uses CPU tensor copy

    Unlike the C++ version (which uses cudaMemcpyAsync and must split copies
    at cudaHostRegister boundaries), this Python fallback does NOT need
    alignment-based chunking because:
    - cudaMemcpy (synchronous) handles cross-cudaHostRegister boundaries
      internally via staging buffers
    - CPU tensor copy has no alignment constraints
    - Tensor mode bypasses raw pointers entirely

    dest:
        - If int: raw memory pointer (used for CUDA/CPU devices where we
          work with pointers).
        - If torch.Tensor: tensor object (used for non-CUDA/CPU devices
          where we operate on tensor objects directly).

    src:
        - If int: raw memory pointer (used for CUDA/CPU devices where we
          work with pointers).
        - If torch.Tensor: tensor object (used for non-CUDA/CPU devices
          where we operate on tensor objects directly).
    """
    # 1. Power of two check (kept for API compatibility)
    if host_buffer_alignments <= 0 or (
        host_buffer_alignments & (host_buffer_alignments - 1) != 0
    ):
        raise ValueError("host_buffer_alignments must be power of two")

    # 2. Validate direction
    if int(direction) not in (int(TransferDirection.H2D), int(TransferDirection.D2H)):
        raise ValueError(f"Unsupported direction: {direction}")

    # 3. Tensor-backed mode.
    # Mixed pointer/tensor are not allowed
    if isinstance(dest, torch.Tensor) or isinstance(src, torch.Tensor):
        if not (isinstance(dest, torch.Tensor) and isinstance(src, torch.Tensor)):
            raise TypeError(
                "Mixed types are not allowed: both dest and src must be torch.Tensor "
                "if either of them is a tensor."
            )
        if nbytes % dest.element_size() != 0:
            raise ValueError("nbytes must align with tensor element size")

        num_elements = nbytes // dest.element_size()

        dest_slice = dest.flatten()[:num_elements]
        src_slice = src.flatten()[:num_elements]

        copied = src_slice.to(dest_slice.device)
        dest_slice.copy_(copied)
        return

    # 4. Pointer mode
    if not isinstance(dest, int) or not isinstance(src, int):
        raise TypeError(
            "dest and src must be both int (pointer mode) "
            "or both torch.Tensor (tensor mode)"
        )

    libcudart = _get_copy_lib()
    if libcudart is not None and hasattr(libcudart, "cudaMemcpy"):
        try:
            # Synchronous cudaMemcpy handles cross-cudaHostRegister boundaries
            # internally — no manual alignment splitting needed.
            ret = libcudart.cudaMemcpy(
                ctypes.c_void_p(dest),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(4),  # cudaMemcpyDefault
            )
            if ret != 0:
                raise RuntimeError(f"cudaMemcpy failed with error code {ret}")
        except AttributeError:
            raise
    else:
        # Pure CPU copy — no alignment constraints.
        _copy_bytes_with_tensor(dest, src, nbytes)
