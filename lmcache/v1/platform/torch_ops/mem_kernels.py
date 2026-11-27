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
from lmcache.v1.platform.torch_ops._kv_format import _format_spec
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

    format_spec = _format_spec(engine_kv_format)

    # TODO: Implement head_size support for HND layouts as next step.
    if format_spec.is_hnd:
        raise NotImplementedError(
            "HND layouts are not supported in the non-CUDA fallback. "
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
    is_mla = format_spec.is_mla
    has_interleaved_kv_blocks = (
        format_spec.is_layer_list
        and not format_spec.is_mla
        and not format_spec.is_fused_packed
        and not format_spec.is_two_major
        and not format_spec.is_kv_second_tuple
    )

    num_layers = key_value.size(1)
    hidden_size = key_value.size(3)

    # For layouts with K/V interleaved after the block axis, pre-compute
    # block-level indices.
    if has_interleaved_kv_blocks:
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

    # Determine the physical shape of the underlying paged tensor
    # (used when wrapping a raw pointer).
    layer_shape: Tuple[int, ...]

    if is_mla:
        layer_shape = (page_buffer_size, hidden_size)
    elif has_interleaved_kv_blocks:
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
        elif has_interleaved_kv_blocks:
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

    format_spec = _format_spec(engine_kv_format)
    is_mla = format_spec.is_mla

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
        is_two_major = format_spec.is_two_major
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
    - Pointer mode with libcudart: uses synchronous cudaMemcpy, split at
      every host_buffer_alignments boundary (see below)
    - Pointer mode without libcudart: uses CPU tensor copy

    Pointer mode with libcudart: the host buffer is pinned in
    host_buffer_alignments-sized chunks (LazyMemoryAllocator), and ONE
    cudaMemcpy spanning two registered chunks fails with
    cudaErrorInvalidValue (error code 1) — measured on 4-bit NVFP4 KV stores
    (LMCache/LMCache#6617). This fallback therefore splits the copy at every
    host_buffer_alignments boundary, mirroring the C++ lmcache_memcpy_async
    in csrc/cuda/mem_kernels.cu. The regression test
    tests/v1/platform/test_memcpy_async_host_register_split.py locks this
    contract in.

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
            # Split host<->device copies at every cudaHostRegister boundary:
            # LazyMemoryAllocator pins host_buffer_alignments-sized chunks
            # separately, and one cudaMemcpy spanning two registered chunks
            # fails with cudaErrorInvalidValue — unlike what the old comment
            # claimed, synchronous cudaMemcpy does NOT handle that case
            # (context: LMCache/LMCache#6617). The C++ lmcache_memcpy_async
            # in csrc/cuda/mem_kernels.cu already splits the same way.
            base_off = int(host_buffer_offset) if host_buffer_offset else 0
            align = int(host_buffer_alignments) if host_buffer_alignments else 0
            remaining = int(nbytes)
            done = 0
            while remaining > 0:
                piece = remaining
                if align > 0:
                    piece = min(piece, align - ((base_off + done) % align))
                ret = libcudart.cudaMemcpy(
                    ctypes.c_void_p(dest + done),
                    ctypes.c_void_p(src + done),
                    ctypes.c_size_t(piece),
                    ctypes.c_int(4),  # cudaMemcpyDefault
                )
                if ret != 0:
                    raise RuntimeError(
                        f"cudaMemcpy failed with error code {ret} "
                        f"(offset={done} piece={piece} nbytes={nbytes})"
                    )
                done += piece
                remaining -= piece
        except AttributeError:
            raise
    else:
        # Pure CPU copy — no alignment constraints.
        _copy_bytes_with_tensor(dest, src, nbytes)
