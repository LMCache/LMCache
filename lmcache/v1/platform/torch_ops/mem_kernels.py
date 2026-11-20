# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Tuple
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


def _memcpy_async_symbol(lib: object) -> Any | None:
    """Return the runtime's asynchronous device copy entry point.

    CUDA and HIP spell it differently, and a runtime may expose neither. The
    symbol is looked up once per call: ``ctypes`` caches the attribute on the
    library object, so the cost is a dictionary lookup.
    """
    for name in ("cudaMemcpyAsync", "hipMemcpyAsync"):
        symbol = getattr(lib, name, None)
        if symbol is not None:
            return symbol
    return None


def _current_stream_handle() -> int:
    """Return the raw handle of the caller's current device stream.

    ``0`` is the legacy default stream, which is also what a runtime uses when
    no stream is recorded, so callers that never entered a stream keep working.
    """
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return 0
    return int(torch.cuda.current_stream().cuda_stream)


def _copy_within_pin_chunks(
    memcpy_async: Any,
    dest: int,
    src: int,
    nbytes: int,
    host_buffer_offset: int,
    host_buffer_alignments: int,
    stream: int,
) -> None:
    """Issue one asynchronous copy per host pin chunk.

    ``cudaHostRegister`` pins the staging buffer in ``host_buffer_alignments``
    chunks, and a device copy that crosses one of those boundaries fails with
    ``cudaErrorInvalidValue`` (see #6617). The synchronous entry point stages
    such a copy internally; the asynchronous one does not, so split it here.

    The split follows the host side, which is the pinned one: the first chunk
    ends at the next boundary after ``host_buffer_offset``. Both pointers move
    together, so one offset describes the whole transfer.

    Args:
        memcpy_async: The runtime's asynchronous copy entry point.
        dest: Destination pointer.
        src: Source pointer.
        nbytes: Bytes to copy in total.
        host_buffer_offset: Byte offset of the host pointer inside the lazy
            allocator's pinned arena.
        host_buffer_alignments: Pin registration alignment, a power of two.
        stream: Raw handle of the stream the copy belongs to.

    Raises:
        RuntimeError: If the runtime rejects a chunk.
    """
    copied = 0
    while copied < nbytes:
        to_boundary = host_buffer_alignments - (
            (host_buffer_offset + copied) % host_buffer_alignments
        )
        chunk = min(nbytes - copied, to_boundary)
        ret = memcpy_async(
            ctypes.c_void_p(dest + copied),
            ctypes.c_void_p(src + copied),
            ctypes.c_size_t(chunk),
            ctypes.c_int(4),  # cudaMemcpyDefault
            ctypes.c_void_p(stream),
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed with error code {ret}")
        copied += chunk


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
    - Pointer mode with a CUDA or HIP runtime: uses the runtime's asynchronous
      copy on the caller's current stream
    - Pointer mode without such a runtime: uses CPU tensor copy

    Ordering: callers queue the copy on the current stream, and the compiled
    ``cuda_ops`` path issues ``cudaMemcpyAsync`` there. A synchronous
    ``cudaMemcpy`` would instead run on the legacy default stream, which does
    not wait for the gather kernel queued on the cache context's non-blocking
    stream; the copy could then read the staging buffer before that kernel
    filled it, storing stale KV with nothing logged. This fallback therefore
    issues the copy asynchronously on the current stream as well, which keeps
    the ordering without ever synchronising the device. The entry point is
    resolved by name, so ROCm's ``hipMemcpyAsync`` takes the same path.

    Chunking: the host side is registered with ``cudaHostRegister`` in
    ``host_buffer_alignments`` chunks, and a copy that crosses one of those
    boundaries fails with ``cudaErrorInvalidValue``. ``cudaMemcpyAsync`` does
    not split a copy for us the way the synchronous entry point does, so the
    copy is issued once per chunk, starting from ``host_buffer_offset``.

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
    memcpy_async = _memcpy_async_symbol(libcudart) if libcudart is not None else None
    if memcpy_async is None:
        # No runtime exposes the async entry point, or there is no GPU runtime
        # at all: copy through tensors, which is ordered by the tensor library's
        # own stream handling.
        _copy_bytes_with_tensor(dest, src, nbytes)
        return
    # The copy belongs to the caller's current stream, exactly as the compiled
    # cuda_ops path issues it. Anything queued there before this call -- the
    # gather kernel that fills a staging buffer -- therefore completes first,
    # and no explicit synchronization is needed. The gate is the async symbol,
    # not `cudaMemcpy`: on ROCm the library exposes `hipMemcpy*` only, and a
    # check for the CUDA spelling would send device pointers to the tensor path.
    _copy_within_pin_chunks(
        memcpy_async,
        dest,
        src,
        nbytes,
        host_buffer_offset,
        host_buffer_alignments,
        _current_stream_handle(),
    )
