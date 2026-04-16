# SPDX-License-Identifier: Apache-2.0
#
# XPU backend for LMCache using Triton kernels for performance-critical
# operations and re-exporting non-CUDA fallbacks for everything else.
#
# This module is loaded by lmcache/__init__.py when XPU is detected.

# Re-export everything from non_cuda_equivalents as the baseline.
# Only the functions overridden below will differ on XPU.
from lmcache.non_cuda_equivalents import *  # noqa: F401, F403
from lmcache.non_cuda_equivalents import (
    GPUKVFormat,
    TransferDirection,
    _tensor_from_ptr,
)

# Third Party
import torch
import triton
import triton.language as tl

# ======================================================================
# GPUKVFormat constants (mirrors the IntEnum values for use in Triton
# constexpr dispatch).
# ======================================================================
_FMT_NB_NL_TWO_BS_NH_HS = int(GPUKVFormat.NB_NL_TWO_BS_NH_HS)  # 0
_FMT_NL_X_TWO_NB_BS_NH_HS = int(GPUKVFormat.NL_X_TWO_NB_BS_NH_HS)  # 1
_FMT_NL_X_NB_TWO_BS_NH_HS = int(GPUKVFormat.NL_X_NB_TWO_BS_NH_HS)  # 2
_FMT_NL_X_NB_BS_HS = int(GPUKVFormat.NL_X_NB_BS_HS)  # 3
_FMT_TWO_X_NL_X_NBBS_NH_HS = int(GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS)  # 4
_FMT_NL_X_NBBS_ONE_HS = int(GPUKVFormat.NL_X_NBBS_ONE_HS)  # 5
_FMT_NL_X_TWO_NB_NH_BS_HS = int(GPUKVFormat.NL_X_TWO_NB_NH_BS_HS)  # 6
_FMT_NL_X_NB_TWO_NH_BS_HS = int(GPUKVFormat.NL_X_NB_TWO_NH_BS_HS)  # 7


# ======================================================================
# Triton kernel: multi-layer KV transfer
#
# Grid: (num_transfer_tokens, num_layers, kv_size)
#   - kv_size = 1 for MLA, 2 otherwise
#
# Each program instance copies one token's worth of data for one layer
# and one of K/V between the LMCache buffer and the paged KV cache.
# ======================================================================
@triton.jit
def _multi_layer_kv_transfer_kernel(
    # Pointers
    key_value_ptr,  # LMCache buffer flat pointer
    paged_buffer_ptrs,  # Device tensor of int64 pointers, one per layer
    slot_mapping_ptr,  # [num_tokens] int64
    # Dimensions
    scalars_per_token: tl.constexpr,  # hidden_size (in elements)
    num_tokens,  # total tokens in key_value (including skipped prefix)
    num_layers,
    page_buffer_size,  # NB * BS
    block_size,
    head_size,  # only used by HND formats
    skip_prefix_n_tokens,
    # Compile-time constants
    DIRECTION: tl.constexpr,  # 0 = H2D (lmc→paged), 1 = D2H (paged→lmc)
    FORMAT: tl.constexpr,  # GPUKVFormat int value
    BLOCK: tl.constexpr,  # tile width for the inner loop
):
    token_id = tl.program_id(0)
    layer_id = tl.program_id(1)
    k_or_v = tl.program_id(2)

    # When key_value_ptr comes as a raw int (CPU pinned buffer passed via
    # data_ptr()), cast it to a proper Triton pointer so tl.load/tl.store work.
    kv_ptr = key_value_ptr.to(tl.pointer_type(tl.int16))

    kv_token_id = token_id + skip_prefix_n_tokens
    slot_idx = tl.load(slot_mapping_ptr + kv_token_id)

    if slot_idx < 0:
        return

    # Load the device pointer for this layer's paged buffer
    layer_ptr_int = tl.load(paged_buffer_ptrs + layer_id)
    layer_ptr = layer_ptr_int.to(tl.pointer_type(tl.int16))

    offsets = tl.arange(0, BLOCK)

    for start in range(0, scalars_per_token, BLOCK):
        i = offsets + start
        mask = i < scalars_per_token

        # ── LMCache buffer offset ──
        # Layout: [kv_size, num_layers, num_tokens, scalars_per_token]
        lmc_off = (
            k_or_v * num_layers * num_tokens * scalars_per_token
            + layer_id * num_tokens * scalars_per_token
            + kv_token_id * scalars_per_token
            + i
        )

        # ── Paged buffer offset (depends on format) ──
        # NB_NL_TWO_BS_NH_HS (0) and NL_X_TWO_NB_BS_NH_HS (1):
        #   k_or_v * PBS * SPT + slot * SPT + i
        # NL_X_NB_TWO_BS_NH_HS (2) — flash infer NHD:
        #   block_idx * 2 * BS * SPT + k_or_v * BS * SPT + block_off * SPT + i
        # NL_X_NB_BS_HS (3) and NL_X_NBBS_ONE_HS (5) — MLA:
        #   slot * SPT + i
        # NL_X_TWO_NB_NH_BS_HS (6) — flash attn HND:
        #   k_or_v * PBS * SPT + block_idx * NH * BS * HS
        #   + head_idx * BS * HS + block_off * HS + head_off
        # NL_X_NB_TWO_NH_BS_HS (7) — flash infer HND:
        #   block_idx * 2 * NH * BS * HS + k_or_v * NH * BS * HS
        #   + head_idx * BS * HS + block_off * HS + head_off

        if FORMAT == 0 or FORMAT == 1:
            # NB_NL_TWO_BS_NH_HS / NL_X_TWO_NB_BS_NH_HS
            page_off = (
                k_or_v * page_buffer_size * scalars_per_token
                + slot_idx * scalars_per_token
                + i
            )
        elif FORMAT == 2:
            # NL_X_NB_TWO_BS_NH_HS (flash infer NHD)
            block_idx = slot_idx // block_size
            block_off = slot_idx % block_size
            page_off = (
                block_idx * 2 * block_size * scalars_per_token
                + k_or_v * block_size * scalars_per_token
                + block_off * scalars_per_token
                + i
            )
        elif FORMAT == 3 or FORMAT == 5:
            # MLA: NL_X_NB_BS_HS / NL_X_NBBS_ONE_HS
            page_off = slot_idx * scalars_per_token + i
        elif FORMAT == 6:
            # NL_X_TWO_NB_NH_BS_HS (flash attn HND)
            block_idx = slot_idx // block_size
            block_off = slot_idx % block_size
            head_idx = i // head_size
            head_off = i % head_size
            page_off = (
                k_or_v * page_buffer_size * scalars_per_token
                + block_idx * (scalars_per_token // head_size) * block_size * head_size
                + head_idx * block_size * head_size
                + block_off * head_size
                + head_off
            )
        elif FORMAT == 7:
            # NL_X_NB_TWO_NH_BS_HS (flash infer HND)
            block_idx = slot_idx // block_size
            block_off = slot_idx % block_size
            head_idx = i // head_size
            head_off = i % head_size
            num_heads = scalars_per_token // head_size
            page_off = (
                block_idx * 2 * num_heads * block_size * head_size
                + k_or_v * num_heads * block_size * head_size
                + head_idx * block_size * head_size
                + block_off * head_size
                + head_off
            )
        else:
            # Unsupported format — should not reach here
            page_off = i

        if DIRECTION == 1:
            # D2H: paged → lmcache
            vals = tl.load(layer_ptr + page_off, mask=mask)
            tl.store(kv_ptr + lmc_off, vals, mask=mask)
        else:
            # H2D: lmcache → paged
            vals = tl.load(kv_ptr + lmc_off, mask=mask)
            tl.store(layer_ptr + page_off, vals, mask=mask)


# ======================================================================
# Triton kernel: multi-layer KV transfer (unilateral / SGLang MHA)
#
# Separate K and V paged buffers:
#   ptrs = [K_layer0, K_layer1, ..., V_layer0, V_layer1, ...]
# ======================================================================
@triton.jit
def _multi_layer_kv_transfer_unilateral_kernel(
    key_value_ptr,
    paged_buffer_ptrs,  # [num_layers * 2] pointers
    slot_mapping_ptr,
    scalars_per_token: tl.constexpr,
    num_tokens,
    num_layers,
    page_buffer_size,
    DIRECTION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    layer_id = tl.program_id(1)
    k_or_v = tl.program_id(2)

    kv_ptr = key_value_ptr.to(tl.pointer_type(tl.int16))

    slot_idx = tl.load(slot_mapping_ptr + token_id)
    if slot_idx < 0:
        return

    # K pointers are at [0..num_layers), V at [num_layers..2*num_layers)
    ptr_idx = layer_id + k_or_v * num_layers
    layer_ptr_int = tl.load(paged_buffer_ptrs + ptr_idx)
    layer_ptr = layer_ptr_int.to(tl.pointer_type(tl.int16))

    offsets = tl.arange(0, BLOCK)

    for start in range(0, scalars_per_token, BLOCK):
        i = offsets + start
        mask = i < scalars_per_token

        lmc_off = (
            k_or_v * num_layers * num_tokens * scalars_per_token
            + layer_id * num_tokens * scalars_per_token
            + token_id * scalars_per_token
            + i
        )
        page_off = slot_idx * scalars_per_token + i

        if DIRECTION == 1:
            vals = tl.load(layer_ptr + page_off, mask=mask)
            tl.store(kv_ptr + lmc_off, vals, mask=mask)
        else:
            vals = tl.load(kv_ptr + lmc_off, mask=mask)
            tl.store(layer_ptr + page_off, vals, mask=mask)


# ======================================================================
# Triton kernel: single-layer KV transfer (for layerwise connector)
#
# Grid: (num_tokens, kv_size)
#   - kv_size = 1 for MLA, 2 otherwise
#
# Transfers one token's K (and V) between lmc buffer and one layer's
# paged KV cache.  Both tensors are passed directly (not via pointers).
# ======================================================================
@triton.jit
def _single_layer_kv_transfer_kernel(
    lmc_ptr,   # LMCache buffer (flat int16 view)
    vllm_ptr,  # vLLM paged KV cache for one layer (flat int16 view)
    slot_mapping_ptr,  # [num_tokens] int64
    # Strides / offsets (in int16 units)
    lmc_stride,        # stride between tokens in lmc buffer
    lmc_value_offset,  # offset from K to V in lmc buffer
    vllm_block_stride, # stride per block in vllm cache
    vllm_value_offset, # offset from K to V in vllm cache
    scalars_per_token,  # num_heads * head_size in int16 units
    block_size,
    head_size,  # in int16 units, only for HND
    # Compile-time
    DIRECTION: tl.constexpr,  # 0=H2D, 1=D2H
    IS_MLA: tl.constexpr,
    IS_HND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Cast raw int64 pointers to typed pointers (needed when CPU pinned
    # buffers are passed as data_ptr() ints to avoid Triton driver
    # rejecting non-XPU tensors).
    lmc_ptr = lmc_ptr.to(tl.pointer_type(tl.int16))
    vllm_ptr = vllm_ptr.to(tl.pointer_type(tl.int16))

    token_idx = tl.program_id(0)
    k_or_v = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    offsets = tl.arange(0, BLOCK)

    for start in range(0, scalars_per_token, BLOCK):
        i = offsets + start
        mask = i < scalars_per_token

        # LMCache offset
        lmc_off = token_idx * lmc_stride + i
        if not IS_MLA:
            lmc_off = lmc_off + k_or_v * lmc_value_offset

        # vLLM offset depends on layout
        if IS_HND:
            head_idx = i // head_size
            head_off = i % head_size
            vllm_off = (
                block_idx * vllm_block_stride
                + head_idx * block_size * head_size
                + block_offset * head_size
                + head_off
            )
        else:
            # NHD (also correct for MLA where num_heads==1)
            vllm_off = (
                block_idx * vllm_block_stride
                + block_offset * scalars_per_token
                + i
            )
        if not IS_MLA:
            vllm_off = vllm_off + k_or_v * vllm_value_offset

        if DIRECTION == 1:
            # D2H: vllm → lmc
            vals = tl.load(vllm_ptr + vllm_off, mask=mask)
            tl.store(lmc_ptr + lmc_off, vals, mask=mask)
        else:
            # H2D: lmc → vllm
            vals = tl.load(lmc_ptr + lmc_off, mask=mask)
            tl.store(vllm_ptr + vllm_off, vals, mask=mask)


# ======================================================================
# Python dispatch wrappers
# ======================================================================

# Determine the appropriate Triton pointer type width based on dtype.
# The CUDA kernel templates on int64 (8 bytes) for best coalescing,
# then falls back to int32/int16/int8. We use int16 (2 bytes) which
# matches bf16/fp16 element size — the most common KV dtype.
_TRITON_BLOCK = 128


def _get_scalars_and_view(key_value: torch.Tensor):
    """Return (num_elements_per_token, key_value_as_int16_view).

    We reinterpret the data as int16 so the Triton kernel works on 2-byte
    elements regardless of the actual dtype (bf16/fp16 are both 2 bytes).
    For dtypes with different element sizes, we adjust the element count.
    """
    elem_size = key_value.element_size()
    num_origin_elements = key_value.size(3)
    # Number of int16 (2-byte) elements per token
    scalars_per_token = num_origin_elements * elem_size // 2
    # Reinterpret as int16 for the kernel
    kv_view = key_value.view(torch.int16)
    return scalars_per_token, kv_view


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor | list[torch.Tensor],
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
    block_size: int = 0,
    head_size: int = 0,
    skip_prefix_n_tokens: int = 0,
):
    """XPU Triton-accelerated multi_layer_kv_transfer.

    When key_value_ptrs is a raw-pointer int64 tensor (the standard path
    from VLLMPagedMemGPUConnectorV2._initialize_pointers), we launch a
    single Triton kernel that processes all layers × tokens × K/V in one
    shot, matching the CUDA kernel's grid=(tokens, layers, kv_size).

    When key_value_ptrs is a list[Tensor] (the non_cuda_equivalents path),
    we fall back to the vectorized PyTorch implementation.
    """
    # If pointers come as a list of tensors, we can't use pointer-of-pointers.
    # Fall back to the per-layer PyTorch path (imported from non_cuda_equivalents).
    if isinstance(key_value_ptrs, list):
        from lmcache.non_cuda_equivalents import (
            multi_layer_kv_transfer as _fallback_multi_layer_kv_transfer,
        )
        _fallback_multi_layer_kv_transfer(
            key_value, key_value_ptrs, slot_mapping, paged_memory_device,
            page_buffer_size, direction, gpu_kv_format, block_size,
            head_size, skip_prefix_n_tokens,
        )
        return

    fmt = int(gpu_kv_format)
    is_mla = fmt in (_FMT_NL_X_NB_BS_HS, _FMT_NL_X_NBBS_ONE_HS)

    scalars_per_token, kv_view = _get_scalars_and_view(key_value)
    # Convert head_size from element units to int16 units
    elem_size = key_value.element_size()
    head_size_i16 = head_size * elem_size // 2 if head_size > 0 else 0

    num_layers = key_value.size(1)
    num_tokens = key_value.size(2)
    num_transfer_tokens = num_tokens - skip_prefix_n_tokens
    kv_size = 1 if is_mla else 2

    direction_int = 1 if direction == TransferDirection.D2H else 0

    # Triton XPU driver rejects CPU-device tensors at the Python level even
    # when the underlying memory is USM-host (pinned).  Passing the raw
    # data_ptr as an int bypasses the tensor-device check; the L0 driver
    # then calls zeMemGetAllocProperties and correctly identifies it as
    # ZE_MEMORY_TYPE_HOST, which is accessible from device kernels.
    kv_arg = kv_view if kv_view.is_xpu else kv_view.data_ptr()

    grid = (num_transfer_tokens, num_layers, kv_size)

    _multi_layer_kv_transfer_kernel[grid](
        kv_arg,
        key_value_ptrs,
        slot_mapping,
        scalars_per_token,
        num_tokens,
        num_layers,
        page_buffer_size,
        block_size,
        head_size_i16,
        skip_prefix_n_tokens,
        DIRECTION=direction_int,
        FORMAT=fmt,
        BLOCK=_TRITON_BLOCK,
    )


def multi_layer_kv_transfer_unilateral(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor | list[torch.Tensor],
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
):
    """XPU Triton-accelerated multi_layer_kv_transfer_unilateral."""
    is_mla = int(gpu_kv_format) in (_FMT_NL_X_NB_BS_HS, _FMT_NL_X_NBBS_ONE_HS)

    # MLA collapses to multi_layer_kv_transfer
    if is_mla:
        return multi_layer_kv_transfer(
            key_value, key_value_ptrs, slot_mapping,
            paged_memory_device, page_buffer_size,
            direction, gpu_kv_format,
        )

    if isinstance(key_value_ptrs, list):
        from lmcache.non_cuda_equivalents import (
            multi_layer_kv_transfer_unilateral as _fallback,
        )
        _fallback(
            key_value, key_value_ptrs, slot_mapping,
            paged_memory_device, page_buffer_size,
            direction, gpu_kv_format,
        )
        return

    scalars_per_token, kv_view = _get_scalars_and_view(key_value)

    num_layers = key_value.size(1)
    num_tokens = key_value.size(2)
    direction_int = 1 if direction == TransferDirection.D2H else 0

    kv_arg = kv_view if kv_view.is_xpu else kv_view.data_ptr()

    grid = (num_tokens, num_layers, 2)

    _multi_layer_kv_transfer_unilateral_kernel[grid](
        kv_arg,
        key_value_ptrs,
        slot_mapping,
        scalars_per_token,
        num_tokens,
        num_layers,
        page_buffer_size,
        DIRECTION=direction_int,
        BLOCK=_TRITON_BLOCK,
    )


def single_layer_kv_transfer(
    lmc_key_value_cache: torch.Tensor,
    vllm_key_value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
    token_major: bool = False,
):
    """XPU Triton-accelerated single_layer_kv_transfer.

    Transfers KV data between an LMCache buffer and one vLLM paged KV
    cache layer.  Falls back to the PyTorch path when either tensor is
    not on XPU.
    """
    fmt = int(gpu_kv_format)
    is_mla = fmt in (_FMT_NL_X_NB_BS_HS, _FMT_NL_X_NBBS_ONE_HS)
    is_hnd = fmt in (_FMT_NL_X_TWO_NB_NH_BS_HS, _FMT_NL_X_NB_TWO_NH_BS_HS)

    # Fall back for non-XPU tensors or unsupported formats
    if not vllm_key_value_cache.is_xpu:
        from lmcache.non_cuda_equivalents import (
            single_layer_kv_transfer as _fallback,
        )
        return _fallback(
            lmc_key_value_cache, vllm_key_value_cache, slot_mapping,
            direction, gpu_kv_format, token_major,
        )

    elem_size = lmc_key_value_cache.element_size()
    direction_int = 1 if direction == TransferDirection.D2H else 0

    if is_mla:
        # lmc: [num_tokens, head_size]
        # vllm: [num_blocks, block_size, head_size]
        num_tokens = lmc_key_value_cache.size(0)
        hidden = lmc_key_value_cache.size(1)
        block_size = vllm_key_value_cache.size(1)
        scalars_per_token = hidden * elem_size // 2
        head_size_i16 = 0

        lmc_view = lmc_key_value_cache.view(torch.int16)
        vllm_view = vllm_key_value_cache.view(torch.int16)

        lmc_stride = scalars_per_token
        lmc_value_offset = 0
        vllm_block_stride = block_size * scalars_per_token
        vllm_value_offset = 0

        lmc_arg = lmc_view if lmc_view.is_xpu else lmc_view.data_ptr()

        grid = (num_tokens, 1)
        _single_layer_kv_transfer_kernel[grid](
            lmc_arg, vllm_view, slot_mapping,
            lmc_stride, lmc_value_offset,
            vllm_block_stride, vllm_value_offset,
            scalars_per_token, block_size, head_size_i16,
            DIRECTION=direction_int, IS_MLA=True, IS_HND=False,
            BLOCK=_TRITON_BLOCK,
        )
    else:
        # Non-MLA
        is_two_major = fmt in (
            _FMT_NL_X_TWO_NB_BS_NH_HS,
            _FMT_NL_X_TWO_NB_NH_BS_HS,
        )
        if is_hnd:
            # [prefix, num_heads, block_size, head_size]
            num_heads = vllm_key_value_cache.size(2)
            block_size = vllm_key_value_cache.size(3)
            head_dim = vllm_key_value_cache.size(4)
        else:
            # [prefix, block_size, num_heads, head_size]
            block_size = vllm_key_value_cache.size(2)
            num_heads = vllm_key_value_cache.size(3)
            head_dim = vllm_key_value_cache.size(4)
        hidden = num_heads * head_dim
        scalars_per_token = hidden * elem_size // 2
        head_size_i16 = head_dim * elem_size // 2

        if token_major:
            # lmc: [num_tokens, 2, hidden]
            num_tokens = lmc_key_value_cache.size(0)
            lmc_stride = 2 * scalars_per_token
            lmc_value_offset = scalars_per_token
        else:
            # lmc: [2, num_tokens, hidden]
            num_tokens = lmc_key_value_cache.size(1)
            lmc_stride = scalars_per_token
            lmc_value_offset = num_tokens * scalars_per_token

        lmc_view = lmc_key_value_cache.view(torch.int16)
        vllm_view = vllm_key_value_cache.view(torch.int16)

        # vllm block stride and value offset in int16 units
        if is_two_major:
            # [2, num_blocks, ...] → value is num_blocks * block_size * hidden away
            num_blocks = vllm_key_value_cache.size(1)
            vllm_block_stride = block_size * scalars_per_token
            vllm_value_offset = num_blocks * block_size * scalars_per_token
        else:
            # [num_blocks, 2, ...] → value is block_size * hidden away
            vllm_block_stride = 2 * block_size * scalars_per_token
            vllm_value_offset = block_size * scalars_per_token

        lmc_arg = lmc_view if lmc_view.is_xpu else lmc_view.data_ptr()

        grid = (num_tokens, 2)
        _single_layer_kv_transfer_kernel[grid](
            lmc_arg, vllm_view, slot_mapping,
            lmc_stride, lmc_value_offset,
            vllm_block_stride, vllm_value_offset,
            scalars_per_token, block_size, head_size_i16,
            DIRECTION=direction_int, IS_MLA=False, IS_HND=is_hnd,
            BLOCK=_TRITON_BLOCK,
        )
