# SPDX-License-Identifier: Apache-2.0
"""Shared, device-agnostic ops *types* for the unified ``DeviceOps`` surface.

These types (``TransferDirection``, ``EngineKVFormat``, ``GPUKVFormat``,
``PageBufferShapeDesc``, ``set_shape_desc_dtype``) were migrated verbatim from
the former ``lmcache.python_ops_fallback`` module. They are deliberately
self-contained (only ``torch`` + ``enum``) so call sites can import them
without pulling in the heavier torch-baseline op implementations in
:mod:`lmcache.v1.platform.torch_ops`.

The object-group transfer plan types (``StagingCopy``, ``LaunchVar``,
``BatchStep``, ``KernelGroupSpec``, and ``CBGroupSpec``) are shared by the
native and torch fallback executors.
"""

# Future
from __future__ import annotations

# Standard
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Third Party
    import torch


class TransferDirection(IntEnum):
    """Specifies the direction of a memory transfer.

    Inherits from IntEnum so that members compare equal to plain ints
    and to native pybind11 enum members with the same integer value.
    Several call sites (and the fallback ops themselves) use
    ``int(direction)`` to compare across backend / fallback boundaries.
    """

    H2D = 0
    D2H = 1


class EngineKVFormat(IntEnum):
    """Enumeration of different engine KV cache memory layouts."""

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

    # used by: vLLM non-MLA flash attention (HND layout)
    NL_X_TWO_NB_NH_BS_HS = 6

    # used by: vLLM non-MLA flash infer (HND layout)
    NL_X_NB_TWO_NH_BS_HS = 7

    # used by: TRT-LLM cross-layer (HND layout)
    NB_NL_TWO_NH_BS_HS = 8

    # used by: SGLang MHA via the MP daemon path
    TWO_X_NL_X_NB_BS_NH_HS = 9

    # DEPRECATED: superseded by NL_X_NB_NH_BS_CS; no longer produced by
    # detection.
    # used by: vLLM non-MLA blocks-first attention with K/V fused into the
    # trailing dim. Per-layer physical shape
    # [num_blocks, num_heads, block_size, 2, head_size] -- the K/V "2" axis is
    # second-to-last, recovered by splitting the fused [..., 2 * head_size].
    NL_X_NB_NH_BS_TWO_HS = 10

    # DEPRECATED: superseded by NL_X_NB_BS_NH_CS; no longer produced by
    # detection.
    # used by: vLLM non-MLA blocks-first attention (NHD layout) with K/V fused
    # into the trailing dim. Per-layer physical shape
    # [num_blocks, block_size, num_heads, 2, head_size] -- like
    # NL_X_NB_NH_BS_TWO_HS but tokens before heads.
    NL_X_NB_BS_NH_TWO_HS = 11

    # used by: vLLM non-MLA blocks-first attention (HND layout, unified KV
    # cache) with K/V fused into the trailing content dim. Per-layer physical
    # shape [num_blocks, num_heads, block_size, content_size].
    NL_X_NB_NH_BS_CS = 12

    # used by: vLLM non-MLA blocks-first attention (NHD layout, unified KV
    # cache) with K/V fused into the trailing content dim. Per-layer physical
    # shape [num_blocks, block_size, num_heads, content_size] -- like
    # NL_X_NB_NH_BS_CS but tokens before heads.
    NL_X_NB_BS_NH_CS = 13

    # vLLM DSA indexer k-cache [NB,BS,132] u8, paged [BSxvals][BSxscales];
    # c_ops only (no pure-torch transfer path)
    NL_X_NB_BSV_BSS = 14


# Backward-compat alias
GPUKVFormat = EngineKVFormat


def is_cross_layer(engine_kv_format: EngineKVFormat) -> bool:
    """Return True if all layers are fused into one tensor.

    Mirrors the C++ ``is_cross_layer`` predicate in ``csrc/engine_kv_format.h``.
    """
    return engine_kv_format in (
        EngineKVFormat.NB_NL_TWO_BS_NH_HS,
        EngineKVFormat.NB_NL_TWO_NH_BS_HS,
    )


def is_kv_list(engine_kv_format: EngineKVFormat) -> bool:
    """Return True if keys and values are two separate top-level lists.

    Mirrors the C++ ``is_kv_list`` predicate in ``csrc/engine_kv_format.h``.
    """
    return engine_kv_format in (
        EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS,
        EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
    )


def is_layer_list(engine_kv_format: EngineKVFormat) -> bool:
    """Return True if there is one list entry per layer.

    Mirrors the C++ ``is_layer_list`` predicate in ``csrc/engine_kv_format.h``.
    """
    return engine_kv_format in (
        EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
        EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
        EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
        EngineKVFormat.NL_X_NB_NH_BS_CS,
        EngineKVFormat.NL_X_NB_BS_NH_CS,
        EngineKVFormat.NL_X_NB_BSV_BSS,
    )


def is_mla(engine_kv_format: EngineKVFormat) -> bool:
    """Return True for MLA formats (single latent KV head, no K/V split).

    Mirrors the C++ ``is_mla`` predicate in ``csrc/engine_kv_format.h``.
    """
    return engine_kv_format in (
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
        EngineKVFormat.NL_X_NB_BSV_BSS,
    )


class PageBufferShapeDesc:
    """Python stand-in for the C++ ``PageBufferShapeDesc`` struct.

    Mirrors the pybind ``def_readwrite`` attributes in ``csrc/pybind.cpp``
    so non-CUDA code paths can construct and inspect shape descriptors
    without the compiled extension.

    ``block_stride_elems`` captures the *physical* per-block step in
    element units (= ``tensor.stride(0)``). For a tightly-packed paged
    buffer it equals ``bs * kv_size * nh * hs`` (non-MLA) /
    ``bs * nh * hs`` (MLA); for a vLLM KV pool where a group's row is
    padded to the pool's maximum row width (e.g. DeepSeek V4 compressor
    / indexer caches), it is strictly larger. Downstream kernels must
    use this value instead of recomputing a "tight" stride from the
    logical shape, otherwise they'll skip into the next block's padding
    region and read/write the wrong slots.
    """

    __slots__ = (
        "kv_size",
        "nl",
        "nb",
        "bs",
        "nh",
        "hs",
        "element_size",
        "block_stride_elems",
        "dtype",
    )

    def __init__(self) -> None:
        self.kv_size: int = 0
        self.nl: int = 0
        self.nb: int = 0
        self.bs: int = 0
        self.nh: int = 0
        self.hs: int = 0
        self.element_size: int = 0
        # 0 means "unset — fall back to tight stride"; any downstream
        # consumer that needs exact addressing must check this.
        self.block_stride_elems: int = 0
        self.dtype: torch.dtype | None = None


class StagingCopy:
    """One host-device copy in an object-group transfer plan."""

    __slots__ = ("_dest", "_src", "_nbytes", "_host_offset")

    def __init__(
        self,
        dest: int | torch.Tensor,
        src: int | torch.Tensor,
        nbytes: int,
        host_offset: int,
    ) -> None:
        self._dest = dest
        self._src = src
        self._nbytes = int(nbytes)
        self._host_offset = int(host_offset)


class LaunchVar:
    """One object-group kernel launch in a transfer plan."""

    __slots__ = (
        "_group_idx",
        "_block_ids_offset",
        "_total_blocks",
        "_num_objects",
        "_skip_prefix_n_blocks",
    )

    def __init__(
        self,
        group_idx: int,
        block_ids_offset: int,
        total_blocks: int,
        num_objects: int,
        skip_prefix_n_blocks: int,
    ) -> None:
        self._group_idx = int(group_idx)
        self._block_ids_offset = int(block_ids_offset)
        self._total_blocks = int(total_blocks)
        self._num_objects = int(num_objects)
        self._skip_prefix_n_blocks = int(skip_prefix_n_blocks)


class BatchStep:
    """Ordered staging copies and launches for one transfer-plan step."""

    __slots__ = ("_staging", "_launches")

    def __init__(self, staging: list[StagingCopy], launches: list[LaunchVar]) -> None:
        self._staging = list(staging)
        self._launches = list(launches)


class KernelGroupSpec:
    """Invariant geometry and buffers for an object-group kernel."""

    __slots__ = (
        "_paged_buffer_ptrs",
        "_lmcache_objects_ptrs",
        "_shape_desc",
        "_lmcache_chunk_size",
        "_engine_kv_format",
        "_block_ids_base",
        "_block_ids_capacity",
    )

    def __init__(
        self,
        paged_buffer_ptrs: int | torch.Tensor | list[int | torch.Tensor],
        lmcache_objects_ptrs: list[int | torch.Tensor],
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: EngineKVFormat,
        block_ids_base: int | torch.Tensor,
        block_ids_capacity: int,
    ) -> None:
        self._paged_buffer_ptrs = paged_buffer_ptrs
        self._lmcache_objects_ptrs = list(lmcache_objects_ptrs)
        self._shape_desc = shape_desc
        self._lmcache_chunk_size = int(lmcache_chunk_size)
        self._engine_kv_format = engine_kv_format
        self._block_ids_base = block_ids_base
        self._block_ids_capacity = int(block_ids_capacity)


class CBGroupSpec:
    """Invariant geometry and buffers for a CacheBlend kernel group."""

    __slots__ = (
        "_paged_kv_ptrs",
        "_temp_buffer_ptrs",
        "_num_layers",
        "_slot_tokens",
        "_hidden_elems",
        "_element_size",
        "_engine_kv_format",
        "_page_buffer_size",
        "_block_size",
        "_head_size",
        "slot_mapping_base",
        "slot_mapping_capacity",
        "_cos_sin_cache",
        "_rot_dim",
        "_rope_num_kv_heads",
        "_rope_head_stride",
        "_key_scalar_type",
        "_is_neox",
        "_rope_base_offset",
    )

    def __init__(
        self,
        paged_kv_ptrs: int | torch.Tensor | list[int | torch.Tensor],
        temp_buffer_ptrs: list[int | torch.Tensor],
        num_layers: int,
        slot_tokens: int,
        hidden_elems: int,
        element_size: int,
        engine_kv_format: EngineKVFormat,
        page_buffer_size: int,
        block_size: int,
        head_size: int,
        slot_mapping_base: int | torch.Tensor,
        slot_mapping_capacity: int,
        cos_sin_cache: int | torch.Tensor,
        rot_dim: int,
        rope_num_kv_heads: int,
        rope_head_stride: int,
        key_scalar_type: int,
        is_neox: bool,
        rope_base_offset: int = 0,
    ) -> None:
        self._paged_kv_ptrs = paged_kv_ptrs
        self._temp_buffer_ptrs = list(temp_buffer_ptrs)
        self._num_layers = int(num_layers)
        self._slot_tokens = int(slot_tokens)
        self._hidden_elems = int(hidden_elems)
        self._element_size = int(element_size)
        self._engine_kv_format = engine_kv_format
        self._page_buffer_size = int(page_buffer_size)
        self._block_size = int(block_size)
        self._head_size = int(head_size)
        self.slot_mapping_base = slot_mapping_base
        self.slot_mapping_capacity = int(slot_mapping_capacity)
        self._cos_sin_cache = cos_sin_cache
        self._rot_dim = int(rot_dim)
        self._rope_num_kv_heads = int(rope_num_kv_heads)
        self._rope_head_stride = int(rope_head_stride)
        self._key_scalar_type = int(key_scalar_type)
        self._is_neox = bool(is_neox)
        self._rope_base_offset = int(rope_base_offset)


def set_shape_desc_dtype(shape_desc: Any, dtype: torch.dtype) -> None:
    """Best-effort ``shape_desc.dtype = dtype``.

    The pure-Python ``PageBufferShapeDesc`` exposes a ``dtype`` slot so
    the CPU fallback kernel can disambiguate float16 vs bfloat16 (both
    have ``element_size == 2``). The pybind C++ struct in
    ``csrc/pybind.cpp`` has no such field; assignment raises
    ``AttributeError`` and is silently swallowed here so call sites
    don't need to branch on the active backend.

    Args:
        shape_desc: A ``PageBufferShapeDesc`` instance (either the
            pure-Python fallback or the C++ pybind struct).
        dtype: The torch dtype to assign.
    """
    try:
        shape_desc.dtype = dtype
    except AttributeError:
        pass
