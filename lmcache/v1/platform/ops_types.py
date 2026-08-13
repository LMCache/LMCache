# SPDX-License-Identifier: Apache-2.0
"""Python-only types for the unified ``DeviceOps`` surface.

The native KV-format and transfer types live in :mod:`lmcache.lmcache_native`.
This module contains only the Python fallback types used by device operations.

The object-group transfer plan types (``StagingCopy``, ``LaunchVar``,
``BatchStep``, ``KernelGroupSpec``, and ``CBGroupSpec``) are shared by the
native and torch fallback executors.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    import lmcache.lmcache_native as lmcache_native


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
        engine_kv_format: lmcache_native.EngineKVFormat,
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
        engine_kv_format: lmcache_native.EngineKVFormat,
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
