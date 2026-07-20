# SPDX-License-Identifier: Apache-2.0
"""MUSA ops backend assembled into ``lmcache.c_ops`` at import time.

The package initializer merges this module over ``python_ops_fallback`` when
MUSA is the active device. Functions not defined here continue to use the
generic Python fallback implementation.
"""

# Future
from __future__ import annotations

# Third Party
import torch

# First Party
import lmcache.python_ops_fallback as py_ops


def _tensor_list(value: object) -> list[torch.Tensor] | None:
    """Return ``value`` as ``list[torch.Tensor]`` when it is tensor-backed."""
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, torch.Tensor) for item in value):
        return None
    return value


def multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: torch.Tensor | list,
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: py_ops.TransferDirection,
    shape_desc: py_ops.PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: py_ops.EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """MUSA block-based multi-layer KV transfer.

    This function intentionally mirrors ``lmcache.c_ops`` and
    ``python_ops_fallback`` so upper layers can call ``lmcache.c_ops`` without
    caring which device backend was selected during package initialization.

    Args:
        paged_buffer_ptrs_tensor: Paged buffer pointers or normalized tensors.
        lmcache_objects_ptrs: LMCache object pointers or chunk tensors.
        block_ids: Ordered engine block IDs for the transfer.
        device: Target device for the transfer.
        direction: Transfer direction (H2D or D2H).
        shape_desc: Shape descriptor of the page buffer.
        lmcache_chunk_size: Chunk size of LMCache objects.
        engine_kv_format: Engine KV cache format.
        skip_prefix_n_blocks: Number of leading blocks to skip.

    Returns:
        None

    Raises:
        ValueError: Propagated from the Python fallback when inputs are invalid.
        TypeError: Propagated from the Python fallback when input types are invalid.
    """
    # First Party
    from lmcache.v1.platform.musa.native_kv_transfer import (
        try_native_multi_layer_block_kv_transfer,
    )

    object_tensors = _tensor_list(lmcache_objects_ptrs)
    if object_tensors is not None and try_native_multi_layer_block_kv_transfer(
        paged_layers=paged_buffer_ptrs_tensor,
        object_tensors=object_tensors,
        block_ids=block_ids,
        direction=direction,
        shape_desc=shape_desc,
        lmcache_chunk_size=lmcache_chunk_size,
        engine_kv_format=engine_kv_format,
        skip_prefix_n_blocks=skip_prefix_n_blocks,
    ):
        return

    py_ops.multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor,
        lmcache_objects_ptrs,
        block_ids,
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    )
