# SPDX-License-Identifier: Apache-2.0
"""MUSA ops backend: the torch baseline with one native override.

:class:`MusaDeviceOps` overrides only ``multi_layer_block_kv_transfer`` to try
the native MUSA path first (when inputs are tensor-backed) and fall back to the
torch baseline otherwise. Every other op inherits the baseline. This ports the
former ``platform/musa/ops.py`` adapter into the unified :class:`DeviceOps`
hierarchy with zero behavior change.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.platform.base_device_ops import DeviceOps


def _tensor_list(value: object) -> list[torch.Tensor] | None:
    """Return ``value`` as ``list[torch.Tensor]`` when it is tensor-backed."""
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, torch.Tensor) for item in value):
        return None
    return value


class MusaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "musa"

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor,
        lmcache_objects_ptrs,
        block_ids,
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    ) -> None:
        """Native MUSA block transfer when tensor-backed; else torch baseline."""
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

        return super().multi_layer_block_kv_transfer(
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
