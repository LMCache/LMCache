# SPDX-License-Identifier: Apache-2.0
"""Neuron ops backend: block transfer without device-side indexing.

Everything except :meth:`NeuronDeviceOps.multi_layer_block_kv_transfer` is
inherited from :class:`DeviceOps`. See :mod:`.kv_ops` for why the shared torch
path cannot be used on Neuron.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar, cast

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.devices.neuron.kv_ops import (
    gather_blocks_to_chunk,
    scatter_chunk_to_blocks,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

#: Rank of a per-layer ``[2, NB, NH, BS, HS]`` tensor.
_PAGED_RANK = 5


def _require_supported_layout(
    engine_kv_format: lmcache_native.EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
    paged_layers: list[torch.Tensor],
) -> None:
    """Check the layout is per-layer, two-major HND, ``[2, NB, NH, BS, HS]``.

    The check reads the format's structural properties rather than matching one
    enum value. The rank check excludes ``NL_X_TWO_NB_NH_ONE_BS_HS``, which has
    the same properties but an extra singleton axis.

    Args:
        engine_kv_format: Engine KV format of ``paged_layers``.
        shape_desc: Paged-buffer shape descriptor.
        paged_layers: Per-layer KV tensors.

    Raises:
        ValueError: If the format or tensor rank is unsupported.
    """
    # Imported lazily, not at module scope: the specs package reads
    # ``lmcache.device_ops``, which is resolved only after this module loads.
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec_class

    format_spec = get_spec_class(engine_kv_format)
    supported = (
        format_spec.is_layer_list
        and format_spec.is_hnd
        and format_spec.is_two_major
        and not format_spec.is_mla
        and not format_spec.is_fused_packed
        and not format_spec.is_kv_second_tuple
        and shape_desc.kv_size == 2
        and all(layer.dim() == _PAGED_RANK for layer in paged_layers)
    )
    if not supported:
        raise ValueError(
            "Neuron block transfer requires a per-layer, two-major HND KV "
            f"layout [2, NB, NH, BS, HS]; got {engine_kv_format!r}"
        )


class NeuronDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "neuron"

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: "torch.Tensor | list[torch.Tensor]",
        lmcache_objects_ptrs: "list[int] | list[torch.Tensor]",
        block_ids: "torch.Tensor | list[int]",
        device: "torch.device | str",
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Move whole paged blocks between Neuron KV and token-major chunks.

        Args:
            paged_buffer_ptrs_tensor: Per-layer KV tensors, ``[2, NB, NH, BS, HS]``.
            lmcache_objects_ptrs: Chunks shaped ``[2, L, T, NH*HS]``.
            block_ids: Flat block ids in chunk-token order.
            device: Unused; taken from the tensors.
            direction: ``D2H`` to store, ``H2D`` to retrieve.
            shape_desc: Paged-buffer shape descriptor.
            lmcache_chunk_size: Tokens per chunk.
            engine_kv_format: A per-layer, two-major HND format with 5-D
                tensors, i.e. ``NL_X_TWO_NB_NH_BS_HS``.
            skip_prefix_n_blocks: Leading blocks neither read nor written.

        Raises:
            ValueError: If the operands are pointers, the format or direction
                is unsupported, or the chunk size is not a block multiple.
        """
        del device
        if isinstance(paged_buffer_ptrs_tensor, torch.Tensor) or not all(
            isinstance(obj, torch.Tensor) for obj in lmcache_objects_ptrs
        ):
            raise ValueError("Neuron block transfer requires tensor operands")
        paged_layers = list(paged_buffer_ptrs_tensor)
        _require_supported_layout(engine_kv_format, shape_desc, paged_layers)
        block_size = int(shape_desc.bs)
        if block_size <= 0 or lmcache_chunk_size % block_size != 0:
            raise ValueError(
                "lmcache_chunk_size must be a positive multiple of shape_desc.bs"
            )
        is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
        if not is_d2h and int(direction) != int(lmcache_native.TransferDirection.H2D):
            raise ValueError(f"Unsupported transfer direction: {direction!r}")

        chunks = cast("list[torch.Tensor]", list(lmcache_objects_ptrs))
        flat_blocks = (
            [int(b) for b in block_ids.tolist()]
            if isinstance(block_ids, torch.Tensor)
            else [int(b) for b in block_ids]
        )
        blocks_per_chunk = lmcache_chunk_size // block_size

        consumed = 0
        for chunk_idx, chunk in enumerate(chunks):
            blocks = flat_blocks[
                chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
            ]
            if not blocks:
                break
            if is_d2h:
                gather_blocks_to_chunk(paged_layers, blocks, chunk)
            else:
                # The prefix skip is global; translate it to this chunk.
                local_skip = min(len(blocks), max(0, skip_prefix_n_blocks - consumed))
                scatter_chunk_to_blocks(
                    paged_layers, blocks, chunk, skip_prefix_n_blocks=local_skip
                )
            consumed += len(blocks)
