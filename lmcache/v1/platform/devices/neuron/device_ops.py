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

#: vllm-neuron's per-layer layout, ``[2, NB, NH, BS, HS]``.
_HND_FORMAT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS


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
            engine_kv_format: Must be ``NL_X_TWO_NB_NH_BS_HS``.
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
        if int(engine_kv_format) != int(_HND_FORMAT):
            raise ValueError(
                f"Neuron block transfer supports only {_HND_FORMAT.name}; "
                f"got {engine_kv_format!r}"
            )
        block_size = int(shape_desc.bs)
        if block_size <= 0 or lmcache_chunk_size % block_size != 0:
            raise ValueError(
                "lmcache_chunk_size must be a positive multiple of shape_desc.bs"
            )
        is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
        if not is_d2h and int(direction) != int(lmcache_native.TransferDirection.H2D):
            raise ValueError(f"Unsupported transfer direction: {direction!r}")

        paged_layers = list(paged_buffer_ptrs_tensor)
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
