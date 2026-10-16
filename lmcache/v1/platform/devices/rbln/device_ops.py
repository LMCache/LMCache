# SPDX-License-Identifier: Apache-2.0
"""RBLN ops backend: block transfer tuned for the device's torch op ordering.

Every op except :meth:`RblnDeviceOps.multi_layer_block_kv_transfer` is
inherited from :class:`DeviceOps`, which routes to the pure torch
implementations in :mod:`lmcache.v1.platform.torch_ops`. That baseline is safe
on RBLN: ``lmcache_memcpy_async`` takes its tensor-mode branch for non-CUDA
devices, and the completion / event recorders degrade to immediate
publication, with ordering supplied by the transfer context's
``torch_dev.synchronize()``.

Block transfer is overridden for the op sequence, not for the layout. Chunks
keep LMCache's canonical token-major wire layout (``[2, L, T, H*D]``), so a
chunk written from an RBLN cache is byte-compatible with every other device --
the case that matters for cross-device KV sharing and PD disaggregation.

For HND, what the override replaces is the shared path's strided device
indexing: RBLN stores heads before block tokens, and the head<->token
transpose is hoisted to the host. The MLA layout (``NL_X_NB_BS_HS``, single
latent plane, no head axis) has no transpose to hoist; its RBLN sequence in
:mod:`kv_ops` exists for the DMA shape. The shared torch path issues one
``index_select`` / ``index_copy_`` per layer, which on torch-rbln means one
v2v submission plus one index read-back per layer and a fresh device
allocation per chunk, with a whole-layer CPU fallback behind each op should
the runtime reject a copy. The MLA sequence instead fans blocks in and out of
a persistent device staging buffer with one batch of contiguous block copies
(direct ``memcpy_v2v``, no index tensor, no fallback path) and crosses the
device boundary with one DMA per chunk.

Both layouts require the engine's KV caches to be real device tensors
(vLLM-RBLN: ``VLLM_RBLN_USE_DEVICE_TENSOR=1``). With the default compile-mode
allocation the per-layer tensors are ``meta`` and any transfer -- this
module's or the shared path's -- dies at the first host copy with "Cannot
copy out of meta tensor".
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar, cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.devices.rbln.kv_layout import (
    squeeze_singleton_axis,
    validate_mla_layers,
)
from lmcache.v1.platform.devices.rbln.kv_ops import (
    gather_blocks_to_chunk_hnd,
    gather_blocks_to_chunk_mla,
    scatter_chunk_to_blocks_hnd,
    scatter_chunk_to_blocks_mla,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


class RblnDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "rbln"

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
        """Move whole paged blocks between RBLN KV and token-major chunks.

        Args:
            paged_buffer_ptrs_tensor: Native per-layer KV tensors --
                ``[2, NB, NH, 1, BS, HS]`` (HND attention) or
                ``[NB, BS, HS]`` (MLA).
            lmcache_objects_ptrs: Staging chunks in the canonical token-major
                layout -- ``[2, L, T, H*D]`` (HND) or ``[L, T, HS]`` (MLA).
            block_ids: Flat paged-block IDs in chunk-token order.
            device: Device the transfer runs on. Unused; taken from the
                tensors.
            direction: ``D2H`` to store, ``H2D`` to retrieve.
            shape_desc: Paged-buffer shape descriptor.
            lmcache_chunk_size: Tokens per staging chunk.
            engine_kv_format: Engine KV layout. Classified by its
                ``KVFormatSpec`` layout facts: a per-layer two-major HND
                layout or a per-layer ``[NB, BS, HS]`` MLA layout.
            skip_prefix_n_blocks: Leading blocks neither read nor written.

        Raises:
            ValueError: If the operands are not tensor lists, the format's
                layout facts match neither supported layout, a paged tensor
                does not match its layout's shape, or the direction is
                unknown.
        """
        del device  # taken from the operands
        if isinstance(paged_buffer_ptrs_tensor, torch.Tensor) or not all(
            isinstance(obj, torch.Tensor) for obj in lmcache_objects_ptrs
        ):
            raise ValueError(
                "RBLN block transfer requires tensor operands; the pointer "
                "form is only produced for compiled backends, and RBLN has "
                "no compiled block-transfer extension in tree."
            )
        # Imported lazily, not at module scope: importing the specs package
        # pulls in ``lmcache.v1.gpu_connector``, which reads
        # ``lmcache.device_ops`` -- still being resolved to this module while
        # ``lmcache`` initializes on an RBLN host (an ImportError otherwise).
        # First Party
        from lmcache.v1.gpu_connector.kv_format.specs.base import describe_shape
        from lmcache.v1.gpu_connector.kv_format.specs.registry import (
            get_spec_class,
        )

        # Dispatch on the format's layout facts rather than its identity, so
        # any per-layer layout the op sequences below address is accepted.
        format_spec = get_spec_class(engine_kv_format)
        is_mla = (
            format_spec.is_layer_list
            and format_spec.is_mla
            and not format_spec.is_pbs_fused
            and not format_spec.is_kv_second_tuple
            and not format_spec.requires_blocked_scale_repacking
        )
        is_hnd = (
            format_spec.is_layer_list
            and format_spec.is_hnd
            and format_spec.is_two_major
            and not format_spec.is_fused_packed
        )
        if not (is_mla or is_hnd):
            raise ValueError(
                "RBLN block transfer supports only per-layer [NB, BS, HS] MLA "
                "and per-layer two-major HND layouts; got "
                f"{engine_kv_format!r} ({describe_shape(engine_kv_format)})"
            )

        # Per-format addressing only; the chunk/block bookkeeping below is
        # shared. HND keeps the singleton axis the RBLN attention backend
        # requires -- drop it here, where the bytes are actually addressed.
        # The squeeze is strict, so an HND layout without that axis fails
        # here. MLA has nothing to squeeze; its rank is pinned instead.
        layers = cast("list[torch.Tensor]", list(paged_buffer_ptrs_tensor))
        paged_layers = (
            validate_mla_layers(layers) if is_mla else squeeze_singleton_axis(layers)
        )
        chunks = cast("list[torch.Tensor]", list(lmcache_objects_ptrs))
        flat_blocks = (
            [int(b) for b in block_ids.tolist()]
            if isinstance(block_ids, torch.Tensor)
            else [int(b) for b in block_ids]
        )

        block_size = int(shape_desc.bs)
        if block_size <= 0 or lmcache_chunk_size % block_size != 0:
            raise ValueError(
                "lmcache_chunk_size must be a positive multiple of shape_desc.bs"
            )
        blocks_per_chunk = lmcache_chunk_size // block_size

        is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
        if not is_d2h and int(direction) != int(lmcache_native.TransferDirection.H2D):
            raise ValueError(f"Unsupported transfer direction: {direction!r}")

        consumed = 0
        for chunk_idx, chunk in enumerate(chunks):
            blocks = flat_blocks[
                chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
            ]
            if not blocks:
                break
            if is_d2h:
                if is_mla:
                    gather_blocks_to_chunk_mla(paged_layers, blocks, chunk)
                else:
                    gather_blocks_to_chunk_hnd(paged_layers, blocks, chunk)
            else:
                # The prefix skip is global across the transfer; translate it
                # into this chunk's local block offset.
                local_skip = min(len(blocks), max(0, skip_prefix_n_blocks - consumed))
                if is_mla:
                    scatter_chunk_to_blocks_mla(
                        paged_layers, blocks, chunk, skip_prefix_n_blocks=local_skip
                    )
                else:
                    scatter_chunk_to_blocks_hnd(
                        paged_layers, blocks, chunk, skip_prefix_n_blocks=local_skip
                    )
            consumed += len(blocks)
