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

HND transfers use chiplet-sharded device staging and a compiled head/token
swap in both directions. A store copies directly from device staging to
the canonical host chunk. Store and restore buffers are independent,
and every compiled swap is serialized. MLA keeps its existing path.

All layouts require the engine's KV caches to be real device tensors
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
from lmcache.v1.platform.devices.rbln.hnd_ops import transfer_hnd
from lmcache.v1.platform.devices.rbln.kv_layout import (
    squeeze_singleton_axis,
    squeeze_singleton_axis_kv_first,
    validate_mla_layers,
)
from lmcache.v1.platform.devices.rbln.kv_ops import (
    gather_blocks_to_chunk_mla,
    scatter_chunk_to_blocks_mla,
)
from lmcache.v1.platform.devices.rbln.pin_memory import skip_host_registration
from lmcache.v1.platform.devices.rbln.staging import hnd_staging
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


class RblnDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "rbln"

    def alloc_shm_pinned_ptr(self, size: int, shm_name: str = "") -> int:
        """Create the server's SHM pool without registering it on an NPU."""
        with skip_host_registration():
            return super().alloc_shm_pinned_ptr(size, shm_name)

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

        The layout is picked by its ``KVFormatSpec`` layout facts. HND uses
        device staging and a compiled swap in both directions. It validates
        the requirements and reserves device buffers before any HND copy.

        Args:
            paged_buffer_ptrs_tensor: Per-layer KV tensors --
                ``[NB, 2, NH, 1, BS, HS]`` (HND), ``[2, NB, NH, 1, BS, HS]``
                (K/V-first HND), or ``[NB, BS, HS]`` (MLA).
            lmcache_objects_ptrs: Staging chunks in the canonical token-major
                layout -- ``[2, L, T, H*D]`` (HND) or ``[L, T, HS]`` (MLA).
            block_ids: Flat paged-block IDs in chunk-token order.
            device: Device the transfer runs on. Unused; taken from the
                tensors.
            direction: ``D2H`` to store, ``H2D`` to retrieve.
            shape_desc: Paged-buffer shape descriptor.
            lmcache_chunk_size: Tokens per staging chunk.
            engine_kv_format: Engine KV layout.
            skip_prefix_n_blocks: Leading blocks to skip during restore.

        Raises:
            ValueError: If the operands are not tensor lists, the format's
                layout facts match none of the supported layouts, a paged
                tensor does not match its layout's shape, the chunk size is
                not a positive block multiple, operands are not contiguous,
                chunks do not match the cache geometry, the direction is unknown,
                or HND inputs are not bf16 with head size a multiple of 64.
            RuntimeError: If torch-rbln lacks the compiled HND copy, or staging
                allocation, placement or a transfer fails.
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
        from lmcache.v1.common.kv_format.specs.base import describe_shape
        from lmcache.v1.common.kv_format.specs.registry import (
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
            and not format_spec.is_fused_packed
        )
        if not (is_mla or is_hnd):
            raise ValueError(
                "RBLN block transfer supports only per-layer [NB, BS, HS] MLA "
                "and per-layer HND layouts; got "
                f"{engine_kv_format!r} ({describe_shape(engine_kv_format)})"
            )

        # Each tuned layout has a ``prepare`` that turns the engine's per-layer
        # tensors into the views its gather/scatter index, failing loudly on
        # a layout drift. The HND singleton axis the RBLN attention backend
        # requires is squeezed there, where the bytes are addressed.
        if is_mla:
            prepare = validate_mla_layers
        elif format_spec.is_two_major:
            prepare = squeeze_singleton_axis_kv_first
        else:
            prepare = squeeze_singleton_axis

        # The chunk/block bookkeeping below is shared by every tuned layout.
        paged_layers = prepare(
            cast("list[torch.Tensor]", list(paged_buffer_ptrs_tensor))
        )
        chunks = cast("list[torch.Tensor]", list(lmcache_objects_ptrs))
        flat_blocks = (
            [int(b) for b in block_ids.tolist()]
            if isinstance(block_ids, torch.Tensor)
            else [int(b) for b in block_ids]
        )

        block_size = int(shape_desc.bs)
        if (
            block_size <= 0
            or lmcache_chunk_size <= 0
            or lmcache_chunk_size % block_size != 0
        ):
            raise ValueError(
                "lmcache_chunk_size must be a positive multiple of shape_desc.bs"
            )
        blocks_per_chunk = lmcache_chunk_size // block_size

        is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
        if not is_d2h and int(direction) != int(lmcache_native.TransferDirection.H2D):
            raise ValueError(f"Unsupported transfer direction: {direction!r}")

        if is_hnd:
            if any(layer.dtype != torch.bfloat16 for layer in paged_layers + chunks):
                raise ValueError("RBLN HND transfer requires bf16 KV and chunks")
            if any(layer.size(-1) % 64 != 0 for layer in paged_layers):
                raise ValueError(
                    "RBLN HND transfer requires head size a multiple of 64"
                )
            try:
                compiled_copy = torch._C._dispatch_has_kernel_for_dispatch_key(
                    "torch_rbln::copy_strided_view", "PrivateUse1"
                )
            except RuntimeError:
                compiled_copy = False
            if not compiled_copy:
                raise RuntimeError(
                    "RBLN HND transfer requires torch_rbln::copy_strided_view "
                    "registered for PrivateUse1"
                )
            if any(not layer.is_contiguous() for layer in paged_layers):
                raise ValueError("RBLN HND transfer requires contiguous paged layers")
            first = paged_layers[0]
            heads, tokens_per_block, head_size = first.shape[2:]
            expected = (
                2,
                len(paged_layers),
                blocks_per_chunk * tokens_per_block,
                heads * head_size,
            )
            if len(chunks) * blocks_per_chunk < len(flat_blocks) or any(
                tuple(chunk.shape) != expected or not chunk.is_contiguous()
                for chunk in chunks
            ):
                raise ValueError(
                    f"RBLN HND transfer requires contiguous chunks shaped {expected}"
                )
            staged = hnd_staging(
                first.device,
                first.dtype,
                len(paged_layers),
                heads,
                tokens_per_block,
                head_size,
                store=is_d2h,
            )
            layers = (
                paged_layers
                if format_spec.is_two_major
                else [layer.transpose(0, 1) for layer in paged_layers]
            )
            transfer_hnd(
                layers,
                flat_blocks,
                chunks,
                blocks_per_chunk,
                skip_prefix_n_blocks,
                staged,
                store=is_d2h,
            )
            return

        consumed = 0
        for chunk_idx, chunk in enumerate(chunks):
            blocks = flat_blocks[
                chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
            ]
            if not blocks:
                break
            if is_d2h:
                gather_blocks_to_chunk_mla(paged_layers, blocks, chunk)
            else:
                # The prefix skip is global across the transfer; translate it
                # into this chunk's local block offset.
                local_skip = min(len(blocks), max(0, skip_prefix_n_blocks - consumed))
                scatter_chunk_to_blocks_mla(paged_layers, blocks, chunk, local_skip)
            consumed += len(blocks)
