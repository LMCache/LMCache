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

HND transfers use device staging split across the chiplets and a compiled
head/token swap in both directions; MLA transfers stage one block of every
layer on the device. Each geometry's staging, one set per direction, and
HND's swap programs are prepared once, on the thread that drives the model,
through :meth:`RblnDeviceOps.prepare_kv_transfer`; stores take turns on the
store set. A transfer of a geometry that was not prepared fails before
copying.

All layouts require the engine's KV caches to be real device tensors
(vLLM-RBLN: ``VLLM_RBLN_USE_DEVICE_TENSOR=1``). With the default compile-mode
allocation the per-layer tensors are ``meta`` and any transfer -- this
module's or the shared path's -- dies at the first host copy with "Cannot
copy out of meta tensor".
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, ClassVar, cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.devices.rbln.kv_layout import (
    squeeze_singleton_axis,
    squeeze_singleton_axis_kv_first,
    validate_mla_layers,
)
from lmcache.v1.platform.devices.rbln.kv_ops import (
    HndTransfer,
    MlaTransfer,
    prepare_transfers,
    transfer,
)
from lmcache.v1.platform.devices.rbln.pin_memory import skip_host_registration
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.kv_format.specs.base import KVFormatSpec

logger = init_logger(__name__)


def _layout(
    engine_kv_format: lmcache_native.EngineKVFormat,
) -> tuple[type[KVFormatSpec], bool]:
    """Return the format's spec and whether it is per-layer MLA (else HND).

    Dispatches on the format's layout facts rather than its identity, so any
    per-layer layout the op sequences address is accepted.

    Raises:
        ValueError: If the format is neither per-layer MLA nor per-layer HND.
    """
    # Imported lazily, not at module scope: importing the specs package
    # pulls in ``lmcache.v1.gpu_connector``, which reads
    # ``lmcache.device_ops`` -- still being resolved to this module while
    # ``lmcache`` initializes on an RBLN host (an ImportError otherwise).
    # First Party
    from lmcache.v1.common.kv_format.specs.base import describe_shape
    from lmcache.v1.common.kv_format.specs.registry import get_spec_class

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
    return format_spec, is_mla


def _paged_layers(
    paged_buffers: list[torch.Tensor], format_spec: type[KVFormatSpec], is_mla: bool
) -> list[torch.Tensor]:
    """Return blocks-first views the transfers index, failing on layout drift.

    MLA layers are validated as they are (``[NB, BS, HS]``). HND loses the
    singleton axis the RBLN attention backend requires, where the bytes are
    addressed, and K/V-first layers are viewed blocks-first, so every HND
    layer reads ``[NB, 2, NH, BS, HS]``. Neither view moves bytes.
    """
    if is_mla:
        return validate_mla_layers(paged_buffers)
    if format_spec.is_two_major:
        return [
            layer.transpose(0, 1)
            for layer in squeeze_singleton_axis_kv_first(paged_buffers)
        ]
    return squeeze_singleton_axis(paged_buffers)


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
        device staging and a compiled swap in both directions, MLA one block
        of device staging per layer; :meth:`prepare_kv_transfer` prepares
        both. It validates the operands and finds the prepared staging before
        any copy.

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
                chunks do not match the cache geometry, or the direction is
                unknown.
            RuntimeError: If :meth:`prepare_kv_transfer` did not prepare this
                geometry, or a transfer fails.
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
        format_spec, is_mla = _layout(engine_kv_format)
        paged_layers = _paged_layers(
            cast("list[torch.Tensor]", list(paged_buffer_ptrs_tensor)),
            format_spec,
            is_mla,
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

        first = paged_layers[0]
        if is_mla:
            _, tokens_per_block, head_size = first.shape
            expected: tuple[int, ...] = (
                len(paged_layers),
                blocks_per_chunk * tokens_per_block,
                head_size,
            )
        else:
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
                f"RBLN block transfer requires contiguous chunks shaped {expected}"
            )
        transfer(
            paged_layers,
            flat_blocks,
            chunks,
            blocks_per_chunk,
            skip_prefix_n_blocks,
            store=is_d2h,
        )

    def prepare_kv_transfer(
        self,
        paged_buffers: list[torch.Tensor],
        engine_kv_format: lmcache_native.EngineKVFormat,
    ) -> None:
        """Prepare everything transfers of these layers need, on this thread.

        HND transfers swap head and token axes with a compiled program that
        torch-rbln would otherwise build the first time a transfer runs.
        Stores run on LMCache's commit workers, where that build races the
        model thread on torch's process-wide export state and stalls the
        forward. This allocates the geometry's device staging, one set per
        direction, and builds and keeps HND's swap programs on it, so
        transfers never allocate staging or build a program; MLA has no
        program. Transfers of a geometry that was not prepared here fail.
        Call it on the engine's model-driving thread once its KV caches are
        final, and before the engine sizes its KV cache so the staging is
        counted. Preparing a geometry again does nothing.

        Args:
            paged_buffers: Per-layer KV tensors of one KV cache group, as a
                transfer receives them.
            engine_kv_format: Their engine KV layout.

        Raises:
            ValueError: If the layout is neither per-layer MLA nor per-layer
                HND, or a layer does not match its layout's shape; transfers of
                it would fail the same way.
            RuntimeError: If allocating staging or building a swap fails,
                including for HND KV torch-rbln builds no swap program for:
                HND needs bf16 KV with a head size a multiple of 64.
        """
        format_spec, is_mla = _layout(engine_kv_format)
        paged_layers = _paged_layers(list(paged_buffers), format_spec, is_mla)
        prepare_transfers(paged_layers, MlaTransfer if is_mla else HndTransfer)
