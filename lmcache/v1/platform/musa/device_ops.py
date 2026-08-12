# SPDX-License-Identifier: Apache-2.0
"""MUSA ops for pointer-backed block transfer and ordered publication.

The generic multiprocess path passes the same pointer operands to every
backend. This module reconstructs non-owning MUSA Tensor views locally, then
uses the existing native MUSA adapter or the torch fallback. Stream ordering
also stays here because TorchMUSA exposes external-stream synchronization, not
the CUDA host-callback ABI.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.musa import native_kv_transfer
from lmcache.v1.platform.musa.tensor_from_ptr import (
    construct_musa_tensor_from_data_pointer,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

_MUSA_MP_BLOCK_TRANSFER_FORMATS = {
    int(EngineKVFormat.NL_X_TWO_NB_BS_NH_HS),
    int(EngineKVFormat.NL_X_NB_BS_HS),
}


def _validate_musa_mp_block_transfer_format(
    engine_kv_format: EngineKVFormat,
) -> None:
    """Reject MUSA handle-transfer layouts outside the validated scope."""
    if int(engine_kv_format) not in _MUSA_MP_BLOCK_TRANSFER_FORMATS:
        raise ValueError(
            "MUSA MP block transfer supports only "
            "NL_X_TWO_NB_BS_NH_HS and NL_X_NB_BS_HS layouts; "
            f"got {engine_kv_format!r}"
        )


def _tensor_list(value: object) -> list[torch.Tensor] | None:
    """Return a flat tensor list, or ``None`` for pointer-form operands."""
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, torch.Tensor) for item in value):
        return None
    return value


def _as_device(
    device: torch.device | str,
    paged_operands: object,
    object_operands: object,
) -> torch.device:
    """Normalize the transfer device, tolerating CPU-only adapter tests."""
    if isinstance(device, torch.device):
        return device
    try:
        return torch.device(device)
    except RuntimeError:
        tensors = (_tensor_list(paged_operands) or []) + (
            _tensor_list(object_operands) or []
        )
        if tensors and all(tensor.device.type == "cpu" for tensor in tensors):
            return tensors[0].device
        raise


def _infer_dtype(
    paged_operands: object,
    object_operands: object,
    shape_desc: PageBufferShapeDesc,
) -> torch.dtype:
    """Resolve the dtype needed for pointer reconstruction."""
    descriptor_dtype = getattr(shape_desc, "dtype", None)
    if isinstance(descriptor_dtype, torch.dtype):
        return descriptor_dtype
    paged_tensors = _tensor_list(paged_operands)
    if paged_tensors:
        return paged_tensors[0].dtype
    object_tensors = _tensor_list(object_operands)
    if object_tensors:
        return object_tensors[0].dtype
    element_size = int(getattr(shape_desc, "element_size", 0))
    dtype_by_size = {1: torch.uint8, 4: torch.float32}
    try:
        return dtype_by_size[element_size]
    except KeyError as exc:
        raise ValueError(
            "MUSA pointer transfer requires an exact shape_desc.dtype when "
            f"element_size is ambiguous or unsupported; got {element_size}"
        ) from exc


def _paged_shape_and_stride(
    engine_kv_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Return per-layer shape and physical stride for a MUSA layout."""
    nb = int(shape_desc.nb)
    bs = int(shape_desc.bs)
    nh = int(shape_desc.nh)
    hs = int(shape_desc.hs)
    if int(engine_kv_format) == int(EngineKVFormat.NL_X_NB_BS_HS):
        block_stride = int(getattr(shape_desc, "block_stride_elems", 0))
        return (nb, bs, hs), (block_stride or bs * hs, hs, 1)
    if int(engine_kv_format) == int(EngineKVFormat.NL_X_TWO_NB_BS_NH_HS):
        return (2, nb, bs, nh, hs), None
    raise ValueError(f"Unsupported MUSA paged layout: {engine_kv_format!r}")


def _staging_shape(
    engine_kv_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
) -> tuple[int, ...]:
    """Return the logical shape of one staging chunk."""
    nl = int(shape_desc.nl)
    nh = int(shape_desc.nh)
    hs = int(shape_desc.hs)
    if int(engine_kv_format) == int(EngineKVFormat.NL_X_NB_BS_HS):
        return (nl, lmcache_chunk_size, hs)
    return (2, nl, lmcache_chunk_size, nh * hs)


def _validate_pointer_tensor(value: torch.Tensor, expected_layers: int) -> None:
    """Validate a packed per-layer pointer tensor."""
    if value.ndim != 1 or value.dtype not in (torch.int64, torch.uint64):
        raise TypeError("MUSA paged pointers must be a 1-D int64 pointer tensor")
    if expected_layers > 0 and value.numel() != expected_layers:
        raise ValueError(
            f"expected {expected_layers} MUSA paged pointers, got {value.numel()}"
        )


def _reconstruct_paged_layers(
    value: torch.Tensor | list[torch.Tensor],
    *,
    engine_kv_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """Normalize pointer-form paged operands to non-owning MUSA views."""
    tensor_layers = _tensor_list(value)
    if tensor_layers is not None:
        expected_layers = int(shape_desc.nl)
        if expected_layers > 0 and len(tensor_layers) != expected_layers:
            raise ValueError(
                f"expected {expected_layers} MUSA paged layers, "
                f"got {len(tensor_layers)}"
            )
        return tensor_layers
    if not isinstance(value, torch.Tensor):
        raise TypeError("MUSA paged operands must be a pointer tensor or tensor list")
    _validate_pointer_tensor(value, int(shape_desc.nl))
    if device.type != "musa":
        raise ValueError(
            f"MUSA pointer reconstruction requires a MUSA device, got {device}"
        )
    shape, stride = _paged_shape_and_stride(engine_kv_format, shape_desc)
    return [
        construct_musa_tensor_from_data_pointer(
            int(pointer.item()),
            shape,
            dtype,
            device,
            stride=stride,
        )
        for pointer in value
    ]


def _reconstruct_staging_tensors(
    value: list[int] | list[torch.Tensor],
    *,
    engine_kv_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """Normalize staging pointers to non-owning MUSA chunk views."""
    tensor_objects = _tensor_list(value)
    if tensor_objects is not None:
        return tensor_objects
    if not isinstance(value, list) or not all(
        isinstance(pointer, int) for pointer in value
    ):
        raise TypeError("MUSA staging operands must be list[int] or list[Tensor]")
    if device.type != "musa":
        raise ValueError(
            f"MUSA pointer reconstruction requires a MUSA device, got {device}"
        )
    shape = _staging_shape(engine_kv_format, shape_desc, lmcache_chunk_size)
    return [
        construct_musa_tensor_from_data_pointer(pointer, shape, dtype, device)
        for pointer in value
    ]


def _synchronize_stream_pointer(stream_ptr: int) -> None:
    """Synchronize a raw MUSA stream pointer through TorchMUSA."""
    if not isinstance(stream_ptr, int):
        raise TypeError("MUSA stream pointer must be an int")
    try:
        # Third Party
        import torch_musa

        external_stream = getattr(torch_musa, "ExternalStream", None)
        if not callable(external_stream):
            raise RuntimeError("TorchMUSA ExternalStream is unavailable")
        external_stream(stream_ptr).synchronize()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to synchronize MUSA stream pointer {stream_ptr}"
        ) from exc


class TorchMusaBlockTransfer:
    """Execute block transfer with the TorchMUSA-compatible torch backend."""

    def execute(
        self,
        paged_layers: list[torch.Tensor],
        object_tensors: list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device,
        direction: TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Transfer normalized tensor operands through the torch backend.

        Args:
            paged_layers: Per-layer MUSA KV-cache tensor views.
            object_tensors: MUSA staging tensors.
            block_ids: Engine block IDs participating in the transfer.
            device: MUSA device on which the transfer runs.
            direction: Store or retrieve transfer direction.
            shape_desc: Engine KV-cache shape descriptor.
            lmcache_chunk_size: Number of slots in each staging object.
            engine_kv_format: Engine KV-cache layout.
            skip_prefix_n_blocks: Leading blocks to skip.

        Returns:
            None.
        """
        torch_ops.multi_layer_block_kv_transfer(
            paged_layers,
            object_tensors,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )


class NativeMusaBlockTransfer:
    """Try the optional native MUSA block-transfer implementation."""

    def execute_if_supported(
        self,
        paged_layers: list[torch.Tensor],
        object_tensors: list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        direction: TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> bool:
        """Run native transfer when enabled and compatible.

        Args:
            paged_layers: Per-layer MUSA KV-cache tensor views.
            object_tensors: MUSA staging tensors.
            block_ids: Engine block IDs participating in the transfer.
            direction: Store or retrieve transfer direction.
            shape_desc: Engine KV-cache shape descriptor.
            lmcache_chunk_size: Number of slots in each staging object.
            engine_kv_format: Engine KV-cache layout.
            skip_prefix_n_blocks: Leading blocks to skip.

        Returns:
            ``True`` when native transfer completed, otherwise ``False``.
        """
        return native_kv_transfer.try_native_multi_layer_block_kv_transfer(
            paged_layers=paged_layers,
            object_tensors=object_tensors,
            block_ids=block_ids,
            direction=direction,
            shape_desc=shape_desc,
            lmcache_chunk_size=lmcache_chunk_size,
            engine_kv_format=engine_kv_format,
            skip_prefix_n_blocks=skip_prefix_n_blocks,
        )


_TORCH_TRANSFER = TorchMusaBlockTransfer()
_NATIVE_TRANSFER = NativeMusaBlockTransfer()


def _musa_multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: torch.Tensor | list[torch.Tensor],
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Reconstruct MUSA operands, then use native or torch transfer."""
    _validate_musa_mp_block_transfer_format(engine_kv_format)
    resolved_device = _as_device(device, paged_buffer_ptrs_tensor, lmcache_objects_ptrs)
    dtype = _infer_dtype(paged_buffer_ptrs_tensor, lmcache_objects_ptrs, shape_desc)
    paged_layers = _reconstruct_paged_layers(
        paged_buffer_ptrs_tensor,
        engine_kv_format=engine_kv_format,
        shape_desc=shape_desc,
        dtype=dtype,
        device=resolved_device,
    )
    object_tensors = _reconstruct_staging_tensors(
        lmcache_objects_ptrs,
        engine_kv_format=engine_kv_format,
        shape_desc=shape_desc,
        lmcache_chunk_size=lmcache_chunk_size,
        dtype=dtype,
        device=resolved_device,
    )
    if _NATIVE_TRANSFER.execute_if_supported(
        paged_layers,
        object_tensors,
        block_ids,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    ):
        return

    _TORCH_TRANSFER.execute(
        paged_layers,
        object_tensors,
        block_ids,
        resolved_device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


class MusaDeviceOps(DeviceOps):
    """MUSA block-transfer and stream-ordering operations."""

    device_type: ClassVar[str] = "musa"

    def record_completion_on_stream(
        self,
        stream_ptr: int,
        kind: str,
        payload: bytes,
    ) -> None:
        """Publish a completion only after prior MUSA stream work finishes.

        TorchMUSA does not yet expose the CUDA host-callback primitive used by
        LMCache's native completion recorder. Synchronizing here preserves the
        storage ownership contract until an asynchronous MUSA callback backend
        is available.

        Args:
            stream_ptr: Raw MUSA stream pointer from the generic recorder path.
            kind: Completion handler key.
            payload: Encoded completion payload.

        Returns:
            None.

        Raises:
            RuntimeError: If the MUSA stream cannot be synchronized.
        """
        _synchronize_stream_pointer(stream_ptr)
        super().record_completion_on_stream(0, kind, payload)

    def record_event_on_stream(
        self,
        stream_ptr: int,
        event_type_name: str,
        session_id: str,
        str_metadata: dict[str, str],
        int_metadata: dict[str, int],
    ) -> None:
        """Record an event only after prior MUSA stream work finishes.

        Args:
            stream_ptr: Raw MUSA stream pointer from the generic recorder path.
            event_type_name: Serialized event type.
            session_id: Session associated with the event.
            str_metadata: String-valued event metadata.
            int_metadata: Integer-valued event metadata.

        Returns:
            None.

        Raises:
            RuntimeError: If the MUSA stream cannot be synchronized.
        """
        _synchronize_stream_pointer(stream_ptr)
        super().record_event_on_stream(
            0,
            event_type_name,
            session_id,
            str_metadata,
            int_metadata,
        )

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: torch.Tensor | list[torch.Tensor],
        lmcache_objects_ptrs: list[int] | list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Transfer MUSA blocks through native code or the torch baseline.

        Args:
            paged_buffer_ptrs_tensor: Packed per-layer pointer tensor or direct
                MUSA Tensor list.
            lmcache_objects_ptrs: Staging data pointers or direct Tensor list.
            block_ids: Ordered engine block IDs for the transfer.
            device: MUSA device on which the transfer runs.
            direction: D2H store or H2D retrieve direction.
            shape_desc: Paged-buffer shape descriptor.
            lmcache_chunk_size: Number of slots in each staging object.
            engine_kv_format: Engine KV layout.
            skip_prefix_n_blocks: Leading blocks to skip.

        Returns:
            None.

        Raises:
            ValueError: If ``engine_kv_format`` is not supported by the MUSA
                handle path.
        """
        _musa_multi_layer_block_kv_transfer(
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
