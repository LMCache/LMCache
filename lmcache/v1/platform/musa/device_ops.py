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
from typing import ClassVar, TypeAlias, cast
import ctypes

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection, is_kv_list
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
    int(EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS),
}

_PagedBufferOperand: TypeAlias = (
    torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]
)
_PagedLayers: TypeAlias = list[torch.Tensor] | list[list[torch.Tensor]]


def _current_musa_device() -> torch.device:
    """Return the MUSA device associated with the current worker thread."""
    musa = getattr(torch, "musa", None)
    if musa is None or not callable(getattr(musa, "current_device", None)):
        raise RuntimeError("torch.musa.current_device is unavailable")
    return torch.device("musa", int(musa.current_device()))


def _host_byte_tensor_from_pointer(pointer: int, nbytes: int) -> torch.Tensor:
    """Create a non-owning CPU byte tensor for a host allocation."""
    if pointer <= 0:
        raise ValueError("host pointer must be a non-zero positive integer")
    buffer_type = ctypes.c_uint8 * nbytes
    return torch.frombuffer(buffer_type.from_address(pointer), dtype=torch.uint8)


def _validate_musa_mp_block_transfer_format(
    engine_kv_format: EngineKVFormat,
) -> None:
    """Reject MUSA handle-transfer layouts outside the validated scope."""
    if int(engine_kv_format) not in _MUSA_MP_BLOCK_TRANSFER_FORMATS:
        raise ValueError(
            "MUSA MP block transfer supports only "
            "NL_X_TWO_NB_BS_NH_HS, NL_X_NB_BS_HS, and "
            "TWO_X_NL_X_NB_BS_NH_HS layouts; "
            f"got {engine_kv_format!r}"
        )


def _tensor_list(value: object) -> list[torch.Tensor] | None:
    """Return a flat tensor list, or ``None`` for pointer-form operands."""
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, torch.Tensor) for item in value):
        return None
    return value


def _tensor_leaves(value: object) -> list[torch.Tensor]:
    """Return all tensor leaves from a tensor or nested list."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, list):
        return [tensor for item in value for tensor in _tensor_leaves(item)]
    return []


def _kv_layer_lists(value: object) -> list[list[torch.Tensor]] | None:
    """Return validated separate ``[key_layers, value_layers]`` tensor lists."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(isinstance(group, list) for group in value):
        return None
    groups = value
    if not all(
        isinstance(tensor, torch.Tensor) for group in groups for tensor in group
    ):
        return None
    return groups


def _paged_tensor_leaves(paged_operands: object) -> list[torch.Tensor]:
    """Return paged KV tensor leaves, ignoring packed pointer tensors."""
    tensor_layers = _tensor_list(paged_operands)
    if tensor_layers is not None:
        return tensor_layers
    nested_layers = _kv_layer_lists(paged_operands)
    if nested_layers is not None:
        return _tensor_leaves(nested_layers)
    return []


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
        tensors = _paged_tensor_leaves(paged_operands) + (
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
    # Flat tensor lists and nested ``[key_layers, value_layers]`` lists carry dtype.
    # A packed int64 pointer tensor is not a dtype source.
    paged_tensors = _paged_tensor_leaves(paged_operands)
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
    if int(engine_kv_format) == int(EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS):
        return (nb, bs, nh, hs), None
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
    value: _PagedBufferOperand,
    *,
    engine_kv_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
    dtype: torch.dtype,
    device: torch.device,
) -> _PagedLayers:
    """Normalize pointer-form paged operands to non-owning MUSA views."""
    expected_layers = int(shape_desc.nl)
    separate_kv_lists = is_kv_list(engine_kv_format)
    if separate_kv_lists:
        nested_layers = _kv_layer_lists(value)
        if nested_layers is not None:
            if any(len(group) != expected_layers for group in nested_layers):
                raise ValueError(
                    f"expected {expected_layers} MUSA layers per key/value "
                    f"group, got {[len(group) for group in nested_layers]}"
                )
            return nested_layers

    tensor_layers = _tensor_list(value)
    if tensor_layers is not None:
        expected_tensors = 2 * expected_layers if separate_kv_lists else expected_layers
        if expected_tensors > 0 and len(tensor_layers) != expected_tensors:
            raise ValueError(
                f"expected {expected_tensors} MUSA paged tensors, "
                f"got {len(tensor_layers)}"
            )
        if separate_kv_lists:
            return [
                tensor_layers[:expected_layers],
                tensor_layers[expected_layers:],
            ]
        return tensor_layers
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "MUSA paged operands must be a pointer tensor or supported tensor list"
        )
    expected_pointers = 2 * expected_layers if separate_kv_lists else expected_layers
    _validate_pointer_tensor(value, expected_pointers)
    if device.type != "musa":
        raise ValueError(
            f"MUSA pointer reconstruction requires a MUSA device, got {device}"
        )
    shape, stride = _paged_shape_and_stride(engine_kv_format, shape_desc)
    reconstructed = [
        construct_musa_tensor_from_data_pointer(
            int(pointer.item()),
            shape,
            dtype,
            device,
            stride=stride,
        )
        for pointer in value
    ]
    if separate_kv_lists:
        return [
            reconstructed[:expected_layers],
            reconstructed[expected_layers:],
        ]
    return reconstructed


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
    pointers = cast(list[int], value)
    return [
        construct_musa_tensor_from_data_pointer(pointer, shape, dtype, device)
        for pointer in pointers
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
        paged_layers: _PagedLayers,
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
            paged_layers: Per-layer MUSA KV-cache tensor views, or separate
                ``[key_layers, value_layers]`` tensor lists for KV-list layouts.
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
        paged_layers: _PagedLayers,
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
            paged_layers: Per-layer MUSA KV-cache tensor views, or separate
                ``[key_layers, value_layers]`` tensor lists for KV-list layouts.
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
    paged_buffer_ptrs_tensor: _PagedBufferOperand,
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

    def lmcache_memcpy_async(
        self,
        dest: int | torch.Tensor,
        src: int | torch.Tensor,
        nbytes: int,
        direction: TransferDirection,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        """Copy lazy staging buffers without dereferencing MUSA pointers on CPU.

        The generic lazy-allocator path supplies raw host and device pointers.
        Reconstruct the device side as a non-owning MUSA tensor and use
        ``Tensor.copy_`` on the current MUSA stream. This preserves the
        asynchronous staging contract without entering CUDA/HIP pointer-copy
        code.

        Args:
            dest: Destination tensor or raw pointer.
            src: Source tensor or raw pointer.
            nbytes: Number of bytes to copy.
            direction: H2D or D2H transfer direction.
            host_buffer_offset: Byte offset of the host pointer from the lazy
                allocator base.
            host_buffer_alignments: Host pin-registration alignment. Must be a
                positive power of two.

        Returns:
            None.

        Raises:
            TypeError: If pointer mode receives non-integer operands.
            ValueError: If sizes, pointers, alignment, or direction are invalid.
            RuntimeError: If the current MUSA device cannot be resolved.
        """
        if isinstance(dest, torch.Tensor) or isinstance(src, torch.Tensor):
            torch_ops.lmcache_memcpy_async(
                dest,
                src,
                nbytes,
                direction,
                host_buffer_offset,
                host_buffer_alignments,
            )
            return
        if not isinstance(dest, int) or not isinstance(src, int):
            raise TypeError("MUSA staging operands must both be pointers or tensors")
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if host_buffer_alignments <= 0 or (
            host_buffer_alignments & (host_buffer_alignments - 1)
        ):
            raise ValueError("host_buffer_alignments must be power of two")
        if int(direction) not in (
            int(TransferDirection.H2D),
            int(TransferDirection.D2H),
        ):
            raise ValueError(f"Unsupported direction: {direction}")
        if nbytes == 0:
            return

        direction_value = int(direction)
        if direction_value == int(TransferDirection.H2D):
            host_base, device_base = src, dest
            is_h2d = True
        else:
            host_base, device_base = dest, src
            is_h2d = False

        device = _current_musa_device()
        copied = 0
        while copied < nbytes:
            bytes_to_boundary = host_buffer_alignments - (
                (host_buffer_offset + copied) % host_buffer_alignments
            )
            copy_size = min(nbytes - copied, bytes_to_boundary)
            host_tensor = _host_byte_tensor_from_pointer(
                host_base + copied,
                copy_size,
            )
            device_tensor = construct_musa_tensor_from_data_pointer(
                device_base + copied,
                (copy_size,),
                torch.uint8,
                device,
            )
            if is_h2d:
                device_tensor.copy_(host_tensor, non_blocking=True)
            else:
                host_tensor.copy_(device_tensor, non_blocking=True)
            copied += copy_size

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
        paged_buffer_ptrs_tensor: _PagedBufferOperand,
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
            paged_buffer_ptrs_tensor: Packed per-layer pointer tensor, direct
                MUSA Tensor list, or separate ``[key_layers, value_layers]``
                tensor lists for KV-list layouts.
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
