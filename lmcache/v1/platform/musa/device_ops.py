# SPDX-License-Identifier: Apache-2.0
"""MUSA ops backend with transfer and stream-ordering adapters.

:class:`MusaDeviceOps` overrides :meth:`multi_layer_block_kv_transfer` to
try the native MUSA path first (when inputs are tensor-backed) and fall
back to the torch baseline otherwise. Stream recorders synchronize external
MUSA streams before publishing through the baseline queues.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import (
    EngineKVFormat,
    PageBufferShapeDesc,
    TransferDirection,
)


def _tensor_list(value: object) -> list[torch.Tensor] | None:
    """Return ``value`` as ``list[torch.Tensor]`` when it is tensor-backed."""
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, torch.Tensor) for item in value):
        return None
    return value


def _synchronize_stream_pointer(stream_ptr: int) -> None:
    """Synchronize a raw MUSA stream pointer through TorchMUSA.

    Args:
        stream_ptr: Process-local pointer to an existing MUSA stream.

    Raises:
        TypeError: If ``stream_ptr`` is not an integer.
        RuntimeError: If TorchMUSA cannot wrap or synchronize the stream.
    """
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


def _musa_multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: torch.Tensor | list,
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: EngineKVFormat,
    skip_prefix_n_blocks: int,
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

    torch_ops.multi_layer_block_kv_transfer(
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


class MusaDeviceOps(DeviceOps):
    """MUSA transfer and stream-ordering operations."""

    device_type: ClassVar[str] = "musa"

    def record_completion_on_stream(
        self,
        stream_ptr: int,
        kind: str,
        payload: bytes,
    ) -> None:
        """Publish a completion after prior MUSA stream work finishes.

        Args:
            stream_ptr: Process-local pointer to the MUSA transfer stream.
            kind: Completion handler key.
            payload: Encoded completion payload.

        Raises:
            RuntimeError: If the stream cannot be synchronized.
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
        """Record an event after prior MUSA stream work finishes.

        Args:
            stream_ptr: Process-local pointer to the MUSA transfer stream.
            event_type_name: Serialized event type.
            session_id: Session associated with the event.
            str_metadata: String-valued event metadata.
            int_metadata: Integer-valued event metadata.

        Raises:
            RuntimeError: If the stream cannot be synchronized.
        """
        _synchronize_stream_pointer(stream_ptr)
        super().record_event_on_stream(
            0,
            event_type_name,
            session_id,
            str_metadata,
            int_metadata,
        )

    def multi_layer_block_kv_transfer(self, *args, **kwargs) -> None:
        _musa_multi_layer_block_kv_transfer(*args, **kwargs)
