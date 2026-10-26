# SPDX-License-Identifier: Apache-2.0
"""Tensor-based MLU transfers with stream-safe host copies.

The engine-driven caller supplies real tensor objects for both the MLU KV
cache and host chunks. Raw staging addresses are deliberately rejected: the
shared torch fallback interprets them as host addresses, whereas the
LMCache-driven path may supply device staging addresses.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar, cast

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc


class MluDeviceOps(DeviceOps):
    """MLU engine-driven ops with explicit unsupported pointer paths."""

    device_type: ClassVar[str] = "mlu"

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
        """Transfer MLU KV blocks to or from canonical LMCache chunks.

        Supports rank-five K/V-first NHD and HND, and rank-three MLA,
        selected through KVFormatSpec layout facts. Transfers use the
        current stream. D2H fills one chunk a layer at a time to bound
        temporary device memory. D2H into a strided host slice blocks: torch_mlu
        1.26.1 does not correctly complete that copy with non_blocking=True.
        Callers must synchronize before consuming other host chunks or
        releasing input storage.

        Args:
            paged_buffer_ptrs_tensor: Per-layer KV tensors, not addresses.
            lmcache_objects_ptrs: Host or device chunk tensors, not addresses.
            block_ids: Engine block IDs in chunk order.
            device: Device owning the paged KV tensors.
            direction: D2H to store, H2D to retrieve.
            shape_desc: Paged-cache dimensions and dtype.
            lmcache_chunk_size: Tokens in each chunk, a block-size multiple.
            engine_kv_format: NHD, HND, or MLA format listed above.
            skip_prefix_n_blocks: Leading blocks left untouched.

        Raises:
            TypeError: If either operand collection contains raw pointers.
            NotImplementedError: If the layout facts or tensor rank are unsupported.
            ValueError: If the format is unknown, or chunk size, skip count,
                or direction is invalid.
        """
        if not isinstance(paged_buffer_ptrs_tensor, list) or not all(
            isinstance(tensor, torch.Tensor) for tensor in paged_buffer_ptrs_tensor
        ):
            raise TypeError("MLU block transfer requires per-layer tensor operands")
        if not all(isinstance(tensor, torch.Tensor) for tensor in lmcache_objects_ptrs):
            raise TypeError("MLU block transfer requires tensor chunks, not pointers")
        # The specs package reads lmcache.device_ops, so load it only after
        # device initialization, as in the shared torch and RBLN paths.
        # First Party
        from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec_class

        format_spec = get_spec_class(engine_kv_format)
        if (
            not format_spec.is_layer_list
            or format_spec.is_fused_packed
            or format_spec.is_pbs_fused
            or format_spec.is_kv_second_tuple
            or format_spec.requires_blocked_scale_repacking
            or not (format_spec.is_mla or format_spec.is_two_major)
        ):
            raise NotImplementedError(f"Unsupported MLU KV format: {engine_kv_format}")
        # HND formats with an extra singleton axis share these layout facts,
        # but the indexing below requires ordinary rank-five paged tensors.
        expected_rank = 3 if format_spec.is_mla else 5
        if any(layer.ndim != expected_rank for layer in paged_buffer_ptrs_tensor):
            raise NotImplementedError(
                f"MLU block transfer requires rank-{expected_rank} paged tensors"
            )
        if direction != TransferDirection.D2H:
            super().multi_layer_block_kv_transfer(
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
            return

        block_size = int(shape_desc.bs)
        if (
            block_size <= 0
            or lmcache_chunk_size <= 0
            or lmcache_chunk_size % block_size
        ):
            raise ValueError(
                "lmcache_chunk_size must be a positive block-size multiple"
            )
        if skip_prefix_n_blocks < 0:
            raise ValueError("skip_prefix_n_blocks must be nonnegative")
        if not paged_buffer_ptrs_tensor or not lmcache_objects_ptrs:
            return
        blocks_per_chunk = lmcache_chunk_size // block_size
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=device)
        is_mla = format_spec.is_mla
        is_hnd = format_spec.is_hnd
        first_layer = paged_buffer_ptrs_tensor[0]
        hidden_size = first_layer.shape[-1]
        if not is_mla:
            hidden_size *= first_layer.shape[2 if is_hnd else 3]
        chunks = cast(list[torch.Tensor], lmcache_objects_ptrs)
        for chunk_idx, chunk in enumerate(chunks):
            begin = max(chunk_idx * blocks_per_chunk, skip_prefix_n_blocks)
            end = min((chunk_idx + 1) * blocks_per_chunk, ids.numel())
            if begin >= end:
                continue
            offset = (begin - chunk_idx * blocks_per_chunk) * block_size
            tokens = (end - begin) * block_size
            chunk_shape: tuple[int, ...] = (
                len(paged_buffer_ptrs_tensor),
                tokens,
                hidden_size,
            )
            if not is_mla:
                chunk_shape = (2, *chunk_shape)
            gathered = first_layer.new_empty(chunk_shape)
            for layer_idx, layer in enumerate(paged_buffer_ptrs_tensor):
                selected = layer.index_select(0 if is_mla else 1, ids[begin:end])
                if is_hnd:
                    selected = selected.permute(0, 1, 3, 2, 4)
                if is_mla:
                    gathered[layer_idx].copy_(selected.reshape(tokens, -1))
                else:
                    gathered[:, layer_idx].copy_(selected.reshape(2, tokens, -1))
                del selected
            target = (
                chunk[:, offset : offset + tokens]
                if is_mla
                else chunk[:, :, offset : offset + tokens]
            )
            # A skipped prefix leaves gaps between layers/planes on the host.
            # torch_mlu's asynchronous strided D2H may return stale data even
            # after device synchronization; its blocking copy is correct.
            target.copy_(
                gathered,
                non_blocking=target.device.type != "cpu" or target.is_contiguous(),
            )
            # Same-stream allocator reuse preserves the queued copy's lifetime
            # without retaining this chunk during the next chunk's allocation.
            del gathered

    def lmcache_memcpy_async(
        self,
        dest: int | torch.Tensor,
        src: int | torch.Tensor,
        nbytes: int,
        direction: TransferDirection,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        """Copy contiguous tensor storage without entering CUDA pointer code.

        Args:
            dest: Contiguous destination tensor.
            src: Contiguous source tensor with the same dtype.
            nbytes: Nonnegative byte count within both operands.
            direction: H2D or D2H.
            host_buffer_offset: Unused for tensor copies.
            host_buffer_alignments: Positive power-of-two alignment, validated
                by the shared tensor-copy implementation.

        Raises:
            TypeError: If either operand is a raw pointer.
            ValueError: If tensors, byte count, direction, or alignment are invalid.
        """
        if not isinstance(dest, torch.Tensor) or not isinstance(src, torch.Tensor):
            raise TypeError("MLU memcpy requires tensor operands, not raw pointers")
        if (
            not dest.is_contiguous()
            or not src.is_contiguous()
            or dest.dtype != src.dtype
        ):
            raise ValueError(
                "MLU memcpy requires contiguous tensors with matching dtype"
            )
        if nbytes < 0 or nbytes > min(dest.nbytes, src.nbytes):
            raise ValueError("nbytes must fit within both tensor operands")
        super().lmcache_memcpy_async(
            dest, src, nbytes, direction, host_buffer_offset, host_buffer_alignments
        )

    def record_completion_on_stream(
        self, stream_ptr: int, kind: str, payload: bytes
    ) -> None:
        """Wait for the MLU queue before publishing a storage completion.

        This conservative implementation blocks the calling thread. The
        stream must belong to the current MLU device and remain alive.

        Args:
            stream_ptr: CNRT queue handle, including zero for the default queue.
            kind: Completion handler key.
            payload: Encoded completion payload.

        Raises:
            RuntimeError: If torch_mlu cannot synchronize the stream.
        """
        self._synchronize_stream(stream_ptr)
        super().record_completion_on_stream(0, kind, payload)

    def record_event_on_stream(
        self,
        stream_ptr: int,
        event_type_name: str,
        session_id: str,
        str_metadata: dict[str, str],
        int_metadata: dict[str, int],
    ) -> None:
        """Timestamp an event after the MLU queue completes, blocking the caller.

        Args:
            stream_ptr: CNRT queue handle on the current MLU device.
            event_type_name: Serialized event type.
            session_id: Session associated with the event.
            str_metadata: String-valued event metadata.
            int_metadata: Integer-valued event metadata.

        Raises:
            RuntimeError: If torch_mlu cannot synchronize the stream.
        """
        self._synchronize_stream(stream_ptr)
        super().record_event_on_stream(
            0, event_type_name, session_id, str_metadata, int_metadata
        )

    def multi_layer_kv_transfer(self, *args: object, **kwargs: object) -> None:
        """Reject the in-process connector; use MP engine_driven on MLU."""
        raise NotImplementedError(
            "MLU requires the MP engine_driven block transfer path"
        )

    def single_layer_kv_transfer(self, *args: object, **kwargs: object) -> None:
        """Reject the in-process connector; use MP engine_driven on MLU."""
        raise NotImplementedError(
            "MLU requires the MP engine_driven block transfer path"
        )

    def _synchronize_stream(self, stream_ptr: int) -> None:
        # CNRT's null queue is not torch_mlu's default stream. Resolve the
        # default explicitly so the zero sentinel cannot publish too early.
        stream = (
            torch.mlu.default_stream()
            if stream_ptr == 0
            else torch.mlu.ExternalStream(stream_ptr)
        )
        stream.synchronize()
