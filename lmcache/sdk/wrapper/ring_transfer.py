# SPDX-License-Identifier: Apache-2.0
"""Transfer KV and query tensors through the QRingBuffer as raw L1 objects.

The server copies each cached chunk's L1 memory objects, byte for byte, into
free blocks of vLLM's Q ring (``sdk_retrieve``), and the SDK decodes them.
A store writes encoded objects into free blocks and the server commits them
(``sdk_store``).
The ring's unit here is a *piece*: one layer's slice of one ring block. Piece
``p`` is layer ``p % num_layers`` of block ``p // num_layers``; an object's
bytes fill its pieces in order.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Sequence
import dataclasses
import math
import threading

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.logging import init_logger
from lmcache.sdk.wrapper.ring_mapping import (
    RingMapping,
    RingReservation,
    RingStateError,
    check_ring_unchanged,
)
from lmcache.v1.common.token_codec import TOKEN_STRIDE
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, QRingInfo
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

logger = init_logger(__name__)

# Whether the copy engine (cudaMemcpyBatchAsync), which moves every transfer
# between the ring and pinned CPU tensors, is available.
_COPY_ENGINE = (
    hasattr(device_ops, "batched_memcpy_async") and device_ops.batch_memcpy_supported()
)


@dataclasses.dataclass(frozen=True)
class RecurrentState:
    """A hybrid model's recurrent-state pages at the end of a cached prefix.

    Attributes:
        pages: One CPU tensor per recurrent layer, in kernel-group then layer
            order, each the layer's slice of the last chunk's recurrent
            object, ``[planes, slots, W]``. The bytes are opaque (e.g. conv |
            ssm | pad for a GDN layer), only meaningful to the engine.
    """

    pages: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class SDKObjectLayout:
    """The L1 object of one chunk: dense models store one object per chunk, in
    one object group, holding one tensor.

    Attributes:
        object_group_idx: The server's object-group index.
        shape: The chunk's tensor shape, ``[planes, L, rows, W]``: ``rows``
            is ``chunk_size``, or a sliding window's rows per chunk.
        dtype: The tensor's dtype.
        tokens_per_block: Hybrid attention only: tokens per engine block.
        kernel_block_size: Hybrid attention only: tokens per kernel page when
            vLLM sub-pages each block into kernel pages of ``[K | V]`` rows;
            0 otherwise.
    """

    object_group_idx: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    tokens_per_block: int = 0
    kernel_block_size: int = 0

    @property
    def sub_paged(self) -> bool:
        """Whether each block's K and V are interleaved per kernel page."""
        return self.shape[0] == 2 and 0 < self.kernel_block_size < self.tokens_per_block

    @property
    def num_bytes(self) -> int:
        """Bytes of one chunk's object."""
        return math.prod(self.shape) * self.dtype.itemsize


def _ring_pieces(layers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """View the ring's per-layer tensors as ``[num_blocks, piece_bytes]``."""
    return [layer.reshape(layer.shape[0], -1).view(torch.uint8) for layer in layers]


class RingSDKTransferWrapper:
    """Retrieve and store cached tensors through free blocks of the Q ring.

    The transfer wrapper of every SDK context, KV or query. A transfer goes
    batch by batch: each batch reserves the free pieces of as many whole chunks
    as are free (``RingMapping.reserve``), so transfers of every context of the
    process run concurrently through disjoint pieces. The engine runs at TP 1:
    one ring. A hybrid model's last chunk goes in its own batch, with the
    recurrent object (the state at the end of the range).

    Args:
        ring: The engine's Q ring, opened (``connect()`` called).
        layout: The model's chunk object.
        chunk_size: Tokens per LMCache chunk.
        timeout: Timeout in seconds for each server call.
        writable: Whether ``store`` is allowed (KV); query tensors are
            read-only.
        recurrent: A hybrid model's recurrent object: one layout per tensor
            it packs, in packing order, all of one object group.

    Raises:
        ValueError: If the chunk object is not ``[planes, L, rows, W]`` with
            ``rows`` up to ``chunk_size`` (exactly it when ``writable``).
        RuntimeError: If the ring is not on CUDA or the copy engine is
            unavailable.
    """

    def __init__(
        self,
        ring: RingMapping,
        layout: SDKObjectLayout,
        chunk_size: int,
        timeout: float,
        writable: bool = True,
        recurrent: Sequence[SDKObjectLayout] = (),
    ) -> None:
        rows = layout.shape[2] if len(layout.shape) == 4 else 0
        if not 0 < rows <= chunk_size or (writable and rows != chunk_size):
            raise ValueError(
                f"the chunk object {layout.shape} is not [planes, L, rows, W] "
                f"with rows up to {chunk_size} (exactly it for KV)"
            )
        if not (_COPY_ENGINE and ring.device.type == "cuda"):
            raise RuntimeError(
                "the SDK needs a CUDA ring and LMCache built against CUDA >= 12.8"
            )
        self._ring = ring
        self._layout = layout
        self._writable = writable
        self._chunk_size = chunk_size
        self._timeout = timeout
        self._recurrent = tuple(recurrent)
        self._copy_plan_cache: tuple[int, torch.Tensor] | None = None

    @property
    def has_recurrent_state(self) -> bool:
        """Whether the model is hybrid (has a recurrent object)."""
        return bool(self._recurrent)

    @property
    def device(self) -> torch.device:
        """The device the ring lives on."""
        return self._ring.device

    @property
    def rows_per_chunk(self) -> int:
        """Rows a retrieve returns per chunk: ``chunk_size``, or a sub-chunk
        sliding window's size."""
        return self._layout.shape[2]

    @property
    def transfer_slot(self) -> threading.Semaphore:
        """A transfer slot; hold it across a lookup and its ``retrieve``."""
        return self._ring.transfer_slot

    def retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[torch.Tensor, RecurrentState | None] | None:
        """Retrieve ``[key.start, key.end)`` through the engine's ring.

        ``[0, key.end)`` must be read-locked by a lookup under
        ``key.request_id``; the prefix ``[0, key.start)`` is released, and on
        failure the unread part of the range. The caller must hold
        ``transfer_slot`` across that lookup and this call.

        Args:
            key: The cache server key; ``key.token_bytes`` covers ``key.end``.
            instance_id: Unused: the ring belongs to the engine's instance.

        Returns:
            ``(tensor, state)``: the cached rows as a CPU tensor ``[planes, L,
            T, W]`` (K and V, or query), and a hybrid model's recurrent state
            at ``key.end``, else None. None if the range is empty or the
            server fails a batch.

        Raises:
            RuntimeError: If the output's pinned memory cannot be allocated.
            RingStateError: If vLLM took ring blocks during the retrieve, the
                ring was re-registered, or it has too few free blocks for one
                chunk.
        """
        del instance_id
        self._ring.free_lookup_locks(key, 0, key.start)
        if key.end <= key.start:
            return None
        # Reserve the pinned memory to hold the retrieved tensor since
        # the SDK kernel requires pinned memory for host operands.
        try:
            out = self._new_output(key.end - key.start)
        except RuntimeError as err:
            # If not enough pinned memory, for now just throw an error.
            # In the future we might want to retry pinned memory allocation.
            self._ring.free_lookup_locks(key, key.start, key.end)
            raise RuntimeError(
                "out of pinned memory for the retrieved tensor; run fewer "
                "concurrent retrieve/modify requests"
            ) from err
        pages: list[torch.Tensor] = []
        start = key.start
        while start < key.end:
            try:
                reservation, num_chunks = self._reserve(start, key.end)
            except RingStateError:
                self._ring.free_lookup_locks(key, start, key.end)
                raise
            end = start + num_chunks * self._chunk_size
            with_state = self.has_recurrent_state and end == key.end
            try:
                ok = self._retrieve_batch(
                    key, reservation, start, end, out, pages if with_state else None
                )
            finally:
                self._ring.release(reservation)
            if not ok:
                self._ring.free_lookup_locks(key, end, key.end)
                return None
            try:
                check_ring_unchanged(
                    reservation.ring.info, self._ring.get_ring(), "cached tensors"
                )
            except RingStateError:
                self._ring.free_lookup_locks(key, end, key.end)
                raise
            start = end
        return out, RecurrentState(tuple(pages)) if self.has_recurrent_state else None

    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        kv: torch.Tensor,
        state: RecurrentState | None = None,
    ) -> bool:
        """Store a contiguous tensor covering ``[key.start, key.end)``.

        Args:
            key: The cache server key; ``key.token_bytes`` covers ``key.end``.
            instance_id: Unused: the ring belongs to the engine's instance.
            kv: Contiguous CPU tensor ``[planes, L, T, W]`` (K and V) with
                ``T == key.end - key.start``, chunk-aligned.
            state: Hybrid models only: the recurrent state stored as the
                state at ``key.end`` (as ``retrieve`` returns it).

        Returns:
            True if every batch was stored, False at the first failed batch.

        Raises:
            NotImplementedError: For read-only (query) tensors.
            ValueError: If ``kv`` or ``state`` does not match the layout.
            RingStateError: If vLLM took ring blocks during the store, the
                ring was re-registered, or it has too few free blocks for one
                chunk.
        """
        del instance_id
        if not self._writable:
            raise NotImplementedError("query tensors are read-only")
        self._check_store_input(key, kv, state)
        kv = kv.to(self._layout.dtype).contiguous().pin_memory()
        if key.end <= key.start:
            return True
        with self._ring.transfer_slot:
            return self._store_batches(key, kv, state)

    def _store_batches(
        self, key: IPCCacheServerKey, kv: torch.Tensor, state: RecurrentState | None
    ) -> bool:
        """Store ``kv`` batch by batch. Must hold a transfer slot."""
        start = key.start
        while start < key.end:
            reservation, num_chunks = self._reserve(start, key.end)
            end = start + num_chunks * self._chunk_size
            try:
                ok = self._store_batch(
                    key, kv, reservation, start, end, state if end == key.end else None
                )
            finally:
                self._ring.release(reservation)
            if not ok:
                logger.warning(
                    "SDK store failed for request_id=%s at [%d, %d)",
                    key.request_id,
                    start,
                    end,
                )
                return False
            check_ring_unchanged(
                reservation.ring.info, self._ring.get_ring(), "KV tensors"
            )
            start = end
        return True

    def close(self) -> None:
        """Drop the opened ring."""
        self._ring.close()

    # Planning

    def _pieces_per_chunk(self) -> int:
        """How many pieces of QRingBuffer needed for one chunk of KV cache."""
        return math.ceil(self._layout.num_bytes / self._ring.piece_bytes)

    def _reserve(self, start: int, end: int) -> tuple[RingReservation, int]:
        """Reserve the next batch of ``[start, end)``: as many whole chunks as
        are free.

        Returns:
            The reservation and the chunks it holds.
        """
        pieces_per_chunk = self._pieces_per_chunk()
        num_chunks = (end - start) // self._chunk_size
        if self.has_recurrent_state and num_chunks == 1:
            state_bytes = sum(layout.num_bytes for layout in self._recurrent)
            state_pieces = math.ceil(state_bytes / self._ring.piece_bytes)
            return self._ring.reserve(pieces_per_chunk + state_pieces, 1), 1
        if self.has_recurrent_state:
            num_chunks -= 1
        reservation = self._ring.reserve(pieces_per_chunk, num_chunks)
        return reservation, len(reservation.pieces) // pieces_per_chunk

    def _split_state_pieces(
        self, piece_ids: list[int], with_state: bool
    ) -> tuple[list[int], list[int]]:
        """Split a batch's pieces into the attention objects' and the
        recurrent object's, in the server's (ascending object group) order."""
        if not with_state:
            return piece_ids, []
        count = self._pieces_per_chunk()
        if self._recurrent[0].object_group_idx < self._layout.object_group_idx:
            return piece_ids[-count:], piece_ids[:-count]
        return piece_ids[:count], piece_ids[count:]

    def _chunk_pieces(self, piece_ids: list[int]) -> list[list[int]]:
        """Divide one chunk of KV cache into list of pieces."""
        count = self._pieces_per_chunk()
        return [piece_ids[i : i + count] for i in range(0, len(piece_ids), count)]

    def _sub_key(
        self, key: IPCCacheServerKey, info: QRingInfo, start: int, end: int
    ) -> IPCCacheServerKey:
        return dataclasses.replace(
            key,
            token_bytes=key.token_bytes[: end * TOKEN_STRIDE],
            start=start,
            end=end,
            worker_id=max(info.worker_id, 0),
        )

    def _call(
        self,
        method: Callable,
        key: IPCCacheServerKey,
        info: QRingInfo,
        piece_ids: list[int],
        device: torch.device,
        with_state: bool = False,
    ) -> bool:
        """Run one SDK transfer, ordered after this stream's work so far.

        Must run with the ring's device current; on return the server's
        copies have completed.

        Args:
            method: The client's ``sdk_retrieve`` or ``sdk_store``.
            key: The batch's key.
            info: The ring the batch goes through.
            piece_ids: The batch's pieces, chunk after chunk.
            device: The ring's device.
            with_state: Whether the batch also moves the recurrent object.

        Returns:
            Whether the server transferred every object.
        """
        groups = [self._layout.object_group_idx]
        if with_state:
            groups = sorted(groups + [self._recurrent[0].object_group_idx])
        event_backend = get_event_ipc_backend(device)
        event = event_backend.create_event(device)
        event_backend.record_event(event, torch_dev.current_stream())
        future = method(
            key,
            info.instance_id,
            groups,
            piece_ids,
            event_backend.export_event(event, device),
        ).to_device_future(device=device, event_backend=event_backend)
        return bool(future.result(timeout=self._timeout))

    # Retrieve

    def _retrieve_batch(
        self,
        key: IPCCacheServerKey,
        reservation: RingReservation,
        start: int,
        end: int,
        out: torch.Tensor,
        pages: list[torch.Tensor] | None = None,
    ) -> bool:
        """Retrieve ``[start, end)`` through the reserved pieces into ``out``,
        and with ``pages``, the recurrent state into it.

        Returns:
            Whether the server transferred every object.
        """
        info = reservation.ring.info
        pieces = _ring_pieces(reservation.ring.layers)
        device = reservation.ring.layers[0].device
        with torch_dev.device(device):
            ok = self._call(
                self._ring.req_client.sdk_retrieve,
                self._sub_key(key, info, start, end),
                info,
                reservation.pieces,
                device,
                with_state=pages is not None,
            )
            if not ok:
                logger.warning(
                    "SDK retrieve failed for request_id=%s at [%d, %d)",
                    key.request_id,
                    start,
                    end,
                )
                return False
            first_chunk = (start - key.start) // self._chunk_size
            attention_ids, state_ids = self._split_state_pieces(
                reservation.pieces, pages is not None
            )
            self._copy_batch(
                pieces, self._chunk_pieces(attention_ids), out, first_chunk
            )
            if pages is not None:
                pages.extend(self._read_state(pieces, state_ids))
        return True

    def _new_output(self, num_tokens: int) -> torch.Tensor:
        """Allocate the contiguous CPU tensor handed to SDK, with shape
        ``[planes, L, T, W]`` (result of retrieve)."""
        planes, num_layers, rows, width = self._layout.shape
        num_rows = num_tokens // self._chunk_size * rows
        return torch.empty(
            planes,
            num_layers,
            num_rows,
            width,
            dtype=self._layout.dtype,
            pin_memory=True,
        )

    def _copy_batch(
        self,
        pieces: list[torch.Tensor],
        chunk_pieces: list[list[int]],
        host: torch.Tensor,
        first_chunk: int,
        to_ring: bool = False,
    ) -> None:
        """Copy every chunk between its pieces and the pinned ``host``
        ``[planes, L, T, W]``, following the copy plan, with one
        cudaMemcpyBatchAsync: into ``host`` (waits for the copy), or with
        ``to_ring`` into the QRingBuffer pieces."""
        num_layers = len(pieces)
        slot, offset, row, within, nbytes = self._chunk_copy_plan(pieces[0].shape[1]).T
        # Each piece's address: its layer's base plus its block's stride.
        ids = torch.tensor(chunk_pieces)
        layer_bases = torch.tensor([piece.data_ptr() for piece in pieces])
        bases = layer_bases[ids % num_layers] + ids // num_layers * pieces[0].stride(0)
        chunks = torch.arange(first_chunk, first_chunk + len(chunk_pieces))
        chunk_bytes = self.rows_per_chunk * host.stride(2) * host.element_size()
        row_bytes = host.stride(1) * host.element_size()
        ring = (bases[:, slot] + offset).flatten().tolist()
        hosts = (
            (host.data_ptr() + chunks[:, None] * chunk_bytes + row * row_bytes + within)
            .flatten()
            .tolist()
        )
        sizes = nbytes.tolist() * len(chunk_pieces)
        src, dst = (hosts, ring) if to_ring else (ring, hosts)
        device_ops.batched_memcpy_async(src, dst, sizes, pieces[0].device)
        if not to_ring:
            torch_dev.current_stream().synchronize()

    def _chunk_copy_plan(self, piece_bytes: int) -> torch.Tensor:
        """The contiguous copies that move one chunk's object between its
        pieces and the host tensor, cached per piece size.

        Returns:
            int64 ``[N, 5]``: piece slot in the chunk, byte offset in that
            piece, host row (``plane * L + layer``), byte offset in the row's
            chunk window, bytes.
        """
        if (
            self._copy_plan_cache is not None
            and self._copy_plan_cache[0] == piece_bytes
        ):
            return self._copy_plan_cache[1]
        layout = self._layout
        planes, num_layers, rows, width = layout.shape
        row_bytes = width * layout.dtype.itemsize
        # Each out token row's index into the object; W never splits.
        src = torch.arange(planes * num_layers * rows).view(planes, num_layers, rows)
        if layout.sub_paged:
            # The registered view reads a block as [2, block] rows, but its
            # bytes are kernel pages of [K(kernel) | V(kernel)] rows.
            block, kernel = layout.tokens_per_block, layout.kernel_block_size
            src = src.reshape(2, num_layers, rows // block, block).permute(1, 2, 0, 3)
            src = src.reshape(num_layers, rows // block, block // kernel, 2, kernel)
            src = src.permute(3, 0, 1, 2, 4)
        src = src.reshape(planes * num_layers, rows)
        starts = torch.ones_like(src, dtype=torch.bool)
        starts[:, 1:] = src[:, 1:] != src[:, :-1] + 1
        out_row, col = starts.nonzero(as_tuple=True)
        lengths = torch.diff(out_row * rows + col, append=torch.tensor([src.numel()]))
        entries = []
        for row, at, start, size in zip(
            out_row.tolist(),
            (col * row_bytes).tolist(),
            (src[out_row, col] * row_bytes).tolist(),
            (lengths * row_bytes).tolist(),
            strict=True,
        ):
            while size:  # split at piece boundaries
                take = min(size, piece_bytes - start % piece_bytes)
                entries.append(
                    (start // piece_bytes, start % piece_bytes, row, at, take)
                )
                start, at, size = start + take, at + take, size - take
        self._copy_plan_cache = (piece_bytes, torch.tensor(entries, dtype=torch.int64))
        return self._copy_plan_cache[1]

    def _read_state(
        self, pieces: list[torch.Tensor], piece_ids: list[int]
    ) -> list[torch.Tensor]:
        """Read the recurrent object from its pieces, as one CPU page per
        recurrent layer."""
        num_layers = len(pieces)
        data = torch.cat([pieces[p % num_layers][p // num_layers] for p in piece_ids])
        state_pages: list[torch.Tensor] = []
        offset = 0
        for layout in self._recurrent:
            tensor = data[offset : offset + layout.num_bytes]
            tensor = tensor.view(layout.dtype).view(layout.shape)
            state_pages.extend(
                tensor[:, layer].to("cpu", copy=True)
                for layer in range(tensor.shape[1])
            )
            offset += layout.num_bytes
        return state_pages

    # Store

    def _check_store_input(
        self, key: IPCCacheServerKey, kv: torch.Tensor, state: RecurrentState | None
    ) -> None:
        planes, num_layers, _, width = self._layout.shape
        num_tokens = key.end - key.start
        if (
            kv.dim() != 4
            or tuple(kv.shape) != (planes, num_layers, num_tokens, width)
            or num_tokens % self._chunk_size
        ):
            raise ValueError(
                f"kv shape {tuple(kv.shape)} does not match "
                f"[{planes}, {num_layers}, {num_tokens}, {width}] with whole chunks"
            )
        expected = [
            (layout.shape[0], *layout.shape[2:])
            for layout in self._recurrent
            for _ in range(layout.shape[1])
        ]
        got = None if state is None else [tuple(page.shape) for page in state.pages]
        if (state is not None or expected) and got != expected:
            raise ValueError(
                f"state must hold one page per recurrent layer, shaped {expected}"
            )

    def _store_batch(
        self,
        key: IPCCacheServerKey,
        kv: torch.Tensor,
        reservation: RingReservation,
        start: int,
        end: int,
        state: RecurrentState | None = None,
    ) -> bool:
        """Store ``[start, end)`` of ``kv``, and ``state`` as the state at
        ``end``, through the reserved pieces.

        Returns:
            Whether the server stored every object.
        """
        info = reservation.ring.info
        pieces = _ring_pieces(reservation.ring.layers)
        device = reservation.ring.layers[0].device
        attention_ids, state_ids = self._split_state_pieces(
            reservation.pieces, state is not None
        )
        with torch_dev.device(device):
            first_chunk = (start - key.start) // self._chunk_size
            chunk_pieces = self._chunk_pieces(attention_ids)
            self._copy_batch(pieces, chunk_pieces, kv, first_chunk, to_ring=True)
            if state is not None:
                self._write_object(pieces, state_ids, self._encode_state(state))
            return self._call(
                self._ring.req_client.sdk_store,
                self._sub_key(key, info, start, end),
                info,
                reservation.pieces,
                device,
                with_state=state is not None,
            )

    def _encode_state(self, state: RecurrentState) -> torch.Tensor:
        """Build the recurrent object's bytes on the ring's device."""
        parts = []
        first = 0
        for layout in self._recurrent:
            pages = state.pages[first : first + layout.shape[1]]
            first += layout.shape[1]
            tensor = torch.stack(pages, dim=1).to(
                device=self.device, dtype=layout.dtype
            )
            parts.append(tensor.contiguous().reshape(-1).view(torch.uint8))
        return torch.cat(parts)

    @staticmethod
    def _write_object(
        pieces: list[torch.Tensor], piece_ids: list[int], data: torch.Tensor
    ) -> None:
        """Copy an object's bytes into its pieces, in order."""
        num_layers, piece_bytes = len(pieces), pieces[0].shape[1]
        for index, piece_id in enumerate(piece_ids):
            segment = data[index * piece_bytes : (index + 1) * piece_bytes]
            pieces[piece_id % num_layers][piece_id // num_layers][
                : segment.numel()
            ].copy_(segment)
