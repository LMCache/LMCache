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
from lmcache import torch_dev
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


@dataclasses.dataclass(frozen=True)
class SDKObjectLayout:
    """The L1 object of one chunk: dense models store one object per chunk, in
    one object group, holding one tensor.

    Attributes:
        object_group_idx: The server's object-group index.
        shape: The chunk's tensor shape, ``[planes, L, chunk_size, W]``.
        dtype: The tensor's dtype.
    """

    object_group_idx: int
    shape: tuple[int, ...]
    dtype: torch.dtype

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
    one ring.

    Args:
        ring: The engine's Q ring, opened (``connect()`` called).
        layout: The model's chunk object.
        chunk_size: Tokens per LMCache chunk.
        timeout: Timeout in seconds for each server call.
        writable: Whether ``store`` is allowed (KV); query tensors are
            read-only.

    Raises:
        ValueError: If the chunk object is not ``[planes, L, chunk_size, W]``.
    """

    def __init__(
        self,
        ring: RingMapping,
        layout: SDKObjectLayout,
        chunk_size: int,
        timeout: float,
        writable: bool = True,
    ) -> None:
        if len(layout.shape) != 4 or layout.shape[2] != chunk_size:
            raise ValueError(
                f"the chunk object {layout.shape} is not [planes, L, {chunk_size}, W]"
            )
        self._ring = ring
        self._layout = layout
        self._writable = writable
        self._chunk_size = chunk_size
        self._timeout = timeout

    @property
    def device(self) -> torch.device:
        """The device the ring lives on."""
        return self._ring.device

    @property
    def transfer_slot(self) -> threading.Semaphore:
        """A transfer slot; hold it across a lookup and its ``retrieve``."""
        return self._ring.transfer_slot

    def retrieve(self, key: IPCCacheServerKey, instance_id: int) -> torch.Tensor | None:
        """Retrieve ``[key.start, key.end)`` through the engine's ring.

        The range must be read-locked by a lookup under ``key.request_id``;
        on failure its unread part is released. The caller must hold
        ``transfer_slot`` across that lookup and this call.

        Args:
            key: The cache server key; ``key.token_bytes`` covers ``key.end``.
            instance_id: Unused: the ring belongs to the engine's instance.

        Returns:
            The cached rows as a CPU tensor ``[planes, L, T, W]`` (K and V for
            KV; ``planes == 1`` for query). None if the range is empty or the
            server fails a batch.

        Raises:
            RingStateError: If vLLM took ring blocks during the retrieve, the
                ring was re-registered, or it has too few free blocks for one
                chunk.
        """
        del instance_id
        if key.end <= key.start:
            return None
        pieces_per_chunk = self._pieces_per_chunk()
        out: torch.Tensor | None = None
        start = key.start
        while start < key.end:
            try:
                reservation = self._ring.reserve(
                    pieces_per_chunk, (key.end - start) // self._chunk_size
                )
            except RingStateError:
                self._ring.free_lookup_locks(key, start, key.end)
                raise
            num_chunks = len(reservation.pieces) // pieces_per_chunk
            end = start + num_chunks * self._chunk_size
            try:
                if out is None:
                    out = self._new_output(key.end - key.start)
                ok = self._retrieve_batch(key, reservation, start, end, out)
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
        return out

    def store(self, key: IPCCacheServerKey, instance_id: int, kv: torch.Tensor) -> bool:
        """Store a contiguous tensor covering ``[key.start, key.end)``.

        Args:
            key: The cache server key; ``key.token_bytes`` covers ``key.end``.
            instance_id: Unused: the ring belongs to the engine's instance.
            kv: Contiguous CPU tensor ``[planes, L, T, W]`` (K and V) with
                ``T == key.end - key.start``, chunk-aligned.

        Returns:
            True if every batch was stored, False at the first failed batch.

        Raises:
            NotImplementedError: For read-only (query) tensors.
            ValueError: If ``kv`` does not match the layout.
            RingStateError: If vLLM took ring blocks during the store, the
                ring was re-registered, or it has too few free blocks for one
                chunk.
        """
        del instance_id
        if not self._writable:
            raise NotImplementedError("query tensors are read-only")
        self._check_store_input(key, kv)
        if key.end <= key.start:
            return True
        with self._ring.transfer_slot:
            return self._store_batches(key, kv)

    def _store_batches(self, key: IPCCacheServerKey, kv: torch.Tensor) -> bool:
        """Store ``kv`` batch by batch. Must hold a transfer slot."""
        pieces_per_chunk = self._pieces_per_chunk()
        start = key.start
        while start < key.end:
            reservation = self._ring.reserve(
                pieces_per_chunk, (key.end - start) // self._chunk_size
            )
            num_chunks = len(reservation.pieces) // pieces_per_chunk
            end = start + num_chunks * self._chunk_size
            try:
                ok = self._store_batch(key, kv, reservation, start, end)
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

        Returns:
            Whether the server transferred every object.
        """
        event_backend = get_event_ipc_backend(device)
        event = event_backend.create_event(device)
        event_backend.record_event(event, torch_dev.current_stream())
        future = method(
            key,
            info.instance_id,
            [self._layout.object_group_idx],
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
    ) -> bool:
        """Retrieve ``[start, end)`` through the reserved pieces into ``out``.

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
            self._read_batch(
                pieces, self._chunk_pieces(reservation.pieces), out, first_chunk
            )
        return True

    def _new_output(self, num_tokens: int) -> torch.Tensor:
        """Allocate the contiguous CPU tensor handed to SDK, with shape
        ``[planes, L, T, W]`` (result of retrieve)."""
        planes, num_layers, _, width = self._layout.shape
        return torch.empty(
            planes, num_layers, num_tokens, width, dtype=self._layout.dtype
        )

    def _read_batch(
        self,
        pieces: list[torch.Tensor],
        chunk_pieces: list[list[int]],
        out: torch.Tensor,
        first_chunk: int,
    ) -> None:
        """Copy a batch's chunk objects from the ring into ``out``, one at a
        time. Each object goes through a pinned CPU staging buffer because
        ``.to("cpu")`` is several times slower.
        """
        num_layers, size = len(pieces), self._layout.num_bytes
        staging = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        chunk_tensor = staging.view(self._layout.dtype).view(self._layout.shape)
        for chunk, piece_ids in enumerate(chunk_pieces, start=first_chunk):
            data = torch.cat(
                [pieces[p % num_layers][p // num_layers] for p in piece_ids]
            )
            staging.copy_(data[:size])
            token = chunk * self._chunk_size
            out[:, :, token : token + self._chunk_size].copy_(chunk_tensor)

    # Store

    def _check_store_input(self, key: IPCCacheServerKey, kv: torch.Tensor) -> None:
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

    def _store_batch(
        self,
        key: IPCCacheServerKey,
        kv: torch.Tensor,
        reservation: RingReservation,
        start: int,
        end: int,
    ) -> bool:
        """Store ``[start, end)`` of ``kv`` through the reserved pieces.

        Returns:
            Whether the server stored every object.
        """
        info = reservation.ring.info
        pieces = _ring_pieces(reservation.ring.layers)
        device = reservation.ring.layers[0].device
        with torch_dev.device(device):
            for chunk, piece_ids in enumerate(self._chunk_pieces(reservation.pieces)):
                token = start - key.start + chunk * self._chunk_size
                data = self._encode_object(kv, token)
                self._write_object(pieces, piece_ids, data)
            return self._call(
                self._ring.req_client.sdk_store,
                self._sub_key(key, info, start, end),
                info,
                reservation.pieces,
                device,
            )

    def _encode_object(self, kv: torch.Tensor, token: int) -> torch.Tensor:
        """Build the bytes of the chunk object starting at ``kv``'s ``token``,
        on the ring's device."""
        tensor = kv[:, :, token : token + self._chunk_size]
        tensor = tensor.to(device=self.device, dtype=self._layout.dtype).contiguous()
        if tuple(tensor.shape) != self._layout.shape:
            raise ValueError(
                f"a chunk encodes as {tuple(tensor.shape)}, not {self._layout.shape}"
            )
        return tensor.reshape(-1).view(torch.uint8)

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
