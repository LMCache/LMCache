# SPDX-License-Identifier: Apache-2.0
"""Transfer KV and query tensors through the Q ring vLLM owns, as raw L1
objects.

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

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.sdk.wrapper.ring_mapping import (
    RingMapping,
    RingStateError,
    check_ring_unchanged,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, QRingInfo
from lmcache.v1.multiprocess.token_codec import TOKEN_STRIDE
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class RecurrentState:
    """Recurrent-state pages at the end of a cached prefix.

    Attributes:
        pages: One CPU tensor per recurrent layer, in kernel-group then layer
            order, each the layer's slice of the prefix's last chunk object,
            ``[planes, slots, width]``. The bytes are opaque (e.g. conv | ssm |
            pad for a GDN layer), only meaningful to the engine that wrote
            them.
    """

    pages: tuple[torch.Tensor, ...]


def subpaged_to_kv(pages: torch.Tensor, block: int, kernel: int) -> torch.Tensor:
    """Decode sub-paged attention rows into true K and V planes.

    Each registered block is viewed as ``[2, block, W]``, but its bytes are
    ``block // kernel`` kernel pages of ``[K(kernel, W) | V(kernel, W)]``.

    Args:
        pages: ``[2, L, T, W]`` rows as the registered view gathers them,
            ``T`` a multiple of ``block``.
        block: Tokens per registered block.
        kernel: Tokens per kernel page.

    Returns:
        ``[2, L, T, W]`` with plane 0 the true K and plane 1 the true V.
    """
    _, num_layers, num_tokens, width = pages.shape
    num_blocks, per_block = num_tokens // block, block // kernel
    # Each block's bytes, in memory order: plane 0 rows, then plane 1 rows.
    raw = pages.reshape(2, num_layers, num_blocks, block * width).permute(1, 2, 0, 3)
    raw = raw.reshape(num_layers, num_blocks, per_block, 2, kernel, width)
    return raw.permute(3, 0, 1, 2, 4, 5).reshape(2, num_layers, num_tokens, width)


def kv_to_subpaged(kv: torch.Tensor, block: int, kernel: int) -> torch.Tensor:
    """Encode true K/V planes into the sub-paged registered rows.

    The inverse of ``subpaged_to_kv``.

    Args:
        kv: ``[2, L, T, W]`` true K and V, ``T`` a multiple of ``block``.
        block: Tokens per registered block.
        kernel: Tokens per kernel page.

    Returns:
        ``[2, L, T, W]`` rows in the registered view's order.
    """
    _, num_layers, num_tokens, width = kv.shape
    num_blocks, per_block = num_tokens // block, block // kernel
    raw = kv.reshape(2, num_layers, num_blocks, per_block, kernel, width)
    raw = raw.permute(1, 2, 3, 0, 4, 5).reshape(num_layers, num_blocks, 2, -1)
    return raw.permute(2, 0, 1, 3).reshape(2, num_layers, num_tokens, width)


@dataclasses.dataclass(frozen=True)
class SDKKernelGroup:
    """One kernel group's tensor inside its object group's chunk object.

    Attributes:
        kernel_group_idx: The server's kernel-group index.
        recurrent: Whether the tensor holds recurrent state pages (its object
            group's kind).
        tokens_per_block: Tokens per paged block of the engine.
        kernel_block_size: Attention only: tokens per physical kernel page;
            smaller than ``tokens_per_block`` when the engine sub-pages each
            block (see ``LMCacheSDKContext._connect_sdk``).
        shape: The tensor's shape for one chunk, ``[planes, L, slots, W]``.
        dtype: The tensor's dtype.
    """

    kernel_group_idx: int
    recurrent: bool
    tokens_per_block: int
    kernel_block_size: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def num_bytes(self) -> int:
        """Bytes of the tensor for one chunk."""
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def sub_paged(self) -> bool:
        """Whether each block's K/V are interleaved at kernel-page granularity."""
        return (
            not self.recurrent
            and self.shape[0] == 2
            and 0 < self.kernel_block_size != self.tokens_per_block
        )


@dataclasses.dataclass(frozen=True)
class SDKObjectGroup:
    """One object group: the server stores one object per chunk for it.

    Attributes:
        object_group_idx: The server's object-group index.
        recurrent: Whether the object holds recurrent state (only the last
            chunk's is ever transferred).
        kernel_groups: The object's tensors, in the server's packing order.
        window_chunks: Trailing chunks of a range the server reads for the
            group (``num_chunks_in_sw``); -1 for all of them.
    """

    object_group_idx: int
    recurrent: bool
    kernel_groups: tuple[SDKKernelGroup, ...]
    window_chunks: int = -1

    @property
    def num_bytes(self) -> int:
        """Bytes of one chunk's object."""
        return sum(kg.num_bytes for kg in self.kernel_groups)


@dataclasses.dataclass(frozen=True)
class _Placement:
    """One object of a transfer and the pieces it occupies.

    Attributes:
        group: The object's group.
        chunk: The chunk's index within the batch (the last for recurrent).
        piece_ids: The pieces, in byte order.
    """

    group: SDKObjectGroup
    chunk: int
    piece_ids: list[int]


def _ring_pieces(layers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """View the ring's per-layer tensors as ``[num_blocks, piece_bytes]``."""
    return [layer.reshape(layer.shape[0], -1).view(torch.uint8) for layer in layers]


class RingSDKTransferWrapper:
    """Retrieve and store cached tensors through free blocks of the Q ring.

    The transfer wrapper of every SDK context, KV or query. A transfer larger
    than the ring's free blocks is split into batches of whole chunks; a
    hybrid model's recurrent state travels with the last batch. Transfers hold
    the ring's lock, shared by every context of the process.

    With several ranks, each rank's objects go through its own ring and the
    results are concatenated along the last (heads) dimension; this needs a
    layout without recurrent groups, and only retrieves.

    Args:
        ring: The engine's Q rings, opened (``connect()`` called).
        object_groups: The model's object groups, in server order.
        chunk_size: Tokens per LMCache chunk.
        timeout: Timeout in seconds for each server call.
        writable: Whether ``store`` is allowed (KV); query tensors are
            read-only.
    """

    def __init__(
        self,
        ring: RingMapping,
        object_groups: Sequence[SDKObjectGroup],
        chunk_size: int,
        timeout: float,
        writable: bool = True,
    ) -> None:
        self._ring = ring
        self._writable = writable
        self._object_groups = tuple(
            sorted(object_groups, key=lambda og: og.object_group_idx)
        )
        self._chunk_size = chunk_size
        self._timeout = timeout
        kernel_groups = [kg for og in self._object_groups for kg in og.kernel_groups]
        self._attention = sorted(
            (kg for kg in kernel_groups if not kg.recurrent),
            key=lambda kg: kg.kernel_group_idx,
        )
        self._recurrent = sorted(
            (kg for kg in kernel_groups if kg.recurrent),
            key=lambda kg: kg.kernel_group_idx,
        )
        if not self._attention:
            raise ValueError("the layout has no attention groups")

    @property
    def has_recurrent_state(self) -> bool:
        """Whether the model has recurrent groups (a hybrid model)."""
        return bool(self._recurrent)

    @property
    def device(self) -> torch.device:
        """The device the ring lives on."""
        return self._ring.device

    @property
    def window_chunks(self) -> int:
        """Trailing chunks of a range the server reads (-1: all of them).

        The attention groups' window; a recurrent group's last-chunk state is
        not part of the returned rows.
        """
        windows = {og.window_chunks for og in self._object_groups if not og.recurrent}
        if len(windows) != 1:
            raise ValueError(f"the attention groups disagree on windows: {windows}")
        return windows.pop()

    @property
    def rows_per_chunk(self) -> int:
        """Rows each chunk's attention object keeps (its slots per chunk)."""
        rows = {kg.shape[2] for kg in self._attention}
        if len(rows) != 1:
            raise ValueError(f"the attention groups disagree on rows: {rows}")
        return rows.pop()

    def retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[torch.Tensor, RecurrentState | None] | None:
        """Retrieve ``[key.start, key.end)`` through the engine's ring.

        ``[0, key.end)`` must be read-locked by a lookup under
        ``key.request_id``; the prefix ``[0, key.start)`` is released, and on
        failure the rest of the range.

        Args:
            key: The cache server key; ``key.token_bytes`` covers ``key.end``.
            instance_id: Unused: the ring belongs to the engine's instance.

        Returns:
            ``(tensor, state)``: the attention groups' rows as a CPU tensor
            ``[planes, L_attn, T, W]`` (true K and V for KV; ``planes == 1``
            for query, whose ``T`` counts the rows each chunk keeps and whose
            ``W`` holds every rank's heads), and for a hybrid model the
            recurrent state at ``key.end``, else None. None if the range is
            empty or the server fails a batch.

        Raises:
            RingStateError: If vLLM took ring blocks during the retrieve, the ring was
                re-registered, it has too few free blocks for one chunk, or
                several ranks serve a layout with recurrent groups.
        """
        del instance_id
        self._ring.free_lookup_locks(key, 0, key.start)
        if key.end <= key.start:
            return None
        with self._ring.lock:
            before = self._ring.get_rings()
            self._ring.map_rings(before)
            try:
                if len(before) > 1 and self._recurrent:
                    raise RingStateError(
                        "recurrent state cannot be split over "
                        f"{len(before)} ranks; SDK mode serves hybrid models at "
                        "TP 1"
                    )
                result = self._retrieve_batches(key, before)
            except RingStateError:
                self._ring.free_lookup_locks(key, key.start, key.end)
                raise
            after = self._ring.get_rings()
        check_ring_unchanged(before, after, "cached tensors")
        return result

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
            kv: Contiguous CPU tensor ``[planes, L_attn, T, W]`` (true K and
                V) with ``T == key.end - key.start``, chunk-aligned.
            state: Hybrid models only: the recurrent state stored as the state
                at ``key.end`` (as ``retrieve`` returns it).

        Returns:
            True if every batch was stored, False at the first failed batch.

        Raises:
            NotImplementedError: For read-only (query) tensors.
            ValueError: If ``kv`` or ``state`` does not match the layout.
            RingStateError: If vLLM took ring blocks during the store, the ring was
                re-registered, it has too few free blocks for one chunk, or
                several ranks are registered.
        """
        del instance_id
        if not self._writable:
            raise NotImplementedError("query tensors are read-only")
        self._check_store_inputs(key, kv, state)
        if key.end <= key.start:
            return True
        with self._ring.lock:
            before = self._ring.get_rings()
            self._ring.map_rings(before)
            stored = self._store_batches(key, kv, state, before)
            after = self._ring.get_rings()
        check_ring_unchanged(before, after, "KV tensors")
        return stored

    def close(self) -> None:
        """Drop the opened ring."""
        self._ring.close()

    # Planning

    def _single_ring(self, rings: Sequence[QRingInfo]) -> QRingInfo:
        if len(rings) != 1:
            raise RingStateError(
                f"SDK mode stores through exactly one Q ring (TP 1); "
                f"{len(rings)} are registered"
            )
        return rings[0]

    def _num_pieces(self, group: SDKObjectGroup, piece_bytes: int) -> int:
        return math.ceil(group.num_bytes / piece_bytes)

    def _free_pieces(self, info: QRingInfo, num_layers: int) -> list[int]:
        """The ring's free pieces, in block order."""
        return [
            block * num_layers + layer
            for block in sorted(info.free_blocks)
            for layer in range(num_layers)
        ]

    def _batches(
        self, key: IPCCacheServerKey, free_counts: Sequence[int], piece_bytes: int
    ) -> list[tuple[int, int]]:
        """Split the range into batches that fit every ring's free pieces.

        Args:
            key: The transfer's key.
            free_counts: Free pieces of each rank's ring.
            piece_bytes: Bytes per piece.

        Returns:
            The batches' token ranges.

        Raises:
            RingStateError: If not even one chunk fits a ring.
        """
        per_chunk = sum(
            self._num_pieces(og, piece_bytes)
            for og in self._object_groups
            if not og.recurrent
        )
        reserved = sum(
            self._num_pieces(og, piece_bytes)
            for og in self._object_groups
            if og.recurrent
        )
        chunks_per_batch = (min(free_counts) - reserved) // per_chunk
        if chunks_per_batch < 1:
            raise RingStateError(
                f"the Q ring has {min(free_counts)} free pieces; one chunk needs "
                f"{per_chunk + reserved}. Enlarge lmcache.mp.q.ring_blocks"
            )
        step = chunks_per_batch * self._chunk_size
        return [
            (start, min(start + step, key.end))
            for start in range(key.start, key.end, step)
        ]

    def _place(
        self,
        num_chunks: int,
        last: bool,
        free: list[int],
        piece_bytes: int,
        store: bool,
    ) -> list[_Placement]:
        """Assign pieces to a batch's objects, in the server's order.

        Object groups in ascending order, chunks in order. A recurrent group
        has only its last chunk, and only in the last batch. A retrieve reads
        an attention group within its window (the batch's trailing
        ``window_chunks``); a store writes all of its chunks.
        """
        placements: list[_Placement] = []
        cursor = 0
        for group in self._object_groups:
            if group.recurrent and not last:
                continue
            if group.recurrent:
                chunks: Sequence[int] = [num_chunks - 1]
            elif store or group.window_chunks < 0:
                chunks = range(num_chunks)
            else:
                chunks = range(max(0, num_chunks - group.window_chunks), num_chunks)
            count = self._num_pieces(group, piece_bytes)
            for chunk in chunks:
                placements.append(
                    _Placement(group, chunk, free[cursor : cursor + count])
                )
                cursor += count
        return placements

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
        placements: list[_Placement],
        device: torch.device,
    ) -> bool:
        """Run one SDK transfer, ordered after this stream's work so far.

        Must run with the ring's device current; on return the server's
        copies have completed.

        Args:
            method: The client's ``sdk_retrieve`` or ``sdk_store``.
            key: The batch's key.
            info: The ring the batch goes through.
            placements: The batch's objects and their pieces.
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
            sorted({p.group.object_group_idx for p in placements}),
            [piece_id for p in placements for piece_id in p.piece_ids],
            event_backend.export_event(event, device),
        ).to_device_future(device=device, event_backend=event_backend)
        return bool(future.result(timeout=self._timeout))

    # Retrieve

    def _retrieve_batches(
        self, key: IPCCacheServerKey, rings: Sequence[QRingInfo]
    ) -> tuple[torch.Tensor, RecurrentState | None] | None:
        """Retrieve the range batch by batch, each batch through every rank."""
        mapped = self._ring.rings
        pieces = [_ring_pieces(ring.layers) for ring in mapped]
        piece_bytes = pieces[0][0].shape[1]
        free = [
            self._free_pieces(info, len(rank_pieces))
            for info, rank_pieces in zip(rings, pieces, strict=True)
        ]
        batches = self._batches(key, [len(f) for f in free], piece_bytes)
        # rows[rank][kernel group], each a list of per-chunk CPU tensors.
        rows: list[dict[int, list[torch.Tensor]]] = [
            {kg.kernel_group_idx: [] for kg in self._attention} for _ in rings
        ]
        pages: list[torch.Tensor] = []
        for start, end in batches:
            for rank, info in enumerate(rings):
                device = mapped[rank].layers[0].device
                placements = self._place(
                    (end - start) // self._chunk_size,
                    end == key.end,
                    free[rank],
                    piece_bytes,
                    store=False,
                )
                with torch_dev.device(device):
                    ok = self._call(
                        self._ring.req_client.sdk_retrieve,
                        self._sub_key(key, info, start, end),
                        info,
                        placements,
                        device,
                    )
                    if not ok:
                        logger.warning(
                            "SDK retrieve failed for request_id=%s at [%d, %d) "
                            "on rank %d",
                            key.request_id,
                            start,
                            end,
                            rank,
                        )
                        self._ring.free_lookup_locks(key, end, key.end)
                        return None
                    for placement in placements:
                        self._decode(
                            self._read_object(pieces[rank], placement),
                            placement,
                            rows[rank],
                            pages,
                        )
        per_rank = [
            torch.cat(
                [
                    torch.cat(rank_rows[kg.kernel_group_idx], dim=2)
                    for kg in self._attention
                ],
                dim=1,
            )
            for rank_rows in rows
        ]
        state = RecurrentState(tuple(pages)) if self._recurrent else None
        return torch.cat(per_rank, dim=-1), state

    def _decode(
        self,
        tensors: list[torch.Tensor],
        placement: _Placement,
        rows: dict[int, list[torch.Tensor]],
        pages: list[torch.Tensor],
    ) -> None:
        """Move one object's tensors to CPU: attention rows, recurrent pages."""
        for kg, tensor in zip(placement.group.kernel_groups, tensors, strict=True):
            if kg.recurrent:
                pages.extend(
                    tensor[:, layer].to("cpu") for layer in range(tensor.shape[1])
                )
                continue
            if kg.sub_paged:
                tensor = subpaged_to_kv(
                    tensor, kg.tokens_per_block, kg.kernel_block_size
                )
            rows[kg.kernel_group_idx].append(tensor.to("cpu"))

    def _read_object(
        self, pieces: list[torch.Tensor], placement: _Placement
    ) -> list[torch.Tensor]:
        """Gather an object's bytes from its pieces and split its tensors."""
        num_layers = len(pieces)
        data = torch.cat(
            [
                pieces[piece_id % num_layers][piece_id // num_layers]
                for piece_id in placement.piece_ids
            ]
        )
        tensors = []
        offset = 0
        for kg in placement.group.kernel_groups:
            chunk = data[offset : offset + kg.num_bytes].clone()
            tensors.append(chunk.view(kg.dtype).view(kg.shape))
            offset += kg.num_bytes
        return tensors

    # Store

    def _check_store_inputs(
        self,
        key: IPCCacheServerKey,
        kv: torch.Tensor,
        state: RecurrentState | None,
    ) -> None:
        planes, width = self._attention[0].shape[0], self._attention[0].shape[3]
        num_layers = sum(kg.shape[1] for kg in self._attention)
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
        if not self._recurrent:
            if state is not None:
                raise ValueError("the model has no recurrent groups to store state in")
            return
        expected = [
            tuple(kg.shape[:1] + kg.shape[2:])
            for kg in self._recurrent
            for _ in range(kg.shape[1])
        ]
        if state is None or [tuple(p.shape) for p in state.pages] != expected:
            raise ValueError(
                f"state must hold one page per recurrent layer, shaped {expected}"
            )

    def _store_batches(
        self,
        key: IPCCacheServerKey,
        kv: torch.Tensor,
        state: RecurrentState | None,
        rings: Sequence[QRingInfo],
    ) -> bool:
        info = self._single_ring(rings)
        pieces = _ring_pieces(self._ring.rings[0].layers)
        piece_bytes = pieces[0].shape[1]
        free = self._free_pieces(info, len(pieces))
        batches = self._batches(key, [len(free)], piece_bytes)
        with torch_dev.device(self.device):
            for start, end in batches:
                placements = self._place(
                    (end - start) // self._chunk_size,
                    end == key.end,
                    free,
                    piece_bytes,
                    store=True,
                )
                for placement in placements:
                    data = self._encode_object(kv, state, placement, start - key.start)
                    self._write_object(pieces, placement.piece_ids, data)
                sub_key = self._sub_key(key, info, start, end)
                if not self._call(
                    self._ring.req_client.sdk_store,
                    sub_key,
                    info,
                    placements,
                    self.device,
                ):
                    logger.warning(
                        "SDK store failed for request_id=%s at [%d, %d)",
                        key.request_id,
                        start,
                        end,
                    )
                    return False
        return True

    def _encode_object(
        self,
        kv: torch.Tensor,
        state: RecurrentState | None,
        placement: _Placement,
        batch_offset: int,
    ) -> torch.Tensor:
        """Build one object's bytes on the ring's device."""
        parts = []
        for kg in placement.group.kernel_groups:
            if kg.recurrent:
                assert state is not None
                first = sum(
                    other.shape[1]
                    for other in self._recurrent
                    if other.kernel_group_idx < kg.kernel_group_idx
                )
                tensor = torch.stack(state.pages[first : first + kg.shape[1]], dim=1)
            else:
                first = sum(
                    other.shape[1]
                    for other in self._attention
                    if other.kernel_group_idx < kg.kernel_group_idx
                )
                token = batch_offset + placement.chunk * self._chunk_size
                tensor = kv[
                    :,
                    first : first + kg.shape[1],
                    token : token + self._chunk_size,
                ]
                if kg.sub_paged:
                    tensor = kv_to_subpaged(
                        tensor, kg.tokens_per_block, kg.kernel_block_size
                    )
            tensor = tensor.to(device=self.device, dtype=kg.dtype).contiguous()
            if tuple(tensor.shape) != kg.shape:
                raise ValueError(
                    f"kernel group {kg.kernel_group_idx} encodes as "
                    f"{tuple(tensor.shape)}, not {kg.shape}"
                )
            parts.append(tensor.reshape(-1).view(torch.uint8))
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
