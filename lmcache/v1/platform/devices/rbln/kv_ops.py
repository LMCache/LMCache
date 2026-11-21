# SPDX-License-Identifier: Apache-2.0
"""RBLN block transfers through device staging kept per geometry.

Chunks keep LMCache's canonical layout, so a chunk stored from an RBLN cache
can be restored into a non-RBLN one and back. What is RBLN-specific is the
sequence of device ops, which depends on the layout:

- **HND** (``[NB, 2, NH, BS, HS]`` views) needs a head<->token swap to reach
  the token-major chunk. The staging is a head-major and a token-major
  tensor, each placed in one slice of rows per chiplet, swapped by a compiled
  program built before serving.
- **MLA** (``[NB, BS, HS]`` per layer, chunk ``[L, T, HS]``) has no transpose:
  the chunk is the blocks laid end to end. What costs is the number of DMAs,
  not bytes -- on RBLN a device<->host copy pays a large fixed cost. One block
  of every layer is staged in an ``[L, BS, HS]`` buffer, so it crosses the
  device boundary in one DMA when a chunk holds one block, which is how RBLN
  runs LMCache.

Each layout is one class. Both move one block at a time: a multi-block
chunk's token window is strided across layers on the host, so it moves as
contiguous per-row DMAs instead of through a temporary host copy, and every
device-to-device copy is a whole contiguous block, which takes torch-rbln's
direct ``memcpy_v2v`` path.

:func:`prepare_transfers` allocates a geometry's staging, one set per
direction, and builds HND's swap programs on it, once, before serving.
Transfers reuse that staging instead of allocating their own: a newly
allocated split tensor makes its first swap and host copy far slower than
later ones, and staging kept from preparation is already allocated when the
engine sizes its KV cache. LMCache runs stores on several commit workers, so
the stores of one geometry take turns on its store set, holding its lock for
the whole store; restores run on the model-driving thread with their own set.
Each kept program is therefore run by one thread at a time, which rebel's
runtime requires (it fails with ``INIT_INTERNAL`` otherwise).
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import Sequence
import functools
import threading

# Third Party
import torch

# Every geometry prepared in this process: (layers, dtype, one block's shape)
# to its prepared transfer. Written only by preparation, before transfers
# start; entries live for the process.
_PREPARED: dict[tuple[int, torch.dtype, tuple[int, ...]], Transfer] = {}


def _geometry(
    layers: Sequence[torch.Tensor],
) -> tuple[int, torch.dtype, tuple[int, ...]]:
    return len(layers), layers[0].dtype, tuple(layers[0].shape[1:])


def _split_staging(first: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate head-major and token-major staging split across the chiplets.

    ``first`` is a blocks-first HND view ``[NB, 2, NH, BS, HS]``. The ``rows``
    (``2 * L``) rows are placed in contiguous slices on ``min(num_chiplet, L)``
    chiplets.
    """
    _, _, num_heads, block_size, head_size = first.shape
    chiplets = torch.rbln.get_device_properties(first.device).num_chiplet
    slices = min(chiplets, rows // 2)
    placements = [
        torch.rbln.DevicePlacement(
            0, s, ((0, rows * s // slices, rows * (s + 1) // slices),)
        )
        for s in range(slices)
    ]
    head = torch.empty(
        (rows, num_heads, block_size, head_size), dtype=first.dtype, device=first.device
    )
    tokens = torch.empty(
        (rows, block_size, num_heads, head_size), dtype=first.dtype, device=first.device
    )
    for buffer in (head, tokens):
        torch.rbln.set_device_alloc(buffer, placements)
    return head, tokens


def _kept_program(src: torch.Tensor, dst: torch.Tensor) -> Callable[..., object]:
    """Run the swap from ``src`` into ``dst`` once and return its program's run.

    The swap runs inside ``torch.rbln.capture_programs()``, and the built
    program's ``runtime.run`` is returned.

    Raises:
        RuntimeError: If the swap does not build exactly one program. torch-rbln
            builds none for KV that is not bf16 with a head size a multiple of
            64, or for a swap of this shape that already ran in the process.
    """
    with torch.rbln.capture_programs() as built:
        dst.copy_(src.permute(0, 2, 1, 3))
    if not built:
        raise RuntimeError(
            f"torch-rbln built no program for the RBLN HND swap of "
            f"{tuple(src.shape)} {src.dtype}: HND needs bf16 KV with a head size "
            "a multiple of 64, and the swap must not have run before LMCache "
            "prepared it"
        )
    if len(built) > 1:
        raise RuntimeError(f"RBLN HND swap built {len(built)} programs; expected one")
    return built[0].runtime.run


def _chunk_pairs(
    chunk: torch.Tensor,
    staged: torch.Tensor,
    token: int,
    block_size: int,
    blocks_per_chunk: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if blocks_per_chunk == 1:
        return [chunk], [staged.view_as(chunk)]
    # Multi-block token slices across layers are strided host views. copy_ may
    # make such a view contiguous on the host before DMA. Pair individual rows
    # instead, so both sides are contiguous and no temporary host copy is needed.
    layers = chunk.size(1)
    return (
        [
            chunk[row // layers, row % layers, token : token + block_size]
            for row in range(2 * layers)
        ],
        [staged[row].view(block_size, -1) for row in range(2 * layers)],
    )


class HndTransfer:
    """One HND geometry's kept staging and swap programs.

    Each direction keeps a head-major ``[2L, NH, BS, HS]`` and a token-major
    ``[2L, BS, NH, HS]`` staging tensor split across the chiplets, and the
    swap program built on them at construction, inside
    ``torch.rbln.capture_programs()``. A transfer only calls the kept
    ``runtime.run``. Building on a commit worker instead would race the model
    thread on torch's process-wide export state, and torch-rbln's own lookup
    can miss later (for example after ``torch._dynamo.reset()``) and build
    again.
    """

    def __init__(self, layers: Sequence[torch.Tensor]) -> None:
        """Allocate both directions' staging and build their swaps.

        Args:
            layers: Blocks-first HND views, each ``[NB, 2, NH, BS, HS]``.

        Raises:
            RuntimeError: If allocation, placement or a swap build fails; see
                :func:`_kept_program`.
        """
        first = layers[0]
        _, _, num_heads, block_size, _ = first.shape
        rows = 2 * len(layers)
        self._staging = {store: _split_staging(first, rows) for store in (True, False)}
        store_head, store_tokens = self._staging[True]
        restore_head, restore_tokens = self._staging[False]
        store_run = _kept_program(store_head, store_tokens)
        # With NH == BS both swaps have one shape and torch-rbln serves the
        # restore swap from the store's program, so the two directions share
        # that program and must share its lock.
        shared = num_heads == block_size
        if shared:
            restore_run = store_run
            restore_run(restore_tokens, out=[restore_head])
        else:
            restore_run = _kept_program(restore_tokens, restore_head)
        self._swaps = {
            True: functools.partial(store_run, store_head, out=[store_tokens]),
            False: functools.partial(restore_run, restore_tokens, out=[restore_head]),
        }
        store_lock = threading.Lock()
        self._locks = {
            True: store_lock,
            False: store_lock if shared else threading.Lock(),
        }

    def move(
        self,
        layers: Sequence[torch.Tensor],
        block_ids: Sequence[int],
        chunks: Sequence[torch.Tensor],
        blocks_per_chunk: int,
        skip_prefix_n_blocks: int,
        *,
        store: bool,
    ) -> None:
        """Move blocks between HND views and canonical chunks.

        Row ``r`` of the staging addresses ``layers[r % L][block][r // L]``.
        Holds the direction's lock for the whole call.

        Args:
            layers: Blocks-first HND views, each ``[NB, 2, NH, BS, HS]``.
            block_ids: Paged block IDs in chunk-token order.
            chunks: Canonical host chunks shaped ``[2, L, bpc*BS, NH*HS]``.
            blocks_per_chunk: Positive number of blocks per chunk.
            skip_prefix_n_blocks: Leading blocks to skip during restore.
            store: Copy from paged blocks to host chunks.

        Raises:
            RuntimeError: If a device copy or a swap fails.
        """
        num_layers = len(layers)
        block_size = layers[0].size(3)
        head, tokens = self._staging[store]
        slots = list(head)
        swap = self._swaps[store]
        start = 0 if store else max(skip_prefix_n_blocks, 0)
        with self._locks[store]:
            for pos in range(start, len(block_ids)):
                block = block_ids[pos]
                blocks = [
                    layers[row % num_layers][block][row // num_layers]
                    for row in range(2 * num_layers)
                ]
                regions, pieces = _chunk_pairs(
                    chunks[pos // blocks_per_chunk],
                    tokens,
                    (pos % blocks_per_chunk) * block_size,
                    block_size,
                    blocks_per_chunk,
                )
                if store:
                    torch._foreach_copy_(slots, blocks)
                    swap()
                    torch._foreach_copy_(regions, pieces)
                else:
                    torch._foreach_copy_(pieces, regions)
                    swap()
                    torch._foreach_copy_(blocks, slots)


class MlaTransfer:
    """One MLA geometry's kept staging; MLA moves blocks with plain copies."""

    def __init__(self, layers: Sequence[torch.Tensor]) -> None:
        """Allocate one ``[L, BS, HS]`` staging buffer per direction.

        Args:
            layers: Per-layer MLA KV tensors, each contiguous ``[NB, BS, HS]``.

        Raises:
            RuntimeError: If allocation fails.
        """
        first = layers[0]
        self._staging = {
            store: torch.empty(
                (len(layers), *first.shape[1:]), dtype=first.dtype, device=first.device
            )
            for store in (True, False)
        }
        self._locks = {store: threading.Lock() for store in (True, False)}

    def move(
        self,
        layers: Sequence[torch.Tensor],
        block_ids: Sequence[int],
        chunks: Sequence[torch.Tensor],
        blocks_per_chunk: int,
        skip_prefix_n_blocks: int,
        *,
        store: bool,
    ) -> None:
        """Move whole MLA blocks between paged layers and canonical chunks.

        Holds the direction's lock for the whole call.

        Args:
            layers: Per-layer MLA KV tensors, each contiguous ``[NB, BS, HS]``.
            block_ids: Paged block IDs in chunk-token order.
            chunks: Canonical host chunks shaped ``[L, bpc*BS, HS]``. A
                trailing chunk may hold fewer blocks than it was sized for;
                its unused tokens are left untouched.
            blocks_per_chunk: Positive number of blocks per chunk.
            skip_prefix_n_blocks: Leading blocks to skip during restore.
            store: Copy from paged blocks to host chunks.

        Raises:
            RuntimeError: If a device copy fails.
        """
        num_layers, block_size = len(layers), layers[0].size(1)
        staged = self._staging[store]
        slots = list(staged)
        start = 0 if store else max(skip_prefix_n_blocks, 0)
        with self._locks[store]:
            for pos in range(start, len(block_ids)):
                chunk = chunks[pos // blocks_per_chunk]
                blocks = [layer[block_ids[pos]] for layer in layers]
                if blocks_per_chunk == 1:
                    regions, pieces = [chunk], [staged]
                else:
                    token = (pos % blocks_per_chunk) * block_size
                    regions = [
                        chunk[layer, token : token + block_size]
                        for layer in range(num_layers)
                    ]
                    pieces = slots
                if store:
                    torch._foreach_copy_(slots, blocks)
                    torch._foreach_copy_(regions, pieces)
                else:
                    torch._foreach_copy_(pieces, regions)
                    torch._foreach_copy_(blocks, slots)


Transfer = HndTransfer | MlaTransfer


def prepare_transfers(layers: Sequence[torch.Tensor], kind: type[Transfer]) -> None:
    """Prepare these layers' geometry for transfers, unless it already is.

    Call this from one thread, the one that drives the model, before
    transfers start and before the engine sizes its KV cache, so HND's
    programs are built before serving and the staging is counted as used.
    Every prepared geometry, with its staging, stays for the life of the
    process. Preparing a geometry again does nothing.

    Args:
        layers: Blocks-first per-layer views (``[NB, ...]``) of one KV cache
            group, as :func:`transfer` will receive them.
        kind: The layout's transfer class.

    Raises:
        RuntimeError: If preparing fails; the geometry stays unprepared.
    """
    geometry = _geometry(layers)
    if geometry not in _PREPARED:
        _PREPARED[geometry] = kind(layers)


def transfer(
    layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    chunks: Sequence[torch.Tensor],
    blocks_per_chunk: int,
    skip_prefix_n_blocks: int,
    *,
    store: bool,
) -> None:
    """Move blocks between paged layers and canonical chunks.

    Called by ``RblnDeviceOps`` after validating the layers and chunks. Safe
    to call from several threads at once: the transfers of one geometry and
    direction take turns on its staging. A restore must run on the
    model-driving thread's current stream.

    Args:
        layers: Blocks-first per-layer views, as :func:`prepare_transfers`
            received them.
        block_ids: Paged block IDs in chunk-token order.
        chunks: Canonical host chunks for the layout.
        blocks_per_chunk: Positive number of blocks per chunk.
        skip_prefix_n_blocks: Leading blocks to skip during restore.
        store: Copy from paged blocks to host chunks.

    Raises:
        RuntimeError: Before any copy, if :func:`prepare_transfers` has not
            prepared this geometry; otherwise if a device copy or a swap
            fails.
    """
    prepared = _PREPARED.get(_geometry(layers))
    if prepared is None:
        raise RuntimeError(
            f"RBLN transfer of {_geometry(layers)} was not prepared; the engine "
            "must call warm_up_kv_transfer with its KV caches before serving"
        )
    prepared.move(
        layers, block_ids, chunks, blocks_per_chunk, skip_prefix_n_blocks, store=store
    )
