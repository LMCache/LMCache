# SPDX-License-Identifier: Apache-2.0
"""HND restores through sharded device staging and a compiled swap."""

# Future
from __future__ import annotations

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.platform.devices.rbln.staging import HndStaging, shard_bounds


def _chunk_pairs(
    chunk: torch.Tensor,
    staged: torch.Tensor,
    lo: int,
    hi: int,
    token: int,
    block_size: int,
    blocks_per_chunk: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    layers = chunk.size(1)
    if blocks_per_chunk == 1:
        if lo == 0 and hi == 2 * layers:
            return [chunk], [staged.view_as(chunk)]
        regions, pieces = [], []
        # A shard may cross the K/V boundary; each half is a contiguous span.
        row = lo
        while row < hi:
            end = min(hi, (row // layers + 1) * layers)
            region = chunk[row // layers, row % layers : row % layers + end - row]
            regions.append(region)
            pieces.append(staged[row - lo : end - lo].view_as(region))
            row = end
        return regions, pieces
    # Multi-block token slices across layers are strided host views. copy_ may
    # make such a view contiguous on the host before DMA. Pair individual rows
    # instead, so both sides are contiguous and no temporary host copy is needed.
    return (
        [
            chunk[row // layers, row % layers, token : token + block_size]
            for row in range(lo, hi)
        ],
        [staged[row - lo].view(block_size, -1) for row in range(lo, hi)],
    )


def restore_hnd(
    layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    chunks: Sequence[torch.Tensor],
    blocks_per_chunk: int,
    skip_prefix_n_blocks: int,
    staged: HndStaging,
) -> None:
    """Restore blocks from canonical chunks into K/V-first paged views.

    Called by ``RblnDeviceOps`` after checking the HND requirements and chunk
    geometry and reserving staging. Each layer is ``[2, NB, NH, BS, HS]``;
    row ``r`` addresses ``layers[r % L][r // L][block]``. The shared set's
    lock covers the whole call, including all compiled swaps. Restores run
    serially on the model-driving thread's current stream.

    Args:
        layers: Prepared K/V-first paged views.
        block_ids: Paged block IDs in chunk-token order.
        chunks: Canonical host chunks shaped ``[2, L, bpc*BS, NH*HS]``.
        blocks_per_chunk: Positive number of blocks per chunk.
        skip_prefix_n_blocks: Leading blocks already present in the cache.
        staged: Shared restore buffers reserved for this geometry.

    Raises:
        RuntimeError: If a device copy or compiled swap fails.
    """
    num_layers = len(layers)
    block_size = layers[0].size(3)
    start = max(skip_prefix_n_blocks, 0)
    bounds = shard_bounds(2 * num_layers, len(staged.head_major))
    with staged.lock:
        for pos in range(start, len(block_ids)):
            slots, blocks, regions, pieces = [], [], [], []
            chunk = chunks[pos // blocks_per_chunk]
            token = (pos % blocks_per_chunk) * block_size
            for (lo, hi), head, tokens in zip(
                bounds, staged.head_major, staged.token_major, strict=True
            ):
                for row in range(lo, hi):
                    slots.append(head[row - lo])
                    blocks.append(
                        layers[row % num_layers][row // num_layers][block_ids[pos]]
                    )
                host, device = _chunk_pairs(
                    chunk, tokens, lo, hi, token, block_size, blocks_per_chunk
                )
                regions.extend(host)
                pieces.extend(device)
            torch._foreach_copy_(pieces, regions)
            for head, tokens in zip(staged.head_major, staged.token_major, strict=True):
                head.copy_(tokens.permute(0, 2, 1, 3))
            torch._foreach_copy_(blocks, slots)
