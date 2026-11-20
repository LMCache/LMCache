# SPDX-License-Identifier: Apache-2.0
"""Block KV transfer for TorchTPU's ``[NB, BS, NHP, P, HS]`` layout.

A TorchTPU tensor's ``data_ptr()`` addresses host-side buffer metadata, not
HBM, so blocks move with whole-tensor ops: ``index_select`` + ``.cpu()`` to
store, ``.to(device)`` + ``index_put_`` to retrieve. TorchTPU compiles one
program per operand shape, so the block index is padded to the next power of
two by repeating its last block; the extra rows are dropped on store and
rewrite that block with its own bytes on retrieve.
"""

# Future
from __future__ import annotations

# Standard
from typing import Sequence

# Third Party
import torch


def _padded_index(block_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    """Return ``block_ids`` as an int32 index on ``device``, padded to the next
    power of two by repeating the last id."""
    bucket = 1 << (len(block_ids) - 1).bit_length()
    padded = [*block_ids, *[block_ids[-1]] * (bucket - len(block_ids))]
    return torch.tensor(padded, dtype=torch.int32, device=device)


def gather_blocks_to_chunk(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole paged blocks into a token-major chunk.

    Args:
        paged_layers: Per-layer KV tensors, each ``[NB, BS, NHP, P, HS]``.
        block_ids: Blocks to gather, in chunk-token order; must be non-empty.
        dst: Host chunk shaped ``[L, T, NHP * P * HS]``; its leading
            ``len(block_ids) * BS`` tokens are written.
    """
    num_tokens = len(block_ids) * paged_layers[0].shape[1]
    index = _padded_index(block_ids, paged_layers[0].device)
    staged = torch.stack([layer.index_select(0, index) for layer in paged_layers])
    rows = staged.cpu()[:, : len(block_ids)]
    dst[:, :num_tokens].copy_(rows.reshape(len(paged_layers), num_tokens, -1))


def scatter_chunk_to_blocks(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter a token-major chunk back into whole paged blocks.

    Args:
        paged_layers: Per-layer KV tensors, each ``[NB, BS, NHP, P, HS]``.
        block_ids: Destination blocks, in chunk-token order.
        src: Host chunk shaped ``[L, T, NHP * P * HS]``.
        skip_prefix_n_blocks: Leading blocks neither read nor written.
    """
    block_shape = paged_layers[0].shape[1:]
    start = min(skip_prefix_n_blocks, len(block_ids))
    targets = block_ids[start:]
    if not targets:
        return
    index = _padded_index(targets, paged_layers[0].device)
    rows = src[:, start * block_shape[0] : len(block_ids) * block_shape[0]].reshape(
        len(paged_layers), len(targets), *block_shape
    )
    padding = index.shape[0] - len(targets)
    if padding:
        rows = torch.cat([rows, rows[:, -1:].expand(-1, padding, *block_shape)], dim=1)
    # Split per layer on the host so each device operand is a dense
    # bucket-shaped tensor rather than a view at a per-layer offset.
    for layer, layer_rows in zip(paged_layers, rows, strict=True):
        layer.index_put_((index,), layer_rows.to(layer.device))
