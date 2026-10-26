# SPDX-License-Identifier: Apache-2.0
"""Block KV transfer for Neuron's ``[2, NB, NH, BS, HS]`` layout.

Neuron rejects device-to-device copies between differently-sized parents and
``index_select``/``index_copy_`` cost time proportional to the whole cache, so
each run of consecutive blocks is copied straight between the paged cache and
a host buffer with ``narrow`` + ``copy_``. The head<->token transpose into the
token-major chunk happens on the host.
"""

# Future
from __future__ import annotations

# Standard
from typing import Sequence
import math
import threading

# Third Party
import torch

_STAGING = threading.local()


def _block_runs(block_ids: Sequence[int]) -> list[tuple[int, int, int]]:
    """Split ``block_ids`` into ``(position, first_block, length)`` runs of
    consecutive block ids."""
    runs: list[tuple[int, int, int]] = []
    for position, block in enumerate(block_ids):
        if runs:
            start, first, length = runs[-1]
            if block == first + length:
                runs[-1] = (start, first, length + 1)
                continue
        runs.append((position, block, 1))
    return runs


def _host_staging(
    num_layers: int,
    num_blocks: int,
    block_shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return this thread's reusable host buffer as ``[L, 2, B, NH, BS, HS]``."""
    buffers = getattr(_STAGING, "buffers", None)
    if buffers is None:
        buffers = _STAGING.buffers = {}
    shape = (num_layers, 2, num_blocks, *block_shape)
    numel = math.prod(shape)
    buf = buffers.get(dtype)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, device="cpu")
        buffers[dtype] = buf
    return buf[:numel].view(shape)


def _chunk_by_block(
    chunk: torch.Tensor, start: int, end: int, num_heads: int, block_size: int
) -> torch.Tensor:
    """View tokens of blocks ``[start, end)`` of a ``[2, L, T, NH*HS]`` chunk as
    ``[2, L, B, BS, NH, HS]``."""
    tokens = chunk.unflatten(-1, (num_heads, -1))
    return tokens[:, :, start * block_size : end * block_size].unflatten(
        2, (end - start, block_size)
    )


def gather_blocks_to_chunk(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole paged blocks into a token-major chunk.

    Args:
        paged_layers: Per-layer KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Blocks to gather, in chunk-token order.
        dst: Chunk shaped ``[2, L, T, NH*HS]``; its leading
            ``len(block_ids) * BS`` tokens are written.
    """
    _kv, _nb, num_heads, block_size, head_size = paged_layers[0].shape
    n_blocks = len(block_ids)
    staged = _host_staging(
        len(paged_layers),
        n_blocks,
        (num_heads, block_size, head_size),
        paged_layers[0].dtype,
    )
    for position, first, length in _block_runs(block_ids):
        for layer_idx, layer in enumerate(paged_layers):
            for kv in (0, 1):
                staged[layer_idx, kv, position : position + length].copy_(
                    layer[kv].narrow(0, first, length)
                )
    # [L, 2, B, NH, BS, HS] -> [2, L, B, BS, NH, HS]
    _chunk_by_block(dst, 0, n_blocks, num_heads, block_size).copy_(
        staged.permute(1, 0, 2, 4, 3, 5)
    )


def scatter_chunk_to_blocks(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter a token-major chunk back into whole paged blocks.

    Args:
        paged_layers: Per-layer KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Destination blocks, in chunk-token order.
        src: Chunk shaped ``[2, L, T, NH*HS]``.
        skip_prefix_n_blocks: Leading blocks neither read nor written.
    """
    _kv, _nb, num_heads, block_size, head_size = paged_layers[0].shape
    n_blocks = len(block_ids)
    start = min(skip_prefix_n_blocks, n_blocks)
    if start >= n_blocks:
        return
    staged = _host_staging(
        len(paged_layers),
        n_blocks - start,
        (num_heads, block_size, head_size),
        src.dtype,
    )
    staged.permute(1, 0, 2, 4, 3, 5).copy_(
        _chunk_by_block(src, start, n_blocks, num_heads, block_size)
    )
    for position, first, length in _block_runs(block_ids[start:]):
        for layer_idx, layer in enumerate(paged_layers):
            for kv in (0, 1):
                layer[kv].narrow(0, first, length).copy_(
                    staged[layer_idx, kv, position : position + length]
                )
