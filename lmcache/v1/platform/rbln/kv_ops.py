# SPDX-License-Identifier: Apache-2.0
"""Head-major block KV transfer for RBLN.

RBLN stores KV heads *before* block tokens (HND). Upstream's block transfer
stages each chunk token-major as ``[2, L, T, H*D]``, so filling it from an HND
paged buffer costs an on-device head<->token permute on every store and
restore.

These kernels write and read the chunk **head-major** as ``[2, L, H, T, D]``
instead, so no permute is ever issued. The chunk keeps the byte size upstream
sized it to -- the two layouts have identical element counts -- and is
reinterpreted with a free ``view``.

This is only sound because the multiprocess engine-driven path writes and reads
the chunk with the same code: the cache server treats it as an opaque byte
range, so the head-major interpretation round-trips. It must not be used where
a chunk crosses between the head-major and token-major worlds.

Implemented with torch ops only -- no compiled extension -- so the layout
contract above is what this module owns. Device behaviour is not: on RBLN the
same lines dispatch to the backend's v2v kernels against a KV cache whose
physical layout is sharded across chiplets, and neither that nor the transfer
cost has an equivalent when the tensors happen to be on CPU.
"""

# Future
from __future__ import annotations

# Standard
from typing import Sequence

# Third Party
import torch


def head_major_view(
    chunk: torch.Tensor,
    num_layers: int,
    num_heads: int,
    chunk_tokens: int,
    head_size: int,
) -> torch.Tensor:
    """Reinterpret a token-major-sized chunk buffer as head-major.

    Args:
        chunk: Contiguous buffer sized ``[2, L, T, H*D]`` by the caller.
        num_layers: Layers in the chunk.
        num_heads: KV heads per layer.
        chunk_tokens: Tokens in the chunk.
        head_size: Per-head dimension.

    Returns:
        torch.Tensor: A ``[2, L, H, T, D]`` view onto the same storage.

    Raises:
        ValueError: If ``chunk`` is not contiguous, so the reinterpretation
            would silently address the wrong bytes.
    """
    if not chunk.is_contiguous():
        raise ValueError(
            "head-major reinterpretation requires a contiguous chunk buffer"
        )
    return chunk.view(2, num_layers, num_heads, chunk_tokens, head_size)


def gather_blocks_head_major(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole paged blocks into a head-major chunk.

    Args:
        paged_layers: Per-layer HND KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Blocks to gather, in chunk-token order.
        dst: Destination view shaped ``[2, L, H, len(block_ids) * BS, D]``.
            May be on device (D2D) or host (D2H); ``copy_`` handles the
            transfer.
    """
    # Keeping the K/V axis in the per-layer view means one stack over layers
    # yields [2, L, H, BS, D] directly -- no separate k/v stacks and no
    # trailing recombine, so the gather is a single submit.
    pieces = [
        torch.stack([layer[:, block] for layer in paged_layers], dim=1)
        for block in block_ids
    ]
    gathered = torch.cat(pieces, dim=3) if len(pieces) > 1 else pieces[0]
    dst.copy_(gathered)


def scatter_head_major_to_blocks(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter a head-major chunk back into whole paged blocks.

    Args:
        paged_layers: Per-layer HND KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Destination blocks, in chunk-token order.
        src: Source view shaped ``[2, L, H, len(block_ids) * BS, D]``.
        skip_prefix_n_blocks: Leading blocks already present in the KV cache;
            neither read from ``src`` nor written.
    """
    block_size = paged_layers[0].shape[3]
    dsts: list[torch.Tensor] = []
    srcs: list[torch.Tensor] = []
    for position, block in enumerate(block_ids):
        if position < skip_prefix_n_blocks:
            continue
        window = src[:, :, :, position * block_size : (position + 1) * block_size, :]
        for layer_idx, layer in enumerate(paged_layers):
            dsts.append(layer[0, block])
            srcs.append(window[0, layer_idx])
            dsts.append(layer[1, block])
            srcs.append(window[1, layer_idx])
    if dsts:
        torch._foreach_copy_(dsts, srcs)
