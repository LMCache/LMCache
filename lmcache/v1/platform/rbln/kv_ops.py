# SPDX-License-Identifier: Apache-2.0
"""Block KV transfer for RBLN, tuned for its torch op ordering.

Chunks are staged in LMCache's canonical token-major wire layout,
``[2, L, T, H*D]`` -- the same bytes any other device writes -- so a chunk
stored from an RBLN cache can be restored into a non-RBLN one and back. What
is RBLN-specific here is the sequence of torch ops used to get there, not the
layout that comes out.

RBLN stores heads before block tokens (HND), so writing a token-major chunk
needs a head<->token transpose. It runs on the host, against host memory
bandwidth, rather than as part of the device<->host copy: folding it in there
would leave every copy across that boundary strided instead of contiguous.

Implemented with torch ops only -- no compiled extension.
"""

# Future
from __future__ import annotations

# Standard
from typing import Sequence
import threading

# Third Party
import torch

#: Host landing buffers for the device<->host leg, per thread.
_STAGING = threading.local()


def _host_staging(
    n_blocks: int, per_block: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Return this thread's contiguous host buffer shaped ``[n_blocks, *per_block]``.

    One buffer per (``per_block``, ``dtype``) per thread, grown to the largest
    block count asked for and sliced down, so a varying prefix skip or a
    trailing short chunk reuses the same allocation instead of adding another
    resident one. Slicing the leading dimension keeps it contiguous, which the
    per-(block, layer) copies depend on.

    Args:
        n_blocks: Blocks the caller needs room for.
        per_block: Shape of one block's staging area.
        dtype: Element type, matching the chunk and the paged tensors.

    Returns:
        torch.Tensor: A contiguous view with ``n_blocks`` leading entries,
        owned by the calling thread.
    """
    buffers = getattr(_STAGING, "buffers", None)
    if buffers is None:
        buffers = _STAGING.buffers = {}
    key = (per_block, dtype)
    buf = buffers.get(key)
    if buf is None or buf.shape[0] < n_blocks:
        buf = torch.empty((n_blocks, *per_block), dtype=dtype, device="cpu")
        buffers[key] = buf
    return buf[:n_blocks]


def gather_blocks_to_chunk(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole paged blocks into a token-major chunk.

    Args:
        paged_layers: Per-layer HND KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Blocks to gather, in chunk-token order.
        dst: Chunk shaped ``[2, L, T, H*D]``. Only its leading
            ``len(block_ids) * BS`` tokens are written, so a trailing chunk
            holding fewer blocks than it was sized for is fine. May be on
            device (D2D) or host (D2H); ``copy_`` handles the transfer.
    """
    _kv, _nb, num_heads, block_size, head_size = paged_layers[0].shape
    n_blocks = len(block_ids)
    # K and V are addressed separately: ``layer[:, block]`` spans both halves
    # of the layer and is not contiguous, ``layer[kv, block]`` is. With
    # ``staged`` laid out [B, L, 2, H, BS, D] its slot matches exactly, so
    # every copy is one contiguous run and nothing is gathered on the device.
    staged = _host_staging(
        n_blocks,
        (len(paged_layers), 2, num_heads, block_size, head_size),
        paged_layers[0].dtype,
    )
    dsts: list[torch.Tensor] = []
    srcs: list[torch.Tensor] = []
    for position, block in enumerate(block_ids):
        for layer_idx, layer in enumerate(paged_layers):
            dsts.append(staged[position, layer_idx, 0])
            srcs.append(layer[0, block])
            dsts.append(staged[position, layer_idx, 1])
            srcs.append(layer[1, block])
    torch._foreach_copy_(dsts, srcs)
    # Both splits stay views; reshape() would not, and would rebuild the very
    # chunk-sized temporary this buffer exists to avoid.
    tokens = dst.unflatten(-1, (num_heads, head_size))
    by_block = tokens[:, :, : n_blocks * block_size].unflatten(
        2, (n_blocks, block_size)
    )
    # staged [B, L, 2, H, BS, D] -> [2, L, B, BS, H, D], matching by_block.
    by_block.copy_(staged.permute(2, 1, 0, 4, 3, 5))


def scatter_chunk_to_blocks(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter a token-major chunk back into whole paged blocks.

    Args:
        paged_layers: Per-layer HND KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Destination blocks, in chunk-token order.
        src: Chunk shaped ``[2, L, T, H*D]``. Only the token windows the
            blocks map to are read, so a trailing chunk holding fewer blocks
            than it was sized for is fine.
        skip_prefix_n_blocks: Leading blocks already present in the KV cache;
            neither read from ``src`` nor written.
    """
    _kv, _nb, num_heads, block_size, head_size = paged_layers[0].shape
    n_blocks = len(block_ids)
    start = min(skip_prefix_n_blocks, n_blocks)
    if start >= n_blocks:
        return
    tokens = src.unflatten(-1, (num_heads, head_size))

    # Mirror of the gather: transpose on the host first, so each copy back is
    # a contiguous [H, BS, D] block.
    n_staged = n_blocks - start
    staged = _host_staging(
        n_staged,
        (len(paged_layers), 2, num_heads, block_size, head_size),
        src.dtype,
    )
    by_block = tokens[:, :, start * block_size : n_blocks * block_size].unflatten(
        2, (n_staged, block_size)
    )
    # staged [B, L, 2, H, BS, D] -> [2, L, B, BS, H, D], matching by_block.
    staged.permute(2, 1, 0, 4, 3, 5).copy_(by_block)

    dsts: list[torch.Tensor] = []
    srcs: list[torch.Tensor] = []
    for position in range(start, n_blocks):
        block = block_ids[position]
        for layer_idx, layer in enumerate(paged_layers):
            # Both sides are contiguous [H, BS, D], so each copy is one run.
            dsts.append(layer[0, block])
            srcs.append(staged[position - start, layer_idx, 0])
            dsts.append(layer[1, block])
            srcs.append(staged[position - start, layer_idx, 1])
    torch._foreach_copy_(dsts, srcs)
