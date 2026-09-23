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
The HND staging buffer is therefore host memory.

The MLA layout (``[NB, BS, HS]`` per layer, chunk ``[L, T, HS]``) has no
transpose: the chunk is the blocks laid end to end. What costs there is the
number of DMAs, not bytes -- on RBLN a device<->host copy pays a large fixed
cost, so ``L * B`` per-block copies run several times slower than the same
bytes in one copy. The MLA path therefore stages on the *device*: blocks are
gathered into (or fanned out of) a persistent device buffer with
device-to-device copies, and the chunk crosses the device boundary in one
contiguous DMA. Every copy is a whole contiguous block, so it takes
torch-rbln's direct ``memcpy_v2v`` path with no index tensor and no
``submit_or_fallback`` CPU fallback behind it.

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


def gather_blocks_to_chunk_hnd(
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


def scatter_chunk_to_blocks_hnd(
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


def _device_staging(
    n_layers: int, n_tokens: int, hidden: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return this thread's contiguous ``[n_layers, n_tokens, hidden]`` device buffer.

    One flat allocation per (``dtype``, ``device``) per thread, grown to the
    largest element count asked for; the requested shape is a view of its
    leading elements, so it is contiguous for any ``n_tokens`` (a trailing
    short chunk or a prefix skip) and the single DMA that moves it stays a
    direct copy. Per thread because the multiprocess server runs transfers on
    a thread pool; a shared buffer would let one transfer overwrite another's
    staged bytes between the two legs.

    Args:
        n_layers: Layers in the chunk.
        n_tokens: Tokens the caller needs room for.
        hidden: Per-token width (``HS`` for MLA).
        dtype: Element type, matching the chunk and the paged tensors.
        device: Device the paged tensors live on.

    Returns:
        torch.Tensor: A contiguous ``[n_layers, n_tokens, hidden]`` view owned
        by the calling thread.
    """
    buffers = getattr(_STAGING, "device_buffers", None)
    if buffers is None:
        buffers = _STAGING.device_buffers = {}
    key = (dtype, device)
    numel = n_layers * n_tokens * hidden
    buf = buffers.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, device=device)
        buffers[key] = buf
    return buf[:numel].view(n_layers, n_tokens, hidden)


def _mla_block_pairs(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    staged: torch.Tensor,
    block_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pair every ``layer[block]`` with its token window in ``staged``.

    Both sides are contiguous ``[BS, HS]`` views -- ``layer[block]`` of a
    contiguous ``[NB, BS, HS]`` layer, and a token slice of one layer of the
    ``[L, B*BS, HS]`` staging buffer -- so each pair is one direct
    device-to-device copy.

    Returns:
        tuple: ``(blocks, slots)`` in matching order, for ``torch._foreach_copy_``.
    """
    blocks: list[torch.Tensor] = []
    slots: list[torch.Tensor] = []
    for layer_idx, layer in enumerate(paged_layers):
        for position, block in enumerate(block_ids):
            blocks.append(layer[block])
            slots.append(
                staged[layer_idx, position * block_size : (position + 1) * block_size]
            )
    return blocks, slots


def gather_blocks_to_chunk_mla(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole MLA paged blocks into a single-plane chunk.

    The blocks are collected into this thread's device staging buffer with
    one batch of device-to-device block copies, then the whole window crosses
    the device boundary in a single contiguous copy. No index tensor is
    involved, so nothing is read back to the host per layer and no op in the
    sequence has a CPU-fallback path behind it.

    Args:
        paged_layers: Per-layer MLA KV tensors, each contiguous ``[NB, BS, HS]``.
        block_ids: Blocks to gather, in chunk-token order.
        dst: Chunk shaped ``[L, T, HS]``. Only its leading
            ``len(block_ids) * BS`` tokens are written, so a trailing chunk
            holding fewer blocks than it was sized for is fine. May be on
            device (D2D) or host (D2H); ``copy_`` handles the transfer.
    """
    n_blocks = len(block_ids)
    _nb, block_size, head_size = paged_layers[0].shape
    staged = _device_staging(
        len(paged_layers),
        n_blocks * block_size,
        head_size,
        paged_layers[0].dtype,
        paged_layers[0].device,
    )
    blocks, slots = _mla_block_pairs(paged_layers, block_ids, staged, block_size)
    torch._foreach_copy_(slots, blocks)
    dst[:, : n_blocks * block_size].copy_(staged)


def scatter_chunk_to_blocks_mla(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter a single-plane chunk back into whole MLA paged blocks.

    Mirror of :func:`gather_blocks_to_chunk_mla`: the chunk window lands in
    this thread's device staging buffer in one contiguous copy, then one batch
    of device-to-device block copies fans it out into the paged layers.

    Args:
        paged_layers: Per-layer MLA KV tensors, each contiguous ``[NB, BS, HS]``.
        block_ids: Destination blocks, in chunk-token order.
        src: Chunk shaped ``[L, T, HS]``. Only the token windows the blocks
            map to are read, so a trailing chunk holding fewer blocks than it
            was sized for is fine.
        skip_prefix_n_blocks: Leading blocks already present in the KV cache;
            neither read from ``src`` nor written.
    """
    n_blocks = len(block_ids)
    start = min(skip_prefix_n_blocks, n_blocks)
    if start >= n_blocks:
        return
    _nb, block_size, head_size = paged_layers[0].shape
    n_valid = n_blocks - start
    staged = _device_staging(
        len(paged_layers),
        n_valid * block_size,
        head_size,
        src.dtype,
        paged_layers[0].device,
    )
    # A prefix skip or a short trailing chunk makes the window a strided host
    # view; ``copy_`` contiguizes it on the host before the one DMA.
    staged.copy_(src[:, start * block_size : n_blocks * block_size])
    blocks, slots = _mla_block_pairs(
        paged_layers, list(block_ids)[start:n_blocks], staged, block_size
    )
    torch._foreach_copy_(blocks, slots)
