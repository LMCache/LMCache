# SPDX-License-Identifier: Apache-2.0
"""Path-agnostic helpers for multiprocess transfer planning."""

# Standard
from itertools import islice
from typing import Generator, Sequence, TypeVar

ItemT = TypeVar("ItemT")


def has_sufficient_block_ids(
    block_ids: Sequence[Sequence[int]],
    blocks_per_chunk: Sequence[int],
    num_chunks: int,
) -> bool:
    """Return whether every group has enough block IDs for all chunks.

    Args:
        block_ids: Per-group block-ID sequences.
        blocks_per_chunk: Per-group number of blocks in one chunk.
        num_chunks: Number of chunks that must be covered.

    Returns:
        True if each group has at least ``num_chunks * blocks_per_chunk[group]``
        block IDs.

    Raises:
        ValueError: If ``num_chunks`` is negative, if any entry in
            ``blocks_per_chunk`` is less than 1, or if the two per-group
            sequences have different lengths.
    """
    if num_chunks < 0:
        raise ValueError("num_chunks must be non-negative")
    if any(group_blocks < 1 for group_blocks in blocks_per_chunk):
        raise ValueError("blocks_per_chunk entries must be at least one")
    return all(
        len(group_block_ids) >= num_chunks * group_blocks
        for group_block_ids, group_blocks in zip(
            block_ids, blocks_per_chunk, strict=True
        )
    )


def select_block_ids_for_window(
    block_ids: Sequence[int],
    total_blocks_per_chunk: int,
    keep_blocks_per_chunk: int,
) -> list[int]:
    """Select the trailing per-chunk block IDs required by an attention window.

    Args:
        block_ids: Block IDs for one kernel group across all chunks.
        total_blocks_per_chunk: Total number of blocks in one LMCache chunk.
        keep_blocks_per_chunk: Number of trailing blocks to keep per chunk.

    Returns:
        A new list containing the selected block IDs.

    Raises:
        ValueError: If block geometry is invalid or if ``block_ids`` does not
            contain complete chunks.
    """
    if total_blocks_per_chunk < 1:
        raise ValueError("total_blocks_per_chunk must be at least one")
    if keep_blocks_per_chunk < 1:
        raise ValueError("keep_blocks_per_chunk must be at least one")
    if keep_blocks_per_chunk > total_blocks_per_chunk:
        raise ValueError(
            "keep_blocks_per_chunk must be less than or equal to total_blocks_per_chunk"
        )
    if len(block_ids) % total_blocks_per_chunk != 0:
        raise ValueError("len(block_ids) must be a multiple of total_blocks_per_chunk")

    selected_block_ids: list[int] = []
    for start_idx in range(0, len(block_ids), total_blocks_per_chunk):
        chunk_block_ids = block_ids[start_idx : start_idx + total_blocks_per_chunk]
        selected_block_ids.extend(chunk_block_ids[-keep_blocks_per_chunk:])
    return selected_block_ids


def downsample_block_ids(
    block_ids: Sequence[Sequence[int]],
    blocks_per_chunk: Sequence[int],
    blocks_per_window: Sequence[int],
) -> list[list[int]]:
    """Downsample block IDs for each kernel group based on its keep window.

    Args:
        block_ids: Per-group block-ID sequences.
        blocks_per_chunk: Per-group total blocks per chunk.
        blocks_per_window: Per-group trailing blocks to keep per chunk.

    Returns:
        A new per-group block-ID list in the original group ordering.

    Raises:
        ValueError: If per-group sequence lengths differ, or group geometry is
            invalid for any group.
    """
    return [
        select_block_ids_for_window(group_block_ids, total_blocks, keep_blocks)
        for group_block_ids, total_blocks, keep_blocks in zip(
            block_ids, blocks_per_chunk, blocks_per_window, strict=True
        )
    ]


def compute_num_objects_to_skip(
    sw_size_chunks: int,
    num_objects: int,
    is_retrieve: bool,
) -> int:
    """Compute how many leading objects should be skipped for transfer.

    Args:
        sw_size_chunks: Attention-window size in chunks for the object group.
            ``-1`` means full attention and ``>=1`` means sliding window.
        num_objects: Number of objects in the transfer list.
        is_retrieve: Whether the transfer direction is retrieve (H2D).

    Returns:
        Number of leading objects to skip.

    Raises:
        ValueError: If ``sw_size_chunks`` is 0 or less than -1, or if
            ``num_objects`` is negative.
    """
    if sw_size_chunks != -1 and sw_size_chunks < 1:
        raise ValueError("sw_size_chunks must be -1 (full) or at least one")
    if num_objects < 0:
        raise ValueError("num_objects must be non-negative")
    if sw_size_chunks == -1:
        return 0
    if not is_retrieve:
        return 0
    return max(0, num_objects - sw_size_chunks)


def batched_iteration_with_skip(
    sequence: Sequence[ItemT],
    batch_size: int,
    skip_count: int,
) -> Generator[tuple[int, tuple[ItemT, ...]], None, None]:
    """Iterate over a sequence in batches after skipping a leading prefix.

    Args:
        sequence: The sequence to iterate over.
        batch_size: Number of items per yielded batch.
        skip_count: Number of leading items to skip.

    Yields:
        Tuples ``(batch_start_idx, batch)`` where ``batch_start_idx`` is in the
        original sequence coordinate space and ``batch`` is a tuple of values.

    Raises:
        ValueError: If ``batch_size`` is less than 1 or ``skip_count`` is
            negative.

    Note:
        If ``skip_count`` exceeds ``len(sequence)``, the iterator is exhausted
        and no batches are yielded.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    if skip_count < 0:
        raise ValueError("skip_count must be non-negative")

    seq_iter = iter(sequence)
    for _ in range(skip_count):
        next(seq_iter, None)

    batch_start_idx = skip_count
    while batch := tuple(islice(seq_iter, batch_size)):
        yield batch_start_idx, batch
        batch_start_idx += len(batch)


def recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Map chunk-space skip blocks into downsampled-window block space.

    Args:
        blocks_per_chunk: Total blocks in one chunk.
        blocks_per_window: Retained trailing blocks in one chunk.
        blocks_to_skip: Blocks to skip in full-chunk coordinates.

    Returns:
        The skip count in downsampled-window coordinates.

    Raises:
        ValueError: If geometry is invalid or ``blocks_to_skip`` is negative.
    """
    if blocks_per_chunk < 1:
        raise ValueError("blocks_per_chunk must be at least one")
    if blocks_per_window < 1:
        raise ValueError("blocks_per_window must be at least one")
    if blocks_per_window > blocks_per_chunk:
        raise ValueError(
            "blocks_per_window must be less than or equal to blocks_per_chunk"
        )
    if blocks_to_skip < 0:
        raise ValueError("blocks_to_skip must be non-negative")

    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    # For the partial tail chunk, drop the discarded prefix
    # (blocks_per_chunk - blocks_per_window) and keep only overlap in the
    # retained trailing window coordinate space.
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)
