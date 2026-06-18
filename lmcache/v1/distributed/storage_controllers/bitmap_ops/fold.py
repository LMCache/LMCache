# SPDX-License-Identifier: Apache-2.0
"""Fold / unfold logic for multi-object-group prefix-cache hit computation.

With hybrid models, an LMCache request is split across several *object groups*
(full attention, sliding window, mamba, ...), each stored as its own
``MemoryObj`` and each with a different token-dependency rule:

- **full attention** needs every chunk of the prefix present to serve it;
- a **sliding window of ``w`` chunks** only needs the last ``w`` chunks of the
  prefix present (mamba is the ``w == 1`` case).

A model-wide prefix-cache hit of length ``L`` requires *every* object group to
be able to serve a prefix of length ``L`` under its own rule. This module turns
the per-group presence bitmaps into that single answer in three steps:

1. **fold** — per group, compute the set of prefix lengths it can serve
   (``_group_serves``);
2. **intersect + right-most-1** — a length is a model-wide hit only if all
   groups can serve it; the longest such length is the hit length ``i*``;
3. **unfold** — expand ``i*`` back into the concrete chunks each group needs
   (``unfold_range``), producing the retain mask used to load / lock / transfer.

When every group is full attention the servable set is a downward-closed prefix,
so ``i*`` equals the leading-ones count of the AND of the per-group presences --
i.e. the plain ``TrimPolicy.PREFIX`` / require-all intersection. Fold/unfold is a
strict generalization of that behavior.

Bitmaps here are laid out **group-major**: bit ``g * num_chunks + j`` is set iff
chunk ``j`` is available for object group ``g``.
"""

# Standard
from collections.abc import Iterable, Sequence

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import TrimPolicy

FULL_ATTENTION_WINDOW = -1
"""Sentinel ``group_windows`` value marking a full-attention object group
(needs the whole prefix). Any value ``<= 0`` is treated as full attention."""


def unfold_range(prefix_len: int, window: int) -> tuple[int, int]:
    """Chunk range ``[lo, hi)`` one object group needs to serve a prefix.

    Full attention needs ``[0, prefix_len)``; a window of ``w`` chunks needs
    ``[max(0, prefix_len - w), prefix_len)``.

    Args:
        prefix_len: model-wide hit length in chunks.
        window: cross-chunk sliding-window size in chunks; ``<= 0`` means full
            attention.

    Returns:
        The half-open chunk range this group must keep. ``(0, 0)`` when
        ``prefix_len <= 0``.
    """
    if prefix_len <= 0:
        return (0, 0)
    lo = 0 if window <= 0 else max(0, prefix_len - window)
    return (lo, prefix_len)


def fold_unfold(
    found: Bitmap,
    num_chunks: int,
    group_windows: Sequence[int],
) -> tuple[int, Bitmap]:
    """Compute the model-wide hit length and the per-group retain mask.

    See the module docstring for the fold -> right-most-1 -> unfold pipeline.

    Args:
        found: group-major presence bitmap of length
            ``len(group_windows) * num_chunks``; bit ``g * num_chunks + j`` set
            iff chunk ``j`` is available for object group ``g``.
        num_chunks: number of LMCache chunks in the request.
        group_windows: per-object-group cross-chunk sliding-window size in
            chunks, in object-group order; ``<= 0`` means full attention.

    Returns:
        ``(hit_length, retain_mask)``: ``hit_length`` is the longest prefix (in
        chunks) every group can serve, and ``retain_mask`` is a group-major
        bitmap (same length as ``found``) with the chunks each group must keep.

    Raises:
        ValueError: If ``group_windows`` is empty or ``num_chunks`` is negative.
    """
    if not group_windows:
        raise ValueError("group_windows must be non-empty")
    if num_chunks < 0:
        raise ValueError(f"num_chunks must be >= 0 (got {num_chunks})")

    num_groups = len(group_windows)
    num_keys = num_groups * num_chunks

    # Fold + intersect in O(num_groups * num_chunks): for each group, ``run`` is
    # the count of consecutive present chunks ending at the current chunk, so a
    # prefix of length L is servable iff the last ``min(window, L)`` chunks are
    # present, i.e. ``run >= min(window, L)``. ``servable[L]`` stays True only if
    # every group can serve a length-L prefix (length 0 is always servable).
    servable = [True] * (num_chunks + 1)
    for group_idx, window in enumerate(group_windows):
        base = group_idx * num_chunks
        effective_window = num_chunks if window <= 0 else window
        run = 0
        for prefix_len in range(1, num_chunks + 1):
            run = run + 1 if found.test(base + prefix_len - 1) else 0
            if servable[prefix_len] and run < min(effective_window, prefix_len):
                servable[prefix_len] = False

    # Right-most servable prefix length (length 0 is always servable).
    hit_length = 0
    for prefix_len in range(num_chunks, -1, -1):
        if servable[prefix_len]:
            hit_length = prefix_len
            break

    # Unfold: the concrete chunks each group needs to serve `hit_length`.
    retain_mask = Bitmap(num_keys)
    for group_idx, window in enumerate(group_windows):
        base = group_idx * num_chunks
        lo, hi = unfold_range(hit_length, window)
        for j in range(lo, hi):
            retain_mask.set(base + j)
    return hit_length, retain_mask


def fold_unfold_ranked(
    found: Bitmap,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> tuple[int, Bitmap]:
    """Fold/unfold over the ``group x chunk x kv_rank`` lookup key layout.

    The lookup key list (see
    :func:`lmcache.v1.distributed.api.ipc_key_to_object_keys`) carries, per
    object group, one key per ``(chunk, kv_rank)`` in chunk-major / rank-minor
    order. A chunk is treated as present for a group only when **every** kv_rank
    shard is present; the returned mask sets **all** ranks of each retained
    ``(group, chunk)``. This reduces the 3-D layout to the ``(group, chunk)``
    grid :func:`fold_unfold` expects, then re-expands the result.

    Args:
        found: presence bitmap of length
            ``len(group_windows) * num_chunks * num_ranks``; bit
            ``g * (num_chunks * num_ranks) + j * num_ranks + r`` set iff chunk
            ``j`` of object group ``g`` is present for kv_rank ``r``.
        num_chunks: number of LMCache chunks in the request.
        num_ranks: number of kv_rank shards per chunk (``world_size`` at lookup).
        group_windows: per-object-group cross-chunk sliding-window size in
            chunks, in object-group order; ``<= 0`` means full attention.

    Returns:
        ``(hit_length, retain_mask)``: ``hit_length`` in chunks and a retain
        mask over the same ranked layout as ``found``.

    Raises:
        ValueError: If ``group_windows`` is empty, ``num_chunks`` is negative,
            or ``num_ranks`` is not positive.
    """
    if num_ranks < 1:
        raise ValueError(f"num_ranks must be >= 1 (got {num_ranks})")
    if not group_windows:
        raise ValueError("group_windows must be non-empty")
    if num_chunks < 0:
        raise ValueError(f"num_chunks must be >= 0 (got {num_chunks})")

    num_groups = len(group_windows)
    group_stride = num_chunks * num_ranks

    # Reduce ranks: chunk present iff all its kv_rank shards are present.
    reduced = Bitmap(num_groups * num_chunks)
    for group_idx in range(num_groups):
        gbase = group_idx * group_stride
        for j in range(num_chunks):
            cbase = gbase + j * num_ranks
            if all(found.test(cbase + r) for r in range(num_ranks)):
                reduced.set(group_idx * num_chunks + j)

    hit_length, retain_gn = fold_unfold(reduced, num_chunks, group_windows)

    # Re-expand the retained (group, chunk) cells back over every kv_rank.
    retain_mask = Bitmap(num_groups * group_stride)
    for idx in retain_gn.get_indices_list():
        group_idx, j = divmod(idx, num_chunks)
        cbase = group_idx * group_stride + j * num_ranks
        for r in range(num_ranks):
            retain_mask.set(cbase + r)
    return hit_length, retain_mask


def merge_bitmaps(bitmaps: Iterable[Bitmap], num_keys: int) -> Bitmap:
    """Merge bitmaps with a bitwise OR into a ``num_keys``-sized bitmap.

    Always returns a ``num_keys``-sized bitmap (empty input -> all zeros), so
    downstream ``&`` operations never hit a size mismatch.

    Args:
        bitmaps: Per-source presence bitmaps to union.
        num_keys: Size of the merged bitmap.

    Returns:
        The bitwise-OR of all inputs as a ``num_keys``-sized bitmap.
    """
    merged = Bitmap(num_keys)
    for bm in bitmaps:
        merged = merged | bm
    return merged


def select_retained(
    found: Bitmap,
    num_keys: int,
    policy: TrimPolicy = TrimPolicy.PREFIX,
) -> Bitmap:
    """Select the retained subset of ``found`` for the non-windowed selections.

    ``PREFIX`` (LONGEST) keeps the leading contiguous run and drops everything
    from the first gap on; any other policy keeps every set bit, gaps included.
    The windowed hybrid fold is handled by :func:`fold_unfold_ranked`, not here.

    Args:
        found: Bitmap of found keys, over key indices ``0..num_keys-1``.
        num_keys: Total number of requested keys.
        policy: Selection to apply (see :class:`TrimPolicy`).

    Returns:
        Bitmap of the retained key indices.
    """
    if policy is TrimPolicy.PREFIX:
        return Bitmap(num_keys, found.count_leading_ones())
    return found
