# SPDX-License-Identifier: Apache-2.0
"""Blend per-leg read sets over a registration's object groups (pure)."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.distributed.api import (
    GroupKind,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


@dataclass(frozen=True)
class _BlendReadGroups:
    """Which of a registration's object groups each lookup leg reads.

    The prefix leg loads contiguous history, so it reads attention +
    recurrent-state groups. The blend (non-prefix) leg loads relocatable
    chunks, so it reads attention + the connector-private aux group and
    never recurrent state. Each leg keys, locks, and reads only its own
    set. All tuples hold ascending group ids.

    Attributes:
        blend_gids: Groups the blend leg reads (attention + aux).
        prefix_gids: Groups the prefix leg reads (attention + recurrent).
        recurrent_gids: The recurrent-state groups alone.
        attn_gid: The single full-attention group's id.
    """

    blend_gids: tuple[int, ...]
    prefix_gids: tuple[int, ...]
    recurrent_gids: tuple[int, ...]
    attn_gid: int


def _classify_cb_read_groups(
    num_object_groups: int, group_kinds: tuple[GroupKind, ...]
) -> _BlendReadGroups:
    """Classify a registration's object groups into the blend read set.

    A single-group (fused) layout maps to group 0.

    Raises:
        RuntimeError: If a multi-group layout has no resolvable read set.
    """
    if num_object_groups <= 1:
        return _BlendReadGroups(
            blend_gids=(0,),
            prefix_gids=(0,),
            recurrent_gids=(),
            attn_gid=0,
        )
    if len(group_kinds) != num_object_groups:
        raise RuntimeError(
            f"Blend: {num_object_groups} object groups but "
            f"{len(group_kinds)} kind label(s); cannot resolve the blend "
            "read set (registration predates group kinds?)."
        )
    attn = [i for i, k in enumerate(group_kinds) if k == "attention"]
    aux = [i for i, k in enumerate(group_kinds) if k == "aux"]
    recurrent = [i for i, k in enumerate(group_kinds) if k == "recurrent"]
    if len(attn) != 1 or len(aux) > 1:
        raise RuntimeError(
            f"Blend supports exactly one attention object group and at "
            f"most one aux (fused-aux) object group; got kinds "
            f"{group_kinds!r}."
        )
    gids = tuple(sorted(attn + aux))
    return _BlendReadGroups(
        blend_gids=gids,
        prefix_gids=tuple(sorted(attn + recurrent)),
        recurrent_gids=tuple(recurrent),
        attn_gid=attn[0],
    )


def _cb_chunks_available(
    committed: list[list[bool]], read_gids: tuple[int, ...]
) -> list[bool]:
    """Mark the chunks of one store that a reader of ``read_gids`` can use.

    The rule: a chunk is available to a reader iff every object group that
    reader reads committed a valid object for it. The transfer commits only
    valid objects (``all_null_chunk_masks`` skips a chunk whose pages are all
    null or whose recurrent-state page is null), so ``committed`` already
    encodes validity; this applies the reader's read set. The blend leg reads
    ``blend_gids`` at every chunk it matches. The prefix leg reads
    ``prefix_gids`` at the chunk its hit ends on (recurrent state has a
    one-chunk window), so over ``prefix_gids`` this marks where a prefix hit
    can end.

    Args:
        committed: ``committed[g][i]`` from the store, per object group.
        read_gids: The object groups the reader reads.

    Returns:
        Per chunk, whether every group in ``read_gids`` committed it; empty
        when ``committed`` is.
    """
    if not committed:
        return []
    return [
        all(committed[gid][i] for gid in read_gids) for i in range(len(committed[0]))
    ]


def _cb_chunk_major_object_keys(
    key: IPCCacheServerKey, chunk_hashes: list[bytes], gids: tuple[int, ...]
) -> list:
    """Expand chunk hashes to object keys, chunk-major.

    Per-chunk stride is uniform ``len(gids) * expansion`` so prefix bitmaps
    stay leading-ones-aligned. Returns the flattened key list.
    """
    per_group = ipc_key_to_object_keys(key, chunk_hashes, list(gids))
    n_hashes = len(chunk_hashes)
    expansion = len(per_group[0]) // n_hashes if n_hashes else 0
    out: list = []
    for i in range(n_hashes):
        for g in range(len(gids)):
            out.extend(per_group[g][i * expansion : (i + 1) * expansion])
    return out
