# SPDX-License-Identifier: Apache-2.0
"""Blend per-leg read sets over a registration's object groups (pure)."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
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


def _narrow_attn_desc(
    attn_desc: AttnWindowDesc, gids: tuple[int, ...]
) -> AttnWindowDesc:
    """Narrow a registration descriptor to one leg's object groups (ascending
    ``gids``).

    The fold stride is ``num_object_groups * world_size``; a leg keying over
    a subset must narrow the descriptor so the stride matches its keys.
    """
    return AttnWindowDesc(
        num_chunks_in_sw=[attn_desc.num_chunks_in_sw[g] for g in gids],
        world_size=attn_desc.world_size,
        group_kinds=(
            tuple(attn_desc.group_kinds[g] for g in gids)
            if attn_desc.group_kinds
            else ()
        ),
    )


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
