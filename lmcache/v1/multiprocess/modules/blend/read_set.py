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
class _CBReadGroups:
    """CacheBlend's per-leg read sets over a registration's object groups.

    Each leg keys, locks, and reads its own set; tuples are ascending.

    Attributes:
        gids: BLEND leg: attention + connector-private aux, never recurrent.
        prefix_gids: PREFIX leg: attention + recurrent, never aux.
        recurrent_gids: The recurrent-state groups alone.
        attn_gid: The full-attention object group's id.
    """

    gids: tuple[int, ...]
    prefix_gids: tuple[int, ...]
    recurrent_gids: tuple[int, ...]
    attn_gid: int


def _classify_cb_read_groups(
    num_object_groups: int, group_kinds: tuple[GroupKind, ...]
) -> _CBReadGroups:
    """Classify a registration's object groups into CacheBlend's read set.

    Args:
        num_object_groups: Total object groups in the registration.
        group_kinds: Per-group kind labels; empty only for single-group
            layouts.

    Returns:
        The read set; a single-group (fused) layout maps to group 0.

    Raises:
        RuntimeError: If a multi-group layout has no resolvable read set.
    """
    if num_object_groups <= 1:
        return _CBReadGroups(
            gids=(0,),
            prefix_gids=(0,),
            recurrent_gids=(),
            attn_gid=0,
        )
    if len(group_kinds) != num_object_groups:
        raise RuntimeError(
            f"CacheBlend: {num_object_groups} object groups but "
            f"{len(group_kinds)} kind label(s); cannot resolve the blend "
            "read set (registration predates group kinds?)."
        )
    attn = [i for i, k in enumerate(group_kinds) if k == "attention"]
    # "standalone" is the deprecated spelling of "aux" (see GroupKind);
    # accepted for one release.
    aux = [i for i, k in enumerate(group_kinds) if k in ("aux", "standalone")]
    recurrent = [i for i, k in enumerate(group_kinds) if k == "recurrent"]
    if len(attn) != 1 or len(aux) > 1:
        raise RuntimeError(
            f"CacheBlend supports exactly one attention object group and at "
            f"most one aux (fused-aux) object group; got kinds "
            f"{group_kinds!r}."
        )
    gids = tuple(sorted(attn + aux))
    return _CBReadGroups(
        gids=gids,
        prefix_gids=tuple(sorted(attn + recurrent)),
        recurrent_gids=tuple(recurrent),
        attn_gid=attn[0],
    )


def _narrow_attn_desc(
    attn_desc: AttnWindowDesc, gids: tuple[int, ...]
) -> AttnWindowDesc:
    """Narrow a registration descriptor to one leg's object groups.

    The fold stride is ``num_object_groups * world_size``; a leg keying over
    a subset must narrow the descriptor so the stride matches its keys.

    Args:
        attn_desc: The registration's full descriptor.
        gids: The leg's object group ids, ascending.

    Returns:
        The descriptor over exactly ``gids``.
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

    Per-chunk stride is uniform ``len(gids) * expansion`` so prefix
    bitmaps stay leading-ones-aligned.

    Args:
        key: The request key.
        chunk_hashes: One content hash per chunk.
        gids: Object group ids, ascending (see :class:`_CBReadGroups`).

    Returns:
        The flattened key list, ``len(chunk_hashes) * len(gids) * expansion``
        entries.
    """
    per_group = ipc_key_to_object_keys(key, chunk_hashes, list(gids))
    n_hashes = len(chunk_hashes)
    expansion = len(per_group[0]) // n_hashes if n_hashes else 0
    out: list = []
    for i in range(n_hashes):
        for g in range(len(gids)):
            out.extend(per_group[g][i * expansion : (i + 1) * expansion])
    return out
