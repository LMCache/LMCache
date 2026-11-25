# SPDX-License-Identifier: Apache-2.0
"""Blend per-leg read sets over a registration's object groups (pure)."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.common.attn_window import GroupKind
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.distributed.api import (
    ShardReplicationDesc,
    ipc_key_to_object_keys,
)


@dataclass(frozen=True)
class _BlendReadGroups:
    """Which of a registration's object groups each lookup leg reads.

    The prefix leg loads contiguous history, so it reads attention +
    recurrent-state groups. The blend (non-prefix) leg loads relocatable
    chunks, so it reads attention + the connector-private aux group and
    never recurrent state. Each leg keys, locks, and reads only its own
    set. All tuples hold ascending group ids.

    A registration may hold several attention groups: object-group
    separation buckets attention layers by cross-chunk window, so a model
    mixing full-attention and sliding-window layers registers one attention
    group per window. Both legs read every one of them.

    Attributes:
        blend_gids: Groups the blend leg reads (attention + aux).
        prefix_gids: Groups the prefix leg reads (attention + recurrent).
        recurrent_gids: The recurrent-state groups alone.
        attn_gids: The attention groups alone (at least one).
    """

    blend_gids: tuple[int, ...]
    prefix_gids: tuple[int, ...]
    recurrent_gids: tuple[int, ...]
    attn_gids: tuple[int, ...]


def _classify_cb_read_groups(
    num_object_groups: int, group_kinds: tuple[GroupKind, ...]
) -> _BlendReadGroups:
    """Classify a registration's object groups into the blend read set.

    A single-group (fused) layout maps to group 0.

    Args:
        num_object_groups: Number of object groups in the registration.
        group_kinds: Kind label of each object group, in object-group order.
            May be empty for a single-group layout.

    Returns:
        The per-leg read sets for the registration.

    Raises:
        RuntimeError: If a multi-group layout has no resolvable read set:
            kind labels are missing, there is no attention group, or there
            is more than one aux group.
    """
    if num_object_groups <= 1:
        return _BlendReadGroups(
            blend_gids=(0,),
            prefix_gids=(0,),
            recurrent_gids=(),
            attn_gids=(0,),
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
    if not attn or len(aux) > 1:
        raise RuntimeError(
            f"Blend needs at least one attention object group and at "
            f"most one aux (fused-aux) object group; got kinds "
            f"{group_kinds!r}."
        )
    return _BlendReadGroups(
        blend_gids=tuple(sorted(attn + aux)),
        prefix_gids=tuple(sorted(attn + recurrent)),
        recurrent_gids=tuple(recurrent),
        attn_gids=tuple(attn),
    )


def _cb_chunk_major_object_keys(
    key: IPCCacheServerKey,
    chunk_hashes: list[bytes],
    gids: tuple[int, ...],
    shard_replication_desc: ShardReplicationDesc,
) -> list:
    """Expand chunk hashes to object keys, chunk-major.

    Per-chunk stride is uniform ``len(gids) * expansion`` so prefix bitmaps
    stay leading-ones-aligned. Returns the flattened key list.
    """
    per_group = ipc_key_to_object_keys(
        key, chunk_hashes, list(gids), shard_replication_desc
    )
    n_hashes = len(chunk_hashes)
    expansion = len(per_group[0]) // n_hashes if n_hashes else 0
    out: list = []
    for i in range(n_hashes):
        for g in range(len(gids)):
            out.extend(per_group[g][i * expansion : (i + 1) * expansion])
    return out
