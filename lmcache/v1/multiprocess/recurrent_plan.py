# SPDX-License-Identifier: Apache-2.0
"""Build sparse recurrent checkpoint reads from the existing prefix keys."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Set
from dataclasses import replace
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.common.token_codec import num_packed_tokens

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext


def earlier_checkpoint_keys(
    context: MPCacheServerContext,
    key: IPCCacheServerKey,
    already_local: Set[int],
) -> tuple[list[int], dict[int, tuple[ObjectKey, ...]]]:
    """Resolve all earlier single-chunk recurrent states for a prefix.

    Args:
        context: Server context for registered layouts and prefix keys.
        key: Scheduler prefix key ending at the primary load boundary.
        already_local: Token positions already usable in the device cache.

    Returns:
        Recurrent group IDs and rank-major keys, empty for unsupported layouts.

    Raises:
        ValueError: If the prefix or resolved object geometry is invalid.
    """
    chunk_size = context.chunk_size
    if (
        key.worker_id is not None
        or key.world_size < 1
        or key.start != 0
        or not 0 <= key.end <= num_packed_tokens(key.token_bytes)
        or key.end % chunk_size
    ):
        raise ValueError("History requires an aligned scheduler prefix key")
    desc = context.layout_desc_registry.find_attn_desc(key.model_name, key.world_size)
    groups = [
        group for group, kind in enumerate(desc.group_kinds) if kind == "recurrent"
    ]
    positions = [
        pos
        for pos in range(chunk_size, key.end, chunk_size)
        if pos not in already_local
    ]
    if (
        not groups
        or not positions
        or any(desc.num_chunks_in_sw[g] != 1 for g in groups)
    ):
        return [], {}
    rows = [
        row
        for rank in range(key.world_size)
        for row in context.resolve_obj_keys(replace(key, worker_id=rank), groups)
    ]
    if len(rows) != key.world_size * len(groups) or any(
        len(row) != key.end // chunk_size for row in rows
    ):
        raise ValueError("History requires one object per group, shard, and chunk")
    return groups, {
        pos: tuple(row[pos // chunk_size - 1] for row in rows) for pos in positions
    }
