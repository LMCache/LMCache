# SPDX-License-Identifier: Apache-2.0
"""Read reservations for complete recurrent checkpoints in L1."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.distributed.error import L1Error

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.internal_api import L1ManagerInterface
    from lmcache.v1.memory_management import MemoryObj


def reserve_complete_recurrent_reads(
    manager: L1ManagerInterface,
    candidates: Mapping[int, tuple[ObjectKey, ...]],
) -> tuple[dict[int, tuple[ObjectKey, ...]], dict[ObjectKey, MemoryObj]]:
    """Reserve only L1-resident checkpoints with every required object.

    Args:
        manager: L1 manager that owns the candidate objects.
        candidates: Token positions mapped to the keys of all required groups
            and shards at that position.

    Returns:
        Complete candidates and their reserved objects. The caller owns one
        read lock per returned key until every consumer finishes using it.
    """
    keys = list(dict.fromkeys(key for group in candidates.values() for key in group))
    if not keys:
        return {}, {}
    acquired = {
        key: obj
        for key, (error, obj) in manager.reserve_read(keys).items()
        if error == L1Error.SUCCESS and obj is not None
    }
    complete = {
        position: group
        for position, group in candidates.items()
        if group and all(key in acquired for key in group)
    }
    retained = {key for group in complete.values() for key in group}
    partial = [key for key in acquired if key not in retained]
    if partial:
        manager.finish_read(partial)
    return complete, {key: acquired[key] for key in keys if key in retained}
