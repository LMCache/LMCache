# SPDX-License-Identifier: Apache-2.0
"""Server-side ownership of the source objects for one history restore."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Set
from dataclasses import dataclass
from typing import TYPE_CHECKING
import time

# First Party
from lmcache.v1.multiprocess.recurrent_plan import earlier_checkpoint_keys
from lmcache.v1.multiprocess.recurrent_progress import RecurrentCopyProgress
from lmcache.v1.multiprocess.recurrent_read import reserve_complete_recurrent_reads

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.internal_api import L1ManagerInterface
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext


@dataclass
class RecurrentReadLease:
    """Own L1 read locks until every started worker copy is fenced.

    The server serializes access and calls release_completed once after
    updating progress for a completion or cancellation.
    """

    groups: list[int]
    keys: dict[int, tuple[ObjectKey, ...]]
    progress: RecurrentCopyProgress
    _objects: dict[ObjectKey, MemoryObj]
    _manager: L1ManagerInterface
    valid_until: float = float("inf")

    def sources_for(self, rank: int, position: int) -> list[MemoryObj]:
        """Return one source per group for a worker's claimed checkpoint."""
        if time.monotonic() >= self.valid_until:
            self.progress.cancel()
            raise RuntimeError("Recurrent source read TTL expired")
        offset = rank * len(self.groups)
        return [
            self._objects[key]
            for key in self.keys[position][offset : offset + len(self.groups)]
        ]

    def release_completed(self) -> bool | None:
        """Release the reads once and return the publish decision if settled."""
        expired = time.monotonic() >= self.valid_until
        if expired:
            self.progress.cancel()
        result = self.progress.take_result()
        if result is not None:
            # An expired lock may have been replaced by another reader's lock.
            if not expired:
                self._manager.finish_read(list(self._objects))
            self._objects.clear()
        return result


def reserve_recurrent_history(
    context: MPCacheServerContext,
    key: IPCCacheServerKey,
    already_local: Set[int],
) -> RecurrentReadLease | None:
    """Reserve complete recurrent checkpoints from a single L1.

    Args:
        context: Server context used to resolve checkpoint keys and L1 ownership.
        key: Scheduler prefix key ending at the primary load boundary.
        already_local: Checkpoints already present in the device cache.

    Returns:
        A lease for all complete checkpoints, or None if there are no
        complete checkpoints or the server has multiple L1 managers.
    """
    managers = context.storage_manager.l1_managers()
    if len(managers) != 1:
        return None
    groups, candidates = earlier_checkpoint_keys(context, key, already_local)
    manager = managers[0]
    valid_until = time.monotonic() + manager.config.read_ttl_seconds
    selected, objects = reserve_complete_recurrent_reads(manager, candidates)
    if not selected:
        return None
    progress = RecurrentCopyProgress(set(range(key.world_size)))
    return RecurrentReadLease(groups, selected, progress, objects, manager, valid_until)
