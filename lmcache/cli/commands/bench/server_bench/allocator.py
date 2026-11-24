# SPDX-License-Identifier: Apache-2.0
"""Request-exclusive block allocation shared by all TP workers."""

# Standard
from dataclasses import dataclass
from typing import Literal
import random


@dataclass(frozen=True)
class GroupBlocks:
    """Physical block IDs in logical order for one engine group."""

    group_id: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class RequestBlocks:
    """Immutable group allocations shared by every TP worker."""

    groups: tuple[GroupBlocks, ...]

    def as_block_ids(self) -> list[list[int]]:
        """Return group-ordered IDs for buffer and transfer APIs."""
        return [list(group.block_ids) for group in self.groups]


class BlockAllocator:
    """Single-threaded, request-exclusive allocation without prefix sharing."""

    def __init__(
        self,
        groups: int,
        capacity: int,
        *,
        allocation_order: Literal["sequential", "random"] = "sequential",
        seed: int = 0,
    ) -> None:
        """Create one free pool per group with a reproducible allocation order."""
        if groups < 0 or capacity < 0:
            raise ValueError("groups and capacity must be non-negative")
        if allocation_order not in ("sequential", "random"):
            raise ValueError(f"unsupported allocation order: {allocation_order}")
        self._free = [set(range(capacity)) for _ in range(groups)]
        self._allocation_order = allocation_order
        self._random = random.Random(seed)

    def allocate(self, count: int) -> RequestBlocks:
        """Atomically reserve count > 0 exclusive blocks in every group."""
        if count <= 0:
            raise ValueError("block count must be positive")
        for group_id, free in enumerate(self._free):
            if len(free) < count:
                raise RuntimeError(
                    f"group {group_id} needs {count} blocks, only {len(free)} free"
                )
        allocated = RequestBlocks(
            tuple(
                GroupBlocks(
                    group_id,
                    tuple(
                        self._random.sample(sorted(free), count)
                        if self._allocation_order == "random"
                        else sorted(free)[:count]
                    ),
                )
                for group_id, free in enumerate(self._free)
            )
        )
        for group in allocated.groups:
            self._free[group.group_id].difference_update(group.block_ids)
        return allocated

    def release(self, allocated: RequestBlocks) -> None:
        """Return owned blocks after all transfers using them have finished."""
        for group in allocated.groups:
            self._free[group.group_id].update(group.block_ids)
