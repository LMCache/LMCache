# SPDX-License-Identifier: Apache-2.0
"""Per-worker/group block allocation."""

# Standard
from collections.abc import Sequence

# [worker][group][logical block]
BlockIds = tuple[tuple[tuple[int, ...], ...], ...]


class BlockAllocator:
    """Single-threaded allocator with one pool per worker/group."""

    def __init__(self, workers: int, groups: int, capacity: int) -> None:
        """Create workers × groups pools of capacity blocks; counts may be zero."""
        self._capacity = capacity
        self._free = [
            [set(range(capacity)) for _ in range(groups)] for _ in range(workers)
        ]

    def allocate(
        self,
        count: int,
        block_ids: Sequence[Sequence[Sequence[int]]] | None = None,
    ) -> BlockIds:
        """Atomically reserve count > 0 blocks per pool and return immutable IDs.

        Use explicit block_ids or each pool's lowest free IDs, in BlockIds order.
        Invalid or occupied IDs raise ValueError; exhaustion raises RuntimeError.
        Failure leaves all pools unchanged.
        """
        if count <= 0:
            raise ValueError("block count must be positive")
        if block_ids is None:
            allocated = tuple(
                tuple(tuple(sorted(free)[:count]) for free in groups)
                for groups in self._free
            )
        else:
            allocated = tuple(
                tuple(tuple(ids) for ids in groups) for groups in block_ids
            )
        if len(allocated) != len(self._free):
            raise ValueError("block_ids must contain one entry per worker")
        for rank, (groups, pools) in enumerate(zip(allocated, self._free, strict=True)):
            if len(groups) != len(pools):
                raise ValueError("block_ids must contain one entry per engine group")
            for group, (ids, free) in enumerate(zip(groups, pools, strict=True)):
                if block_ids is None and len(free) < count:
                    raise RuntimeError(
                        f"worker {rank}/group {group} needs {count} blocks, "
                        f"only {len(free)} free"
                    )
                if len(ids) != count:
                    raise ValueError(
                        "block_ids length must cover the full token chunks"
                    )
                if any(type(i) is not int or not 0 <= i < self._capacity for i in ids):
                    raise ValueError(
                        "block_ids must be integers within the KV allocation"
                    )
                if len(set(ids)) != len(ids):
                    raise ValueError(
                        "block_ids must not contain duplicates within a group"
                    )
                if not free.issuperset(ids):
                    raise ValueError(
                        f"worker {rank}/group {group}: block_ids already occupied"
                    )
        for groups, pools in zip(allocated, self._free, strict=True):
            for ids, free in zip(groups, pools, strict=True):
                free.difference_update(ids)
        return allocated

    def release(self, allocated: BlockIds) -> None:
        """Return allocated IDs; caller must ensure ownership."""
        for groups, pools in zip(allocated, self._free, strict=True):
            for ids, free in zip(groups, pools, strict=True):
                free.update(ids)
