# SPDX-License-Identifier: Apache-2.0
"""Find pending prefix writes touched by block eviction or hash invalidation."""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from hashlib import sha256
from inspect import getfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock

    # Local
    from .base import BlockHashes
    from .eviction_aware import PendingStoreOp


# vLLM 0.34.0. Additional builds require an audit of all hash mutation paths.
_SUPPORTED_BLOCK_POOL_SOURCES = {
    "f4cf43b3d84a701659a90a1f256343109d84a2f97484b6dfc956ecfdbaf5b5b8",
}


class PrefixWriteIndex:
    """Map block IDs to the pending requests that still depend on them.

    A shared block may occur in several operations for one request. Its owner
    must remain indexed until the final reference leaves the pending queue.
    """

    def __init__(self) -> None:
        self._owners: dict[int, Counter[str]] = {}
        self._order: dict[str, int] = {}
        self._next_order = 0
        self._dirty: set[str] = set()
        self.num_ops = 0

    def add(self, request_id: str, block_hashes: BlockHashes) -> None:
        """Register one admitted operation and its original admission order."""
        if request_id not in self._order:
            self._order[request_id] = self._next_order
            self._next_order += 1
        for block_id in block_hashes:
            owners = self._owners.get(block_id)
            if owners is None:
                owners = self._owners[block_id] = Counter()
            owners[request_id] += 1
        self.num_ops += 1

    def remove(self, request_id: str, ops: list[PendingStoreOp], empty: bool) -> None:
        """Release references for emitted or discarded ops; empty ends this owner."""
        for op in ops:
            for block_id in op.block_hashes:
                owners = self._owners[block_id]
                owners[request_id] -= 1
                if not owners[request_id]:
                    del owners[request_id]
                if not owners:
                    del self._owners[block_id]
        self.num_ops -= len(ops)
        if empty:
            self._order.pop(request_id, None)
            self._dirty.discard(request_id)

    def invalidate(self, block_id: int) -> None:
        """Schedule each owner of a changed block for a snapshot check."""
        self._dirty.update(self._owners.get(block_id, ()))

    def invalidate_all(self) -> None:
        """A successful pool reset invalidates all pending hash snapshots."""
        self._dirty.update(self._order)

    def candidates(self, ranks: dict[int, int], blocked: set[str]) -> list[str]:
        """Return affected requests in admission order to preserve tie breaks."""
        # A blocked writer still needs its old snapshots checked after its receipt.
        due = self._dirty.difference(blocked)
        self._dirty.intersection_update(blocked)
        for block_id in ranks:
            due.update(self._owners.get(block_id, ()))
        return sorted(due, key=self._order.__getitem__)

    def observe_pool(self, pool: BlockPool) -> bool:
        """Instrument only this bound instance of the audited vLLM block pool.

        Every invalidation of a non-null primary hash in this build goes
        through the removal path (including private copies and explicit eviction)
        or a whole-pool reset. A scheduler allocation delta alone misses both.
        """
        cls = type(pool)
        if (cls.__module__, cls.__name__) != ("vllm.v1.core.block_pool", "BlockPool"):
            return False
        try:
            digest = sha256(Path(getfile(cls)).read_bytes()).hexdigest()
        except (TypeError, OSError):
            return False
        if digest not in _SUPPORTED_BLOCK_POOL_SOURCES:
            return False
        remove_hashes = pool._remove_cached_block_hashes
        reset_cache = pool.reset_prefix_cache

        def remove_and_notify(block: KVCacheBlock) -> list[BlockHashWithGroupId]:
            if block.block_hash is not None:
                self.invalidate(block.block_id)
            return remove_hashes(block)

        def reset_and_notify() -> bool:
            success = reset_cache()
            if success:
                self.invalidate_all()
            return success

        pool._remove_cached_block_hashes = remove_and_notify
        pool.reset_prefix_cache = reset_and_notify
        return True
