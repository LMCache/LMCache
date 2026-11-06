# SPDX-License-Identifier: Apache-2.0
"""Shared block-pool fake for the lazy-offload suites.

``FakeBlockPool`` models the two ``BlockPool`` properties the lazy-offload paths
depend on: the free queue is a linked list threaded through the blocks
themselves, reachable from ``free_block_queue.fake_free_list_head``, and queue
membership is exactly ``ref_cnt == 0``. Fidelity against a real pool is guarded
in ``test_fake_block_pool.py``.
"""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

#: Tokens per GPU block used throughout the lazy-offload suites.
TOKENS_PER_BLOCK = 16

#: Block id of the free list's head and tail sentinels. vLLM uses -1 for
#: both, and neither is ever reported as a free block.
_SENTINEL_ID = -1


@dataclass
class FakeBlock:
    """The ``KVCacheBlock`` fields the lazy-offload paths read."""

    block_id: int
    block_hash: bytes | None = None
    ref_cnt: int = 1
    pool: "FakeBlockPool | None" = None
    _next: "FakeBlock | None" = field(default=None, repr=False)

    @property
    def next_free_block(self) -> "FakeBlock | None":
        """The next block in the free queue, counting the link read."""
        if self.pool is not None:
            self.pool.link_reads += 1
        return self._next


class FakeFreeQueue:
    """The free queue in the shape the eviction-aware policy walks."""

    def __init__(self, owner: "FakeBlockPool") -> None:
        self._owner = owner

    @property
    def fake_free_list_head(self) -> FakeBlock:
        """Head sentinel whose successor is the next eviction victim."""
        head = FakeBlock(_SENTINEL_ID, pool=self._owner)
        previous = head
        for block in self._owner.free_list:
            previous._next = block
            previous = block
        previous._next = FakeBlock(_SENTINEL_ID, pool=self._owner)
        return head


class FakeBlockPool:
    """In-memory stand-in for vLLM's ``BlockPool``."""

    def __init__(self, num_blocks: int = 0) -> None:
        """Create ``num_blocks`` hashless blocks, all held and out of the queue."""
        self.blocks: dict[int, FakeBlock] = {}
        self.free_list: list[FakeBlock] = []
        self.free_block_queue = FakeFreeQueue(self)
        self.link_reads = 0
        self.touched: list[list[int]] = []
        self.freed: list[list[int]] = []
        for block_id in range(1, num_blocks + 1):
            self.blocks[block_id] = FakeBlock(block_id, pool=self)

    def seed_free(self, block_ids: list[int]) -> None:
        """Create hashed blocks at the tail of the free queue."""
        self._seed(block_ids, free=True)

    def seed_held(self, block_ids: list[int]) -> None:
        """Create hashed blocks that are in use and out of the queue."""
        self._seed(block_ids, free=False)

    def _seed(self, block_ids: list[int], free: bool) -> None:
        for block_id in block_ids:
            block = FakeBlock(block_id, pool=self)
            block.block_hash = f"hash-{block_id}".encode()
            block.ref_cnt = 0 if free else 1
            self.blocks[block_id] = block
            if free:
                self.free_list.append(block)

    def evict(self, block_id: int) -> None:
        """Take a block out of the queue and clear its hash."""
        block = self.blocks[block_id]
        self.free_list.remove(block)
        block.block_hash = None
        block.ref_cnt = 1

    def set_hash(self, block_id: int, block_hash: bytes | None) -> None:
        """Set one block's current prefix-cache hash."""
        self.blocks[block_id].block_hash = block_hash

    def make_free(self, block_ids: list[int]) -> None:
        """Release blocks to the tail of the free queue."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_cnt = 0
            self.free_list.append(block)

    def free_block_ids(self) -> list[int]:
        """The free queue's block ids, next victim first."""
        return [block.block_id for block in self.free_list]

    def get_num_free_blocks(self) -> int:
        """How many blocks the queue currently holds."""
        return len(self.free_list)

    def touch(self, blocks: list[FakeBlock]) -> None:
        """Pin blocks, dequeuing each only on its 0 -> 1 transition."""
        self.touched.append([block.block_id for block in blocks])
        for block in blocks:
            if block.ref_cnt == 0 and block in self.free_list:
                self.free_list.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks: list[FakeBlock]) -> None:
        """Unpin blocks, enqueuing only those that reach zero references.

        The real pool splits what it releases: a block carrying no hash can
        never match the prefix cache, so it goes to the head of the queue and
        is evicted first, while a hashed block goes to the tail. Lazy offload
        only ever releases through this path, so the split decides whether an
        unpinned block comes back at eviction rank 0 or last.
        """
        self.freed.append([block.block_id for block in blocks])
        without_hash: list[FakeBlock] = []
        with_hash: list[FakeBlock] = []
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                if block.block_hash is None:
                    without_hash.append(block)
                else:
                    with_hash.append(block)
        self.free_list = without_hash + self.free_list
        self.free_list.extend(with_hash)


def make_real_pool(num_gpu_blocks: int = 8) -> "BlockPool":
    """Build a real vLLM ``BlockPool`` with prefix caching on."""
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    return BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=TOKENS_PER_BLOCK,
    )


def real_free_block_ids(pool: "BlockPool") -> list[int]:
    """The free queue as vLLM's own materialiser reports it."""
    return [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]


def give_real_hash(block: "KVCacheBlock", block_hash: bytes) -> None:
    """Put a prefix-cache hash on a real block, whatever vLLM exposes.

    ``block_hash`` is a plain attribute up to vLLM 0.26 and a read-only
    property with a ``set_block_hash`` setter from 0.27, so assigning to it
    directly passes on an older pin and raises on a newer one.
    """
    setter = getattr(block, "set_block_hash", None)
    if setter is None:
        block.block_hash = block_hash  # type: ignore[assignment]
    else:
        setter(block_hash)


def clear_real_hash(block: "KVCacheBlock") -> None:
    """Take the prefix-cache hash off a real block, whatever vLLM exposes."""
    reset = getattr(block, "reset_hash", None)
    if reset is None:
        block.block_hash = None  # type: ignore[assignment]
    else:
        reset()
