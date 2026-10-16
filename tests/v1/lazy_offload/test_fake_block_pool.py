# SPDX-License-Identifier: Apache-2.0
"""Fidelity guards for ``FakeBlockPool`` against vLLM's real ``BlockPool``.

The fake encodes three assumptions the policy's rank walk relies on: the free
queue is a linked list between two sentinels reachable from
``free_block_queue.fake_free_list_head``; ``ref_cnt == 0`` is exactly queue
membership; and a released block rejoins at the head when it carries no hash
and at the tail when it does. A vLLM upgrade that changes any of them fails
here rather than silently desynchronising the policy from the pool.
"""

# Standard
import inspect

# Third Party
import pytest

pytest.importorskip("vllm", reason="the fidelity guards build a real BlockPool")

# Third Party
from vllm.v1.core.block_pool import BlockPool  # noqa: E402

# First Party
from tests.v1.lazy_offload.block_pool_fake import (  # noqa: E402
    FakeBlockPool,
    clear_real_hash,
    give_real_hash,
    make_real_pool,
    real_free_block_ids,
)

#: vLLM released every block to the tail, behind a ``prepend`` flag, up to
#: 0.23. The hashless-first split arrived in 0.24 and is what the fake models,
#: matching the unpinned vLLM the project builds against.
_PRE_SPLIT_VLLM = "prepend" in inspect.signature(BlockPool.free_blocks).parameters

pytestmark = pytest.mark.skipif(
    _PRE_SPLIT_VLLM,
    reason="vLLM <= 0.23 releases every block to the tail; the fake models 0.24+",
)


def walk_free_queue_ids(pool: "BlockPool | FakeBlockPool") -> list[int]:
    """Follow the free-queue links the way the policy's rank walk does."""
    ids: list[int] = []
    block = pool.free_block_queue.fake_free_list_head.next_free_block
    if block is None:
        raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
    while block.next_free_block is not None:
        ids.append(block.block_id)
        block = block.next_free_block
    return ids


def test_link_walk_matches_vllm_own_queue_listing() -> None:
    """The policy-style link walk yields what vLLM says the queue holds, in order, and
    walks an emptied queue as nothing.
    """
    pool = make_real_pool()
    (block,) = pool.get_new_blocks(1)
    give_real_hash(block, b"hash-1")
    pool.free_blocks([block])

    expected = real_free_block_ids(pool)
    assert walk_free_queue_ids(pool) == expected
    assert expected[-1] == block.block_id  # hashed: released at the tail

    pool.get_new_blocks(pool.get_num_free_blocks())
    assert walk_free_queue_ids(pool) == []


def test_fake_pool_free_queue_is_indistinguishable_from_the_real_pool() -> None:
    """One pin/unpin history, two pools, one reader."""
    real = make_real_pool()
    fake = FakeBlockPool(len(real.blocks) - 1)  # same ids 1..7, no null twin
    fake.seed_free(sorted(fake.blocks))
    for block_id in sorted(fake.blocks):
        give_real_hash(real.blocks[block_id], f"hash-{block_id}".encode())

    def assert_agreement() -> None:
        assert walk_free_queue_ids(real) == walk_free_queue_ids(fake)
        assert real.get_num_free_blocks() == fake.get_num_free_blocks()

    def touch(block_ids: list[int]) -> None:
        real.touch([real.blocks[bid] for bid in block_ids])
        fake.touch([fake.blocks[bid] for bid in block_ids])
        assert_agreement()

    def release(block_ids: list[int]) -> None:
        real.free_blocks([real.blocks[bid] for bid in block_ids])
        fake.free_blocks([fake.blocks[bid] for bid in block_ids])
        assert_agreement()

    def drop_hash(block_id: int) -> None:
        clear_real_hash(real.blocks[block_id])
        fake.set_hash(block_id, None)

    assert_agreement()  # construction order, 1..7
    touch([2, 3])  # 0 -> 1 dequeues both
    touch([3])  # shared pin: no queue change
    release([3])  # 2 -> 1: stays out of the queue
    release([3])  # 1 -> 0: rejoins at the tail, still hashed
    drop_hash(2)
    release([2])  # hashless: rejoins at the head, evicted next

    assert walk_free_queue_ids(fake)[0] == 2
