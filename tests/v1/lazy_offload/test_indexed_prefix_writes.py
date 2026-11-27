# SPDX-License-Identifier: Apache-2.0
"""Indexed and full-scan drains must agree on complete ordered prefix writes."""

# Standard
from importlib.util import find_spec
from typing import Any
import random

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_policy import create_offload_policy
from lmcache.integration.vllm.lazy_offload_policy.prefix_write_index import (
    PrefixWriteIndex,
)
from tests.v1.lazy_offload.block_pool_fake import FakeBlockPool, make_real_pool
from tests.v1.lazy_offload.test_eviction_aware_policy import add_op, drain, make_queue

requires_vllm = pytest.mark.skipif(
    find_spec("vllm") is None, reason="requires a real vLLM BlockPool"
)


def test_indexed_prefix_decisions_match_full_scan() -> None:
    rng = random.Random(4499)
    pool = FakeBlockPool(64)
    pool.seed_held(list(range(1, 60)))
    full, indexed = [make_queue(pool, max_drain_per_step=3) for _ in range(2)]
    index = indexed.prefix_index = PrefixWriteIndex()
    ends = {}
    for step in range(1500):
        req = str(rng.randrange(20))
        bid = rng.randrange(1, 60)
        action = rng.randrange(12)
        if action < 5:
            for candidate in (bid, 60 - bid):
                if pool.blocks[candidate].block_hash is None:
                    pool.set_hash(candidate, f"{step}-{candidate}".encode())
            start = ends.get(req, 0)
            for queue in (full, indexed):
                add_op(queue, pool, req, [bid, 60 - bid], start + 256, start)
            ends[req] = start + 256
        elif action < 7:
            index.invalidate(bid)
            pool.set_hash(bid, f"reused-{step}".encode())
        elif action == 7:
            index.invalidate_all()
            for block in pool.blocks.values():
                block.block_hash = None
        elif action < 11:
            for queue in (full, indexed):
                if action == 8:
                    queue.mark_store_failed(req)
                elif action == 9:
                    queue.drop_request(req)
                else:
                    queue.discard_for_reuse(req)
            ends.pop(req, None)
        order = list(range(1, 60))
        rng.shuffle(order)
        pool.free_list[:] = [pool.blocks[b] for b in order[:10]]
        blocked = frozenset(str(rng.randrange(20)) for _ in range(3))
        pressure = rng.randrange(4)
        assert drain(full, est_next=pressure, blocked=blocked) == drain(
            indexed, est_next=pressure, blocked=blocked
        )
        assert full._ledger_line() == indexed._ledger_line()
    for req in list(indexed._pending):
        indexed.discard_for_reuse(req)
    assert not index._owners and not index._order and not index._dirty
    assert index.num_ops == 0


@requires_vllm
def test_indexed_prefix_writes_survive_real_pool_lifecycle() -> None:
    pool = make_real_pool(32)
    q = create_offload_policy({"lmcache.mp.indexed_prefix_writes": True}, pool)
    if q.prefix_index is None:
        pytest.skip("Indexed write integration requires the audited vLLM build")
    a, b, c, dst = pool.get_new_blocks(4)
    for i, block in enumerate([a, b, c], start=1):
        pool._insert_block_hash(bytes([i]) * 32, block, None)
    # Each op covers a different token range; two adjacent ops share a block.
    add_op(q, pool, "A", [a.block_id], 256)
    add_op(q, pool, "A", [a.block_id, b.block_id], 512)
    add_op(q, pool, "A", [c.block_id], 768)
    add_op(q, pool, "B", [b.block_id], 256)
    pool.move_block_hashes(b, dst)
    assert not drain(q, blocked=frozenset({"A"})).items
    assert not q.has_pending_request("B")
    assert len(q._pending["A"]) == 3
    assert not drain(q).items
    assert len(q._pending["A"]) == 1
    # Failed reset keeps the live prefix. A successful reset invalidates it.
    assert not pool.reset_prefix_cache()
    assert not drain(q).items and q.has_pending_request("A")
    pool.free_blocks([a, b, c, dst])
    assert pool.reset_prefix_cache()
    assert drain(q).emptied_request_ids == ["A"]
    assert not q.prefix_index.num_ops


def test_indexed_write_emits_prefix_before_victim_with_step_cap() -> None:
    pool = FakeBlockPool(8)
    pool.seed_held([1, 2, 3, 4])
    pool.make_free([3])
    q = make_queue(pool, max_drain_per_step=2)
    q.prefix_index = PrefixWriteIndex()
    for i in range(1, 5):
        add_op(q, pool, "A", [i], i * 256)
    first = drain(q, est_next=1)
    assert [m.op.end for m, _ in first.items[0].metadatas] == [256, 512]
    second = drain(q, est_next=1)
    assert [m.op.end for m, _ in second.items[0].metadatas] == [768]
    assert [op.store_metadata.op.end for op in q._pending["A"]] == [1024]


@requires_vllm
def test_real_pool_reassignment_invalidates_all_owners_even_away_from_front() -> None:
    pool = make_real_pool(32)
    queue = create_offload_policy({"lmcache.mp.indexed_prefix_writes": True}, pool)
    if queue.prefix_index is None:
        pytest.skip("Indexed write integration requires the audited vLLM build")
    victim, other = pool.get_new_blocks(2)
    for block in [victim, other]:
        pool._insert_block_hash(bytes([block.block_id]) * 32, block, None)
    add_op(queue, pool, "A", [victim.block_id, other.block_id], 256)
    add_op(queue, pool, "B", [victim.block_id], 256)
    pool.free_blocks([victim], evict_first=True)
    (reused,) = pool.get_new_blocks(1)
    assert reused is victim
    assert drain(queue).emptied_request_ids == ["A", "B"]
    add_op(queue, pool, "C", [other.block_id], 256)
    pool.evict_blocks({other.block_id})
    assert drain(queue).emptied_request_ids == ["C"]


@requires_vllm
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"lmcache.mp.indexed_prefix_writes": False},
        {
            "lmcache.mp.indexed_prefix_writes": True,
            "lmcache.mp.lazy_offload_max_deferral_seconds": 0.01,
        },
    ],
)
def test_index_is_opt_in_and_deadlines_keep_the_full_scan(
    extra: dict[str, Any],
) -> None:
    pool = make_real_pool(8)
    q = create_offload_policy(extra, pool)
    assert q.prefix_index is None
    assert "_remove_cached_block_hashes" not in vars(pool)
    assert "reset_prefix_cache" not in vars(pool)


def test_unrecognized_pool_keeps_the_full_scan() -> None:
    pool = FakeBlockPool(8)
    q = create_offload_policy({"lmcache.mp.indexed_prefix_writes": True}, pool)
    assert q.prefix_index is None


def test_index_config_requires_a_json_boolean() -> None:
    with pytest.raises(ValueError, match="JSON boolean"):
        create_offload_policy(
            {"lmcache.mp.indexed_prefix_writes": "false"}, FakeBlockPool(8)
        )
