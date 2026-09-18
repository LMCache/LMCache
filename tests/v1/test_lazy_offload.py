# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral tests for scheduler-side lazy offload."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_manager import LazyOffloadManager
from lmcache.integration.vllm.lazy_offload_policy.base import BlockHashes, DrainSignals
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    EvictionAwarePolicyConfig,
    EvictionAwareStoreQueue,
)


@dataclass
class FakeBlock:
    """Minimal vLLM block used by the policy and manager tests."""

    block_id: int
    block_hash: bytes | None
    next_free_block: FakeBlock | None = None
    ref_count: int = 0


class FakeBlockPool:
    """Block-pool double with a linked free queue and pin accounting."""

    def __init__(self, free_order: list[int]) -> None:
        self.blocks = {
            block_id: FakeBlock(block_id, f"hash-{block_id}".encode())
            for block_id in free_order
        }
        tail = FakeBlock(-1, b"tail")
        next_block = tail
        for block_id in reversed(free_order):
            block = self.blocks[block_id]
            block.next_free_block = next_block
            next_block = block
        self.free_block_queue = SimpleNamespace(
            fake_free_list_head=SimpleNamespace(next_free_block=next_block)
        )
        self.touched: list[list[int]] = []
        self.freed: list[list[int]] = []

    def touch(self, blocks: list[FakeBlock]) -> None:
        """Record and pin blocks as vLLM's pool does before a store."""
        self.touched.append([block.block_id for block in blocks])
        for block in blocks:
            block.ref_count += 1

    def free_blocks(self, blocks: list[FakeBlock]) -> None:
        """Record and unpin blocks when all workers report completion."""
        self.freed.append([block.block_id for block in blocks])
        for block in blocks:
            block.ref_count -= 1


@dataclass
class FakeLoadStoreOp:
    """Dependency-free stand-in for the vLLM integration store operation."""

    token_ids: list[int]
    block_ids: list[list[int]]
    start: int
    end: int

    @property
    def flat_block_ids(self) -> list[int]:
        """Return block ids from every KV-cache group."""
        return [block_id for group in self.block_ids for block_id in group]


@dataclass
class FakeRequestMetadata:
    """Dependency-free stand-in for ``LMCacheMPRequestMetadata``."""

    request_id: str
    direction: str
    op: FakeLoadStoreOp
    cache_salt: str = ""
    request_configs: dict[str, object] | None = None


@dataclass
class FakeCompletionTracker:
    """Aggregate a configurable number of worker receipts per request."""

    expected_workers: int
    completed: dict[str, int] = field(default_factory=dict)

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        """Return true after all expected worker receipts arrive."""
        total = self.completed.get(request_id, 0) + count
        self.completed[request_id] = total
        return total >= self.expected_workers


def _metadata(
    request_id: str,
    block_ids: list[list[int]],
    *,
    start: int = 0,
    end: int = 16,
) -> Any:
    """Build one fake STORE operation."""
    return FakeRequestMetadata(
        request_id=request_id,
        direction="STORE",
        op=FakeLoadStoreOp(
            token_ids=list(range(end)),
            block_ids=block_ids,
            start=start,
            end=end,
        ),
        cache_salt="salt",
        request_configs={"tenant": "test"},
    )


def _hashes(pool: FakeBlockPool, metadata: FakeRequestMetadata) -> BlockHashes:
    """Snapshot all block hashes covered by an operation."""
    return {
        block_id: pool.blocks[block_id].block_hash
        for block_id in metadata.op.flat_block_ids
    }


def _signals(
    *,
    new_blocks: int = 0,
    next_blocks: int = 0,
    finished: set[str] | None = None,
    blocked: set[str] | None = None,
) -> DrainSignals:
    """Build one drain input with concise defaults."""
    return DrainSignals(
        new_blocks_allocated=new_blocks,
        est_next_step_blocks=next_blocks,
        finished_request_ids=finished or set(),
        blocked_request_ids=blocked or set(),
    )


def _scheduler_output(
    *, total_tokens: int = 16, new_block_groups: list[list[int]] | None = None
) -> SimpleNamespace:
    """Build the scheduler fields consumed by ``LazyOffloadManager``."""
    scheduled_new_reqs = []
    if new_block_groups is not None:
        scheduled_new_reqs.append(SimpleNamespace(block_ids=new_block_groups))
    return SimpleNamespace(
        total_num_scheduled_tokens=total_tokens,
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
    )


def test_eviction_aware_prefers_the_nearest_free_queue_block() -> None:
    """The most imminent request wins even when it was admitted later."""
    pool = FakeBlockPool([1, 2])
    policy = EvictionAwareStoreQueue(
        EvictionAwarePolicyConfig(horizon_steps=1, max_drain_per_step=1), pool
    )
    farther = _metadata("farther", [[2]])
    nearest = _metadata("nearest", [[1]])
    policy.add(farther, _hashes(pool, farther))
    policy.add(nearest, _hashes(pool, nearest))

    drain = policy.drain(_signals(next_blocks=2))

    assert [item.request_id for item in drain.items] == ["nearest"]
    assert policy.has_pending_request("farther")


def test_eviction_pressure_ema_decays_without_double_counting_history() -> None:
    """A zero-allocation step decays a 10-block EMA to seven blocks."""
    pool = FakeBlockPool(list(range(10)))
    policy = EvictionAwareStoreQueue(
        EvictionAwarePolicyConfig(horizon_steps=1, max_drain_per_step=1), pool
    )
    policy.drain(_signals(new_blocks=10))
    metadata = _metadata("outside-decayed-window", [[7]])
    policy.add(metadata, _hashes(pool, metadata))

    drain = policy.drain(_signals())

    assert drain.items == []
    assert policy.has_pending_request(metadata.request_id)


def test_block_hash_reuse_drops_the_lost_suffix_and_closes_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recycled middle block preserves only the still-valid prefix."""
    pool = FakeBlockPool([1, 2, 3, 4])
    policy = EvictionAwareStoreQueue(
        EvictionAwarePolicyConfig(horizon_steps=1, max_drain_per_step=4), pool
    )
    for index, block_id in enumerate((1, 2, 3)):
        metadata = _metadata(
            "request", [[block_id]], start=index * 16, end=(index + 1) * 16
        )
        policy.add(metadata, _hashes(pool, metadata))
    pool.blocks[2].block_hash = b"reused"

    drain = policy.drain(_signals(next_blocks=4))

    assert [meta.op.start for meta, _ in drain.items[0].metadatas] == [0]
    assert drain.emptied_request_ids == ["request"]
    rejected = _metadata("request", [[4]], start=48, end=64)
    policy.add(rejected, _hashes(pool, rejected))
    assert not policy.has_pending_request("request")

    fake_logger = MagicMock()
    monkeypatch.setattr(
        "lmcache.integration.vllm.lazy_offload_policy.eviction_aware.logger",
        fake_logger,
    )
    policy.log_final_stats()
    ledger = fake_logger.info.call_args.args[1]
    assert "admitted=3" in ledger
    assert "emitted=1" in ledger
    assert "dropped_evicted=2" in ledger
    assert "rejected_prefix_broken=1" in ledger
    assert ledger.endswith("pending=0")


def test_deadline_drains_a_whole_request_without_eviction_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline releases all operations of a request as one item."""
    now = 10.0
    monkeypatch.setattr(
        "lmcache.integration.vllm.lazy_offload_policy.eviction_aware.time.monotonic",
        lambda: now,
    )
    pool = FakeBlockPool([1, 2])
    policy = EvictionAwareStoreQueue(
        EvictionAwarePolicyConfig(
            horizon_steps=1,
            max_drain_per_step=4,
            max_deferral_seconds=5,
        ),
        pool,
    )
    for index, block_id in enumerate((1, 2)):
        metadata = _metadata(
            "request", [[block_id]], start=index * 16, end=(index + 1) * 16
        )
        policy.add(metadata, _hashes(pool, metadata))

    now = 15.1
    drain = policy.drain(_signals())

    assert len(drain.items) == 1
    assert len(drain.items[0].metadatas) == 2
    assert drain.emptied_request_ids == ["request"]


def test_manager_balances_hybrid_block_pins_after_multiworker_receipts() -> None:
    """A multi-group batch stays pinned until every worker completes."""
    pool = FakeBlockPool([1, 2, 3, 4, 5, 6])
    tracker = FakeCompletionTracker(expected_workers=2)
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": "EVICTION_AWARE",
            "lmcache.mp.lazy_offload_horizon_steps": 1,
            "lmcache.mp.lazy_offload_max_drain_per_step": 8,
        },
        group_tokens_per_block=[16, 8],
        completion_tracker=tracker,
    )
    manager.bind_block_pool(pool)
    manager.on_request_arrived("request")
    manager.add_store_candidate(_metadata("request", [[1], [2, 3]]))
    manager.add_store_candidate(_metadata("request", [[4], [5, 6]], start=16, end=32))

    actions = manager.on_scheduler_step(
        _scheduler_output(
            total_tokens=32,
            new_block_groups=[[90, 91], [92, 93, 94, 95]],
        )
    )

    assert len(actions.stores_to_submit) == 1
    submitted = actions.stores_to_submit[0]
    assert submitted.op.block_ids == [[1, 4], [2, 3, 5, 6]]
    assert submitted.op.start == 0
    assert submitted.op.end == 32
    assert submitted.request_configs == {"tenant": "test"}
    assert pool.touched == [[1, 2, 3], [4, 5, 6]]
    assert [pool.blocks[block_id].ref_count for block_id in range(1, 7)] == [1] * 6
    assert manager.has_inflight_store_work()
    assert manager.on_request_finished("request").sessions_to_end == []

    first_receipt = manager.on_store_results(set(), {"request": 1})
    assert first_receipt.sessions_to_end == []
    assert pool.freed == []
    assert manager.has_inflight_store_work()

    final_receipt = manager.on_store_results(set(), {"request": 1})
    assert final_receipt.sessions_to_end == ["request"]
    assert pool.freed == [[1, 2, 3, 4, 5, 6]]
    assert [pool.blocks[block_id].ref_count for block_id in range(1, 7)] == [0] * 6
    assert not manager.has_inflight_store_work()


def test_manager_drops_preempted_generation_before_resume() -> None:
    """A resumed request never emits the preemption-invalidated operation."""
    pool = FakeBlockPool([1, 2])
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": "EVICTION_AWARE",
            "lmcache.mp.lazy_offload_horizon_steps": 1,
        },
        group_tokens_per_block=[16],
        completion_tracker=FakeCompletionTracker(expected_workers=1),
    )
    manager.bind_block_pool(pool)
    manager.on_request_arrived("request")
    manager.add_store_candidate(_metadata("request", [[1]]))
    manager.on_request_reset("request")
    manager.on_request_arrived("request")
    manager.add_store_candidate(_metadata("request", [[2]]))

    actions = manager.on_scheduler_step(_scheduler_output(new_block_groups=[[90, 91]]))

    assert len(actions.stores_to_submit) == 1
    assert actions.stores_to_submit[0].op.block_ids == [[2]]


def test_manager_discards_finished_request_state_before_id_reuse() -> None:
    """A new request generation cannot inherit the predecessor's buffer."""
    pool = FakeBlockPool([1, 2])
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": "EVICTION_AWARE",
            "lmcache.mp.lazy_offload_horizon_steps": 1,
        },
        group_tokens_per_block=[16],
        completion_tracker=FakeCompletionTracker(expected_workers=1),
    )
    manager.bind_block_pool(pool)
    manager.on_request_arrived("request")
    manager.add_store_candidate(_metadata("request", [[1]]))
    assert manager.on_request_finished("request").sessions_to_end == []

    reuse_actions = manager.on_request_arrived("request")
    manager.add_store_candidate(_metadata("request", [[2]]))
    drain_actions = manager.on_scheduler_step(
        _scheduler_output(new_block_groups=[[90, 91]])
    )

    assert reuse_actions.sessions_to_end == ["request"]
    assert len(drain_actions.stores_to_submit) == 1
    assert drain_actions.stores_to_submit[0].op.block_ids == [[2]]


def test_manager_ignores_orphaned_failure_after_inflight_id_reuse() -> None:
    """A predecessor's failed receipt cannot break the successor's prefix."""
    pool = FakeBlockPool([1, 2])
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": "EVICTION_AWARE",
            "lmcache.mp.lazy_offload_horizon_steps": 1,
        },
        group_tokens_per_block=[16],
        completion_tracker=FakeCompletionTracker(expected_workers=1),
    )
    manager.bind_block_pool(pool)
    manager.on_request_arrived("request")
    manager.add_store_candidate(_metadata("request", [[1]]))
    first_drain = manager.on_scheduler_step(
        _scheduler_output(new_block_groups=[[90, 91]])
    )
    assert first_drain.stores_to_submit[0].op.block_ids == [[1]]

    manager.on_request_finished("request")
    assert manager.on_request_arrived("request").sessions_to_end == []
    manager.add_store_candidate(_metadata("request", [[2]]))
    blocked_drain = manager.on_scheduler_step(
        _scheduler_output(new_block_groups=[[90, 91]])
    )
    assert blocked_drain.stores_to_submit == []

    receipt = manager.on_store_results(
        failed_request_ids={"request"},
        completed_store_counts={"request": 1},
    )
    assert receipt.sessions_to_end == []
    assert pool.freed == [[1]]

    successor_drain = manager.on_scheduler_step(
        _scheduler_output(new_block_groups=[[90, 91]])
    )
    assert successor_drain.stores_to_submit[0].op.block_ids == [[2]]
