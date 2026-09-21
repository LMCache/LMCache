# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for scheduler-side lazy-offload orchestration.

``LazyOffloadManager``: scheduler pressure translation, store actions, GPU
block pin ownership, completion and failure receipts, and request lifecycle
transitions, per ``docs/design/integration/vllm/lazy_offload.md``. The block
pool is the shared ``FakeBlockPool``.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# First Party
from lmcache.integration.vllm import (  # noqa: E402
    lazy_offload_manager as manager_module,
)
from lmcache.integration.vllm.kv_cache_groups import (  # noqa: E402
    KVGroupRetentionKind,
    KVGroupRetentionSpec,
)
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadActions,
    LazyOffloadManager,
)
from lmcache.integration.vllm.lazy_offload_policy import (  # noqa: E402
    POLICY_CONFIG_KEY,
    LazyOffloadMode,
)
from lmcache.integration.vllm.lazy_offload_policy.base import (  # noqa: E402
    BlockHashes,
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)
from tests.v1.lazy_offload.block_pool_fake import (  # noqa: E402
    TOKENS_PER_BLOCK,
    FakeBlockPool,
    give_real_hash,
    make_real_pool,
    real_free_block_ids,
)


class _FakeCompletionTracker:
    """Counts store-completion receipts until the expected worker count."""

    def __init__(self, expected_worker_count: int = 1) -> None:
        self._expected = expected_worker_count
        self._counts: dict[str | int, int] = {}

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        """Accumulate receipts and report whether the batch is complete."""
        total = self._counts.get(request_id, 0) + count
        if total >= self._expected:
            self._counts.pop(request_id, None)
            return True
        self._counts[request_id] = total
        return False

    def update_pending_store_operation_count(
        self, operation_id: int, count: int, /
    ) -> bool:
        """Accumulate receipts for one scheduler-assigned operation."""
        total = self._counts.get(operation_id, 0) + count
        if total >= self._expected:
            self._counts.pop(operation_id, None)
            return True
        self._counts[operation_id] = total
        return False


def _make_store_metadata(
    request_id: str,
    group_block_ids: list[list[int]],
    start: int,
    end: int,
    cache_salt: str = "",
    request_configs: dict[str, Any] | None = None,
    retention_group_id: int | None = None,
    selected_engine_group_ids: tuple[int, ...] | None = None,
) -> LMCacheMPRequestMetadata:
    """Build a STORE metadata with per-group block ids over ``[start, end)``."""
    return LMCacheMPRequestMetadata(
        request_id=request_id,
        direction="STORE",
        op=LoadStoreOp(
            token_ids=list(range(end)),
            block_ids=[list(group) for group in group_block_ids],
            start=start,
            end=end,
            selected_engine_group_ids=selected_engine_group_ids,
        ),
        cache_salt=cache_salt,
        request_configs=request_configs,
        retention_group_id=retention_group_id,
    )


def _make_scheduler_output(
    total_num_scheduled_tokens: int,
    new_request_block_ids: list[list[list[int]]] | None = None,
    cached_new_block_ids: list[list[list[int]] | None] | None = None,
) -> SimpleNamespace:
    """Duck-typed ``SchedulerOutput`` with the fields the drain path reads."""
    new_reqs = [
        SimpleNamespace(req_id=f"new-req-{index}", block_ids=block_ids)
        for index, block_ids in enumerate(new_request_block_ids or [])
    ]
    cached = SimpleNamespace(new_block_ids=cached_new_block_ids or [])
    return SimpleNamespace(
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
    )


@dataclass
class _Harness:
    """A manager wired to fakes, plus the fakes for assertions."""

    pool: FakeBlockPool
    manager: LazyOffloadManager
    ended_sessions: list[str]
    operation_ids: dict[str, list[int]]


def _make_manager_harness(
    num_blocks: int = 64,
    extra_config: dict[str, ConfigValue] | None = None,
    expected_worker_count: int = 1,
    group_tokens_per_block: list[int] | None = None,
    group_retention_specs: list[KVGroupRetentionSpec] | None = None,
    lmcache_tokens_per_chunk: int | None = None,
) -> _Harness:
    """Build a bound manager over a fake pool, EVICTION_AWARE by default."""
    pool = FakeBlockPool(num_blocks)
    manager = LazyOffloadManager(
        {
            POLICY_CONFIG_KEY: LazyOffloadMode.EVICTION_AWARE.value,
            **(extra_config or {}),
        },
        (
            group_tokens_per_block
            if group_tokens_per_block is not None
            else [TOKENS_PER_BLOCK]
        ),
        _FakeCompletionTracker(expected_worker_count),
        group_retention_specs=group_retention_specs,
        lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
    )
    manager.bind_block_pool(pool)  # type: ignore[arg-type]
    return _Harness(pool=pool, manager=manager, ended_sessions=[], operation_ids={})


def _admit_op(
    harness: _Harness,
    request_id: str,
    group_block_ids: list[list[int]],
    start: int,
    end: int,
    cache_salt: str = "",
    request_configs: dict[str, Any] | None = None,
    retention_group_id: int | None = None,
    selected_engine_group_ids: tuple[int, ...] | None = None,
) -> LMCacheMPRequestMetadata:
    """Give the blocks hashes and buffer one store op for them."""
    for group in group_block_ids:
        for bid in group:
            if harness.pool.blocks[bid].block_hash is None:
                harness.pool.set_hash(bid, f"hash-{bid}".encode())
    meta = _make_store_metadata(
        request_id,
        group_block_ids,
        start,
        end,
        cache_salt=cache_salt,
        request_configs=request_configs,
        retention_group_id=retention_group_id,
        selected_engine_group_ids=selected_engine_group_ids,
    )
    harness.manager.add_store_candidate(meta)
    return meta


def _apply_actions(harness: _Harness, actions: LazyOffloadActions) -> None:
    """Record the session-release actions the connector would apply."""
    harness.ended_sessions.extend(actions.sessions_to_end)


@dataclass
class _DrainResult:
    """Readable view of the public actions returned by one scheduler step."""

    actions: LazyOffloadActions

    def __len__(self) -> int:
        return len(self.actions.stores_to_submit)

    @property
    def stores(self) -> list[LMCacheMPRequestMetadata]:
        """The stores this step asked the connector to submit."""
        return self.actions.stores_to_submit


def _drain(
    harness: _Harness,
    total_num_scheduled_tokens: int = 2 * TOKENS_PER_BLOCK,
    new_request_block_ids: list[list[list[int]]] | None = None,
    cached_new_block_ids: list[list[list[int]] | None] | None = None,
) -> _DrainResult:
    """Run one public manager scheduler-step hook."""
    scheduler_output = _make_scheduler_output(
        total_num_scheduled_tokens, new_request_block_ids, cached_new_block_ids
    )
    actions = harness.manager.on_scheduler_step(scheduler_output)  # type: ignore[arg-type]
    for store in actions.stores_to_submit:
        assert store.store_operation_id is not None
        harness.operation_ids.setdefault(store.request_id, []).append(
            store.store_operation_id
        )
    _apply_actions(harness, actions)
    return _DrainResult(actions)


def _finish_request(harness: _Harness, request_id: str) -> LazyOffloadActions:
    actions = harness.manager.on_request_finished(request_id)
    _apply_actions(harness, actions)
    return actions


def _report_store_complete(harness: _Harness, request_id: str, count: int = 1) -> None:
    operation_ids = harness.operation_ids.get(request_id, [])
    operation_id = operation_ids[0] if operation_ids else -1
    actions = harness.manager.on_store_results(set(), {operation_id: count})
    if operation_ids and operation_id not in (
        harness.manager._requests.in_flight_operation_ids()
    ):
        operation_ids.remove(operation_id)
    _apply_actions(harness, actions)


def _report_store_failed(harness: _Harness, request_id: str, count: int = 1) -> None:
    operation_ids = harness.operation_ids.get(request_id, [])
    operation_id = operation_ids[0] if operation_ids else -1
    actions = harness.manager.on_store_results(
        {operation_id},
        {operation_id: count},
    )
    if operation_ids and operation_id not in (
        harness.manager._requests.in_flight_operation_ids()
    ):
        operation_ids.remove(operation_id)
    _apply_actions(harness, actions)


def _arrive_new_request(harness: _Harness, request_id: str) -> None:
    """Deliver the manager lifecycle event for a newly observed request id."""
    _apply_actions(harness, harness.manager.on_request_arrived(request_id))


def _make_fifo_harness(threshold: int = 1) -> _Harness:
    return _make_manager_harness(
        extra_config={
            POLICY_CONFIG_KEY: LazyOffloadMode.FIFO.value,
            "lmcache.mp.lazy_offload_threshold": threshold,
        }
    )


class TestRealBlockPool:
    """The manager and the policy against a real vLLM ``BlockPool``."""

    def test_manager_drains_and_completes_against_a_real_vllm_pool(self) -> None:
        """The full pin/emit/receipt path runs against a real ``BlockPool``: live hash
        reads, the rank walk, ``touch`` and ``free_blocks`` all hit the real structures
        the fakes stand in for elsewhere.
        """
        pool = make_real_pool()
        manager = LazyOffloadManager(
            {POLICY_CONFIG_KEY: LazyOffloadMode.EVICTION_AWARE.value},
            [TOKENS_PER_BLOCK],
            _FakeCompletionTracker(),
        )
        manager.bind_block_pool(pool)
        blocks = pool.get_new_blocks(pool.get_num_free_blocks())
        stored, spare = blocks[0], blocks[1:]
        give_real_hash(stored, b"hash-real")

        manager.add_store_candidate(
            _make_store_metadata("req", [[stored.block_id]], 0, TOKENS_PER_BLOCK)
        )
        pool.free_blocks([stored])  # rank 0: the next eviction victim
        assert real_free_block_ids(pool) == [stored.block_id]

        actions = manager.on_scheduler_step(
            _make_scheduler_output(2 * TOKENS_PER_BLOCK)  # type: ignore[arg-type]
        )
        assert len(actions.stores_to_submit) == 1
        assert stored.ref_cnt == 1  # pinned out of the queue
        assert pool.get_num_free_blocks() == 0

        manager.on_request_finished("req")
        operation_id = actions.stores_to_submit[0].store_operation_id
        assert operation_id is not None
        actions = manager.on_store_results(set(), {operation_id: 1})
        assert actions.sessions_to_end == ["req"]
        assert stored.ref_cnt == 0
        assert real_free_block_ids(pool) == [stored.block_id]
        pool.free_blocks(spare)  # leave the pool balanced

    def test_rank_walk_rejects_a_queue_missing_its_tail_sentinel(self) -> None:
        """A head with no successor is not a shape vLLM maintains: the drain must fail
        loudly instead of reporting an empty queue over a structure that changed.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.free_block_queue = SimpleNamespace(  # type: ignore[assignment]
            fake_free_list_head=SimpleNamespace(next_free_block=None)
        )

        with pytest.raises(RuntimeError, match="no successor"):
            _drain(harness)


class _RoguePolicy(OffloadPolicy):
    """A policy that ignores ``blocked_request_ids`` and re-emits one item."""

    def __init__(self, items: list[PendingStoreItem]) -> None:
        self._items = items

    def add(
        self,
        meta: LMCacheMPRequestMetadata,
        block_hashes: BlockHashes,
        *,
        requires_prefix: bool = True,
        retire_at_token: int | None = None,
    ) -> None:
        """Drop the operation: the items to emit are fixed at construction."""

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Emit the fixed items every step, ignoring every signal."""
        return LazyOffloadDrain(items=list(self._items))

    def has_pending_request(self, request_id: str) -> bool:
        """Report nothing buffered."""
        return False

    def drop_request(self, request_id: str) -> int:
        """Discard nothing."""
        return 0

    def discard_for_reuse(self, request_id: str) -> None:
        """Keep no per-request state, so there is nothing to discard."""

    def release_request(self, request_id: str) -> None:
        """Keep no per-request state, so there is nothing to release."""

    def mark_store_failed(
        self,
        request_id: str,
        retention_group_id: int = 0,
        *,
        requires_prefix: bool = True,
    ) -> int:
        """Drop nothing on failure."""
        return 0

    def log_final_stats(self) -> None:
        """Keep no counters."""


class TestDrainWiring:
    """One scheduler step: pressure translation, pinning, and emission."""

    def test_zero_token_step_does_not_emit_or_pin(self) -> None:
        """No-forward steps cannot carry metadata, so the manager must hold stores."""
        harness = _make_manager_harness()
        # Warm the pressure estimate before the idle step.
        _drain(
            harness,
            total_num_scheduled_tokens=2 * TOKENS_PER_BLOCK,
            new_request_block_ids=[[[30, 31]]],
        )
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])

        result = _drain(harness, total_num_scheduled_tokens=0)

        assert len(result) == 0
        assert harness.pool.touched == []

    def test_drain_without_pressure_emits_nothing(self) -> None:
        """Ops whose blocks sit deep in the free queue are not released."""
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[41, 42]], 0, 32)
        # 40 blocks ahead of the op's blocks; the step consumes ~2 blocks.
        harness.pool.make_free(list(range(1, 41)))
        harness.pool.make_free([41, 42])

        drained = _drain(harness)

        assert len(drained) == 0
        assert harness.pool.touched == []
        assert harness.ended_sessions == []

    def test_drain_under_pressure_pins_and_emits(self) -> None:
        """An op at the head of the free queue is pinned and submitted."""
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])

        drained = _drain(harness)

        assert len(drained) == 1
        assert harness.pool.touched == [[1, 2]]
        # Pinned blocks left the free queue.
        assert harness.pool.get_num_free_blocks() == 0

    def test_drain_pressure_from_gross_allocation(self) -> None:
        """This step's observed block allocation alone must be able to trigger a drain:
        the estimate from scheduled tokens is small, but the step allocated many blocks,
        so deeper free-queue ranks come into danger.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        # Three blocks ahead: ranks 3 and 4.
        harness.pool.make_free([11, 12, 13])
        harness.pool.make_free([1, 2])

        # Scheduled tokens alone give est 1 -> a shallow window. The gross
        # allocation of 4 blocks seeds the EMA -> the window reaches ranks 3-4.
        drained = _drain(
            harness,
            total_num_scheduled_tokens=8,
            new_request_block_ids=[[[20, 21, 22, 23]]],
        )

        assert len(drained) == 1
        assert harness.pool.touched == [[1, 2]]

    def test_drain_pressure_from_cached_request_allocation(self) -> None:
        """Decode-heavy steps allocate blocks exclusively through
        ``scheduled_cached_reqs``; that allocation alone must be able to trigger a
        drain, exactly as new-request allocation does.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        # Three blocks ahead: ranks 3 and 4.
        harness.pool.make_free([11, 12, 13])
        harness.pool.make_free([1, 2])

        drained = _drain(
            harness,
            total_num_scheduled_tokens=8,
            cached_new_block_ids=[[[20, 21, 22, 23]]],
        )

        assert len(drained) == 1
        assert harness.pool.touched == [[1, 2]]

    def test_cached_allocation_placeholders_do_not_hide_pressure(self) -> None:
        """vLLM reports ``None`` for cached requests that allocated nothing; the
        pressure counter must skip the placeholders and still see the step's real
        allocations.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([11, 12, 13])
        harness.pool.make_free([1, 2])

        drained = _drain(
            harness,
            total_num_scheduled_tokens=8,
            cached_new_block_ids=[None, [[20, 21, 22, 23]], None],
        )

        assert len(drained) == 1
        assert harness.pool.touched == [[1, 2]]

    def test_drain_estimate_rounds_partial_block_up(self) -> None:
        """A step scheduling less than one block of tokens still consumes a block; the
        next-step estimate must round up, not down.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])

        # 8 tokens, block size 16: ceil gives est 1 -> ranks 0 and 1 are due.
        # Floor division would give est 0 -> depth 0 -> nothing due.
        drained = _drain(harness, total_num_scheduled_tokens=8)

        assert len(drained) == 1

    def test_drain_coalesces_one_request_into_one_store_op(self) -> None:
        """One retention group's contiguous ranges become one store operation."""
        harness = _make_manager_harness()
        request_configs = {"lmcache.skip_save": False}
        _admit_op(
            harness,
            "req",
            [[1, 2]],
            0,
            32,
            cache_salt="salt-a",
            request_configs=request_configs,
        )
        _admit_op(
            harness,
            "req",
            [[3, 4]],
            32,
            64,
            cache_salt="salt-a",
            request_configs=request_configs,
        )
        harness.pool.make_free([1, 2, 3, 4])

        drained = _drain(harness)

        assert len(drained) == 1
        merged = drained.stores[0]
        assert merged.op.start == 0
        assert merged.op.end == 64
        assert merged.op.block_ids == [[1, 2, 3, 4]]
        assert merged.cache_salt == "salt-a"
        assert merged.request_configs == request_configs
        # The ledger carries the whole coalesced range, not just the first
        # op's prefix: the worker keys the store by the full range, so a
        # ledger that stopped growing would file truncated tokens under it.
        assert len(merged.op.token_ids) == 64
        # All four blocks are pinned for the single in-flight store.
        assert sorted(bid for pin in harness.pool.touched for bid in pin) == [
            1,
            2,
            3,
            4,
        ]

    def test_coalescing_a_gapped_batch_is_rejected(self) -> None:
        """A store's block list must cover its whole token range."""
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _admit_op(harness, "req", [[3, 4]], 48, 80)
        harness.pool.make_free([1, 2, 3, 4])

        with pytest.raises(ValueError, match="non-contiguous store ops"):
            _drain(harness)

        assert harness.pool.freed == [[1, 2, 3, 4]]
        assert all(
            harness.pool.blocks[block_id].ref_cnt == 0 for block_id in range(1, 5)
        )

    def test_drain_multi_group_op_pins_all_groups(self) -> None:
        """Hybrid-model batches merge each group independently and pin all blocks."""
        harness = _make_manager_harness(group_tokens_per_block=[16, 32])
        _admit_op(harness, "req", [[1, 2], [9]], 0, 32)
        _admit_op(harness, "req", [[3, 4], [10]], 32, 64)
        harness.pool.make_free([1, 2, 3, 4, 9, 10])

        drained = _drain(harness, total_num_scheduled_tokens=64)

        assert len(drained) == 1
        assert drained.stores[0].op.block_ids == [[1, 2, 3, 4], [9, 10]]
        assert sorted(bid for pin in harness.pool.touched for bid in pin) == [
            1,
            2,
            3,
            4,
            9,
            10,
        ]

    def test_null_blocks_are_not_hashed_pinned_or_owned_by_receipt(self) -> None:
        """Null placeholders stay in transfer geometry but own no scheduler pin."""
        harness = _make_manager_harness()
        null_block = harness.pool.blocks[1]
        null_block.is_null = True  # type: ignore[attr-defined]
        harness.pool.set_hash(1, b"null-placeholder")
        harness.pool.set_hash(2, b"hash-2")
        harness.manager.add_store_candidate(
            _make_store_metadata("req", [[1, 2]], 0, 32)
        )
        harness.pool.make_free([2])

        drained = _drain(harness)

        assert len(drained) == 1
        assert drained.stores[0].op.block_ids == [[1, 2]]
        assert harness.pool.touched == [[2]]
        operation_id = drained.stores[0].store_operation_id
        assert operation_id is not None
        harness.manager.on_store_results(set(), {operation_id: 1})
        assert harness.pool.freed == [[2]]


class TestHybridRetentionManager:
    """Manager behavior when one request owns independent retention groups."""

    @staticmethod
    def _specs() -> list[KVGroupRetentionSpec]:
        return [
            KVGroupRetentionSpec(
                engine_group_id=0,
                tokens_per_block=TOKENS_PER_BLOCK,
                kind=KVGroupRetentionKind.FULL_ATTENTION,
                window_size_tokens=None,
                retention_group_id=0,
            ),
            KVGroupRetentionSpec(
                engine_group_id=1,
                tokens_per_block=TOKENS_PER_BLOCK,
                kind=KVGroupRetentionKind.SLIDING_WINDOW,
                window_size_tokens=32,
                retention_group_id=1,
            ),
        ]

    def _harness(self) -> _Harness:
        return _make_manager_harness(
            group_tokens_per_block=[TOKENS_PER_BLOCK, TOKENS_PER_BLOCK],
            group_retention_specs=self._specs(),
            lmcache_tokens_per_chunk=32,
        )

    def test_sliding_retirement_token_tracks_first_retained_block(self) -> None:
        harness = _make_manager_harness(
            group_tokens_per_block=[TOKENS_PER_BLOCK, TOKENS_PER_BLOCK],
            group_retention_specs=self._specs(),
            lmcache_tokens_per_chunk=256,
        )
        metadata = _make_store_metadata(
            "req",
            [[], [1] * 16],
            256,
            512,
            retention_group_id=1,
            selected_engine_group_ids=(1,),
        )
        specs = [
            KVGroupRetentionSpec(
                engine_group_id=1,
                tokens_per_block=16,
                kind=KVGroupRetentionKind.SLIDING_WINDOW,
                window_size_tokens=128,
                retention_group_id=1,
            )
        ]

        # The 256-token object keeps only [384, 512). Its first 16-token block
        # can be retired before the step after progress reaches 527.
        assert harness.manager._retirement_token(metadata, specs) == 527

    def test_two_groups_of_one_request_complete_independently(self) -> None:
        harness = self._harness()
        _admit_op(
            harness,
            "req",
            [[1], []],
            0,
            16,
            retention_group_id=0,
            selected_engine_group_ids=(0,),
        )
        _admit_op(
            harness,
            "req",
            [[], [2]],
            0,
            16,
            retention_group_id=1,
            selected_engine_group_ids=(1,),
        )
        harness.pool.make_free([1, 2])

        drained = _drain(harness)
        assert {store.retention_group_id for store in drained.stores} == {0, 1}
        operation_ids = {
            store.retention_group_id: store.store_operation_id
            for store in drained.stores
        }
        first_operation = operation_ids[0]
        second_operation = operation_ids[1]
        assert first_operation is not None
        assert second_operation is not None
        _finish_request(harness, "req")

        first = harness.manager.on_store_results(set(), {first_operation: 1})
        assert first.sessions_to_end == []
        assert harness.pool.blocks[1].ref_cnt == 0
        assert harness.pool.blocks[2].ref_cnt == 1

        second = harness.manager.on_store_results(set(), {second_operation: 1})
        assert second.sessions_to_end == ["req"]
        assert harness.pool.blocks[2].ref_cnt == 0

    def test_full_group_failure_does_not_cancel_other_inflight_group(self) -> None:
        harness = self._harness()
        _admit_op(
            harness,
            "req",
            [[1], []],
            0,
            16,
            retention_group_id=0,
            selected_engine_group_ids=(0,),
        )
        # This full-attention suffix stays buffered while only block 1 is at
        # risk; a failed prefix must drop it.
        _admit_op(
            harness,
            "req",
            [[3], []],
            16,
            32,
            retention_group_id=0,
            selected_engine_group_ids=(0,),
        )
        _admit_op(
            harness,
            "req",
            [[], [2]],
            0,
            16,
            retention_group_id=1,
            selected_engine_group_ids=(1,),
        )
        harness.pool.make_free([1, 2])

        drained = _drain(harness)
        operation_ids = {
            store.retention_group_id: store.store_operation_id
            for store in drained.stores
        }
        assert set(operation_ids) == {0, 1}
        full_operation = operation_ids[0]
        window_operation = operation_ids[1]
        assert full_operation is not None
        assert window_operation is not None
        _finish_request(harness, "req")

        failed = harness.manager.on_store_results({full_operation}, {full_operation: 1})
        assert failed.sessions_to_end == []
        assert harness.pool.blocks[1].ref_cnt == 0
        assert harness.pool.blocks[2].ref_cnt == 1
        harness.pool.make_free([3])
        assert len(_drain(harness)) == 0

        completed = harness.manager.on_store_results(set(), {window_operation: 1})
        assert completed.sessions_to_end == ["req"]
        assert harness.pool.blocks[2].ref_cnt == 0

    def test_drain_drops_evicted_op_and_ends_finished_session(self) -> None:
        """An op whose block was reallocated is dropped, and the finished request's
        session ends at the drain that drops its last op.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _finish_request(harness, "req")
        assert harness.ended_sessions == []
        # Block 1 is evicted and reallocated to other content: new hash, not free.
        harness.pool.set_hash(1, b"other-content")
        harness.pool.make_free([2])

        drained = _drain(harness)

        assert len(drained) == 0
        assert harness.pool.touched == []
        assert harness.ended_sessions == ["req"]

    def test_drain_holds_back_while_store_in_flight(self) -> None:
        """A request with an in-flight store must not emit again until the completion
        receipt arrives.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1

        _admit_op(harness, "req", [[3, 4]], 32, 64)
        harness.pool.make_free([3, 4])
        assert len(_drain(harness)) == 0, "second batch emitted while first in flight"

        _report_store_complete(harness, "req")
        assert len(_drain(harness)) == 1

    def test_drain_shared_block_pins_are_reference_counted(self) -> None:
        """Two requests' ops can cover the same block (shared prefix)."""
        harness = _make_manager_harness()
        _admit_op(harness, "req-a", [[1, 2]], 0, 32)
        _admit_op(harness, "req-b", [[1, 3]], 0, 32)
        harness.pool.make_free([1, 2, 3])

        drained = _drain(harness)
        assert len(drained) == 2
        assert harness.pool.blocks[1].ref_cnt == 2

        _report_store_complete(harness, "req-a")
        # Block 1 is still pinned by req-b's in-flight store.
        assert 1 not in harness.pool.free_block_ids()

        _report_store_complete(harness, "req-b")
        assert sorted(harness.pool.free_block_ids()) == [1, 2, 3]
        assert harness.pool.free_block_ids().count(1) == 1

    def test_receipt_unpin_leaves_resurrected_block_pinned(self) -> None:
        """A pinned block can be resurrected by the engine (prefix-cache hit touches it)
        while the store is in flight; the receipt unpin must not push a block the engine
        still holds back into the free queue.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        _drain(harness)
        # The engine resurrects block 2 for a new request while in flight.
        harness.pool.touch([harness.pool.blocks[2]])

        _report_store_complete(harness, "req")

        assert harness.pool.free_block_ids() == [1]
        assert harness.pool.blocks[2].ref_cnt == 1

    def test_emitting_a_blocked_request_is_a_logic_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A policy that emits a request with an in-flight batch anyway must trigger the
        manager's RuntimeError instead of a second submission.
        """
        pool = FakeBlockPool(8)
        pool.set_hash(1, b"hash-1")
        item = PendingStoreItem(request_id="req")
        item.metadatas.append(
            (_make_store_metadata("req", [[1]], 0, 16), {(0, 1): b"hash-1"})
        )
        monkeypatch.setattr(
            manager_module,
            "create_offload_policy",
            lambda configs, gpu_block_pool: _RoguePolicy([item]),
        )
        manager = LazyOffloadManager({}, [TOKENS_PER_BLOCK], _FakeCompletionTracker())
        manager.bind_block_pool(pool)  # type: ignore[arg-type]

        actions = manager.on_scheduler_step(
            _make_scheduler_output(TOKENS_PER_BLOCK)  # type: ignore[arg-type]
        )
        assert len(actions.stores_to_submit) == 1  # first emission registers

        with pytest.raises(RuntimeError, match="in flight"):
            manager.on_scheduler_step(
                _make_scheduler_output(TOKENS_PER_BLOCK)  # type: ignore[arg-type]
            )


class TestStoreReceipts:
    """What a store receipt unpins, and when it ends the session."""

    def test_receipt_unpins_at_free_queue_tail(self) -> None:
        """A completed store is unpinned with no placement argument, so the blocks
        rejoin the free queue behind entries already there, and a duplicate receipt
        cannot unpin them a second time.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[5, 6]], 0, 32)
        harness.pool.make_free([5, 6])
        _drain(harness)
        # Other blocks joined the free queue while the store was in flight.
        harness.pool.make_free([10, 11])

        _report_store_complete(harness, "req")

        assert harness.pool.freed == [[5, 6]]
        assert harness.pool.free_block_ids() == [10, 11, 5, 6]
        _report_store_complete(harness, "req")
        assert harness.pool.freed == [[5, 6]]

    def test_receipt_for_running_request_keeps_session(self) -> None:
        """A store completing while the request is still running must not end the
        session.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        _drain(harness)

        _report_store_complete(harness, "req")

        assert harness.ended_sessions == []

    def test_receipt_after_finish_ends_session(self) -> None:
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        _drain(harness)
        _finish_request(harness, "req")
        assert harness.ended_sessions == []

        _report_store_complete(harness, "req")

        assert harness.ended_sessions == ["req"]

    def test_duplicate_receipt_is_ignored(self) -> None:
        """A resent receipt after the batch was fully processed must not unpin again or
        end the session twice.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        _drain(harness)
        _finish_request(harness, "req")
        _report_store_complete(harness, "req")
        assert harness.ended_sessions == ["req"]

        _report_store_complete(harness, "req")

        assert len(harness.pool.freed) == 1
        assert harness.ended_sessions == ["req"]

    def test_receipt_for_unknown_request_is_ignored(self) -> None:
        harness = _make_manager_harness()
        _report_store_complete(harness, "never-drained")
        assert harness.pool.freed == []
        assert harness.ended_sessions == []

    def test_partial_worker_receipts_do_not_unpin(self) -> None:
        """With multiple workers, the store completes only when every worker has
        reported; earlier receipts must not unpin or end the session.
        """
        harness = _make_manager_harness(expected_worker_count=2)
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        _drain(harness)
        _finish_request(harness, "req")

        _report_store_complete(harness, "req", count=1)
        assert harness.pool.freed == []
        assert harness.ended_sessions == []

        _report_store_complete(harness, "req", count=1)
        assert harness.pool.freed == [[1, 2]]
        assert harness.ended_sessions == ["req"]

    def test_manager_requires_bound_pool(self) -> None:
        manager = LazyOffloadManager(
            {POLICY_CONFIG_KEY: LazyOffloadMode.EVICTION_AWARE.value},
            [TOKENS_PER_BLOCK],
            _FakeCompletionTracker(),
        )

        with pytest.raises(ValueError, match="not bound"):
            manager.on_store_results(set(), {-1: 1})
        with pytest.raises(ValueError, match="not bound"):
            manager.add_store_candidate(_make_store_metadata("req", [[1]], 0, 16))
        # log_final_stats is one of the two documented exemptions: a
        # connector that aborts before the scheduler binds the pool still
        # calls it from shutdown().
        manager.log_final_stats()

    def test_rebinding_the_same_pool_is_idempotent(self) -> None:
        harness = _make_manager_harness()

        harness.manager.bind_block_pool(harness.pool)  # type: ignore[arg-type]

        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1

    def test_binding_a_different_pool_is_rejected(self) -> None:
        """Every buffered hash snapshot names blocks of the bound pool, so a rebind
        would silently compare them against another pool's blocks.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)

        with pytest.raises(ValueError, match="different GPU block pool"):
            harness.manager.bind_block_pool(FakeBlockPool(8))  # type: ignore[arg-type]


class TestTokenLedger:
    """One shared token list per request, instead of one copy per op."""

    def test_two_operations_of_one_request_carry_the_same_list_object(
        self,
    ) -> None:
        """The tracker hands the manager a fresh copy of the request's whole token
        sequence per operation, so a long request would hold one copy per buffered op.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        first = _drain(harness).stores[0]
        _report_store_complete(harness, "req")

        _admit_op(harness, "req", [[3, 4]], 32, 64)
        harness.pool.make_free([3, 4])
        second = _drain(harness).stores[0]

        assert first.op.token_ids is second.op.token_ids
        # The shared list absorbed the second op's longer tail.
        assert len(second.op.token_ids) == 64

    def test_the_callers_metadata_is_not_mutated(self) -> None:
        """``add_store_candidate`` buffers a rebound copy: the tracker keeps producing
        fresh lists and must not see the manager's ledger.
        """
        harness = _make_manager_harness()
        offered = _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])

        stored = _drain(harness).stores[0]

        assert len(offered.op.token_ids) == 32
        assert stored.op.token_ids is not offered.op.token_ids

    def test_a_reused_request_id_does_not_inherit_the_ledger(self) -> None:
        """The predecessor's ledger is longer than the successor's tokens, so a ledger
        kept across the teardown would never be overwritten and the successor would
        store the predecessor's tokens.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2, 3, 4]], 0, 64)
        harness.pool.make_free([1, 2, 3, 4])
        assert len(_drain(harness)) == 1
        _finish_request(harness, "req")
        _report_store_complete(harness, "req")
        assert harness.ended_sessions == ["req"]

        _admit_op(harness, "req", [[5, 6]], 0, 32)
        harness.pool.make_free([5, 6])

        resubmitted = _drain(harness).stores[0]

        assert len(resubmitted.op.token_ids) == 32


class TestKeepalive:
    """What ``has_inflight_store_work`` keeps the engine awake for."""

    def test_buffered_operations_alone_are_not_inflight_work(self) -> None:
        """Buffered ops are advanced only by a step that schedules model tokens, so
        reporting them here would spin the engine on connector-only steps that can never
        drain them.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)

        assert not harness.manager.has_inflight_store_work()

    def test_a_submitted_batch_is_inflight_until_its_receipt(self) -> None:
        """A submitted batch pins GPU blocks and its receipt arrives without further
        scheduling, so the engine must stay awake for it.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1

        assert harness.manager.has_inflight_store_work()

        _report_store_complete(harness, "req")

        assert not harness.manager.has_inflight_store_work()


class TestRequestFinished:
    """A finished request keeps its session until its buffer drains."""

    def test_request_finished_releases_idle_session(self) -> None:
        harness = _make_manager_harness()

        actions = _finish_request(harness, "req")

        assert actions.sessions_to_end == ["req"]
        assert harness.ended_sessions == ["req"]

    def test_request_finished_defers_session_while_ops_pending(self) -> None:
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)

        actions = _finish_request(harness, "req")

        assert actions.sessions_to_end == []
        assert harness.ended_sessions == []


class TestFullLifecycle:
    """Admission through teardown, with pins balanced end to end."""

    def test_lifecycle_store_completes_with_balanced_pins_and_one_teardown(
        self,
    ) -> None:
        """Admit -> finish -> pressure drain -> receipt: pins and unpins pair up exactly
        and the session ends exactly once.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _finish_request(harness, "req")
        harness.pool.make_free([1, 2])

        drained = _drain(harness)
        assert len(drained) == 1
        _report_store_complete(harness, "req")

        pinned = sorted(bid for pin in harness.pool.touched for bid in pin)
        unpinned = sorted(bid for freed in harness.pool.freed for bid in freed)
        assert pinned == unpinned == [1, 2]
        assert harness.ended_sessions == ["req"]
        # The blocks ended up back in the free queue with no reference left
        # behind.
        assert harness.pool.free_block_ids() == [1, 2]
        assert harness.pool.blocks[1].ref_cnt == 0


class TestFailedStoreReceipts:
    """A failed store unpins, breaks the prefix chain, and ends the session."""

    def test_failed_store_receipt_unpins_and_drops_held_back_ops(self) -> None:
        """A failure receipt still unpins the batch's blocks, but the request's held-
        back ops must be dropped: without the failed prefix they would be stored
        unreachable.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _admit_op(harness, "req", [[3, 4]], 32, 64)
        harness.pool.make_free([1, 2])
        drained = _drain(harness)
        assert len(drained) == 1  # first op in flight, second held back

        _report_store_failed(harness, "req")

        # Unpinned regardless of the failure.
        assert harness.pool.blocks[1].ref_cnt == 0
        # The held-back op is gone: pressure on its blocks emits nothing.
        harness.pool.make_free([3, 4])
        assert len(_drain(harness)) == 0
        # Nothing pending or in flight: the finished request tears down now.
        _finish_request(harness, "req")
        assert harness.ended_sessions == ["req"]

    def test_failed_store_receipt_for_finished_request_ends_session_once(self) -> None:
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _finish_request(harness, "req")
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1

        _report_store_failed(harness, "req")

        assert harness.pool.blocks[1].ref_cnt == 0
        assert harness.ended_sessions == ["req"]

    def test_failed_store_blacklists_later_ops_of_the_same_generation(self) -> None:
        """A non-orphaned failure breaks the request's prefix chain: operations admitted
        after the failure are rejected, since without the failed prefix they would be
        stored unreachable.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1

        _report_store_failed(harness, "req")

        _admit_op(harness, "req", [[3, 4]], 32, 64)
        harness.pool.make_free([3, 4])
        assert len(_drain(harness)) == 0, "op stored without its failed prefix"

    def test_failed_receipt_of_a_finished_request_ends_its_session(self) -> None:
        """One call carrying both the failure and its receipt must end the session:
        ``on_store_results`` applies failures first so that dropping the finished
        request's held-back suffix makes it releasable by the receipt that arrives with
        it.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _admit_op(harness, "req", [[3, 4]], 32, 64)
        harness.pool.make_free([1, 2])

        assert len(_drain(harness)) == 1  # first op in flight, second held back
        _finish_request(harness, "req")
        assert harness.ended_sessions == []

        _report_store_failed(harness, "req")

        assert harness.ended_sessions == ["req"]


class TestPreemptionReset:
    """A preemption frees the blocks, so what the request buffered is stale."""

    def test_preemption_reset_drops_buffered_ops_so_resume_cannot_overlap(self) -> None:
        """After preempt+resume the recreated tracker restarts at
        ``num_stored_tokens=0`` and re-produces store metadata from token zero.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _admit_op(harness, "req", [[3, 4]], 32, 64)

        harness.manager.on_request_reset("req")

        # Resume: APC resurrected the same blocks (hashes still match) and the
        # fresh tracker re-emits the first op.
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2, 3, 4])

        drained = _drain(harness, total_num_scheduled_tokens=4 * TOKENS_PER_BLOCK)

        assert len(drained) == 1
        op = drained.stores[0].op
        assert (op.start, op.end) == (0, 32)

    def test_preemption_reset_keeps_in_flight_batch_and_its_pins(self) -> None:
        """A batch already drained when the preemption hits stays in flight: its blocks
        remain pinned until the receipt, and no second batch may be submitted meanwhile
        (the worker keys store futures by request id).
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1  # in flight, blocks pinned

        harness.manager.on_request_reset("req")
        _admit_op(harness, "req", [[3, 4]], 0, 32)
        harness.pool.make_free([3, 4])

        assert len(_drain(harness)) == 0, "second batch while one is in flight"
        assert harness.pool.blocks[1].ref_cnt == 1

        _report_store_complete(harness, "req")
        assert harness.pool.blocks[1].ref_cnt == 0
        assert len(_drain(harness)) == 1

    def test_orphaned_batch_failure_receipt_spares_resumed_request(self) -> None:
        """The failure receipt of a batch emitted before the preemption must not drop
        the ops the resumed request re-buffered from token zero, nor blacklist its later
        ops -- they do not depend on the failed prefix.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1  # batch in flight

        harness.manager.on_request_reset("req")
        _admit_op(harness, "req", [[3, 4]], 0, 32)  # resumed, from token zero

        _report_store_failed(harness, "req")  # the orphaned batch's receipt
        assert harness.pool.blocks[1].ref_cnt == 0  # receipt still unpins

        harness.pool.make_free([3, 4])
        drained = _drain(harness)
        assert len(drained) == 1, "post-resume op dropped by the stale failure"
        # The current generation's prefix chain is intact: its next op is
        # accepted and drains once the first batch's receipt clears it.
        _admit_op(harness, "req", [[5, 6]], 32, 64)
        _report_store_complete(harness, "req")
        harness.pool.make_free([5, 6])
        drained = _drain(harness)
        assert len(drained) == 1
        op = drained.stores[0].op
        assert (op.start, op.end) == (32, 64)

    def test_orphaned_batch_failure_receipt_spares_id_reusing_successor(self) -> None:
        """The failure receipt of a finished predecessor's batch must not blacklist the
        successor now using the id.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "X", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1  # predecessor batch in flight
        _finish_request(harness, "X")

        _apply_actions(harness, harness.manager.on_request_arrived("X"))
        _admit_op(harness, "X", [[3, 4]], 0, 32)  # the successor's first op

        _report_store_failed(harness, "X")  # the predecessor batch's receipt
        assert harness.pool.blocks[1].ref_cnt == 0  # receipt still unpins
        # The successor is running: its merged session must not end here.
        assert harness.ended_sessions == []

        harness.pool.make_free([3, 4])
        drained = _drain(harness)
        assert len(drained) == 1, "successor blacklisted by predecessor failure"
        assert (drained.stores[0].op.start, drained.stores[0].op.end) == (0, 32)

    def test_orphaned_batch_completion_releases_deferred_session(self) -> None:
        """A completion receipt for an orphaned batch still frees its pins, and it can
        release the session when the request finished after the reset with nothing else
        pending.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1  # in flight

        harness.manager.on_request_reset("req")  # orphans the batch
        _finish_request(harness, "req")
        assert harness.ended_sessions == []  # receipt still outstanding

        _report_store_complete(harness, "req")

        assert harness.pool.blocks[1].ref_cnt == 0
        assert harness.pool.free_block_ids() == [1, 2]
        assert harness.ended_sessions == ["req"]


class TestRequestIdReuse:
    """A new request arriving under a finished id must not inherit its state."""

    def test_id_reuse_arrival_releases_predecessor_and_protects_successor(self) -> None:
        """A new request reusing a finished-deferred id must end the predecessor's
        session at arrival and discard its buffered ops.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "X", [[1, 2]], 0, 32)
        _finish_request(harness, "X")
        assert harness.ended_sessions == []  # teardown deferred

        _arrive_new_request(harness, "X")
        assert harness.ended_sessions == ["X"]  # released at arrival

        # The successor buffers its own first op; the predecessor's blocks
        # recycle; the drain must not touch the successor's state.
        _admit_op(harness, "X", [[3, 4]], 0, 32)
        harness.pool.set_hash(1, b"recycled")
        assert len(_drain(harness)) == 0
        assert harness.ended_sessions == ["X"]  # no second teardown

        # The successor's op is intact, chained, and storable under pressure.
        harness.pool.make_free([3, 4])
        drained = _drain(harness)
        assert len(drained) == 1
        assert (drained.stores[0].op.start, drained.stores[0].op.end) == (0, 32)

    def test_id_reuse_then_preemption_cannot_swallow_predecessor_release(self) -> None:
        """With the arrival reclaim in place, a preemption of the successor finds no
        finished-deferred state under the id, so the predecessor's release is never
        swallowed and the successor's own teardown stays independent.
        """
        harness = _make_manager_harness()
        _admit_op(harness, "X", [[1, 2]], 0, 32)
        _finish_request(harness, "X")
        _arrive_new_request(harness, "X")
        assert harness.ended_sessions == ["X"]

        _admit_op(harness, "X", [[3, 4]], 0, 32)
        harness.manager.on_request_reset("X")  # successor preempted: plain reset

        # Resumed successor re-buffers from token zero and finishes cleanly.
        _admit_op(harness, "X", [[3, 4]], 0, 32)
        harness.pool.make_free([3, 4])
        assert (
            len(_drain(harness, total_num_scheduled_tokens=4 * TOKENS_PER_BLOCK)) == 1
        )
        _report_store_complete(harness, "X")
        _finish_request(harness, "X")
        assert harness.ended_sessions == ["X", "X"]

    def test_id_reuse_with_in_flight_batch_defers_release_to_successor_finish(
        self,
    ) -> None:
        """The session must outlive an in-flight store AND the live successor: when the
        predecessor's batch is still awaiting its receipt at reuse time, the merged
        session ends through the successor's own finish, not at the receipt (the
        successor is running when it lands).
        """
        harness = _make_manager_harness()
        _admit_op(harness, "X", [[1, 2]], 0, 32)
        harness.pool.make_free([1, 2])
        assert len(_drain(harness)) == 1  # in flight
        _finish_request(harness, "X")

        _arrive_new_request(harness, "X")
        assert harness.ended_sessions == []  # receipt still outstanding

        # The receipt lands while the successor is running: no teardown yet.
        _report_store_complete(harness, "X")
        assert harness.ended_sessions == []

        # The successor's own finish ends the merged session, exactly once.
        _finish_request(harness, "X")
        assert harness.ended_sessions == ["X"]


class TestFIFODrain:
    """The manager drives the FIFO policy through the same hooks."""

    def test_fifo_drain_submits_intact_request_after_finish(self) -> None:
        harness = _make_fifo_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _finish_request(harness, "req")

        drained = _drain(harness)

        assert len(drained) == 1
        assert harness.pool.touched == [[1, 2]]

    def test_fifo_reused_id_waits_for_predecessor_receipt_before_draining(self) -> None:
        harness = _make_fifo_harness()
        _admit_op(harness, "X", [[1, 2]], 0, 32)
        _finish_request(harness, "X")
        assert len(_drain(harness)) == 1  # predecessor batch in flight

        _arrive_new_request(harness, "X")
        _admit_op(harness, "X", [[3, 4]], 0, 32)
        _finish_request(harness, "X")
        assert len(_drain(harness)) == 0  # successor must not overlap by id

        _report_store_complete(harness, "X")
        assert harness.ended_sessions == []
        assert len(_drain(harness)) == 1
        _report_store_complete(harness, "X")
        assert harness.ended_sessions == ["X"]

    def test_fifo_no_drain_below_finished_request_threshold(self) -> None:
        harness = _make_fifo_harness(threshold=2)
        _admit_op(harness, "req-a", [[1, 2]], 0, 32)
        _finish_request(harness, "req-a")
        assert len(_drain(harness)) == 0, "drained below the finished-count threshold"

        _admit_op(harness, "req-b", [[3, 4]], 0, 32)
        _finish_request(harness, "req-b")
        assert len(_drain(harness)) == 2

    def test_fifo_first_op_mismatch_unpins_and_ends_the_session(self) -> None:
        """On a hash mismatch the drain unpins what it pinned and skips the request
        rather than storing stale data.
        """
        harness = _make_fifo_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _finish_request(harness, "req")
        harness.pool.make_free([1, 2])
        harness.pool.set_hash(1, b"other-content")

        drained = _drain(harness)

        assert len(drained) == 0
        assert harness.pool.freed == [[1, 2]]
        assert harness.ended_sessions == ["req"]

    def test_fifo_drain_drops_op_with_unhashed_block(self) -> None:
        """A block with no hash at buffering time cannot prove at drain time that its
        content survived: None == None must fail validation, not pass it (an evicted-
        and-reallocated block also reads None).
        """
        harness = _make_fifo_harness()
        # Bypass the hash-seeding helper: blocks 1-2 keep block_hash=None.
        harness.manager.add_store_candidate(
            _make_store_metadata("req", [[1, 2]], 0, 32)
        )
        _finish_request(harness, "req")

        drained = _drain(harness)

        assert len(drained) == 0
        assert harness.ended_sessions == ["req"]

    def test_fifo_drain_coalesces_ops_into_one_store(self) -> None:
        """A request's buffered ops must go out as one store: the worker keys its in-
        flight store future by request id, so per-op submission overwrites futures and
        loses completion receipts.
        """
        harness = _make_fifo_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _admit_op(harness, "req", [[3, 4]], 32, 64)
        _finish_request(harness, "req")

        drained = _drain(harness)

        assert len(drained) == 1
        op = drained.stores[0].op
        assert (op.start, op.end) == (0, 64)
        assert op.block_ids[0] == [1, 2, 3, 4]

    def test_fifo_mid_request_mismatch_submits_valid_prefix(self) -> None:
        """A mismatch drops the remaining ops but the intact prefix is still stored, and
        the receipt path ends the session.
        """
        harness = _make_fifo_harness()
        _admit_op(harness, "req", [[1, 2]], 0, 32)
        _admit_op(harness, "req", [[3, 4]], 32, 64)
        _finish_request(harness, "req")
        harness.pool.set_hash(3, b"other-content")

        drained = _drain(harness)

        assert len(drained) == 1
        op = drained.stores[0].op
        assert (op.start, op.end) == (0, 32)
        # The mismatched op was unpinned right away; nothing else freed yet.
        assert harness.pool.freed == [[3, 4]]
        assert harness.ended_sessions == []

        _report_store_complete(harness, "req")
        assert harness.ended_sessions == ["req"]
