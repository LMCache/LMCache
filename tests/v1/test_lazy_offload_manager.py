# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for scheduler-side lazy-offload orchestration.

The policy and pending-store facade have their own pure-logic test suites.
These tests exercise :class:`LazyOffloadManager` directly: scheduler pressure
translation, store actions, GPU-block pin ownership, completion receipts, and
request lifecycle transitions. Connector delegation is covered separately in
``test_mp_connector_lazy_offload.py``.

The fake pool mirrors the real pool's reference-counting semantics:
``touch`` increments ``ref_cnt`` and dequeues a block only on the 0 -> 1
transition; ``free_blocks`` decrements and enqueues only blocks that reach
0. This makes double-pin/double-unpin bugs observable (a block pinned by
two owners must not re-enter the free queue until both release it). Block
ids start at 1, matching vLLM's convention of reserving id 0 for the null
block, which never enters the free queue.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# First Party
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadActions,
    LazyOffloadManager,
)
from lmcache.integration.vllm.lazy_offload_pending_store import AddOutcome  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)

TOKENS_PER_BLOCK = 16


#: Block id of the free list's head and tail sentinels. vLLM uses -1 for
#: both, and neither is ever reported as a free block.
_SENTINEL_ID = -1


@dataclass
class _FakeBlock:
    """The ``KVCacheBlock`` fields the lazy-offload paths read."""

    block_id: int
    block_hash: bytes | None
    ref_cnt: int = 0
    next_free_block: "_FakeBlock | None" = None


class _FakeFreeQueue:
    """Free-queue facade in the shape the production pool view walks.

    vLLM holds the free queue as a linked list threaded through the blocks
    themselves, between a head and a tail sentinel, and the pool view walks
    it from the head and stops at the depth it was asked for -- it never
    materialises the queue. So the facade has to offer links, not a list.

    ``free_list`` stays the readable source of truth for the tests; the
    links are rebuilt from it on every access, so a test that reorders or
    edits the queue needs no extra bookkeeping.
    """

    def __init__(self, owner: "_FakeBlockPool") -> None:
        self._owner = owner

    @property
    def fake_free_list_head(self) -> _FakeBlock:
        """Head sentinel whose successor is the next eviction victim."""
        head = _FakeBlock(_SENTINEL_ID, None)
        previous = head
        for block in self._owner.free_list:
            previous.next_free_block = block
            previous = block
        previous.next_free_block = _FakeBlock(_SENTINEL_ID, None)
        return head


class _FakeBlockPool:
    """In-memory stand-in for vLLM's ``BlockPool``.

    ``free_list`` is the eviction queue, head first (index 0 is the next
    victim). Pinning follows the real pool's reference counting: ``touch``
    increments ``ref_cnt`` and removes a block from the queue only when it
    was 0; ``free_blocks`` decrements and enqueues (at the head when
    ``prepend`` is set) only the blocks that reach 0. Every call is
    recorded verbatim so tests can assert exact pin/unpin pairing.
    """

    def __init__(self, num_blocks: int) -> None:
        # Ids start at 1: vLLM reserves block id 0 for the null block,
        # which never appears in the free queue.
        self.blocks: dict[int, _FakeBlock] = {
            bid: _FakeBlock(bid, None) for bid in range(1, num_blocks + 1)
        }
        self.free_list: list[_FakeBlock] = []
        self.free_block_queue = _FakeFreeQueue(self)
        self.touched: list[list[int]] = []
        self.freed: list[tuple[list[int], bool]] = []

    def get_num_free_blocks(self) -> int:
        return len(self.free_list)

    def touch(self, blocks: list[_FakeBlock]) -> None:
        self.touched.append([b.block_id for b in blocks])
        for block in blocks:
            if block.ref_cnt == 0 and block in self.free_list:
                self.free_list.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks: list[_FakeBlock], prepend: bool = False) -> None:
        self.freed.append(([b.block_id for b in blocks], prepend))
        released = []
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                released.append(block)
        if prepend:
            self.free_list = released + self.free_list
        else:
            self.free_list.extend(released)

    def set_hash(self, block_id: int, block_hash: bytes | None) -> None:
        self.blocks[block_id].block_hash = block_hash

    def make_free(self, block_ids: list[int]) -> None:
        """Append unreferenced blocks to the tail of the free queue."""
        for bid in block_ids:
            self.free_list.append(self.blocks[bid])

    def free_block_ids(self) -> list[int]:
        return [b.block_id for b in self.free_list]


class _FakeSchedulerAdapter:
    """Records session teardowns and counts store-completion receipts.

    ``update_pending_store_count`` mirrors the production adapter's
    accumulate-until-world-size logic; the counting semantics themselves
    are the adapter's responsibility (and its test surface). What these
    tests verify is only the connector's reaction to the returned bool.
    """

    def __init__(self, expected_worker_count: int = 1) -> None:
        self.ended_sessions: list[str] = []
        self.shutdown_calls: int = 0
        self.lookup_result: int | None = 0
        self._expected = expected_worker_count
        self._counts: dict[str, int] = {}

    def end_session(self, request_id: str) -> None:
        self.ended_sessions.append(request_id)

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int], cache_salt: str
    ) -> None:
        pass

    def check_lookup_result(self, request_id: str) -> int | None:
        return self.lookup_result

    def update_pending_store_count(self, req_id: str, count: int) -> bool:
        total = self._counts.get(req_id, 0) + count
        if total >= self._expected:
            self._counts.pop(req_id, None)
            return True
        self._counts[req_id] = total
        return False


def _make_store_metadata(
    request_id: str,
    group_block_ids: list[list[int]],
    start: int,
    end: int,
    cache_salt: str = "",
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
        ),
        cache_salt=cache_salt,
    )


def _make_scheduler_output(
    total_num_scheduled_tokens: int,
    new_request_block_ids: list[list[list[int]]] | None = None,
    cached_new_block_ids: list[Any] | None = None,
) -> SimpleNamespace:
    """Duck-typed ``SchedulerOutput`` with the fields the drain path reads.

    Args:
        total_num_scheduled_tokens: Tokens scheduled this step.
        new_request_block_ids: Per new request, per group, the allocated
            block ids.
        cached_new_block_ids: Per cached request, either per-group block id
            lists or a falsy placeholder (vLLM uses ``None`` for requests
            without new allocations).
    """
    new_reqs = [
        SimpleNamespace(block_ids=block_ids)
        for block_ids in (new_request_block_ids or [])
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

    pool: _FakeBlockPool
    adapter: _FakeSchedulerAdapter
    manager: LazyOffloadManager


def _make_lazy_connector(
    num_blocks: int = 64,
    extra_config: dict[str, Any] | None = None,
    expected_worker_count: int = 1,
    group_tokens_per_block: list[int] | None = None,
) -> _Harness:
    pool = _FakeBlockPool(num_blocks)
    adapter = _FakeSchedulerAdapter(expected_worker_count)
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": "EVICTION_AWARE",
            **(extra_config or {}),
        },
        group_tokens_per_block or [TOKENS_PER_BLOCK],
        adapter,
    )
    manager.bind_block_pool(pool)  # type: ignore[arg-type]
    return _Harness(pool=pool, adapter=adapter, manager=manager)


def _admit_op(
    harness: _Harness,
    request_id: str,
    group_block_ids: list[list[int]],
    start: int,
    end: int,
    cache_salt: str = "",
) -> LMCacheMPRequestMetadata:
    """Give the blocks hashes and buffer one store op for them."""
    for group in group_block_ids:
        for bid in group:
            if harness.pool.blocks[bid].block_hash is None:
                harness.pool.set_hash(bid, f"hash-{bid}".encode())
    meta = _make_store_metadata(
        request_id, group_block_ids, start, end, cache_salt=cache_salt
    )
    harness.manager.add_store_candidate(meta)
    return meta


def _apply_actions(harness: _Harness, actions: LazyOffloadActions) -> None:
    for request_id in actions.sessions_to_end:
        harness.adapter.end_session(request_id)


@dataclass
class _DrainResult:
    """Readable view of the public actions returned by one scheduler step."""

    actions: LazyOffloadActions

    def __len__(self) -> int:
        return len(self.actions.stores_to_submit)

    @property
    def requests(self) -> list[LMCacheMPRequestMetadata]:
        return self.actions.stores_to_submit


def _drain(
    harness: _Harness,
    total_num_scheduled_tokens: int = 2 * TOKENS_PER_BLOCK,
    new_request_block_ids: list[list[list[int]]] | None = None,
) -> _DrainResult:
    """Run one public manager scheduler-step hook."""
    scheduler_output = _make_scheduler_output(
        total_num_scheduled_tokens, new_request_block_ids
    )
    actions = harness.manager.on_scheduler_step(scheduler_output)  # type: ignore[arg-type]
    _apply_actions(harness, actions)
    return _DrainResult(actions)


def _finish_request(harness: _Harness, request_id: str) -> LazyOffloadActions:
    actions = harness.manager.on_request_finished(request_id)
    _apply_actions(harness, actions)
    return actions


def _report_store_complete(harness: _Harness, request_id: str, count: int = 1) -> None:
    actions = harness.manager.on_store_results(set(), {request_id: count})
    _apply_actions(harness, actions)


def _report_store_failed(harness: _Harness, request_id: str, count: int = 1) -> None:
    actions = harness.manager.on_store_results({request_id}, {request_id: count})
    _apply_actions(harness, actions)


####
# Eviction-aware drain wiring
####


def test_zero_token_step_does_not_emit_or_pin() -> None:
    """No-forward steps cannot carry metadata, so the manager must hold stores."""
    harness = _make_lazy_connector()
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


def test_drain_without_pressure_emits_nothing() -> None:
    """Ops whose blocks sit deep in the free queue are not released."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[41, 42]], 0, 32)
    # 40 blocks ahead of the op's blocks; the step consumes ~2 blocks.
    harness.pool.make_free(list(range(1, 41)))
    harness.pool.make_free([41, 42])

    metadata = _drain(harness)

    assert len(metadata) == 0
    assert harness.pool.touched == []
    assert harness.adapter.ended_sessions == []


def test_drain_under_pressure_pins_and_emits() -> None:
    """An op at the head of the free queue is pinned and submitted."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])

    metadata = _drain(harness)

    assert len(metadata) == 1
    assert harness.pool.touched == [[1, 2]]
    # Pinned blocks left the free queue.
    assert harness.pool.get_num_free_blocks() == 0


def test_drain_pressure_from_gross_allocation() -> None:
    """This step's observed block allocation alone must be able to trigger
    a drain: the estimate from scheduled tokens is small, but the step
    allocated many blocks, so deeper free-queue ranks come into danger."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    # Three blocks ahead: ranks 3 and 4.
    harness.pool.make_free([11, 12, 13])
    harness.pool.make_free([1, 2])

    # Scheduled tokens alone give est 1 -> depth 2 (< rank 3). The gross
    # allocation of 4 blocks seeds the EMA -> depth 8 -> due.
    metadata = _drain(
        harness,
        total_num_scheduled_tokens=8,
        new_request_block_ids=[[[20, 21, 22, 23]]],
    )

    assert len(metadata) == 1
    assert harness.pool.touched == [[1, 2]]


def test_drain_estimate_rounds_partial_block_up() -> None:
    """A step scheduling less than one block of tokens still consumes a
    block; the next-step estimate must round up, not down."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])

    # 8 tokens, block size 16: ceil gives est 1 -> depth 2 -> ranks 0 and 1
    # are due. Floor division would give est 0 -> depth 0 -> nothing due.
    metadata = _drain(harness, total_num_scheduled_tokens=8)

    assert len(metadata) == 1


def test_drain_coalesces_one_request_into_one_store_op() -> None:
    """The worker tracks one in-flight store per request, so a drained
    batch must arrive as a single coalesced operation."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32, cache_salt="salt-a")
    _admit_op(harness, "req", [[3, 4]], 32, 64, cache_salt="salt-a")
    harness.pool.make_free([1, 2, 3, 4])

    metadata = _drain(harness)

    assert len(metadata) == 1
    merged = metadata.requests[0]
    assert merged.op.start == 0
    assert merged.op.end == 64
    assert merged.op.block_ids == [[1, 2, 3, 4]]
    assert merged.cache_salt == "salt-a"
    # All four blocks are pinned for the single in-flight store.
    assert sorted(bid for pin in harness.pool.touched for bid in pin) == [1, 2, 3, 4]


def test_drain_multi_group_op_pins_all_groups() -> None:
    """Hybrid-model batches merge each group independently and pin all blocks."""
    harness = _make_lazy_connector(group_tokens_per_block=[16, 32])
    _admit_op(harness, "req", [[1, 2], [9]], 0, 32)
    _admit_op(harness, "req", [[3, 4], [10]], 32, 64)
    harness.pool.make_free([1, 2, 3, 4, 9, 10])

    metadata = _drain(harness, total_num_scheduled_tokens=64)

    assert len(metadata) == 1
    assert metadata.requests[0].op.block_ids == [[1, 2, 3, 4], [9, 10]]
    assert sorted(bid for pin in harness.pool.touched for bid in pin) == [
        1,
        2,
        3,
        4,
        9,
        10,
    ]


def test_drain_drops_evicted_op_and_ends_finished_session() -> None:
    """An op whose block was reallocated is dropped, and the finished
    request's session ends at the drain that drops its last op."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _finish_request(harness, "req")
    assert harness.adapter.ended_sessions == []
    # Block 1 is evicted and reallocated to other content: new hash, not free.
    harness.pool.set_hash(1, b"other-content")
    harness.pool.make_free([2])

    metadata = _drain(harness)

    assert len(metadata) == 0
    assert harness.pool.touched == []
    assert harness.adapter.ended_sessions == ["req"]


def test_drain_holds_back_while_store_in_flight() -> None:
    """A request with an in-flight store must not emit again until the
    completion receipt arrives."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    assert len(_drain(harness)) == 1

    _admit_op(harness, "req", [[3, 4]], 32, 64)
    harness.pool.make_free([3, 4])
    assert len(_drain(harness)) == 0, "second batch emitted while first in flight"

    _report_store_complete(harness, "req")
    assert len(_drain(harness)) == 1


def test_drain_shared_block_pins_are_reference_counted() -> None:
    """Two requests' ops can cover the same block (shared prefix). The
    block is pinned once per op and must re-enter the free queue only
    after the last receipt releases it."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req-a", [[1, 2]], 0, 32)
    _admit_op(harness, "req-b", [[1, 3]], 0, 32)
    harness.pool.make_free([1, 2, 3])

    metadata = _drain(harness)
    assert len(metadata) == 2
    assert harness.pool.blocks[1].ref_cnt == 2

    _report_store_complete(harness, "req-a")
    # Block 1 is still pinned by req-b's in-flight store.
    assert 1 not in harness.pool.free_block_ids()

    _report_store_complete(harness, "req-b")
    assert sorted(harness.pool.free_block_ids()) == [1, 2, 3]
    assert harness.pool.free_block_ids().count(1) == 1


def test_receipt_unpin_leaves_resurrected_block_pinned() -> None:
    """A pinned block can be resurrected by the engine (prefix-cache hit
    touches it) while the store is in flight; the receipt unpin must not
    push a block the engine still holds back into the free queue."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    _drain(harness)
    # The engine resurrects block 2 for a new request while in flight.
    harness.pool.touch([harness.pool.blocks[2]])

    _report_store_complete(harness, "req")

    assert harness.pool.free_block_ids() == [1]
    assert harness.pool.blocks[2].ref_cnt == 1


####
# Store-completion receipts (update_connector_output)
####


def test_receipt_unpins_to_free_queue_head() -> None:
    """Completed stores are unpinned with ``prepend=True``: the block has a
    copy below the GPU, so it should be the next eviction victim."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[5, 6]], 0, 32)
    harness.pool.make_free([5, 6])
    _drain(harness)
    # Other blocks joined the free queue while the store was in flight.
    harness.pool.make_free([10, 11])

    _report_store_complete(harness, "req")

    assert harness.pool.freed == [([5, 6], True)]
    assert harness.pool.free_block_ids() == [5, 6, 10, 11]
    # A duplicate receipt cannot unpin the blocks a second time.
    _report_store_complete(harness, "req")
    assert harness.pool.freed == [([5, 6], True)]


def test_receipt_for_running_request_keeps_session() -> None:
    """A store completing while the request is still running must not end
    the session."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    _drain(harness)

    _report_store_complete(harness, "req")

    assert harness.adapter.ended_sessions == []


def test_receipt_after_finish_ends_session() -> None:
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    _drain(harness)
    _finish_request(harness, "req")
    assert harness.adapter.ended_sessions == []

    _report_store_complete(harness, "req")

    assert harness.adapter.ended_sessions == ["req"]


def test_duplicate_receipt_is_ignored() -> None:
    """A resent receipt after the batch was fully processed must not unpin
    again or end the session twice."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    _drain(harness)
    _finish_request(harness, "req")
    _report_store_complete(harness, "req")
    assert harness.adapter.ended_sessions == ["req"]

    _report_store_complete(harness, "req")

    assert len(harness.pool.freed) == 1
    assert harness.adapter.ended_sessions == ["req"]


def test_receipt_for_unknown_request_is_ignored() -> None:
    harness = _make_lazy_connector()
    _report_store_complete(harness, "never-drained")
    assert harness.pool.freed == []
    assert harness.adapter.ended_sessions == []


def test_partial_worker_receipts_do_not_unpin() -> None:
    """With multiple workers, the store completes only when every worker
    has reported; earlier receipts must not unpin or end the session."""
    harness = _make_lazy_connector(expected_worker_count=2)
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    _drain(harness)
    _finish_request(harness, "req")

    _report_store_complete(harness, "req", count=1)
    assert harness.pool.freed == []
    assert harness.adapter.ended_sessions == []

    _report_store_complete(harness, "req", count=1)
    assert harness.pool.freed == [([1, 2], True)]
    assert harness.adapter.ended_sessions == ["req"]


def test_store_results_require_bound_pool() -> None:
    adapter = _FakeSchedulerAdapter()
    manager = LazyOffloadManager(
        {"lmcache.mp.lazy_offload_policy": "EVICTION_AWARE"},
        [TOKENS_PER_BLOCK],
        adapter,
    )

    with pytest.raises(ValueError, match="block pool"):
        manager.on_store_results(set(), {"req": 1})


####
# request_finished in lazy mode
####


def test_request_finished_releases_idle_session() -> None:
    harness = _make_lazy_connector()

    actions = _finish_request(harness, "req")

    assert actions.sessions_to_end == ["req"]
    assert harness.adapter.ended_sessions == ["req"]


def test_request_finished_defers_session_while_ops_pending() -> None:
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)

    actions = _finish_request(harness, "req")

    assert actions.sessions_to_end == []
    assert harness.adapter.ended_sessions == []


####
# Full lifecycle
####


def test_lifecycle_store_completes_with_balanced_pins_and_one_teardown() -> None:
    """Admit -> finish -> pressure drain -> receipt: pins and unpins pair
    up exactly and the session ends exactly once."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _finish_request(harness, "req")
    harness.pool.make_free([1, 2])

    metadata = _drain(harness)
    assert len(metadata) == 1
    _report_store_complete(harness, "req")

    pinned = sorted(bid for pin in harness.pool.touched for bid in pin)
    unpinned = sorted(bid for freed, _ in harness.pool.freed for bid in freed)
    assert pinned == unpinned == [1, 2]
    assert all(prepend for _, prepend in harness.pool.freed)
    assert harness.adapter.ended_sessions == ["req"]
    # The blocks ended up back in the free queue, ready for eviction,
    # with no reference left behind.
    assert harness.pool.free_block_ids() == [1, 2]
    assert harness.pool.blocks[1].ref_cnt == 0


####
# Failed-store receipts
####


def test_failed_store_receipt_unpins_and_drops_held_back_ops() -> None:
    """A failure receipt still unpins the batch's blocks, but the
    request's held-back ops must be dropped: without the failed prefix
    they would be stored unreachable."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _admit_op(harness, "req", [[3, 4]], 32, 64)
    harness.pool.make_free([1, 2])
    metadata = _drain(harness)
    assert len(metadata) == 1  # first chunk in flight, second held back

    _report_store_failed(harness, "req")

    # Unpinned regardless of the failure.
    assert harness.pool.blocks[1].ref_cnt == 0
    # The held-back chunk is gone: pressure on its blocks emits nothing.
    harness.pool.make_free([3, 4])
    assert len(_drain(harness)) == 0
    # Nothing pending or in flight: the finished request tears down now.
    _finish_request(harness, "req")
    assert harness.adapter.ended_sessions == ["req"]


def test_failed_store_receipt_for_finished_request_ends_session_once() -> None:
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _finish_request(harness, "req")
    harness.pool.make_free([1, 2])
    assert len(_drain(harness)) == 1

    _report_store_failed(harness, "req")

    assert harness.pool.blocks[1].ref_cnt == 0
    assert harness.adapter.ended_sessions == ["req"]


####
# Content deduplication across requests
####


def test_shared_prefix_content_buffered_once_across_requests() -> None:
    """Two requests covering the same blocks buffer one op: the duplicate
    does not defer its request's session teardown, and the pressure drain
    emits a single store."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req-a", [[1, 2]], 0, 32)
    _admit_op(harness, "req-b", [[1, 2]], 0, 32)

    _finish_request(harness, "req-b")
    assert harness.adapter.ended_sessions == ["req-b"]

    harness.pool.make_free([1, 2])
    metadata = _drain(harness)
    assert len(metadata) == 1
    assert metadata.requests[0].request_id == "req-a"


def test_drain_never_coalesces_across_dedup_hole() -> None:
    """Deduplication can leave a hole in a request's pending list; the
    drain must emit only the contiguous front run (the batch is coalesced
    into one store op), leaving the post-hole ops for a later batch.

    Request C buffers chunks 1+2 and drains chunk 1 alone (emission
    releases its content key while chunk 2's stays held). Request B with
    identical content then admits chunk 1, is DEDUPLICATED on chunk 2, and
    admits chunk 3 -- a pending list with a hole at [32, 64)."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req-c", [[1, 2]], 0, 32)
    _admit_op(harness, "req-c", [[3, 4]], 32, 64)
    harness.pool.make_free([1, 2])
    metadata = _drain(harness)
    assert len(metadata) == 1
    assert (metadata.requests[0].op.start, metadata.requests[0].op.end) == (0, 32)
    _report_store_complete(harness, "req-c")  # chunk 2 still pending under C

    _admit_op(harness, "req-b", [[1, 2]], 0, 32)
    dedup = harness.manager.add_store_candidate(
        _make_store_metadata("req-b", [[3, 4]], 32, 64)
    )
    assert dedup is AddOutcome.DEDUPLICATED
    _admit_op(harness, "req-b", [[5, 6]], 64, 96)

    # Pressure on B's chunk-3 blocks: prefix closure pulls chunk 1 along,
    # but the batch must stop at the hole instead of coalescing across it.
    harness.pool.make_free([5, 6])
    metadata = _drain(harness)
    assert len(metadata) == 1
    op = metadata.requests[0].op
    assert (metadata.requests[0].request_id, op.start, op.end) == ("req-b", 0, 32)

    # After the receipt the post-hole op is emitted as its own batch.
    _report_store_complete(harness, "req-b")
    metadata = _drain(harness)
    assert len(metadata) == 1
    op = metadata.requests[0].op
    assert (metadata.requests[0].request_id, op.start, op.end) == ("req-b", 64, 96)


####
# Preemption: tracker reset must drop stale buffered ops
####


def _preempt_reset(harness: _Harness, request_id: str, num_tokens: int = 64) -> None:
    """Deliver the manager lifecycle event emitted for a tracker reset."""
    del num_tokens
    harness.manager.on_request_reset(request_id)


def test_preemption_reset_drops_buffered_ops_so_resume_cannot_overlap() -> None:
    """After preempt+resume the recreated tracker restarts at
    ``num_stored_tokens=0`` and re-produces store metadata from token zero.
    The pre-preemption ops must be dropped at tracker reset; otherwise the
    next pressure drain coalesces overlapping ranges and raises, killing
    the scheduler step."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _admit_op(harness, "req", [[3, 4]], 32, 64)

    _preempt_reset(harness, "req")

    # Resume: APC resurrected the same blocks (hashes still match) and the
    # fresh tracker re-emits the first chunk.
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2, 3, 4])

    metadata = _drain(harness, total_num_scheduled_tokens=4 * TOKENS_PER_BLOCK)

    assert len(metadata) == 1
    op = metadata.requests[0].op
    assert (op.start, op.end) == (0, 32)


def test_preemption_reset_keeps_in_flight_batch_and_its_pins() -> None:
    """A batch already drained when the preemption hits stays in flight:
    its blocks remain pinned until the receipt, and no second batch may be
    submitted meanwhile (the worker keys store futures by request id)."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    assert len(_drain(harness)) == 1  # in flight, blocks pinned

    _preempt_reset(harness, "req")
    _admit_op(harness, "req", [[3, 4]], 0, 32)
    harness.pool.make_free([3, 4])

    assert len(_drain(harness)) == 0, "second batch while one is in flight"
    assert harness.pool.blocks[1].ref_cnt == 1

    _report_store_complete(harness, "req")
    assert harness.pool.blocks[1].ref_cnt == 0
    assert len(_drain(harness)) == 1


def test_stale_batch_failure_receipt_spares_resumed_request() -> None:
    """The failure receipt of a batch emitted before the preemption must
    not drop the ops the resumed request re-buffered from token zero, nor
    reject its later chunks -- they do not depend on the failed prefix."""
    harness = _make_lazy_connector()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    assert len(_drain(harness)) == 1  # batch in flight

    _preempt_reset(harness, "req")
    _admit_op(harness, "req", [[3, 4]], 0, 32)  # resumed, from token zero

    _report_store_failed(harness, "req")  # the stale batch's receipt
    assert harness.pool.blocks[1].ref_cnt == 0  # receipt still unpins

    harness.pool.make_free([3, 4])
    metadata = _drain(harness)
    assert len(metadata) == 1, "post-resume op dropped by the stale failure"
    for bid in (5, 6):
        harness.pool.set_hash(bid, f"hash-{bid}".encode())
    outcome = harness.manager.add_store_candidate(
        _make_store_metadata("req", [[5, 6]], 32, 64)
    )
    assert outcome is AddOutcome.BUFFERED


####
# Request-id reuse after a finished-deferred teardown
####


def _arrive_new_request(
    harness: _Harness, request_id: str, num_tokens: int = 64
) -> None:
    """Deliver the manager lifecycle event for a newly observed request id."""
    del num_tokens
    _apply_actions(harness, harness.manager.on_request_arrived(request_id))


def test_id_reuse_arrival_releases_predecessor_and_protects_successor() -> None:
    """A new request reusing a finished-deferred id must end the
    predecessor's session at arrival and discard its buffered ops.
    Otherwise the two requests' pending lists conflate: the predecessor's
    eviction drop prefix-closes over the successor's intact ops and fires
    end_session while the successor is live."""
    harness = _make_lazy_connector()
    _admit_op(harness, "X", [[1, 2]], 0, 32)
    _finish_request(harness, "X")
    assert harness.adapter.ended_sessions == []  # teardown deferred

    _arrive_new_request(harness, "X")
    assert harness.adapter.ended_sessions == ["X"]  # released at arrival

    # The successor buffers its own first chunk; the predecessor's blocks
    # recycle; the drain must not touch the successor's state.
    _admit_op(harness, "X", [[3, 4]], 0, 32)
    harness.pool.set_hash(1, b"recycled")
    assert len(_drain(harness)) == 0
    assert harness.adapter.ended_sessions == ["X"]  # no second teardown

    # The successor's chunk is intact, chained, and storable under pressure.
    harness.pool.make_free([3, 4])
    metadata = _drain(harness)
    assert len(metadata) == 1
    assert (metadata.requests[0].op.start, metadata.requests[0].op.end) == (0, 32)


def test_id_reuse_then_preemption_cannot_swallow_predecessor_release() -> None:
    """With the arrival reclaim in place, a preemption of the successor
    finds no finished-deferred state under the id (drop_request's
    precondition holds), so the predecessor's release is never swallowed
    and the successor's own teardown stays independent."""
    harness = _make_lazy_connector()
    _admit_op(harness, "X", [[1, 2]], 0, 32)
    _finish_request(harness, "X")
    _arrive_new_request(harness, "X")
    assert harness.adapter.ended_sessions == ["X"]

    _admit_op(harness, "X", [[3, 4]], 0, 32)
    _preempt_reset(harness, "X")  # successor preempted: a plain reset

    # Resumed successor re-buffers from token zero and finishes cleanly.
    _admit_op(harness, "X", [[3, 4]], 0, 32)
    harness.pool.make_free([3, 4])
    assert len(_drain(harness, total_num_scheduled_tokens=4 * TOKENS_PER_BLOCK)) == 1
    _report_store_complete(harness, "X")
    _finish_request(harness, "X")
    assert harness.adapter.ended_sessions == ["X", "X"]


def test_id_reuse_with_in_flight_batch_defers_release_to_successor_finish() -> None:
    """The session must outlive an in-flight store AND the live successor:
    when the predecessor's batch is still awaiting its receipt at reuse
    time, the merged session ends through the successor's own finish, not
    at the receipt (the successor is running when it lands)."""
    harness = _make_lazy_connector()
    _admit_op(harness, "X", [[1, 2]], 0, 32)
    harness.pool.make_free([1, 2])
    assert len(_drain(harness)) == 1  # in flight
    _finish_request(harness, "X")

    _arrive_new_request(harness, "X")
    assert harness.adapter.ended_sessions == []  # receipt still outstanding

    # The receipt lands while the successor is running: no teardown yet.
    _report_store_complete(harness, "X")
    assert harness.adapter.ended_sessions == []

    # The successor's own finish ends the merged session, exactly once.
    _finish_request(harness, "X")
    assert harness.adapter.ended_sessions == ["X"]


####
# Count-triggered FIFO drain (legacy placeholder policy)
####


def _make_fifo_harness(threshold: int = 1) -> _Harness:
    return _make_lazy_connector(
        extra_config={
            "lmcache.mp.lazy_offload_policy": "FIFO",
            "lmcache.mp.lazy_offload_threshold": threshold,
        }
    )


def test_fifo_drain_submits_intact_request_after_finish() -> None:
    harness = _make_fifo_harness()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _finish_request(harness, "req")

    metadata = _drain(harness)

    assert len(metadata) == 1
    assert harness.pool.touched == [[1, 2]]


def test_fifo_reused_id_waits_for_predecessor_receipt_before_draining() -> None:
    harness = _make_fifo_harness()
    _admit_op(harness, "X", [[1, 2]], 0, 32)
    _finish_request(harness, "X")
    assert len(_drain(harness)) == 1  # predecessor batch in flight

    _arrive_new_request(harness, "X")
    _admit_op(harness, "X", [[3, 4]], 0, 32)
    _finish_request(harness, "X")
    assert len(_drain(harness)) == 0  # successor must not overlap by id

    _report_store_complete(harness, "X")
    assert harness.adapter.ended_sessions == []
    assert len(_drain(harness)) == 1
    _report_store_complete(harness, "X")
    assert harness.adapter.ended_sessions == ["X"]


def test_fifo_predecessor_receipt_does_not_end_live_reused_id() -> None:
    harness = _make_fifo_harness()
    _admit_op(harness, "X", [[1, 2]], 0, 32)
    _finish_request(harness, "X")
    assert len(_drain(harness)) == 1

    _arrive_new_request(harness, "X")
    _report_store_complete(harness, "X")
    assert harness.adapter.ended_sessions == []

    _finish_request(harness, "X")
    assert harness.adapter.ended_sessions == ["X"]


def test_fifo_no_drain_below_finished_request_threshold() -> None:
    harness = _make_fifo_harness(threshold=2)
    _admit_op(harness, "req-a", [[1, 2]], 0, 32)
    _finish_request(harness, "req-a")
    assert len(_drain(harness)) == 0, "drained below the finished-count threshold"

    _admit_op(harness, "req-b", [[3, 4]], 0, 32)
    _finish_request(harness, "req-b")
    assert len(_drain(harness)) == 2


def test_fifo_drain_skips_request_with_reallocated_block() -> None:
    """On a hash mismatch the FIFO path must unpin what it pinned and skip
    the request instead of storing stale data."""
    harness = _make_fifo_harness()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _finish_request(harness, "req")
    harness.pool.make_free([1, 2])
    harness.pool.set_hash(1, b"other-content")

    metadata = _drain(harness)

    assert len(metadata) == 0
    # Legacy behavior kept as-is: the mismatch unpin appends to the tail
    # (no prepend), unlike the receipt path's head re-insertion.
    assert harness.pool.freed == [([1, 2], False)]


def test_fifo_drain_drops_chunk_with_unhashed_block() -> None:
    """A block with no hash at buffering time cannot prove at drain time
    that its content survived: None == None must fail validation, not pass
    it (an evicted-and-reallocated block also reads None)."""
    harness = _make_fifo_harness()
    # Bypass the hash-seeding helper: blocks 1-2 keep block_hash=None.
    harness.manager.add_store_candidate(_make_store_metadata("req", [[1, 2]], 0, 32))
    _finish_request(harness, "req")

    metadata = _drain(harness)

    assert len(metadata) == 0
    assert harness.adapter.ended_sessions == ["req"]


def test_fifo_drain_coalesces_chunks_into_one_store() -> None:
    """A request's buffered chunks must go out as one store: the worker
    keys its in-flight store future by request id, so per-chunk submission
    overwrites futures and loses completion receipts."""
    harness = _make_fifo_harness()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _admit_op(harness, "req", [[3, 4]], 32, 64)
    _finish_request(harness, "req")

    metadata = _drain(harness)

    assert len(metadata) == 1
    op = metadata.requests[0].op
    assert (op.start, op.end) == (0, 64)
    assert op.block_ids[0] == [1, 2, 3, 4]


def test_fifo_first_chunk_mismatch_still_ends_session() -> None:
    """A request whose drain submits nothing gets no completion receipt,
    so its session must be ended at the drain itself."""
    harness = _make_fifo_harness()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _finish_request(harness, "req")
    harness.pool.set_hash(1, b"other-content")

    metadata = _drain(harness)

    assert len(metadata) == 0
    assert harness.adapter.ended_sessions == ["req"]


def test_fifo_mid_request_mismatch_submits_valid_prefix() -> None:
    """A mismatch drops the remaining chunks but the intact prefix is
    still stored, and the receipt path ends the session."""
    harness = _make_fifo_harness()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _admit_op(harness, "req", [[3, 4]], 32, 64)
    _finish_request(harness, "req")
    harness.pool.set_hash(3, b"other-content")

    metadata = _drain(harness)

    assert len(metadata) == 1
    op = metadata.requests[0].op
    assert (op.start, op.end) == (0, 32)
    # The mismatched chunk was unpinned right away; nothing else freed yet.
    assert harness.pool.freed == [([3, 4], False)]
    assert harness.adapter.ended_sessions == []

    _report_store_complete(harness, "req")
    assert harness.adapter.ended_sessions == ["req"]


def test_fifo_duplicate_receipt_is_ignored() -> None:
    """FIFO's ``notify_store_complete`` unconditionally allows teardown, so
    without the in-flight guard a resent receipt would end the session a
    second time."""
    harness = _make_fifo_harness()
    _admit_op(harness, "req", [[1, 2]], 0, 32)
    _finish_request(harness, "req")
    _drain(harness)
    _report_store_complete(harness, "req")
    assert harness.adapter.ended_sessions == ["req"]

    _report_store_complete(harness, "req")

    assert harness.adapter.ended_sessions == ["req"]
    assert len(harness.pool.freed) == 1
