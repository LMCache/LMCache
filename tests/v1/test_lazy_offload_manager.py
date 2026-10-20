# SPDX-License-Identifier: Apache-2.0
"""LazyOffloadManager admission of store operations on hybrid block tables."""

# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
import pytest

pytest.importorskip("vllm", reason="the manager reads a vLLM BlockPool")

# Third Party
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)

# First Party
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadManager,
)
from lmcache.integration.vllm.lazy_offload_policy import (  # noqa: E402
    eviction_aware,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)

TOKENS_PER_BLOCK = 4
# Large enough that every admission here is scheduled in one chunk.
MAX_STEP_TOKENS = 1024
# Emit every buffered operation on the next drain, by its deferral deadline.
CONFIGS = {"lmcache.mp.lazy_offload_max_deferral_seconds": 1e-9}


class _NoReceipts:
    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return False


def _step(
    scheduled: dict[str, int] | None = None,
    preempted: set[str] | None = None,
    tables: dict[str, tuple[list[int], ...]] | None = None,
) -> SimpleNamespace:
    """A scheduler step running newly admitted requests from token zero.

    Args:
        scheduled: Request id -> tokens scheduled. None schedules one token
            of a request the manager never saw admitted, so no running
            request bounds the danger window.
        preempted: Requests the step preempted.
        tables: Request id -> its block table per group, for requests
            whose table differs from the one holding just their tokens.
    """
    return SimpleNamespace(
        total_num_scheduled_tokens=sum((scheduled or {}).values()) or 1,
        num_scheduled_tokens=dict(scheduled or {}),
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id=request_id,
                num_computed_tokens=0,
                # A table holding the scheduled tokens' blocks, null ones:
                # the null block is block 0 of a BlockPool.
                block_ids=(tables or {}).get(
                    request_id, ([0] * -(-num_tokens // TOKENS_PER_BLOCK),) * 2
                ),
            )
            for request_id, num_tokens in (scheduled or {}).items()
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            num_computed_tokens=[],
            new_block_ids=[],
            resumed_req_ids=set(),
        ),
        finished_req_ids=set(),
        preempted_req_ids=preempted,
    )


def _manager(
    pool: BlockPool,
    configs: dict | None = None,
    receipts: object | None = None,
    max_step_tokens: int = MAX_STEP_TOKENS,
    mamba_groups: dict[int, int] | None = None,
    prefill_alignment: int = 0,
    num_lookahead_tokens: int = 0,
) -> LazyOffloadManager:
    manager = LazyOffloadManager(
        CONFIGS if configs is None else configs,
        [TOKENS_PER_BLOCK, TOKENS_PER_BLOCK],
        receipts or _NoReceipts(),
        max_step_tokens=max_step_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        mamba_groups=mamba_groups,
        prefill_alignment=prefill_alignment,
    )
    manager.bind_block_pool(pool)
    return manager


def _store(block_ids: list[list[int]]) -> LMCacheMPRequestMetadata:
    n_tokens = TOKENS_PER_BLOCK * len(block_ids[0])
    return LMCacheMPRequestMetadata(
        request_id="req",
        direction="STORE",
        op=LoadStoreOp(
            token_ids=list(range(n_tokens)),
            block_ids=block_ids,
            start=0,
            end=n_tokens,
        ),
    )


def _hashed_blocks(pool: BlockPool, n: int) -> list[int]:
    blocks = pool.get_new_blocks(n)
    for i, block in enumerate(blocks):
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(bytes([i + 1]) * 32), 0)
        )
    return [block.block_id for block in blocks]


def test_null_blocks_do_not_block_the_store() -> None:
    """A mamba group's null blocks (align mode) leave the store to its live blocks."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    attention = _hashed_blocks(pool, 2)
    state = _hashed_blocks(pool, 1)
    null = pool.null_block.block_id
    manager = _manager(pool)

    manager.add_store_candidate(_store([attention, [null, state[0]]]))
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1
    stored = actions.stores_to_submit[0]
    assert stored.request_id == "req"
    assert stored.op.block_ids == [attention, [null, state[0]]]


def test_unhashed_live_block_still_rejects_the_store() -> None:
    """A live block without a hash may have been recycled: nothing is stored."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    attention = _hashed_blocks(pool, 1) + [b.block_id for b in pool.get_new_blocks(1)]
    manager = _manager(pool)

    manager.add_store_candidate(_store([attention, [pool.null_block.block_id] * 2]))
    actions = manager.on_scheduler_step(_step())

    assert actions.stores_to_submit == []


@pytest.mark.parametrize(
    ("num_tokens", "stored"),
    [(2 * TOKENS_PER_BLOCK - 1, False), (2 * TOKENS_PER_BLOCK, True)],
    ids=["inside-block", "at-boundary"],
)
def test_decode_crossing_a_block_boundary_deepens_the_window(
    num_tokens: int, stored: bool
) -> None:
    """Only a running request that can need a new block puts one at risk.

    The request has computed its whole prompt, so its next step decodes one
    token. With a full last block that token takes a new block per group,
    which can be the buffered one at the free-queue head; with room left it
    takes none, and the block stays buffered.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    manager = _manager(pool, configs={})
    manager.on_admission("running", num_tokens, 0, 0, load_async=False)
    manager.add_store_candidate(_store([ids[:1], [pool.null_block.block_id]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(_step({"running": num_tokens}))

    assert bool(actions.stores_to_submit) is stored


class _Receipts:
    """Every receipt settles its batch, as with a single worker."""

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return True


def _op(start: int, block_ids: list[list[int]]) -> LMCacheMPRequestMetadata:
    end = start + TOKENS_PER_BLOCK * len(block_ids[0])
    return LMCacheMPRequestMetadata(
        request_id="req",
        direction="STORE",
        op=LoadStoreOp(
            token_ids=list(range(end)), block_ids=block_ids, start=start, end=end
        ),
    )


def _blocked_with_freed_successor(
    pool: BlockPool,
) -> tuple[LazyOffloadManager, int]:
    """A request whose first store is in flight and whose next op is due.

    The next op's block is free at the queue head, where a running request
    that has just filled its last block can recycle it next step.

    Returns:
        The manager after that step, and the next op's block id.
    """
    null = pool.null_block.block_id
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    first, second = ids[0], ids[1]
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[first], [null]]))
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [[second], [null]]))
    # Like a mamba state retired mid-request: free while the request runs.
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[1:]])
    actions = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    assert actions.stores_to_submit == []
    return manager, second


def test_blocked_request_pins_only_blocks_in_reach() -> None:
    """An in-flight store leaves the request's other free blocks unpinned.

    No request runs, so nothing can recycle the next op's block before the
    next drain: pinning it would only cost a free block.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    null = pool.null_block.block_id
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[ids[0]], [null]]))
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [[ids[1]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[1:]])
    num_free = pool.get_num_free_blocks()

    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    assert pool.get_num_free_blocks() == num_free


def test_freed_block_of_blocked_request_survives_until_its_batch_settles() -> None:
    """A slow in-flight store does not cost the request its next op.

    The whole free queue is allocated while the first batch is in flight;
    the next op is stored once the receipt arrives, and every pin is
    released by the time its own receipt does.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, second = _blocked_with_freed_successor(pool)

    pool.get_new_blocks(pool.get_num_free_blocks())
    manager.on_store_results(set(), {"req": 1})
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1
    assert actions.stores_to_submit[0].op.block_ids[0] == [second]
    manager.on_store_results(set(), {"req": 1})
    assert pool.blocks[second].ref_cnt == 0


# An admission of this many tokens allocates at most this many blocks over
# the manager's two groups: per group, a partial block at each end and one
# for a copy-on-write or checkpoint block (no draft tokens here).
ADMISSION_TOKENS = 4 * TOKENS_PER_BLOCK
ADMISSION_BLOCKS = 2 * (4 + 2)


def _buffered_block_in_queue(
    pool: BlockPool,
    position: int,
    running_tokens: int = 0,
    mamba_groups: dict[int, int] | None = None,
    num_lookahead_tokens: int = 0,
) -> tuple[LazyOffloadManager, int, list[int]]:
    """A buffered block at ``position`` of a free queue of cached blocks.

    No request runs, so the danger depth is zero and the drain leaves it
    buffered.

    Args:
        pool: The block pool, with every block free.
        position: Where the buffered block sits in the free queue.
        running_tokens: When set, a request ``"running"`` of that many
            tokens is admitted first, to be scheduled by the caller.
        mamba_groups: The manager's align-mode Mamba groups.
        num_lookahead_tokens: The manager's draft tokens per step.

    Returns:
        The manager, the buffered block's id, and every other free block.
    """
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    block = ids[position]
    manager = _manager(
        pool,
        configs={},
        receipts=_Receipts(),
        mamba_groups=mamba_groups,
        num_lookahead_tokens=num_lookahead_tokens,
    )
    if running_tokens:
        manager.on_admission("running", running_tokens, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[block], [pool.null_block.block_id]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])
    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    return manager, block, [block_id for block_id in ids if block_id != block]


@pytest.mark.parametrize("admitted", [False, True], ids=["no-admission", "admission"])
def test_admission_keeps_buffered_block_its_allocation_would_recycle(
    admitted: bool,
) -> None:
    """An admission recycling past a buffered block does not cost a store.

    Only the admission notice can save the block, and the notice must not
    shrink the free queue the admission allocates from. Out of reach, the
    block stays buffered rather than stored; releasing the buffer, as the
    request's preemption does, still finds it intact.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, _, _ = _buffered_block_in_queue(pool, position=5)
    num_free = pool.get_num_free_blocks()

    if admitted:
        manager.on_admission("new", ADMISSION_TOKENS, 0, 0, load_async=False)
        assert pool.get_num_free_blocks() == num_free
    pool.get_new_blocks(ADMISSION_BLOCKS)
    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    actions = manager.on_scheduler_step(_step(preempted={"req"}))

    assert bool(actions.stores_to_submit) is admitted


@pytest.mark.parametrize(
    ("mamba_groups", "pinned"),
    [(None, True), ({0: 0, 1: 0}, False)],
    ids=["attention", "mamba"],
)
def test_long_admission_of_mamba_groups_takes_a_few_blocks(
    mamba_groups: dict[int, int] | None, pinned: bool
) -> None:
    """An align-mode Mamba group allocates its state block, not the prompt.

    The admission's ten blocks of prompt per group would reach the buffered
    block at the queue tail, which a squeeze pins. Mamba groups null all
    but the state block, so the allocation stays at the head and the block
    is left in the queue, buffered.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    num_free = pool.get_num_free_blocks()
    manager, _, _ = _buffered_block_in_queue(
        pool, position=num_free - 1, mamba_groups=mamba_groups
    )

    manager.on_admission("new", 10 * TOKENS_PER_BLOCK, 0, 0, load_async=False)

    assert (pool.get_num_free_blocks() < num_free) is pinned


def test_external_hit_of_mamba_groups_takes_a_state_block_of_its_own() -> None:
    """vLLM gives an external hit a Mamba state block before allocating.

    Each Mamba group takes one block for the external hit and one state
    block for the chunk, four in all, which reach the buffered block third
    in the queue. Saved by the admission notice, it is still intact when
    the request's preemption releases it.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, _, _ = _buffered_block_in_queue(
        pool, position=2, mamba_groups={0: 0, 1: 0}
    )

    manager.on_admission(
        "new", 20 * TOKENS_PER_BLOCK, 0, 10 * TOKENS_PER_BLOCK, load_async=False
    )
    pool.get_new_blocks(4)
    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    actions = manager.on_scheduler_step(_step(preempted={"req"}))

    assert actions.stores_to_submit != []


def test_resumed_load_of_mamba_groups_reserves_no_draft_blocks_again() -> None:
    """A resumed load's Mamba state takes one block per group, no drafts.

    The admission's allocation for the hit reserved the draft-token state
    blocks, so vLLM caps the resumed allocation at the state block alone:
    two blocks in all, which stop short of the buffered block third in the
    queue. Counting the drafts again would reach the tail and pin it.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    num_free = pool.get_num_free_blocks()
    manager, _, _ = _buffered_block_in_queue(
        pool,
        position=num_free - 1,
        mamba_groups={0: 0, 1: 0},
        num_lookahead_tokens=3,
    )
    manager.on_admission(
        "new", TOKENS_PER_BLOCK + ADMISSION_TOKENS, 0, TOKENS_PER_BLOCK, load_async=True
    )
    pool.get_new_blocks(num_free - 3)
    num_free = pool.get_num_free_blocks()

    manager.on_loads_finished({"new"})

    assert pool.get_num_free_blocks() == num_free


def test_squeezed_admission_pins_rather_than_loses_the_store() -> None:
    """An admission that needs every free block cannot take a buffered one.

    The tail is no refuge when the allocation reaches it, so the block is
    pinned: vLLM gets the rest of the queue, and the next drain stores it.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, _, others = _buffered_block_in_queue(pool, position=5)

    manager.on_admission(
        "new", TOKENS_PER_BLOCK * (len(others) + 1), 0, 0, load_async=False
    )
    pool.get_new_blocks(pool.get_num_free_blocks())
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1


def test_step_scheduling_nothing_stores_what_a_squeezed_admission_pinned() -> None:
    """A pin that defers the admission is stored without a forward pass.

    With the buffered block pinned the admission does not fit, so if no
    request runs, the next steps schedule nothing. Waiting for a step that
    schedules tokens would keep the pin, and the admission, waiting forever.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, _, others = _buffered_block_in_queue(pool, position=5)
    manager.on_admission(
        "new", TOKENS_PER_BLOCK * (len(others) + 1), 0, 0, load_async=False
    )
    nothing_scheduled = SimpleNamespace(
        total_num_scheduled_tokens=0,
        num_scheduled_tokens={},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], num_computed_tokens=[], new_block_ids=[], resumed_req_ids=set()
        ),
        finished_req_ids=set(),
        preempted_req_ids=None,
    )

    actions = manager.on_scheduler_step(nothing_scheduled)

    assert len(actions.stores_to_submit) == 1


@pytest.mark.parametrize("notified", [False, True], ids=["no-notice", "notice"])
def test_resumed_load_keeps_buffered_block_its_allocation_would_recycle(
    notified: bool,
) -> None:
    """A finished asynchronous load does not cost a store when it resumes.

    vLLM allocates only for the hit at admission, and the rest of the
    request's blocks once the load finishes, without asking for a hit
    again: the load-finished notice is all that can save the block.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    hit_blocks = 2 * (1 + 2)
    manager, _, _ = _buffered_block_in_queue(pool, position=hit_blocks + 2)

    manager.on_admission(
        "new", TOKENS_PER_BLOCK + ADMISSION_TOKENS, 0, TOKENS_PER_BLOCK, load_async=True
    )
    pool.get_new_blocks(hit_blocks)
    if notified:
        manager.on_loads_finished({"new"})
    pool.get_new_blocks(ADMISSION_BLOCKS)
    actions = manager.on_scheduler_step(_step(preempted={"req"}))

    assert bool(actions.stores_to_submit) is notified


def test_resumed_load_left_waiting_keeps_its_protection() -> None:
    """A finished load vLLM schedules steps later still cannot cost a store.

    The load-finished notice moves the buffered block to the tail, but no
    step schedules the request yet. Another admission takes the queue head
    and a finished request's cached blocks queue up behind the block, so
    when the resumed request finally allocates, its chunk reaches the block.
    The drain in between must store it.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, _, _ = _buffered_block_in_queue(pool, position=10)
    manager.on_admission(
        "new", TOKENS_PER_BLOCK + ADMISSION_TOKENS, 0, TOKENS_PER_BLOCK, load_async=True
    )
    pool.get_new_blocks(2 * (1 + 2))
    manager.on_loads_finished({"new"})

    other_tokens = 6 * TOKENS_PER_BLOCK
    manager.on_admission("other", other_tokens, 0, 0, load_async=False)
    finished = pool.get_new_blocks(2 * (6 + 2))
    for i, block in enumerate(finished):
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(bytes([100 + i]) * 32), 0)
        )
    pool.free_blocks(finished)
    stores = manager.on_scheduler_step(_step()).stores_to_submit
    pool.get_new_blocks(ADMISSION_BLOCKS)
    stores += manager.on_scheduler_step(_step(preempted={"req"})).stores_to_submit

    assert stores != []


def test_store_over_the_drain_budget_survives_until_the_next_drain() -> None:
    """A due op the budget defers is not recycled while it waits.

    Two requests have a block at the free-queue head and a running request
    is about to take two blocks; one drain stores only one op. The other
    moves out of reach and stays buffered, still intact once the running
    request has allocated.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool,
        configs={"lmcache.mp.lazy_offload_max_drain_per_step": 1},
        receipts=_Receipts(),
    )
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    for request_id, block_id in (("first", ids[0]), ("second", ids[1])):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.get_new_blocks(2)
    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    second = manager.on_scheduler_step(_step(preempted={"second"}))

    assert [store.request_id for store in first.stores_to_submit] == ["first"]
    assert [store.request_id for store in second.stores_to_submit] == ["second"]


def test_stored_block_returns_to_queue_head() -> None:
    """Storing a free block does not refresh it in vLLM's eviction order.

    The block was next to be recycled when the store pinned it; once the
    store settles it is next again, rather than the most recently used.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    running = 2 * TOKENS_PER_BLOCK
    manager, block, _ = _buffered_block_in_queue(
        pool, position=0, running_tokens=running
    )
    stores = manager.on_scheduler_step(_step({"running": running}))
    assert len(stores.stores_to_submit) == 1

    manager.on_store_results(set(), {"req": 1})

    assert pool.get_new_blocks(1)[0].block_id == block


def test_preemption_unpins_held_blocks() -> None:
    """A reset discards held ops, and with them their pins."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, second = _blocked_with_freed_successor(pool)
    free_while_held = pool.get_num_free_blocks()

    manager.on_request_reset("req")

    assert pool.blocks[second].ref_cnt == 0
    assert pool.get_num_free_blocks() == free_while_held + 1


def test_window_reaches_past_blocks_the_drain_takes_out() -> None:
    """Pinning a due block does not bring the next one into reach unseen.

    The running request can take two blocks next step. The first buffered
    block is at the head; once it is pinned, the second, two places behind
    it, is among the first two free blocks, so it is stored too.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, configs={}, receipts=_Receipts())
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    for request_id, block_id in (("first", ids[0]), ("second", ids[2])):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.get_new_blocks(2)
    second = manager.on_scheduler_step(_step())

    stored = first.stores_to_submit + second.stores_to_submit
    assert {store.request_id for store in stored} == {"first", "second"}


def test_preempted_request_releases_its_whole_buffer() -> None:
    """A preempted request's buffered ops are stored before its restart.

    Rescheduled, the request starts over from token zero, which discards
    its buffer; storing it at preemption keeps what eager offload stores.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, _, _ = _buffered_block_in_queue(pool, position=5)

    actions = manager.on_scheduler_step(_step(preempted={"req"}))

    assert [store.request_id for store in actions.stores_to_submit] == ["req"]


def test_admission_counts_the_local_hit_it_takes_from_the_queue() -> None:
    """A local prefix hit does not pull a buffered block into the allocation.

    vLLM touches the hit blocks, taking them out of the free queue, before
    it allocates from the head: every hit block ahead of the buffered one
    brings it one place closer.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    hit = 2 * TOKENS_PER_BLOCK
    new_blocks = 2 * (2 + 2)
    block = ids[new_blocks + 1]
    manager = _manager(pool, configs={}, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[block], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    manager.on_admission("new", hit + 2 * TOKENS_PER_BLOCK, hit, 0, load_async=False)
    pool.touch([pool.blocks[block_id] for block_id in ids[:2]])
    pool.get_new_blocks(new_blocks)
    actions = manager.on_scheduler_step(_step(preempted={"req"}))

    assert [store.request_id for store in actions.stores_to_submit] == ["req"]


def test_token_budget_caps_the_window() -> None:
    """Prefilling requests share one step's tokens, so only so many advance.

    Four requests each need a full block's worth of tokens to cross into a
    new one, and the step has room for two: at most two crossings per
    group. A buffered block past those four is not in reach, however much
    each request could take alone.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool, configs={}, receipts=_Receipts(), max_step_tokens=2 * TOKENS_PER_BLOCK
    )
    running = [f"running-{index}" for index in range(4)]
    for request_id in running:
        manager.on_admission(request_id, 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[4]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step({request_id: TOKENS_PER_BLOCK + 1 for request_id in running})
    )

    assert actions.stores_to_submit == []


@pytest.mark.parametrize(
    ("position", "stored"), [(1, True), (2, False)], ids=["in-reach", "past-reach"]
)
def test_mamba_group_takes_one_block_per_step(position: int, stored: bool) -> None:
    """An align-mode Mamba group allocates one state block per step.

    The prefilling request could cross many blocks of an attention group
    in one chunk, but each Mamba group takes a single block: two groups
    reach two blocks into the queue, and no further.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool, configs={}, receipts=_Receipts(), mamba_groups={0: 0, 1: 0}
    )
    manager.on_admission("running", 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[position]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(_step({"running": TOKENS_PER_BLOCK}))

    assert (actions.stores_to_submit != []) is stored


@pytest.mark.parametrize(
    ("position", "stored"), [(1, True), (2, False)], ids=["in-reach", "past-reach"]
)
def test_scratch_group_adds_no_reach(position: int, stored: bool) -> None:
    """A scratch group (``0`` tokens per block) is left out of every bound.

    It holds one block per request for the request's lifetime, so a step
    never grows it: with a scratch group between two Mamba groups, the
    window reaches the same two blocks as without it.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = LazyOffloadManager(
        {},
        [TOKENS_PER_BLOCK, 0, TOKENS_PER_BLOCK],
        _Receipts(),
        max_step_tokens=MAX_STEP_TOKENS,
        num_lookahead_tokens=0,
        mamba_groups={0: 0, 2: 0},
    )
    manager.bind_block_pool(pool)
    manager.on_admission("running", 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[position]], [null], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step({"running": TOKENS_PER_BLOCK}, tables={"running": ([null],) * 3})
    )

    assert (actions.stores_to_submit != []) is stored


@pytest.mark.parametrize(
    ("prefill_alignment", "stored"), [(0, True), (TOKENS_PER_BLOCK, False)]
)
def test_aligned_prefill_crossing_costs_a_whole_block(
    prefill_alignment: int, stored: bool
) -> None:
    """A chunk short of the prompt's end ends on an aligned position.

    Four requests sit on a block boundary, so one token each would cross
    into a new block, and the eight-token step reaches five blocks per
    group. Aligned chunks spend a whole block's tokens per crossing, so the
    step reaches two per group, short of the buffered block.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool,
        configs={},
        receipts=_Receipts(),
        max_step_tokens=2 * TOKENS_PER_BLOCK,
        prefill_alignment=prefill_alignment,
    )
    running = [f"running-{index}" for index in range(4)]
    for request_id in running:
        manager.on_admission(request_id, 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[4]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step({request_id: TOKENS_PER_BLOCK for request_id in running})
    )

    assert (actions.stores_to_submit != []) is stored


def test_leftover_tokens_short_of_a_crossing_reach_no_block() -> None:
    """A crossing is paid in whole tokens or not at all.

    Four requests each need a full block's worth of tokens to cross into a
    new one, and the step has room for two crossings and one token more:
    the token cannot buy part of a third block, so each group still takes
    at most two and the fifth buffered block is out of reach.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool,
        configs={},
        receipts=_Receipts(),
        max_step_tokens=2 * TOKENS_PER_BLOCK + 1,
    )
    running = [f"running-{index}" for index in range(4)]
    for request_id in running:
        manager.on_admission(request_id, 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[4]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step({request_id: TOKENS_PER_BLOCK + 1 for request_id in running})
    )

    assert actions.stores_to_submit == []


@pytest.mark.parametrize(
    ("table_length", "stored"), [(2, True), (3, False)], ids=["filled", "ahead"]
)
def test_block_already_in_the_table_is_not_allocated_again(
    table_length: int, stored: bool
) -> None:
    """A decode token landing in a block the table holds takes no new one.

    The request has computed two blocks' worth of tokens, so its next token
    crosses into a third block, unless the table already holds it (as
    lookahead slots of an earlier step leave it). Then the step allocates
    nothing, and the buffered block at the queue head is out of reach.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, configs={}, receipts=_Receipts())
    num_tokens = 2 * TOKENS_PER_BLOCK
    manager.on_admission("running", num_tokens, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[0]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step(
            {"running": num_tokens},
            tables={"running": ([null] * table_length,) * 2},
        )
    )

    assert (actions.stores_to_submit != []) is stored


def test_block_another_group_held_counts_as_allocated() -> None:
    """Allocated blocks are counted per group.

    Groups share one block pool, so a block one group of the request freed
    can come back to another group of the same request; it still comes off
    the free queue, and the ledger counts it. LMCache loggers do not
    propagate, so the module logger is patched.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={})
    num_tokens = TOKENS_PER_BLOCK
    manager.on_admission("running", num_tokens, 0, 0, load_async=False)
    manager.on_scheduler_step(
        _step({"running": num_tokens}, tables={"running": ([1], [2])})
    )
    decode = SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"running": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["running"],
            num_computed_tokens=[num_tokens],
            new_block_ids=[([3], [1])],
            resumed_req_ids=set(),
        ),
        finished_req_ids=set(),
        preempted_req_ids=None,
    )

    with patch.object(eviction_aware.logger, "info") as mock_info:
        manager.on_scheduler_step(decode)

    ledgers = [
        call.args[1]
        for call in mock_info.call_args_list
        if call.args[0] == "Lazy offload counters: %s"
    ]
    assert " bound_allocated_blocks=2 " in ledgers[-1]


@pytest.mark.parametrize(
    ("new_block_ids", "exceeded"),
    [(([1], [2]), False), (([1, 2], [3]), True)],
    ids=["within-bound", "over-bound"],
)
def test_allocation_over_the_bound_is_reported(
    new_block_ids: tuple[list[int], ...], exceeded: bool
) -> None:
    """A running request allocating past the danger depth is logged.

    The decoding request has just filled its last block, so the bound is
    one new block per group; a third block means the bound does not hold.
    Either way the ledger sums the bound and the allocation, whose ratio is
    the bound's slack. LMCache loggers do not propagate, so the module
    logger is patched.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={})
    num_tokens = 2 * TOKENS_PER_BLOCK
    manager.on_admission("running", num_tokens, 0, 0, load_async=False)
    manager.on_scheduler_step(_step({"running": num_tokens}))
    decode = SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"running": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["running"],
            num_computed_tokens=[num_tokens],
            new_block_ids=[new_block_ids],
            resumed_req_ids=set(),
        ),
        finished_req_ids=set(),
        preempted_req_ids=None,
    )

    with patch.object(eviction_aware.logger, "info") as mock_info:
        manager.on_scheduler_step(decode)

    reported = any(
        "over the previous drain's bound" in call.args[0]
        for call in mock_info.call_args_list
    )
    ledgers = [
        call.args[1]
        for call in mock_info.call_args_list
        if call.args[0] == "Lazy offload counters: %s"
    ]
    allocated = sum(len(group) for group in new_block_ids)
    assert reported is exceeded
    assert f"bound_blocks=2 bound_allocated_blocks={allocated} " in ledgers[-1]


def test_deferred_op_is_pinned_when_the_next_step_can_take_the_whole_queue() -> None:
    """The tail is no refuge for a deferred op when the window reaches it.

    Only three blocks are free and the running request can take two: past
    the drain budget, the second op is pinned rather than moved to the
    tail, so allocating the whole queue leaves it to store next drain.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool,
        configs={"lmcache.mp.lazy_offload_max_drain_per_step": 1},
        receipts=_Receipts(),
    )
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    for request_id, block_id in (("first", ids[0]), ("second", ids[1])):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[:3]])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.get_new_blocks(pool.get_num_free_blocks())
    second = manager.on_scheduler_step(_step())

    assert [store.request_id for store in first.stores_to_submit] == ["first"]
    assert [store.request_id for store in second.stores_to_submit] == ["second"]


def test_squeezed_drain_pins_a_block_its_request_still_owns() -> None:
    """A window reaching the tail also covers blocks not yet in the queue.

    vLLM frees a running request's older blocks, as align-mode Mamba drops
    past states, before the next step allocates; they join a tail already
    in reach. Past the drain budget, the op on such a block is pinned while
    its request still owns it, so it survives to the next drain.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool,
        configs={"lmcache.mp.lazy_offload_max_drain_per_step": 1},
        receipts=_Receipts(),
    )
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    for request_id, block_id in (("first", ids[0]), ("owned", ids[5])):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[:2]])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.free_blocks([pool.blocks[ids[5]]])
    pool.get_new_blocks(pool.get_num_free_blocks())
    second = manager.on_scheduler_step(_step())

    assert [store.request_id for store in first.stores_to_submit] == ["first"]
    assert [store.request_id for store in second.stores_to_submit] == ["owned"]


def test_squeezed_admission_pins_a_block_its_request_still_owns() -> None:
    """An allocation reaching the tail takes blocks freed before it runs.

    A resumed load allocates after the running requests, which may free a
    buffered block first. The walk met no buffered block, yet reached the
    tail, so the op is pinned while its request still owns the block.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    manager = _manager(pool, configs={}, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[ids[0]], [pool.null_block.block_id]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[1:]])
    assert manager.on_scheduler_step(_step()).stores_to_submit == []

    manager.on_admission(
        "new",
        TOKENS_PER_BLOCK * (pool.get_num_free_blocks() + 1),
        0,
        0,
        load_async=False,
    )
    pool.free_blocks([pool.blocks[ids[0]]])
    pool.get_new_blocks(pool.get_num_free_blocks())
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1
