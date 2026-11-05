# SPDX-License-Identifier: Apache-2.0
"""Scheduler-step and store-receipt contracts of ``LazyOffloadManager``."""

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest

pytest.importorskip("vllm", reason="lazy offload imports vLLM metadata types")

# Third Party
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)

# First Party
from lmcache.integration.vllm import lazy_offload_manager  # noqa: E402
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    MAX_STORE_ATTEMPTS,
    LazyOffloadActions,
    LazyOffloadManager,
    StoreCompletionTracker,
)
from lmcache.integration.vllm.lazy_offload_policy.base import (  # noqa: E402
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import (  # noqa: E402
    FIFOOffloadPolicy,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids  # noqa: E402

TOKENS_PER_BLOCK = 4
# Pool size of the store-receipt tests.
NUM_BLOCKS = 16

pytestmark = pytest.mark.no_shared_allocator


class _RecordingFIFOPolicy(FIFOOffloadPolicy):
    """FIFO policy that keeps every drain signal it receives."""

    def __init__(self, configs: dict[str, ConfigValue]) -> None:
        super().__init__(configs)
        self.signals: list[DrainSignals] = []

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        self.signals.append(signals)
        return super().drain(signals)


class _NoopCompletionTracker:
    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return False


def _scheduler_output(total_num_scheduled_tokens: int) -> Any:
    return SimpleNamespace(
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
    )


@pytest.mark.parametrize(
    "group_tokens_per_block, scheduled_tokens, expected_blocks",
    [
        # Qwen3.8-Flash-Next / GLM-5.3-Flash style: a scratch ring group
        # (span 0) sits between cacheable groups.
        ([16, 0, 32], 48, 3 + 2),
        # Partial blocks round up per cacheable group.
        ([16, 0, 32], 17, 2 + 1),
        # Position does not matter to the manager: it takes any span list
        # the connector accepts.
        ([0, 64], 65, 2),
    ],
)
def test_scheduler_step_skips_scratch_groups_in_block_estimate(
    monkeypatch: pytest.MonkeyPatch,
    group_tokens_per_block: list[int],
    scheduled_tokens: int,
    expected_blocks: int,
) -> None:
    """A scratch group (span ``0``) adds no next-step blocks and never divides."""
    policies: list[_RecordingFIFOPolicy] = []

    def create_policy(
        configs: dict[str, ConfigValue], gpu_block_pool: object
    ) -> _RecordingFIFOPolicy:
        policies.append(_RecordingFIFOPolicy(configs))
        return policies[-1]

    monkeypatch.setattr(lazy_offload_manager, "create_offload_policy", create_policy)
    manager = LazyOffloadManager({}, group_tokens_per_block, _NoopCompletionTracker())
    manager.bind_block_pool(cast(Any, SimpleNamespace(blocks=[])))

    actions = manager.on_scheduler_step(_scheduler_output(scheduled_tokens))

    assert actions == LazyOffloadActions()
    assert [s.est_next_step_blocks for s in policies[0].signals] == [expected_blocks]


class _Receipts:
    """Every receipt settles its batch, as with a single worker."""

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        """Settle the batch on its first receipt."""
        return True


class _TwoWorkerReceipts:
    """A batch settles once two worker receipts have arrived."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        """Count ``count`` receipts for the request.

        Returns:
            True on the second receipt, which also resets the count.
        """
        self._counts[request_id] = self._counts.get(request_id, 0) + count
        if self._counts[request_id] < 2:
            return False
        del self._counts[request_id]
        return True


def _store_step() -> Any:
    """A step scheduling as many tokens as the pool holds blocks of.

    Its next-step estimate covers the whole free queue, so every buffered
    op whose blocks are free is due.
    """
    return _scheduler_output(NUM_BLOCKS * TOKENS_PER_BLOCK)


def _manager(
    pool: BlockPool, receipts: StoreCompletionTracker | None = None
) -> LazyOffloadManager:
    """Build a one-group EVICTION_AWARE manager bound to ``pool``.

    Request "req" has arrived.

    Args:
        pool: The vLLM block pool the manager watches.
        receipts: Store completion tracker; None never settles a batch.

    Returns:
        The manager, already bound to ``pool``.
    """
    manager = LazyOffloadManager(
        {}, [TOKENS_PER_BLOCK], receipts or _NoopCompletionTracker()
    )
    manager.bind_block_pool(pool)
    manager.on_request_arrived("req")
    return manager


def _hashed_blocks(pool: BlockPool, n: int) -> list[int]:
    """Allocate ``n`` blocks and give each a distinct prefix-cache hash."""
    blocks = pool.get_new_blocks(n)
    for i, block in enumerate(blocks):
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(bytes([i + 1]) * 32), 0)
        )
    return [block.block_id for block in blocks]


def _op(start: int, block_ids: list[int]) -> LMCacheMPRequestMetadata:
    """A STORE op of request "req" over ``block_ids``, from token ``start``."""
    end = start + TOKENS_PER_BLOCK * len(block_ids)
    return LMCacheMPRequestMetadata(
        request_id="req",
        direction="STORE",
        op=LoadStoreOp(
            token_bytes=pack_token_ids(range(end)),
            block_ids=[block_ids],
            start=start,
            end=end,
        ),
    )


def test_coalesced_store_keeps_the_request_configs() -> None:
    """Ops of one request stored in one drain carry the request's configs."""
    pool = BlockPool(num_gpu_blocks=NUM_BLOCKS, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, 2)
    configs = {"lmcache.tag.tenant": "a"}
    manager = _manager(pool)
    for i, block in enumerate(ids):
        op = _op(i * TOKENS_PER_BLOCK, [block])
        op.request_configs = configs
        manager.add_store_candidate(op)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(_store_step())

    assert len(actions.stores_to_submit) == 1
    stored = actions.stores_to_submit[0]
    assert stored.op.block_ids[0] == ids
    assert stored.request_configs == configs


def test_failed_store_is_submitted_again_ahead_of_later_ops() -> None:
    """A store the worker rejected is retried with the request's next op.

    The data is still on the GPU, so the failed range is buffered again in
    front of the op that arrived while it was in flight, and the request's
    prefix chain stays whole.
    """
    pool = BlockPool(num_gpu_blocks=NUM_BLOCKS, enable_caching=True, hash_block_size=4)
    first, second = _hashed_blocks(pool, 2)
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [first]))
    pool.free_blocks([pool.blocks[first], pool.blocks[second]])
    assert len(manager.on_scheduler_step(_store_step()).stores_to_submit) == 1
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [second]))

    manager.on_store_results({"req"}, {"req": 1})
    retried = manager.on_scheduler_step(_store_step()).stores_to_submit

    assert len(retried) == 1
    assert (retried[0].op.start, retried[0].op.end) == (0, 2 * TOKENS_PER_BLOCK)
    assert retried[0].op.block_ids[0] == [first, second]


def test_failed_store_waits_for_every_worker_before_its_retry() -> None:
    """One worker's failure does not resubmit while another still reads.

    The worker accepts one store per request, and the other worker may
    still be reading the pinned blocks, so the retry waits for the batch to
    settle; the settled retry leaves no pin behind.
    """
    pool = BlockPool(num_gpu_blocks=NUM_BLOCKS, enable_caching=True, hash_block_size=4)
    (block,) = _hashed_blocks(pool, 1)
    manager = _manager(pool, receipts=_TwoWorkerReceipts())
    manager.add_store_candidate(_op(0, [block]))
    pool.free_blocks([pool.blocks[block]])
    assert len(manager.on_scheduler_step(_store_step()).stores_to_submit) == 1

    manager.on_store_results({"req"}, {"req": 1})
    unsettled = manager.on_scheduler_step(_store_step()).stores_to_submit
    manager.on_store_results(set(), {"req": 1})
    retried = manager.on_scheduler_step(_store_step()).stores_to_submit
    manager.on_store_results(set(), {"req": 2})

    assert unsettled == []
    assert len(retried) == 1
    assert retried[0].op.block_ids[0] == [block]
    assert pool.blocks[block].ref_cnt == 0


def test_failed_store_of_a_preempted_request_is_not_retried() -> None:
    """A reset detaches the failed batch from the restarted request.

    The restart recomputes from token zero, so the old range would be stored
    from blocks the request no longer owns; its receipt only unpins them.
    """
    pool = BlockPool(num_gpu_blocks=NUM_BLOCKS, enable_caching=True, hash_block_size=4)
    (block,) = _hashed_blocks(pool, 1)
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [block]))
    pool.free_blocks([pool.blocks[block]])
    assert len(manager.on_scheduler_step(_store_step()).stores_to_submit) == 1

    manager.on_request_reset("req")
    manager.on_store_results({"req"}, {"req": 1})

    assert manager.on_scheduler_step(_store_step()).stores_to_submit == []
    assert pool.blocks[block].ref_cnt == 0


def test_store_that_keeps_failing_is_given_up() -> None:
    """A store failing ``MAX_STORE_ATTEMPTS`` times in a row is dropped.

    Retrying it forever would keep the finished request's session open and
    its blocks pinned; giving up ends the session and leaves no pin.
    """
    pool = BlockPool(num_gpu_blocks=NUM_BLOCKS, enable_caching=True, hash_block_size=4)
    (block,) = _hashed_blocks(pool, 1)
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [block]))
    assert manager.on_request_finished("req").sessions_to_end == []
    pool.free_blocks([pool.blocks[block]])

    submissions = 0
    ended: list[str] = []
    for _ in range(2 * MAX_STORE_ATTEMPTS):
        stores = manager.on_scheduler_step(_store_step()).stores_to_submit
        submissions += len(stores)
        if stores:
            ended += manager.on_store_results({"req"}, {"req": 1}).sessions_to_end

    assert submissions == MAX_STORE_ATTEMPTS
    assert ended == ["req"]
    assert not manager.has_inflight_store_work()
    assert pool.blocks[block].ref_cnt == 0
