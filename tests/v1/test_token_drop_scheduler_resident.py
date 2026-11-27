# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    TokenDropScheduler,
    get_resident_kv_state,
)
from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPWorkerMetadata


def _connector(*, private=True, token_drop=True):
    blocks = [
        SimpleNamespace(block_id=i, ref_cnt=1 if private else 2, block_hash=None)
        for i in [3, 6, 9]
    ]
    row = [blocks]
    released = []
    tracker = SimpleNamespace(
        token_drop_spec=object() if token_drop else None,
        token_drop_generation=1,
        num_scheduled_tokens=12,
        allocated_block_ids={0: [3, 6, 9]},
    )
    pool = Mock()
    pool.free_blocks.side_effect = lambda block_iter: released.extend(block_iter)
    scheduler = TokenDropScheduler()
    scheduler.bind_block_pool(pool)
    # Register private blocks first; the rejection case mutates refcount after
    # registration to prove reclaim revalidates native ownership.
    for block in blocks:
        block.ref_cnt = 1
    scheduler.register_allocation("td", SimpleNamespace(blocks=row))
    if not private:
        blocks[0].ref_cnt = 2
    return scheduler, pool, tracker, blocks, released


def _commit(scheduler, pool, tracker, length):
    scheduler.commit_completed_updates(
        LMCacheMPWorkerMetadata(
            completed_store_requests={}, resident_kv_updates={"td": (1, length)}
        ),
        {"td": tracker},
        pool,
        4,
    )


def test_completed_compaction_reclaims_only_private_tail():
    scheduler, pool, tracker, blocks, released = _connector()
    state = get_resident_kv_state(pool)
    state.set("td", 12)
    _commit(scheduler, pool, tracker, 5)
    assert [b.block_id for b in released] == [9]
    assert [b.block_id for b in blocks] == [3, 6]
    assert tracker.allocated_block_ids[0] == [3, 6]
    assert state.get("td") == 5


@pytest.mark.parametrize(
    "private,td,length",
    [
        (False, True, 5),
        (True, False, 5),
        (True, True, 0),
        (True, True, 13),
    ],
)
def test_invalid_resident_commit_never_reclaims(private, td, length):
    scheduler, pool, tracker, blocks, released = _connector(
        private=private, token_drop=td
    )
    state = get_resident_kv_state(pool)
    state.set("td", 12)
    if td:
        with pytest.raises((RuntimeError, ValueError)):
            _commit(scheduler, pool, tracker, length)
    else:
        _commit(scheduler, pool, tracker, length)
    assert released == []
    assert [b.block_id for b in blocks] == [3, 6, 9]
    assert tracker.allocated_block_ids[0] == [3, 6, 9]
    assert state.get("td") == 12
