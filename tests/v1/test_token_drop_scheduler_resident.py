# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.token_drop_allocator_adapter import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
)


def _connector(*, private=True, token_drop=True):
    blocks = [
        SimpleNamespace(block_id=i, ref_cnt=1 if private else 2, block_hash=None)
        for i in [3, 6, 9]
    ]
    row = [blocks]
    released = []
    tracker = SimpleNamespace(
        token_drop_spec=object() if token_drop else None,
        num_scheduled_tokens=12,
        allocated_block_ids={0: [3, 6, 9]},
    )
    connector = SimpleNamespace(
        _gpu_block_pool=SimpleNamespace(
            free_blocks=lambda block_iter: released.extend(block_iter)
        ),
        _group_tokens_per_block=[4],
        _token_drop_allocations={"td": SimpleNamespace(blocks=row)},
        _get_request_tracker=lambda req_id: tracker,
    )
    connector._require_private_token_drop_blocks = lambda req_id: (
        LMCacheMPConnector._require_private_token_drop_blocks(connector, req_id)
    )
    return connector, tracker, blocks, released


@pytest.fixture(autouse=True)
def _reset_resident():
    clear_resident_kv_tokens("td")
    yield
    clear_resident_kv_tokens("td")


def test_completed_compaction_reclaims_only_private_tail():
    connector, tracker, blocks, released = _connector()
    set_resident_kv_tokens("td", 12)
    LMCacheMPConnector._commit_token_drop_resident_updates(connector, {"td": 5})
    assert [b.block_id for b in released] == [9]
    assert connector._token_drop_allocations["td"].blocks[0] == blocks[:2]
    assert tracker.allocated_block_ids[0] == [3, 6]
    assert get_resident_kv_tokens("td") == 5


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
    connector, tracker, blocks, released = _connector(private=private, token_drop=td)
    set_resident_kv_tokens("td", 12)
    with pytest.raises((RuntimeError, ValueError)):
        LMCacheMPConnector._commit_token_drop_resident_updates(
            connector, {"td": length}
        )
    assert released == []
    assert connector._token_drop_allocations["td"].blocks[0] == blocks
    assert tracker.allocated_block_ids[0] == [3, 6, 9]
    assert get_resident_kv_tokens("td") == 12
