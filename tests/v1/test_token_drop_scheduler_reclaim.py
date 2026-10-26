# SPDX-License-Identifier: Apache-2.0
"""Scheduler resident-KV commits, private-block reclaim, and preemption."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    commit_completed_token_drop_updates,
    commit_token_drop_resident_updates,
    get_resident_kv_tokens,
    require_private_token_drop_blocks,
    reset_preempted_token_drop_requests,
    reset_token_drop_request,
    set_resident_kv_tokens,
)


@pytest.fixture(autouse=True)
def _clear_resident_frontiers():
    for request_id in ("td", "td2"):
        clear_resident_kv_tokens(request_id)
    yield
    for request_id in ("td", "td2"):
        clear_resident_kv_tokens(request_id)


def _connector(*, private=True, hashed=False, token_drop=True):
    block_ids = [3, 6, 9]
    blocks = [
        SimpleNamespace(
            block_id=i,
            ref_cnt=1 if private else 2,
            block_hash="cached" if hashed else None,
        )
        for i in block_ids
    ]
    tracker = SimpleNamespace(
        token_drop_spec=object() if token_drop else None,
        num_scheduled_tokens=12,
        allocated_block_ids={0: block_ids.copy()},
    )
    released = []
    connector = SimpleNamespace(
        _gpu_block_pool=SimpleNamespace(
            free_blocks=lambda iterable: released.extend(iterable)
        ),
        _group_tokens_per_block=[4],
        _token_drop_allocations={
            "td": SimpleNamespace(blocks=[blocks]),
        },
        _token_drop_finished_before_output=set(),
        _get_request_tracker=lambda request_id: tracker,
    )
    return connector, tracker, blocks, released


def test_reclaim_completed_private_tail_only():
    connector, tracker, blocks, released = _connector()
    set_resident_kv_tokens("td", 12)
    commit_completed_token_drop_updates(connector, {"td": 5})
    assert [block.block_id for block in released] == [9]
    assert connector._token_drop_allocations["td"].blocks[0] == blocks[:2]
    assert tracker.allocated_block_ids[0] == [3, 6]
    assert get_resident_kv_tokens("td") == 5


@pytest.mark.parametrize(
    "private,hashed,token_drop,resident_length",
    [
        (False, False, True, 5),
        (True, True, True, 5),
        (True, False, False, 5),
        (True, False, True, 0),
        (True, False, True, 13),
    ],
)
def test_invalid_commit_does_not_reclaim(private, hashed, token_drop, resident_length):
    connector, tracker, blocks, released = _connector(
        private=private, hashed=hashed, token_drop=token_drop
    )
    set_resident_kv_tokens("td", 12)
    with pytest.raises((RuntimeError, ValueError)):
        commit_token_drop_resident_updates(connector, {"td": resident_length})
    assert released == []
    assert tracker.allocated_block_ids[0] == [3, 6, 9]
    assert connector._token_drop_allocations["td"].blocks[0] == blocks
    assert get_resident_kv_tokens("td") == 12


def test_same_step_preemption_preserves_new_allocation():
    connector, _, blocks, released = _connector()
    set_resident_kv_tokens("td", 7)
    output = SimpleNamespace(
        preempted_req_ids={"td"},
        num_scheduled_tokens={"td": 12},
    )
    assert reset_preempted_token_drop_requests(connector, output) == {"td"}
    assert connector._token_drop_allocations["td"].blocks[0] is blocks
    assert get_resident_kv_tokens("td") is None
    assert connector._token_drop_finished_before_output == set()
    assert released == []


def test_preemption_without_reschedule_removes_request_state():
    connector, _, _, _ = _connector()
    set_resident_kv_tokens("td", 7)
    output = SimpleNamespace(preempted_req_ids={"td"}, num_scheduled_tokens={})
    assert reset_preempted_token_drop_requests(connector, output) == {"td"}
    assert "td" not in connector._token_drop_allocations
    assert connector._token_drop_finished_before_output == {"td"}
    assert get_resident_kv_tokens("td") is None


def test_finished_request_stale_completion_cannot_reclaim_new_blocks():
    connector, _, blocks, released = _connector()
    set_resident_kv_tokens("td", 12)
    reset_token_drop_request(connector, "td")
    # A new allocation for the same request ID must not consume stale output.
    connector._token_drop_allocations["td"] = SimpleNamespace(blocks=[blocks])
    set_resident_kv_tokens("td", 12)
    commit_completed_token_drop_updates(connector, {"td": 4})
    assert released == []
    assert get_resident_kv_tokens("td") == 12
    assert connector._token_drop_allocations["td"].blocks[0] == blocks


def test_ownership_validation_rejects_missing_allocation():
    connector, _, _, _ = _connector()
    with pytest.raises(RuntimeError, match="registered KV group"):
        require_private_token_drop_blocks(connector, "unknown")


def test_native_allocator_receives_committed_resident_length():
    connector, _, _, released = _connector()
    native_updates = []
    connector._kv_cache_manager = SimpleNamespace(
        set_resident_kv_tokens=lambda request_id, length: native_updates.append(
            (request_id, length)
        )
    )
    set_resident_kv_tokens("td", 12)

    commit_completed_token_drop_updates(connector, {"td": 5})

    assert [block.block_id for block in released] == [9]
    assert native_updates == [("td", 5)]
    assert get_resident_kv_tokens("td") == 5
