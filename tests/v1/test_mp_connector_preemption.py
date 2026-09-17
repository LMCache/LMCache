# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side connector contracts for preempted / resumed requests.

The end-to-end behaviour is covered by ``tests/v1/mp_preemption`` against a
real vLLM scheduler; these tests pin the individual contracts directly.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.lmcache_mp_metrics import (  # noqa: E402
    LMCacheMPConnectorStats,
)

CHUNK = 4


class _Request:
    def __init__(self, request_id: str, tokens: list[int], status: RequestStatus):
        self.request_id = request_id
        self.status = status
        self.num_computed_tokens = 0
        self.num_preemptions = 0
        self.cache_salt = ""
        self.prompt_token_ids = list(tokens)
        self.all_token_ids = list(tokens)
        self.mm_features: list[object] = []
        self.sampling_params = None
        self.kv_transfer_params = None


def _connector(hit_tokens: int | None) -> tuple[LMCacheMPConnector, MagicMock]:
    """A scheduler-role connector over a mocked adapter answering ``hit_tokens``."""
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector._finished_without_store = set()
    connector._connector_stats = LMCacheMPConnectorStats()
    connector._hit_alignment_tokens = 2
    connector.lazy_offload = False
    connector._can_store = True
    adapter = MagicMock(name="scheduler_adapter")
    adapter.lmcache_tokens_per_chunk = CHUNK
    adapter.check_lookup_result.return_value = hit_tokens
    connector.scheduler_adapter = adapter
    return connector, adapter


def test_lookup_hit_is_idempotent_across_polls():
    """vLLM re-polls after a failed allocate_slots; the hit must not double."""
    tracker = LMCacheMPRequestTracker(
        _Request("r", list(range(12)), RequestStatus.WAITING)
    )
    tracker.set_lookup_hit(8)
    tracker.set_lookup_hit(8)
    assert tracker.num_stored_tokens == 8
    assert tracker.num_lmcache_hit_tokens == 8


def test_lookup_hit_after_stores_is_rejected():
    tracker = LMCacheMPRequestTracker(
        _Request("r", list(range(12)), RequestStatus.WAITING)
    )
    tracker.set_lookup_hit(4)
    tracker.increase_num_stored_tokens(4)  # a store [4, 8) was emitted
    with pytest.raises(ValueError):
        tracker.set_lookup_hit(4)


def test_connector_polls_twice_without_double_counting():
    connector, adapter = _connector(hit_tokens=8)
    request = _Request("r", list(range(12)), RequestStatus.WAITING)

    first = connector.get_num_new_matched_tokens(request, 0)
    second = connector.get_num_new_matched_tokens(request, 0)

    assert first == second == (8, True)
    tracker = connector.request_trackers["r"]
    assert tracker.num_stored_tokens == 8
    assert tracker.num_lmcache_hit_tokens == 8
    adapter.maybe_submit_lookup_request.assert_called()


def test_preempted_request_gets_fresh_tracker_and_loads():
    """A resumed request looks up over prompt + generated tokens and loads."""
    connector, adapter = _connector(hit_tokens=8)
    request = _Request("r", list(range(12)), RequestStatus.RUNNING)
    stale = LMCacheMPRequestTracker(request)
    stale.state = LMCacheMPRequestState.READY
    stale.allocated_block_ids = {0: [7, 8, 9]}
    stale.num_stored_tokens = 12
    connector.request_trackers["r"] = stale

    request.status = RequestStatus.PREEMPTED
    request.num_preemptions = 1
    request.all_token_ids = list(range(12)) + [100, 101, 102, 103]

    need, is_async = connector.get_num_new_matched_tokens(request, 4)

    tracker = connector.request_trackers["r"]
    assert tracker is not stale
    assert tracker.state == LMCacheMPRequestState.PREFETCHING
    assert tracker.allocated_block_ids == {}
    # Lookup used every token the request has now, not just the prompt.
    submitted = adapter.maybe_submit_lookup_request.call_args.kwargs["token_ids"]
    assert submitted == request.all_token_ids
    assert (need, is_async) == (8 - 4, True)
    assert tracker.num_vllm_hit_tokens == 4
    assert tracker.num_stored_tokens == 8


def test_full_hit_on_resume_recomputes_last_token():
    connector, _adapter = _connector(hit_tokens=16)
    request = _Request("r", list(range(16)), RequestStatus.PREEMPTED)
    request.num_preemptions = 1
    need, is_async = connector.get_num_new_matched_tokens(request, 0)
    assert (need, is_async) == (15, True)


def test_request_finished_without_forward_frees_via_vllm_and_releases_locks():
    """Aborted while waiting: delay_free=False, skip set, locks freed."""
    connector, adapter = _connector(hit_tokens=8)
    request = _Request("r", list(range(12)), RequestStatus.WAITING)
    connector.get_num_new_matched_tokens(request, 0)

    delay_free, params = connector.request_finished(request, [])

    assert delay_free is False
    assert params is None
    assert connector._finished_without_store == {"r"}
    adapter.free_lookup_locks.assert_called_once()
    kwargs = adapter.free_lookup_locks.call_args.kwargs
    assert (kwargs["start"], kwargs["end"]) == (0, 8)
    adapter.end_session.assert_called_once_with("r")
    adapter.cleanup_lookup_result.assert_called_with("r")
    assert "r" not in connector.request_trackers


def test_request_finished_after_forward_delays_free_and_keeps_locks_alone():
    connector, adapter = _connector(hit_tokens=0)
    request = _Request("r", list(range(12)), RequestStatus.WAITING)
    connector.get_num_new_matched_tokens(request, 0)
    tracker = connector.request_trackers["r"]
    tracker.state = LMCacheMPRequestState.READY
    tracker.increase_num_scheduled_tokens(12)

    delay_free, _params = connector.request_finished(request, [1, 2, 3])

    assert delay_free is True
    assert connector._finished_without_store == set()
    adapter.free_lookup_locks.assert_not_called()
