# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side tests for MP connector preemption loading."""

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)


class _FakeRequest:
    """Minimal request exposing the fields used by the MP connector."""

    def __init__(self, token_ids: list[int]) -> None:
        self.request_id = "req-0"
        self.cache_salt = ""
        self.prompt_token_ids = list(token_ids[:4])
        self._token_ids = list(token_ids)
        self.all_token_ids = ConstantList(self._token_ids)
        self.mm_features = []
        self.status = RequestStatus.PREEMPTED

    @property
    def num_tokens(self) -> int:
        return len(self._token_ids)


class _FakeSchedulerAdapter:
    """Deterministic lookup adapter used by scheduler-side unit tests."""

    lmcache_tokens_per_chunk = 4

    def __init__(self, results: list[int | None]) -> None:
        self._results = iter(results)
        self.lookup_token_ids: list[list[int]] = []

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int], cache_salt: str
    ) -> None:
        del request_id, cache_salt
        self.lookup_token_ids.append(list(token_ids))

    def check_lookup_result(self, request_id: str) -> int | None:
        del request_id
        return next(self._results)


def _make_connector(
    adapter: _FakeSchedulerAdapter,
) -> LMCacheMPConnector:
    connector = object.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter
    connector._hit_alignment_tokens = 4
    return connector


def test_preempted_request_reloads_cached_output_prefix() -> None:
    request = _FakeRequest(list(range(12)))
    adapter = _FakeSchedulerAdapter([8])
    connector = _make_connector(adapter)

    old_tracker = LMCacheMPRequestTracker(request)
    old_tracker.allocated_block_ids = {0: [10, 11, 12]}
    old_tracker.state = LMCacheMPRequestState.READY
    connector.request_trackers[request.request_id] = old_tracker

    result = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    assert result == (8, True)
    assert adapter.lookup_token_ids == [list(range(12))]
    new_tracker = connector.request_trackers[request.request_id]
    assert new_tracker is not old_tracker
    assert new_tracker.allocated_block_ids == {}


def test_preemption_lookup_polling_keeps_fresh_tracker() -> None:
    request = _FakeRequest(list(range(12)))
    adapter = _FakeSchedulerAdapter([None, 8])
    connector = _make_connector(adapter)

    old_tracker = LMCacheMPRequestTracker(request)
    old_tracker.state = LMCacheMPRequestState.READY
    connector.request_trackers[request.request_id] = old_tracker

    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    fresh_tracker = connector.request_trackers[request.request_id]
    assert fresh_tracker is not old_tracker

    assert connector.get_num_new_matched_tokens(request, 0) == (8, True)
    assert connector.request_trackers[request.request_id] is fresh_tracker


def test_preempted_full_hit_recomputes_last_token() -> None:
    request = _FakeRequest(list(range(8)))
    adapter = _FakeSchedulerAdapter([8])
    connector = _make_connector(adapter)

    result = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    assert result == (7, True)
