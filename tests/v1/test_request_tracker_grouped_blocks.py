# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker


def test_request_tracker_update_accepts_grouped_block_ids() -> None:
    tracker = RequestTracker(
        req_id="req-1",
        prompt_len=2,
        token_ids=[10, 11],
        allocated_block_ids_by_group=([1], [5]),
    )

    tracker.update(
        new_token_ids=[12],
        new_block_ids=([2], [6]),
    )

    assert tracker.token_ids == [10, 11, 12]
    assert tracker.allocated_block_ids_by_group == ([1, 2], [5, 6])


def test_connector_delegates_grouped_request_finished() -> None:
    expected = (True, {"first_tok": 7})
    request = SimpleNamespace(request_id="req-1")
    block_ids = ([1, 2], [5, 6])

    class _FakeEngine:
        def request_finished_all_groups(
            self,
            actual_request,
            actual_block_ids,
        ) -> tuple[bool, dict[str, int]]:
            assert actual_request is request
            assert actual_block_ids == block_ids
            return expected

    connector = LMCacheConnectorV1Dynamic.__new__(LMCacheConnectorV1Dynamic)
    connector._lmcache_engine = _FakeEngine()

    assert connector.request_finished_all_groups(request, block_ids) == expected
