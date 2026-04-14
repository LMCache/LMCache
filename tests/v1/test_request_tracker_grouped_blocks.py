# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
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
