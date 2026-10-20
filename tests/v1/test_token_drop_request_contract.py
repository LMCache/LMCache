# SPDX-License-Identifier: Apache-2.0
"""Scheduler-to-worker token-dropping request state."""

# Standard
from typing import Literal

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.token_drop import (
    LMCacheMPTokenDropRequestState,
)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_request_state_has_no_worker_batch_row_until_assigned(
    phase: Literal["prefill", "decode"],
):
    state = LMCacheMPTokenDropRequestState(
        request_id="req",
        algorithm="rkv",
        config={},
        resident_kv_tokens=32,
        physical_block_ids=None,
        phase=phase,
        num_decoded_tokens=0,
        num_new_tokens=8,
    )

    assert state.worker_row == -1
    assert state.phase == phase
