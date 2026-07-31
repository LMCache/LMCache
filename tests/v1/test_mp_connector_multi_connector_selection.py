# SPDX-License-Identifier: Apache-2.0
"""Tests that the MP connector honours vLLM's ``num_external_tokens``.

See the comment on ``update_state_after_alloc`` in ``lmcache_mp_connector.py``
for why a losing ``MultiConnector`` sub-connector must not retrieve. These
tests drive that method directly with the scheduler adapter stubbed out.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)

TOKENS_PER_CHUNK = 256


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(self, request_id: str, num_tokens: int):
        self.request_id = request_id
        self.cache_salt = ""
        self.prompt_token_ids = list(range(num_tokens))
        self._live_token_ids = list(self.prompt_token_ids)
        self.all_token_ids = ConstantList(self._live_token_ids)
        self.mm_features: list[object] = []


class _FakeBlocks:
    """Duck-typed ``KVCacheBlocks`` exposing only ``get_block_ids``."""

    def __init__(self, block_ids: tuple[list[int], ...]):
        self._block_ids = block_ids

    def get_block_ids(self) -> tuple[list[int], ...]:
        return self._block_ids


def _make_connector_and_tracker(
    *,
    num_vllm_hit_tokens: int,
    num_lmcache_hit_tokens: int,
) -> tuple[LMCacheMPConnector, LMCacheMPRequestTracker, _FakeRequest]:
    """Connector/tracker pair in the post-lookup PREFETCHING state, with
    ``__init__`` bypassed since it would otherwise talk to a cache server."""
    request = _FakeRequest("req-0", num_lmcache_hit_tokens)
    tracker = LMCacheMPRequestTracker(request)
    tracker.num_vllm_hit_tokens = num_vllm_hit_tokens
    tracker.num_lmcache_hit_tokens = num_lmcache_hit_tokens
    tracker.state = LMCacheMPRequestState.PREFETCHING

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {request.request_id: tracker}
    connector.scheduler_adapter = MagicMock(name="scheduler_adapter")
    connector.scheduler_adapter.lmcache_tokens_per_chunk = TOKENS_PER_CHUNK
    return connector, tracker, request


def test_losing_connector_emits_no_retrieve():
    """``num_external_tokens == 0`` with a hit means another connector won."""
    connector, tracker, request = _make_connector_and_tracker(
        num_vllm_hit_tokens=TOKENS_PER_CHUNK,
        num_lmcache_hit_tokens=TOKENS_PER_CHUNK * 3,
    )

    connector.update_state_after_alloc(request, _FakeBlocks(([0, 1, 2],)), 0)

    assert tracker.state is LMCacheMPRequestState.READY


def test_losing_connector_releases_every_lookup_lock():
    """The cancelled retrieve would have released the tail locks; do it here."""
    connector, tracker, request = _make_connector_and_tracker(
        num_vllm_hit_tokens=TOKENS_PER_CHUNK,
        num_lmcache_hit_tokens=TOKENS_PER_CHUNK * 3,
    )

    connector.update_state_after_alloc(request, _FakeBlocks(([0, 1, 2],)), 0)

    kwargs = connector.scheduler_adapter.free_lookup_locks.call_args.kwargs
    assert kwargs["start"] == 0
    assert kwargs["end"] == TOKENS_PER_CHUNK * 3


def test_winning_connector_still_retrieves():
    """A non-zero count means this connector was selected."""
    connector, tracker, request = _make_connector_and_tracker(
        num_vllm_hit_tokens=TOKENS_PER_CHUNK,
        num_lmcache_hit_tokens=TOKENS_PER_CHUNK * 3,
    )

    connector.update_state_after_alloc(
        request, _FakeBlocks(([0, 1, 2],)), TOKENS_PER_CHUNK * 2
    )

    assert tracker.state is LMCacheMPRequestState.WAITING_FOR_LOAD
    kwargs = connector.scheduler_adapter.free_lookup_locks.call_args.kwargs
    assert kwargs["end"] == TOKENS_PER_CHUNK


def test_zero_tokens_without_a_hit_is_unchanged():
    """Single-connector deployments pass 0 whenever the lookup missed."""
    connector, tracker, request = _make_connector_and_tracker(
        num_vllm_hit_tokens=0,
        num_lmcache_hit_tokens=0,
    )

    connector.update_state_after_alloc(request, _FakeBlocks(([0],)), 0)

    assert tracker.state is LMCacheMPRequestState.READY
    connector.scheduler_adapter.free_lookup_locks.assert_not_called()
