# SPDX-License-Identifier: Apache-2.0
"""Regression tests for retry-safe MP connector lookup accounting."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module scope")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.lmcache_mp_metrics import (  # noqa: E402
    LMCacheMPConnectorStats,
)

CHUNK_SIZE = 256
TOKENS_PER_BLOCK = 16


class _LookupAdapter:
    """Test adapter exposing the scheduler lookup contract."""

    lmcache_tokens_per_chunk = CHUNK_SIZE

    def __init__(self, hit_tokens: int | None) -> None:
        self.hit_tokens = hit_tokens
        self.cleanup_calls = 0

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        """Accept a lookup submission without contacting a cache server."""

    def check_lookup_result(self, request_id: str) -> int | None:
        """Return the current lookup result, including health-driven changes."""
        return self.hit_tokens

    def cleanup_lookup_result(self, request_id: str) -> None:
        """Record that allocation committed and cleared the lookup result."""
        self.cleanup_calls += 1

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        """Accept lock releases without contacting a cache server."""


class _LookupConnector(LMCacheMPConnector):
    """Use production scheduler callbacks with an in-memory lookup adapter."""

    def __init__(self, adapter: _LookupAdapter) -> None:
        self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}
        self.scheduler_adapter = adapter  # type: ignore[assignment]
        self._hit_alignment_tokens = TOKENS_PER_BLOCK
        self._connector_stats = LMCacheMPConnectorStats()


def _make_connector(
    hit_tokens: int | None,
) -> tuple[LMCacheMPConnector, _LookupAdapter]:
    adapter = _LookupAdapter(hit_tokens)
    return _LookupConnector(adapter), adapter


def _make_request(num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-0",
        cache_salt=None,
        all_token_ids=list(range(num_tokens)),
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
    )


def _make_blocks(num_blocks: int) -> SimpleNamespace:
    return SimpleNamespace(get_block_ids=lambda: ([*range(num_blocks)],))


def test_lookup_retry_accounts_for_stored_watermark_once() -> None:
    """Repeated lookup polling must not compound the stored watermark."""
    hit_tokens = 2 * CHUNK_SIZE
    connector, adapter = _make_connector(hit_tokens)
    request = _make_request(num_tokens=1600)

    assert connector.get_num_new_matched_tokens(request, 0) == (hit_tokens, True)
    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_stored_tokens == hit_tokens

    assert connector.get_num_new_matched_tokens(request, 0) == (hit_tokens, True)
    assert tracker.num_stored_tokens == hit_tokens

    connector.update_state_after_alloc(
        request, _make_blocks(num_blocks=100), hit_tokens
    )
    assert tracker.num_stored_tokens == hit_tokens

    connector.update_state_after_alloc(
        request, _make_blocks(num_blocks=100), hit_tokens
    )
    assert tracker.num_stored_tokens == hit_tokens
    assert adapter.cleanup_calls == 1


def test_store_after_lookup_retry_starts_at_true_watermark() -> None:
    """Store metadata after a retry must not leave a hole after the hit."""
    hit_tokens = 2 * CHUNK_SIZE
    connector, _ = _make_connector(hit_tokens)
    request = _make_request(num_tokens=1600)

    connector.get_num_new_matched_tokens(request, 0)
    connector.get_num_new_matched_tokens(request, 0)
    connector.update_state_after_alloc(
        request, _make_blocks(num_blocks=100), hit_tokens
    )

    tracker = connector.request_trackers[request.request_id]
    tracker.increase_num_scheduled_tokens(1600 - hit_tokens)
    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK_SIZE, [TOKENS_PER_BLOCK]
    )

    assert metadata is not None
    assert metadata.op.start == hit_tokens


@pytest.mark.parametrize("num_computed_tokens", [0, 3 * CHUNK_SIZE])
@pytest.mark.parametrize(
    "lookup_results",
    [
        [2 * CHUNK_SIZE, 0],
        [2 * CHUNK_SIZE, CHUNK_SIZE],
        [CHUNK_SIZE, 2 * CHUNK_SIZE],
        [0, 2 * CHUNK_SIZE],
        [None, 2 * CHUNK_SIZE],
        [2 * CHUNK_SIZE, None, 0],
    ],
    ids=[
        "hit-to-zero",
        "smaller-hit",
        "larger-hit",
        "miss-to-hit",
        "pending",
        "hit-pending-zero",
    ],
)
def test_lookup_changes_before_allocation_do_not_leave_store_holes(
    lookup_results: list[int | None], num_computed_tokens: int
) -> None:
    """Store the locally computed suffix from the latest completed lookup."""
    connector, adapter = _make_connector(lookup_results[0])
    request = _make_request(num_tokens=1600)
    request.num_computed_tokens = num_computed_tokens
    latest_hit = 0

    for result in lookup_results:
        adapter.hit_tokens = result
        matched = connector.get_num_new_matched_tokens(request, num_computed_tokens)
        if result is None:
            assert matched == (None, True)
        else:
            latest_hit = result
            external_tokens = max(0, result - num_computed_tokens)
            assert matched == (external_tokens, external_tokens > 0)
        tracker = connector.request_trackers[request.request_id]
        assert tracker.num_stored_tokens == latest_hit
        assert tracker.num_lmcache_hit_tokens == latest_hit

    external_tokens = max(0, latest_hit - num_computed_tokens)
    connector.update_state_after_alloc(
        request, _make_blocks(num_blocks=100), external_tokens
    )
    connector.update_state_after_alloc(
        request, _make_blocks(num_blocks=100), external_tokens
    )
    assert adapter.cleanup_calls == 1
    assert tracker.num_stored_tokens == latest_hit
    assert tracker.num_vllm_hit_tokens == num_computed_tokens
    retrieve_metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK_SIZE, [TOKENS_PER_BLOCK]
    )
    if external_tokens == 0:
        assert not tracker.needs_retrieve()
        assert retrieve_metadata is None
    else:
        assert retrieve_metadata is not None
        assert retrieve_metadata.direction == "RETRIEVE"
        assert retrieve_metadata.op.start == num_computed_tokens
        assert retrieve_metadata.op.end == latest_hit

    tracker.increase_num_scheduled_tokens(
        len(request.all_token_ids) - max(latest_hit, num_computed_tokens)
    )
    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK_SIZE, [TOKENS_PER_BLOCK]
    )

    assert metadata is not None
    assert metadata.direction == "STORE"
    assert metadata.op.start == latest_hit
    assert metadata.op.end == 6 * CHUNK_SIZE
    assert tracker.num_stored_tokens == metadata.op.end


def test_lookup_accounting_preserves_separately_stored_tokens() -> None:
    """Replacing a lookup contribution must preserve separate store accounting."""
    tracker = LMCacheMPRequestTracker(_make_request(num_tokens=1600))
    tracker.account_lookup_result(2 * CHUNK_SIZE)
    tracker.increase_num_stored_tokens(CHUNK_SIZE)
    tracker.account_lookup_result(2 * CHUNK_SIZE)
    assert tracker.num_stored_tokens == 3 * CHUNK_SIZE

    tracker.account_lookup_result(0)
    assert tracker.num_stored_tokens == CHUNK_SIZE
    tracker.account_lookup_result(0)
    assert tracker.num_stored_tokens == CHUNK_SIZE
