# SPDX-License-Identifier: Apache-2.0
"""Regression tests for retry-safe MP connector lookup accounting."""

# Standard
from types import SimpleNamespace

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
)

CHUNK_SIZE = 256
TOKENS_PER_BLOCK = 16


class _LookupAdapter:
    """Test adapter exposing the scheduler lookup contract."""

    lmcache_tokens_per_chunk = CHUNK_SIZE

    def __init__(self, hit_tokens: int) -> None:
        self.hit_tokens = hit_tokens
        self.cleanup_calls = 0

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int], cache_salt: str = ""
    ) -> None:
        """Accept a lookup submission without contacting a cache server."""

    def check_lookup_result(self, request_id: str) -> int:
        """Return the cached lookup result for every scheduler retry."""
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
    ) -> None:
        """Accept lock releases without contacting a cache server."""


def _make_connector(hit_tokens: int) -> tuple[LMCacheMPConnector, _LookupAdapter]:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    adapter = _LookupAdapter(hit_tokens)
    connector.scheduler_adapter = adapter  # type: ignore[assignment]
    connector._hit_alignment_tokens = TOKENS_PER_BLOCK
    return connector, adapter


def _make_request(num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-0",
        cache_salt=None,
        all_token_ids=list(range(num_tokens)),
        status=RequestStatus.WAITING,
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
