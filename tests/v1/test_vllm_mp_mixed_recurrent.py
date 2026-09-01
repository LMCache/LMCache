# SPDX-License-Identifier: Apache-2.0
"""Fail-closed mixed-prefix tests for the vLLM MP connector."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module load")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    LMCacheMPRequestState,
    _has_recurrent_cache,
)


class _Request:
    """Duck-typed text request carrying the fields the connector reads."""

    def __init__(self, request_id: str, num_tokens: int) -> None:
        self.request_id = request_id
        self.cache_salt = ""
        self.prompt_token_ids = list(range(num_tokens))
        self.all_token_ids = ConstantList(self.prompt_token_ids)
        self.mm_features: list[object] = []
        self.status = object()
        self.kv_transfer_params = None


class _SchedulerAdapter:
    """Minimal scheduler adapter with observable lookup lifecycle calls."""

    lmcache_tokens_per_chunk = 3072

    def __init__(self, lookup_tokens: int) -> None:
        self.lookup_tokens = lookup_tokens
        self.submit_count = 0
        self.check_count = 0
        self.cleaned_request_ids: list[str] = []
        self.freed_ranges: list[tuple[int, int, str]] = []
        self.ended_request_ids: list[str] = []

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str,
    ) -> None:
        del request_id, token_ids, cache_salt
        self.submit_count += 1

    def check_lookup_result(self, request_id: str) -> int:
        del request_id
        self.check_count += 1
        return self.lookup_tokens

    def cleanup_lookup_result(self, request_id: str) -> None:
        self.cleaned_request_ids.append(request_id)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str,
    ) -> None:
        del token_ids, cache_salt
        self.freed_ranges.append((start, end, request_id))

    def end_session(self, request_id: str) -> None:
        self.ended_request_ids.append(request_id)


class _NoBlocks:
    """Empty allocation returned when the external tail is recomputed."""

    @staticmethod
    def get_block_ids() -> tuple[list[int], ...]:
        return ()


class MambaSpec:
    """Compatibility stand-in detected by the helper's public class name."""


class AttentionSpec:
    """Ordinary attention control for recurrent-cache detection."""


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, False),
        (SimpleNamespace(has_mamba_layers=True, kv_cache_groups=[]), True),
        (
            SimpleNamespace(
                has_mamba_layers=False,
                kv_cache_groups=[SimpleNamespace(kv_cache_spec=MambaSpec())],
            ),
            True,
        ),
        (
            SimpleNamespace(
                has_mamba_layers=False,
                kv_cache_groups=[SimpleNamespace(kv_cache_spec=AttentionSpec())],
            ),
            False,
        ),
    ],
)
def test_recurrent_cache_detection_supports_old_and_new_vllm_configs(
    config: object,
    expected: bool,
) -> None:
    assert _has_recurrent_cache(config) is expected  # type: ignore[arg-type]


def _scheduler_connector(
    *, has_recurrent_cache: bool, lookup_tokens: int
) -> LMCacheMPConnector:
    connector: Any = object.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector._has_recurrent_cache = has_recurrent_cache
    connector._hit_alignment_tokens = 3072
    connector.scheduler_adapter = _SchedulerAdapter(lookup_tokens)
    connector.lazy_offload = False
    return connector


def test_recurrent_mixed_local_and_external_prefix_recomputes_tail() -> None:
    """A deeper external tail must not be spliced onto local recurrent state."""
    connector = _scheduler_connector(
        has_recurrent_cache=True,
        lookup_tokens=46080,
    )
    request = _Request("mixed-recurrent", 47963)

    matched = connector.get_num_new_matched_tokens(
        request,
        num_computed_tokens=36864,
    )

    assert matched == (0, False)
    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_vllm_hit_tokens == 36864
    assert tracker.num_lmcache_hit_tokens == 46080
    assert tracker.num_stored_tokens == 46080

    connector.update_state_after_alloc(request, _NoBlocks(), 0)

    assert tracker.state == LMCacheMPRequestState.READY
    adapter = connector.scheduler_adapter
    assert isinstance(adapter, _SchedulerAdapter)
    assert adapter.cleaned_request_ids == [request.request_id]
    assert adapter.freed_ranges == [(0, 46080, request.request_id)]

    connector.request_finished(request, [])
    assert adapter.cleaned_request_ids == [request.request_id]
    assert adapter.freed_ranges == [(0, 46080, request.request_id)]


def test_suppressed_retrieve_is_idempotent_under_repeated_polling() -> None:
    """Repeated scheduler polling must not re-lookup or double-count a hit."""
    connector = _scheduler_connector(
        has_recurrent_cache=True,
        lookup_tokens=46080,
    )
    request = _Request("repeated-mixed-recurrent", 47963)

    assert connector.get_num_new_matched_tokens(request, 36864) == (0, False)
    assert connector.get_num_new_matched_tokens(request, 36864) == (0, False)

    tracker = connector.request_trackers[request.request_id]
    adapter = connector.scheduler_adapter
    assert isinstance(adapter, _SchedulerAdapter)
    assert tracker.num_stored_tokens == 46080
    assert adapter.submit_count == 1
    assert adapter.check_count == 1


def test_cancelled_suppressed_retrieve_releases_lookup_locks() -> None:
    """Cancellation before allocation must not strand the retained locks."""
    connector = _scheduler_connector(
        has_recurrent_cache=True,
        lookup_tokens=46080,
    )
    request = _Request("cancelled-mixed-recurrent", 47963)

    assert connector.get_num_new_matched_tokens(request, 36864) == (0, False)
    connector.request_finished(request, [])

    adapter = connector.scheduler_adapter
    assert isinstance(adapter, _SchedulerAdapter)
    assert adapter.cleaned_request_ids == [request.request_id]
    assert adapter.freed_ranges == [(0, 46080, request.request_id)]
    assert adapter.ended_request_ids == [request.request_id]
    assert request.request_id not in connector.request_trackers


@pytest.mark.parametrize(
    ("has_recurrent_cache", "num_computed_tokens", "expected"),
    [
        (True, 0, (46080, True)),
        (False, 36864, (9216, True)),
        (True, 46080, (0, False)),
    ],
)
def test_mixed_recurrent_guard_preserves_qualified_retrievals(
    has_recurrent_cache: bool,
    num_computed_tokens: int,
    expected: tuple[int, bool],
) -> None:
    """Full external, non-recurrent, and non-deeper hits retain behavior."""
    connector = _scheduler_connector(
        has_recurrent_cache=has_recurrent_cache,
        lookup_tokens=46080,
    )
    request = _Request("qualified-retrieve", 47963)

    matched = connector.get_num_new_matched_tokens(
        request,
        num_computed_tokens=num_computed_tokens,
    )

    assert matched == expected
    tracker = connector.request_trackers[request.request_id]
    assert tracker.needs_retrieve() is (expected[0] > 0)
