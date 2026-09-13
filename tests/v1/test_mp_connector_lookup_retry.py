# SPDX-License-Identifier: Apache-2.0
"""Regression tests for retrying MP connector prefix lookups."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm import (  # noqa: E402
    lmcache_mp_connector as connector_module,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)

CHUNK_SIZE = 256
TOKENS_PER_BLOCK = 16


class _FakeLookupAdapter:
    """Return one cached lookup result until allocation commits it."""

    lmcache_tokens_per_chunk = CHUNK_SIZE

    def __init__(self, hit_tokens: int) -> None:
        self.hit_tokens = hit_tokens
        self.cleanup_calls = 0
        self.freed_ranges: list[tuple[int, int]] = []

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
    ) -> None:
        return None

    def check_lookup_result(self, request_id: str) -> int | None:
        return self.hit_tokens

    def cleanup_lookup_result(self, request_id: str) -> None:
        self.cleanup_calls += 1

    def report_block_allocations(self, records: list[object]) -> None:
        return None

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> None:
        self.freed_ranges.append((start, end))


class _FakeKVTransferConfig:
    """Provide connector settings through vLLM's config interface."""

    def __init__(self) -> None:
        self.kv_connector_extra_config: dict[str, object] = {}

    def get_from_extra_config(self, key: str, default: object) -> object:
        return default


class _FakeRequest:
    """Provide the request fields used by the scheduler-side connector."""

    def __init__(self, num_tokens: int) -> None:
        self.request_id = "req-0"
        self.cache_salt = None
        self.prompt_token_ids = list(range(num_tokens))
        self.all_token_ids = ConstantList(self.prompt_token_ids)
        self.mm_features: list[object] = []
        self.status = RequestStatus.WAITING


class _FakeBlocks:
    """Provide allocated block IDs through vLLM's public block interface."""

    def __init__(self, num_blocks: int) -> None:
        self.block_ids = (list(range(num_blocks)),)

    def get_block_ids(self) -> tuple[list[int], ...]:
        return self.block_ids


def _make_connector(
    hit_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LMCacheMPConnector, _FakeLookupAdapter]:
    """Build a scheduler connector with a fixed lookup result.

    Args:
        hit_tokens: Number of tokens the fake LMCache lookup reports.
        monkeypatch: Pytest fixture used to replace external dependencies.

    Returns:
        The connector and its fake scheduler adapter.
    """
    adapter = _FakeLookupAdapter(hit_tokens)
    monkeypatch.setattr(
        connector_module,
        "LMCacheMPSchedulerAdapter",
        lambda **kwargs: adapter,
    )
    monkeypatch.setattr(connector_module, "print_banner_once", lambda stream: None)
    config = SimpleNamespace(
        kv_transfer_config=_FakeKVTransferConfig(),
        cache_config=SimpleNamespace(
            block_size=TOKENS_PER_BLOCK,
            mamba_cache_mode="none",
        ),
        model_config=SimpleNamespace(model="test-model", use_mla=False),
        parallel_config=SimpleNamespace(
            world_size=1,
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )
    connector = LMCacheMPConnector(  # type: ignore[arg-type]
        config,
        KVConnectorRole.SCHEDULER,
    )
    return connector, adapter


def test_lookup_retry_commits_stored_tokens_once_after_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hit_tokens = 2 * CHUNK_SIZE
    request = _FakeRequest(num_tokens=1600)
    connector, adapter = _make_connector(hit_tokens, monkeypatch)

    first_result = connector.get_num_new_matched_tokens(request, 0)
    retry_result = connector.get_num_new_matched_tokens(request, 0)

    assert first_result == (hit_tokens, True)
    assert retry_result == first_result

    connector.update_state_after_alloc(
        request,
        _FakeBlocks(num_blocks=hit_tokens // TOKENS_PER_BLOCK),  # type: ignore[arg-type]
        hit_tokens,
    )
    assert adapter.cleanup_calls == 1

    initial_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={request.request_id: 0},
        preempted_req_ids=[],
    )
    initial_metadata = connector.build_connector_meta(  # type: ignore[arg-type]
        initial_output
    )
    retrieve_metadata = next(
        item for item in initial_metadata.requests if item.direction == "RETRIEVE"
    )

    assert retrieve_metadata.op.start == 0
    assert retrieve_metadata.op.end == hit_tokens

    connector.update_state_after_alloc(
        request,
        _FakeBlocks(num_blocks=100),  # type: ignore[arg-type]
        hit_tokens,
    )

    assert adapter.cleanup_calls == 1

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id)],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={request.request_id: 1600 - hit_tokens},
        preempted_req_ids=[],
    )
    metadata = connector.build_connector_meta(scheduler_output)  # type: ignore[arg-type]
    store_metadata = next(
        item for item in metadata.requests if item.direction == "STORE"
    )

    assert store_metadata.op.start == hit_tokens
    assert store_metadata.op.end == 6 * CHUNK_SIZE


def test_zero_lookup_hit_stores_from_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _FakeRequest(num_tokens=1600)
    connector, adapter = _make_connector(0, monkeypatch)

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(
        request,
        _FakeBlocks(num_blocks=100),  # type: ignore[arg-type]
        0,
    )

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id)],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={request.request_id: 1600},
        preempted_req_ids=[],
    )
    metadata = connector.build_connector_meta(scheduler_output)  # type: ignore[arg-type]
    store_metadata = next(
        item for item in metadata.requests if item.direction == "STORE"
    )

    assert store_metadata.op.start == 0
    assert store_metadata.op.end == 6 * CHUNK_SIZE
    assert adapter.freed_ranges == []


def test_local_hit_beyond_lookup_hit_preserves_remote_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hit_tokens = 2 * CHUNK_SIZE
    local_hit_tokens = 704
    request = _FakeRequest(num_tokens=1600)
    connector, adapter = _make_connector(hit_tokens, monkeypatch)

    assert connector.get_num_new_matched_tokens(request, local_hit_tokens) == (
        0,
        False,
    )
    connector.update_state_after_alloc(
        request,
        _FakeBlocks(num_blocks=100),  # type: ignore[arg-type]
        0,
    )

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id)],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={request.request_id: 1600 - local_hit_tokens},
        preempted_req_ids=[],
    )
    metadata = connector.build_connector_meta(scheduler_output)  # type: ignore[arg-type]
    store_metadata = next(
        item for item in metadata.requests if item.direction == "STORE"
    )

    assert all(item.direction != "RETRIEVE" for item in metadata.requests)
    assert store_metadata.op.start == hit_tokens
    assert store_metadata.op.end == 6 * CHUNK_SIZE
    assert adapter.freed_ranges == [(0, hit_tokens)]
