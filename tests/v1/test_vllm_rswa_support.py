# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for prompt-only R-SWA caching in LMCache MP."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_connector_v1 import (  # noqa: E402
    LMCacheConnectorV1Dynamic,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    get_cache_model_name,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.lmcache_mp_metrics import (  # noqa: E402
    LMCacheMPConnectorStats,
)
from lmcache.integration.vllm.utils import is_rswa_model  # noqa: E402
from lmcache.integration.vllm.vllm_v1_adapter import (  # noqa: E402
    LMCacheConnectorV1Impl,
)


class RSWASpec:
    """Duck-typed stand-in for vLLM releases that do not expose RSWASpec."""


class FullAttentionSpec:
    """Non-R-SWA attention-spec stand-in."""


class UniformTypeKVCacheSpecs:
    """Duck-typed stand-in for vLLM's per-layer spec wrapper."""

    def __init__(self, specs: dict[str, object]) -> None:
        self.kv_cache_specs = specs


class _FakeRequest:
    """Small public-contract double for the fields read by the MP connector."""

    def __init__(
        self,
        prompt_tokens: int,
        total_tokens: int,
        *,
        resumable: bool = False,
    ) -> None:
        self.request_id = "req-rswa"
        self.cache_salt = ""
        self.prompt_token_ids = list(range(prompt_tokens))
        self.num_prompt_tokens = prompt_tokens
        self._token_ids = list(range(total_tokens))
        self.all_token_ids = ConstantList(self._token_ids)
        self.mm_features: list[object] = []
        self.sampling_params = SimpleNamespace(extra_args=None)
        self.resumable = resumable
        self.status = RequestStatus.WAITING
        self.num_computed_tokens = 0


def _vllm_config(
    *,
    rswa_window: int | None = None,
    dcp_size: int = 1,
    interleave: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="baidu/Unlimited-OCR",
            rswa_window=rswa_window,
        ),
        kv_transfer_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave,
        ),
    )


def _kv_cache_config(spec: object) -> SimpleNamespace:
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )


def _lookup_connector(
    adapter: MagicMock,
    *,
    eager_prefetch: bool = False,
) -> LMCacheMPConnector:
    """Build callback state without starting network or device resources."""
    # These public-callback tests bypass the networked constructor, so seed
    # its scheduler state explicitly, including the real per-instance stats.
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    KVConnectorBase_V1.__init__(
        connector,
        _vllm_config(rswa_window=128),  # type: ignore[arg-type]
        KVConnectorRole.SCHEDULER,
        _kv_cache_config(RSWASpec()),  # type: ignore[arg-type]
    )
    connector._prompt_only_cache = True
    connector._hit_alignment_tokens = 16
    connector._eager_prefetch = eager_prefetch
    connector._connector_stats = LMCacheMPConnectorStats()
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter
    return connector


@pytest.mark.parametrize(
    "spec",
    [
        RSWASpec(),
        UniformTypeKVCacheSpecs({"model.layers.0.attn": RSWASpec()}),
        UniformTypeKVCacheSpecs(
            {
                "model.layers.0.attn": FullAttentionSpec(),
                "model.layers.1.attn": RSWASpec(),
            }
        ),
    ],
)
def test_rswa_detection_reads_resolved_leaf_specs(spec: object) -> None:
    """Direct and UniformType-wrapped RSWASpec leaves are both detected."""
    assert is_rswa_model(_vllm_config(), _kv_cache_config(spec))


def test_rswa_detection_falls_back_to_model_config() -> None:
    """The model flag survives connector-side KV-spec merging or omission."""
    assert is_rswa_model(_vllm_config(rswa_window=128))
    assert not is_rswa_model(
        _vllm_config(),
        _kv_cache_config(FullAttentionSpec()),
    )


def test_rswa_detection_reads_nested_hf_config() -> None:
    """Older ModelConfig shapes can expose the marker only on HF config."""
    config = _vllm_config()
    del config.model_config.rswa_window
    config.model_config.hf_config = SimpleNamespace(
        text_config=SimpleNamespace(rswa_window=128)
    )

    assert is_rswa_model(config)


def test_rswa_prompt_cache_has_an_isolated_namespace() -> None:
    """Prompt-only R-SWA entries cannot hit legacy decode-cache objects."""
    config = _vllm_config(rswa_window=128, dcp_size=4, interleave=4)

    assert get_cache_model_name(config) == (
        "baidu/Unlimited-OCR"
        "##lmcache-dcp-layout-v1-d4-interleave4"
        "##lmcache-rswa-prompt-v1"
    )
    worker_kv_config = _kv_cache_config(
        UniformTypeKVCacheSpecs(
            {
                "model.layers.0.attn": FullAttentionSpec(),
                "model.layers.1.attn": RSWASpec(),
            }
        )
    )
    assert get_cache_model_name(config, worker_kv_config) == get_cache_model_name(
        config
    )
    assert get_cache_model_name(_vllm_config()) == "baidu/Unlimited-OCR"


def test_rswa_resolved_spec_only_gets_prompt_cache_namespace() -> None:
    """Resolved R-SWA groups namespace the cache without a model marker."""
    config = _vllm_config(dcp_size=4, interleave=4)
    kv_cache_config = _kv_cache_config(RSWASpec())

    assert get_cache_model_name(config, kv_cache_config) == (
        "baidu/Unlimited-OCR"
        "##lmcache-dcp-layout-v1-d4-interleave4"
        "##lmcache-rswa-prompt-v1"
    )


def test_rswa_mp_rejects_renderer_local_multimodal_ids() -> None:
    """R-SWA never starts with restart-unsafe multimodal cache keys."""
    config = _vllm_config(rswa_window=128)
    config.model_config.multimodal_config = SimpleNamespace(
        mm_processor_cache_gb=0,
    )
    config.cache_config = SimpleNamespace(enable_prefix_caching=False)

    with pytest.raises(ValueError, match="stable multimodal identifiers"):
        LMCacheMPConnector(
            config,  # type: ignore[arg-type]
            role=KVConnectorRole.SCHEDULER,
        )


def test_non_mp_connector_fails_fast_for_rswa() -> None:
    """The dynamic non-MP path forwards resolved R-SWA groups to its guard."""
    with pytest.raises(ValueError, match="LMCache-shipped LMCacheMPConnector"):
        LMCacheConnectorV1Dynamic(
            _vllm_config(),  # type: ignore[arg-type]
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=_kv_cache_config(RSWASpec()),
        )


def test_non_mp_adapter_fails_fast_for_vllm_builtin_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM's default non-native path reaches the in-process safety guard."""
    banner = MagicMock()
    load_config = MagicMock()
    monkeypatch.setattr(
        "lmcache.integration.vllm.vllm_v1_adapter.print_banner_once",
        banner,
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.vllm_v1_adapter.lmcache_get_or_create_config",
        load_config,
    )

    with pytest.raises(ValueError, match="LMCache-shipped LMCacheMPConnector"):
        LMCacheConnectorV1Impl(
            _vllm_config(rswa_window=128),  # type: ignore[arg-type]
            role=KVConnectorRole.SCHEDULER,
            parent=MagicMock(),
        )
    banner.assert_not_called()
    load_config.assert_not_called()


def test_prompt_only_tracker_excludes_decode_tokens() -> None:
    """R-SWA key derivation remains fixed at the initial prompt boundary."""
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)

    tracker = LMCacheMPRequestTracker(request, prompt_only=True)

    assert tracker.num_cache_tokens == 256
    assert tracker.get_token_ids() == list(range(512))
    assert tracker.get_cache_token_ids() == list(range(256))


@pytest.mark.parametrize("max_offload_tokens", [None, 1024])
def test_resumable_rswa_request_fails_closed(
    max_offload_tokens: int | None,
) -> None:
    """A mutable streaming prompt is not assigned a stale cache boundary."""
    request = _FakeRequest(prompt_tokens=256, total_tokens=256, resumable=True)
    request.sampling_params.extra_args = {
        "kv_transfer_params": {"lmcache.max_offload_tokens": max_offload_tokens}
    }
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)

    assert tracker.num_cache_tokens == 0
    assert tracker.get_cache_token_ids() == []


@pytest.mark.parametrize("resumable", [False, True])
def test_uncacheable_rswa_lookup_does_not_start_timer_or_report_metrics(
    resumable: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither prefetch nor polling times a lookup that was never submitted."""
    request = _FakeRequest(
        prompt_tokens=256 if resumable else 0,
        total_tokens=512,
        resumable=resumable,
    )
    clock = MagicMock()
    monkeypatch.setattr(
        "lmcache.integration.vllm.lmcache_mp_connector.time",
        SimpleNamespace(monotonic=clock),
    )
    adapter = MagicMock()
    connector = _lookup_connector(adapter, eager_prefetch=True)

    connector.on_new_request(request)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.get_kv_connector_stats() is None
    assert connector.request_trackers[request.request_id].lookup_started_at is None
    adapter.maybe_submit_lookup_request.assert_not_called()
    adapter.check_lookup_result.assert_not_called()
    clock.assert_not_called()


@pytest.mark.parametrize("eager_prefetch", [False, True])
def test_rswa_lookup_latency_spans_pending_polls(
    eager_prefetch: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt-only lookup retains one latency interval through pending replies."""
    now = [10.0]
    monkeypatch.setattr(
        "lmcache.integration.vllm.lmcache_mp_connector.time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = 256
    adapter.check_lookup_result.side_effect = [None, 256]
    connector = _lookup_connector(adapter, eager_prefetch=eager_prefetch)

    if eager_prefetch:
        connector.on_new_request(request)
        now[0] = 11.0
    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    assert connector.get_kv_connector_stats() is None
    assert connector.request_trackers[request.request_id].lookup_started_at == 10.0

    now[0] = 12.0
    assert connector.get_num_new_matched_tokens(request, 0) == (255, True)
    stats = connector.get_kv_connector_stats()
    assert stats is not None
    assert stats.reduce() == {
        "LMCache MP lookup count": 1,
        "LMCache MP lookup avg latency (ms)": 2000.0,
    }
    assert connector.get_kv_connector_stats() is None
    assert connector.request_trackers[request.request_id].lookup_started_at is None
    for call in adapter.maybe_submit_lookup_request.call_args_list:
        assert call.kwargs["token_ids"] == list(range(256))


@pytest.mark.parametrize("preempted", [False, True])
def test_uncacheable_rswa_request_does_not_submit_lookup(preempted: bool) -> None:
    """Mutable prompts and preempted requests do not issue remote lookups."""
    request = _FakeRequest(
        prompt_tokens=256,
        total_tokens=512,
        resumable=not preempted,
    )
    if preempted:
        request.status = RequestStatus.PREEMPTED
    adapter = MagicMock()

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._prompt_only_cache = True
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    adapter.maybe_submit_lookup_request.assert_not_called()
    adapter.check_lookup_result.assert_not_called()
    tracker = connector.request_trackers[request.request_id]
    assert tracker.get_cache_token_ids() == ([] if not preempted else list(range(256)))


def test_rswa_store_stops_before_freed_decode_gap() -> None:
    """Only complete prompt chunks are stored after decode advances.

    vLLM's RSWAManager frees logical block-table indices 16..23 in this
    geometry once decode reaches token 512. The MP tracker may still retain
    those original IDs, but prompt-only metadata must never slice them.
    """
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    tracker.allocated_block_ids = {0: list(range(1000, 1032))}
    tracker.increase_num_scheduled_tokens(256)

    prompt_store = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16],
    )
    tracker.increase_num_scheduled_tokens(256)
    decode_store = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16],
    )

    assert prompt_store is not None
    assert (prompt_store.op.start, prompt_store.op.end) == (0, 256)
    assert prompt_store.op.block_ids == [list(range(1000, 1016))]
    assert decode_store is None


def test_rswa_store_drops_partial_prompt_tail() -> None:
    """A chunk crossing the prompt/decode boundary is never committed."""
    request = _FakeRequest(prompt_tokens=300, total_tokens=512)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    tracker.allocated_block_ids = {0: list(range(1000, 1032))}
    tracker.increase_num_scheduled_tokens(512)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16],
    )
    second = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16],
    )

    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (0, 256)
    assert second is None


@pytest.mark.parametrize(
    ("max_offload_tokens", "expected_end"),
    [(None, 512), (0, 0), (255, 0), (256, 256), (300, 256), (512, 512), (768, 512)],
)
def test_rswa_store_respects_both_prompt_and_offload_limits(
    max_offload_tokens: int | None,
    expected_end: int,
) -> None:
    """A write budget can shorten prompt stores but cannot include decode KV."""
    request = _FakeRequest(prompt_tokens=512, total_tokens=1024)
    request.sampling_params.extra_args = {
        "kv_transfer_params": {"lmcache.max_offload_tokens": max_offload_tokens}
    }
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    tracker.allocated_block_ids = {0: list(range(1000, 1064))}
    tracker.increase_num_scheduled_tokens(1024)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, lmcache_tokens_per_chunk=256, group_tokens_per_block=[16]
    )

    assert tracker.num_cache_tokens == 512
    assert tracker.get_cache_token_ids() == list(range(512))
    assert tracker.num_stored_tokens == expected_end
    if expected_end:
        assert metadata is not None
        assert (metadata.op.start, metadata.op.end) == (0, expected_end)
        assert metadata.op.token_ids == list(range(512))
        assert metadata.op.block_ids == [list(range(1000, 1000 + expected_end // 16))]
    else:
        assert metadata is None
    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, lmcache_tokens_per_chunk=256, group_tokens_per_block=[16]
        )
        is None
    )


@pytest.mark.parametrize("max_offload_tokens", [0, 256])
def test_rswa_offload_limit_does_not_shorten_prompt_retrieval(
    max_offload_tokens: int,
) -> None:
    """An offload budget limits writes, not already-cached prompt reads."""
    request = _FakeRequest(prompt_tokens=512, total_tokens=1024)
    request.sampling_params.extra_args = {
        "kv_transfer_params": {"lmcache.max_offload_tokens": max_offload_tokens}
    }
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    tracker.allocated_block_ids = {0: list(range(1000, 1064))}
    tracker.num_lmcache_hit_tokens = 512
    tracker.num_stored_tokens = 512
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, lmcache_tokens_per_chunk=256, group_tokens_per_block=[16]
    )

    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (0, 512)
    assert metadata.op.token_ids == list(range(512))
    assert metadata.op.block_ids == [list(range(1000, 1032))]
    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, lmcache_tokens_per_chunk=256, group_tokens_per_block=[16]
        )
        is None
    )


def test_rswa_retrieve_uses_only_prompt_blocks() -> None:
    """A prompt hit never maps retrieval through the later sparse gap."""
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    tracker.allocated_block_ids = {0: list(range(1000, 1032))}
    tracker.num_lmcache_hit_tokens = 256
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16],
    )

    assert metadata is not None
    assert metadata.op.token_ids == list(range(256))
    assert metadata.op.block_ids == [list(range(1000, 1016))]


def test_rswa_mixed_groups_store_and_retrieve_only_prompt_blocks() -> None:
    """Prompt-only metadata composes with heterogeneous group geometry."""
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    tracker.allocated_block_ids = {
        0: list(range(1000, 1032)),
        1: list(range(2000, 2016)),
    }
    tracker.increase_num_scheduled_tokens(512)

    store = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16, 32],
    )

    assert store is not None
    assert (store.op.start, store.op.end) == (0, 256)
    assert store.op.block_ids == [
        list(range(1000, 1016)),
        list(range(2000, 2008)),
    ]

    tracker.num_lmcache_hit_tokens = 256
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    retrieve = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16, 32],
    )

    assert retrieve is not None
    assert (retrieve.op.start, retrieve.op.end) == (0, 256)
    assert retrieve.op.block_ids == store.op.block_ids
    assert retrieve.op.token_ids == list(range(256))


def test_rswa_full_prompt_hit_recomputes_last_prompt_token() -> None:
    """The full-hit adjustment compares against the prompt, not decode length."""
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = 256
    adapter.check_lookup_result.return_value = 256

    connector = _lookup_connector(adapter)

    matched_tokens, load_async = connector.get_num_new_matched_tokens(request, 0)

    assert (matched_tokens, load_async) == (255, True)
    adapter.maybe_submit_lookup_request.assert_called_once_with(
        request.request_id,
        token_ids=list(range(256)),
        cache_salt="",
        request_configs=None,
    )


def test_rswa_lock_release_uses_prompt_cache_key() -> None:
    """APC-overlapped lock release derives the same prompt-only key as lookup."""
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = 256
    adapter.check_lookup_result.return_value = 256

    connector = _lookup_connector(adapter)

    assert connector.get_num_new_matched_tokens(request, 256) == (0, False)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (list(range(1000, 1016)),)
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    adapter.free_lookup_locks.assert_called_once_with(
        token_ids=list(range(256)),
        start=0,
        end=256,
        request_id=request.request_id,
        cache_salt="",
        request_configs=None,
    )


def test_rswa_allocation_telemetry_keeps_decode_tail_tokens() -> None:
    """Prompt-only cache policy must not truncate allocation telemetry."""
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    # At token 384, the tracker has seen the 16 prompt blocks followed by
    # eight blocks that R-SWA will free as the decode window advances.
    tracker.allocated_block_ids = {0: list(range(1000, 1024))}
    tracker.num_scheduled_tokens = 384
    tracker.num_stored_tokens = 256
    tracker.state = LMCacheMPRequestState.READY

    tail_block_ids = list(range(2000, 2008))
    cached_reqs = SimpleNamespace(
        req_ids=[request.request_id],
        resumed_req_ids=set(),
        new_block_ids=[(tail_block_ids,)],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens={request.request_id: 128},
        total_num_scheduled_tokens=128,
        preempted_req_ids=set(),
    )
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = 256

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._prompt_only_cache = True
    connector._group_tokens_per_block = [16]
    connector.request_trackers = {request.request_id: tracker}
    connector.scheduler_adapter = adapter
    connector.lazy_offload = False

    metadata = connector.build_connector_meta(  # type: ignore[arg-type]
        scheduler_output
    )

    assert len(metadata) == 0
    adapter.report_block_allocations.assert_called_once()
    records = adapter.report_block_allocations.call_args.args[0]
    assert len(records) == 1
    assert records[0].req_id == request.request_id
    assert records[0].new_block_ids == tail_block_ids
    assert records[0].new_token_ids == list(range(384, 512))


@pytest.mark.parametrize("has_pending_store", [False, True])
def test_lazy_offload_finish_handles_requests_without_store_metadata(
    has_pending_store: bool,
) -> None:
    """A bypassed or sub-chunk request has no lazy item to mark finished."""
    request = _FakeRequest(prompt_tokens=128, total_tokens=128)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    pending_store = MagicMock()
    pending_store.has_pending_request.return_value = has_pending_store

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {request.request_id: tracker}
    connector.scheduler_adapter = MagicMock()
    connector.lazy_offload = True
    connector._pending_store = pending_store

    delay_free, return_params = connector.request_finished(request, [])

    assert delay_free is False
    assert return_params is None
    pending_store.has_pending_request.assert_called_once_with(request.request_id)
    if has_pending_store:
        pending_store.mark_req_finished.assert_called_once_with(request.request_id)
    else:
        pending_store.mark_req_finished.assert_not_called()
    assert request.request_id not in connector.request_trackers


def test_lazy_offload_finish_skips_session_for_bypassed_request() -> None:
    """A cap-zero request has neither a pending item nor a server session."""
    request = _FakeRequest(prompt_tokens=128, total_tokens=128, resumable=True)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    pending_store = MagicMock()
    pending_store.has_pending_request.return_value = False

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {request.request_id: tracker}
    connector.scheduler_adapter = MagicMock()
    connector.lazy_offload = True
    connector._pending_store = pending_store

    delay_free, return_params = connector.request_finished(request, [])

    assert delay_free is False
    assert return_params is None
    connector.scheduler_adapter.end_session.assert_not_called()
    connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with(
        request.request_id
    )
    pending_store.mark_req_finished.assert_not_called()


def test_uncacheable_request_uses_non_lazy_finished_handshake() -> None:
    """The MP worker must acknowledge even a no-store non-lazy request."""
    request = _FakeRequest(prompt_tokens=128, total_tokens=128, resumable=True)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {request.request_id: tracker}
    connector.scheduler_adapter = MagicMock()
    connector.lazy_offload = False

    delay_free, return_params = connector.request_finished(request, [])

    assert delay_free is True
    assert return_params is None
    connector.scheduler_adapter.end_session.assert_not_called()
    connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with(
        request.request_id
    )
    assert request.request_id not in connector.request_trackers


@pytest.mark.parametrize(
    ("lazy_offload", "expected_delay_free"),
    [(False, True), (True, False)],
)
def test_request_finished_tolerates_early_abort_without_tracker(
    lazy_offload: bool,
    expected_delay_free: bool,
) -> None:
    """An abort before lookup/tracker creation still completes cleanly."""
    request = _FakeRequest(prompt_tokens=128, total_tokens=128)
    pending_store = MagicMock()
    pending_store.has_pending_request.return_value = False

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = MagicMock()
    connector.lazy_offload = lazy_offload
    connector._pending_store = pending_store

    delay_free, return_params = connector.request_finished(request, [])

    assert delay_free is expected_delay_free
    assert return_params is None
    connector.scheduler_adapter.end_session.assert_called_once_with(request.request_id)
    connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with(
        request.request_id
    )
    if lazy_offload:
        pending_store.has_pending_request.assert_called_once_with(request.request_id)
    else:
        pending_store.has_pending_request.assert_not_called()
    pending_store.mark_req_finished.assert_not_called()


def test_uncacheable_request_skips_allocation_telemetry() -> None:
    """A full cache bypass does not create L0 ownership to tear down."""
    request = _FakeRequest(prompt_tokens=128, total_tokens=128, resumable=True)
    tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    tracker.allocated_block_ids = {0: list(range(8))}
    adapter = MagicMock()

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._group_tokens_per_block = [16]
    connector.request_trackers = {request.request_id: tracker}
    connector.scheduler_adapter = adapter
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id)],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
        ),
    )

    connector._report_block_allocation_deltas(scheduler_output)

    adapter.report_block_allocations.assert_not_called()


def test_rswa_preemption_replaces_stale_tracker_and_stays_prompt_only() -> None:
    """A resumed allocation cannot revive stale decode-gap block IDs."""
    request = _FakeRequest(prompt_tokens=256, total_tokens=512)
    stale_tracker = LMCacheMPRequestTracker(request, prompt_only=True)
    stale_tracker.allocated_block_ids = {0: list(range(1000, 1032))}
    stale_tracker.num_scheduled_tokens = 512
    stale_tracker.num_stored_tokens = 256
    stale_tracker.state = LMCacheMPRequestState.READY
    request.status = RequestStatus.PREEMPTED

    adapter = MagicMock()
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._prompt_only_cache = True
    connector.request_trackers = {request.request_id: stale_tracker}
    connector.scheduler_adapter = adapter

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    fresh_tracker = connector.request_trackers[request.request_id]
    assert fresh_tracker is not stale_tracker
    assert fresh_tracker.state == LMCacheMPRequestState.PREFETCHING
    assert fresh_tracker.allocated_block_ids == {}
    assert fresh_tracker.get_token_ids() == list(range(512))
    assert fresh_tracker.get_cache_token_ids() == list(range(256))

    # Repeated scheduler polls keep the fresh tracker instead of recreating it.
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.request_trackers[request.request_id] is fresh_tracker
    adapter.maybe_submit_lookup_request.assert_not_called()
    adapter.check_lookup_result.assert_not_called()

    # A resumed R-SWA table contains nulls for the freed middle gap. Even after
    # accepting that full replacement table, only the prompt blocks may store.
    replacement_block_ids = [
        *range(2000, 2016),
        *([-1] * 8),
        *range(3000, 3008),
    ]
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (replacement_block_ids,)
    request.status = RequestStatus.WAITING
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)
    fresh_tracker.increase_num_scheduled_tokens(512)

    prompt_store = LMCacheMPRequestMetadata.GetStoreMetadata(
        fresh_tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16],
    )
    decode_store = LMCacheMPRequestMetadata.GetStoreMetadata(
        fresh_tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[16],
    )

    assert fresh_tracker.allocated_block_ids == {0: replacement_block_ids}
    assert prompt_store is not None
    assert (prompt_store.op.start, prompt_store.op.end) == (0, 256)
    assert prompt_store.op.block_ids == [list(range(2000, 2016))]
    assert decode_store is None


@pytest.mark.parametrize(
    ("prompt_tokens", "total_tokens", "remote_hit", "expected"),
    [
        (128, 384, 0, (0, False)),
        (300, 512, 256, (256, True)),
    ],
)
def test_rswa_lookup_respects_short_and_unaligned_prompt_boundaries(
    prompt_tokens: int,
    total_tokens: int,
    remote_hit: int,
    expected: tuple[int, bool],
) -> None:
    """Lookup exposes only the prompt and never rounds a hit into decode."""
    request = _FakeRequest(
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
    )
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = 256
    adapter.check_lookup_result.return_value = remote_hit

    connector = _lookup_connector(adapter)

    assert connector.get_num_new_matched_tokens(request, 0) == expected
    adapter.maybe_submit_lookup_request.assert_called_once_with(
        request.request_id,
        token_ids=list(range(prompt_tokens)),
        cache_salt="",
        request_configs=None,
    )
    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_stored_tokens == remote_hit
