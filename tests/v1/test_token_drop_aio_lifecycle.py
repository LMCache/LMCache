# SPDX-License-Identifier: Apache-2.0
"""Scheduler/worker token-drop metadata and output barrier contracts."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

pytest.importorskip("vllm")
# Third Party
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

# First Party
from lmcache.integration.vllm.experimental.token_drop import TokenDropSpec
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    add_token_drop_states,
    admit_token_drop_request,
    clear_resident_kv_tokens,
    commit_completed_token_drop_updates,
    get_resident_kv_tokens,
    require_private_token_drop_blocks,
    reset_preempted_token_drop_requests,
    set_resident_kv_tokens,
)
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPWorkerMetadata,
)


@pytest.fixture(autouse=True)
def _clear_frontier():
    clear_resident_kv_tokens("td")
    yield
    clear_resident_kv_tokens("td")


def _scheduler(*, logical=65, prompt=64):
    tracker = SimpleNamespace(
        token_drop_spec=TokenDropSpec("rkv", {"budget": 32}),
        num_scheduled_tokens=logical,
        num_prompt_tokens=prompt,
        token_drop_generation=1,
    )
    blocks = [
        SimpleNamespace(block_id=7, ref_cnt=1, block_hash=None),
        SimpleNamespace(block_id=9, ref_cnt=1, block_hash=None),
    ]
    scheduler = SimpleNamespace(
        request_trackers={
            "td": tracker,
            "normal": SimpleNamespace(token_drop_spec=None),
        },
        _token_drop_allocations={"td": SimpleNamespace(blocks=[blocks])},
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["normal", "td"]),
        num_scheduled_tokens={"normal": 1, "td": 1},
    )
    return scheduler, output


def _admission_connector(*, block_stride=False):
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False, async_scheduling=False
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(world_size=1, use_ubatching=False),
        use_v2_model_runner=False,
        attention_config=SimpleNamespace(backend=None),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
    )
    connector = SimpleNamespace(
        _vllm_config=cfg,
        _group_tokens_per_block=[16],
        _kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(
                        {
                            "layer.0": FullAttentionSpec(
                                block_size=16,
                                num_kv_heads=2,
                                head_size=64,
                                dtype=torch.bfloat16,
                                indexes_kv_by_block_stride=block_stride,
                            )
                        }
                    )
                )
            ]
        ),
    )
    request = SimpleNamespace(skip_reading_prefix_cache=False)
    return connector, request


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("chunked", "chunked prefill"),
        ("speculative", "speculative decoding"),
        ("async", "synchronous scheduling"),
        ("parallel", "single GPU"),
        ("groups", "one KV cache group"),
    ],
)
def test_unsupported_td_is_rejected_before_installing_adaptors(
    kind, error, monkeypatch
):
    connector, request = _admission_connector()
    cfg = connector._vllm_config
    if kind == "chunked":
        cfg.scheduler_config.enable_chunked_prefill = True
    elif kind == "speculative":
        cfg.speculative_config = object()
    elif kind == "async":
        cfg.scheduler_config.async_scheduling = True
    elif kind == "parallel":
        cfg.parallel_config.world_size = 2
    else:
        connector._group_tokens_per_block = [16, 16]

    def forbidden():
        raise AssertionError("Adaptor installed before admission validated")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    with pytest.raises(ValueError, match=error):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


@pytest.mark.parametrize("block_stride", [False, True])
def test_supported_td_admission_installs_both_adaptors(monkeypatch, block_stride):
    connector, request = _admission_connector(block_stride=block_stride)
    called = []
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        lambda: called.append("apc"),
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_allocator_adapter",
        lambda: called.append("allocator"),
    )
    admit_token_drop_request(connector, request)
    assert called == ["apc", "allocator"]
    assert request.skip_reading_prefix_cache is True


def test_disabled_normal_request_preserves_original_eager_prefetch_early_return():
    create_tracker = Mock(side_effect=AssertionError("Unexpected tracker creation"))
    connector = SimpleNamespace(
        role=KVConnectorRole.SCHEDULER,
        _token_drop_enabled=False,
        _eager_prefetch=False,
        _get_or_create_request_tracker=create_tracker,
    )
    request = SimpleNamespace(
        resumable=False,
        sampling_params=SimpleNamespace(extra_args=None),
    )
    LMCacheMPConnector.on_new_request(connector, request)
    create_tracker.assert_not_called()


def test_disabled_token_drop_opt_in_fails_before_any_hook_or_tracker():
    create_tracker = Mock(side_effect=AssertionError("Unexpected tracker creation"))
    connector = SimpleNamespace(
        role=KVConnectorRole.SCHEDULER,
        _token_drop_enabled=False,
        _eager_prefetch=False,
        _get_or_create_request_tracker=create_tracker,
    )
    request = SimpleNamespace(
        resumable=False,
        sampling_params=SimpleNamespace(
            extra_args={
                "kv_transfer_params": {
                    "lmcache.token_drop": {"algorithm": "rkv", "config": {}}
                }
            }
        ),
    )
    with pytest.raises(ValueError, match="lmcache.experimental.token_dropping"):
        LMCacheMPConnector.on_new_request(connector, request)
    create_tracker.assert_not_called()


def test_publish_private_resident_frontier_and_decode_facts():
    scheduler, output = _scheduler()
    set_resident_kv_tokens("td", 33)
    metadata = LMCacheMPConnectorMetadata()

    add_token_drop_states(scheduler, output, metadata)

    assert len(metadata.token_drop_requests) == 1
    state = metadata.token_drop_requests[0]
    assert state.request_id == "td"
    assert state.resident_kv_tokens == 33
    assert state.physical_block_ids == [7, 9]
    assert state.phase == "decode"
    assert state.num_decoded_tokens == 0
    assert state.num_new_tokens == 1
    assert len(metadata) == 1


def test_first_prefill_keeps_vanilla_physical_addressing():
    scheduler, output = _scheduler(logical=64, prompt=64)
    output.scheduled_cached_reqs.req_ids = []
    output.scheduled_new_reqs = [SimpleNamespace(req_id="td")]
    output.num_scheduled_tokens["td"] = 64
    metadata = LMCacheMPConnectorMetadata()

    add_token_drop_states(scheduler, output, metadata)

    state = metadata.token_drop_requests[0]
    assert state.physical_block_ids is None
    assert state.resident_kv_tokens == 64
    assert state.phase == "prefill"


def test_preemption_clears_only_token_drop_physical_state():
    scheduler, output = _scheduler()
    scheduler._token_drop_finished_before_output = set()
    set_resident_kv_tokens("td", 33)
    output.preempted_req_ids = {"normal", "td"}
    output.num_scheduled_tokens.pop("td")
    metadata = LMCacheMPConnectorMetadata()

    reset_preempted_token_drop_requests(scheduler, output, metadata)

    assert metadata.token_drop_reset_ids == {"td"}
    assert get_resident_kv_tokens("td") is None
    assert "td" not in scheduler._token_drop_allocations
    assert scheduler._token_drop_finished_before_output == {"td"}


def test_preempted_and_rescheduled_same_step_preserves_new_allocation():
    scheduler, output = _scheduler()
    scheduler._token_drop_finished_before_output = set()
    set_resident_kv_tokens("td", 33)
    output.preempted_req_ids = {"td"}
    new_blocks = scheduler._token_drop_allocations["td"].blocks[0]
    metadata = LMCacheMPConnectorMetadata()

    reset_preempted_token_drop_requests(scheduler, output, metadata)

    assert metadata.token_drop_reset_ids == {"td"}
    assert scheduler._token_drop_allocations["td"].blocks[0] is new_blocks
    assert get_resident_kv_tokens("td") is None
    assert scheduler._token_drop_finished_before_output == set()


def test_preempted_stale_worker_update_cannot_reclaim_blocks():
    scheduler, _ = _scheduler()
    scheduler._token_drop_finished_before_output = {"td"}
    blocks = scheduler._token_drop_allocations["td"].blocks[0]
    set_resident_kv_tokens("td", 33)
    metadata = LMCacheMPWorkerMetadata(
        completed_store_requests={}, resident_kv_updates={"td": (1, 16)}
    )

    commit_completed_token_drop_updates(scheduler, metadata)

    assert get_resident_kv_tokens("td") == 33
    assert scheduler._token_drop_allocations["td"].blocks[0] is blocks
    assert scheduler._token_drop_finished_before_output == set()


def test_worker_meta_publishes_updates_without_lazy_offload_or_kv_events():
    worker = SimpleNamespace(
        _kv_events_hook_called=False,
        get_kv_connector_kv_cache_events=lambda: None,
        lazy_offload=False,
        _pending_token_drop_updates={"td": (1, 21)},
    )

    metadata = LMCacheMPConnector.build_connector_worker_meta(worker)

    assert isinstance(metadata, LMCacheMPWorkerMetadata)
    assert metadata.resident_kv_updates == {"td": (1, 21)}
    assert worker._pending_token_drop_updates == {}
    assert LMCacheMPConnector.build_connector_worker_meta(worker) is None


def test_forward_end_collects_only_completed_compaction_reports():
    metadata = LMCacheMPConnectorMetadata()
    worker = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
        _token_drop_worker=SimpleNamespace(compact=lambda: {"td": (1, 12)}),
        _pending_token_drop_updates={},
        dispatcher=None,
    )
    LMCacheMPConnector.wait_for_save(worker)
    assert worker._pending_token_drop_updates == {"td": (1, 12)}


def test_worker_does_not_publish_partial_failure():
    metadata = LMCacheMPConnectorMetadata()

    def fail():
        raise RuntimeError("compaction failed after partial write")

    worker = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
        _token_drop_worker=SimpleNamespace(compact=fail),
        _pending_token_drop_updates={},
        dispatcher=None,
    )
    with pytest.raises(RuntimeError, match="partial write"):
        LMCacheMPConnector.wait_for_save(worker)
    assert worker._pending_token_drop_updates == {}


def test_reset_only_meta_counts_as_nonempty():
    metadata = LMCacheMPConnectorMetadata()
    assert len(metadata) == 0
    metadata.token_drop_reset_ids.add("td")
    assert len(metadata) == 1
    metadata.token_drop_finished_ids.add("finished")
    assert len(metadata) == 2


def test_stale_generation_cannot_reclaim_new_request_blocks():
    scheduler, _ = _scheduler(logical=32, prompt=24)
    scheduler._token_drop_finished_before_output = set()
    scheduler._group_tokens_per_block = [16]
    scheduler._gpu_block_pool = Mock()
    scheduler._get_request_tracker = scheduler.request_trackers.__getitem__
    blocks = scheduler._token_drop_allocations["td"].blocks[0]
    set_resident_kv_tokens("td", 32)

    # Reusing a request id must not grant its old worker completion access
    # to the new request's native allocation.
    metadata = LMCacheMPWorkerMetadata(
        completed_store_requests={}, resident_kv_updates={"td": (0, 16)}
    )
    commit_completed_token_drop_updates(scheduler, metadata)
    assert len(blocks) == 2
    assert get_resident_kv_tokens("td") == 32
    scheduler._gpu_block_pool.free_blocks.assert_not_called()


def test_current_generation_reclaims_native_tail_once_completed():
    scheduler, _ = _scheduler(logical=32, prompt=24)
    scheduler._token_drop_finished_before_output = set()
    scheduler._group_tokens_per_block = [16]
    scheduler._gpu_block_pool = Mock()
    scheduler._get_request_tracker = scheduler.request_trackers.__getitem__
    scheduler.request_trackers["td"].allocated_block_ids = {0: [7, 9]}
    blocks = scheduler._token_drop_allocations["td"].blocks[0]
    set_resident_kv_tokens("td", 32)
    metadata = LMCacheMPWorkerMetadata(
        completed_store_requests={}, resident_kv_updates={"td": (1, 16)}
    )
    commit_completed_token_drop_updates(scheduler, metadata)
    assert [b.block_id for b in blocks] == [7]
    assert scheduler.request_trackers["td"].allocated_block_ids[0] == [7]
    assert get_resident_kv_tokens("td") == 16
    assert scheduler._gpu_block_pool.free_blocks.call_count == 1


def test_multi_connector_empty_native_blocks_rejected_before_compaction():
    # MultiConnector forwards empty KVCacheBlocks to non-selected children.
    # TD must fail closed instead of compacting KV with untracked ownership.
    scheduler, _ = _scheduler()
    scheduler._token_drop_allocations["td"] = SimpleNamespace(blocks=[[]])
    with pytest.raises(RuntimeError, match="nonempty native private blocks"):
        require_private_token_drop_blocks(scheduler, "td")


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_full_cudagraph_rejected_before_td_hooks(mode, monkeypatch):
    connector, request = _admission_connector()
    connector._vllm_config.compilation_config.cudagraph_mode = mode

    def forbidden():
        raise AssertionError("TD allocator/APC hook installed before mode validation")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_allocator_adapter",
        forbidden,
    )
    with pytest.raises(ValueError, match="CUDA Graph mode NONE or PIECEWISE"):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


def test_retrieve_failure_does_not_install_query_hooks():
    events = []
    metadata = LMCacheMPConnectorMetadata()
    metadata.requests = [
        SimpleNamespace(
            direction="RETRIEVE",
            request_id="normal",
            op=object(),
            cache_salt="",
            request_configs={},
        )
    ]

    def submit(*args, **kwargs):
        events.append("retrieve")
        raise RuntimeError("retrieve submit failed")

    worker = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
        _token_drop_worker=SimpleNamespace(
            prepare_forward=lambda *args: events.append("query_hook")
        ),
        worker_adapter=SimpleNamespace(
            create_recorded_event=lambda: object(),
            batched_submit_retrieve_requests=submit,
        ),
    )
    with pytest.raises(RuntimeError, match="retrieve submit failed"):
        LMCacheMPConnector.start_load_kv(worker, object())
    assert events == ["retrieve"]


def test_mixed_retrieve_is_submitted_before_query_hook_install():
    events = []
    metadata = LMCacheMPConnectorMetadata()
    metadata.requests = [
        SimpleNamespace(
            direction="RETRIEVE",
            request_id="normal",
            op=object(),
            cache_salt="",
            request_configs={},
        )
    ]
    metadata.token_drop_requests.append(object())

    def prepare(_ctx, states):
        assert states is metadata.token_drop_requests
        events.append("query_hook")

    worker = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
        _token_drop_worker=SimpleNamespace(prepare_forward=prepare),
        worker_adapter=SimpleNamespace(
            create_recorded_event=lambda: object(),
            batched_submit_retrieve_requests=lambda *args, **kwargs: events.append(
                "retrieve"
            ),
        ),
    )
    LMCacheMPConnector.start_load_kv(worker, object())
    assert events == ["retrieve", "query_hook"]


def test_td_only_step_installs_query_hook_without_retrieve():
    observed = []
    metadata = LMCacheMPConnectorMetadata()
    worker = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
        _token_drop_worker=SimpleNamespace(
            prepare_forward=lambda ctx, states: observed.append(states)
        ),
    )
    LMCacheMPConnector.start_load_kv(worker, object())
    assert observed == [metadata.token_drop_requests]


def test_preemption_reset_does_not_claim_request_finished():
    scheduler, output = _scheduler()
    scheduler._token_drop_pending_finished_ids = set()
    scheduler._token_drop_finished_before_output = set()
    set_resident_kv_tokens("td", 33)
    output.preempted_req_ids = {"td"}
    output.num_scheduled_tokens.pop("td")
    metadata = LMCacheMPConnectorMetadata()

    reset_preempted_token_drop_requests(scheduler, output, metadata)

    assert metadata.token_drop_reset_ids == {"td"}
    assert metadata.token_drop_finished_ids == set()
    assert scheduler._token_drop_pending_finished_ids == set()


@pytest.mark.parametrize(
    "backend", [AttentionBackendEnum.FLASHINFER, AttentionBackendEnum.TRITON_ATTN]
)
def test_explicit_unsupported_attention_backend_rejected_before_hooks(
    backend, monkeypatch
):
    connector, request = _admission_connector()
    connector._vllm_config.attention_config = SimpleNamespace(backend=backend)

    def forbidden():
        raise AssertionError("TD hooks installed before backend validation")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_allocator_adapter",
        forbidden,
    )
    with pytest.raises(ValueError, match="requires FlashAttention backend"):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


@pytest.mark.parametrize("kind", ["v2_runner", "microbatching"])
def test_td_admission_rejects_unsupported_runner_before_hooks(kind, monkeypatch):
    connector, request = _admission_connector()
    if kind == "v2_runner":
        connector._vllm_config.use_v2_model_runner = True
    else:
        connector._vllm_config.parallel_config.use_ubatching = True

    def forbidden():
        raise AssertionError("TD hooks installed before runner validation")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    with pytest.raises(ValueError, match="classic|microbatch"):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


@pytest.mark.parametrize(
    "kind",
    [
        "sliding_window",
        "sliding_alloc",
        "chunked_local",
        "non_causal",
        "sink",
        "quantized",
        "asymmetric_kv",
        "mixed",
        "missing_config",
        "missing_groups",
    ],
)
def test_td_admission_rejects_unsupported_kv_specs_before_hooks(kind, monkeypatch):
    connector, request = _admission_connector()
    base = dict(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    if kind == "sliding_window":
        spec = SlidingWindowSpec(**base, sliding_window=64)
    elif kind == "sliding_alloc":
        spec = FullAttentionSpec(**base, sliding_window=64)
    elif kind == "chunked_local":
        spec = FullAttentionSpec(**base, attention_chunk_size=64)
    elif kind == "non_causal":
        spec = FullAttentionSpec(**base, non_causal=True)
    elif kind == "sink":
        spec = SinkFullAttentionSpec(**base, sink_len=4)
    elif kind == "quantized":
        spec = FullAttentionSpec(**base, kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)
    elif kind == "asymmetric_kv":
        spec = FullAttentionSpec(**base, head_size_v=32)
    else:
        spec = FullAttentionSpec(**base)
    if kind == "missing_config":
        connector._kv_cache_config = None
    elif kind == "missing_groups":
        connector._kv_cache_config.kv_cache_groups = []
    elif kind == "mixed":
        connector._kv_cache_config.kv_cache_groups[
            0
        ].kv_cache_spec = UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={
                "layer.0": spec,
                "layer.1": SlidingWindowSpec(**base, sliding_window=64),
            },
        )
    else:
        connector._kv_cache_config.kv_cache_groups[
            0
        ].kv_cache_spec = UniformTypeKVCacheSpecs.from_specs({"layer.0": spec})

    def forbidden():
        raise AssertionError("TD hooks installed before topology validation")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    with pytest.raises(ValueError, match="KV cache group|full-attention"):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


def test_td_on_new_request_arms_apc_before_external_lookup(monkeypatch):
    connector, request = _admission_connector()
    connector.role = KVConnectorRole.SCHEDULER
    connector._token_drop_enabled = True
    connector._eager_prefetch = False
    connector._get_or_create_request_tracker = lambda _: SimpleNamespace(
        token_drop_spec=object()
    )
    connector.scheduler_adapter = SimpleNamespace(
        maybe_submit_lookup_request=Mock(
            side_effect=AssertionError("TD must not enter external lookup")
        )
    )
    request.resumable = False
    request.sampling_params = SimpleNamespace(
        extra_args={
            "kv_transfer_params": {
                "lmcache.token_drop": {"algorithm": "rkv", "config": {}}
            }
        }
    )
    seen = []
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        lambda: seen.append("apc"),
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_allocator_adapter",
        lambda: seen.append("allocator"),
    )
    assert request.skip_reading_prefix_cache is False
    LMCacheMPConnector.on_new_request(connector, request)
    assert request.skip_reading_prefix_cache is True
    assert seen == ["apc", "allocator"]
