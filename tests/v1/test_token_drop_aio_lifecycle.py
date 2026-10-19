# SPDX-License-Identifier: Apache-2.0
"""Scheduler/worker token-drop metadata and output barrier contracts."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

pytest.importorskip("vllm")
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# First Party
from lmcache.integration.vllm.experimental.token_drop import TokenDropSpec
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    add_token_drop_states,
    admit_token_drop_request,
    commit_completed_token_drop_updates,
    reset_preempted_token_drop_requests,
)
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    clear_resident_kv_tokens,
    get_resident_kv_tokens,
    set_resident_kv_tokens,
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


def _admission_connector():
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False, async_scheduling=False
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(world_size=1),
    )
    connector = SimpleNamespace(
        _vllm_config=cfg,
        _group_tokens_per_block=[16],
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
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_apc_guard",
        forbidden,
    )
    with pytest.raises(ValueError, match=error):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


def test_supported_td_admission_installs_both_adaptors(monkeypatch):
    connector, request = _admission_connector()
    called = []
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_apc_guard",
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
    assert state.is_genuine_decode is True
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
    assert state.is_genuine_decode is False


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
        completed_store_requests={}, resident_kv_updates={"td": 16}
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
        _pending_token_drop_updates={"td": 21},
    )

    metadata = LMCacheMPConnector.build_connector_worker_meta(worker)

    assert isinstance(metadata, LMCacheMPWorkerMetadata)
    assert metadata.resident_kv_updates == {"td": 21}
    assert worker._pending_token_drop_updates == {}
    assert LMCacheMPConnector.build_connector_worker_meta(worker) is None


def test_forward_end_collects_only_completed_compaction_reports():
    metadata = LMCacheMPConnectorMetadata()
    worker = SimpleNamespace(
        _get_connector_metadata=lambda: metadata,
        _token_drop_worker=SimpleNamespace(compact=lambda: {"td": 12}),
        _pending_token_drop_updates={},
        dispatcher=None,
    )
    LMCacheMPConnector.wait_for_save(worker)
    assert worker._pending_token_drop_updates == {"td": 12}
    # The lifecycle calls compact() once per step; there is no second validator.


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


@pytest.mark.parametrize(
    ("finished", "expected_stores"),
    [
        ({"normal", "td"}, {"normal"}),
        ({"td"}, set()),
        ({"normal"}, {"normal"}),
    ],
)
def test_worker_completion_excludes_token_drop_without_hiding_normal_store(
    finished, expected_stores
):
    # vLLM has already reclaimed TD's private KV blocks at request_finished().
    # Reporting that ID as finished_sending would crash the scheduler.
    forwarded = []
    cleaned = []

    def from_adapter(ids):
        forwarded.append(set(ids))
        return set(ids), set()

    def finish_requests(ids):
        cleaned.extend(ids)
        return ids & {"td"}

    worker = SimpleNamespace(
        lazy_offload=False,
        _can_store=True,
        worker_adapter=SimpleNamespace(get_finished=from_adapter),
        _token_drop_worker=SimpleNamespace(finish_requests=finish_requests),
    )
    sending, receiving = LMCacheMPConnector.get_finished(worker, finished)
    assert sending == expected_stores
    assert receiving == set()
    assert forwarded == [expected_stores]
    assert set(cleaned) == finished
