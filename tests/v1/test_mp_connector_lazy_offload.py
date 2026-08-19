# SPDX-License-Identifier: Apache-2.0
"""Thin delegation tests for lazy offload in ``LMCacheMPConnector``.

Manager behavior is covered in ``test_lazy_offload_manager.py``. These tests
keep only the connector boundary: lifecycle events are forwarded to the
manager, returned actions are applied, and the pre-existing connector return
contracts remain intact.
"""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadActions,
    LazyOffloadManager,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    KVConnectorRole,
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)
import lmcache.integration.vllm.lmcache_mp_connector as mp_connector_module  # noqa: E402

TOKENS_PER_BLOCK = 16


@dataclass
class _RecordingManager:
    """Configurable test double for the connector's manager boundary."""

    scheduler_actions: LazyOffloadActions = field(default_factory=LazyOffloadActions)
    store_actions: LazyOffloadActions = field(default_factory=LazyOffloadActions)
    finished_actions: LazyOffloadActions = field(default_factory=LazyOffloadActions)
    arrival_actions: LazyOffloadActions = field(default_factory=LazyOffloadActions)
    scheduler_steps: list[object] = field(default_factory=list)
    store_results: list[tuple[set[str], dict[str, int]]] = field(default_factory=list)
    finished_requests: list[str] = field(default_factory=list)
    reset_requests: list[str] = field(default_factory=list)
    arrived_requests: list[str] = field(default_factory=list)
    candidates: list[LMCacheMPRequestMetadata] = field(default_factory=list)
    bound_pools: list[object] = field(default_factory=list)
    final_log_calls: int = 0

    def on_scheduler_step(self, scheduler_output: object) -> LazyOffloadActions:
        self.scheduler_steps.append(scheduler_output)
        return self.scheduler_actions

    def on_store_results(
        self, failed_request_ids: set[str], completed_store_counts: dict[str, int]
    ) -> LazyOffloadActions:
        self.store_results.append((failed_request_ids, completed_store_counts))
        return self.store_actions

    def on_request_finished(self, request_id: str) -> LazyOffloadActions:
        self.finished_requests.append(request_id)
        return self.finished_actions

    def on_request_reset(self, request_id: str) -> int:
        self.reset_requests.append(request_id)
        return 0

    def on_request_arrived(self, request_id: str) -> LazyOffloadActions:
        self.arrived_requests.append(request_id)
        return self.arrival_actions

    def add_store_candidate(self, metadata: LMCacheMPRequestMetadata) -> None:
        self.candidates.append(metadata)

    def bind_block_pool(self, pool: object) -> None:
        self.bound_pools.append(pool)

    def log_final_stats(self) -> None:
        self.final_log_calls += 1


class _FakeSchedulerAdapter:
    def __init__(self) -> None:
        self.ended_sessions: list[str] = []
        self.shutdown_calls = 0
        self.lookup_result: int | None = 0
        self.lmcache_tokens_per_chunk = TOKENS_PER_BLOCK

    def end_session(self, request_id: str) -> None:
        self.ended_sessions.append(request_id)

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int], cache_salt: str
    ) -> None:
        pass

    def check_lookup_result(self, request_id: str) -> int | None:
        return self.lookup_result


class _ConstructorFakeSchedulerAdapter:
    """Stands in for the real adapter in full-constructor tests.

    The real adapter opens a ZMQ handshake with the LMCache server at
    construction; the constructor tests only need the attributes the
    connector's ``__init__`` reads afterwards.
    """

    lmcache_tokens_per_chunk = 16 * TOKENS_PER_BLOCK

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeKVTransferConfig:
    """The two ``KVTransferConfig`` members the connector constructor reads."""

    def __init__(self, extra_config: dict[str, Any]) -> None:
        self.kv_connector_extra_config = extra_config

    def get_from_extra_config(self, key: str, default: Any) -> Any:
        return self.kv_connector_extra_config.get(key, default)


def _make_vllm_config(
    lazy_offload: bool, enable_prefix_caching: bool
) -> SimpleNamespace:
    """Duck-typed single-rank ``VllmConfig`` for constructor validation."""
    return SimpleNamespace(
        kv_transfer_config=_FakeKVTransferConfig(
            {"lmcache.mp.lazy_offload": lazy_offload}
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
            block_size=TOKENS_PER_BLOCK,
        ),
        parallel_config=SimpleNamespace(
            world_size=1,
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            model="test-model", use_mla=False, is_hybrid=False
        ),
    )


@dataclass
class _Harness:
    connector: LMCacheMPConnector
    manager: _RecordingManager
    adapter: _FakeSchedulerAdapter


def _make_connector() -> _Harness:
    """Construct the smallest scheduler-side connector needed by smoke tests."""
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    manager = _RecordingManager()
    adapter = _FakeSchedulerAdapter()
    connector.lazy_offload = True
    connector.request_trackers = {}
    connector._group_tokens_per_block = [TOKENS_PER_BLOCK]
    connector._hit_alignment_tokens = TOKENS_PER_BLOCK
    connector._lazy_offload_manager = manager  # type: ignore[assignment]
    connector.scheduler_adapter = adapter  # type: ignore[assignment]
    return _Harness(connector, manager, adapter)


def _store_metadata(request_id: str = "req") -> LMCacheMPRequestMetadata:
    return LMCacheMPRequestMetadata(
        request_id=request_id,
        direction="STORE",
        op=LoadStoreOp(
            token_ids=list(range(32)),
            block_ids=[[1, 2]],
            start=0,
            end=32,
        ),
        cache_salt="",
    )


def _scheduler_output(total_tokens: int = 32) -> SimpleNamespace:
    return SimpleNamespace(
        total_num_scheduled_tokens=total_tokens,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=[]
        ),
        preempted_req_ids=[],
    )


def _stub_regular_step_processing(connector: LMCacheMPConnector) -> None:
    def _no_op(*args: Any, **kwargs: Any) -> None:
        return None

    connector._process_retrieve_requests = _no_op  # type: ignore[method-assign]
    connector._process_new_requests = _no_op  # type: ignore[method-assign]
    connector._process_cached_requests = _no_op  # type: ignore[method-assign]
    connector._report_block_allocation_deltas = _no_op  # type: ignore[method-assign]


def test_constructor_rejects_lazy_offload_without_prefix_caching() -> None:
    """Eviction detection reads vLLM block hashes, which only exist while
    prefix caching maintains them; the misconfiguration must fail fast,
    before the server handshake."""
    with pytest.raises(ValueError, match="prefix caching"):
        LMCacheMPConnector(
            _make_vllm_config(lazy_offload=True, enable_prefix_caching=False),  # type: ignore[arg-type]
            KVConnectorRole.SCHEDULER,
        )


def test_constructor_accepts_eager_without_prefix_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is scoped to lazy mode; eager deployments keep working
    with prefix caching off."""
    monkeypatch.setattr(
        mp_connector_module,
        "LMCacheMPSchedulerAdapter",
        _ConstructorFakeSchedulerAdapter,
    )

    connector = LMCacheMPConnector(
        _make_vllm_config(lazy_offload=False, enable_prefix_caching=False),  # type: ignore[arg-type]
        KVConnectorRole.SCHEDULER,
    )

    assert connector.lazy_offload is False


def test_constructor_wires_the_manager_for_the_lazy_scheduler_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler-role constructor must build the manager the lazy
    routing path delegates to; the harness elsewhere in this file wires a
    recording double by hand, so only this test proves the production
    wiring exists."""
    monkeypatch.setattr(
        mp_connector_module,
        "LMCacheMPSchedulerAdapter",
        _ConstructorFakeSchedulerAdapter,
    )

    connector = LMCacheMPConnector(
        _make_vllm_config(lazy_offload=True, enable_prefix_caching=True),  # type: ignore[arg-type]
        KVConnectorRole.SCHEDULER,
    )

    assert connector.lazy_offload is True
    assert isinstance(connector._lazy_offload_manager, LazyOffloadManager)


def test_build_connector_meta_forwards_step_and_applies_actions() -> None:
    harness = _make_connector()
    _stub_regular_step_processing(harness.connector)
    store = _store_metadata()
    harness.manager.scheduler_actions = LazyOffloadActions(
        stores_to_submit=[store], sessions_to_end=["finished"]
    )
    output = _scheduler_output()

    metadata = harness.connector.build_connector_meta(output)

    assert harness.manager.scheduler_steps == [output]
    assert metadata.requests == [store]
    assert harness.adapter.ended_sessions == ["finished"]


def test_build_connector_meta_forwards_zero_token_step_to_manager() -> None:
    """The manager owns no-forward gating; the connector does not duplicate it."""
    harness = _make_connector()
    _stub_regular_step_processing(harness.connector)
    output = _scheduler_output(total_tokens=0)

    metadata = harness.connector.build_connector_meta(output)

    assert harness.manager.scheduler_steps == [output]
    assert len(metadata) == 0


def test_update_connector_output_forwards_receipts_and_applies_actions() -> None:
    harness = _make_connector()
    harness.manager.store_actions = LazyOffloadActions(sessions_to_end=["req"])
    output = SimpleNamespace(
        kv_connector_worker_meta=LMCacheMPWorkerMetadata(
            completed_store_requests={"req": 2},
            failed_store_requests={"req"},
        )
    )

    harness.connector.update_connector_output(output)

    assert harness.manager.store_results == [({"req"}, {"req": 2})]
    assert harness.adapter.ended_sessions == ["req"]


def test_update_connector_output_ignores_foreign_metadata() -> None:
    harness = _make_connector()

    harness.connector.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=None)
    )

    assert harness.manager.store_results == []


def test_request_finished_forwards_event_and_keeps_lazy_return_contract() -> None:
    harness = _make_connector()
    harness.manager.finished_actions = LazyOffloadActions(sessions_to_end=["finished"])
    request = SimpleNamespace(request_id="req", kv_transfer_params=None)

    delay_free, params = harness.connector.request_finished(request, [])

    assert (delay_free, params) == (False, None)
    assert harness.manager.finished_requests == ["req"]
    assert harness.adapter.ended_sessions == ["finished"]


def test_new_request_arrival_forwards_event_and_applies_release() -> None:
    harness = _make_connector()
    harness.manager.arrival_actions = LazyOffloadActions(
        sessions_to_end=["predecessor"]
    )
    request = SimpleNamespace(
        request_id="X",
        status=RequestStatus.WAITING,
        cache_salt="",
        all_token_ids=list(range(32)),
    )

    tracker = harness.connector._get_or_create_request_tracker(request)

    assert tracker is harness.connector.request_trackers["X"]
    assert harness.manager.arrived_requests == ["X"]
    assert harness.adapter.ended_sessions == ["predecessor"]


def test_preempted_tracker_reset_forwards_reset_before_rearrival() -> None:
    harness = _make_connector()
    request = SimpleNamespace(
        request_id="X",
        status=RequestStatus.PREEMPTED,
        cache_salt="",
        all_token_ids=list(range(32)),
    )
    stale = LMCacheMPRequestTracker(request)  # type: ignore[arg-type]
    stale.state = LMCacheMPRequestState.READY
    harness.connector.request_trackers["X"] = stale

    tracker = harness.connector._get_or_create_request_tracker(request)  # type: ignore[arg-type]

    assert tracker is not stale
    assert harness.manager.reset_requests == ["X"]
    assert harness.manager.arrived_requests == ["X"]


def test_lookup_miss_still_records_the_vllm_prefix_hit() -> None:
    harness = _make_connector()
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="F",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )

    need_to_load, is_async = harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    )

    assert (need_to_load, is_async) == (0, False)
    tracker = harness.connector.request_trackers["F"]
    assert tracker.num_vllm_hit_tokens == 3 * TOKENS_PER_BLOCK
    assert tracker.num_lmcache_hit_tokens == 0


def test_eager_lookup_miss_backfills_the_apc_prefix_without_duplicates() -> None:
    """The mode-independent APC accounting fixes eager under-store.

    Only four tokens were newly scheduled, which is below one LMCache chunk.
    The APC prefix nevertheless has valid GPU KV and must be copied down before
    eviction. Once copied, later eager metadata starts exactly after it.
    """
    harness = _make_connector()
    harness.connector.lazy_offload = False
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="E",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )

    need_to_load, is_async = harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    )

    assert (need_to_load, is_async) == (0, False)
    tracker = harness.connector.request_trackers["E"]
    assert tracker.num_vllm_hit_tokens == 3 * TOKENS_PER_BLOCK
    assert tracker.num_lmcache_hit_tokens == 0
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    tracker.num_scheduled_tokens = 4

    backfill = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
        group_tokens_per_block=[TOKENS_PER_BLOCK],
    )

    assert backfill is not None
    assert (backfill.op.start, backfill.op.end) == (0, 3 * TOKENS_PER_BLOCK)
    assert tracker.num_stored_tokens == 3 * TOKENS_PER_BLOCK

    tracker.increase_num_scheduled_tokens(TOKENS_PER_BLOCK - 4)
    next_chunk = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
        group_tokens_per_block=[TOKENS_PER_BLOCK],
    )
    assert next_chunk is not None
    assert (next_chunk.op.start, next_chunk.op.end) == (
        3 * TOKENS_PER_BLOCK,
        4 * TOKENS_PER_BLOCK,
    )


def test_eager_apc_backfill_uses_the_existing_immediate_store_path() -> None:
    """The eager side effect changes coverage, not offload orchestration."""
    harness = _make_connector()
    harness.connector.lazy_offload = False
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="E-immediate",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )
    assert harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    ) == (0, False)
    tracker = harness.connector.request_trackers["E-immediate"]
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["E-immediate"],
            new_block_ids=[[]],
            resumed_req_ids={"E-immediate"},
        ),
        num_scheduled_tokens={"E-immediate": 4},
    )
    connector_metadata = LMCacheMPConnectorMetadata()

    harness.connector._process_cached_requests(
        scheduler_output,
        connector_metadata,  # type: ignore[arg-type]
    )

    assert len(connector_metadata.requests) == 1
    store = connector_metadata.requests[0]
    assert store.direction == "STORE"
    assert (store.op.start, store.op.end) == (0, 3 * TOKENS_PER_BLOCK)
    assert harness.manager.candidates == []


def test_lazy_new_request_store_routes_to_manager_not_metadata() -> None:
    """In lazy mode a new request's store becomes a deferred candidate;
    nothing may reach the step's connector metadata, or the worker would
    store it immediately and the policy would have decided nothing."""
    harness = _make_connector()
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="L-new",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )
    assert harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    ) == (0, False)
    tracker = harness.connector.request_trackers["L-new"]
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="L-new")],
        num_scheduled_tokens={"L-new": 4 * TOKENS_PER_BLOCK},
    )
    connector_metadata = LMCacheMPConnectorMetadata()

    harness.connector._process_new_requests(
        scheduler_output,
        connector_metadata,  # type: ignore[arg-type]
    )

    assert connector_metadata.requests == []
    assert [c.request_id for c in harness.manager.candidates] == ["L-new"]
    store = harness.manager.candidates[0]
    assert store.direction == "STORE"
    assert (store.op.start, store.op.end) == (0, 4 * TOKENS_PER_BLOCK)


def test_lazy_cached_request_store_routes_to_manager_not_metadata() -> None:
    """The cached-request path (decode steps, APC backfill) must take the
    same lazy fork as new requests: candidates to the manager, an empty
    step metadata. Mirrors the eager immediate-store test above."""
    harness = _make_connector()
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="L-cached",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )
    assert harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    ) == (0, False)
    tracker = harness.connector.request_trackers["L-cached"]
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["L-cached"],
            new_block_ids=[[]],
            resumed_req_ids={"L-cached"},
        ),
        num_scheduled_tokens={"L-cached": 4},
    )
    connector_metadata = LMCacheMPConnectorMetadata()

    harness.connector._process_cached_requests(
        scheduler_output,
        connector_metadata,  # type: ignore[arg-type]
    )

    assert connector_metadata.requests == []
    assert [c.request_id for c in harness.manager.candidates] == ["L-cached"]
    store = harness.manager.candidates[0]
    assert (store.op.start, store.op.end) == (0, 3 * TOKENS_PER_BLOCK)


def test_eager_without_an_aligned_apc_hit_keeps_the_old_chunk_threshold() -> None:
    """The eager fix is inert when APC contributes no complete chunk."""
    harness = _make_connector()
    harness.connector.lazy_offload = False
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="E-no-apc",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )

    assert harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=4
    ) == (0, False)
    tracker = harness.connector.request_trackers["E-no-apc"]
    tracker.allocated_block_ids = {0: [1, 2]}
    tracker.num_scheduled_tokens = 4

    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
            group_tokens_per_block=[TOKENS_PER_BLOCK],
        )
        is None
    )


def test_eager_apc_backfill_excludes_the_prefix_already_in_lmcache() -> None:
    """APC and LMCache hits overlap; they are never summed or re-stored."""
    harness = _make_connector()
    harness.connector.lazy_offload = False
    harness.adapter.lookup_result = 2 * TOKENS_PER_BLOCK
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="E-partial",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )

    assert harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    ) == (0, False)
    tracker = harness.connector.request_trackers["E-partial"]
    assert tracker.num_lmcache_hit_tokens == 2 * TOKENS_PER_BLOCK
    assert tracker.num_vllm_hit_tokens == 3 * TOKENS_PER_BLOCK
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    tracker.num_scheduled_tokens = 4

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
        group_tokens_per_block=[TOKENS_PER_BLOCK],
    )

    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (
        2 * TOKENS_PER_BLOCK,
        3 * TOKENS_PER_BLOCK,
    )


def test_store_metadata_covers_the_vllm_hit_tokens() -> None:
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="F",
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    tracker.num_scheduled_tokens = 2
    tracker.num_vllm_hit_tokens = 3 * TOKENS_PER_BLOCK

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
        group_tokens_per_block=[TOKENS_PER_BLOCK],
    )

    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (0, 3 * TOKENS_PER_BLOCK)


def test_shutdown_delegates_final_log_before_adapter_shutdown() -> None:
    harness = _make_connector()

    harness.connector.shutdown()

    assert harness.manager.final_log_calls == 1
    assert harness.adapter.shutdown_calls == 1
