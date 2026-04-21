# SPDX-License-Identifier: Apache-2.0
"""
Tests for the SYNC_LOOKUP protocol and the paired sync-lookup
scheduler / worker adapters.

Covers:
- Protocol registration: enum, payload, response, handler type.
- Message-queue round-trip with a stub server handler.
- Server-side sync_lookup behavior: return value, job lifecycle for
  zero-hits vs. non-zero-hits, thread-pool registration.
- end_session fallback cleanup of a lingering prefetch job.
- LMCacheMPSyncLookupSchedulerAdapter: single SYNC_LOOKUP round-trip,
  cached result for repeated check_lookup_result.
- LMCacheMPSyncLookupWorkerAdapter: queued retrieves, dispatch after
  QUERY_PREFETCH_STATUS returns non-None, error_block_ids lock.
"""

# Standard
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
import threading
import time

# First Party
from lmcache.v1.distributed.api import PrefetchHandle
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType
from lmcache.v1.multiprocess.server import MPCacheEngine, _PrefetchJob

# Test helpers
from tests.v1.multiprocess.test_mq import MessageQueueTestHelper

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSyncLookupSchedulerAdapter,
        LMCacheMPSyncLookupWorkerAdapter,
    )

# ============================================================================
# Protocol definition tests
# ============================================================================


def test_sync_lookup_in_request_type() -> None:
    """SYNC_LOOKUP should be a member of RequestType."""
    assert hasattr(RequestType, "SYNC_LOOKUP")
    assert isinstance(RequestType.SYNC_LOOKUP, RequestType)


def test_sync_lookup_payload_classes() -> None:
    """SYNC_LOOKUP payload should be [IPCCacheEngineKey, int]."""
    payload_classes = get_payload_classes(RequestType.SYNC_LOOKUP)
    assert len(payload_classes) == 2
    assert payload_classes[0] is IPCCacheEngineKey
    assert payload_classes[1] is int


def test_sync_lookup_response_class() -> None:
    """SYNC_LOOKUP response should be int (not Optional)."""
    response_class = get_response_class(RequestType.SYNC_LOOKUP)
    assert response_class is int


def test_sync_lookup_handler_type() -> None:
    """SYNC_LOOKUP should use BLOCKING handler type."""
    handler_type = get_handler_type(RequestType.SYNC_LOOKUP)
    assert handler_type == HandlerType.BLOCKING


# ============================================================================
# Message-queue round-trip test
# ============================================================================


def _sync_lookup_handler(key: IPCCacheEngineKey, tp_size: int) -> int:
    """Stub handler: accept the expected payload and return a fixed count."""
    assert isinstance(tp_size, int)
    return 7


def test_mq_sync_lookup_round_trip() -> None:
    """Client SYNC_LOOKUP request returns the handler-provided int."""
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5591")
    helper.register_handler(RequestType.SYNC_LOOKUP, _sync_lookup_handler)

    key = IPCCacheEngineKey(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=tuple(range(256)),
        start=0,
        end=256,
        request_id="req-mq",
    )
    helper.run_test(
        request_type=RequestType.SYNC_LOOKUP,
        payloads=[key, 1],
        expected_response=7,
        num_requests=1,
    )


# ============================================================================
# Server handler tests
# ============================================================================


def _install_prefetch_job(
    engine: MagicMock,
    request_id: str,
    world_size: int = 1,
) -> _PrefetchJob:
    """Create a _PrefetchJob and register it on the mock engine."""
    handle = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id=request_id,
        l1_prefix_hit_count=0,
        total_requested_keys=10,
        submit_time=time.monotonic(),
    )
    job = _PrefetchJob(handle=handle, world_size=world_size, request_id=request_id)
    engine._prefetch_jobs[request_id] = job
    return job


def _make_sync_lookup_engine() -> tuple[MagicMock, IPCCacheEngineKey]:
    """Return a mock engine and a cache key suitable for sync_lookup()."""
    engine = MagicMock()
    engine._prefetch_job_lock = threading.Lock()
    engine._prefetch_jobs = {}
    # Ultra-short poll so tests that exercise the "still in progress"
    # branch don't pay a real sleep tax.
    engine._SYNC_LOOKUP_POLL_INTERVAL_S = 0.001

    # lookup() is mocked: the real method would register the job via
    # _register_prefetch_job. For tests we install the job directly
    # after calling sync_lookup -> lookup.
    key = IPCCacheEngineKey(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=tuple(range(256)),
        start=0,
        end=256,
        request_id="req-sl",
    )
    return engine, key


def test_server_sync_lookup_returns_hit_count_non_zero() -> None:
    """sync_lookup returns hit_count = storage_return // world_size."""
    engine, key = _make_sync_lookup_engine()

    def fake_lookup(k: IPCCacheEngineKey, tp: int) -> None:
        _install_prefetch_job(engine, k.request_id, world_size=2)

    engine.lookup.side_effect = fake_lookup
    engine.storage_manager.query_prefetch_lookup_hits.return_value = 8

    result = MPCacheEngine.sync_lookup(engine, key, 1)

    assert result == 4  # 8 // world_size (2)
    # Non-zero hits: job must remain alive for worker polling.
    assert key.request_id in engine._prefetch_jobs
    engine.storage_manager.query_prefetch_status.assert_not_called()


def test_server_sync_lookup_zero_hits_drains_job() -> None:
    """Zero hits should pop the job and drain the prefetch controller."""
    engine, key = _make_sync_lookup_engine()

    def fake_lookup(k: IPCCacheEngineKey, tp: int) -> None:
        _install_prefetch_job(engine, k.request_id, world_size=1)

    engine.lookup.side_effect = fake_lookup
    engine.storage_manager.query_prefetch_lookup_hits.return_value = 0

    result = MPCacheEngine.sync_lookup(engine, key, 1)

    assert result == 0
    # Zero hits: job must be popped, and query_prefetch_status must have
    # been called to consume the prefetch controller's completed_results.
    assert key.request_id not in engine._prefetch_jobs
    engine.storage_manager.query_prefetch_status.assert_called_once()


def test_server_sync_lookup_waits_for_lookup_phase() -> None:
    """sync_lookup polls while query_prefetch_lookup_hits returns None."""
    engine, key = _make_sync_lookup_engine()

    def fake_lookup(k: IPCCacheEngineKey, tp: int) -> None:
        _install_prefetch_job(engine, k.request_id, world_size=1)

    engine.lookup.side_effect = fake_lookup
    # First two polls still in progress, third returns a hit count.
    engine.storage_manager.query_prefetch_lookup_hits.side_effect = [None, None, 3]

    result = MPCacheEngine.sync_lookup(engine, key, 1)

    assert result == 3
    assert engine.storage_manager.query_prefetch_lookup_hits.call_count == 3


def test_server_sync_lookup_missing_job_returns_zero() -> None:
    """If lookup() fails to register a job, sync_lookup returns 0."""
    engine, key = _make_sync_lookup_engine()
    # lookup() is a no-op: no job is registered.
    engine.lookup.return_value = None

    result = MPCacheEngine.sync_lookup(engine, key, 1)

    assert result == 0
    engine.storage_manager.query_prefetch_lookup_hits.assert_not_called()


def test_server_sync_lookup_handler_registered() -> None:
    """MPCacheEngine should have a sync_lookup method."""
    engine = MPCacheEngine.__new__(MPCacheEngine)
    assert hasattr(engine, "sync_lookup")
    assert callable(engine.sync_lookup)


def test_sync_lookup_uses_dedicated_thread_pool() -> None:
    """run_cache_server should assign SYNC_LOOKUP to its own normal pool.

    The failure mode this guards against: if SYNC_LOOKUP shares a pool
    with short operations like PING / END_SESSION, concurrent blocking
    polls can starve the heartbeat and falsely mark the server
    unhealthy.  Verified by inspecting the assigned executors.
    """
    # First Party
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.protocol import _PROTOCOL_DEFINITIONS
    import lmcache.v1.multiprocess.server as server_mod

    # run_cache_server needs real handler registration but we don't want
    # to boot the ZMQ loop or engine. Stub out the expensive bits.
    with (
        patch.object(
            server_mod, "init_observability", return_value=MagicMock(stop=lambda: None)
        ),
        patch.object(server_mod, "maybe_initialize_trace_recorder"),
        patch.object(server_mod, "MPCacheEngine") as MockEngine,
        patch.object(server_mod, "MessageQueueServer") as MockServer,
        patch("torch.cuda.init"),
    ):
        engine_mock = MagicMock()
        # Every method used by add_handler_helper must exist as a callable.
        for rt in _PROTOCOL_DEFINITIONS:
            setattr(engine_mock, rt.name.lower(), MagicMock())
        engine_mock.sync_lookup = MagicMock()
        engine_mock.query_prefetch_status = MagicMock()
        engine_mock.query_prefetch_lookup_hits = MagicMock()
        engine_mock.free_lookup_locks = MagicMock()
        engine_mock.lookup = MagicMock()
        MockEngine.return_value = engine_mock

        server_instance = MagicMock()
        MockServer.return_value = server_instance

        # First Party
        from lmcache.v1.distributed.config import StorageManagerConfig
        from lmcache.v1.mp_observability.config import ObservabilityConfig

        server_mod.run_cache_server(
            mp_config=MPServerConfig(max_sync_lookup_workers=4),
            storage_manager_config=MagicMock(spec=StorageManagerConfig),
            obs_config=MagicMock(spec=ObservabilityConfig),
            return_engine=True,
        )

    # Collect the request-type sets passed to add_normal_thread_pool.
    normal_pool_calls = server_instance.add_normal_thread_pool.call_args_list
    pool_request_sets: list[set[RequestType]] = []
    sync_lookup_max_workers: int | None = None
    for call in normal_pool_calls:
        rts = call.kwargs.get("request_types") or call.args[0]
        max_workers = call.kwargs.get("max_workers") or call.args[1]
        pool_request_sets.append(set(rts))
        if RequestType.SYNC_LOOKUP in rts:
            sync_lookup_max_workers = max_workers

    # SYNC_LOOKUP must be in exactly one pool, alone.
    sync_lookup_pools = [s for s in pool_request_sets if RequestType.SYNC_LOOKUP in s]
    assert len(sync_lookup_pools) == 1, (
        f"Expected exactly one pool with SYNC_LOOKUP, got {sync_lookup_pools}"
    )
    assert sync_lookup_pools[0] == {RequestType.SYNC_LOOKUP}, (
        f"SYNC_LOOKUP must be on its own pool, but pool has {sync_lookup_pools[0]}"
    )
    # Pool size must come from the config knob.
    assert sync_lookup_max_workers == 4

    # And the shared pool must still cover the other normal ops.
    shared_pool = next(
        (s for s in pool_request_sets if RequestType.SYNC_LOOKUP not in s), None
    )
    assert shared_pool is not None
    for rt in (
        RequestType.LOOKUP,
        RequestType.QUERY_PREFETCH_STATUS,
        RequestType.END_SESSION,
        RequestType.PING,
    ):
        assert rt in shared_pool, f"{rt} should remain on the shared normal pool"


# ============================================================================
# end_session fallback cleanup test
# ============================================================================


def test_end_session_drains_lingering_prefetch_job() -> None:
    """end_session must pop a lingering _prefetch_jobs entry and drain it.

    Worker-crash / cancellation scenario: SYNC_LOOKUP returned a non-zero
    count, the worker never issued RETRIEVE, and the prefetch job is
    still registered.  end_session must clean it up or the prefetch
    controller's _completed_results leaks per request.
    """
    engine = MagicMock()
    engine._event_bus = MagicMock()
    engine._prefetch_job_lock = threading.Lock()
    engine._prefetch_jobs = {}
    engine.session_manager.remove.return_value = None  # no session state

    # Install a lingering job.
    job = _install_prefetch_job(engine, "req-linger", world_size=1)

    MPCacheEngine.end_session(engine, "req-linger")

    # Job removed and drained.
    assert "req-linger" not in engine._prefetch_jobs
    engine.storage_manager.query_prefetch_status.assert_called_once_with(job.handle)


def test_end_session_no_lingering_job_skips_drain() -> None:
    """When no prefetch job exists, end_session must not call query_prefetch_status."""
    engine = MagicMock()
    engine._event_bus = MagicMock()
    engine._prefetch_job_lock = threading.Lock()
    engine._prefetch_jobs = {}
    engine.session_manager.remove.return_value = None

    MPCacheEngine.end_session(engine, "req-nothing")

    engine.storage_manager.query_prefetch_status.assert_not_called()


# ============================================================================
# Scheduler adapter tests
# ============================================================================


def _make_scheduler_adapter() -> tuple[
    "LMCacheMPSyncLookupSchedulerAdapter", MagicMock
]:
    """Construct a minimally-initialized SyncLookup scheduler adapter."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSyncLookupSchedulerAdapter,
        ParallelStrategy,
    )

    adapter = LMCacheMPSyncLookupSchedulerAdapter.__new__(
        LMCacheMPSyncLookupSchedulerAdapter
    )
    adapter.model_name = "test_model"
    adapter.chunk_size = 256
    adapter.blocks_in_chunk = 16
    adapter.parallel_strategy = ParallelStrategy(False, 1, 0, 1, 0, 1, 1)
    adapter._health_event = threading.Event()
    adapter._health_event.set()
    adapter._mq_timeout = 30.0
    adapter._pending_lookups = set()
    adapter._finished_lookup_results = {}
    adapter._heartbeat = None
    adapter._heartbeat_lock = threading.Lock()
    adapter._heartbeat_interval = 5.0

    mock_client = MagicMock(spec=MessageQueueClient)
    adapter.mq_client = mock_client
    return adapter, mock_client


def test_scheduler_adapter_single_sync_lookup_rpc() -> None:
    """maybe_submit_lookup_request sends exactly one SYNC_LOOKUP request."""
    adapter, mock_client = _make_scheduler_adapter()

    mock_future = MagicMock()
    mock_future.result.return_value = 3  # 3 chunk hits
    mock_client.submit_request.return_value = mock_future

    token_ids = list(range(512))
    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.maybe_submit_lookup_request("req-1", token_ids)

    mock_client.submit_request.assert_called_once()
    req_type = mock_client.submit_request.call_args[0][0]
    assert req_type == RequestType.SYNC_LOOKUP

    # Token count is cached for check_lookup_result.
    assert adapter._finished_lookup_results["req-1"] == 3 * 256


def test_scheduler_adapter_check_lookup_result_returns_cached() -> None:
    """check_lookup_result returns the cached SYNC_LOOKUP count without RPCs."""
    adapter, mock_client = _make_scheduler_adapter()
    mock_future = MagicMock()
    mock_future.result.return_value = 5
    mock_client.submit_request.return_value = mock_future

    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.maybe_submit_lookup_request("req-1", list(range(512)))

    mock_client.submit_request.reset_mock()

    # Repeated checks should be pure reads.
    for _ in range(3):
        assert adapter.check_lookup_result("req-1") == 5 * 256
    mock_client.submit_request.assert_not_called()


def test_scheduler_adapter_check_lookup_result_unknown_returns_zero() -> None:
    """Unknown request_ids must yield 0 (not None) so callers don't spin."""
    adapter, _mock_client = _make_scheduler_adapter()
    assert adapter.check_lookup_result("never-submitted") == 0


def test_scheduler_adapter_unhealthy_skips_rpc() -> None:
    """An unhealthy server short-circuits maybe_submit_lookup_request."""
    adapter, mock_client = _make_scheduler_adapter()
    adapter._health_event.clear()

    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.maybe_submit_lookup_request("req-1", list(range(512)))
    mock_client.submit_request.assert_not_called()


def test_scheduler_adapter_timeout_clears_pending() -> None:
    """SYNC_LOOKUP timeout marks server unhealthy and cleans up pending set."""
    adapter, mock_client = _make_scheduler_adapter()
    mock_future = MagicMock()
    mock_future.result.side_effect = TimeoutError()
    mock_client.submit_request.return_value = mock_future

    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.maybe_submit_lookup_request("req-1", list(range(512)))

    assert "req-1" not in adapter._pending_lookups
    assert not adapter._health_event.is_set()
    assert adapter.check_lookup_result("req-1") == 0


# ============================================================================
# Worker adapter tests
# ============================================================================


def _make_worker_adapter() -> tuple["LMCacheMPSyncLookupWorkerAdapter", MagicMock]:
    """Construct a minimally-initialized SyncLookup worker adapter."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSyncLookupWorkerAdapter,
        ParallelStrategy,
    )

    adapter = LMCacheMPSyncLookupWorkerAdapter.__new__(LMCacheMPSyncLookupWorkerAdapter)
    adapter.model_name = "test_model"
    adapter.instance_id = 123
    adapter.parallel_strategy = ParallelStrategy(False, 1, 0, 1, 0, 1, 1)
    adapter.blocks_in_chunk = 16
    adapter._health_event = threading.Event()
    adapter._health_event.set()
    adapter._mq_timeout = 30.0
    adapter._heartbeat = None
    adapter._heartbeat_lock = threading.Lock()
    adapter._heartbeat_interval = 5.0

    adapter.store_futures = {}
    adapter.retrieve_futures = {}
    adapter.error_block_ids = set()
    adapter.finished_stores = set()
    adapter.previously_finished = set()
    adapter._returned_finished = set()

    # SyncLookup-specific state.
    adapter._pending_retrieves = {}
    adapter._pending_retrieves_lock = threading.Lock()
    adapter._prefetch_poll_stop = threading.Event()
    adapter._prefetch_poll_thread = None
    adapter._error_block_ids_lock = threading.Lock()

    mock_client = MagicMock(spec=MessageQueueClient)
    adapter.mq_client = mock_client
    return adapter, mock_client


def test_worker_adapter_submit_retrieve_queues_instead_of_sending() -> None:
    """submit_retrieve_request buffers the op; no RPC is sent synchronously."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    adapter, mock_client = _make_worker_adapter()

    op = LoadStoreOp(token_ids=list(range(256)), block_ids=[1, 2, 3], start=0, end=256)
    event = MagicMock()

    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.submit_retrieve_request("req-1", op, event)

    # No RETRIEVE/QUERY_* RPC yet — the retrieve is queued.
    mock_client.submit_request.assert_not_called()
    assert "req-1" in adapter._pending_retrieves


def test_worker_adapter_submit_retrieve_unhealthy_marks_error() -> None:
    """Unhealthy server: retrieve is not queued; block ids become errors."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    adapter, mock_client = _make_worker_adapter()
    adapter._health_event.clear()

    op = LoadStoreOp(token_ids=list(range(256)), block_ids=[5, 6, 7], start=0, end=256)
    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.submit_retrieve_request("req-unhealthy", op, MagicMock())

    assert adapter._pending_retrieves == {}
    assert adapter.error_block_ids == {5, 6, 7}
    mock_client.submit_request.assert_not_called()


def test_worker_adapter_dispatches_retrieve_after_prefetch_ready() -> None:
    """Background loop iteration: non-None QUERY_PREFETCH_STATUS -> RETRIEVE dispatched.

    Runs one iteration of _prefetch_ready_loop manually (no real thread)
    by stopping the loop after the first RETRIEVE dispatch.
    """
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    adapter, mock_client = _make_worker_adapter()

    op = LoadStoreOp(token_ids=list(range(256)), block_ids=[1, 2, 3], start=0, end=256)
    event = MagicMock()
    event.ipc_handle.return_value = b"ipc"
    adapter._pending_retrieves["req-1"] = (op, event, "")

    # QUERY_PREFETCH_STATUS -> positive count (ready).
    query_future = MagicMock()
    query_future.result.return_value = 3
    # RETRIEVE future (what base class submit_retrieve_request will request).
    retrieve_future = MagicMock()

    def submit_request_side_effect(req_type, payload, response_cls=None):  # noqa: ARG001
        if req_type == RequestType.QUERY_PREFETCH_STATUS:
            return query_future
        return retrieve_future

    mock_client.submit_request.side_effect = submit_request_side_effect

    # Stop the loop after one tick by having the wait() immediately return.
    adapter._prefetch_poll_stop.set()
    # But we still want the body to execute once — patch the set() to do so.
    adapter._prefetch_poll_stop.clear()

    def one_shot_wait(timeout: float) -> bool:  # noqa: ARG001
        adapter._prefetch_poll_stop.set()
        return True

    with patch.object(adapter._prefetch_poll_stop, "wait", side_effect=one_shot_wait):
        adapter._prefetch_ready_loop()

    # Retrieve was dispatched and pending buffer drained.
    assert adapter._pending_retrieves == {}
    request_types = [c[0][0] for c in mock_client.submit_request.call_args_list]
    assert RequestType.QUERY_PREFETCH_STATUS in request_types
    assert RequestType.RETRIEVE in request_types


def test_worker_adapter_get_finished_drains_on_unhealthy() -> None:
    """Unhealthy get_finished drains pending retrieves into error_block_ids."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    adapter, _mock_client = _make_worker_adapter()

    op = LoadStoreOp(token_ids=list(range(256)), block_ids=[7, 8, 9], start=0, end=256)
    adapter._pending_retrieves["req-drained"] = (op, MagicMock(), "")
    adapter._health_event.clear()

    adapter.get_finished(set())

    assert "req-drained" not in adapter._pending_retrieves
    assert adapter.error_block_ids == {7, 8, 9}


def test_worker_adapter_error_block_ids_lock_is_held() -> None:
    """get_block_ids_with_load_errors holds _error_block_ids_lock across copy+clear.

    Directly proves the lock is held by seeing that a thread blocked
    inside the lock cannot interleave with the drain.
    """
    adapter, _mock_client = _make_worker_adapter()
    adapter.error_block_ids = {1, 2, 3}

    # Hold the lock from another thread, then call get_block_ids_with_load_errors
    # with a short deadline — it should block until we release.
    blocked = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with adapter._error_block_ids_lock:
            blocked.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    blocked.wait(timeout=2.0)

    # The main thread can't acquire the lock yet, so get_block_ids_with_load_errors
    # would block. Use a short timer to assert that behavior.
    done = threading.Event()
    result: list[set[int]] = []

    def drain() -> None:
        result.append(adapter.get_block_ids_with_load_errors())
        done.set()

    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    assert not done.wait(timeout=0.1), (
        "get_block_ids_with_load_errors should block while the lock is held"
    )

    release.set()
    holder.join(timeout=2.0)
    assert done.wait(timeout=2.0)
    assert result == [{1, 2, 3}]
    assert adapter.error_block_ids == set()
