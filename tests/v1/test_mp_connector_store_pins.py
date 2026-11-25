# SPDX-License-Identifier: Apache-2.0
"""Sliding-window / recurrent blocks stay pinned until every worker reports
the STORE that reads them."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import functools
import itertools
import threading

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.config import KVTransferConfig, VllmConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.v1 import kv_cache_interface  # noqa: E402
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_mod  # noqa: E402
from lmcache.integration.vllm import (  # noqa: E402
    vllm_multi_process_adapter as adapter_mod,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    STORE_PIN_HINT,
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
)
from lmcache.v1.client.transport.errors import REQUEST_FAILURES  # noqa: E402
from lmcache.v1.common.group_view import EngineGroupInfo  # noqa: E402
from lmcache.v1.transport.futures import DeviceMessagingFuture  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator

CHUNK = 4


# ------------------------------------------------------------------ #
#  Scheduler helpers                                                   #
# ------------------------------------------------------------------ #


def _scheduler(workers: int = 1) -> tuple[LMCacheMPConnector, MagicMock]:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    pool = MagicMock()
    pool.blocks = {i: f"block-{i}" for i in range(100)}
    connector._gpu_block_pool = pool
    # The adapter's own receipt counter, as lazy offload uses it.
    counter = SimpleNamespace(
        _store_request_pending_counts={}, _expected_worker_count=workers
    )
    connector.scheduler_adapter = MagicMock(
        lmcache_tokens_per_chunk=8,
        update_pending_store_count=functools.partial(
            LMCacheMPSchedulerAdapter.update_pending_store_count, cast(Any, counter)
        ),
    )
    # Group 0: full attention, 8 tokens/block. Group 1: sliding window of 3
    # tokens over 2-token blocks (2 window blocks). Group 2: Mamba, 2-token
    # blocks, one state block per chunk.
    connector._group_tokens_per_block = [8, 2, 2]
    connector._windowed_group_tokens = {1: 3, 2: 2}
    connector._store_pins = {}
    connector._store_pin_seq = 0
    connector._warned_unpinned_stores = False
    return connector, pool


def _store(
    start: int, end: int, block_ids: list[list[int]]
) -> LMCacheMPRequestMetadata:
    return LMCacheMPRequestMetadata(
        request_id="req-0",
        direction="STORE",
        op=LoadStoreOp(token_bytes=b"", block_ids=block_ids, start=start, end=end),
        request_configs={"lmcache.tag.x": "y"},
    )


def test_scheduler_pins_the_window_blocks_of_each_full_chunk() -> None:
    connector, pool = _scheduler()
    # Two full chunks; group 2's first chunk ends on a null state block.
    r_meta = _store(
        0, 16, [[1, 2], [10, 11, 12, 13, 14, 15, 16, 17], [0, 0, 0, 0, 0, 0, 0, 27]]
    )

    connector._pin_store_blocks(r_meta)

    pinned = [12, 13, 16, 17, 27]
    pool.touch.assert_called_once_with([f"block-{i}" for i in pinned])
    assert r_meta.request_configs is not None
    key = r_meta.request_configs[STORE_PIN_HINT]
    assert r_meta.request_configs["lmcache.tag.x"] == "y"
    assert connector._store_pins == {key: pinned}


def test_scheduler_pins_the_unfull_window_with_its_lead_block() -> None:
    connector, pool = _scheduler()
    # Full chunk [0, 8) plus unfull [8, 14): group 1 blocks 4..6 of the tail.
    r_meta = _store(
        0, 14, [[1, 2], [10, 11, 12, 13, 20, 21, 22], [0, 0, 0, 7, 0, 0, 9]]
    )

    connector._pin_store_blocks(r_meta)

    (blocks,), _ = pool.touch.call_args
    assert blocks == [f"block-{i}" for i in [7, 9, 12, 13, 20, 21, 22]]


def test_scheduler_does_not_pin_without_windowed_groups() -> None:
    connector, pool = _scheduler()
    connector._windowed_group_tokens = {}
    r_meta = _store(0, 8, [[1], [10, 11, 12, 13], [0, 0, 0, 7]])

    connector._pin_store_blocks(r_meta)

    pool.touch.assert_not_called()
    assert STORE_PIN_HINT not in (r_meta.request_configs or {})


def test_scheduler_stores_unpinned_without_a_block_pool() -> None:
    # vLLM before bind_gpu_block_pool: nothing to pin with.
    connector, _ = _scheduler()
    connector._gpu_block_pool = None
    first = _store(0, 8, [[1], [10, 11, 12, 13], [0, 0, 0, 7]])
    second = _store(8, 16, [[2], [14, 15, 16, 17], [0, 0, 0, 9]])

    connector._pin_store_blocks(first)
    connector._pin_store_blocks(second)

    assert connector._store_pins == {}
    assert STORE_PIN_HINT not in (first.request_configs or {})
    assert STORE_PIN_HINT not in (second.request_configs or {})
    assert connector._warned_unpinned_stores


def test_scheduler_does_not_pin_a_store_of_null_blocks() -> None:
    connector, pool = _scheduler()
    r_meta = _store(0, 8, [[1], [0, 0, 0, 0], [0, 0, 0, 0]])

    connector._pin_store_blocks(r_meta)

    pool.touch.assert_not_called()
    assert connector._store_pins == {}
    assert STORE_PIN_HINT not in (r_meta.request_configs or {})


def test_pins_are_released_once_every_worker_has_reported() -> None:
    connector, pool = _scheduler(workers=2)
    r_meta = _store(0, 8, [[1], [10, 11, 12, 13], [0, 0, 0, 7]])
    connector._pin_store_blocks(r_meta)
    assert r_meta.request_configs is not None
    key = r_meta.request_configs[STORE_PIN_HINT]

    # One of two workers, and a key that is not pinned.
    connector._release_reported_store_pins({key: 1, "other#9": 2})
    pool.free_blocks.assert_not_called()
    assert key in connector._store_pins

    connector._release_reported_store_pins({key: 1})
    pool.free_blocks.assert_called_once_with([f"block-{i}" for i in [7, 12, 13]])
    assert connector._store_pins == {}
    adapter: Any = connector.scheduler_adapter
    counts = adapter.update_pending_store_count.args[0]._store_request_pending_counts
    assert counts == {}

    # A late duplicate report is ignored.
    connector._release_reported_store_pins({key: 1})
    pool.free_blocks.assert_called_once()


# ------------------------------------------------------------------ #
#  Worker helpers                                                      #
# ------------------------------------------------------------------ #


def _pending_future() -> MagicMock:
    future = MagicMock()
    future.query.return_value = False
    return future


def _store_metadata(pin_key: str | None) -> LMCacheMPConnectorMetadata:
    metadata = LMCacheMPConnectorMetadata()
    metadata.add_request_metadata(
        cast(
            Any,
            SimpleNamespace(
                request_id="req-0",
                direction="STORE",
                op=MagicMock(),
                cache_salt="",
                request_configs=(
                    {STORE_PIN_HINT: pin_key} if pin_key is not None else None
                ),
            ),
        )
    )
    return metadata


def _worker(
    pin_key: str | None = "req-0#0", *, healthy: bool = True, is_kv_writer: bool = True
) -> tuple[LMCacheMPConnector, MagicMock]:
    """A worker bound to one STORE; a real adapter whose submit creates
    ``future`` (and then the next futures ``submit_store`` is given)."""
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.dispatcher = None
    connector.lazy_offload = False
    connector._unsubmitted_store_pins = set()
    connector._saved_metadata = None
    future = _pending_future()
    adapter: Any = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.lazy_offload = False
    adapter.dispatcher = None
    adapter._health_event = threading.Event()
    if healthy:
        adapter._health_event.set()
    # A non-None sentinel makes _ensure_heartbeat_started a no-op.
    adapter._heartbeat = object()
    # Rank 1 of two replicas holding only a replicated group is not a writer.
    adapter.engine_group_infos = [EngineGroupInfo(0, (0,), tp_replicas=2)]
    adapter.parallel_strategy = SimpleNamespace(kv_worker_id=0 if is_kv_writer else 1)
    adapter.store_futures = {}
    adapter.store_events = {}
    adapter._completed_store_requests = {}
    adapter._failed_store_requests = set()
    adapter._pinned_store_futures = {}
    adapter.kv_caches = MagicMock()
    adapter.blocks_in_chunk = 1
    adapter._create_key = MagicMock()
    adapter._block_ids_per_group = MagicMock()
    adapter.create_recorded_event = MagicMock(return_value=None)
    adapter.transfer_ctx = MagicMock()
    adapter.transfer_ctx.submit_store.return_value = future
    adapter._require_transfer_ctx = MagicMock(return_value=adapter.transfer_ctx)
    connector.worker_adapter = adapter
    connector._connector_metadata = _store_metadata(pin_key)
    return connector, future


def _reported(connector: LMCacheMPConnector) -> dict[str, int]:
    meta = connector.build_connector_worker_meta()
    return {} if meta is None else meta.completed_store_requests


def test_worker_reports_a_pinned_store_once_its_future_ends() -> None:
    connector, future = _worker()

    connector.wait_for_save()
    # The worker never blocks on the store.
    future.wait.assert_not_called()
    future.result.assert_not_called()
    transfer_ctx: Any = connector.worker_adapter.transfer_ctx
    transfer_ctx.submit_store.assert_called_once()
    transfer_ctx.flush_inflight_stores.assert_not_called()
    assert _reported(connector) == {}
    assert _reported(connector) == {}

    future.query.return_value = True
    assert _reported(connector) == {"req-0#0": 1}
    assert _reported(connector) == {}


def test_answered_store_is_not_waited_for_while_its_copy_runs() -> None:
    # The server answers before its copy is done. The worker keeps polling
    # and never synchronizes on the copy event inside the engine step.
    connector, _ = _worker()
    future = MagicMock(spec=DeviceMessagingFuture)
    future.raw_future_ = MagicMock()
    future.query.return_value = False
    future.raw_future_.query.return_value = True
    adapter: Any = connector.worker_adapter
    adapter.transfer_ctx.submit_store.return_value = future

    connector.wait_for_save()
    for _ in range(5):
        assert _reported(connector) == {}
    future.result.assert_not_called()
    future.wait.assert_not_called()

    future.query.return_value = True
    assert _reported(connector) == {"req-0#0": 1}
    assert _reported(connector) == {}


def test_worker_reports_a_failed_pinned_store() -> None:
    # A failed request has ended: the server no longer reads the blocks.
    connector, future = _worker()
    connector.wait_for_save()
    future.query.side_effect = REQUEST_FAILURES[0]("server went away")

    assert _reported(connector) == {"req-0#0": 1}
    assert _reported(connector) == {}


def test_worker_does_not_report_an_unpinned_store() -> None:
    connector, future = _worker(pin_key=None)

    connector.wait_for_save()
    future.query.return_value = True

    adapter: Any = connector.worker_adapter
    adapter.transfer_ctx.submit_store.assert_called_once()
    assert connector.build_connector_worker_meta() is None
    assert adapter._pinned_store_futures == {}


@pytest.mark.parametrize(
    ("healthy", "is_kv_writer"),
    [(True, False), (False, True)],
    ids=["non-writer", "lost"],
)
def test_worker_reports_a_pinned_store_it_did_not_submit(
    healthy: bool, is_kv_writer: bool
) -> None:
    # A non-writer rank or an unhealthy server creates no store future.
    connector, _ = _worker(healthy=healthy, is_kv_writer=is_kv_writer)

    connector.wait_for_save()

    adapter: Any = connector.worker_adapter
    adapter.transfer_ctx.submit_store.assert_not_called()
    assert _reported(connector) == {"req-0#0": 1}
    assert _reported(connector) == {}


def test_worker_reports_pinned_stores_of_a_lost_server() -> None:
    connector, future = _worker()
    connector.wait_for_save()
    adapter: Any = connector.worker_adapter
    adapter._health_event.clear()

    assert _reported(connector) == {"req-0#0": 1}
    future.query.assert_not_called()
    assert _reported(connector) == {}


def test_worker_reports_each_pinned_store_of_one_request() -> None:
    # A request has several STOREs in flight without lazy offload, so a
    # receipt per request id could not tell them apart.
    connector, first = _worker()
    second = _pending_future()
    adapter: Any = connector.worker_adapter
    adapter.transfer_ctx.submit_store.side_effect = [first, second]

    connector.wait_for_save()
    connector._connector_metadata = _store_metadata("req-0#1")
    connector.wait_for_save()
    assert _reported(connector) == {}

    second.query.return_value = True
    assert _reported(connector) == {"req-0#1": 1}
    first.query.return_value = True
    assert _reported(connector) == {"req-0#0": 1}
    assert _reported(connector) == {}


def test_deferred_finalize_does_not_report_before_the_store_is_submitted() -> None:
    # With a draft model vLLM builds the worker metadata before wait_for_save.
    connector, future = _worker()

    assert _reported(connector) == {}
    connector.wait_for_save()
    connector._connector_metadata = LMCacheMPConnectorMetadata()
    assert _reported(connector) == {}

    future.query.return_value = True
    assert _reported(connector) == {"req-0#0": 1}


def test_worker_reports_a_pinned_store_whose_step_skipped_wait_for_save() -> None:
    # vLLM before vllm-project/vllm#57775 skips wait_for_save on a step
    # without a forward; the STORE of such a step is never submitted.
    connector, _ = _worker()

    assert _reported(connector) == {}
    connector._connector_metadata = LMCacheMPConnectorMetadata()
    assert _reported(connector) == {"req-0#0": 1}
    assert _reported(connector) == {}
    adapter: Any = connector.worker_adapter
    adapter.transfer_ctx.submit_store.assert_not_called()


def test_skipped_store_is_reported_when_the_next_step_saves_first() -> None:
    # Same, on a vLLM that calls wait_for_save before it builds the worker
    # metadata: the next step's own STORE is submitted, the skipped one is not.
    connector, future = _worker()

    assert _reported(connector) == {}
    connector._connector_metadata = _store_metadata("req-0#1")
    connector.wait_for_save()
    assert _reported(connector) == {"req-0#0": 1}
    assert _reported(connector) == {}

    future.query.return_value = True
    assert _reported(connector) == {"req-0#1": 1}


# ------------------------------------------------------------------ #
#  Wiring through the public connector API                             #
# ------------------------------------------------------------------ #


def _config(world_size: int = 1) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(model="test-model", use_mla=False),
            parallel_config=SimpleNamespace(
                world_size=world_size,
                rank=0,
                tensor_parallel_size=world_size,
                pipeline_parallel_size=1,
            ),
            cache_config=SimpleNamespace(block_size=CHUNK, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheMPConnector", kv_role="kv_both"
            ),
        ),
    )


def _request(request_id: str = "request") -> Request:
    return cast(
        Request,
        SimpleNamespace(
            request_id=request_id,
            cache_salt="",
            lora_request=None,
            all_token_ids=list(range(12)),
            num_prompt_tokens=12,
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            kv_transfer_params=None,
        ),
    )


def _schedule(
    num_tokens: int = 0,
    block_ids: tuple[list[int], ...] = (),
    *,
    new: bool = False,
    num_computed_tokens: int = 0,
    request_id: str = "request",
) -> SchedulerOutput:
    scheduled = bool(num_tokens)
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            scheduled_new_reqs=(
                [
                    SimpleNamespace(
                        req_id=request_id,
                        block_ids=block_ids,
                        num_computed_tokens=num_computed_tokens,
                    )
                ]
                if new and scheduled
                else []
            ),
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[request_id] if not new and scheduled else [],
                new_block_ids=[block_ids] if not new and scheduled else [],
                resumed_req_ids=set(),
                num_computed_tokens=(
                    [num_computed_tokens] if not new and scheduled else []
                ),
            ),
            num_scheduled_tokens={request_id: num_tokens} if scheduled else {},
            total_num_scheduled_tokens=num_tokens,
        ),
    )


@pytest.fixture
def mock_io(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stub transport, telemetry and GPU resources, keeping connector logic real."""
    scheduler = MagicMock(lmcache_tokens_per_chunk=CHUNK)
    scheduler.check_lookup_result.return_value = 0
    transfer = MagicMock()
    transfer.create_recorded_event.return_value = None
    # One future per submitted STORE, unfinished until a test ends it.
    futures: list[MagicMock] = []

    def submit_store(*args: Any, **kwargs: Any) -> MagicMock:
        future = MagicMock()
        future.query.return_value = False
        future.result.return_value = True
        futures.append(future)
        return future

    transfer.submit_store.side_effect = submit_store
    pool = MagicMock(blocks=[f"block-{i}" for i in range(8)])
    monkeypatch.setattr(
        connector_mod, "LMCacheMPSchedulerAdapter", lambda **kw: scheduler
    )
    monkeypatch.setattr(connector_mod, "print_banner_once", MagicMock())
    monkeypatch.setattr(connector_mod, "vllm_layout_hints", lambda config: {})
    monkeypatch.setattr(adapter_mod.RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: CHUNK)
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", MagicMock())
    monkeypatch.setattr(adapter_mod, "set_ipc_policy", MagicMock())
    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", lambda *a, **kw: transfer
    )
    monkeypatch.setattr(
        adapter_mod.RequestTelemetryFactory, "create", lambda **kw: MagicMock()
    )
    return SimpleNamespace(
        transfer=transfer,
        futures=futures,
        pool=pool,
        kv_caches={"layer": torch.empty(2, 8, CHUNK, 1, 8)},
    )


@pytest.fixture
def connectors(
    mock_io: SimpleNamespace,
) -> Iterator[tuple[LMCacheMPConnector, LMCacheMPConnector]]:
    """Real scheduler and worker connectors with the default configuration.

    The single group is made a recurrent-state group of one block per chunk,
    as ``__init__`` derives it from a Mamba or sliding-window cache group.
    """
    config = _config()
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    worker = LMCacheMPConnector(config, KVConnectorRole.WORKER)
    scheduler._windowed_group_tokens = {0: CHUNK}
    scheduler.bind_gpu_block_pool(mock_io.pool)
    worker.register_kv_caches(mock_io.kv_caches)
    try:
        yield scheduler, worker
    finally:
        worker.shutdown()
        scheduler.shutdown()


def _run_worker_step(
    worker: LMCacheMPConnector, metadata: Any, *, forward: bool = True
) -> KVConnectorOutput:
    """One worker step in vLLM's order; ``forward=False`` skips wait_for_save
    like a vLLM before vllm-project/vllm#57775 on a step without a forward."""
    worker.bind_connector_metadata(metadata)
    if forward:
        worker.wait_for_save()
    worker.get_finished(set())
    output = KVConnectorOutput(
        kv_connector_worker_meta=worker.build_connector_worker_meta()
    )
    worker.clear_connector_metadata()
    return output


def _start_request(
    scheduler: LMCacheMPConnector, block_ids: list[int], request_id: str = "request"
) -> None:
    request = _request(request_id)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: (block_ids,)), 0
    )


def test_store_blocks_stay_pinned_until_the_worker_reports_the_store(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    mock_io: SimpleNamespace,
) -> None:
    scheduler, worker = connectors
    pool = mock_io.pool
    _start_request(scheduler, [1])

    # Step 1 (new request) stores chunk 0 from block 1.
    metadata = scheduler.build_connector_meta(_schedule(CHUNK, ([1],), new=True))
    (store,) = metadata.requests
    assert store.direction == "STORE"
    assert STORE_PIN_HINT in store.request_configs
    pool.touch.assert_called_once_with(["block-1"])
    assert scheduler.has_pending_push_work()
    scheduler.update_connector_output(_run_worker_step(worker, metadata))
    mock_io.transfer.submit_store.assert_called_once()
    transfer_ctx: Any = worker.worker_adapter.transfer_ctx
    transfer_ctx.flush_inflight_stores.assert_not_called()

    # Step 2 (running request) stores chunk 1 from block 2.
    metadata = scheduler.build_connector_meta(
        _schedule(CHUNK, ([2],), num_computed_tokens=CHUNK)
    )
    assert pool.touch.call_args_list[-1].args == (["block-2"],)
    scheduler.update_connector_output(_run_worker_step(worker, metadata))

    # Connector-only steps: the stores are still running, so no step count
    # releases the pins.
    for _ in range(4):
        metadata = scheduler.build_connector_meta(_schedule())
        scheduler.update_connector_output(
            _run_worker_step(worker, metadata, forward=False)
        )
    pool.free_blocks.assert_not_called()
    assert scheduler.has_pending_push_work()

    # The first store ends: only its block is released.
    first, second = mock_io.futures
    first.query.return_value = True
    scheduler.update_connector_output(
        _run_worker_step(worker, scheduler.build_connector_meta(_schedule()))
    )
    pool.free_blocks.assert_called_once_with(["block-1"])
    assert scheduler.has_pending_push_work()

    second.query.return_value = True
    scheduler.update_connector_output(
        _run_worker_step(worker, scheduler.build_connector_meta(_schedule()))
    )
    assert pool.free_blocks.call_args_list[-1].args == (["block-2"],)
    assert pool.free_blocks.call_count == 2
    assert not scheduler.has_pending_push_work()
    assert scheduler._store_pins == {}


@pytest.mark.parametrize("finalizes_steps_without_a_forward", [True, False])
def test_no_step_pattern_releases_a_pin_before_its_store_ends(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    mock_io: SimpleNamespace,
    finalizes_steps_without_a_forward: bool,
) -> None:
    """A STORE, then five steps with or without a forward in all 32 orders,
    with the store ending before each of them or after all of them.

    A vLLM before vllm-project/vllm#57775 skips ``wait_for_save`` on a step
    without a forward; a later one calls it on every step.
    """
    scheduler, worker = connectors
    pool = mock_io.pool
    runs = 0
    for pattern in itertools.product([True, False], repeat=5):
        for ends_before_step in range(6):
            request_id = f"request-{runs}"
            runs += 1
            _start_request(scheduler, [1], request_id)
            metadata = scheduler.build_connector_meta(
                _schedule(CHUNK, ([1],), new=True, request_id=request_id)
            )
            scheduler.update_connector_output(_run_worker_step(worker, metadata))
            future = mock_io.futures[-1]
            pool.free_blocks.reset_mock()

            for step, forward in enumerate(pattern):
                if step == ends_before_step:
                    future.query.return_value = True
                output = _run_worker_step(
                    worker,
                    scheduler.build_connector_meta(_schedule()),
                    forward=forward or finalizes_steps_without_a_forward,
                )
                scheduler.update_connector_output(output)
                # Pinned while the store runs, released in the step it ends.
                assert pool.free_blocks.called is (step >= ends_before_step)
                assert scheduler.has_pending_push_work() is (step < ends_before_step)

            future.query.return_value = True
            scheduler.update_connector_output(
                _run_worker_step(worker, scheduler.build_connector_meta(_schedule()))
            )
            pool.free_blocks.assert_called_once_with(["block-1"])
            assert scheduler._store_pins == {}
    assert runs == 192


def test_request_finished_does_not_release_a_running_store_pin(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    mock_io: SimpleNamespace,
) -> None:
    scheduler, worker = connectors
    _start_request(scheduler, [1])
    metadata = scheduler.build_connector_meta(_schedule(CHUNK, ([1],), new=True))
    scheduler.update_connector_output(_run_worker_step(worker, metadata))

    scheduler.request_finished(_request(), [1])
    scheduler.update_connector_output(
        _run_worker_step(worker, scheduler.build_connector_meta(_schedule()))
    )
    mock_io.pool.free_blocks.assert_not_called()
    assert scheduler.has_pending_push_work()

    mock_io.futures[0].query.return_value = True
    scheduler.update_connector_output(
        _run_worker_step(worker, scheduler.build_connector_meta(_schedule()))
    )
    mock_io.pool.free_blocks.assert_called_once_with(["block-1"])
    assert not scheduler.has_pending_push_work()


def test_models_without_windowed_groups_pin_nothing(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    mock_io: SimpleNamespace,
) -> None:
    scheduler, worker = connectors
    scheduler._windowed_group_tokens = {}
    _start_request(scheduler, [1])

    metadata = scheduler.build_connector_meta(_schedule(CHUNK, ([1],), new=True))

    (store,) = metadata.requests
    assert STORE_PIN_HINT not in (store.request_configs or {})
    mock_io.pool.touch.assert_not_called()
    assert not scheduler.has_pending_push_work()
    output = _run_worker_step(worker, metadata)
    assert output.kv_connector_worker_meta is None


def test_init_finds_the_sliding_window_groups(mock_io: SimpleNamespace) -> None:
    if hasattr(kv_cache_interface, "KVCacheGroup"):
        pytest.skip("this vLLM declares the window on the group's manager")
    full = kv_cache_interface.FullAttentionSpec(
        block_size=CHUNK, num_kv_heads=1, head_size=8, dtype=torch.float32
    )
    windowed = kv_cache_interface.SlidingWindowSpec(
        block_size=CHUNK,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=2 * CHUNK,
    )
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            kv_cache_interface.KVCacheGroupSpec(["layer0"], full),
            kv_cache_interface.KVCacheGroupSpec(["layer1"], windowed),
        ],
    )

    scheduler = LMCacheMPConnector(_config(), KVConnectorRole.SCHEDULER, cache_config)

    try:
        assert scheduler._windowed_group_tokens == {1: 2 * CHUNK}
    finally:
        scheduler.shutdown()
