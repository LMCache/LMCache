# SPDX-License-Identifier: Apache-2.0
"""Lifecycle and metric tests for paged-Q capture and asynchronous stores."""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.sdk.qringbuffer import (
    QRingBuffer,
    QRingBufferAdapter,
    QRingBufferCapture,
)


@dataclass
class _FakeKey:
    """Minimal cache-key stand-in accepted by dataclasses.replace."""

    model_name: str


@dataclass
class _FakeRequest:
    """Minimal connector request metadata."""

    request_id: str
    direction: str
    op: LoadStoreOp
    cache_salt: str = ""


class _FakeFuture:
    """Controllable MessagingFuture stand-in."""

    def __init__(
        self,
        result: bool = True,
        ready: bool = True,
        error: Exception | None = None,
        query_error: Exception | None = None,
    ) -> None:
        self.ready = ready
        self._result = result
        self._error = error
        self._query_error = query_error

    def query(self) -> bool:
        """Return whether the synthetic operation is terminal."""
        if self._query_error is not None:
            raise self._query_error
        return self.ready

    def result(self) -> bool:
        """Return or raise the configured terminal result."""
        if self._error is not None:
            raise self._error
        return self._result


class _FakeTransferContext:
    """Transfer context returning queued futures or a configured error."""

    def __init__(self, outcomes: list[_FakeFuture | Exception]) -> None:
        self._outcomes = outcomes

    def submit_q_store(self, *args, **kwargs) -> _FakeFuture:
        """Return the next synthetic submission outcome."""
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeWorkerAdapter:
    """Worker state used by QRingBufferAdapter tests."""

    def __init__(self, transfer_ctx: _FakeTransferContext | None) -> None:
        self.transfer_ctx = transfer_ctx
        self.is_healthy = True
        self.is_kv_writer = True
        self.instance_id = 7
        self.blocks_in_chunk = 1
        self.heartbeat_started = 0

    def _ensure_heartbeat_started(self) -> None:
        self.heartbeat_started += 1

    def _create_key(self, *args, **kwargs) -> _FakeKey:
        return _FakeKey(model_name="base")


def _ring(num_blocks: int = 8) -> QRingBuffer:
    return QRingBuffer(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=4,
        hidden_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _adapter(
    outcomes: list[_FakeFuture | Exception], num_blocks: int = 8
) -> tuple[QRingBufferAdapter, _FakeWorkerAdapter]:
    worker = _FakeWorkerAdapter(_FakeTransferContext(outcomes))
    adapter = QRingBufferAdapter(cast(Any, worker), "model##query")
    adapter.q_ring = _ring(num_blocks)
    return adapter, worker


def _op() -> LoadStoreOp:
    return LoadStoreOp(
        block_ids=[[0]],
        token_ids=[1, 2, 3, 4],
        start=0,
        end=4,
    )


def _allocate(adapter: QRingBufferAdapter, blocks: int) -> list[int]:
    assert adapter.q_ring is not None
    block_ids = adapter.q_ring.allocate(blocks)
    assert block_ids is not None
    return block_ids


def test_store_metrics_track_inflight_high_water_and_terminal_results() -> None:
    """Success and failure both reclaim blocks and drain in-flight gauges."""
    success = _FakeFuture(result=True, ready=False)
    failure = _FakeFuture(result=False, ready=False)
    adapter, _ = _adapter([success, failure])
    first_blocks = _allocate(adapter, 2)
    second_blocks = _allocate(adapter, 3)

    adapter.submit_q_store_request("req-ok", _op(), first_blocks, SimpleNamespace())
    adapter.submit_q_store_request("req-fail", _op(), second_blocks, SimpleNamespace())

    pending = adapter.metrics_snapshot()
    assert pending.requests_submitted == 2
    assert pending.in_flight_requests == 2
    assert pending.in_flight_blocks == 5
    assert pending.high_watermark_in_flight_requests == 2
    assert pending.high_watermark_in_flight_blocks == 5

    success.ready = True
    failure.ready = True
    adapter.reclaim_finished_q_stores()

    finished = adapter.metrics_snapshot()
    assert finished.requests_completed == 1
    assert finished.requests_failed == 1
    assert finished.blocks_submitted == 5
    assert finished.blocks_released == 5
    assert finished.in_flight_requests == 0
    assert finished.in_flight_blocks == 0
    assert finished.completion_duration_ns_total > 0
    assert finished.completion_duration_ns_max > 0
    assert adapter.q_ring is not None
    assert adapter.q_ring.num_free_blocks() == 8


def test_completion_exception_is_counted_and_reclaimed() -> None:
    """A future exception cannot strand its ring block."""
    future = _FakeFuture(error=RuntimeError("store failed"))
    adapter, _ = _adapter([future])
    blocks = _allocate(adapter, 2)
    adapter.submit_q_store_request("req", _op(), blocks, SimpleNamespace())

    adapter.reclaim_finished_q_stores()

    metrics = adapter.metrics_snapshot()
    assert metrics.requests_failed == 1
    assert metrics.in_flight_requests == 0
    assert metrics.blocks_released == 2
    assert adapter.q_ring is not None
    assert adapter.q_ring.num_free_blocks() == 8


def test_completion_query_exception_is_counted_and_reclaimed() -> None:
    """A readiness-query exception is terminal and cannot strand blocks."""
    future = _FakeFuture(query_error=RuntimeError("query failed"))
    adapter, _ = _adapter([future])
    blocks = _allocate(adapter, 2)
    adapter.submit_q_store_request("req", _op(), blocks, SimpleNamespace())

    adapter.reclaim_finished_q_stores()

    metrics = adapter.metrics_snapshot()
    assert metrics.requests_failed == 1
    assert metrics.in_flight_requests == 0
    assert metrics.blocks_released == 2
    assert adapter.q_ring is not None
    assert adapter.q_ring.num_free_blocks() == 8


def test_synchronous_submission_failure_is_counted_and_reclaimed() -> None:
    """A submit exception preserves allocator capacity before propagating."""
    adapter, _ = _adapter([RuntimeError("submit failed")])
    blocks = _allocate(adapter, 3)

    with pytest.raises(RuntimeError, match="submit failed"):
        adapter.submit_q_store_request("req", _op(), blocks, SimpleNamespace())

    metrics = adapter.metrics_snapshot()
    assert metrics.requests_submission_failed == 1
    assert metrics.requests_submitted == 0
    assert metrics.blocks_released == 3
    assert adapter.q_ring is not None
    assert adapter.q_ring.num_free_blocks() == 8


def test_pre_submit_drop_reasons_release_blocks() -> None:
    """Unavailable, unhealthy, and invalid-key paths have bounded counters."""
    adapter, worker = _adapter([])

    unavailable_blocks = _allocate(adapter, 1)
    worker.transfer_ctx = None
    adapter.submit_q_store_request(
        "unavailable", _op(), unavailable_blocks, SimpleNamespace()
    )

    worker.transfer_ctx = _FakeTransferContext([])
    worker.is_healthy = False
    unhealthy_blocks = _allocate(adapter, 2)
    adapter.submit_q_store_request(
        "unhealthy", _op(), unhealthy_blocks, SimpleNamespace()
    )

    worker.is_healthy = True
    missing_ids_blocks = _allocate(adapter, 3)
    op = _op()
    cast(Any, op).token_ids = None
    adapter.submit_q_store_request(
        "missing-token-ids", op, missing_ids_blocks, SimpleNamespace()
    )

    metrics = adapter.metrics_snapshot()
    assert metrics.requests_dropped_unavailable == 1
    assert metrics.requests_dropped_unhealthy == 1
    assert metrics.requests_dropped_missing_token_ids == 1
    assert metrics.blocks_released == 6
    assert adapter.q_ring is not None
    assert adapter.q_ring.num_free_blocks() == 8


def test_missing_forward_event_discards_capture_without_ring_leak() -> None:
    """A state built before a no-store forward exit returns every block."""
    adapter, worker = _adapter([])
    capture = QRingBufferCapture(cast(Any, worker), adapter)
    query = torch.ones((4, 8), dtype=torch.float32)
    op = _op()
    metadata = SimpleNamespace(requests=[_FakeRequest("req", "STORE", op)])
    attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    )
    capture.q_step_state = capture._build_q_step_state(
        query, cast(Any, metadata), attn_metadata
    )
    assert capture.q_step_state is not None

    capture.batched_submit_qstore_requests(event=None)

    metrics = adapter.metrics_snapshot()
    assert metrics.requests_dropped_missing_event == 1
    assert metrics.blocks_released == 1
    assert adapter.q_ring is not None
    assert adapter.q_ring.num_free_blocks() == 8


def test_batch_submit_continues_after_one_request_raises() -> None:
    """A synchronous failure cannot strand later requests in the same plan."""
    second_future = _FakeFuture(result=True)
    adapter, worker = _adapter([RuntimeError("first failed"), second_future])
    capture = QRingBufferCapture(cast(Any, worker), adapter)
    query = torch.ones((8, 8), dtype=torch.float32)
    metadata = SimpleNamespace(
        requests=[
            _FakeRequest(
                "first",
                "STORE",
                LoadStoreOp(
                    token_ids=[1, 2, 3, 4],
                    block_ids=[[0]],
                    start=0,
                    end=4,
                ),
            ),
            _FakeRequest(
                "second",
                "STORE",
                LoadStoreOp(
                    token_ids=[5, 6, 7, 8],
                    block_ids=[[1]],
                    start=0,
                    end=4,
                ),
            ),
        ]
    )
    attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor(range(8), dtype=torch.int64)
    )
    capture.q_step_state = capture._build_q_step_state(
        query, cast(Any, metadata), attn_metadata
    )
    assert capture.q_step_state is not None

    with pytest.raises(RuntimeError, match="first failed"):
        capture.batched_submit_qstore_requests(SimpleNamespace())
    adapter.reclaim_finished_q_stores()

    metrics = adapter.metrics_snapshot()
    assert metrics.requests_submission_failed == 1
    assert metrics.requests_submitted == 1
    assert metrics.requests_completed == 1
    assert metrics.blocks_released == 2
    assert adapter.q_ring is not None
    assert adapter.q_ring.num_free_blocks() == 8


def test_unhealthy_reclaim_abandons_all_inflight_stores() -> None:
    """Health loss drains every in-flight gauge and releases every block."""
    adapter, worker = _adapter([_FakeFuture(ready=False), _FakeFuture(ready=False)])
    adapter.submit_q_store_request(
        "first", _op(), _allocate(adapter, 2), SimpleNamespace()
    )
    adapter.submit_q_store_request(
        "second", _op(), _allocate(adapter, 1), SimpleNamespace()
    )
    worker.is_healthy = False

    adapter.reclaim_finished_q_stores()

    metrics = adapter.metrics_snapshot()
    assert metrics.requests_abandoned_unhealthy == 2
    assert metrics.in_flight_requests == 0
    assert metrics.in_flight_blocks == 0
    assert metrics.blocks_released == 3
    assert adapter.q_ring is not None
    assert adapter.q_ring.num_free_blocks() == 8
