# SPDX-License-Identifier: Apache-2.0
"""Worker-adapter tests for the lazy offload store-receipt contract.

The scheduler unpins a drained store batch's blocks (and possibly ends the
request's session) only after collecting one store-completion receipt per
worker rank. These tests pin the adapter-side half of that contract: every
submitted store batch yields exactly one receipt from this rank, regardless
of writer role or server health -- a rank that will never produce a store
future must report completion immediately, or the scheduler waits forever
and the blocks stay pinned.

The adapter is built via ``__new__`` with only the attributes the tested
paths read, mirroring the connector-side suite in
``test_mp_connector_lazy_offload.py``.
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast
import threading

# Third Party
import pytest

pytest.importorskip("vllm", reason="the MP adapter imports vLLM at module top")

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    HeartbeatThread,
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
)


def _make_worker_adapter(
    healthy: bool = True,
    is_kv_writer: bool = True,
    lazy_offload: bool = True,
) -> LMCacheMPWorkerAdapter:
    """Build an adapter with only the attributes the tested paths read."""
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.lazy_offload = lazy_offload
    adapter.dispatcher = None
    adapter._health_event = threading.Event()
    if healthy:
        adapter._health_event.set()
    # A non-None sentinel makes _ensure_heartbeat_started a no-op.
    adapter._heartbeat = cast(HeartbeatThread, object())
    adapter.parallel_strategy = SimpleNamespace(  # type: ignore[assignment]
        is_kv_writer=is_kv_writer
    )
    adapter.store_futures = {}
    adapter.retrieve_futures = {}
    adapter.store_events = {}
    adapter.retrieve_events = {}
    adapter._dropped_retrieves = set()
    adapter.error_block_ids = set()
    adapter._completed_store_requests = {}
    adapter._failed_store_requests = set()
    return adapter


def _make_op() -> LoadStoreOp:
    return LoadStoreOp(token_ids=list(range(32)), block_ids=[[1, 2]], start=0, end=32)


def _submit_store(adapter: LMCacheMPWorkerAdapter, request_id: str = "req") -> None:
    adapter.submit_store_request(request_id, _make_op(), event=None)  # type: ignore[arg-type]


####
# Receipt completeness: submit-time drops must still produce receipts
####


def test_non_writer_rank_reports_completion_at_submit() -> None:
    """MLA TP>1: a non-writer rank never creates a store future, so it must
    report completion immediately -- the scheduler counts one receipt per
    rank of the whole world before unpinning."""
    adapter = _make_worker_adapter(is_kv_writer=False)

    _submit_store(adapter)

    assert adapter.store_futures == {}
    assert adapter.get_completed_store_requests() == {"req": 1}
    # Exactly once: the receipt is not re-reported on later calls.
    assert adapter.get_completed_store_requests() is None


def test_unhealthy_submit_reports_completion_instead_of_silent_drop() -> None:
    """A store dropped at submit time while the server is unhealthy will
    never get a future; without an immediate receipt the pinned blocks and
    the session leak forever."""
    adapter = _make_worker_adapter(healthy=False)

    _submit_store(adapter)

    assert adapter.store_futures == {}
    assert adapter.get_completed_store_requests() == {"req": 1}
    assert adapter.get_completed_store_requests() is None


def test_non_lazy_submit_drops_do_not_accumulate_receipts() -> None:
    """Outside lazy offload nothing drains the receipt dict; submit-time
    drops must not grow it."""
    adapter = _make_worker_adapter(is_kv_writer=False, lazy_offload=False)

    _submit_store(adapter)

    assert adapter.get_completed_store_requests() is None


####
# Receipts from finished store futures
####


class _FakeStoreFuture:
    """A store future in a fixed state."""

    def __init__(self, done: bool, result: Any = True) -> None:
        self._done = done
        self._result = result

    def query(self) -> bool:
        return self._done

    def result(self, timeout: int = 0) -> Any:
        return self._result


def _prepare_for_get_finished(adapter: LMCacheMPWorkerAdapter) -> None:
    """Pin the extra attributes the healthy get_finished path reads."""
    adapter.model_name = "model"
    adapter.request_telemetry = SimpleNamespace(  # type: ignore[assignment]
        on_request_store_finished=lambda **kwargs: None
    )
    adapter.parallel_strategy = SimpleNamespace(  # type: ignore[assignment]
        is_kv_writer=True, kv_world_size=1, kv_worker_id=0
    )


def test_finished_store_future_yields_receipt_and_is_untracked() -> None:
    adapter = _make_worker_adapter()
    _prepare_for_get_finished(adapter)
    adapter.store_futures["req"] = _FakeStoreFuture(done=True)  # type: ignore[assignment]

    adapter.get_finished_with_lazy_offload()

    assert adapter.get_completed_store_requests() == {"req": 1}
    assert adapter.store_futures == {}
    # The receipt is delivered exactly once.
    adapter.get_finished_with_lazy_offload()
    assert adapter.get_completed_store_requests() is None


def test_pending_store_future_yields_no_receipt_yet() -> None:
    adapter = _make_worker_adapter()
    _prepare_for_get_finished(adapter)
    adapter.store_futures["req"] = _FakeStoreFuture(done=False)  # type: ignore[assignment]

    adapter.get_finished_with_lazy_offload()

    assert adapter.get_completed_store_requests() is None
    assert "req" in adapter.store_futures


def test_unhealthy_drain_receipts_all_outstanding_stores() -> None:
    """When the server turns unhealthy, every outstanding store future is
    drained and its receipt still reaches the scheduler (the store may be
    lost, but the blocks must be unpinned)."""
    adapter = _make_worker_adapter(healthy=False)
    adapter.store_futures["req"] = _FakeStoreFuture(done=False)  # type: ignore[assignment]

    stores, retrieves = adapter.get_finished_with_lazy_offload()

    assert stores is None
    assert retrieves == set()
    assert adapter.store_futures == {}
    assert adapter.get_completed_store_requests() == {"req": 1}


def test_get_finished_with_lazy_offload_requires_lazy_mode() -> None:
    adapter = _make_worker_adapter(lazy_offload=False)
    with pytest.raises(ValueError, match="lazy offload"):
        adapter.get_finished_with_lazy_offload()


####
# Failed stores: the receipt still arrives, the integrity signal separately
####


def test_failed_store_future_reports_failure_alongside_receipt() -> None:
    """A failed store must still produce its completion receipt (the
    pinned blocks have to be unpinned either way); the failure travels as
    a separate signal so the scheduler can break the prefix chain."""
    adapter = _make_worker_adapter()
    _prepare_for_get_finished(adapter)
    adapter.store_futures["req"] = _FakeStoreFuture(done=True, result=None)  # type: ignore[assignment]

    adapter.get_finished_with_lazy_offload()

    assert adapter.get_completed_store_requests() == {"req": 1}
    assert adapter.get_failed_store_requests() == {"req"}
    # Exactly once.
    assert adapter.get_failed_store_requests() is None


def test_successful_store_future_reports_no_failure() -> None:
    adapter = _make_worker_adapter()
    _prepare_for_get_finished(adapter)
    adapter.store_futures["req"] = _FakeStoreFuture(done=True)  # type: ignore[assignment]

    adapter.get_finished_with_lazy_offload()

    assert adapter.get_failed_store_requests() is None


def test_unhealthy_submit_drop_reports_failure() -> None:
    adapter = _make_worker_adapter(healthy=False)
    _submit_store(adapter)
    assert adapter.get_failed_store_requests() == {"req"}


def test_unhealthy_drain_reports_failure_for_unknown_outcomes() -> None:
    """A future drained by the unhealthy branch has an unknown outcome;
    the data cannot be assumed stored."""
    adapter = _make_worker_adapter(healthy=False)
    adapter.store_futures["req"] = _FakeStoreFuture(done=False)  # type: ignore[assignment]

    adapter.get_finished_with_lazy_offload()

    assert adapter.get_failed_store_requests() == {"req"}


def test_non_writer_receipt_is_not_a_failure() -> None:
    """A non-writer rank stores nothing by design; its synthetic receipt
    must not break the prefix chain (the writer rank stores the data)."""
    adapter = _make_worker_adapter(is_kv_writer=False)
    _submit_store(adapter)
    assert adapter.get_failed_store_requests() is None


####
# Scheduler-side receipt counting
####


def _make_receipt_counter(expected_worker_count: int) -> LMCacheMPSchedulerAdapter:
    """Build a scheduler adapter with only the counting state pinned."""
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter._expected_worker_count = expected_worker_count
    adapter._store_request_pending_counts = {}
    return adapter


def test_pending_store_count_completes_exactly_at_worker_count() -> None:
    adapter = _make_receipt_counter(expected_worker_count=4)
    assert adapter.update_pending_store_count("req", 4) is True


def test_pending_store_count_accumulates_across_steps() -> None:
    adapter = _make_receipt_counter(expected_worker_count=4)
    for _ in range(3):
        assert adapter.update_pending_store_count("req", 1) is False
    assert adapter.update_pending_store_count("req", 1) is True


def test_pending_store_count_resets_after_completion() -> None:
    """A later batch of the same request starts counting from zero."""
    adapter = _make_receipt_counter(expected_worker_count=2)
    adapter.update_pending_store_count("req", 1)
    assert adapter.update_pending_store_count("req", 1) is True
    assert adapter.update_pending_store_count("req", 1) is False


def test_pending_store_count_is_per_request() -> None:
    adapter = _make_receipt_counter(expected_worker_count=2)
    assert adapter.update_pending_store_count("req-a", 1) is False
    assert adapter.update_pending_store_count("req-b", 1) is False
    assert adapter.update_pending_store_count("req-a", 1) is True


####
# Worker metadata aggregation
####


def test_worker_metadata_aggregate_sums_per_request_counts() -> None:
    first = LMCacheMPWorkerMetadata(completed_store_requests={"r1": 1, "r2": 1})
    second = LMCacheMPWorkerMetadata(completed_store_requests={"r1": 1, "r3": 1})

    merged = first.aggregate(second)

    assert isinstance(merged, LMCacheMPWorkerMetadata)
    assert merged.completed_store_requests == {"r1": 2, "r2": 1, "r3": 1}
    # Inputs are not mutated.
    assert first.completed_store_requests == {"r1": 1, "r2": 1}
    assert second.completed_store_requests == {"r1": 1, "r3": 1}


def test_worker_metadata_aggregate_unions_failed_stores() -> None:
    """One rank's failure breaks the request's prefix chain even when the
    other ranks succeeded."""
    first = LMCacheMPWorkerMetadata(
        completed_store_requests={"r1": 1}, failed_store_requests={"r1"}
    )
    second = LMCacheMPWorkerMetadata(completed_store_requests={"r1": 1})

    merged = first.aggregate(second)

    assert isinstance(merged, LMCacheMPWorkerMetadata)
    assert merged.failed_store_requests == {"r1"}
    assert second.failed_store_requests == set()
