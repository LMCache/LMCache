# SPDX-License-Identifier: Apache-2.0
"""Worker-adapter tests for the lazy offload store-receipt contract.

The scheduler unpins a drained batch's blocks only after one store-completion
receipt per worker rank. Every submitted batch must yield exactly one receipt
from this rank regardless of writer role or server health: a rank that will
never produce a store future has to report completion immediately, or the
scheduler waits forever and the blocks stay pinned.
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
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
from lmcache.v1.multiprocess.group_view import (  # noqa: E402
    PARTIAL_STORE_GROUPS_CAPABILITY,
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
    adapter._lazy_store_futures = {}
    adapter.retrieve_futures = {}
    adapter.store_events = {}
    adapter._lazy_store_events = {}
    adapter.retrieve_events = {}
    adapter._dropped_retrieves = set()
    adapter.error_block_ids = set()
    adapter._completed_store_requests = {}
    adapter._failed_store_requests = set()
    adapter._completed_store_operations = {}
    adapter._failed_store_operations = set()
    adapter.experimental = set()
    adapter._kv_events_enabled = False
    adapter._pending_store_kv_events = {}
    return adapter


def _make_op() -> LoadStoreOp:
    return LoadStoreOp(token_ids=list(range(32)), block_ids=[[1, 2]], start=0, end=32)


def _submit_store(
    adapter: LMCacheMPWorkerAdapter,
    request_id: str = "req",
    operation_id: int = 10,
) -> None:
    adapter.submit_store_request(  # type: ignore[arg-type]
        request_id,
        _make_op(),
        event=None,
        store_operation_id=operation_id,
    )


def _prepare_successful_submit(adapter: LMCacheMPWorkerAdapter) -> MagicMock:
    """Install the minimal healthy-writer dependencies for store submission."""
    future = MagicMock()
    adapter.transfer_ctx = SimpleNamespace(  # type: ignore[assignment]
        submit_store=MagicMock(return_value=future)
    )
    adapter.kv_caches = {}
    adapter.blocks_in_chunk = 1
    adapter._create_key = MagicMock(return_value=object())  # type: ignore[method-assign]
    adapter._block_ids_per_group = MagicMock(  # type: ignore[method-assign]
        return_value=[[1, 2]]
    )
    build_events = MagicMock(return_value=["store-event"])
    adapter._build_store_kv_events = build_events  # type: ignore[method-assign]
    return build_events


def test_partial_group_store_does_not_publish_model_wide_kv_event() -> None:
    adapter = _make_worker_adapter()
    build_events = _prepare_successful_submit(adapter)
    adapter.experimental = {PARTIAL_STORE_GROUPS_CAPABILITY}
    adapter._kv_events_enabled = True
    op = LoadStoreOp(
        token_ids=list(range(32)),
        block_ids=[[1, 2]],
        start=0,
        end=32,
        selected_engine_group_ids=(0,),
    )

    adapter.submit_store_request(
        "req",
        op,
        event=None,
        store_operation_id=10,
    )

    build_events.assert_not_called()
    assert adapter._pending_store_kv_events == {}


def test_whole_cache_store_still_records_kv_event() -> None:
    adapter = _make_worker_adapter()
    build_events = _prepare_successful_submit(adapter)
    adapter._kv_events_enabled = True

    adapter.submit_store_request(
        "req",
        _make_op(),
        event=None,
        store_operation_id=10,
    )

    build_events.assert_called_once()
    assert adapter._pending_store_kv_events == {10: ["store-event"]}


####
# Receipt completeness: submit-time drops must still produce receipts
####


def test_non_writer_rank_reports_completion_at_submit() -> None:
    """MLA TP>1: a non-writer rank never creates a store future, so it must report
    completion immediately -- the scheduler counts one receipt per rank of the whole
    world before unpinning. Storing nothing by design is not a failure, so the
    synthetic receipt must not break the prefix chain either.
    """
    adapter = _make_worker_adapter(is_kv_writer=False)

    _submit_store(adapter)

    assert adapter.store_futures == {}
    assert adapter.get_completed_store_operations() == {10: 1}
    # Exactly once: the receipt is not re-reported on later calls.
    assert adapter.get_completed_store_operations() is None
    assert adapter.get_failed_store_operations() is None


def test_unhealthy_submit_reports_completion_and_failure() -> None:
    """A store dropped at submit time while the server is unhealthy will never get a
    future; without an immediate receipt the pinned blocks and the session leak forever.
    Nothing was written, so the failure travels with the receipt.
    """
    adapter = _make_worker_adapter(healthy=False)

    _submit_store(adapter)

    assert adapter.store_futures == {}
    assert adapter.get_completed_store_operations() == {10: 1}
    assert adapter.get_completed_store_operations() is None
    assert adapter.get_failed_store_operations() == {10}


def test_non_lazy_submit_drops_do_not_accumulate_receipts() -> None:
    """Outside lazy offload nothing drains the receipt dict; submit-time drops must not
    grow it.
    """
    adapter = _make_worker_adapter(is_kv_writer=False, lazy_offload=False)

    _submit_store(adapter)

    assert adapter.get_completed_store_operations() is None


####
# Receipts from store futures
####


class _FakeStoreFuture:
    """A store future in a fixed state."""

    def __init__(self, done: bool, result: Any = True) -> None:
        self._done = done
        self._result = result

    def query(self) -> bool:
        """Whether the store has completed."""
        return self._done

    def result(self, timeout: int = 0) -> Any:
        """The store's outcome."""
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
    adapter._lazy_store_futures[10] = (  # type: ignore[assignment]
        "req",
        _FakeStoreFuture(done=True),
    )

    adapter.get_finished_with_lazy_offload()

    assert adapter.get_completed_store_operations() == {10: 1}
    assert adapter._lazy_store_futures == {}
    assert adapter.get_failed_store_operations() is None
    # The receipt is delivered exactly once.
    adapter.get_finished_with_lazy_offload()
    assert adapter.get_completed_store_operations() is None


def test_failed_store_future_reports_failure_alongside_receipt() -> None:
    """A failed store must still produce its completion receipt (the pinned blocks have
    to be unpinned either way); the failure travels as a separate signal so the
    scheduler can break the prefix chain.
    """
    adapter = _make_worker_adapter()
    _prepare_for_get_finished(adapter)
    adapter._lazy_store_futures[10] = (  # type: ignore[assignment]
        "req",
        _FakeStoreFuture(done=True, result=None),
    )

    adapter.get_finished_with_lazy_offload()

    assert adapter.get_completed_store_operations() == {10: 1}
    assert adapter.get_failed_store_operations() == {10}
    # Exactly once.
    assert adapter.get_failed_store_operations() is None


def test_pending_store_future_yields_no_receipt_yet() -> None:
    adapter = _make_worker_adapter()
    _prepare_for_get_finished(adapter)
    adapter._lazy_store_futures[10] = (  # type: ignore[assignment]
        "req",
        _FakeStoreFuture(done=False),
    )

    adapter.get_finished_with_lazy_offload()

    assert adapter.get_completed_store_operations() is None
    assert 10 in adapter._lazy_store_futures


def test_unhealthy_drain_receipts_all_outstanding_stores() -> None:
    """When the server turns unhealthy, every outstanding store future is drained and
    its receipt still reaches the scheduler (the store may be lost, but the blocks must
    be unpinned). The outcome is unknown, so the data cannot be assumed stored.
    """
    adapter = _make_worker_adapter(healthy=False)
    adapter._lazy_store_futures[10] = (  # type: ignore[assignment]
        "req",
        _FakeStoreFuture(done=False),
    )

    stores, retrieves = adapter.get_finished_with_lazy_offload()

    assert stores is None
    assert retrieves == set()
    assert adapter._lazy_store_futures == {}
    assert adapter.get_completed_store_operations() == {10: 1}
    assert adapter.get_failed_store_operations() == {10}


def test_get_finished_with_lazy_offload_requires_lazy_mode() -> None:
    adapter = _make_worker_adapter(lazy_offload=False)
    with pytest.raises(ValueError, match="lazy offload"):
        adapter.get_finished_with_lazy_offload()


####
# Scheduler-side receipt counting
####


def _make_receipt_counter(expected_worker_count: int) -> LMCacheMPSchedulerAdapter:
    """Build a scheduler adapter with only the counting state pinned."""
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter._expected_worker_count = expected_worker_count
    adapter._store_request_pending_counts = {}
    adapter._store_operation_pending_counts = {}
    return adapter


def test_pending_store_count_accumulates_across_steps() -> None:
    adapter = _make_receipt_counter(expected_worker_count=4)
    for _ in range(3):
        assert adapter.update_pending_store_operation_count(10, 1) is False
    assert adapter.update_pending_store_operation_count(10, 1) is True


def test_pending_store_count_resets_after_completion() -> None:
    """A later batch of the same request starts counting from zero."""
    adapter = _make_receipt_counter(expected_worker_count=2)
    adapter.update_pending_store_operation_count(10, 1)
    assert adapter.update_pending_store_operation_count(10, 1) is True
    assert adapter.update_pending_store_operation_count(10, 1) is False


def test_pending_store_count_is_per_operation() -> None:
    adapter = _make_receipt_counter(expected_worker_count=2)
    assert adapter.update_pending_store_operation_count(10, 1) is False
    assert adapter.update_pending_store_operation_count(11, 1) is False
    assert adapter.update_pending_store_operation_count(10, 1) is True


####
# Worker metadata aggregation
####


def test_worker_metadata_aggregate_sums_per_operation_counts() -> None:
    first = LMCacheMPWorkerMetadata(completed_store_operations={10: 1, 20: 1})
    second = LMCacheMPWorkerMetadata(completed_store_operations={10: 1, 30: 1})

    merged = first.aggregate(second)

    assert isinstance(merged, LMCacheMPWorkerMetadata)
    assert merged.completed_store_operations == {10: 2, 20: 1, 30: 1}
    # Inputs are not mutated.
    assert first.completed_store_operations == {10: 1, 20: 1}
    assert second.completed_store_operations == {10: 1, 30: 1}


def test_worker_metadata_aggregate_unions_failed_stores() -> None:
    """One rank's failure breaks the request's prefix chain even when the other ranks
    succeeded.
    """
    first = LMCacheMPWorkerMetadata(
        completed_store_operations={10: 1}, failed_store_operations={10}
    )
    second = LMCacheMPWorkerMetadata(completed_store_operations={10: 1})

    merged = first.aggregate(second)

    assert isinstance(merged, LMCacheMPWorkerMetadata)
    assert merged.failed_store_operations == {10}
    assert second.failed_store_operations == set()
