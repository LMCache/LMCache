# SPDX-License-Identifier: Apache-2.0
"""Operation-level lifetime tests for LMCache MP STORE operations."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LoadStoreOp,
)


def _store_metadata(request_id: str, block_ids: list[int]) -> LMCacheMPRequestMetadata:
    return LMCacheMPRequestMetadata(
        request_id=request_id,
        direction="STORE",
        op=LoadStoreOp(
            token_ids=list(range(256)),
            block_ids=[block_ids],
            start=0,
            end=256,
        ),
    )


def _bare_scheduler_adapter(expected_workers: int) -> LMCacheMPSchedulerAdapter:
    adapter = object.__new__(LMCacheMPSchedulerAdapter)
    adapter._expected_worker_count = expected_workers
    adapter._store_operation_terminal_workers = {}
    adapter._store_operation_failed_workers = {}
    return adapter


def _bare_connector(expected_workers: int = 2) -> tuple[LMCacheMPConnector, MagicMock]:
    connector = object.__new__(LMCacheMPConnector)
    connector._role = KVConnectorRole.SCHEDULER
    connector.lazy_offload = False
    connector._next_store_op_id = 0
    connector._pinned_store_operations = {}
    connector.scheduler_adapter = _bare_scheduler_adapter(expected_workers)
    pool = MagicMock(name="block_pool")
    pool.blocks = {index: MagicMock(name=f"block_{index}") for index in range(8)}
    connector._gpu_block_pool = pool
    return connector, pool


def test_worker_metadata_aggregates_terminals_and_failures() -> None:
    merged = LMCacheMPWorkerMetadata(
        completed_store_requests={"lazy": 1},
        terminal_store_operations={1: {0}, 2: {0}},
        failed_store_operations={2: {0}},
    ).aggregate(
        LMCacheMPWorkerMetadata(
            completed_store_requests={"lazy": 1},
            terminal_store_operations={1: {1}, 3: {1}},
            failed_store_operations={3: {1}},
        )
    )

    assert isinstance(merged, LMCacheMPWorkerMetadata)
    assert merged.completed_store_requests == {"lazy": 2}
    assert merged.terminal_store_operations == {1: {0, 1}, 2: {0}, 3: {1}}
    assert merged.failed_store_operations == {2: {0}, 3: {1}}


def test_scheduler_aggregates_operation_across_steps() -> None:
    adapter = _bare_scheduler_adapter(expected_workers=2)

    assert adapter.update_pending_store_workers(4, {1}, {1}) is None
    assert adapter.update_pending_store_workers(4, {0}, set()) == {1}
    assert adapter._store_operation_terminal_workers == {}
    assert adapter._store_operation_failed_workers == {}


def test_scheduler_rejects_invalid_worker_id() -> None:
    adapter = _bare_scheduler_adapter(expected_workers=2)

    try:
        adapter.update_pending_store_workers(5, {0, 2}, set())
    except ValueError as exc:
        assert "invalid worker IDs" in str(exc)
    else:
        raise AssertionError("an invalid worker ID was accepted")


def test_scheduler_rejects_failure_without_terminal_report() -> None:
    adapter = _bare_scheduler_adapter(expected_workers=2)

    try:
        adapter.update_pending_store_workers(5, {0}, {1})
    except ValueError as exc:
        assert "subset" in str(exc)
    else:
        raise AssertionError("a failure without a terminal report was accepted")


def test_duplicate_worker_report_does_not_replace_delayed_rank() -> None:
    adapter = _bare_scheduler_adapter(expected_workers=2)

    assert adapter.update_pending_store_workers(6, {0}, set()) is None
    assert adapter.update_pending_store_workers(6, {0}, set()) is None
    assert adapter._store_operation_terminal_workers[6] == {0}
    assert adapter.update_pending_store_workers(6, {1}, set()) == set()


def test_store_pins_unique_blocks_and_releases_after_all_ranks() -> None:
    connector, pool = _bare_connector(expected_workers=2)
    connector_metadata = LMCacheMPConnectorMetadata()
    store_metadata = _store_metadata("same-request", [1, 2, 1, 2])

    connector._add_store_metadata(connector_metadata, store_metadata)

    assert store_metadata.store_op_id == 0
    pool.touch.assert_called_once_with([pool.blocks[1], pool.blocks[2]])
    assert connector.has_pending_push_work() is True

    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                terminal_store_operations={0: {0}},
            )
        )
    )
    pool.free_blocks.assert_not_called()

    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                terminal_store_operations={0: {1}},
            )
        )
    )
    assert list(pool.free_blocks.call_args.args[0]) == [
        pool.blocks[2],
        pool.blocks[1],
    ]
    assert pool.free_blocks.call_count == 1
    assert connector.has_pending_push_work() is False


def test_same_request_gets_independent_operation_ids() -> None:
    connector, _pool = _bare_connector(expected_workers=1)
    connector_metadata = LMCacheMPConnectorMetadata()
    first = _store_metadata("same-request", [1])
    second = _store_metadata("same-request", [2])

    connector._add_store_metadata(connector_metadata, first)
    connector._add_store_metadata(connector_metadata, second)

    assert first.store_op_id == 0
    assert second.store_op_id == 1
    assert set(connector._pinned_store_operations) == {0, 1}


def test_failed_rank_is_terminal_and_releases_after_all_ranks() -> None:
    connector, pool = _bare_connector(expected_workers=2)
    connector_metadata = LMCacheMPConnectorMetadata()
    connector._add_store_metadata(
        connector_metadata,
        _store_metadata("req-failure", [3]),
    )

    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                terminal_store_operations={0: {0, 1}},
                failed_store_operations={0: {1}},
            )
        )
    )

    assert list(pool.free_blocks.call_args.args[0]) == [pool.blocks[3]]
    assert pool.free_blocks.call_count == 1
    assert connector.has_pending_push_work() is False


def test_request_finish_uses_operation_owned_block_references() -> None:
    connector, pool = _bare_connector(expected_workers=1)
    connector._cleanup_request_tracker = MagicMock()
    connector.scheduler_adapter.end_session = MagicMock()
    request = SimpleNamespace(request_id="req-finished", kv_transfer_params=None)

    delay_free, return_params = connector.request_finished(request, [1, 2])

    assert delay_free is False
    assert return_params is None
    pool.free_blocks.assert_not_called()
    connector._cleanup_request_tracker.assert_called_once_with("req-finished")
    connector.scheduler_adapter.end_session.assert_called_once_with("req-finished")


def test_worker_metadata_drains_operation_results() -> None:
    connector = object.__new__(LMCacheMPConnector)
    connector.lazy_offload = False
    connector.worker_adapter = MagicMock()
    connector.worker_adapter.get_completed_store_operations.return_value = (
        {7: {0, 1}},
        {7: {1}},
    )

    metadata = connector.build_connector_worker_meta()

    assert metadata == LMCacheMPWorkerMetadata(
        terminal_store_operations={7: {0, 1}},
        failed_store_operations={7: {1}},
    )
