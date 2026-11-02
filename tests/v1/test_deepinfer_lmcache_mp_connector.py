# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DeepInfer multiprocess client adapter."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("deepinfer.kv_transfer.base_connector")

# First Party
from lmcache.integration.deepinfer import lmcache_mp_connector as connector
from lmcache.integration.deepinfer import multi_process_adapter as adapter_module
from lmcache.v1.multiprocess.futures import MessagingFuture


class _NonBlockingFuture(MessagingFuture[Any]):
    """Reject a read that could wait on the scheduler thread."""

    def result(self, timeout: float | None = None) -> Any:
        assert timeout == 0
        assert self.query()
        return super().result(timeout)


def _ready(value: Any) -> MessagingFuture[Any]:
    future: MessagingFuture[Any] = MessagingFuture()
    future.set_result(value)
    return future


@pytest.mark.parametrize(
    ("is_mla", "kv_world_size", "kv_worker_id", "num_kv_readers", "is_writer"),
    [
        (True, 1, 0, 4, False),
        (False, 4, 3, 1, True),
    ],
)
def test_parallel_mapping_follows_model_cache_layout(
    is_mla: bool,
    kv_world_size: int,
    kv_worker_id: int,
    num_kv_readers: int,
    is_writer: bool,
) -> None:
    """Only replicated MLA collapses ranks and reserves N readers."""
    parallel = adapter_module.DeepInferMPParallelConfig.for_model(4, 3, is_mla=is_mla)

    assert parallel.kv_world_size == kv_world_size
    assert parallel.kv_worker_id == kv_worker_id
    assert parallel.num_kv_readers == num_kv_readers
    assert parallel.is_kv_writer is is_writer


def test_scheduler_adapter_polls_lookup_without_blocking(monkeypatch) -> None:
    """LOOKUP acknowledgement and status are advanced across scheduler ticks."""
    lookup = _NonBlockingFuture()
    status = _NonBlockingFuture()
    client = MagicMock()
    client.get_chunk_size.return_value = _ready(128)
    client.lookup.return_value = lookup
    client.query_prefetch_status.return_value = status
    monkeypatch.setattr(
        adapter_module.RequestClientFactory,
        "create",
        lambda *args, **kwargs: client,
    )
    adapter = adapter_module.DeepInferMPSchedulerAdapter(
        server_url="tcp://test",
        context=MagicMock(),
        model_name="deepinfer-test",
        block_size=128,
        parallel_config=adapter_module.DeepInferMPParallelConfig.for_model(
            4, 0, is_mla=True
        ),
    )

    adapter.maybe_submit_lookup_request("request-1", list(range(256)))
    lookup_key, legacy_parallel_size = client.lookup.call_args.args
    assert lookup_key.world_size == 1
    assert lookup_key.worker_id is None
    assert lookup_key.num_kv_readers == 4
    assert legacy_parallel_size == 4
    assert adapter.check_lookup_result("request-1") is None
    client.query_prefetch_status.assert_not_called()

    lookup.set_result(None)
    assert adapter.check_lookup_result("request-1") is None
    client.query_prefetch_status.assert_called_once_with("request-1")
    assert adapter.check_lookup_result("request-1") is None

    status.set_result(2)
    assert adapter.check_lookup_result("request-1") == 256
    assert adapter.check_lookup_result("request-1") == 256
    adapter.shutdown()


def test_collect_page_caches_discovers_components_and_deduplicates_aliases() -> None:
    """CA/ICA ownership follows tensor identity instead of model layer names."""
    index = torch.empty(8, 64, 1, 68, dtype=torch.uint8)
    latent = torch.empty(8, 64, 1, 288, dtype=torch.uint8)
    cache_list = [
        SimpleNamespace(ica_cache=index, ca_cache=latent),
        SimpleNamespace(ica_cache=index, ca_cache=None),
    ]

    caches, groups = connector._collect_page_caches(cache_list, block_size=128)

    assert list(caches) == ["layer.0.ica_cache", "layer.0.ca_cache"]
    assert all(tensor.dim() == 3 for tensor in caches.values())
    assert [group.engine_group_id for group in groups] == [0, 0]
    assert [group.tokens_per_block for group in groups] == [128, 128]


def test_connector_selects_adapter_from_role(monkeypatch) -> None:
    """One connector class owns both role-specific adapter lifecycles."""
    scheduler_adapter = MagicMock(lmcache_tokens_per_chunk=128)
    worker_adapter = MagicMock(lmcache_tokens_per_chunk=128)
    scheduler_factory = MagicMock(return_value=scheduler_adapter)
    worker_factory = MagicMock(return_value=worker_adapter)
    monkeypatch.setattr(
        connector,
        "DeepInferMPSchedulerAdapter",
        scheduler_factory,
    )
    monkeypatch.setattr(
        connector,
        "DeepInferMPWorkerAdapter",
        worker_factory,
    )
    cache_engine = SimpleNamespace(
        gpu_cache=SimpleNamespace(
            cache_list=[
                SimpleNamespace(
                    ica_cache=torch.empty(2, 128, 1, 1),
                    ca_cache=None,
                )
            ],
            total_bytes_inference=0,
        )
    )

    scheduler = connector.LMCacheMPConnector(
        "scheduler",
        {"server_url": "tcp://127.0.0.1:5555"},
        model_name="deepinfer-test",
        block_size=128,
        world_size=2,
        is_mla=True,
    )
    worker = connector.LMCacheMPConnector(
        "worker",
        {"server_url": "tcp://127.0.0.1:5555"},
        model_name="deepinfer-test",
        block_size=128,
        world_size=2,
        is_mla=True,
        worker_id=1,
        cache_engine=cache_engine,
    )

    assert type(scheduler) is type(worker)
    assert scheduler.role == "scheduler"
    assert worker.role == "worker"
    assert scheduler.scheduler_adapter is scheduler_adapter
    assert worker.worker_adapter is worker_adapter
    scheduler_parallel = scheduler_factory.call_args.kwargs["parallel_config"]
    worker_parallel = worker_factory.call_args.kwargs["parallel_config"]
    assert scheduler_parallel.kv_world_size == 1
    assert scheduler_parallel.num_kv_readers == 2
    assert worker_parallel.engine_worker_id == 1
    assert worker_parallel.kv_worker_id == 0
    assert worker_parallel.num_kv_readers == 2
    assert not worker._is_kv_writer
    worker_adapter.register_kv_caches.assert_called_once()


def test_worker_key_targets_shared_mla_object() -> None:
    """Every execution rank retrieves the same replicated MLA object."""
    adapter = adapter_module.DeepInferMPWorkerAdapter.__new__(
        adapter_module.DeepInferMPWorkerAdapter
    )
    adapter._model_name = "deepinfer-test"
    adapter._parallel = adapter_module.DeepInferMPParallelConfig.for_model(
        4, 3, is_mla=True
    )
    spec = adapter_module.DeepInferMPTransferSpec(
        token_ids=list(range(128)),
        block_ids=[[7]],
        start=0,
        end=128,
    )

    key = adapter._create_key("request-1", spec)

    assert key.world_size == 1
    assert key.worker_id == 0
    assert key.num_kv_readers == 4


def test_non_writer_rank_skips_replicated_kv_store() -> None:
    """Only the canonical engine rank stores a replicated MLA object."""
    worker = connector.LMCacheMPConnector.__new__(connector.LMCacheMPConnector)
    worker._role = "worker"
    worker._is_kv_writer = False
    worker.worker_adapter = MagicMock()
    worker._pending = {}
    worker._pending_loads = {}
    worker._stored_tokens = {}
    worker._submitted_tokens = {}
    worker._active_sends = set()
    worker._finished_sends = set()
    worker._finished_requests = set()

    worker.wait_for_save(
        request_id="request-1",
        token_ids=list(range(128)),
        block_ids=[7],
        state_id=3,
        end=128,
    )

    worker.worker_adapter.submit_store_request.assert_not_called()
    assert worker.get_transfer_results().finished_sending == {"request-1"}


def test_scheduler_lookup_caps_and_aligns_query(monkeypatch) -> None:
    """DeepInfer always leaves a chunk-aligned endpoint for state restore."""
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = 128
    adapter.check_lookup_result.return_value = 256
    monkeypatch.setattr(
        connector, "DeepInferMPSchedulerAdapter", lambda **kwargs: adapter
    )
    scheduler = connector.LMCacheMPConnector(
        "scheduler",
        {"server_url": "tcp://127.0.0.1:5555"},
        model_name="deepinfer-test",
        block_size=128,
        world_size=2,
        is_mla=True,
    )

    hit = scheduler.get_num_new_matched_tokens(
        request_id="request-1", token_ids=list(range(400)), end=399
    )

    assert hit == (256, True)
    submitted_tokens = adapter.maybe_submit_lookup_request.call_args.args[1]
    assert len(submitted_tokens) == 384


def test_scheduler_lookup_returns_pending_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unfinished lookup leaves the request parked in DeepInfer."""
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = 128
    adapter.check_lookup_result.return_value = None
    monkeypatch.setattr(
        connector, "DeepInferMPSchedulerAdapter", lambda **kwargs: adapter
    )
    scheduler = connector.LMCacheMPConnector(
        "scheduler",
        {"server_url": "tcp://127.0.0.1:5555"},
        model_name="deepinfer-test",
        block_size=128,
        world_size=2,
        is_mla=True,
    )

    result = scheduler.get_num_new_matched_tokens(
        request_id="request-pending", token_ids=list(range(128)), end=128
    )

    assert result == (None, True)


def test_worker_block_ids_use_sparse_endpoint_state() -> None:
    """Only the final LMCache chunk references the runtime-state snapshot."""
    worker = connector.LMCacheMPConnector.__new__(connector.LMCacheMPConnector)
    worker._role = "worker"
    worker._block_size = 128
    worker._chunk_size = 128
    worker._state_group = True

    block_ids = worker._block_ids([7, 8, 9, 10], 3, start=128, end=512)

    assert block_ids == [[8, 9, 10], [0, 0, 3]]


def test_failed_load_does_not_bulk_release_shared_reader_locks() -> None:
    """Worker retrieves settle locks individually, including on failure."""
    scheduler = connector.LMCacheMPConnector.__new__(connector.LMCacheMPConnector)
    scheduler._role = "scheduler"
    scheduler.scheduler_adapter = MagicMock()
    scheduler._scheduled_loads = {}

    scheduler.update_state_after_alloc(
        request_id="request-1",
        token_ids=list(range(256)),
        num_external_tokens=256,
    )
    scheduler.update_connector_output(
        connector.KVConnectorTransferResults(
            finished_recving={"request-1"},
            failed_recving={"request-1"},
        )
    )

    scheduler.scheduler_adapter.free_lookup_locks.assert_not_called()
    scheduler.scheduler_adapter.cleanup_lookup_result.assert_called_once_with(
        "request-1"
    )


def test_worker_load_is_submitted_then_polled() -> None:
    """The worker connector does not wait in start_load_kv."""
    future = MagicMock()
    future.query.side_effect = [False, True]
    future.result.return_value = True
    worker = connector.LMCacheMPConnector.__new__(connector.LMCacheMPConnector)
    worker._role = "worker"
    worker._block_size = 128
    worker._chunk_size = 128
    worker._state_group = False
    worker._pending_loads = {}
    worker._pending = {}
    worker._finished_sends = set()
    worker.worker_adapter = MagicMock()
    worker.worker_adapter.submit_retrieve_request.return_value = future

    worker.start_load_kv(
        request_id="request-1",
        token_ids=list(range(128)),
        block_ids=[7],
        state_id=3,
        hit_tokens=128,
    )

    assert worker.get_transfer_results().finished_recving == set()
    output = worker.get_transfer_results()
    assert output.finished_recving == {"request-1"}
    assert output.failed_recving == set()


def test_unsubmitted_worker_load_releases_one_shared_reader() -> None:
    """A locally dropped retrieve releases only that rank's reader share."""
    worker = connector.LMCacheMPConnector.__new__(connector.LMCacheMPConnector)
    worker._role = "worker"
    worker._block_size = 128
    worker._chunk_size = 128
    worker._state_group = False
    worker._pending_loads = {}
    worker._pending = {}
    worker._finished_sends = set()
    worker.worker_adapter = MagicMock()
    worker.worker_adapter.submit_retrieve_request.return_value = None

    worker.start_load_kv(
        request_id="request-1",
        token_ids=list(range(128)),
        block_ids=[7],
        state_id=3,
        hit_tokens=128,
    )

    spec = worker.worker_adapter.submit_retrieve_request.call_args.args[1]
    worker.worker_adapter.release_lookup_reader.assert_called_once_with(
        "request-1", spec
    )
    output = worker.get_transfer_results()
    assert output.finished_recving == {"request-1"}
    assert output.failed_recving == {"request-1"}


def test_store_uses_distinct_snapshot_slots_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second store snapshots into another slot while the first D2H runs."""
    first = _NonBlockingFuture()
    second = _NonBlockingFuture()
    worker_adapter = MagicMock(lmcache_tokens_per_chunk=128)
    worker_adapter.submit_store_request.side_effect = [first, second]
    monkeypatch.setattr(
        connector,
        "DeepInferMPWorkerAdapter",
        lambda **kwargs: worker_adapter,
    )
    cache_engine = SimpleNamespace(
        gpu_states=torch.empty(2, 8, dtype=torch.uint8),
        gpu_cache=SimpleNamespace(
            cache_list=[
                SimpleNamespace(
                    ica_cache=torch.empty(2, 128, 1, 1),
                    ca_cache=None,
                )
            ],
            total_bytes_inference=8,
            select_prefix_state_to_store=MagicMock(),
        ),
    )
    worker = connector.LMCacheMPConnector(
        "worker",
        {
            "server_url": "tcp://127.0.0.1:5555",
            "max_inflight_loads": 1,
            "state_snapshot_pool_size": 2,
        },
        model_name="deepinfer-test",
        block_size=128,
        world_size=1,
        is_mla=False,
        worker_id=0,
        cache_engine=cache_engine,
    )

    worker.wait_for_save(
        request_id="request-1",
        token_ids=list(range(256)),
        block_ids=[7, 8],
        state_id=1,
        end=128,
    )
    worker.wait_for_save(
        request_id="request-1",
        token_ids=list(range(256)),
        block_ids=[7, 8],
        state_id=1,
        end=256,
    )

    assert worker_adapter.submit_store_request.call_count == 2
    first_spec = worker_adapter.submit_store_request.call_args_list[0].args[1]
    second_spec = worker_adapter.submit_store_request.call_args_list[1].args[1]
    assert first_spec.block_ids[1] == [2]
    assert second_spec.block_ids[1] == [3]
    assert worker.get_transfer_results().finished_sending == set()

    # Request completion is bookkeeping, not a synchronous D2H barrier.
    worker.request_finished("request-1")
    first.set_result(True)
    second.set_result(True)
    output = worker.get_transfer_results()

    assert output.finished_sending == {"request-1"}
