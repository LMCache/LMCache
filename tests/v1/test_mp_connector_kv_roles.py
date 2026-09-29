# SPDX-License-Identifier: Apache-2.0
"""Role contracts through public MP connector APIs; network/GPU I/O is stubbed."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.config import KVTransferConfig, VllmConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (  # noqa: E402
    MultiConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_mod  # noqa: E402
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
)

pytestmark = pytest.mark.no_shared_allocator


def _config(transfer_config: KVTransferConfig) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(model="test-model", use_mla=False),
            parallel_config=SimpleNamespace(
                world_size=1, rank=0, tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            cache_config=SimpleNamespace(block_size=4, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
            kv_transfer_config=transfer_config,
        ),
    )


def _request(request_id: str = "request") -> Request:
    return cast(
        Request,
        SimpleNamespace(
            request_id=request_id,
            cache_salt="",
            all_token_ids=list(range(12)),
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            kv_transfer_params=None,
        ),
    )


def _schedule(
    request_id: str = "request",
    num_tokens: int = 0,
    block_ids: tuple[list[int], ...] = (),
    *,
    new: bool = False,
) -> SchedulerOutput:
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            scheduled_new_reqs=(
                [SimpleNamespace(req_id=request_id, block_ids=block_ids)]
                if new and num_tokens
                else []
            ),
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[request_id] if not new and num_tokens else [],
                new_block_ids=[block_ids] if not new and num_tokens else [],
                resumed_req_ids=set(),
            ),
            num_scheduled_tokens={request_id: num_tokens} if num_tokens else {},
            total_num_scheduled_tokens=num_tokens,
        ),
    )


@pytest.fixture(params=["kv_consumer", "kv_producer", "kv_both"])
def kv_role(request: pytest.FixtureRequest) -> str:
    """Exercise all roles supported by KVTransferConfig."""
    return request.param


@pytest.fixture(params=[False, True])
def lazy_offload(request: pytest.FixtureRequest) -> bool:
    """Exercise eager storage and actual FIFO lazy draining."""
    return request.param


@pytest.fixture
def mock_io(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stub transport, telemetry and GPU resources, keeping worker logic real."""
    scheduler = MagicMock(lmcache_tokens_per_chunk=4)
    scheduler.check_lookup_result.return_value = 0
    transfer = MagicMock()
    transfer.create_recorded_event.return_value = None
    for submit in (transfer.submit_store, transfer.submit_retrieve):
        submit.return_value.query.return_value = True
        submit.return_value.result.return_value = True
    telemetry = MagicMock()
    pool = MagicMock(blocks=[SimpleNamespace(block_hash=bytes([i])) for i in range(8)])
    monkeypatch.setattr(
        connector_mod, "LMCacheMPSchedulerAdapter", lambda **kw: scheduler
    )
    monkeypatch.setattr(connector_mod, "print_banner_once", MagicMock())
    monkeypatch.setattr(connector_mod, "vllm_layout_hints", lambda config: {})
    monkeypatch.setattr(adapter_mod.RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 4)
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", MagicMock())
    monkeypatch.setattr(adapter_mod, "set_ipc_policy", MagicMock())
    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", lambda *a, **kw: transfer
    )
    monkeypatch.setattr(
        adapter_mod.RequestTelemetryFactory, "create", lambda **kw: telemetry
    )
    return SimpleNamespace(
        scheduler=scheduler,
        transfer=transfer,
        telemetry=telemetry,
        pool=pool,
        kv_caches={"layer": torch.empty(2, 8, 4, 1, 8)},
    )


@pytest.fixture
def connectors(
    kv_role: str, lazy_offload: bool, mock_io: SimpleNamespace
) -> Iterator[tuple[LMCacheMPConnector, LMCacheMPConnector]]:
    """Construct real scheduler/worker connectors and a bound lazy manager."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role=kv_role,
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": lazy_offload,
                "lmcache.mp.lazy_offload_policy": "FIFO",
                "lmcache.mp.lazy_offload_threshold": 1,
                "lmcache.mp.lazy_offload_select_count": 1,
            },
        )
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    worker = LMCacheMPConnector(config, KVConnectorRole.WORKER)
    scheduler.bind_gpu_block_pool(mock_io.pool)
    worker.register_kv_caches(mock_io.kv_caches)
    try:
        yield scheduler, worker
    finally:
        worker.shutdown()
        scheduler.shutdown()


def test_chunked_prefill_stores_and_completion(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    kv_role: str,
    lazy_offload: bool,
    mock_io: SimpleNamespace,
) -> None:
    """Only writers store new/cached chunks and wait for real save completions."""
    scheduler, worker = connectors
    can_store = kv_role != "kv_consumer"
    request = _request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([1],)), 0
    )
    mock_io.transfer.submit_store.return_value.query.return_value = False
    for step in range(3):
        metadata = scheduler.build_connector_meta(
            _schedule(num_tokens=4, block_ids=([step + 1],), new=step == 0)
        )
        assert isinstance(metadata, LMCacheMPConnectorMetadata)
        tracker = scheduler.request_trackers[request.request_id]
        assert tracker.num_scheduled_tokens == (step + 1) * 4
        assert tracker.allocated_block_ids == {0: list(range(1, step + 2))}
        assert tracker.num_stored_tokens == ((step + 1) * 4 if can_store else 0)
        if can_store and not lazy_offload:
            assert len(metadata.requests) == 1
            meta = metadata.requests[0]
            assert (meta.direction, meta.op.start, meta.op.end, meta.op.block_ids) == (
                "STORE",
                step * 4,
                (step + 1) * 4,
                [[step + 1]],
            )
        else:
            assert metadata.requests == []
        worker.bind_connector_metadata(metadata)
        worker.wait_for_save()
        assert not any(worker.get_finished(set()))

    mock_io.scheduler.cleanup_lookup_result.reset_mock()
    assert scheduler.request_finished(request, [1, 2, 3]) == (
        can_store and not lazy_offload,
        None,
    )
    assert scheduler.request_trackers == {}
    mock_io.scheduler.cleanup_lookup_result.assert_called_once_with("request")

    if lazy_offload:
        # A later token step drains a writer's completed request, never a consumer's.
        trigger = _request("trigger")
        assert scheduler.get_num_new_matched_tokens(trigger, 0) == (0, False)
        scheduler.update_state_after_alloc(
            trigger, MagicMock(get_block_ids=lambda: ([4],)), 0
        )
        metadata = scheduler.build_connector_meta(
            _schedule("trigger", 1, ([4],), new=True)
        )
        assert len(metadata.requests) == int(can_store)
        if can_store:
            mock_io.scheduler.end_session.assert_not_called()
            meta = metadata.requests[0]
            assert (meta.request_id, meta.op.start, meta.op.end, meta.op.block_ids) == (
                "request",
                0,
                12,
                [[1, 2, 3]],
            )
        assert scheduler.has_pending_push_work() is can_store
        worker.bind_connector_metadata(metadata)
        worker.wait_for_save()

    assert not any(worker.get_finished({"request"}))
    mock_io.telemetry.on_request_store_finished.assert_not_called()
    mock_io.transfer.submit_store.return_value.query.return_value = True
    sending, receiving = worker.get_finished({"request"})
    assert (sending or set()) == (
        {"request"} if can_store and not lazy_offload else set()
    )
    assert not receiving
    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=worker.build_connector_worker_meta())
    )
    assert not scheduler.has_pending_push_work()
    mock_io.scheduler.end_session.assert_called_once_with("request")
    assert mock_io.transfer.submit_store.call_count == (
        (1 if lazy_offload else 3) if can_store else 0
    )
    if can_store:
        mock_io.telemetry.on_request_store_finished.assert_called_once_with(
            request_ids_set={"request"},
            model_name="test-model",
            world_size=1,
            kv_rank=0,
        )
    else:
        mock_io.telemetry.on_request_store_finished.assert_not_called()
    assert not any(worker.get_finished({"request"}))


@pytest.mark.parametrize("retrieve_success", [False, True])
def test_lookup_retrieve_and_cleanup(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    kv_role: str,
    lazy_offload: bool,
    mock_io: SimpleNamespace,
    retrieve_success: bool,
) -> None:
    """Every role retains lookup, receive completion/error reporting and stats."""
    scheduler, worker = connectors
    request = _request()
    request.kv_transfer_params = {"cached_token_stats": True}
    mock_io.scheduler.check_lookup_result.side_effect = [None, 8]
    assert scheduler.get_num_new_matched_tokens(request, 4) == (None, True)
    assert scheduler.get_num_new_matched_tokens(request, 4) == (4, True)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([1, 2],)), 4
    )
    metadata = scheduler.build_connector_meta(_schedule())
    assert len(metadata.requests) == 1
    meta = metadata.requests[0]
    assert (meta.direction, meta.op.start, meta.op.end, meta.op.block_ids) == (
        "RETRIEVE",
        4,
        8,
        [[2]],
    )
    worker.bind_connector_metadata(metadata)
    future = mock_io.transfer.submit_retrieve.return_value
    future.query.return_value = False
    future.result.return_value = retrieve_success
    worker.start_load_kv(MagicMock())
    with pytest.MonkeyPatch.context() as patcher:
        submit = MagicMock(wraps=worker.worker_adapter.batched_submit_store_requests)
        patcher.setattr(worker.worker_adapter, "batched_submit_store_requests", submit)
        worker.wait_for_save()
        submit.assert_not_called()
    mock_io.transfer.submit_store.assert_not_called()
    mock_io.transfer.submit_retrieve.assert_called_once()
    assert not any(worker.get_finished(set()))
    future.query.return_value = True
    sending, receiving = worker.get_finished(set())
    assert not sending
    assert receiving == {"request"}
    assert worker.get_block_ids_with_load_errors() == (
        set() if retrieve_success else {2}
    )
    assert not any(worker.get_finished(set()))
    mock_io.scheduler.cleanup_lookup_result.reset_mock()
    delay_free, params = scheduler.request_finished(request, [1, 2])
    assert delay_free is (kv_role != "kv_consumer" and not lazy_offload)
    assert params == {
        "cached_token_stats": {
            "num_vllm_cached_tokens": 4,
            "num_lmcache_cached_tokens": 8,
            "num_lmcache_extra_cached_tokens": 4,
        }
    }
    assert scheduler.request_trackers == {}
    mock_io.scheduler.cleanup_lookup_result.assert_called_once_with("request")
    mock_io.scheduler.end_session.assert_called_once_with("request")
    sending, receiving = worker.get_finished({"request"})
    assert (sending or set()) == ({"request"} if delay_free else set())
    assert not receiving


@pytest.mark.parametrize("mp_role", ["kv_both", "kv_consumer"])
@pytest.mark.parametrize("peer_delays_free", [False, True])
def test_multi_connector_child_role_and_completion(
    mp_role: str,
    peer_delays_free: bool,
    mock_io: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real MultiConnector construction honors child roles and combines completions."""
    scheduler_peer = MagicMock()
    scheduler_peer.get_num_new_matched_tokens.return_value = (0, False)
    scheduler_peer.request_finished.return_value = (
        peer_delays_free,
        {"remote_engine_id": "peer"},
    )
    worker_peer = MagicMock()
    worker_peer.get_finished.side_effect = [
        (None, None),
        ({"request"} if peer_delays_free else None, None),
    ]
    # MultiConnector polls children through get_transfer_results(), whose base
    # implementation wraps get_finished(). A bare MagicMock would answer that
    # call with a mock whose result sets iterate empty, silently dropping the
    # peer's completions, so route it through the real base implementation.
    worker_peer.get_transfer_results.side_effect = lambda finished_req_ids: (
        KVConnectorBase_V1.get_transfer_results(worker_peer, finished_req_ids)
    )
    peer_factory = MagicMock(side_effect=[scheduler_peer, worker_peer])

    def connector_class(config: KVTransferConfig) -> Any:
        return (
            LMCacheMPConnector
            if config.kv_connector == "LMCacheMPConnector"
            else peer_factory
        )

    monkeypatch.setattr(KVConnectorFactory, "get_connector_class", connector_class)
    config = _config(
        KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "NixlConnector", "kv_role": "kv_consumer"},
                    {"kv_connector": "LMCacheMPConnector", "kv_role": mp_role},
                ]
            },
        )
    )
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, None)
    worker = MultiConnector(config, KVConnectorRole.WORKER, None)
    worker.register_kv_caches(mock_io.kv_caches)
    request = _request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([1],)), 0
    )
    metadata = scheduler.build_connector_meta(
        _schedule(num_tokens=4, block_ids=([1],), new=True)
    )
    can_store = mp_role == "kv_both"
    assert len(metadata.metadata[1].requests) == int(can_store)
    worker.bind_connector_metadata(metadata)
    worker.wait_for_save()
    assert mock_io.transfer.submit_store.call_count == int(can_store)
    assert scheduler.request_finished(request, [1]) == (
        can_store or peer_delays_free,
        {"remote_engine_id": "peer"},
    )
    worker.bind_connector_metadata(scheduler.build_connector_meta(_schedule()))
    sending, receiving = worker.get_finished({"request"})
    assert (sending or set()) == (
        {"request"} if can_store and not peer_delays_free else set()
    )
    assert not receiving
    sending, receiving = worker.get_finished(set())
    assert (sending or set()) == ({"request"} if peer_delays_free else set())
    assert not receiving
    mock_io.scheduler.end_session.assert_called_once_with("request")
    worker.shutdown()
    scheduler.shutdown()
