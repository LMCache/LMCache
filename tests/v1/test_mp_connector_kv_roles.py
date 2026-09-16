# SPDX-License-Identifier: Apache-2.0
"""Role handling tests for the vLLM MP connector."""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (  # noqa: E402
    MultiConnector,
)

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)


def _make_vllm_config(kv_transfer_config: KVTransferConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(model="test-model", use_mla=False),
        parallel_config=SimpleNamespace(
            world_size=1,
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(block_size=4, enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        kv_transfer_config=kv_transfer_config,
    )


def _make_connector(
    kv_role: str,
    lazy_offload: bool,
    connector_role: KVConnectorRole = KVConnectorRole.SCHEDULER,
) -> LMCacheMPConnector:
    config = _make_vllm_config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role=kv_role,
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": lazy_offload,
            },
        )
    )
    scheduler_adapter = MagicMock(lmcache_tokens_per_chunk=4)
    worker_adapter = MagicMock()
    with (
        patch(
            "lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPSchedulerAdapter",
            return_value=scheduler_adapter,
        ),
        patch(
            "lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPWorkerAdapter",
            return_value=worker_adapter,
        ),
        patch("lmcache.integration.vllm.lmcache_mp_connector.print_banner_once"),
    ):
        return LMCacheMPConnector(config, connector_role)


def _make_tracker(request_id: str, block_ids: list[int]) -> LMCacheMPRequestTracker:
    request = SimpleNamespace(
        request_id=request_id,
        cache_salt=None,
        all_token_ids=list(range(8)),
        prompt_token_ids=list(range(8)),
        mm_features=[],
        sampling_params=None,
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: block_ids}
    return tracker


@pytest.mark.parametrize(
    "kv_role,can_store",
    [("kv_consumer", False), ("kv_producer", True), ("kv_both", True)],
)
@pytest.mark.parametrize("lazy_offload", [False, True])
def test_role_gates_new_and_cached_stores_after_accounting(
    kv_role: str,
    can_store: bool,
    lazy_offload: bool,
) -> None:
    """Consumers update request state without creating eager or lazy stores."""
    connector = _make_connector(kv_role, lazy_offload)
    lazy_manager = MagicMock()
    connector._lazy_offload_manager = lazy_manager
    new_tracker = _make_tracker("new", [10])
    cached_tracker = _make_tracker("cached", [])
    connector.request_trackers = {
        "new": new_tracker,
        "cached": cached_tracker,
    }
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="new")],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["cached"],
            new_block_ids=[([20],)],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={"new": 4, "cached": 4},
    )
    metadata = LMCacheMPConnectorMetadata()

    connector._process_new_requests(scheduler_output, metadata)
    connector._process_cached_requests(scheduler_output, metadata)

    assert connector._can_store is can_store
    assert new_tracker.num_scheduled_tokens == 4
    assert cached_tracker.num_scheduled_tokens == 4
    assert cached_tracker.allocated_block_ids == {0: [20]}
    if not can_store:
        assert metadata.requests == []
        lazy_manager.add_store_candidate.assert_not_called()
        assert new_tracker.num_stored_tokens == 0
        assert cached_tracker.num_stored_tokens == 0
    elif lazy_offload:
        assert metadata.requests == []
        assert lazy_manager.add_store_candidate.call_count == 2
    else:
        assert [request.direction for request in metadata.requests] == [
            "STORE",
            "STORE",
        ]
        lazy_manager.add_store_candidate.assert_not_called()


def test_consumer_retrieval_metadata_submits_only_retrieve() -> None:
    """A consumer retrieval reaches the worker without a store submission."""
    scheduler_connector = _make_connector("kv_consumer", False)
    tracker = _make_tracker("retrieve", [30])
    tracker.num_lmcache_hit_tokens = 4
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    scheduler_connector.request_trackers = {"retrieve": tracker}
    metadata = LMCacheMPConnectorMetadata()

    scheduler_connector._process_retrieve_requests(metadata)

    worker_connector = _make_connector(
        "kv_consumer", False, connector_role=KVConnectorRole.WORKER
    )
    worker_connector._connector_metadata = metadata
    worker_connector.start_load_kv(SimpleNamespace())
    worker_connector.wait_for_save()

    assert [request.direction for request in metadata.requests] == ["RETRIEVE"]
    worker_connector.worker_adapter.batched_submit_retrieve_requests.assert_called_once()
    worker_connector.worker_adapter.batched_submit_store_requests.assert_not_called()


@pytest.mark.parametrize(
    "kv_role,finished_store_ids",
    [
        ("kv_consumer", set()),
        ("kv_producer", {"finished"}),
        ("kv_both", {"finished"}),
    ],
)
def test_get_finished_only_forwards_engine_completions_for_writers(
    kv_role: str,
    finished_store_ids: set[str],
) -> None:
    """Consumers still poll retrieves without reporting phantom stores."""
    connector = _make_connector(kv_role, False, connector_role=KVConnectorRole.WORKER)
    connector.worker_adapter.get_finished.return_value = (set(), {"retrieved"})

    result = connector.get_finished({"finished"})

    connector.worker_adapter.get_finished.assert_called_once_with(finished_store_ids)
    assert result == (set(), {"retrieved"})


@pytest.mark.parametrize(
    "kv_role,can_store",
    [("kv_consumer", False), ("kv_producer", True), ("kv_both", True)],
)
@pytest.mark.parametrize("lazy_offload", [False, True])
def test_request_finished_preserves_cleanup_and_store_wait(
    kv_role: str,
    can_store: bool,
    lazy_offload: bool,
) -> None:
    """Completion always cleans up, but only non-lazy writers delay frees."""
    connector = _make_connector(kv_role, lazy_offload)
    tracker = _make_tracker("request", [40])
    tracker.num_vllm_hit_tokens = 2
    tracker.num_lmcache_hit_tokens = 4
    connector.request_trackers = {"request": tracker}
    lazy_manager = MagicMock()
    lazy_manager.on_request_finished.return_value = SimpleNamespace(
        sessions_to_end=["request"]
    )
    connector._lazy_offload_manager = lazy_manager
    request = SimpleNamespace(
        request_id="request",
        kv_transfer_params={"cached_token_stats": True},
    )

    delay_free, return_params = connector.request_finished(request, [40])

    assert delay_free is (can_store and not lazy_offload)
    assert return_params == {
        "cached_token_stats": {
            "num_vllm_cached_tokens": 2,
            "num_lmcache_cached_tokens": 4,
            "num_lmcache_extra_cached_tokens": 2,
        }
    }
    assert connector.request_trackers == {}
    scheduler_adapter = cast(MagicMock, connector.scheduler_adapter)
    scheduler_adapter.cleanup_lookup_result.assert_called_once_with("request")
    scheduler_adapter.end_session.assert_called_once_with("request")
    if lazy_offload:
        lazy_manager.on_request_finished.assert_called_once_with("request")
    else:
        lazy_manager.on_request_finished.assert_not_called()


def test_nested_mp_role_is_not_inherited_from_outer_consumer() -> None:
    """MultiConnector constructs the MP child with its own kv_both role."""
    outer_config = _make_vllm_config(
        KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "NixlConnector",
                        "kv_role": "kv_consumer",
                    },
                    {
                        "kv_connector": "LMCacheMPConnector",
                        "kv_role": "kv_both",
                    },
                ]
            },
        )
    )

    def connector_class(config: KVTransferConfig) -> type:
        return (
            LMCacheMPConnector
            if config.kv_connector == "LMCacheMPConnector"
            else MagicMock
        )

    with patch.object(
        KVConnectorFactory,
        "get_connector_class",
        side_effect=connector_class,
    ):
        child_configs = MultiConnector._get_connector_classes_and_configs(outer_config)

    mp_config = next(
        config for connector, config in child_configs if connector is LMCacheMPConnector
    )
    scheduler_adapter = MagicMock(lmcache_tokens_per_chunk=4)
    with (
        patch(
            "lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPSchedulerAdapter",
            return_value=scheduler_adapter,
        ),
        patch("lmcache.integration.vllm.lmcache_mp_connector.print_banner_once"),
    ):
        mp_connector = LMCacheMPConnector(mp_config, KVConnectorRole.SCHEDULER)

    tracker = _make_tracker("nested", [50])
    mp_connector.request_trackers = {"nested": tracker}
    metadata = LMCacheMPConnectorMetadata()
    mp_connector._process_new_requests(
        SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id="nested")],
            num_scheduled_tokens={"nested": 4},
        ),
        metadata,
    )

    assert outer_config.kv_transfer_config.is_kv_producer is False
    assert mp_config.kv_transfer_config.kv_role == "kv_both"
    assert mp_connector._can_store is True
    assert [request.direction for request in metadata.requests] == ["STORE"]
