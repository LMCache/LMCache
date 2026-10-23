# SPDX-License-Identifier: Apache-2.0
"""Keep token-drop requests out of the LMCache STORE and lazy-offload paths."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.experimental.token_drop import (  # noqa: E402
    TokenDropSpec,
)
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadManager,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)


@pytest.mark.parametrize("phase", ["new", "cached"])
def test_token_drop_does_not_generate_store_metadata(phase, monkeypatch) -> None:
    tracker = SimpleNamespace(
        token_drop_spec=TokenDropSpec("rkv", {}),
        increase_num_scheduled_tokens=MagicMock(),
        append_block_ids=MagicMock(),
    )
    store_builder = MagicMock()
    monkeypatch.setattr(LMCacheMPRequestMetadata, "GetStoreMetadata", store_builder)
    connector = SimpleNamespace(
        scheduler_adapter=SimpleNamespace(lmcache_tokens_per_chunk=16),
        _get_request_tracker=lambda _: tracker,
        _can_store=True,
        lazy_offload=True,
        _mamba_relocation_window=0,
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="req")] if phase == "new" else [],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req"] if phase == "cached" else [],
            new_block_ids=[([3],)] if phase == "cached" else [],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={"req": 2},
    )
    metadata = MagicMock()
    if phase == "new":
        LMCacheMPConnector._process_new_requests(connector, output, metadata)  # type: ignore[arg-type]
    else:
        LMCacheMPConnector._process_cached_requests(connector, output, metadata)  # type: ignore[arg-type]
    tracker.increase_num_scheduled_tokens.assert_called_once_with(2)
    store_builder.assert_not_called()
    metadata.add_request_metadata.assert_not_called()


def make_request():
    return SimpleNamespace(
        request_id="req",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        mm_features=[],
        sampling_params=SimpleNamespace(
            extra_args={
                "kv_transfer_params": {
                    "lmcache.token_drop": {"algorithm": "rkv", "config": {}}
                }
            }
        ),
    )


def test_token_drop_registers_for_lazy_offload_lifecycle_only() -> None:
    manager = MagicMock()
    connector = SimpleNamespace(
        request_trackers={},
        lazy_offload=True,
        _lazy_offload_manager=manager,
        scheduler_adapter=MagicMock(),
    )
    tracker = LMCacheMPConnector._get_or_create_request_tracker(
        connector,  # type: ignore[arg-type]
        make_request(),
    )
    assert isinstance(tracker, LMCacheMPRequestTracker)
    assert tracker.token_drop_spec == TokenDropSpec("rkv", {})
    manager.on_request_arrived.assert_called_once_with("req")


def test_token_drop_finish_runs_lazy_registry_cleanup() -> None:
    manager = MagicMock()
    connector = SimpleNamespace(
        request_trackers={
            "req": SimpleNamespace(
                token_drop_spec=TokenDropSpec("rkv", {}),
            )
        },
        _cleanup_request_tracker=MagicMock(),
        scheduler_adapter=MagicMock(),
        lazy_offload=True,
        _lazy_offload_manager=manager,
        _can_store=True,
    )
    request = SimpleNamespace(request_id="req", kv_transfer_params=None)
    assert LMCacheMPConnector.request_finished(connector, request, []) == (False, None)  # type: ignore[arg-type]
    connector._cleanup_request_tracker.assert_called_once_with("req")
    manager.on_request_finished.assert_called_once_with("req")
    connector.scheduler_adapter.end_session.assert_not_called()


def test_normal_request_still_enters_lazy_offload_registry() -> None:
    manager = MagicMock()
    connector = SimpleNamespace(
        request_trackers={},
        lazy_offload=True,
        _lazy_offload_manager=manager,
        scheduler_adapter=MagicMock(),
    )
    request = make_request()
    request.sampling_params.extra_args = None
    tracker = LMCacheMPConnector._get_or_create_request_tracker(connector, request)  # type: ignore[arg-type]
    assert tracker.token_drop_spec is None
    manager.on_request_arrived.assert_called_once_with("req")


def test_token_drop_preemption_resets_lazy_offload_registry() -> None:
    manager = MagicMock()
    connector = SimpleNamespace(
        request_trackers={},
        lazy_offload=True,
        _lazy_offload_manager=manager,
        scheduler_adapter=MagicMock(),
    )
    request = make_request()
    tracker = LMCacheMPConnector._get_or_create_request_tracker(connector, request)  # type: ignore[arg-type]
    tracker.state = LMCacheMPRequestState.READY
    request.status = RequestStatus.PREEMPTED
    LMCacheMPConnector._get_or_create_request_tracker(connector, request)  # type: ignore[arg-type]
    manager.on_request_reset.assert_called_once_with("req")
    assert manager.on_request_arrived.call_count == 2


def test_normal_pending_store_is_discarded_when_token_drop_reuses_id() -> None:
    policy = MagicMock()
    policy.has_pending_request.side_effect = [True, False]
    manager = LazyOffloadManager({}, [16], MagicMock())
    manager._policy = policy
    connector = SimpleNamespace(
        request_trackers={},
        lazy_offload=True,
        _lazy_offload_manager=manager,
        scheduler_adapter=MagicMock(),
        _can_store=True,
    )
    connector._cleanup_request_tracker = lambda rid: connector.request_trackers.pop(
        rid, None
    )

    normal = make_request()
    normal.sampling_params.extra_args = None
    LMCacheMPConnector._get_or_create_request_tracker(connector, normal)  # type: ignore[arg-type]
    manager._token_ledgers["req"] = [1, 2, 3]
    assert LMCacheMPConnector.request_finished(connector, normal, []) == (False, None)  # type: ignore[arg-type]
    assert manager._requests.is_finished("req")
    connector.scheduler_adapter.end_session.assert_not_called()

    token_drop = make_request()
    LMCacheMPConnector._get_or_create_request_tracker(connector, token_drop)  # type: ignore[arg-type]
    policy.discard_for_reuse.assert_called_once_with("req")
    assert "req" not in manager._token_ledgers
    assert not manager._requests.is_finished("req")
    connector.scheduler_adapter.end_session.assert_called_once_with("req")

    assert LMCacheMPConnector.request_finished(connector, token_drop, []) == (  # type: ignore[arg-type]
        False,
        None,
    )
    assert "req" not in manager._requests._slots
    assert connector.scheduler_adapter.end_session.call_count == 2


def test_inflight_normal_store_is_orphaned_when_token_drop_reuses_id() -> None:
    completion = MagicMock()
    completion.update_pending_store_count.return_value = True
    policy = MagicMock()
    policy.has_pending_request.return_value = False
    manager = LazyOffloadManager({}, [16], completion)
    manager._policy = policy
    pool = MagicMock()
    manager._gpu_block_pool = pool
    manager.on_request_arrived("req")
    manager._requests.register_batch("req", [7])
    assert manager.on_request_finished("req").sessions_to_end == []

    assert manager.on_request_arrived("req").sessions_to_end == []
    policy.discard_for_reuse.assert_called_once_with("req")
    assert manager._requests.in_flight_is_orphaned("req")

    assert manager.on_store_results(set(), {"req": 1}).sessions_to_end == []
    pool.free_blocks.assert_called_once_with([pool.blocks[7]])
    assert manager.on_request_finished("req").sessions_to_end == ["req"]
    assert "req" not in manager._requests._slots
