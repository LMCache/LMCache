# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the standalone DeepInfer multiprocess adapter."""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.deepinfer import multi_process_adapter as adapter
from lmcache.integration.deepinfer.multi_process_adapter import (
    DeepInferMPParallelConfig,
    DeepInferMPTransferSpec,
    DeepInferMPWorkerAdapter,
)
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
import lmcache.lmcache_native as lmcache_native


class _FakeHeartbeatThread:
    """No-thread heartbeat double used by the registration test."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args
        del kwargs
        self.stop_requested = False

    def register_recover_callback(self, callback: Any) -> None:
        """Accept the recovery callback without starting a thread."""
        del callback

    def register_unhealthy_callback(self, callback: Any) -> None:
        """Accept the unhealthy callback without starting a thread."""
        del callback

    def register_healthy_callback(self, callback: Any) -> None:
        """Accept the healthy callback without starting a thread."""
        del callback

    def start(self) -> None:
        """Represent a successfully started heartbeat."""

    def stop_and_wait(self) -> None:
        """Represent a synchronously stopped heartbeat."""
        self.stop_requested = True


def test_deepinfer_detector_recognizes_registered_page_views() -> None:
    """DeepInfer owns format discovery for its three-dimensional page views."""
    caches = [
        torch.empty(4, 64, 68, dtype=torch.uint8),
        torch.empty(4, 64, 288, dtype=torch.uint8),
    ]

    detected, normalized = detect_format(caches, EngineType.DEEPINFER, {})

    assert detected is lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    assert normalized is caches


def test_worker_registration_and_store_use_deepinfer_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration and event ordering stay in the DeepInfer adapter."""
    client = MagicMock(name="request_client")
    transfer_context = MagicMock(name="transfer_context")
    factory = MagicMock(name="request_client_factory")
    factory.create.return_value = client
    monkeypatch.setattr(adapter, "RequestClientFactory", factory)
    monkeypatch.setattr(adapter, "_get_chunk_size", lambda *args: 128)
    monkeypatch.setattr(
        adapter,
        "create_transfer_context",
        lambda *args, **kwargs: transfer_context,
    )
    monkeypatch.setattr(adapter, "_HeartbeatThread", _FakeHeartbeatThread)
    worker = DeepInferMPWorkerAdapter(
        server_url="tcp://127.0.0.1:5555",
        context=MagicMock(name="zmq_context"),
        model_name="deepinfer-test-model",
        block_size=128,
        parallel_config=DeepInferMPParallelConfig.for_model(2, 1, is_mla=True),
        transfer_mode="lmcache_driven",
    )
    caches = {"layer.0.ica_cache": torch.empty(4, 64, 68)}
    groups = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0,),
            tokens_per_block=128,
        )
    ]

    worker.register_kv_caches(caches, engine_group_infos=groups)

    register_call = transfer_context.register.call_args
    assert register_call.args[2] == 1
    assert register_call.kwargs["engine_type"] is EngineType.DEEPINFER
    assert register_call.kwargs["engine_group_infos"] == groups

    event = object()
    future = MagicMock(name="store_future")
    transfer_context.create_recorded_event.return_value = event
    transfer_context.submit_store.return_value = future
    spec = DeepInferMPTransferSpec(
        token_ids=list(range(128)),
        block_ids=[[1]],
        start=0,
        end=128,
    )
    assert worker.submit_store_request("request-1", spec) is future
    transfer_context.create_recorded_event.assert_called_once_with()
    store_key = transfer_context.submit_store.call_args.args[1]
    assert store_key.world_size == 1
    assert store_key.worker_id == 0
    assert store_key.num_kv_readers == 2
    future.retain_reference.assert_called_once_with(event)

    worker.release_lookup_reader("request-1", spec)
    release_key = client.free_lookup_locks.call_args.args[0]
    assert release_key.world_size == 1
    assert release_key.worker_id == 0
    assert release_key.num_kv_readers == 1
    worker.shutdown()
