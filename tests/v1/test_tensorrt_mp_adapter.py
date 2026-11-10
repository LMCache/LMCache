# SPDX-License-Identifier: Apache-2.0
"""CPU-only caller tests for the TensorRT-LLM multiprocess adapter."""

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
import importlib
import sys

# Third Party
import pytest


@pytest.fixture
def trt_mp_module(monkeypatch: pytest.MonkeyPatch):
    """Import the adapter with its optional TensorRT-LLM API stubbed."""

    class _ConnectorBase:
        def __init__(self, llm_args: object) -> None:
            self._llm_args = llm_args

    modules = {
        name: ModuleType(name)
        for name in (
            "tensorrt_llm",
            "tensorrt_llm._torch",
            "tensorrt_llm._torch.pyexecutor",
            "tensorrt_llm._torch.pyexecutor.connectors",
            "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector",
            "tensorrt_llm.bindings",
            "tensorrt_llm.bindings.internal",
            "tensorrt_llm.bindings.internal.batch_manager",
            "tensorrt_llm.llmapi",
            "tensorrt_llm.llmapi.llm_args",
        )
    }
    modules["tensorrt_llm"].mpi_rank = lambda: 0  # type: ignore[attr-defined]
    connector_module = modules[
        "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"
    ]
    connector_module.KvCacheConnectorScheduler = _ConnectorBase  # type: ignore[attr-defined]
    connector_module.KvCacheConnectorWorker = _ConnectorBase  # type: ignore[attr-defined]
    connector_module.SchedulerOutput = object  # type: ignore[attr-defined]
    modules["tensorrt_llm.bindings.internal.batch_manager"].LlmRequest = object  # type: ignore[attr-defined]
    modules["tensorrt_llm.llmapi.llm_args"].TorchLlmArgs = object  # type: ignore[attr-defined]

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "lmcache.integration.tensorrt_llm.tensorrt_mp_adapter"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_failed_retrieve_waits_for_device_result_and_fails_closed(
    trt_mp_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A False device transfer result cannot be treated as a loaded KV hit."""
    module = trt_mp_module
    worker = module.LMCacheMPKvConnectorWorker.__new__(
        module.LMCacheMPKvConnectorWorker
    )
    worker._metadata = module.LMCacheMPConnectorMetadata(
        loads={7: module._BlockSpec(tokens=[1, 2], block_ids=[3])}
    )
    worker._req_client = MagicMock(name="req_client")
    worker._mq_timeout = 5.0
    worker._instance_id = 42
    worker._device = "cpu"
    worker._create_key = MagicMock(return_value=SimpleNamespace())

    event = MagicMock(name="event")
    event.ipc_handle.return_value = b"producer-event"
    event_backend = MagicMock(name="event_backend")
    event_backend.create_event.return_value = event
    event_backend.export_event.side_effect = (
        lambda exported_event, device: exported_event.ipc_handle()
    )
    worker._event_backend = event_backend

    device_future = MagicMock(name="device_future")
    device_future.result.return_value = False
    raw_future = MagicMock(name="raw_future")
    raw_future.to_device_future.return_value = device_future
    worker._req_client.retrieve.return_value = raw_future

    with pytest.raises(RuntimeError, match="refusing to use unloaded KV blocks"):
        worker.start_load_kv(MagicMock(name="stream"))

    raw_future.to_device_future.assert_called_once_with(
        device="cpu",
        event_backend=event_backend,
    )
    raw_future.retain_reference.assert_called_once_with(event)
    device_future.result.assert_called_once_with(timeout=5.0)


def test_store_wait_retains_event_on_raw_future(
    trt_mp_module,
) -> None:
    """A timed store keeps its exported event alive on the raw future."""
    module = trt_mp_module
    worker = module.LMCacheMPKvConnectorWorker.__new__(
        module.LMCacheMPKvConnectorWorker
    )
    worker._metadata = module.LMCacheMPConnectorMetadata(
        saves={7: module._BlockSpec(tokens=[1, 2], block_ids=[3])}
    )
    worker._req_client = MagicMock(name="req_client")
    worker._mq_timeout = 5.0
    worker._instance_id = 42
    worker._device = "cpu"
    worker._create_key = MagicMock(return_value=SimpleNamespace())

    event = MagicMock(name="event")
    event.ipc_handle.return_value = b"producer-event"
    event_backend = MagicMock(name="event_backend")
    event_backend.create_event.return_value = event
    event_backend.export_event.side_effect = (
        lambda exported_event, device: exported_event.ipc_handle()
    )
    worker._event_backend = event_backend

    device_future = MagicMock(name="device_future")
    device_future.result.return_value = True
    raw_future = MagicMock(name="raw_future")
    raw_future.to_device_future.return_value = device_future
    worker._req_client.store.return_value = raw_future

    worker.wait_for_save(MagicMock(name="stream"))

    raw_future.to_device_future.assert_called_once_with(
        device="cpu",
        event_backend=event_backend,
    )
    raw_future.retain_reference.assert_called_once_with(event)
    device_future.result.assert_called_once_with(timeout=5.0)


def test_lookup_carries_trt_prefix_hit_hint(trt_mp_module) -> None:
    """The LOOKUP reports TensorRT-LLM's prefix-cache hit to the server's KV
    load policy."""
    module = trt_mp_module
    scheduler = module.LMCacheMPKvConnectorScheduler.__new__(
        module.LMCacheMPKvConnectorScheduler
    )
    scheduler._block_size = 4
    scheduler._chunk_size = 4
    scheduler._pending = {}
    scheduler._mq_timeout = 5.0
    scheduler._model_name = "m"
    scheduler._world_size = 1
    scheduler._req_client = MagicMock(name="req_client")
    scheduler._req_client.query_prefetch_status.return_value.result.return_value = 0
    request = SimpleNamespace(request_id=7, get_tokens=lambda beam: list(range(16)))

    assert scheduler.get_num_new_matched_tokens(request, 4) == (0, False)

    lookup_key = scheduler._req_client.lookup.call_args.args[0]
    assert lookup_key.request_configs == {"lmcache.hint.engine_computed_tokens": 4}


@pytest.mark.parametrize(
    ("num_computed", "cached", "start", "skip"),
    [(0, 80, 0, 0), (32, 80, 32, 0), (36, 80, 32, 4), (12, 16, 0, 12)],
    ids=[
        "no-trt-prefix",
        "trt-holds-two-chunks",
        "trt-prefix-mid-chunk",
        "hit-within-first-chunk",
    ],
)
def test_retrieve_covers_only_the_locked_range(
    trt_mp_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    num_computed: int,
    cached: int,
    start: int,
    skip: int,
) -> None:
    """A lookup hit frees the chunks inside TensorRT-LLM's prefix and
    retrieves the rest, so each lookup lock is released exactly once."""
    module = trt_mp_module
    client = MagicMock(spec=module.RequestClient)
    client.get_chunk_size.return_value.result.return_value = 16
    client.query_prefetch_status.return_value.result.return_value = cached
    monkeypatch.setattr(module.zmq, "Context", MagicMock())
    monkeypatch.setattr(
        module.RequestClientFactory, "create", MagicMock(return_value=client)
    )
    monkeypatch.setattr(module, "RawCudaIPCWrapper", MagicMock())
    monkeypatch.setattr(module, "get_event_ipc_backend", MagicMock())
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda name: SimpleNamespace(hidden_size=128, num_attention_heads=8),
    )
    llm_args = SimpleNamespace(
        kv_cache_config=SimpleNamespace(tokens_per_block=4),
        kv_connector_config=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        model="m",
    )
    scheduler = module.LMCacheMPKvConnectorScheduler(llm_args)
    worker = module.LMCacheMPKvConnectorWorker(llm_args)
    worker.register_kv_caches(module.torch.empty(32, 1, 2, 512))

    tokens = list(range(128))
    request = SimpleNamespace(request_id=7, get_tokens=lambda beam: tokens)
    assert scheduler.get_num_new_matched_tokens(request, num_computed) == (
        cached - num_computed,
        False,
    )
    new_request = SimpleNamespace(
        request_id=7,
        new_block_ids=list(range(32)),
        computed_position=num_computed,
        new_tokens=tokens,
    )
    worker._metadata = scheduler.build_connector_meta(
        SimpleNamespace(new_requests=[new_request])
    )
    worker.start_load_kv(MagicMock())

    if start:
        free_key = client.free_lookup_locks.call_args.args[0]
        assert (free_key.start, free_key.end) == (0, start)
    else:
        client.free_lookup_locks.assert_not_called()
    key, _, block_ids, _, skip_first_n_tokens = client.retrieve.call_args.args
    assert (key.start, key.end) == (start, cached)
    assert block_ids == [list(range(start // 4, cached // 4))]
    assert skip_first_n_tokens == skip
