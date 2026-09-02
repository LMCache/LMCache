# SPDX-License-Identifier: Apache-2.0
"""CPU-only caller tests for the TensorRT-LLM multiprocess adapter."""

# Standard
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import importlib
import sys

# Third Party
import pytest


def _make_worker(
    trt_mp_module: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, MagicMock]:
    """Construct a worker through its public constructor with an MQ stub."""
    module = trt_mp_module
    monkeypatch.setattr(
        module.zmq, "Context", MagicMock(return_value=MagicMock(name="zmq_context"))
    )
    mq_client = MagicMock(name="mq_client")
    chunk_future = MagicMock(name="chunk_future")
    chunk_future.result.return_value = 256
    mq_client.submit_request.return_value = chunk_future
    monkeypatch.setattr(
        module,
        "MessageQueueClient",
        MagicMock(return_value=mq_client),
    )
    llm_args = SimpleNamespace(
        kv_cache_config=SimpleNamespace(tokens_per_block=32),
        kv_connector_config=SimpleNamespace(server_url=None),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        model="test-model",
    )
    worker = module.LMCacheMPKvConnectorWorker(llm_args)
    mq_client.submit_request.reset_mock()
    return worker, mq_client.submit_request


def _successful_transfer_future(name: str) -> MagicMock:
    """Build a completed device-transfer future for the worker test."""
    device_future = MagicMock(name=f"{name}_device_future")
    device_future.result.return_value = True
    transfer_future = MagicMock(name=f"{name}_future")
    transfer_future.to_device_future.return_value = device_future
    return transfer_future


@pytest.fixture
def trt_mp_module(monkeypatch: pytest.MonkeyPatch):
    """Import the adapter with its optional TensorRT-LLM API stubbed."""

    class _ConnectorBase:
        def __init__(self, llm_args: object) -> None:
            self._llm_args = llm_args

        def bind_connector_meta(self, metadata: object) -> None:
            """Bind metadata supplied by the TRT-LLM scheduler."""
            self._metadata = metadata

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
    key = module.IPCCacheServerKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        num_kv_readers=1,
        token_ids=tuple(range(32)),
        start=0,
        end=32,
        request_id="7",
    )
    worker._metadata = module.LMCacheMPConnectorMetadata(
        loads={7: module._BlockSpec(tokens=list(range(32)), block_ids=[3])}
    )
    worker._block_size = 32
    worker._mq_client = MagicMock(name="mq_client")
    worker._mq_timeout = 5.0
    worker._instance_id = 42
    worker._create_key = MagicMock(return_value=key)

    event = MagicMock(name="event")
    event.ipc_handle.return_value = b"producer-event"
    monkeypatch.setattr(module, "check_interprocess_event_support", MagicMock())
    monkeypatch.setattr(module.torch_dev, "Event", MagicMock(return_value=event))

    device_future = MagicMock(name="device_future")
    device_future.result.return_value = False
    raw_future = MagicMock(name="raw_future")
    raw_future.to_device_future.return_value = device_future
    send_request = MagicMock(return_value=raw_future)
    monkeypatch.setattr(module, "_send_request", send_request)

    with pytest.raises(RuntimeError, match="refusing to use unloaded KV blocks"):
        worker.start_load_kv(MagicMock(name="stream"))

    raw_future.to_device_future.assert_called_once_with()
    device_future.result.assert_called_once_with(timeout=5.0)


@pytest.mark.parametrize(
    ("token_count", "block_count", "expected_block_count"),
    [
        pytest.param(2176, 68, 64, id="partial-final-chunk"),
        pytest.param(2048, 64, 64, id="chunk-aligned"),
        pytest.param(128, 4, 0, id="short-request"),
        pytest.param(2176, 63, 63, id="short-block-list-reaches-server"),
    ],
)
def test_transfer_block_ids_match_key_range(
    trt_mp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    token_count: int,
    block_count: int,
    expected_block_count: int,
) -> None:
    """STORE and RETRIEVE use whole key chunks without mutating metadata."""
    module = trt_mp_module
    worker, submit_request = _make_worker(trt_mp_module, monkeypatch)
    tokens = list(range(token_count))
    block_ids = list(range(block_count))
    spec = SimpleNamespace(tokens=tokens, block_ids=block_ids)
    worker.bind_connector_meta(
        module.LMCacheMPConnectorMetadata(
            loads={7: spec},
            saves={7: spec},
        )
    )

    event = MagicMock(name="event")
    event.ipc_handle.return_value = b"producer-event"
    monkeypatch.setattr(module, "check_interprocess_event_support", MagicMock())
    monkeypatch.setattr(module.torch_dev, "Event", MagicMock(return_value=event))
    submit_request.side_effect = [
        _successful_transfer_future("retrieve"),
        _successful_transfer_future("store"),
    ]

    worker.start_load_kv(MagicMock(name="load_stream"))
    worker.wait_for_save(MagicMock(name="store_stream"))

    assert block_ids == list(range(block_count))
    if expected_block_count == 0:
        assert submit_request.call_count == 0
        return

    assert submit_request.call_count == 2
    retrieve_call, store_call = submit_request.call_args_list
    assert retrieve_call.args[0] == module.RequestType.RETRIEVE
    assert store_call.args[0] == module.RequestType.STORE
    retrieve_payload = retrieve_call.args[1]
    store_payload = store_call.args[1]
    expected_end = (token_count // 256) * 256
    assert retrieve_payload[0].start == 0
    assert retrieve_payload[0].end == expected_end
    assert retrieve_payload[2] == [list(range(expected_block_count))]
    assert store_payload[0].start == 0
    assert store_payload[0].end == expected_end
    assert store_payload[2] == [list(range(expected_block_count))]
