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
    worker._mq_client = MagicMock(name="mq_client")
    worker._mq_timeout = 5.0
    worker._instance_id = 42
    worker._create_key = MagicMock(return_value=SimpleNamespace())

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
