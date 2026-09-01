# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the multi-process TensorRT-LLM adapter cleanup logic."""

# Standard
from typing import Any
from unittest.mock import MagicMock
import importlib
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.protocol import RequestType

_ADAPTER_MODULE = "lmcache.integration.tensorrt_llm.tensorrt_mp_adapter"
_PARENT_MODULE = "lmcache.integration.tensorrt_llm"
_PARENT_PACKAGE = "lmcache.integration"


def _build_tensorrt_llm_stub_modules() -> dict[str, types.ModuleType]:
    """Build fake ``tensorrt_llm`` modules for adapter import under test."""
    module_names = [
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
    ]
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["tensorrt_llm"].mpi_rank = lambda: 0  # type: ignore[attr-defined]

    class MockKvCacheConnector:
        """Mock TRT-LLM KV cache connector base class."""

        def __init__(self, llm_args: Any) -> None:
            self._llm_args = llm_args
            self._metadata = None

    kv_cache_connector_module = modules[
        "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"
    ]
    kv_cache_connector_module.KvCacheConnectorScheduler = (  # type: ignore[attr-defined]
        MockKvCacheConnector
    )
    kv_cache_connector_module.KvCacheConnectorWorker = (  # type: ignore[attr-defined]
        MockKvCacheConnector
    )
    kv_cache_connector_module.SchedulerOutput = MagicMock  # type: ignore[attr-defined]

    modules["tensorrt_llm.bindings.internal.batch_manager"].LlmRequest = (  # type: ignore[attr-defined]
        MagicMock
    )
    modules["tensorrt_llm.llmapi.llm_args"].TorchLlmArgs = MagicMock  # type: ignore[attr-defined]
    return modules


@pytest.fixture
def adapter_mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the adapter under stubbed ``tensorrt_llm`` modules.

    Returns:
        Any: The imported TensorRT-LLM MP adapter module.
    """
    for name, module in _build_tensorrt_llm_stub_modules().items():
        monkeypatch.setitem(sys.modules, name, module)

    parent_pkg = importlib.import_module(_PARENT_PACKAGE)
    monkeypatch.delattr(parent_pkg, "tensorrt_llm", raising=False)
    monkeypatch.delitem(sys.modules, _ADAPTER_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, _PARENT_MODULE, raising=False)

    return importlib.import_module(_ADAPTER_MODULE)


def test_worker_cleanup_unregisters_registered_worker(
    monkeypatch: pytest.MonkeyPatch,
    adapter_mod: Any,
) -> None:
    """Test LMCacheMPKvConnectorWorker cleanup when registered."""
    mq_client = MagicMock(name="mq_client")
    zmq_context = MagicMock(name="zmq_context")
    future = MagicMock(name="future")
    send_mock = MagicMock(name="_send_request", return_value=future)
    monkeypatch.setattr(adapter_mod, "_send_request", send_mock)

    worker = adapter_mod.LMCacheMPKvConnectorWorker.__new__(
        adapter_mod.LMCacheMPKvConnectorWorker
    )
    worker._closed = False
    worker._registered = True
    worker._instance_id = 123
    worker._mq_client = mq_client
    worker._zmq_context = zmq_context
    worker.__del__()

    send_mock.assert_called_once_with(
        mq_client,
        RequestType.UNREGISTER_KV_CACHE,
        [123],
    )
    future.result.assert_called_once_with(
        timeout=adapter_mod.DEFAULT_CLEANUP_UNREGISTER_TIMEOUT
    )
    mq_client.close.assert_called_once()
    zmq_context.destroy.assert_called_once_with(linger=0)

    send_mock.reset_mock()
    worker.__del__()
    send_mock.assert_not_called()


def test_worker_cleanup_unregistered_does_not_unregister(
    monkeypatch: pytest.MonkeyPatch,
    adapter_mod: Any,
) -> None:
    """Test LMCacheMPKvConnectorWorker cleanup when unregistered."""
    mq_client = MagicMock(name="mq_client")
    zmq_context = MagicMock(name="zmq_context")
    send_mock = MagicMock(name="_send_request")
    monkeypatch.setattr(adapter_mod, "_send_request", send_mock)

    worker = adapter_mod.LMCacheMPKvConnectorWorker.__new__(
        adapter_mod.LMCacheMPKvConnectorWorker
    )
    worker._closed = False
    worker._registered = False
    worker._mq_client = mq_client
    worker._zmq_context = zmq_context
    worker.__del__()

    send_mock.assert_not_called()
    mq_client.close.assert_called_once()
    zmq_context.destroy.assert_called_once_with(linger=0)


def test_cleanup_handles_partial_init(adapter_mod: Any) -> None:
    """_cleanup() must not raise when __init__ failed before all attrs exist."""
    scheduler = adapter_mod.LMCacheMPKvConnectorScheduler.__new__(
        adapter_mod.LMCacheMPKvConnectorScheduler
    )
    scheduler._zmq_context = MagicMock(name="zmq_context")
    scheduler._cleanup()
    scheduler._zmq_context.destroy.assert_called_once_with(linger=0)

    worker = adapter_mod.LMCacheMPKvConnectorWorker.__new__(
        adapter_mod.LMCacheMPKvConnectorWorker
    )
    worker._cleanup()
