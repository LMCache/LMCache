# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the multi-process TensorRT-LLM adapter cleanup logic."""

# Standard
from typing import Any
from unittest.mock import MagicMock, patch
import sys
import types

# Mock tensorrt_llm before any other imports to prevent ImportError
tb_module: Any = types.ModuleType("tensorrt_llm")
tb_module.mpi_rank = lambda: 0
sys.modules["tensorrt_llm"] = tb_module

torch_module: Any = types.ModuleType("tensorrt_llm._torch")
sys.modules["tensorrt_llm._torch"] = torch_module

pyexecutor_module: Any = types.ModuleType("tensorrt_llm._torch.pyexecutor")
sys.modules["tensorrt_llm._torch.pyexecutor"] = pyexecutor_module

connectors_module: Any = types.ModuleType("tensorrt_llm._torch.pyexecutor.connectors")
sys.modules["tensorrt_llm._torch.pyexecutor.connectors"] = connectors_module

kv_cache_connector_module: Any = types.ModuleType(
    "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"
)
sys.modules["tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"] = (
    kv_cache_connector_module
)


class MockKvCacheConnectorScheduler:
    """Mock KvCacheConnectorScheduler base class."""

    def __init__(self, llm_args: Any) -> None:
        self._llm_args = llm_args


class MockKvCacheConnectorWorker:
    """Mock KvCacheConnectorWorker base class."""

    def __init__(self, llm_args: Any) -> None:
        self._llm_args = llm_args
        self._metadata = None


kv_cache_connector_module.KvCacheConnectorScheduler = MockKvCacheConnectorScheduler
kv_cache_connector_module.KvCacheConnectorWorker = MockKvCacheConnectorWorker
kv_cache_connector_module.SchedulerOutput = MagicMock

bindings_module: Any = types.ModuleType("tensorrt_llm.bindings")
sys.modules["tensorrt_llm.bindings"] = bindings_module
internal_module: Any = types.ModuleType("tensorrt_llm.bindings.internal")
sys.modules["tensorrt_llm.bindings.internal"] = internal_module
batch_manager_module: Any = types.ModuleType(
    "tensorrt_llm.bindings.internal.batch_manager"
)
sys.modules["tensorrt_llm.bindings.internal.batch_manager"] = batch_manager_module
batch_manager_module.LlmRequest = MagicMock

llmapi_module: Any = types.ModuleType("tensorrt_llm.llmapi")
sys.modules["tensorrt_llm.llmapi"] = llmapi_module
llm_args_module: Any = types.ModuleType("tensorrt_llm.llmapi.llm_args")
sys.modules["tensorrt_llm.llmapi.llm_args"] = llm_args_module
llm_args_module.TorchLlmArgs = MagicMock

# Third Party
import pytest  # noqa: E402

# First Party
from lmcache.integration.tensorrt_llm.tensorrt_mp_adapter import (  # noqa: E402
    LMCacheMPKvConnectorScheduler,
    LMCacheMPKvConnectorWorker,
)
from lmcache.v1.multiprocess.protocol import RequestType  # noqa: E402


@pytest.fixture
def mock_llm_args() -> MagicMock:
    """Fixture to provide mocked llm_args.

    Returns:
        MagicMock: The mocked llm_args object.
    """
    args = MagicMock()
    args.kv_cache_config.tokens_per_block = 16
    args.tensor_parallel_size = 1
    args.pipeline_parallel_size = 1
    args.model = "mock_model"
    args.kv_connector_config = MagicMock()
    args.kv_connector_config.server_url = "ipc:///tmp/mock"
    return args


@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter.zmq.Context")
@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter.MessageQueueClient")
@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter._send_request")
def test_scheduler_cleanup(
    mock_send_request: MagicMock,
    mock_mq_client_cls: MagicMock,
    mock_zmq_context_cls: MagicMock,
    mock_llm_args: MagicMock,
) -> None:
    """Test LMCacheMPKvConnectorScheduler close and shutdown functionality.

    Args:
        mock_send_request: Mocked _send_request function.
        mock_mq_client_cls: Mocked MessageQueueClient class.
        mock_zmq_context_cls: Mocked zmq.Context class.
        mock_llm_args: Mocked llm_args fixture.
    """
    # Mock future returned by _send_request for GET_CHUNK_SIZE
    mock_future = MagicMock()
    mock_future.result.return_value = 256
    mock_send_request.return_value = mock_future

    scheduler = LMCacheMPKvConnectorScheduler(mock_llm_args)

    assert scheduler._mq_client is not None
    assert scheduler._zmq_context is not None

    mq_client: Any = scheduler._mq_client
    zmq_context: Any = scheduler._zmq_context

    # Call shutdown
    scheduler.shutdown()

    # Verify closed
    mq_client.close.assert_called_once()
    zmq_context.destroy.assert_called_once_with(linger=0)

    # Calling shutdown again is safe (idempotency)
    scheduler.shutdown()


@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter.zmq.Context")
@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter.MessageQueueClient")
@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter._send_request")
def test_worker_cleanup(
    mock_send_request: MagicMock,
    mock_mq_client_cls: MagicMock,
    mock_zmq_context_cls: MagicMock,
    mock_llm_args: MagicMock,
) -> None:
    """Test LMCacheMPKvConnectorWorker cleanup when registered.

    Args:
        mock_send_request: Mocked _send_request function.
        mock_mq_client_cls: Mocked MessageQueueClient class.
        mock_zmq_context_cls: Mocked zmq.Context class.
        mock_llm_args: Mocked llm_args fixture.
    """
    mock_future = MagicMock()
    mock_future.result.return_value = 256
    mock_send_request.return_value = mock_future

    worker = LMCacheMPKvConnectorWorker(mock_llm_args)

    # Mock registration
    worker._registered = True

    mq_client: Any = worker._mq_client
    zmq_context: Any = worker._zmq_context
    instance_id = worker._instance_id

    # Call shutdown
    worker.shutdown()

    # Verify unregister request was sent
    mock_send_request.assert_any_call(
        mq_client,
        RequestType.UNREGISTER_KV_CACHE,
        [instance_id],
    )

    # Verify closed
    mq_client.close.assert_called_once()
    zmq_context.destroy.assert_called_once_with(linger=0)

    # Calling shutdown again does not call unregister or close again
    mock_send_request.reset_mock()
    worker.shutdown()
    mock_send_request.assert_not_called()


@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter.zmq.Context")
@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter.MessageQueueClient")
@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter._send_request")
def test_worker_cleanup_unregistered(
    mock_send_request: MagicMock,
    mock_mq_client_cls: MagicMock,
    mock_zmq_context_cls: MagicMock,
    mock_llm_args: MagicMock,
) -> None:
    """Test LMCacheMPKvConnectorWorker cleanup when unregistered.

    Args:
        mock_send_request: Mocked _send_request function.
        mock_mq_client_cls: Mocked MessageQueueClient class.
        mock_zmq_context_cls: Mocked zmq.Context class.
        mock_llm_args: Mocked llm_args fixture.
    """
    mock_future = MagicMock()
    mock_future.result.return_value = 256
    mock_send_request.return_value = mock_future

    worker = LMCacheMPKvConnectorWorker(mock_llm_args)

    # Ensure unregistered
    worker._registered = False

    mq_client: Any = worker._mq_client
    zmq_context: Any = worker._zmq_context

    # Call shutdown
    worker.shutdown()

    # Verify unregister request was NOT sent
    for call in mock_send_request.call_args_list:
        assert call[0][1] != RequestType.UNREGISTER_KV_CACHE

    # Verify closed
    mq_client.close.assert_called_once()
    zmq_context.destroy.assert_called_once_with(linger=0)


@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter.zmq.Context")
@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter.MessageQueueClient")
@patch("lmcache.integration.tensorrt_llm.tensorrt_mp_adapter._send_request")
def test_destructors(
    mock_send_request: MagicMock,
    mock_mq_client_cls: MagicMock,
    mock_zmq_context_cls: MagicMock,
    mock_llm_args: MagicMock,
) -> None:
    """Test that __del__ and close triggers shutdown for both scheduler and worker.

    Args:
        mock_send_request: Mocked _send_request function.
        mock_mq_client_cls: Mocked MessageQueueClient class.
        mock_zmq_context_cls: Mocked zmq.Context class.
        mock_llm_args: Mocked llm_args fixture.
    """
    mock_future = MagicMock()
    mock_future.result.return_value = 256
    mock_send_request.return_value = mock_future

    mock_mq_client_1 = MagicMock()
    mock_mq_client_2 = MagicMock()
    mock_mq_client_cls.side_effect = [mock_mq_client_1, mock_mq_client_2]

    mock_zmq_ctx_1 = MagicMock()
    mock_zmq_ctx_2 = MagicMock()
    mock_zmq_context_cls.side_effect = [mock_zmq_ctx_1, mock_zmq_ctx_2]

    scheduler = LMCacheMPKvConnectorScheduler(mock_llm_args)
    worker = LMCacheMPKvConnectorWorker(mock_llm_args)
    worker._registered = True

    # Call __del__
    scheduler.__del__()
    worker.__del__()

    mock_mq_client_1.close.assert_called_once()
    mock_mq_client_2.close.assert_called_once()

    mock_zmq_ctx_1.destroy.assert_called_once_with(linger=0)
    mock_zmq_ctx_2.destroy.assert_called_once_with(linger=0)

    # Call close (idempotent, won't fail even if mq_client is cleared)
    scheduler.close()
    worker.close()
