# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import os
import sys


class DummyKvCacheConnectorWorker:
    def __init__(self, llm_args, *args, **kwargs):
        self._llm_args = llm_args


class DummyKvCacheConnectorScheduler:
    def __init__(self, llm_args, *args, **kwargs):
        self._llm_args = llm_args


# Mock tensorrt_llm before importing the adapter
mock_tensorrt_llm = MagicMock()
sys.modules["tensorrt_llm"] = mock_tensorrt_llm
sys.modules["tensorrt_llm._torch"] = mock_tensorrt_llm
sys.modules["tensorrt_llm._torch.pyexecutor"] = mock_tensorrt_llm
sys.modules["tensorrt_llm._torch.pyexecutor.connectors"] = mock_tensorrt_llm

mock_kv_cache_connector = MagicMock()
mock_kv_cache_connector.KvCacheConnectorWorker = DummyKvCacheConnectorWorker
mock_kv_cache_connector.KvCacheConnectorScheduler = DummyKvCacheConnectorScheduler
sys.modules["tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"] = (
    mock_kv_cache_connector
)
sys.modules["tensorrt_llm.bindings"] = mock_tensorrt_llm
sys.modules["tensorrt_llm.bindings.internal"] = mock_tensorrt_llm
sys.modules["tensorrt_llm.bindings.internal.batch_manager"] = mock_tensorrt_llm
sys.modules["tensorrt_llm.llmapi"] = mock_tensorrt_llm
sys.modules["tensorrt_llm.llmapi.llm_args"] = mock_tensorrt_llm

# Third Party
import pytest  # noqa: E402

# First Party
from lmcache.integration.tensorrt_llm import (  # noqa: E402
    tensorrt_mp_adapter as adapter_mod,
)
from lmcache.integration.tensorrt_llm.tensorrt_mp_adapter import (  # noqa: E402
    LMCacheMPKvConnectorWorker,
)


def _make_trtllm_worker() -> LMCacheMPKvConnectorWorker:
    mock_tensorrt_llm.mpi_rank.return_value = 0
    llm_args = MagicMock()
    llm_args.tensor_parallel_size = 1
    llm_args.pipeline_parallel_size = 1
    llm_args.model = "test-model"
    llm_args.kv_connector_config = None
    llm_args.kv_cache_config.tokens_per_block = 16
    return LMCacheMPKvConnectorWorker(llm_args)


@pytest.fixture
def mock_trtllm_adapter(monkeypatch):
    # Stub the MQ boundary and chunk size
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "_send_request", lambda *a, **kw: MagicMock())

    # Mock zmq
    monkeypatch.setattr(adapter_mod, "zmq", MagicMock())


def test_trtllm_instance_id_is_uuid_derived_63_bit_int(
    mock_trtllm_adapter,
) -> None:
    """instance_id is a 63-bit int, not the PID, and unique per worker."""
    worker = _make_trtllm_worker()

    assert isinstance(worker._instance_id, int)
    assert not isinstance(worker._instance_id, bool)
    assert 0 <= worker._instance_id < (1 << 63)
    assert worker._instance_id != os.getpid()

    worker2 = _make_trtllm_worker()
    assert worker._instance_id != worker2._instance_id


def test_trtllm_instance_id_logged_at_info_on_construction(
    mock_trtllm_adapter, monkeypatch
) -> None:
    """The constructor logs instance_id at INFO for correlating server-side
    reap warnings. The module logger does not propagate (``propagate=False``),
    so the test spies on it directly instead of using ``caplog``."""
    messages: list[str] = []

    def spy_info(msg: object, *args: object, **kwargs: object) -> None:
        messages.append(str(msg) % args if args else str(msg))

    monkeypatch.setattr(adapter_mod.logger, "info", spy_info)

    worker = _make_trtllm_worker()

    assert any(str(worker._instance_id) in msg for msg in messages)
