# SPDX-License-Identifier: Apache-2.0
"""CPU-only caller tests for the TensorRT-LLM multiprocess adapter."""

# Standard
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock
import importlib
import sys

# Third Party
import pytest


def _make_worker(
    trt_mp_module: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, MagicMock]:
    """Construct a worker with a stub for the public request-client contract."""
    module = trt_mp_module
    monkeypatch.setenv("LMCACHE_MQ_TIMEOUT", "5")
    monkeypatch.setattr(
        module.zmq, "Context", MagicMock(return_value=MagicMock(name="zmq_context"))
    )
    req_client = MagicMock(spec=module.RequestClient)
    req_client.get_chunk_size.return_value.result.return_value = 256
    monkeypatch.setattr(
        module.RequestClientFactory,
        "create",
        MagicMock(return_value=req_client),
    )
    llm_args = SimpleNamespace(
        kv_cache_config=SimpleNamespace(tokens_per_block=32),
        kv_connector_config=SimpleNamespace(server_url=None),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        model="test-model",
    )
    worker = module.LMCacheMPKvConnectorWorker(llm_args)
    return worker, req_client


def _successful_transfer_future(name: str) -> MagicMock:
    """Build a completed device-transfer future for the worker test."""
    device_future = MagicMock(name=f"{name}_device_future")
    device_future.result.return_value = True
    transfer_future = MagicMock(name=f"{name}_future")
    transfer_future.to_device_future.return_value = device_future
    return transfer_future


@pytest.fixture
def trt_mp_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
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
    trt_mp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A False device transfer result cannot be treated as a loaded KV hit."""
    module = trt_mp_module
    worker, req_client = _make_worker(module, monkeypatch)
    worker.bind_connector_meta(
        module.LMCacheMPConnectorMetadata(
            loads={
                7: SimpleNamespace(tokens=list(range(256)), block_ids=list(range(8)))
            }
        )
    )

    event = MagicMock(name="event")
    event.ipc_handle.return_value = b"producer-event"
    monkeypatch.setattr(module, "check_interprocess_event_support", MagicMock())
    monkeypatch.setattr(module.torch_dev, "Event", MagicMock(return_value=event))

    device_future = MagicMock(name="device_future")
    device_future.result.return_value = False
    raw_future = MagicMock(name="raw_future")
    raw_future.to_device_future.return_value = device_future
    req_client.retrieve.return_value = raw_future

    with pytest.raises(RuntimeError, match="refusing to use unloaded KV blocks"):
        worker.start_load_kv(MagicMock(name="stream"))

    raw_future.to_device_future.assert_called_once_with()
    device_future.result.assert_called_once_with(timeout=5.0)


@pytest.mark.parametrize(
    ("token_count", "block_count", "expected_block_count"),
    [
        pytest.param(2176, 68, 64, id="partial-final-chunk"),
        pytest.param(2049, 65, 64, id="partial-final-block"),
        pytest.param(2048, 64, 64, id="chunk-aligned"),
        pytest.param(128, 4, 0, id="short-request"),
        pytest.param(0, 0, 0, id="empty-request"),
        pytest.param(2176, 0, 0, id="no-allocated-blocks"),
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
    worker, req_client = _make_worker(trt_mp_module, monkeypatch)
    tokens = list(range(token_count))
    # Physical pages need not be contiguous or ordered by their IDs.
    original_block_ids = [3 * i + 1 for i in reversed(range(block_count))]
    block_ids = list(original_block_ids)
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
    req_client.retrieve.return_value = _successful_transfer_future("retrieve")
    req_client.store.return_value = _successful_transfer_future("store")

    worker.start_load_kv(MagicMock(name="load_stream"))
    worker.wait_for_save(MagicMock(name="store_stream"))

    assert block_ids == original_block_ids
    assert tokens == list(range(token_count))
    if expected_block_count == 0:
        req_client.retrieve.assert_not_called()
        req_client.store.assert_not_called()
        return

    req_client.retrieve.assert_called_once()
    req_client.store.assert_called_once()
    retrieve_payload = req_client.retrieve.call_args.args
    store_payload = req_client.store.call_args.args
    expected_end = (token_count // 256) * 256
    assert retrieve_payload[0].start == 0
    assert retrieve_payload[0].end == expected_end
    assert retrieve_payload[2] == [original_block_ids[:expected_block_count]]
    assert store_payload[0].start == 0
    assert store_payload[0].end == expected_end
    assert store_payload[2] == [original_block_ids[:expected_block_count]]
