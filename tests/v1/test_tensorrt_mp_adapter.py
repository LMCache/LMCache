# SPDX-License-Identifier: Apache-2.0
"""CPU-only caller tests for the TensorRT-LLM multiprocess adapter."""

# Standard
from collections import Counter
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


CHUNK_SIZE = 16
BLOCK_SIZE = 4
PROMPT_TOKENS = 8 * CHUNK_SIZE


@pytest.mark.parametrize(
    ("num_computed", "hit_chunks", "retrieved"),
    [
        pytest.param(0, 8, (0, 128), id="cold-exact-repeat"),
        pytest.param(0, 5, (0, 80), id="partial-hit"),
        pytest.param(32, 8, (32, 128), id="held-aligned-prefix"),
        pytest.param(40, 8, (32, 128), id="held-unaligned-prefix"),
        pytest.param(32, 5, (32, 80), id="held-prefix-partial-hit"),
        pytest.param(72, 5, (64, 80), id="held-prefix-in-last-hit-chunk"),
        pytest.param(96, 5, None, id="held-prefix-beyond-hit"),
        pytest.param(0, 0, None, id="miss"),
    ],
)
def test_load_retrieves_the_chunks_the_lookup_left_locked(
    trt_mp_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    num_computed: int,
    hit_chunks: int,
    retrieved: tuple[int, int] | None,
) -> None:
    """Free/retrieve ranges account for each hit chunk once.

    The retrieve payload covers the tokens offered to TRT-LLM.
    """
    module = trt_mp_module
    client = MagicMock(spec=module.RequestClient)
    client.get_chunk_size.return_value.result.return_value = CHUNK_SIZE
    client.query_prefetch_status.return_value.result.return_value = hit_chunks
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
        kv_cache_config=SimpleNamespace(tokens_per_block=BLOCK_SIZE),
        kv_connector_config=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        model="m",
    )
    scheduler = module.LMCacheMPKvConnectorScheduler(llm_args)
    worker = module.LMCacheMPKvConnectorWorker(llm_args)
    worker.register_kv_caches(
        module.torch.empty(PROMPT_TOKENS // BLOCK_SIZE + 1, 1, 2, 128 * BLOCK_SIZE)
    )
    tokens = list(range(PROMPT_TOKENS))
    request = SimpleNamespace(request_id=7, get_tokens=lambda beam: tokens)
    new_request = SimpleNamespace(
        request_id=7,
        new_block_ids=list(range(1, PROMPT_TOKENS // BLOCK_SIZE + 1)),
        computed_position=num_computed,
        new_tokens=tokens,
    )

    new_matched, _ = scheduler.get_num_new_matched_tokens(request, num_computed)
    worker.bind_connector_meta(
        scheduler.build_connector_meta(SimpleNamespace(new_requests=[new_request]))
    )
    worker.start_load_kv(MagicMock(name="stream"))
    worker.wait_for_save(MagicMock(name="stream"))

    releases = Counter(
        chunk
        for call in client.free_lookup_locks.call_args_list
        + client.retrieve.call_args_list
        for chunk in range(
            call.args[0].start // CHUNK_SIZE, call.args[0].end // CHUNK_SIZE
        )
    )
    assert releases == Counter(range(hit_chunks))
    for call in client.store.call_args_list:
        assert (call.args[0].start, call.args[0].end) == (0, PROMPT_TOKENS)
    if retrieved is None:
        assert new_matched == 0
        client.retrieve.assert_not_called()
        return
    key, _, block_ids, _, skip_first_n_tokens = client.retrieve.call_args.args
    assert (key.start, key.end) == retrieved
    assert key.num_tokens == PROMPT_TOKENS
    assert new_matched == key.end - num_computed
    assert block_ids == [
        list(range(key.start // BLOCK_SIZE + 1, key.end // BLOCK_SIZE + 1))
    ]
    assert skip_first_n_tokens == num_computed - key.start
