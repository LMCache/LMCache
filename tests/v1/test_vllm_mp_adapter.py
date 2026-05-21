# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for ``LMCacheMPWorkerAdapter.register_kv_caches``.

Behavioural coverage of the heartbeat-driven recovery path
(``HeartbeatThread.register_recover_callback`` →
worker re-registration) lives in the buildkite end-to-end test
``.buildkite/k3_tests/multiprocess/scripts/run-restart-recovery.sh``.
That path requires driving the periodic-thread tick loop, which is
deliberately not reachable through any public interface.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.protocol import RequestType


@pytest.fixture
def fake_adapter(monkeypatch):
    """Build an adapter through its real ``__init__`` with the network
    boundary stubbed out. Returns ``(adapter, send_mock, future)`` where
    ``send_mock`` is the patched ``send_lmcache_request`` and ``future``
    is its return value (a ``MagicMock`` whose ``result()`` defaults to
    succeed; tests can attach ``side_effect`` to simulate failures).
    """
    # Stub the MQ boundary so __init__'s chunk-size query and any later
    # send_lmcache_request call don't touch a real socket.
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)

    future = MagicMock(name="future")
    future.result.return_value = None
    send_mock = MagicMock(name="send_lmcache_request", return_value=future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    # KV-cache wrapping pulls in CUDA IPC; bypass for unit tests.
    monkeypatch.setattr(adapter_mod, "wrap_kv_caches", lambda kv: list(kv.values()))
    # ``vllm_layout_hints`` returns a ``LayoutHints`` (TypedDict / dict at
    # runtime); the production path performs item assignment on it
    # (``layout_hints["inference_engine_logical_block_size"] = ...``), so
    # the stub must also be a real dict — a string would raise
    # ``TypeError: 'str' object does not support item assignment``.
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
    )

    parallel_strategy = ParallelStrategy(
        use_mla=False,
        kv_world_size=1,
        kv_worker_id=0,
        actual_world_size=1,
        actual_worker_id=0,
        tp_size=1,
        pp_size=1,
    )
    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=parallel_strategy,
        mq_timeout=5.0,
    )
    # __init__ issues exactly one MQ call (the chunk-size query). Reset
    # so individual tests start with a clean call count.
    send_mock.reset_mock()
    return adapter, send_mock, future


def test_register_kv_caches_updates_kv_caches_and_submits(fake_adapter):
    """Public register_kv_caches stores the dict and submits one request."""
    adapter, send_mock, _ = fake_adapter
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    new_caches = {"layer.0": fake_tensor, "layer.1": fake_tensor}

    adapter.register_kv_caches(new_caches)

    assert adapter.kv_caches is new_caches
    assert send_mock.call_count == 1
    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE


def test_register_kv_caches_raises_connection_error_on_timeout(fake_adapter):
    """Public register_kv_caches surfaces ConnectionError on MQ timeout."""
    adapter, _send_mock, future = fake_adapter
    future.result.side_effect = TimeoutError("server down")

    with pytest.raises(ConnectionError, match="did not respond"):
        fake_tensor = MagicMock()
        fake_tensor.device.type = "cuda"
        adapter.register_kv_caches({"layer.0": fake_tensor})


def test_register_kv_caches_cpu_submits_non_gpu_context_registration(
    fake_adapter, monkeypatch
):
    """CPU KV cache registration routes to REGISTER_KV_CACHE_NON_GPU_CONTEXT."""
    adapter, send_mock, _ = fake_adapter
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
        raising=False,
    )
    cpu_kv = {"layer.0": torch.randn(2, 8, 4, 2, 8)}

    adapter.register_kv_caches(cpu_kv)

    assert adapter.kv_caches is cpu_kv
    assert send_mock.call_count == 1
    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE_NON_GPU_CONTEXT
    assert len(args[2]) == 1


def test_submit_store_request_tracks_returned_future(fake_adapter, monkeypatch):
    """submit_store_request stores the returned future in store_futures."""
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    transfer_ctx = MagicMock()
    fake_future = MagicMock()
    transfer_ctx.submit_store.return_value = fake_future
    adapter.transfer_ctx = transfer_ctx
    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[0], start=0, end=4)

    adapter.submit_store_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_store.called
    assert transfer_ctx.submit_store.call_args.kwargs == {}
    assert adapter.store_futures["req-1"] is fake_future


def test_submit_retrieve_request_tracks_returned_future(fake_adapter, monkeypatch):
    """submit_retrieve_request stores returned future and block IDs."""
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    transfer_ctx = MagicMock()
    fake_future = MagicMock()
    transfer_ctx.submit_retrieve.return_value = fake_future
    adapter.transfer_ctx = transfer_ctx
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[0],
        start=0,
        end=4,
        skip_first_n_tokens=1,
    )

    adapter.submit_retrieve_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_retrieve.called
    assert transfer_ctx.submit_retrieve.call_args.kwargs == {"skip_first_n_tokens": 1}
    assert adapter.retrieve_futures["req-1"] == (fake_future, [0])


def test_cacheblend_register_kv_caches_uses_cb_protocol(fake_adapter):
    """CacheBlend mode registers the CB GPU cache, not the normal MP cache."""
    adapter, send_mock, _future = fake_adapter
    adapter.enable_cacheblend = True

    adapter.register_kv_caches({"layer.0": object()})

    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.CB_REGISTER_KV_CACHE
    assert len(args[2]) == 4


def test_cacheblend_store_slices_tokens_for_cb_protocol(fake_adapter):
    """CB store keys contain only the stored chunk while offset points at vLLM KV."""
    adapter, send_mock, future = fake_adapter
    adapter.enable_cacheblend = True
    adapter._heartbeat = MagicMock(name="heartbeat")
    future.to_cuda_future.return_value = future
    event = MagicMock(name="event")
    event.ipc_handle.return_value = b"event-handle"
    op = LoadStoreOp(
        token_ids=list(range(64)),
        block_ids=[10, 11],
        start=16,
        end=48,
    )

    adapter.submit_store_request("req-1", op, event)

    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.CB_STORE_PRE_COMPUTED
    key, offset, instance_id, event_handle = args[2]
    assert tuple(key.token_ids) == tuple(range(16, 48))
    assert key.start == 0
    assert key.end == 32
    assert offset == 16
    assert instance_id == adapter.instance_id
    assert event_handle == b"event-handle"


def test_cacheblend_store_reports_telemetry_when_store_future_finishes(
    fake_adapter,
    monkeypatch,
):
    """CB precomputed stores must unblock strict disagg handoff directly.

    CacheBlend prefill submits CB_STORE_PRE_COMPUTED futures; strict proxy mode
    cannot wait for a later scheduler get_finished() call before decoder handoff.
    """
    adapter, _send_mock, future = fake_adapter
    adapter.enable_cacheblend = True
    adapter._heartbeat = MagicMock(name="heartbeat")
    adapter.request_telemetry = MagicMock(name="request_telemetry")
    future.to_cuda_future.return_value = future
    future.result.return_value = True
    event = MagicMock(name="event")
    event.ipc_handle.return_value = b"event-handle"

    class InlineThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(adapter_mod.threading, "Thread", InlineThread)

    adapter.submit_store_request(
        "chatcmpl-cb-prefill",
        LoadStoreOp(
            token_ids=list(range(64)),
            block_ids=[10, 11],
            start=0,
            end=64,
        ),
        event,
    )

    adapter.request_telemetry.on_request_store_finished.assert_called_once_with(
        request_ids_set={"chatcmpl-cb-prefill"},
        model_name="test-model",
        world_size=1,
        kv_rank=0,
    )


def test_get_finished_emits_request_telemetry_when_store_future_finishes(
    fake_adapter,
):
    """Telemetry must unblock disagg decode as soon as LMCache store is done.

    vLLM's finished_req_ids reconciliation is needed for scheduler block
    lifetime, but the disagg proxy waits for KV-store readiness before decoder
    handoff. If telemetry waits for the engine-finished side too, strict mode can
    deadlock after a successful prefill store.
    """
    adapter, _send_mock, future = fake_adapter
    future.query.return_value = True
    future.result.return_value = True
    adapter.store_futures["req-1"] = future
    adapter.request_telemetry = MagicMock(name="request_telemetry")

    adapter.get_finished(set())

    adapter.request_telemetry.on_request_store_finished.assert_called_once_with(
        request_ids_set={"req-1"},
        model_name="test-model",
        world_size=1,
        kv_rank=0,
    )
