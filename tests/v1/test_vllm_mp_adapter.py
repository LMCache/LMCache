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
import gc
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.group_view import LMCacheGroupView
from lmcache.v1.multiprocess.protocol import RequestType


class FakeMQClient:
    """MessageQueueClient double that records close calls."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.closed = False

    def close(self) -> None:
        """Record that the client was closed."""
        self.closed = True


def _parallel_strategy(
    *,
    kv_worker_id: int = 0,
    actual_worker_id: int = 0,
) -> ParallelStrategy:
    world_size = max(kv_worker_id, actual_worker_id) + 1
    return ParallelStrategy(
        use_mla=False,
        kv_world_size=world_size,
        kv_worker_id=kv_worker_id,
        actual_world_size=world_size,
        actual_worker_id=actual_worker_id,
        tp_size=1,
        pp_size=1,
    )


class FakeCudaEvent:
    def ipc_handle(self) -> bytes:
        return b"fake-ipc-handle"


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

    class FakeHeartbeatThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.recover_callback: object | None = None

        def register_recover_callback(self, callback: object) -> None:
            self.recover_callback = callback

        def start(self) -> None:
            return None

    monkeypatch.setattr(adapter_mod, "HeartbeatThread", FakeHeartbeatThread)

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


def test_scheduler_adapter_does_not_autostart(monkeypatch) -> None:
    """Scheduler adapter keeps connect-only behavior for MP server startup."""
    maybe_start = MagicMock(name="maybe_start_mp_server_from_url")
    wait_for_server = MagicMock(name="wait_for_mp_server_from_url")

    monkeypatch.setattr(adapter_mod, "maybe_start_mp_server_from_url", maybe_start)
    monkeypatch.setattr(adapter_mod, "wait_for_mp_server_from_url", wait_for_server)
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", FakeMQClient)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)

    LMCacheMPSchedulerAdapter(
        server_url="tcp://localhost:5555",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_parallel_strategy(),
        extra_config={"lmcache.mp.autostart": True},
    )

    maybe_start.assert_not_called()
    wait_for_server.assert_not_called()


def test_worker_zero_autostarts_before_mq_client(monkeypatch) -> None:
    """Actual worker 0 ensures the MP server is ready before MQ connects."""
    events: list[str] = []
    launcher = MagicMock(name="launcher")

    def fake_maybe_start(**kwargs):
        events.append("maybe_start")
        assert kwargs["server_url"] == "tcp://localhost:5555"
        return launcher

    class RecordingMQClient(FakeMQClient):
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("mq_client")
            super().__init__(*_args, **_kwargs)

    monkeypatch.setattr(adapter_mod, "maybe_start_mp_server_from_url", fake_maybe_start)
    wait_for_server = MagicMock(name="wait_for_mp_server_from_url")
    monkeypatch.setattr(adapter_mod, "wait_for_mp_server_from_url", wait_for_server)
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", RecordingMQClient)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)

    LMCacheMPWorkerAdapter(
        server_url="tcp://localhost:5555",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_parallel_strategy(actual_worker_id=0),
        extra_config={"lmcache.mp.autostart": True},
    )

    assert events == ["maybe_start", "mq_client"]
    wait_for_server.assert_not_called()
    launcher.shutdown.assert_not_called()


def test_nonzero_worker_waits_before_mq_client(monkeypatch) -> None:
    """Nonzero actual workers wait even when ``kv_worker_id`` is zero."""
    events: list[str] = []

    def fake_wait_for_server(**kwargs):
        events.append("wait_for_server")
        assert kwargs["server_url"] == "tcp://localhost:5555"

    class RecordingMQClient(FakeMQClient):
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("mq_client")
            super().__init__(*_args, **_kwargs)

    maybe_start = MagicMock(name="maybe_start_mp_server_from_url")
    monkeypatch.setattr(adapter_mod, "maybe_start_mp_server_from_url", maybe_start)
    monkeypatch.setattr(
        adapter_mod, "wait_for_mp_server_from_url", fake_wait_for_server
    )
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", RecordingMQClient)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)

    LMCacheMPWorkerAdapter(
        server_url="tcp://localhost:5555",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_parallel_strategy(kv_worker_id=0, actual_worker_id=1),
        extra_config={"lmcache.mp.autostart": True},
    )

    assert events == ["wait_for_server", "mq_client"]
    maybe_start.assert_not_called()


def test_worker_adapter_does_not_shutdown_autostarted_server_on_init_failure(
    monkeypatch,
) -> None:
    """A shared auto-started server is not stopped by worker init failure."""
    launcher = MagicMock(name="launcher")
    fake_client = FakeMQClient()

    monkeypatch.setattr(
        adapter_mod,
        "maybe_start_mp_server_from_url",
        MagicMock(return_value=launcher),
    )
    monkeypatch.setattr(
        adapter_mod,
        "wait_for_mp_server_from_url",
        MagicMock(name="wait_for_mp_server_from_url"),
    )
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(
        adapter_mod,
        "get_lmcache_chunk_size",
        MagicMock(side_effect=TimeoutError),
    )

    with pytest.raises(ConnectionError, match="Cannot reach the LMCache MP server"):
        LMCacheMPWorkerAdapter(
            server_url="tcp://localhost:5555",
            context=MagicMock(name="zmq_context"),
            model_name="test-model",
            vllm_block_size=16,
            parallel_strategy=_parallel_strategy(),
            extra_config={"lmcache.mp.autostart": True},
        )

    assert fake_client.closed
    launcher.shutdown.assert_not_called()


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
    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[[0]], start=0, end=4)

    adapter.submit_store_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_store.called
    assert transfer_ctx.submit_store.call_args.kwargs == {}
    assert transfer_ctx.submit_store.call_args.args[4] == [[0]]
    assert adapter.store_futures["req-1"] is fake_future


def test_submit_store_request_expands_block_ids_to_views(fake_adapter, monkeypatch):
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    adapter.group_views = [
        LMCacheGroupView(0, (0, 2)),
        LMCacheGroupView(0, (4,)),
        LMCacheGroupView(1, (1, 3)),
    ]
    transfer_ctx = MagicMock()
    fake_future = MagicMock()
    transfer_ctx.submit_store.return_value = fake_future
    adapter.transfer_ctx = transfer_ctx
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[[0, 1], [10, 11]],
        start=0,
        end=4,
    )

    adapter.submit_store_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_store.call_args.args[4] == [
        [0, 1],
        [0, 1],
        [10, 11],
    ]


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
        block_ids=[[0]],
        start=0,
        end=4,
        skip_first_n_tokens=1,
    )

    adapter.submit_retrieve_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_retrieve.called
    assert transfer_ctx.submit_retrieve.call_args.kwargs == {"skip_first_n_tokens": 1}
    assert transfer_ctx.submit_retrieve.call_args.args[4] == [[0]]
    assert adapter.retrieve_futures["req-1"] == (fake_future, [0])


def test_load_store_op_accepts_per_group_block_ids():
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[[0, 1], [10, 11]],
        start=0,
        end=4,
    )

    assert op.block_ids == [[0, 1], [10, 11]]
    assert op.flat_block_ids == [0, 1, 10, 11]


def test_store_keeps_event_until_future_finishes(fake_adapter):
    """Store requests keep the exported CUDA event alive while pending."""
    adapter, _send_mock, _future = fake_adapter
    cuda_future = MagicMock(name="cuda_future")
    cuda_future.query.return_value = False
    transfer_ctx = MagicMock()
    transfer_ctx.submit_store.return_value = cuda_future
    adapter.transfer_ctx = transfer_ctx

    event = FakeCudaEvent()
    event_ref = weakref.ref(event)
    op = LoadStoreOp(token_ids=[1, 2], block_ids=[[7]], start=0, end=2)

    adapter.submit_store_request("req-1", op, event)
    del event
    gc.collect()
    assert event_ref() is not None

    cuda_future.query.return_value = True
    cuda_future.result.return_value = True
    finished_stores, finished_retrieves = adapter.get_finished({"req-1"})

    assert finished_stores == {"req-1"}
    assert finished_retrieves == set()
    assert "req-1" not in adapter.store_events
    transfer_ctx.reset_mock()
    gc.collect()
    assert event_ref() is None


def test_retrieve_keeps_event_until_future_finishes(fake_adapter):
    """Retrieve requests keep the exported CUDA event alive while pending."""
    adapter, _send_mock, _future = fake_adapter
    cuda_future = MagicMock(name="cuda_future")
    cuda_future.query.return_value = False
    transfer_ctx = MagicMock()
    transfer_ctx.submit_retrieve.return_value = cuda_future
    adapter.transfer_ctx = transfer_ctx

    event = FakeCudaEvent()
    event_ref = weakref.ref(event)
    op = LoadStoreOp(token_ids=[1, 2], block_ids=[[7]], start=0, end=2)

    adapter.submit_retrieve_request("req-1", op, event)
    del event
    gc.collect()
    assert event_ref() is not None

    cuda_future.query.return_value = True
    cuda_future.result.return_value = True
    finished_stores, finished_retrieves = adapter.get_finished(set())

    assert finished_stores == set()
    assert finished_retrieves == {"req-1"}
    assert "req-1" not in adapter.retrieve_events
    transfer_ctx.reset_mock()
    gc.collect()
    assert event_ref() is None
