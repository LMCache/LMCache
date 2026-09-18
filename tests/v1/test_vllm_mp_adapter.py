# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for ``LMCacheMPWorkerAdapter``. The MQ boundary is
stubbed (see ``fake_adapter``); no GPU or live server needed. End-to-end
recovery: ``.buildkite/k3_tests/multiprocess/scripts/run-restart-recovery.sh``."""

# Standard
from typing import Callable, ClassVar
from unittest.mock import MagicMock
import collections
import gc
import os
import threading
import time
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.experimental.dispatcher import Dispatcher
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    HeartbeatThread,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.utils import CacheEvent, CacheRemoveEvent, CacheStoreEvent
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_CAPABILITY,
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    KV_EVENT_MEDIUM_STORAGE,
    KVEventPollResult,
    KVEventRecord,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform.ipc_policy import (
    is_isolated_ipc,
    is_use_vmm_api,
    set_ipc_policy,
    set_isolated_ipc,
    set_use_vmm_api,
)


class FakeCudaEvent:
    def ipc_handle(self) -> bytes:
        return b"fake-ipc-handle"


class FakeHeartbeatThread:
    """Test double mirroring ``HeartbeatThread``'s public surface.
    ``start()`` invokes class-level ``start_hook`` when set, otherwise
    simulates a successful first ping. Class state reset per test."""

    instances: ClassVar[list["FakeHeartbeatThread"]] = []
    start_hook: ClassVar[Callable[["FakeHeartbeatThread"], None] | None] = None

    def __init__(
        self,
        req_client: object = None,
        health_event: threading.Event | None = None,
        interval: float = 0.0,
        instance_id: int | None = None,
    ) -> None:
        self.req_client = req_client
        self.health_event = (
            health_event if health_event is not None else threading.Event()
        )
        self.interval = interval
        self.instance_id = instance_id
        # Snapshot of the health event at construction time: lets tests
        # assert the adapter starts the heartbeat healthy (event still set).
        self.health_event_set_at_init = self.health_event.is_set()
        self.recover_callback: Callable[[], bool] | None = None
        # Ordered record of public calls ("register_recover_callback",
        # "start", "stop") for call-order assertions.
        self.calls: list[str] = []
        self.stop_requested = False
        FakeHeartbeatThread.instances.append(self)

    def register_recover_callback(self, callback: Callable[[], bool]) -> None:
        self.calls.append("register_recover_callback")
        self.recover_callback = callback

    def start(self) -> None:
        self.calls.append("start")
        hook = FakeHeartbeatThread.start_hook
        if hook is not None:
            hook(self)
        else:
            self.simulate_successful_ping()

    def stop(self, timeout: float = 5.0) -> None:
        self.calls.append("stop")
        self.stop_requested = True

    def simulate_successful_ping(self) -> None:
        """Mimic one successful heartbeat cycle: on the unhealthy->healthy
        edge the recover callback runs first, and the event is set only
        when the callback returns ``True``."""
        was_healthy = self.health_event.is_set()
        ok = True
        if not was_healthy and self.recover_callback is not None:
            ok = self.recover_callback()
        if ok:
            self.health_event.set()


# Own completed-store events are the fallback path: they are published only
# while the server's cache-event log is not polled.
_OWN_STORE_EVENTS: dict[str, object] = {"lmcache.mp.kv_event_poll_interval": 0}


def _make_worker_adapter(
    extra_config: dict[str, object] | None = None,
    enable_kv_events: bool = False,
    parallel_strategy: ParallelStrategy | None = None,
) -> LMCacheMPWorkerAdapter:
    """Construct a worker adapter with the standard test arguments; the
    network boundary must already be patched (see ``fake_adapter``).
    ``extra_config`` forwards ``lmcache.mp.*`` overrides, and
    ``parallel_strategy`` overrides the default single-rank placement."""
    if parallel_strategy is None:
        parallel_strategy = ParallelStrategy(
            mla_only=False,
            vllm_world_size=1,
            vllm_worker_id=0,
            tp_size=1,
            pp_size=1,
            n_servers=1,
        )
    return LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=parallel_strategy,
        mq_timeout=5.0,
        extra_config=extra_config,
        enable_kv_events=enable_kv_events,
    )


def _op(block_ids: list[list[int]]) -> LoadStoreOp:
    """Build a minimal four-token ``LoadStoreOp`` over *block_ids*."""
    return LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=block_ids, start=0, end=4)


def _patch_transfer_context_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> list[MagicMock]:
    """Patch ``create_transfer_context`` to mint recorded MagicMocks,
    returning the list every created context is appended to."""
    contexts: list[MagicMock] = []

    def fake_create_transfer_context(
        kv_caches: dict[str, torch.Tensor],
        *,
        instance_id: int,
        req_client: RequestClient,
        mode: str,
    ) -> MagicMock:
        del kv_caches, instance_id, req_client, mode
        ctx = MagicMock(name=f"transfer_ctx_{len(contexts)}")
        contexts.append(ctx)
        return ctx

    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", fake_create_transfer_context
    )
    return contexts


def _patch_request_client_factory(
    monkeypatch: pytest.MonkeyPatch,
    client: MagicMock,
) -> None:
    """Make the transport-neutral factory return the supplied client."""
    factory = MagicMock(name="request_client_factory")
    factory.create.return_value = client
    monkeypatch.setattr(adapter_mod, "RequestClientFactory", factory)


def _return_future_from_request_methods(
    client: MagicMock,
    future: MagicMock,
) -> None:
    """Configure every request method on a client mock with one future."""
    for name, method in RequestClient.__dict__.items():
        if not name.startswith("_") and name != "close" and callable(method):
            getattr(client, name).return_value = future


@pytest.fixture
def fake_adapter(monkeypatch):
    """Build an adapter with the network boundary stubbed. Returns
    ``(adapter, req_client, future)``; ``future.result()`` defaults to succeed.
    ``HeartbeatThread`` is replaced by ``FakeHeartbeatThread``."""
    req_client = MagicMock(name="req_client", spec=RequestClient)
    _patch_request_client_factory(monkeypatch, req_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)
    monkeypatch.setattr(
        adapter_mod, "get_experimental", lambda *a, **kw: {KV_EVENT_CAPABILITY}
    )

    future = MagicMock(name="future")
    future.result.return_value = None
    _return_future_from_request_methods(req_client, future)

    FakeHeartbeatThread.instances.clear()
    FakeHeartbeatThread.start_hook = None
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", FakeHeartbeatThread)

    # KV-cache wrapping pulls in CUDA IPC; bypass for unit tests.
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    monkeypatch.setattr(
        worker_transfer,
        "wrap_kv_caches",
        lambda kv: list(kv.values()),
    )
    # ``vllm_layout_hints`` returns a ``LayoutHints`` (TypedDict / dict at
    # runtime); stub it with an empty dict.
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
    )

    adapter = _make_worker_adapter()
    req_client.reset_mock()
    return adapter, req_client, future


def test_register_kv_caches_updates_kv_caches_and_submits(fake_adapter):
    """Public register_kv_caches stores the dict and submits one request."""
    adapter, req_client, _ = fake_adapter
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    new_caches = {"layer.0": fake_tensor, "layer.1": fake_tensor}

    adapter.register_kv_caches(new_caches)

    assert adapter.kv_caches is new_caches
    req_client.register_kv_cache.assert_called_once()


def test_register_kv_caches_raises_connection_error_on_timeout(fake_adapter):
    """Public register_kv_caches surfaces ConnectionError on MQ timeout."""
    adapter, _send_mock, future = fake_adapter
    future.result.side_effect = TimeoutError("server down")

    with pytest.raises(ConnectionError, match="did not respond"):
        fake_tensor = MagicMock()
        fake_tensor.device.type = "cuda"
        adapter.register_kv_caches({"layer.0": fake_tensor})


def test_register_kv_caches_cpu_submits_engine_driven_context_registration(
    fake_adapter, monkeypatch
):
    """CPU-only capability fallback registers without requiring an event."""
    adapter, req_client, _ = fake_adapter
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.transfer_context.worker_transfer._supports_async_primitives",
        lambda: False,
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
        raising=False,
    )
    cpu_kv = {"layer.0": torch.randn(2, 8, 4, 2, 8)}

    adapter.register_kv_caches(cpu_kv)

    assert adapter.kv_caches is cpu_kv
    req_client.register_kv_cache_engine_driven_context.assert_called_once()
    assert len(req_client.register_kv_cache_engine_driven_context.call_args.args) == 1
    assert adapter.create_recorded_event() is None


def test_register_kv_caches_tuple_caches_use_engine_driven_context(
    fake_adapter, monkeypatch
):
    """CPU-only tuple caches register without requiring an IPC event."""
    adapter, req_client, _ = fake_adapter
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.transfer_context.worker_transfer._supports_async_primitives",
        lambda: False,
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
        raising=False,
    )
    # NL_X_TWO_X_NB_BS_NH_HS layout: [NB, BS, NH, HS] per plane.
    k = torch.randn(2, 8, 4, 8)
    v = torch.randn(2, 8, 4, 8)
    tuple_kv = {"layer.0": (k, v), "layer.1": (k, v)}

    adapter.register_kv_caches(tuple_kv)

    assert adapter.kv_caches is tuple_kv
    req_client.register_kv_cache_engine_driven_context.assert_called_once()
    assert adapter.create_recorded_event() is None


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

    adapter.submit_store_request(
        "req-1",
        op,
        event=MagicMock(),
        request_configs={"lmcache.skip_save": True},
    )

    assert transfer_ctx.submit_store.called
    assert transfer_ctx.submit_store.call_args.kwargs == {}
    assert transfer_ctx.submit_store.call_args.args[1].request_configs == {
        "lmcache.skip_save": True
    }
    assert transfer_ctx.submit_store.call_args.args[3] == [[0]]
    assert adapter.store_futures["req-1"] is fake_future


def test_submit_store_request_expands_block_ids_to_views(fake_adapter, monkeypatch):
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    adapter.engine_group_infos = [
        EngineGroupInfo(0, (0, 2)),
        EngineGroupInfo(0, (4,)),
        EngineGroupInfo(1, (1, 3)),
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

    assert transfer_ctx.submit_store.call_args.args[3] == [
        [0, 1],
        [0, 1],
        [10, 11],
    ]


def test_store_kv_events_are_reported_after_successful_store(
    fake_adapter,
    monkeypatch,
):
    """Completed MP stores are exposed once as LMCache cache-store events."""
    # First Party
    from lmcache.v1.multiprocess.token_hasher import TokenHasher

    adapter = _make_worker_adapter(
        extra_config=_OWN_STORE_EVENTS, enable_kv_events=True
    )
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    transfer_ctx = MagicMock()
    store_future = MagicMock()
    store_future.query.return_value = True
    store_future.result.return_value = True
    transfer_ctx.submit_store.return_value = store_future
    adapter.transfer_ctx = transfer_ctx

    chunk_size = adapter.lmcache_tokens_per_chunk
    token_ids = list(range(chunk_size * 2))
    op = LoadStoreOp(
        token_ids=token_ids,
        block_ids=[[1]],
        start=chunk_size,
        end=chunk_size * 2,
    )
    adapter.submit_store_request("req-1", op, event=None)

    assert adapter.get_kv_events() == []

    adapter.get_finished({"req-1"})
    events = adapter.get_kv_events()
    expected_hashes = TokenHasher(chunk_size=chunk_size).compute_chunk_hashes(
        token_ids,
        end=chunk_size * 2,
    )

    assert len(events) == 1
    assert events[0].block_hashes == [expected_hashes[1]]
    assert events[0].parent_block_hash == expected_hashes[0]
    assert events[0].token_ids == token_ids[chunk_size : chunk_size * 2]
    assert events[0].block_size == chunk_size
    assert events[0].medium == "CPU"
    assert adapter.get_kv_events() == []


def test_store_kv_events_are_discarded_after_failed_store(
    fake_adapter,
    monkeypatch,
):
    """Failed MP stores must not emit cache-store events."""
    adapter = _make_worker_adapter(
        extra_config=_OWN_STORE_EVENTS, enable_kv_events=True
    )
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    transfer_ctx = MagicMock()
    store_future = MagicMock()
    store_future.query.return_value = True
    store_future.result.return_value = False
    transfer_ctx.submit_store.return_value = store_future
    adapter.transfer_ctx = transfer_ctx

    chunk_size = adapter.lmcache_tokens_per_chunk
    op = LoadStoreOp(
        token_ids=list(range(chunk_size)),
        block_ids=[[0]],
        start=0,
        end=chunk_size,
    )
    adapter.submit_store_request("req-1", op, event=None)

    adapter.get_finished({"req-1"})

    assert adapter.get_kv_events() == []


@pytest.mark.parametrize("store_result", [True, False, None])
def test_lazy_store_kv_events_preserve_completion_and_failure_reporting(
    fake_adapter,
    store_result: bool | None,
) -> None:
    """Only successful stores emit events; every lazy store reports completion."""
    adapter = _make_worker_adapter(
        extra_config={"lmcache.mp.lazy_offload": True, **_OWN_STORE_EVENTS},
        enable_kv_events=True,
    )
    adapter.transfer_ctx = MagicMock()
    future = adapter.transfer_ctx.submit_store.return_value
    future.query.return_value = True
    future.result.return_value = store_result
    chunk_size = adapter.lmcache_tokens_per_chunk
    op = LoadStoreOp(
        token_ids=list(range(chunk_size)),
        block_ids=[[0]],
        start=0,
        end=chunk_size,
    )
    adapter.submit_store_request("req-1", op, event=None)
    assert adapter.get_kv_events() == []
    if store_result is None:
        # Lose server health while the store is pending.
        FakeHeartbeatThread.instances[-1].health_event.clear()

    adapter.get_finished_with_lazy_offload()

    assert len(adapter.get_kv_events()) == (1 if store_result else 0)
    assert adapter.get_kv_events() == []
    assert adapter.get_completed_store_requests() == {"req-1": 1}
    assert adapter.get_completed_store_requests() is None
    assert adapter.get_failed_store_requests() == (None if store_result else {"req-1"})
    assert adapter.get_failed_store_requests() is None


@pytest.mark.parametrize("lazy_offload", [False, True])
@pytest.mark.parametrize("enable_kv_events", [False, True])
def test_kv_event_buffer_metrics(
    fake_adapter: object,
    lazy_offload: bool,
    enable_kv_events: bool,
) -> None:
    """A stalled drain is visible; failed/pending stores do not count as events."""
    adapter = _make_worker_adapter(
        extra_config={"lmcache.mp.lazy_offload": lazy_offload, **_OWN_STORE_EVENTS},
        enable_kv_events=enable_kv_events,
    )
    adapter.transfer_ctx = MagicMock()
    future = adapter.transfer_ctx.submit_store.return_value
    future.result.return_value = True
    chunk_size = adapter.lmcache_tokens_per_chunk
    op = LoadStoreOp(
        token_ids=list(range(chunk_size * 2)),
        block_ids=[[0, 1]],
        start=0,
        end=chunk_size * 2,
    )

    def metrics() -> tuple[float, ...]:
        if not enable_kv_events:
            return (0, 0, 0)
        labels = {"model_name": "test-model", "worker_id": "0"}
        values = []
        for metric, name in (
            (adapter_mod._KV_EVENTS_BUFFERED, "buffered"),
            (adapter_mod._KV_EVENTS_GENERATED, "generated_total"),
            (adapter_mod._KV_EVENTS_DRAINED, "drained_total"),
        ):
            sample_name = f"vllm:lmcache_mp_kv_events_{name}"
            values.append(
                next(
                    sample.value
                    for family in metric.collect()
                    for sample in family.samples
                    if sample.name == sample_name and sample.labels == labels
                )
            )
        return tuple(values)

    def finish() -> None:
        if lazy_offload:
            adapter.get_finished_with_lazy_offload()
        else:
            adapter.get_finished(set())

    buffered, generated, drained = metrics()
    count = 0
    for request_id in ("first", "second"):
        future.query.return_value = False
        adapter.submit_store_request(request_id, op, event=None)
        finish()
        assert metrics() == (buffered + count, generated + count, drained)
        future.query.return_value = True
        finish()
        count += 2 if enable_kv_events else 0
        assert metrics() == (buffered + count, generated + count, drained)

    future.result.return_value = False
    adapter.submit_store_request("failed", op, event=None)
    finish()
    assert metrics() == (buffered + count, generated + count, drained)

    future.query.return_value = False
    adapter.submit_store_request("interrupted", op, event=None)
    FakeHeartbeatThread.instances[-1].health_event.clear()
    finish()
    assert metrics() == (buffered + count, generated + count, drained)

    assert len(adapter.get_kv_events()) == count
    assert metrics() == (buffered, generated + count, drained + count)
    assert adapter.get_kv_events() == []
    assert metrics() == (buffered, generated + count, drained + count)


def test_store_kv_events_use_hash_algorithm_extra_config(
    fake_adapter,
    monkeypatch,
):
    """KV event hashes use the hash algorithm configured for the MP server."""
    captured: dict[str, object] = {}

    class FakeTokenHasher:
        def __init__(self, chunk_size: int, hash_algorithm: str) -> None:
            captured["chunk_size"] = chunk_size
            captured["hash_algorithm"] = hash_algorithm

        def compute_chunk_hashes(
            self,
            token_ids: list[int],
            end: int | None = None,
        ) -> list[bytes]:
            return [b"hash"]

    monkeypatch.setattr(adapter_mod, "TokenHasher", FakeTokenHasher)

    adapter = _make_worker_adapter(
        extra_config={"lmcache.mp.hash_algorithm": "builtin", **_OWN_STORE_EVENTS},
        enable_kv_events=True,
    )

    assert captured == {
        "chunk_size": adapter.lmcache_tokens_per_chunk,
        "hash_algorithm": "builtin",
    }


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

    adapter.submit_retrieve_request(
        "req-1",
        op,
        event=MagicMock(),
        request_configs={"lmcache.skip_save": True},
    )

    assert transfer_ctx.submit_retrieve.called
    assert transfer_ctx.submit_retrieve.call_args.kwargs == {"skip_first_n_tokens": 1}
    assert transfer_ctx.submit_retrieve.call_args.args[1].request_configs == {
        "lmcache.skip_save": True
    }
    assert transfer_ctx.submit_retrieve.call_args.args[3] == [[0]]
    assert adapter.retrieve_futures["req-1"] == (fake_future, [0])


@pytest.mark.parametrize(
    "method_name",
    ["batched_submit_store_requests", "batched_submit_retrieve_requests"],
)
def test_batched_submit_rejects_mismatched_parallel_lists(
    fake_adapter, method_name: str
) -> None:
    adapter, _send_mock, _future = fake_adapter
    method = getattr(adapter, method_name)

    with pytest.raises(ValueError, match="must have the same length"):
        method(["req-1"], [], MagicMock())

    with pytest.raises(ValueError, match="must have the same length"):
        method(
            ["req-1"],
            [_op([[0]])],
            MagicMock(),
            request_configs_list=[],
        )


def test_load_store_op_accepts_per_group_block_ids():
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[[0, 1], [10, 11]],
        start=0,
        end=4,
    )

    assert op.block_ids == [[0, 1], [10, 11]]
    assert op.flat_block_ids == [0, 1, 10, 11]


@pytest.fixture
def restore_isolated_ipc():
    """Restore the process-global isolated-IPC switch after the test."""
    previous = is_isolated_ipc()
    yield
    set_isolated_ipc(previous)


@pytest.mark.parametrize(
    "raw,expected",
    [(False, False), (True, True), ("false", False), ("true", True)],
)
def test_isolated_ipc_extra_config_sets_process_switch(
    fake_adapter, restore_isolated_ipc, raw, expected
):
    """The lmcache.mp.isolated_ipc key drives the process-global switch,
    accepting both JSON booleans and their string spellings."""
    _make_worker_adapter(extra_config={"lmcache.mp.isolated_ipc": raw})
    assert is_isolated_ipc() is expected


def test_isolated_ipc_untouched_without_extra_config(
    fake_adapter, restore_isolated_ipc
):
    """Legacy callers passing no extra_config leave the process switch alone."""
    set_isolated_ipc(True)
    _make_worker_adapter(extra_config=None)
    assert is_isolated_ipc() is True


def test_isolated_ipc_is_set_before_transfer_context_creation(
    fake_adapter, restore_isolated_ipc, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend selection sees isolated IPC before transfer registration."""
    calls: list[tuple[str, bool, bool | str | None]] = []
    original_set_ipc_policy = adapter_mod.set_ipc_policy

    def record_ipc_policy(
        *,
        isolated_ipc: bool | None = None,
        use_vmm_api: bool | None = None,
    ) -> None:
        original_set_ipc_policy(
            isolated_ipc=isolated_ipc,
            use_vmm_api=use_vmm_api,
        )
        calls.append(("set_ipc_policy", is_isolated_ipc(), use_vmm_api))

    transfer_ctx = MagicMock(name="transfer_ctx")

    def create_context(
        _kv_caches: dict[str, torch.Tensor],
        *,
        instance_id: int,
        req_client: RequestClient,
        mode: str | None,
    ) -> MagicMock:
        del instance_id, req_client
        calls.append(("create_transfer_context", is_isolated_ipc(), mode))
        return transfer_ctx

    monkeypatch.setattr(adapter_mod, "set_ipc_policy", record_ipc_policy)
    monkeypatch.setattr(adapter_mod, "create_transfer_context", create_context)

    set_ipc_policy(isolated_ipc=False, use_vmm_api=False)
    adapter = _make_worker_adapter(extra_config={"lmcache.mp.isolated_ipc": True})
    adapter.register_kv_caches({"layer.0": torch.zeros(1)})

    assert calls == [
        ("set_ipc_policy", True, False),
        ("create_transfer_context", True, None),
    ]
    transfer_ctx.register.assert_called_once()


@pytest.fixture
def restore_use_vmm_api():
    """Restore the process-global VMM-API switch after the test."""
    previous = is_use_vmm_api()
    yield
    set_use_vmm_api(previous)


@pytest.mark.parametrize(
    "raw,expected",
    [(False, False), (True, True), ("false", False), ("true", True)],
)
def test_use_vmm_api_extra_config_sets_process_switch(
    fake_adapter, restore_use_vmm_api, raw, expected
):
    """The lmcache.mp.use_vmm_api key drives the process-global switch,
    accepting both JSON booleans and their string spellings."""
    _make_worker_adapter(extra_config={"lmcache.mp.use_vmm_api": raw})
    assert is_use_vmm_api() is expected


def test_create_recorded_event_delegates_to_transfer_context(fake_adapter, monkeypatch):
    """The active transfer context owns ordering-event creation."""
    adapter, _send_mock, _future = fake_adapter
    contexts = _patch_transfer_context_factory(monkeypatch)
    kv = torch.zeros(1)

    created_event = MagicMock(name="event")
    adapter.register_kv_caches({"layer.0": kv})
    context = contexts[0]
    context.create_recorded_event.return_value = created_event
    event = adapter.create_recorded_event()
    adapter.create_recorded_event()

    assert event is created_event
    assert context.create_recorded_event.call_count == 2


def test_create_recorded_event_before_registration_raises(fake_adapter):
    adapter, _send_mock, _future = fake_adapter
    with pytest.raises(RuntimeError, match="register_kv_caches"):
        adapter.create_recorded_event()


def test_create_recorded_event_skips_unregistered_recovery_context(fake_adapter):
    """Degraded-mode forwards must not touch a context still registering."""
    adapter, _send_mock, _future = fake_adapter
    recovery_context = MagicMock()
    adapter.transfer_ctx = recovery_context
    adapter._health_event.clear()

    assert adapter.create_recorded_event() is None
    recovery_context.create_recorded_event.assert_not_called()


def test_none_event_is_not_retained(fake_adapter, monkeypatch):
    """Synchronous engine-driven requests do not retain a placeholder event."""
    adapter, _send_mock, _future = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    transfer_ctx = MagicMock()
    transfer_ctx.submit_store.return_value = MagicMock()
    transfer_ctx.submit_retrieve.return_value = MagicMock()
    adapter.transfer_ctx = transfer_ctx

    adapter.submit_store_request("store", _op([[0]]), None)
    adapter.submit_retrieve_request("retrieve", _op([[1]]), None)

    assert "store" not in adapter.store_events
    assert "retrieve" not in adapter.retrieve_events


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


@pytest.mark.parametrize("lazy_offload", [False, True])
def test_failed_retrieve_marks_blocks_for_recompute(
    fake_adapter,
    lazy_offload: bool,
) -> None:
    """A terminal False retrieve never exposes unloaded KV to vLLM."""
    adapter, _send_mock, _future = fake_adapter
    adapter.lazy_offload = lazy_offload
    retrieve_future = MagicMock(name="retrieve_future")
    retrieve_future.query.return_value = True
    retrieve_future.result.return_value = False
    adapter.retrieve_futures["req-1"] = (retrieve_future, [7, 8])

    if lazy_offload:
        finished_stores, finished_retrieves = adapter.get_finished_with_lazy_offload()
        assert finished_stores is None
    else:
        finished_stores, finished_retrieves = adapter.get_finished(set())
        assert finished_stores == set()

    assert finished_retrieves == {"req-1"}
    assert adapter.get_block_ids_with_load_errors() == {7, 8}
    assert "req-1" not in adapter.retrieve_futures


def test_failed_full_retrieve_is_recomputed_instead_of_retried_remotely() -> None:
    """A failed full async load must not re-enter remote wait forever."""
    pytest.importorskip("vllm")

    # Third Party
    from vllm.v1.request import RequestStatus

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestState,
        LMCacheMPRequestTracker,
    )

    class _Request:
        def __init__(self) -> None:
            self.request_id = "req-1"
            self.status = RequestStatus.WAITING
            self.num_computed_tokens = 0
            self.num_preemptions = 0
            self.cache_salt = ""
            self.prompt_token_ids = [1, 2, 3, 4]
            self.all_token_ids = [1, 2, 3, 4]
            self.mm_features: list[object] = []

    request = _Request()
    tracker = LMCacheMPRequestTracker(request)  # type: ignore[arg-type]
    tracker.state = LMCacheMPRequestState.READY
    tracker.num_lmcache_hit_tokens = 4
    tracker.num_stored_tokens = 4
    tracker.allocated_block_ids = {0: [7]}

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {request.request_id: tracker}
    connector.scheduler_adapter = MagicMock(name="scheduler_adapter")

    matched_tokens, load_async = connector.get_num_new_matched_tokens(
        request,
        num_computed_tokens=0,  # type: ignore[arg-type]
    )
    second_result = connector.get_num_new_matched_tokens(
        request,
        num_computed_tokens=0,  # type: ignore[arg-type]
    )

    assert (matched_tokens, load_async) == (0, False)
    assert second_result == (0, False)
    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()
    connector.scheduler_adapter.free_lookup_locks.assert_not_called()
    connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with("req-1")
    assert tracker.state == LMCacheMPRequestState.BYPASS_LMCACHE
    assert tracker.allocated_block_ids == {}
    assert tracker.num_stored_tokens == 0
    assert tracker.num_vllm_hit_tokens == 0
    assert tracker.num_lmcache_hit_tokens == 0

    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([7],)
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)
    assert tracker.state == LMCacheMPRequestState.READY


def test_instance_id_is_uuid_derived_63_bit_int(fake_adapter) -> None:
    """instance_id is a 63-bit int, not the PID, and unique per adapter."""
    adapter, _send_mock, _ = fake_adapter

    assert isinstance(adapter.instance_id, int)
    assert not isinstance(adapter.instance_id, bool)
    assert 0 <= adapter.instance_id < 2**63
    assert adapter.instance_id != os.getpid()

    other = _make_worker_adapter()
    assert other.instance_id != adapter.instance_id


def test_instance_id_logged_at_info_on_construction(fake_adapter, monkeypatch) -> None:
    """The constructor logs instance_id at INFO for correlating server-side
    reap warnings. The module logger does not propagate (``propagate=False``),
    so the test spies on it directly instead of using ``caplog``."""
    _adapter, _send_mock, _ = fake_adapter
    messages: list[str] = []

    def spy_info(msg: object, *args: object, **kwargs: object) -> None:
        messages.append(str(msg) % args if args else str(msg))

    monkeypatch.setattr(adapter_mod.logger, "info", spy_info)

    adapter = _make_worker_adapter()

    assert any(str(adapter.instance_id) in msg for msg in messages)


def test_heartbeat_lazy_start_wires_callback_before_start(fake_adapter) -> None:
    """The lazy create path starts the heartbeat healthy (no pessimistic
    clear) and wires the recover callback before ``start()``; the first
    store is not gated. Idempotent on re-entry (no second thread)."""
    adapter, _send_mock, _ = fake_adapter
    adapter.transfer_ctx = MagicMock()
    assert adapter.is_healthy  # the constructor leaves the event set

    adapter.submit_store_request("req-1", _op([[0]]), MagicMock())

    assert len(FakeHeartbeatThread.instances) == 1
    heartbeat = FakeHeartbeatThread.instances[0]
    # Started healthy: the event was NOT cleared before construction, so
    # the first store is not dropped.
    assert heartbeat.health_event_set_at_init is True
    # The recover callback is wired before start() (for genuine recovery).
    assert heartbeat.calls == ["register_recover_callback", "start"]
    assert adapter.is_healthy
    assert adapter.transfer_ctx.submit_store.call_count == 1

    # Re-entry is idempotent: no new thread.
    adapter.submit_store_request("req-2", _op([[1]]), MagicMock())
    assert len(FakeHeartbeatThread.instances) == 1
    assert adapter.transfer_ctx.submit_store.call_count == 2


def test_heartbeat_first_ping_runs_callback_before_setting_event(
    monkeypatch,
) -> None:
    """Real HeartbeatThread: started with the health event cleared, the
    first successful ping invokes the recover callback while the event
    is still cleared, then sets the event."""
    monkeypatch.setattr(
        adapter_mod, "send_ping", lambda req_client, timeout, instance_id=None: True
    )
    health_event = threading.Event()  # cleared: pessimistic start state
    heartbeat = HeartbeatThread(
        req_client=MagicMock(name="req_client"),
        health_event=health_event,
        interval=60.0,
    )
    event_state_during_callback: list[bool] = []

    def recover() -> bool:
        event_state_during_callback.append(health_event.is_set())
        return True

    heartbeat.register_recover_callback(recover)
    try:
        heartbeat.start()
        assert health_event.wait(timeout=10.0)
    finally:
        heartbeat.stop(timeout=10.0)

    assert event_state_during_callback == [False]


def test_dropped_retrieve_reported_once_via_unhealthy_get_finished(
    fake_adapter,
) -> None:
    """A retrieve submitted while unhealthy is dropped (blocks flagged,
    nothing sent) and reported exactly once by the unhealthy branch of
    ``get_finished``."""
    adapter, _send_mock, _ = fake_adapter
    transfer_ctx = MagicMock()
    adapter.transfer_ctx = transfer_ctx
    # Simulate a failed first ping: the heartbeat start clears the event.
    FakeHeartbeatThread.start_hook = lambda hb: hb.health_event.clear()

    adapter.submit_retrieve_request("req-1", _op([[3, 4]]), MagicMock())

    assert not adapter.is_healthy
    assert not transfer_ctx.submit_retrieve.called

    ret_stores, finished_retrieves = adapter.get_finished(set())
    assert ret_stores == set()
    assert finished_retrieves == {"req-1"}
    assert adapter.get_block_ids_with_load_errors() == {3, 4}

    # Exactly once: a second poll must not re-report the request.
    _ret_stores, finished_retrieves = adapter.get_finished(set())
    assert finished_retrieves == set()


def test_dropped_retrieve_reported_once_via_healthy_get_finished(
    fake_adapter,
) -> None:
    """A retrieve dropped while unhealthy is still reported exactly once
    by the healthy branch of ``get_finished`` after the server
    recovers."""
    adapter, _send_mock, _ = fake_adapter
    adapter.transfer_ctx = MagicMock()
    # Simulate a failed first ping: the heartbeat start clears the event.
    FakeHeartbeatThread.start_hook = lambda hb: hb.health_event.clear()

    adapter.submit_retrieve_request("req-1", _op([[5]]), MagicMock())
    assert not adapter.is_healthy

    # Server recovers: the next heartbeat cycle takes the edge.
    FakeHeartbeatThread.instances[0].simulate_successful_ping()
    assert adapter.is_healthy

    _ret_stores, finished_retrieves = adapter.get_finished(set())
    assert finished_retrieves == {"req-1"}
    assert adapter.get_block_ids_with_load_errors() == {5}

    _ret_stores, finished_retrieves = adapter.get_finished(set())
    assert finished_retrieves == set()


def test_shutdown_stops_heartbeat_before_unregister(fake_adapter) -> None:
    """shutdown() stops the heartbeat before sending UNREGISTER, so no
    stray heartbeat ping can race the closing req_client."""
    adapter, req_client, future = fake_adapter
    transfer_context = MagicMock()
    adapter.transfer_ctx = transfer_context
    adapter.submit_store_request("req-1", _op([[0]]), MagicMock())
    heartbeat = FakeHeartbeatThread.instances[0]

    stop_state_at_unregister: list[bool] = []

    def record_unregister() -> MagicMock:
        stop_state_at_unregister.append(heartbeat.stop_requested)
        return future

    transfer_context.unregister.side_effect = record_unregister

    adapter.shutdown()

    assert "stop" in heartbeat.calls
    assert stop_state_at_unregister == [True]
    transfer_context.unregister.assert_called_once_with()
    req_client.unregister_kv_cache.assert_not_called()


def test_cold_shutdown_skips_unregister(fake_adapter) -> None:
    """shutdown() on an adapter whose heartbeat was never lazily started
    (cold shutdown before registration) does not send UNREGISTER."""
    adapter, req_client, _future = fake_adapter

    adapter.shutdown()

    assert FakeHeartbeatThread.instances == []
    req_client.unregister_kv_cache.assert_not_called()


def test_straggler_cycle_after_stop_skips_callback_and_event(monkeypatch) -> None:
    """Real HeartbeatThread: a ping still in flight when ``stop()`` returns
    completes without firing the recover callback or setting the health
    event — a straggler success must not re-register a ghost context."""
    ping_entered = threading.Event()
    release_ping = threading.Event()

    def slow_ping(
        req_client: object, timeout: float, instance_id: int | None = None
    ) -> bool:
        ping_entered.set()
        release_ping.wait(timeout=10.0)
        return True

    monkeypatch.setattr(adapter_mod, "send_ping", slow_ping)
    health_event = threading.Event()  # cleared: a success would take the edge
    heartbeat = HeartbeatThread(
        req_client=MagicMock(name="req_client"),
        health_event=health_event,
        interval=60.0,
    )
    callback = MagicMock(name="recover_callback", return_value=True)
    heartbeat.register_recover_callback(callback)

    heartbeat.start()
    assert ping_entered.wait(timeout=10.0)
    # The join times out while the ping is still in flight.
    heartbeat.stop(timeout=0.05)
    release_ping.set()

    # Wait for the straggler cycle to complete.
    deadline = time.time() + 10.0
    while heartbeat.total_runs == 0 and time.time() < deadline:
        time.sleep(0.01)

    assert heartbeat.total_runs == 1
    callback.assert_not_called()
    assert not health_event.is_set()


def test_recover_callback_skips_register_after_stop_requested(
    fake_adapter, monkeypatch
) -> None:
    """A recover callback that observes a requested stop bails out before
    submitting REGISTER: a REGISTER submitted after UNREGISTER would
    re-create a ghost server-side context."""
    adapter, _send_mock, _ = fake_adapter
    contexts = _patch_transfer_context_factory(monkeypatch)

    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.register_kv_caches({"layer.0": fake_tensor})
    adapter.submit_store_request("req-1", _op([[0]]), MagicMock())
    heartbeat = FakeHeartbeatThread.instances[0]
    assert heartbeat.recover_callback is not None
    rebuilds_before = len(contexts)

    # Simulate a stop landing while a recovery cycle is in flight: the
    # pre-submission re-check must refuse to re-register.
    heartbeat.stop()
    assert heartbeat.recover_callback() is False

    assert len(contexts) == rebuilds_before  # no new transfer context
    assert contexts[-1].register.call_count == 1  # no second REGISTER


def test_register_uses_local_context_when_self_transfer_ctx_nulled(
    monkeypatch,
) -> None:
    """register must call register() on the local context, not re-read
    self.transfer_ctx: a concurrent shutdown() can null the attribute
    between publish and the call, which previously raised AttributeError."""

    class _NullingTransferCtxAdapter(LMCacheMPWorkerAdapter):
        # Models self.transfer_ctx already nulled by a racing shutdown():
        # the getter always reports None, so any code re-reading the
        # attribute (rather than the local) hits None.register.
        @property
        def transfer_ctx(self):
            return None

        @transfer_ctx.setter
        def transfer_ctx(self, value):
            pass

    req_client = MagicMock(name="req_client", spec=RequestClient)
    _patch_request_client_factory(monkeypatch, req_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())
    future = MagicMock(name="future")
    future.result.return_value = None
    _return_future_from_request_methods(req_client, future)
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", FakeHeartbeatThread)
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    monkeypatch.setattr(
        worker_transfer,
        "wrap_kv_caches",
        lambda kv: list(kv.values()),
    )
    monkeypatch.setattr("lmcache.integration.vllm.utils.vllm_layout_hints", lambda: {})
    local_ctx = MagicMock(name="local_transfer_ctx")
    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", lambda kv, **_kwargs: local_ctx
    )

    parallel_strategy = ParallelStrategy(
        mla_only=False,
        vllm_world_size=1,
        vllm_worker_id=0,
        tp_size=1,
        pp_size=1,
        n_servers=1,
    )
    adapter = _NullingTransferCtxAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=parallel_strategy,
        mq_timeout=5.0,
    )

    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    # Under the bug this raises AttributeError (None.register).
    adapter.register_kv_caches({"layer.0": fake_tensor})

    local_ctx.register.assert_called_once()


def test_startup_warns_when_heartbeat_interval_exceeds_reap_floor(
    fake_adapter, monkeypatch
) -> None:
    """3 x heartbeat_interval > 30 s emits a startup WARNING to raise the
    server's worker reap timeout. The module logger does not propagate
    (``propagate=False``), so the test spies on it instead of ``caplog``."""
    _adapter, _send_mock, _ = fake_adapter
    warnings: list[str] = []
    monkeypatch.setattr(
        adapter_mod.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(str(msg)),
    )

    _make_worker_adapter(extra_config={"lmcache.mp.heartbeat_interval": 15})

    assert any("reap" in msg for msg in warnings)


def test_startup_does_not_warn_for_default_heartbeat_interval(
    fake_adapter, monkeypatch
) -> None:
    """The default 10 s heartbeat interval (3 x 10 s == 30 s floor) must
    not emit the reap-timeout startup WARNING."""
    _adapter, _send_mock, _ = fake_adapter
    warnings: list[str] = []
    monkeypatch.setattr(
        adapter_mod.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(str(msg)),
    )

    _make_worker_adapter()

    assert not any("reap" in msg for msg in warnings)


def test_recover_callback_rebuilds_transfer_ctx_without_closing_previous(
    fake_adapter, monkeypatch
) -> None:
    """Pin current behavior: every recover-callback invocation rebuilds
    ``transfer_ctx`` without closing the previous context (known IPC leak;
    in-flight submissions may still hold a reference to the old context)."""
    adapter, _send_mock, _ = fake_adapter
    contexts = _patch_transfer_context_factory(monkeypatch)

    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.register_kv_caches({"layer.0": fake_tensor})  # contexts[0]
    # Start the heartbeat (healthy, no recover) so the callback is wired.
    adapter.submit_store_request("req-1", _op([[0]]), MagicMock())
    heartbeat = FakeHeartbeatThread.instances[0]
    assert heartbeat.recover_callback is not None
    assert len(contexts) == 1
    assert adapter.transfer_ctx is contexts[0]

    # Each recover-callback invocation rebuilds transfer_ctx without closing
    # the previous context (known IPC leak; in-flight submissions may still
    # hold a reference to the old context).
    assert heartbeat.recover_callback() is True
    assert len(contexts) == 2
    assert adapter.transfer_ctx is contexts[1]
    contexts[0].close.assert_not_called()

    assert heartbeat.recover_callback() is True
    assert len(contexts) == 3
    assert adapter.transfer_ctx is contexts[2]
    contexts[1].close.assert_not_called()


# For the experimental dispatcher
def test_enabled_feature_receives_reclaim_and_shutdown(fake_adapter):
    """get_finished reclaims ring blocks and shutdown unregisters the ring."""
    adapter, _, _ = fake_adapter
    assert adapter.dispatcher is None

    dispatcher = MagicMock(spec=Dispatcher)
    adapter.dispatcher = dispatcher

    adapter.get_finished(set())
    adapter.shutdown()

    dispatcher.reclaim.assert_called_once_with()
    dispatcher.shutdown.assert_called_once_with()


@pytest.mark.parametrize("ring_ok", [True, False])
def test_recovery_reports_the_ring_re_registration_result(fake_adapter, ring_ok):
    """A worker whose Q ring failed to re-register must not be reported
    healthy: the server would have no Q context, so every later STORE_Q would
    raise. The success path must still report healthy so the health event can
    be set."""
    adapter, _, _ = fake_adapter
    dispatcher = MagicMock(spec=Dispatcher)
    dispatcher.reregister.return_value = ring_ok
    adapter.dispatcher = dispatcher
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.register_kv_caches({"layer.0": fake_tensor})

    assert adapter._reregister_kv_caches_callback() is ring_ok


# -- Server-side KV event polling ---------------------------------------------


class _FakePollServer:
    """Answers ``poll_kv_events``. A poll completes once an answer is queued,
    so each ``get_kv_events`` call consumes at most one answer and issues at
    most one poll, in a deterministic order."""

    def __init__(self, req_client: MagicMock) -> None:
        self.answers: collections.deque[KVEventPollResult] = collections.deque()
        self.calls: list[tuple[str, int, int]] = []
        req_client.poll_kv_events.side_effect = self._poll

    def answer(self, *results: KVEventPollResult) -> None:
        self.answers.extend(results)

    def _poll(self, model_name: str, cursor: int, max_events: int) -> object:
        self.calls.append((model_name, cursor, max_events))
        server = self

        class _Future:
            def query(self) -> bool:
                return bool(server.answers)

            def result(self, timeout: float | None = None) -> KVEventPollResult:
                return server.answers.popleft()

        return _Future()


def _poll_result(
    events: list[KVEventRecord] | None = None,
    *,
    incarnation: int = 7,
    next_cursor: int = 0,
    lost: bool = False,
    enabled: bool = True,
) -> KVEventPollResult:
    return KVEventPollResult(
        enabled=enabled,
        incarnation=incarnation,
        next_cursor=next_cursor,
        lost=lost,
        events=list(events or []),
    )


def _record(
    seq: int,
    kind: str,
    hashes: list[bytes],
    *,
    medium: str = KV_EVENT_MEDIUM_CPU,
    token_ids: list[int] | None = None,
    parent: bytes | None = None,
) -> KVEventRecord:
    return KVEventRecord(
        seq=seq,
        kind=kind,
        medium=medium,
        model_name="test-model",
        block_hashes=list(hashes),
        parent_block_hash=parent,
        token_ids=list(token_ids or []),
    )


def _polling_adapter(
    fake_adapter,
    parallel_strategy: ParallelStrategy | None = None,
    extra_config: dict[str, object] | None = None,
) -> tuple[LMCacheMPWorkerAdapter, _FakePollServer]:
    """A KV-event-enabled adapter whose poll interval never throttles."""
    _adapter, req_client, _future = fake_adapter
    config: dict[str, object] = {"lmcache.mp.kv_event_poll_interval": 1e-6}
    config.update(extra_config or {})
    adapter = _make_worker_adapter(
        extra_config=config,
        enable_kv_events=True,
        parallel_strategy=parallel_strategy,
    )
    adapter.transfer_ctx = MagicMock()
    return adapter, _FakePollServer(req_client)


def _complete_own_store(
    adapter: LMCacheMPWorkerAdapter, request_id: str, chunk_index: int
) -> bytes:
    """Store one chunk and report it finished; return its chunk hash."""
    # First Party
    from lmcache.v1.multiprocess.token_hasher import TokenHasher

    chunk_size = adapter.lmcache_tokens_per_chunk
    token_ids = [chunk_index * 1000 + i for i in range(chunk_size)]
    transfer_ctx = MagicMock()
    future = MagicMock()
    future.query.return_value = True
    future.result.return_value = True
    transfer_ctx.submit_store.return_value = future
    adapter.transfer_ctx = transfer_ctx
    adapter.submit_store_request(
        request_id,
        LoadStoreOp(token_ids=token_ids, block_ids=[[0]], start=0, end=chunk_size),
        event=None,
    )
    adapter.get_finished({request_id})
    return TokenHasher(chunk_size=chunk_size).compute_chunk_hashes(token_ids)[0]


def _resync_count(reason: str) -> float:
    labels = {"model_name": "test-model", "worker_id": "0", "reason": reason}
    return sum(
        sample.value
        for family in adapter_mod._KV_EVENT_RESYNCS.collect()
        for sample in family.samples
        if sample.name == "vllm:lmcache_mp_kv_event_resyncs_total"
        and sample.labels == labels
    )


def _step(adapter: LMCacheMPWorkerAdapter) -> list[CacheEvent]:
    """One model-runner step: let the poll interval elapse, then drain."""
    time.sleep(0.001)
    return adapter.get_kv_events()


def test_polling_reports_evictions_of_announced_chunks_only(fake_adapter) -> None:
    """A server removal becomes a CacheRemoveEvent for chunks announced from
    the server's records; chunks never announced are not withdrawn."""
    adapter, server = _polling_adapter(fake_adapter)
    server.answer(
        _poll_result(
            [_record(1, KV_EVENT_KIND_STORED, [b"h1"], token_ids=[1])], next_cursor=1
        )
    )
    assert _step(adapter) == []  # issues the first poll
    assert [type(e).__name__ for e in _step(adapter)] == ["CacheStoreEvent"]
    assert server.calls[0] == ("test-model", 0, 1024)

    server.answer(
        _poll_result(
            [_record(2, KV_EVENT_KIND_REMOVED, [b"never-announced", b"h1"])],
            next_cursor=2,
        )
    )
    events = _step(adapter)
    assert len(events) == 1
    assert isinstance(events[0], CacheRemoveEvent)
    assert (events[0].block_hashes, events[0].medium) == ([b"h1"], "CPU")
    # The next poll continues from the returned cursor.
    assert server.calls[-1] == ("test-model", 2, 1024)

    # A second removal of the same chunk is not reported again.
    server.answer(
        _poll_result([_record(3, KV_EVENT_KIND_REMOVED, [b"h1"])], next_cursor=3)
    )
    assert _step(adapter) == []


def test_polling_reports_server_stores_once_and_lets_them_be_evicted(
    fake_adapter,
) -> None:
    adapter, server = _polling_adapter(fake_adapter)
    stored = _record(
        1,
        KV_EVENT_KIND_STORED,
        [b"server-chunk"],
        token_ids=[1, 2, 3, 4],
        parent=b"parent",
    )
    server.answer(_poll_result([stored, stored], next_cursor=2))
    assert _step(adapter) == []  # issues the first poll

    events = _step(adapter)
    assert len(events) == 1
    assert isinstance(events[0], CacheStoreEvent)
    assert events[0].block_hashes == [b"server-chunk"]
    assert events[0].parent_block_hash == b"parent"
    assert events[0].token_ids == [1, 2, 3, 4]
    assert (events[0].block_size, events[0].medium) == (
        adapter.lmcache_tokens_per_chunk,
        "CPU",
    )

    server.answer(
        _poll_result(
            [_record(3, KV_EVENT_KIND_REMOVED, [b"server-chunk"])], next_cursor=3
        )
    )
    assert [type(e).__name__ for e in _step(adapter)] == ["CacheRemoveEvent"]


def test_own_store_completions_are_not_announced_while_polling(fake_adapter) -> None:
    """With the server's log polled, only its write-finished records announce
    stores: a store result does not say which chunks were written."""
    adapter, server = _polling_adapter(fake_adapter)
    own = _complete_own_store(adapter, "req-1", 1)
    assert _step(adapter) == []  # the own completion announces nothing

    server.answer(
        _poll_result(
            [_record(1, KV_EVENT_KIND_STORED, [own], token_ids=[1])], next_cursor=1
        )
    )
    events = _step(adapter)
    assert [(type(e).__name__, e.block_hashes) for e in events] == [
        ("CacheStoreEvent", [own])
    ]


def test_server_restart_withdraws_every_announced_chunk(fake_adapter) -> None:
    adapter, server = _polling_adapter(fake_adapter)
    server.answer(
        _poll_result(
            [
                _record(1, KV_EVENT_KIND_STORED, [b"cpu-chunk"], token_ids=[1]),
                _record(
                    2,
                    KV_EVENT_KIND_STORED,
                    [b"l2-chunk"],
                    medium=KV_EVENT_MEDIUM_STORAGE,
                    token_ids=[1],
                ),
            ],
            incarnation=7,
            next_cursor=2,
        )
    )
    _step(adapter)
    assert [e.medium for e in _step(adapter)] == ["CPU", "STORAGE"]
    before = _resync_count("server_restart")

    server.answer(_poll_result(incarnation=8, next_cursor=0))
    events = _step(adapter)
    assert all(isinstance(e, CacheRemoveEvent) for e in events)
    assert {(e.medium, tuple(e.block_hashes)) for e in events} == {
        ("CPU", (b"cpu-chunk",)),
        ("STORAGE", (b"l2-chunk",)),
    }
    assert _resync_count("server_restart") == before + 1

    # Nothing is announced any more, so nothing is withdrawn twice.
    server.answer(_poll_result(incarnation=9, next_cursor=0))
    assert _step(adapter) == []


def test_lost_events_withdraw_announced_chunks(fake_adapter) -> None:
    adapter, server = _polling_adapter(fake_adapter)
    server.answer(
        _poll_result(
            [_record(1, KV_EVENT_KIND_STORED, [b"h1"], token_ids=[1])], next_cursor=1
        )
    )
    _step(adapter)
    assert len(_step(adapter)) == 1
    before = _resync_count("events_lost")

    server.answer(
        _poll_result(
            [_record(41, KV_EVENT_KIND_STORED, [b"after-loss"], token_ids=[1])],
            lost=True,
            next_cursor=41,
        )
    )
    events = _step(adapter)
    assert [(type(e).__name__, e.block_hashes) for e in events] == [
        ("CacheRemoveEvent", [b"h1"]),
        ("CacheStoreEvent", [b"after-loss"]),
    ]
    assert _resync_count("events_lost") == before + 1
    assert server.calls[-1] == ("test-model", 41, 1024)


def test_lost_events_on_first_contact_do_not_resync(fake_adapter) -> None:
    adapter, server = _polling_adapter(fake_adapter)
    before = _resync_count("events_lost")
    server.answer(_poll_result(lost=True, next_cursor=40))
    _step(adapter)
    assert _step(adapter) == []
    assert _resync_count("events_lost") == before
    assert server.calls[-1] == ("test-model", 40, 1024)


@pytest.mark.parametrize("failure", ["disabled_on_server", "no_client_method"])
def test_polling_stops_when_the_server_cannot_serve_it(
    fake_adapter, failure: str
) -> None:
    """Polling stops for good and the rank resumes announcing its own stores."""
    adapter, server = _polling_adapter(fake_adapter)
    _adapter, req_client, _future = fake_adapter
    if failure == "disabled_on_server":
        server.answer(_poll_result(enabled=False))
        _step(adapter)
        _step(adapter)
        assert len(server.calls) == 1
    else:
        req_client.poll_kv_events = None
        _step(adapter)

    # Own store events keep flowing, and no further poll is issued.
    _complete_own_store(adapter, "req-1", 1)
    assert len(_step(adapter)) == 1
    assert len(server.calls) == (1 if failure == "disabled_on_server" else 0)


def test_pending_poll_times_out_only_while_healthy(fake_adapter) -> None:
    adapter, server = _polling_adapter(
        fake_adapter, extra_config={"lmcache.mp.mq_timeout": 0.0}
    )
    _complete_own_store(adapter, "req-1", 1)  # starts the (fake) heartbeat
    heartbeat = FakeHeartbeatThread.instances[-1]
    _step(adapter)
    assert len(server.calls) == 1  # unanswered

    # Unhealthy: the stale poll is dropped, but polling resumes on recovery.
    heartbeat.health_event.clear()
    _step(adapter)
    _step(adapter)
    assert len(server.calls) == 1
    heartbeat.health_event.set()
    _step(adapter)
    assert len(server.calls) == 2

    # Healthy and unanswered: the server predates the request; stop for good.
    _step(adapter)
    _step(adapter)
    assert len(server.calls) == 2


def test_a_full_page_is_followed_by_an_immediate_poll(
    fake_adapter, monkeypatch
) -> None:
    monkeypatch.setattr(adapter_mod, "_KV_EVENT_POLL_PAGE", 1)
    adapter, server = _polling_adapter(
        fake_adapter,
        extra_config={"lmcache.mp.kv_event_poll_interval": 100.0},
    )
    server.answer(
        _poll_result([_record(1, KV_EVENT_KIND_REMOVED, [b"x"])], next_cursor=1)
    )
    _step(adapter)
    _step(adapter)  # consumes the full page and polls again at once
    assert server.calls == [("test-model", 0, 1), ("test-model", 1, 1)]

    server.answer(_poll_result(next_cursor=1))
    _step(adapter)
    _step(adapter)  # an empty page: the 100 s interval applies again
    assert len(server.calls) == 2


def test_no_polling_without_kv_events_or_with_a_zero_interval(fake_adapter) -> None:
    _adapter, req_client, _future = fake_adapter
    cases: list[tuple[bool, dict[str, object]]] = [
        (False, {"lmcache.mp.kv_event_poll_interval": 1e-6}),
        (True, {"lmcache.mp.kv_event_poll_interval": 0}),
    ]
    for enable_kv_events, extra_config in cases:
        adapter = _make_worker_adapter(
            extra_config=extra_config, enable_kv_events=enable_kv_events
        )
        assert adapter.get_kv_events() == []
    req_client.poll_kv_events.assert_not_called()


# -- Single publisher and server capability ------------------------------------


def _strategy(worker_id: int, world_size: int, n_servers: int = 1) -> ParallelStrategy:
    return ParallelStrategy(
        mla_only=False,
        vllm_world_size=world_size,
        vllm_worker_id=worker_id,
        tp_size=world_size // n_servers,
        pp_size=1,
        n_servers=n_servers,
    )


@pytest.mark.parametrize(
    ("worker_id", "world_size", "n_servers", "is_poller"),
    [
        (0, 1, 1, True),
        (0, 2, 1, True),
        (1, 2, 1, False),
        (3, 4, 1, False),
        # Two servers, two ranks each: the first rank of each block polls
        # its own server's log.
        (0, 4, 2, True),
        (1, 4, 2, False),
        (2, 4, 2, True),
        (3, 4, 2, False),
    ],
)
def test_one_rank_per_server_is_the_poller(
    worker_id: int, world_size: int, n_servers: int, is_poller: bool
) -> None:
    """Every rank reads the same records, so only one may republish them."""
    assert _strategy(worker_id, world_size, n_servers).is_kv_event_poller is is_poller


@pytest.mark.parametrize(("worker_id", "is_poller"), [(0, True), (1, False)])
def test_only_one_rank_per_engine_publishes(
    fake_adapter, worker_id: int, is_poller: bool
) -> None:
    """Every rank of an engine reads the same records, so only one may
    republish them: a repeated BlockRemoved is an error for the router. The
    silent ranks must not fall back to their own store events either, or the
    engine would have two publishers."""
    adapter, server = _polling_adapter(fake_adapter, _strategy(worker_id, 2))
    server.answer(
        _poll_result(
            [_record(1, KV_EVENT_KIND_STORED, [b"h1"], token_ids=[1])], next_cursor=1
        )
    )
    _step(adapter)
    events = _step(adapter)

    assert bool(server.calls) is is_poller
    assert [type(e).__name__ for e in events] == (
        ["CacheStoreEvent"] if is_poller else []
    )

    # While the server's log is the source, no rank announces its own stores.
    _complete_own_store(adapter, "req-own", 9)
    assert _step(adapter) == []


def test_an_unadvertised_server_is_never_polled(fake_adapter, monkeypatch) -> None:
    """A server that predates POLL_KV_EVENTS aborts its request loop on the
    unknown request type, so the worker keeps to its own completed stores."""
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())
    adapter, server = _polling_adapter(fake_adapter)

    own = _complete_own_store(adapter, "req-1", 1)
    events = _step(adapter)

    assert server.calls == []
    assert [(type(e).__name__, e.block_hashes) for e in events] == [
        ("CacheStoreEvent", [own])
    ]
