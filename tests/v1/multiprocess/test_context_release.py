# SPDX-License-Identifier: Apache-2.0
"""Context releases run off the MQ main loop and the reaper (LMCache#4676).

``GPUCacheContext.close()`` drains the context stream and unmaps the
worker's IPC segments. Behind a stream that waits on the event of a worker
that died, it never returns. These tests stand in for that close with one
that blocks on an event, and check that only later releases on the same
device wait for it.
"""

# Standard
from typing import Any, Literal, cast
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as gpu_mod

#: Generous bound for a call that must not wait on the blocked release.
_PROMPT_S = 5.0


class _Contexts:
    """Stands in for ``create_cache_context``; each instance gets a mock
    context on its own device, whose ``close`` can be made to block."""

    def __init__(self) -> None:
        self.by_instance: dict[int, MagicMock] = {}
        self.devices: dict[int, str] = {}
        self.blocked: dict[int, threading.Event] = {}
        self.closed: list[int] = []

    def create(self, *args: Any, instance_id: int, **kwargs: Any) -> MagicMock:
        ctx = MagicMock(name=f"cache_context-{instance_id}")
        ctx.num_layers = 1
        ctx.device = self.devices.get(instance_id, "cuda:0")

        def close() -> None:
            gate = self.blocked.get(instance_id)
            if gate is not None:
                assert gate.wait(timeout=30), "test never unblocked the close"
            self.closed.append(instance_id)

        ctx.close.side_effect = close
        self.by_instance[instance_id] = ctx
        return ctx


@pytest.fixture
def contexts(monkeypatch) -> _Contexts:
    contexts = _Contexts()
    monkeypatch.setattr(gpu_mod, "DeviceHostFuncDispatcher", MagicMock())
    monkeypatch.setattr(
        gpu_mod, "get_event_ipc_backend", lambda device: MagicMock(name="backend")
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(gpu_mod, "get_unfull_chunk_layout_desc", MagicMock())
    monkeypatch.setattr(gpu_mod, "torch_dev", MagicMock(name="torch_dev"))
    monkeypatch.setattr(
        LMCacheDrivenTransferModule,
        "_create_cache_context",
        lambda self, kv_caches, *, instance_id, **kw: contexts.create(
            instance_id=instance_id
        ),
    )
    return contexts


def _module() -> LMCacheDrivenTransferModule:
    ctx = MagicMock(name="ctx")
    ctx.chunk_size = 256
    ctx.finalize_chunk_size.return_value = 256
    return LMCacheDrivenTransferModule(ctx)


def _register(
    module: LMCacheDrivenTransferModule, instance_id: int, model: str = "m"
) -> None:
    module.register_kv_cache(
        instance_id,
        kv_caches=[],
        model_name=model,
        world_size=1,
        engine_type=EngineType.VLLM,
        layout_hints={},
        engine_group_infos=[],
        event_profile=None,
    )


def _block(contexts: _Contexts, instance_id: int) -> threading.Event:
    gate = threading.Event()
    contexts.blocked[instance_id] = gate
    return gate


def test_unregister_returns_while_its_release_is_blocked(contexts) -> None:
    module = _module()
    _register(module, 1)
    gate = _block(contexts, 1)
    try:
        start = time.monotonic()
        module.unregister_kv_cache(1)
        assert time.monotonic() - start < _PROMPT_S

        assert module.context_entries_snapshot() == {}
        assert module.report_status()["pending_releases"] == {"cuda:0": 1}
        assert not module.wait_for_releases(timeout=0.2)
        cast(
            MagicMock, module.context.layout_desc_registry.unregister
        ).assert_called_once_with("m", 1)
    finally:
        gate.set()
    assert module.wait_for_releases(timeout=_PROMPT_S)
    assert contexts.closed == [1]
    assert module.report_status()["pending_releases"] == {}
    module.close()


def test_blocked_release_does_not_delay_other_devices(contexts) -> None:
    module = _module()
    contexts.devices.update({1: "cuda:0", 2: "cuda:0", 3: "cuda:1"})
    for iid in (1, 2, 3):
        _register(module, iid)
    gate = _block(contexts, 1)
    try:
        module.unregister_kv_cache(1)
        module.unregister_kv_cache(2)
        module.unregister_kv_cache(3)

        deadline = time.monotonic() + _PROMPT_S
        while (
            module.report_status()["pending_releases"] != {"cuda:0": 2}
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        # cuda:1 is released; cuda:0 keeps its order behind the blocked one.
        assert module.report_status()["pending_releases"] == {"cuda:0": 2}
        assert contexts.closed == [3]
    finally:
        gate.set()
    assert module.wait_for_releases(timeout=_PROMPT_S)
    assert contexts.closed == [3, 1, 2]
    module.close()


def test_reaper_keeps_scanning_while_a_reaped_release_is_blocked(
    contexts, monkeypatch
) -> None:
    module = _module()
    contexts.devices.update({1: "cuda:0", 2: "cuda:1"})
    _register(module, 1)
    _register(module, 2)
    module.touch_instance(1)
    module.touch_instance(2)
    gate = _block(contexts, 1)
    real_monotonic = time.monotonic
    try:
        # Instance 1 goes silent first and is reaped; its release blocks.
        monkeypatch.setattr(gpu_mod.time, "monotonic", lambda: real_monotonic() + 100)
        module.touch_instance(2)
        start = real_monotonic()
        assert module.reap_stale_instances(60.0, 3600.0) == [1]
        assert real_monotonic() - start < _PROMPT_S

        # A later scan still runs and releases instance 2 on its device.
        monkeypatch.setattr(gpu_mod.time, "monotonic", lambda: real_monotonic() + 200)
        assert module.reap_stale_instances(60.0, 3600.0) == [2]
        deadline = real_monotonic() + _PROMPT_S
        while 2 not in contexts.closed and real_monotonic() < deadline:
            time.sleep(0.01)
        assert contexts.closed == [2]
    finally:
        monkeypatch.setattr(gpu_mod.time, "monotonic", real_monotonic)
        gate.set()
    assert module.wait_for_releases(timeout=_PROMPT_S)
    module.close()


def test_failed_release_does_not_wedge_its_device(contexts) -> None:
    module = _module()
    _register(module, 1)
    _register(module, 2)
    contexts.by_instance[1].close.side_effect = RuntimeError("unmap failed")

    module.unregister_kv_cache(1)
    module.unregister_kv_cache(2)

    assert module.wait_for_releases(timeout=_PROMPT_S)
    assert contexts.closed == [2]
    assert module.report_status()["pending_releases"] == {}
    module.close()


def test_close_does_not_wait_forever_for_a_blocked_release(
    contexts, monkeypatch
) -> None:
    monkeypatch.setattr(gpu_mod, "_SHUTDOWN_RELEASE_TIMEOUT_S", 0.2)
    module = _module()
    _register(module, 1)
    _register(module, 2)
    gate = _block(contexts, 1)
    try:
        module.unregister_kv_cache(1)
        start = time.monotonic()
        module.close()
        assert time.monotonic() - start < _PROMPT_S
        assert module.context_entries_snapshot() == {}
    finally:
        gate.set()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_blocked_release_does_not_stall_the_message_loop(
    contexts, transport: Literal["zmq", "grpc"]
) -> None:
    """The report on LMCache#4676: with one engine's close() blocked, an
    UNREGISTER from it left another engine without an answer on PING,
    GET_CHUNK_SIZE and REGISTER_KV_CACHE."""
    module = _module()
    management = ManagementModule(module.context, liveness_targets=[module])
    config = MPServerConfig(
        transport=transport,
        host="127.0.0.1",
        port=0,
        max_cpu_workers=2,
        max_gpu_workers=2,
        grpc_server_workers=4,
    )
    server: Any = create_request_server([module, management], config)
    if transport == "grpc":
        url = f"grpc://127.0.0.1:{server.bound_port}"
    else:
        url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    server.start()
    stuck = RequestClientFactory.create(url, context=zmq.Context.instance())
    healthy = RequestClientFactory.create(url, context=zmq.Context.instance())
    gate = _block(contexts, 1)

    def register(client: Any, instance_id: int) -> None:
        future = client.register_kv_cache(
            instance_id, [], "m", 1, EngineType.VLLM, {}, [], None
        )
        assert future.result(timeout=_PROMPT_S) is None

    try:
        register(stuck, 1)
        register(healthy, 2)

        assert stuck.unregister_kv_cache(1).result(timeout=_PROMPT_S) is None
        assert module.report_status()["pending_releases"] == {"cuda:0": 1}

        assert healthy.ping(2).result(timeout=_PROMPT_S) is True
        assert healthy.get_chunk_size().result(timeout=_PROMPT_S) == 256
        register(healthy, 3)
        assert healthy.unregister_kv_cache(3).result(timeout=_PROMPT_S) is None
        assert 1 not in contexts.closed
    finally:
        gate.set()
        stuck.close()
        healthy.close()
        server.close()
        management.close()
        module.close()
    assert contexts.closed[0] == 1
