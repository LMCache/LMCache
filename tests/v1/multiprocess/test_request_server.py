# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request server behavior tests."""

# Standard
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock, patch
import socket
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import affinity_pool as affinity_pool_mod
from lmcache.v1.multiprocess.custom_types import CBMatchResult, IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import EngineModule, InstanceLivenessTarget
from lmcache.v1.multiprocess.modules import management as management_mod
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.request_handler import (
    HandlerType,
    get_affinity_key_index,
    request_handler,
    wrap_affinity_release,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

# Test helpers
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)


@dataclass
class _DispatchState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: int = 0
    max_active: int = 0
    calls: int = 0


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


@pytest.mark.parametrize("request_transport", REQUEST_TRANSPORTS)
def test_sync_handlers_are_serialized(request_transport: RequestTransport) -> None:
    """SYNC handlers must retain the single-main-loop execution contract."""
    state = _DispatchState()

    class SyncModule(EngineModule):
        @property
        def context(self) -> MPCacheServerContext:
            """The dispatch-only test does not use an engine context."""
            raise NotImplementedError

        def report_status(self) -> dict:
            """Return the empty status of this test module."""
            return {}

        def close(self) -> None:
            """Release no resources for this test module."""

        @request_handler()
        def noop(self) -> str:
            with state.lock:
                state.active += 1
                state.calls += 1
                state.max_active = max(state.max_active, state.active)
            try:
                time.sleep(0.05)
            finally:
                with state.lock:
                    state.active -= 1
            return "ok"

    worker_count = 8
    server_url = request_server_url(request_transport, _unused_tcp_port())
    config = request_server_config(request_transport, server_url)
    server = create_request_server([SyncModule()], config)
    server.start()
    clients: list[RequestClient] = []
    try:
        clients = [RequestClientFactory.create(server_url) for _ in range(worker_count)]
        barrier = threading.Barrier(worker_count)

        def call_noop(client: RequestClient) -> str:
            barrier.wait(timeout=5)
            return cast(str, client.noop().result(timeout=5))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(call_noop, client) for client in clients]
            assert [future.result(timeout=10) for future in futures] == [
                "ok"
            ] * worker_count
    finally:
        for client in clients:
            client.close()
        server.close()

    assert state.calls == worker_count
    assert state.max_active == 1


class _AffinityModule(EngineModule):
    @property
    def context(self) -> MPCacheServerContext:
        """The dispatch-only test does not use an engine context."""
        raise NotImplementedError

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        pass

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        return threading.current_thread().name.encode(), True

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def cb_retrieve_pre_computed(
        self,
        key: IPCCacheServerKey,
        cb_match_result: list[CBMatchResult],
        gpu_block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        return threading.current_thread().name.encode(), True

    @request_handler(releases_client_affinity=True)
    def unregister_kv_cache(self, instance_id: int) -> None:
        if instance_id < 0:
            raise ValueError("unregister failed")

    @request_handler(releases_client_affinity=True)
    def unregister_kv_cache_engine_driven_context(self, instance_id: int) -> None:
        pass

    @request_handler(releases_client_affinity=True)
    def unregister_q_cache(self, instance_id: int) -> None:
        pass


@pytest.fixture(params=REQUEST_TRANSPORTS)
def affinity_server(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, Callable[[int], None]]]:
    """Run real transports/pools with a manually ticked reaper and no GPU state."""
    transport = cast(RequestTransport, request.param)
    server_url = request_server_url(transport, _unused_tcp_port())
    config = request_server_config(transport, server_url, max_gpu_workers=2)
    timer_factory = MagicMock()
    monkeypatch.setattr(management_mod, "create_periodic_thread", timer_factory)
    target = MagicMock(spec=InstanceLivenessTarget)
    target.reap_stale_instances.return_value = []
    target.tracked_instance_count.return_value = 0
    management = ManagementModule(
        MagicMock(spec=MPCacheServerContext),
        liveness_targets=[target],
        worker_reap_timeout_seconds=120,
        worker_registration_grace_seconds=3600,
    )
    server = create_request_server([management, _AffinityModule()], config)
    server.start()

    def reap(instance_id: int) -> None:
        target.reap_stale_instances.return_value = [instance_id]
        timer_factory.call_args.kwargs["execute_fn"]()

    try:
        yield server_url, reap
    finally:
        management.close()
        server.close()


def _store_thread(client: RequestClient, instance_id: int) -> bytes:
    key = IPCCacheServerKey("model", 1, 0, (1,), 0, 1, "affinity-test")
    name, success = client.store(key, instance_id, [[0]], b"").result(timeout=5)
    assert success
    return name


@pytest.mark.parametrize("cleanup", ["reap", "kv", "engine", "q"])
def test_affinity_cleanup_after_repeated_restarts(
    affinity_server: tuple[str, Callable[[int], None]], cleanup: str
) -> None:
    """Both transports retire stale bindings through reaping and unregister RPCs."""
    server_url, reap = affinity_server
    clients = [RequestClientFactory.create(server_url) for _ in range(2)]
    try:
        old, survivor = clients
        retired_key = 1001
        retired_thread = _store_thread(old, retired_key)
        survivor_thread = _store_thread(survivor, 1002)
        assert retired_thread != survivor_thread
        with patch.object(affinity_pool_mod.logger, "warning") as warning:
            for replacement in range(1003, 1008):
                if cleanup != "reap":
                    unregister = {
                        "kv": old.unregister_kv_cache,
                        "engine": old.unregister_kv_cache_engine_driven_context,
                        "q": old.unregister_q_cache,
                    }[cleanup]
                    unregister(retired_key).result(timeout=5)
                old.close()
                clients.remove(old)
                if cleanup == "reap":
                    reap(retired_key)
                old = RequestClientFactory.create(server_url)
                clients.append(old)
                assert _store_thread(old, replacement) == retired_thread
                assert _store_thread(survivor, 1002) == survivor_thread
                retired_key = replacement
            warning.assert_not_called()
    finally:
        for client in clients:
            client.close()


def test_affinity_uses_instance_id_across_connections_and_operations(
    affinity_server: tuple[str, Callable[[int], None]],
) -> None:
    """One instance keeps its thread across connections and payload positions."""
    server_url, _ = affinity_server
    clients = [RequestClientFactory.create(server_url) for _ in range(2)]
    try:
        first, second = clients
        thread = _store_thread(first, 1)
        assert _store_thread(second, 1) == thread
        assert _store_thread(first, 2) != thread
        key = IPCCacheServerKey("model", 1, 0, (1,), 0, 1, "affinity-test")
        assert second.cb_retrieve_pre_computed(key, [], [[0]], 1, b"").result(
            timeout=5
        ) == (thread, True)
    finally:
        for client in clients:
            client.close()


def test_failed_unregister_does_not_release_affinity() -> None:
    release = MagicMock()
    handler = wrap_affinity_release(
        "unregister_kv_cache", _AffinityModule().unregister_kv_cache, release
    )
    with pytest.raises(ValueError, match="unregister failed"):
        handler(-1)
    release.assert_not_called()


@pytest.mark.parametrize("operation", ["noop", "ping"])
def test_affinity_requires_an_integer_instance_id(operation: str) -> None:
    with pytest.raises(ValueError, match="requires an integer instance_id"):
        get_affinity_key_index(operation)
