# SPDX-License-Identifier: Apache-2.0
"""Startup-order regressions for the multiprocess gRPC server."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import MPServerConfig


def _storage_config() -> StorageManagerConfig:
    """Return a minimal storage manager config for startup tests."""
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1,
                use_lazy=False,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


def test_run_cache_server_applies_isolated_ipc_before_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--isolated-ipc`` must be process-global before context construction."""
    calls: list[tuple[str, object]] = []

    class FakeContext:
        def __init__(self, **_kwargs: object) -> None:
            calls.append(("context", None))
            self.storage_manager = SimpleNamespace(report_status=lambda: {})
            self.chunk_size = 256
            self.token_hasher = SimpleNamespace(hash_algorithm_name="blake3")
            self.session_manager = SimpleNamespace(active_count=lambda: 0)

        def close(self) -> None:
            calls.append(("context_close", None))

    class FakeGrpcServer:
        def __init__(self, bind_url: str) -> None:
            calls.append(("grpc_server", bind_url))
            self.services: list[str] = []

        def add_service(self, service_name: str, _implementation: object) -> None:
            self.services.append(service_name)

        def assign_thread_pools(
            self,
            *,
            max_cpu_workers: int,
            max_gpu_workers: int,
        ) -> None:
            calls.append(("assign_thread_pools", (max_cpu_workers, max_gpu_workers)))

        def start(self) -> None:
            calls.append(("server_start", None))

        def close(self) -> None:
            calls.append(("server_close", None))

    fake_rpc_services = SimpleNamespace(
        engine_service=object(),
        controller_service=object(),
        debug_service=object(),
        observability_service=object(),
        p2p_service=object(),
        blend_service=None,
        management=SimpleNamespace(clear=lambda: None),
        lmcache_driven_transfer=None,
        status_reporters=(),
        closeables=(),
    )

    monkeypatch.setattr(
        server_mod,
        "set_isolated_ipc",
        lambda enabled: calls.append(("set_isolated_ipc", enabled)),
    )
    monkeypatch.setattr(
        server_mod,
        "init_observability",
        lambda *_args, **_kwargs: SimpleNamespace(stop=lambda: None),
    )
    monkeypatch.setattr(server_mod, "init_gc_monitor", lambda _config: None)
    monkeypatch.setattr(
        server_mod,
        "maybe_initialize_trace_recorder",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(server_mod, "MPCacheServerContext", FakeContext)
    monkeypatch.setattr(
        server_mod,
        "_build_rpc_services",
        lambda *_args, **_kwargs: fake_rpc_services,
    )
    monkeypatch.setattr(server_mod, "InitializeMPUsageContext", lambda *_args: None)
    monkeypatch.setattr(server_mod, "InitializeMPContinuousUsage", lambda *_args: None)
    monkeypatch.setattr(server_mod, "InitializeL2ConnectorUsage", lambda *_args: None)
    monkeypatch.setattr(server_mod, "InitializeL1Usage", lambda *_args: None)
    monkeypatch.setattr(server_mod, "MultiprocessGrpcServer", FakeGrpcServer)
    monkeypatch.setattr(server_mod, "torch_dev", SimpleNamespace(init=lambda: None))

    result = server_mod.run_cache_server(
        mp_config=MPServerConfig(host="localhost", port=0, isolated_ipc=True),
        storage_manager_config=_storage_config(),
        obs_config=ObservabilityConfig(enabled=False),
        return_engine=True,
        start_prometheus_http_server=False,
    )

    assert result is not None
    assert calls[0] == ("set_isolated_ipc", True)
    assert calls.index(("set_isolated_ipc", True)) < calls.index(("context", None))
