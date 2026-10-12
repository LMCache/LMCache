# SPDX-License-Identifier: Apache-2.0
"""Tests for engine-module discovery, composition, and lifecycle."""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock
import socket

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.blend_client import BlendCoordinatorClient
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
    discover_modules,
    order_modules,
)
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreModule
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)


class _TestModule(DiscoverableModule):
    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        return None


def _build(**config: Any) -> list[EngineModule]:
    return server_mod._build_modules(
        MagicMock(name="ctx"),
        MPServerConfig(worker_reap_timeout_seconds=0, **config),
        CoordinatorConfig(),
    )


def test_discovery_finds_every_builtin_module() -> None:
    found = discover_modules()
    assert set(found) == {
        LookupModule,
        P2PController,
        ManagementModule,
        LMCacheDrivenTransferModule,
        EngineDrivenTransferModule,
        KVEventModule,
        QStoreModule,
        BlendModule,
    }
    assert len(found) == len(set(found))
    assert all(dep in found for cls in found for dep in cls.module_dependencies)


def test_dependency_resolution_survives_module_rename(
    monkeypatch: pytest.MonkeyPatch, stub_module_resources
) -> None:
    monkeypatch.setattr(LMCacheDrivenTransferModule, "module_name", "renamed_transfer")
    modules = _build(engine_type="blend")
    dependency = next(m for m in modules if isinstance(m, LMCacheDrivenTransferModule))
    assert stub_module_resources[BlendModule].call_args.args[1] is dependency


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("lmcache_driven", {LMCacheDrivenTransferModule}),
        ("engine_driven", {EngineDrivenTransferModule}),
        ("auto", {LMCacheDrivenTransferModule, EngineDrivenTransferModule}),
    ],
)
def test_transfer_mode_gates_modules(mode, expected, stub_module_resources) -> None:
    modules = _build(supported_transfer_mode=mode)
    assert {
        type(m)
        for m in modules
        if isinstance(m, (LMCacheDrivenTransferModule, EngineDrivenTransferModule))
    } == expected


def test_modules_close_before_dependencies_and_shared_context(
    monkeypatch: pytest.MonkeyPatch, stub_module_resources
) -> None:
    modules = _build(
        engine_type="blend", supported_transfer_mode="auto", enable=["transfer_query"]
    )
    closed: list[object] = []
    for module in modules:
        module_type = type(module)
        monkeypatch.setattr(module, "close", lambda cls=module_type: closed.append(cls))
    ctx = MagicMock()
    ctx.close.side_effect = lambda: closed.append(ctx)

    server_mod.MPCacheServer(ctx, modules).close()

    assert closed[0] is ManagementModule
    assert closed[-1] is ctx
    for module in modules:
        assert isinstance(module, DiscoverableModule)
        for dependency in module.module_dependencies:
            assert closed.index(type(module)) < closed.index(dependency)


def test_blend_gating(stub_module_resources) -> None:
    assert not any(isinstance(m, BlendModule) for m in _build())
    assert any(isinstance(m, BlendModule) for m in _build(engine_type="blend"))
    with pytest.raises(ValueError, match="blend engine requires"):
        _build(engine_type="blend", supported_transfer_mode="engine_driven")


@pytest.mark.parametrize("enabled", [False, True])
def test_blend_factory_forwards_configuration(
    enabled: bool, monkeypatch: pytest.MonkeyPatch, stub_module_resources
) -> None:
    coordinator = MagicMock()
    factory = MagicMock(return_value=coordinator)
    monkeypatch.setattr(BlendCoordinatorClient, "maybe_create", factory)
    config = MPServerConfig(
        engine_type="blend",
        worker_reap_timeout_seconds=0,
        enable_segmented_prefix=enabled,
        enable_dedup_content=enabled,
        enable_blend_reorder=enabled,
    )
    coordinator_config = CoordinatorConfig(
        url="http://coordinator",
        event_reporting=enabled,
        blend_timeout=2.5,
        blend_match_concurrency=3,
    )
    modules = server_mod._build_modules(MagicMock(), config, coordinator_config)
    transfer = next(m for m in modules if isinstance(m, LMCacheDrivenTransferModule))
    args, kwargs = stub_module_resources[BlendModule].call_args
    assert args[1] is transfer
    assert kwargs == {
        "coordinator": coordinator,
        "enable_segmented_prefix": enabled,
        "enable_dedup_content": enabled,
        "enable_reorder": enabled,
    }
    factory.assert_called_once_with(
        coordinator_config.url if enabled else "", timeout=2.5, match_concurrency=3
    )


@pytest.mark.parametrize("transport", REQUEST_TRANSPORTS)
def test_blend_store_overrides_transfer_handler(
    transport: RequestTransport,
    monkeypatch: pytest.MonkeyPatch,
    stub_module_resources,
) -> None:
    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def store(
        self: object,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        return type(self).__name__.encode(), True

    monkeypatch.setattr(BlendModule, "store", store)
    monkeypatch.setattr(LMCacheDrivenTransferModule, "store", store)
    modules = _build(engine_type="blend")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = request_server_url(transport, port)
    server = create_request_server(modules, request_server_config(transport, url))
    key = IPCCacheServerKey.from_token_ids("model", 1, 0, [1], end=1)
    try:
        server.start()
        client = RequestClientFactory.create(url)
        try:
            assert client.store(key, 1, [[0]], b"").result(timeout=5) == (
                b"BlendModule",
                True,
            )
        finally:
            client.close()
    finally:
        server.close()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unnamed", "does not define module_name"),
        ("duplicate", "Duplicate module_name"),
        ("unknown", "not a discoverable module"),
        ("cycle", "Module dependency cycle"),
    ],
)
def test_dependency_graph_rejects_invalid_modules(case: str, message: str) -> None:
    first = cast(
        type[DiscoverableModule],
        type("First", (_TestModule,), {"module_name": "first"}),
    )
    second = cast(
        type[DiscoverableModule],
        type("Second", (_TestModule,), {"module_name": "second"}),
    )
    if case == "unnamed":
        first.module_name = ""
    elif case == "duplicate":
        second.module_name = first.module_name
    elif case == "unknown":
        first.module_dependencies = (_TestModule,)
    else:
        first.module_dependencies = (second,)
        second.module_dependencies = (first,)

    with pytest.raises(ValueError, match=message):
        order_modules([first, second])


def test_require_rejects_unbuilt_dependency() -> None:
    build_ctx = ModuleBuildContext(MagicMock(), MPServerConfig(), CoordinatorConfig())
    with pytest.raises(ValueError, match="is not built yet"):
        build_ctx.require(QStoreModule)
