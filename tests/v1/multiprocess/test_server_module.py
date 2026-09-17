# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamically loaded multiprocess server modules."""

# Standard
from types import ModuleType
from typing import Any, cast
from unittest.mock import MagicMock
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache.v1.multiprocess.server_module import (
    ServerModuleBuildContext,
    ServerModuleComponents,
    ServerModuleSpec,
    build_server_module_router,
    load_server_module_components,
    load_server_modules,
    parse_server_module_specs,
    register_grpc_services,
    register_zmq_services,
    server_module_handler,
)


class _FakeLMCacheDriven:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakeEngineDriven:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakePluginModule:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self.touched: list[int] = []

    @property
    def context(self):
        return self._ctx

    def report_status(self) -> dict:
        return {"fake_plugin": {"is_healthy": True}}

    def close(self) -> None:
        return None

    def touch_instance(self, instance_id: int) -> None:
        self.touched.append(instance_id)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        return []

    def tracked_instance_count(self) -> int:
        return len(self.touched)

    def drop_instance_state(self, instance_id: int) -> None:
        return None


class _FakePluginProtocolModule(_FakePluginModule):
    @server_module_handler("fake.echo")
    def echo(self, payload: bytes) -> bytes:
        return b"echo:" + payload

    @server_module_handler("fake.fail")
    def fail(self, payload: bytes) -> bytes:
        raise RuntimeError(payload.decode())


class _FakePluginTransportServiceModule(_FakePluginModule):
    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.grpc_server: object | None = None
        self.zmq_server: object | None = None

    def register_grpc_services(self, server: object) -> None:
        self.grpc_server = server

    def register_zmq_services(self, server: object) -> None:
        self.zmq_server = server


def _install_fake_factory(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    *,
    module_name: str = "fake_server_module_plugin",
    factory_name: str = "build_server_modules",
) -> str:
    module = ModuleType(module_name)
    setattr(module, factory_name, factory)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def test_parse_server_module_specs_accepts_object_and_list() -> None:
    specs = parse_server_module_specs(
        [
            '{"module_path":"plugin.one","config":{"a":1}}',
            (
                '[{"module_path":"plugin.two",'
                '"factory_name":"build_two"},'
                '{"module_path":"plugin.three"}]'
            ),
        ]
    )

    assert [spec.module_path for spec in specs] == [
        "plugin.one",
        "plugin.two",
        "plugin.three",
    ]
    assert specs[0].config == {"a": 1}
    assert specs[1].factory_name == "build_two"
    assert specs[2].factory_name == "build_server_modules"


def test_load_server_modules_passes_context_and_accumulates_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = MagicMock(name="ctx")
    built_module = MagicMock(name="built_module")
    first_plugin = _FakePluginModule(ctx)
    second_plugin = _FakePluginModule(ctx)
    seen_contexts: list[ServerModuleBuildContext] = []

    def factory(build_context: ServerModuleBuildContext):
        seen_contexts.append(build_context)
        if len(seen_contexts) == 1:
            return first_plugin
        return [second_plugin]

    module_name = _install_fake_factory(monkeypatch, factory)
    specs = [
        ServerModuleSpec(module_name, config={"name": "first"}),
        ServerModuleSpec(module_name, config={"name": "second"}),
    ]

    modules = load_server_modules(
        specs,
        server_context=ctx,
        mp_config=MPServerConfig(),
        coordinator_config=MagicMock(url=""),
        built_modules=[built_module],
    )

    assert modules == [first_plugin, second_plugin]
    assert seen_contexts[0].server_context is ctx
    assert seen_contexts[0].config == {"name": "first"}
    assert seen_contexts[0].modules == (built_module,)
    assert seen_contexts[1].config == {"name": "second"}
    assert seen_contexts[1].modules == (built_module, first_plugin)


def test_load_server_modules_rejects_non_module_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _install_fake_factory(monkeypatch, lambda _: object())

    with pytest.raises(TypeError, match="must return an EngineModule"):
        load_server_modules(
            [ServerModuleSpec(module_name)],
            server_context=MagicMock(name="ctx"),
            mp_config=MPServerConfig(),
            coordinator_config=MagicMock(url=""),
            built_modules=[],
        )


def test_load_server_module_components_accepts_service_only_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grpc_registrar = MagicMock(name="grpc_registrar")
    zmq_registrar = MagicMock(name="zmq_registrar")

    def factory(build_context: ServerModuleBuildContext):
        assert build_context.config == {"mode": "service-only"}
        return ServerModuleComponents(
            grpc_service_registrars=[grpc_registrar],
            zmq_service_registrars=[zmq_registrar],
        )

    module_name = _install_fake_factory(monkeypatch, factory)
    components = load_server_module_components(
        [ServerModuleSpec(module_name, config={"mode": "service-only"})],
        server_context=MagicMock(name="ctx"),
        mp_config=MPServerConfig(),
        coordinator_config=MagicMock(url=""),
        built_modules=[],
    )

    assert components.modules == ()
    assert components.grpc_service_registrars == (grpc_registrar,)
    assert components.zmq_service_registrars == (zmq_registrar,)


def test_load_server_module_components_rejects_non_callable_service_registrar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _install_fake_factory(
        monkeypatch,
        lambda _: ServerModuleComponents(
            grpc_service_registrars=[cast(Any, "not-callable")]
        ),
    )

    with pytest.raises(TypeError, match="grpc_service_registrars"):
        load_server_module_components(
            [ServerModuleSpec(module_name)],
            server_context=MagicMock(name="ctx"),
            mp_config=MPServerConfig(),
            coordinator_config=MagicMock(url=""),
            built_modules=[],
        )


def test_server_module_router_dispatches_extension_handlers() -> None:
    plugin_module = _FakePluginProtocolModule(MagicMock(name="ctx"))
    router = build_server_module_router(plugin_module.context, [plugin_module])
    assert router is not None

    registered = iter_request_handlers(router)
    assert len(registered) == 1
    assert registered[0].options.request_type is RequestType.SERVER_MODULE_CALL

    ok = router.server_module_call(
        ServerModuleCallRequest(method="fake.echo", payload=b"hello")
    )
    missing = router.server_module_call(
        ServerModuleCallRequest(method="fake.missing", payload=b"hello")
    )
    failed = router.server_module_call(
        ServerModuleCallRequest(method="fake.fail", payload=b"boom")
    )

    assert ok.success
    assert ok.payload == b"echo:hello"
    assert ok.error == ""
    assert not missing.success
    assert missing.payload == b""
    assert "fake.missing" in missing.error
    assert not failed.success
    assert failed.error == "boom"


def test_server_module_router_rejects_duplicate_extension_methods() -> None:
    ctx = MagicMock(name="ctx")

    with pytest.raises(ValueError, match="duplicate server module method"):
        build_server_module_router(
            ctx,
            [
                _FakePluginProtocolModule(ctx),
                _FakePluginProtocolModule(ctx),
            ],
        )


def test_build_server_module_router_returns_none_without_handlers() -> None:
    assert (
        build_server_module_router(
            MagicMock(name="ctx"),
            [_FakePluginModule(MagicMock(name="plugin_ctx"))],
        )
        is None
    )


def test_transport_service_registrars_are_called() -> None:
    module = _FakePluginTransportServiceModule(MagicMock(name="ctx"))
    grpc_server = object()
    zmq_server = object()

    register_grpc_services([module], grpc_server)
    register_zmq_services([module], zmq_server)

    assert module.grpc_server is grpc_server
    assert module.zmq_server is zmq_server


def test_transport_service_registrars_must_be_callable() -> None:
    class BadModule:
        register_grpc_services = "not-callable"

    with pytest.raises(TypeError, match="register_grpc_services must be callable"):
        register_grpc_services([BadModule()], object())


def test_build_modules_loads_plugin_and_registers_liveness_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = MagicMock(name="ctx")
    plugin_module = _FakePluginModule(ctx)

    def factory(build_context: ServerModuleBuildContext):
        assert build_context.server_context is ctx
        return plugin_module

    module_name = _install_fake_factory(monkeypatch, factory)
    monkeypatch.setattr(server_mod, "LookupModule", lambda ctx: MagicMock())
    monkeypatch.setattr(server_mod, "P2PController", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(server_mod, "LMCacheDrivenTransferModule", _FakeLMCacheDriven)
    monkeypatch.setattr(server_mod, "EngineDrivenTransferModule", _FakeEngineDriven)
    management = MagicMock(name="ManagementModule")
    monkeypatch.setattr(server_mod, "ManagementModule", management)

    modules = server_mod._build_modules(
        ctx,
        MPServerConfig(server_modules=[ServerModuleSpec(module_name)]),
        MagicMock(url=""),
    )

    assert modules[-1] is plugin_module
    assert plugin_module in management.call_args.kwargs["liveness_targets"]


def test_build_modules_adds_server_module_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = MagicMock(name="ctx")
    plugin_module = _FakePluginProtocolModule(ctx)

    def factory(build_context: ServerModuleBuildContext):
        assert build_context.server_context is ctx
        return plugin_module

    module_name = _install_fake_factory(monkeypatch, factory)
    monkeypatch.setattr(server_mod, "LookupModule", lambda ctx: MagicMock())
    monkeypatch.setattr(server_mod, "P2PController", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(server_mod, "LMCacheDrivenTransferModule", _FakeLMCacheDriven)
    monkeypatch.setattr(server_mod, "EngineDrivenTransferModule", _FakeEngineDriven)
    monkeypatch.setattr(server_mod, "ManagementModule", MagicMock())

    modules = server_mod._build_modules(
        ctx,
        MPServerConfig(server_modules=[ServerModuleSpec(module_name)]),
        MagicMock(url=""),
    )

    assert modules[-2] is plugin_module
    assert iter_request_handlers(modules[-1])[0].options.request_type is (
        RequestType.SERVER_MODULE_CALL
    )


def test_build_server_components_collects_transport_service_registrars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = MagicMock(name="ctx")
    grpc_registrar = MagicMock(name="grpc_registrar")
    zmq_registrar = MagicMock(name="zmq_registrar")

    def factory(build_context: ServerModuleBuildContext):
        assert build_context.server_context is ctx
        return ServerModuleComponents(
            grpc_service_registrars=[grpc_registrar],
            zmq_service_registrars=[zmq_registrar],
        )

    module_name = _install_fake_factory(monkeypatch, factory)
    monkeypatch.setattr(server_mod, "LookupModule", lambda ctx: MagicMock())
    monkeypatch.setattr(server_mod, "P2PController", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(server_mod, "LMCacheDrivenTransferModule", _FakeLMCacheDriven)
    monkeypatch.setattr(server_mod, "EngineDrivenTransferModule", _FakeEngineDriven)
    monkeypatch.setattr(server_mod, "ManagementModule", MagicMock())

    components = server_mod._build_server_components(
        ctx,
        MPServerConfig(server_modules=[ServerModuleSpec(module_name)]),
        MagicMock(url=""),
    )

    assert grpc_registrar in components.grpc_service_registrars
    assert zmq_registrar in components.zmq_service_registrars
