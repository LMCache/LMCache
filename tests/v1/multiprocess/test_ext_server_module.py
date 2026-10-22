# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamically loaded multiprocess server modules."""

# Future
from __future__ import annotations

# Standard
from types import ModuleType
from typing import Any, ClassVar
from unittest.mock import MagicMock
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleSpec,
    build_server_module_router,
    load_server_module_classes,
    parse_server_module_specs,
    register_grpc_services,
    register_zmq_services,
    server_module_handler,
)
from lmcache.v1.multiprocess.modules import lookup as lookup_mod
from lmcache.v1.multiprocess.modules import management as management_mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
from lmcache.v1.multiprocess.request_handler import iter_request_handlers


class _FakePluginModule(DiscoverableModule):
    module_name = "fake_plugin"
    module_dependencies = (lookup_mod.LookupModule,)
    last_created: ClassVar[_FakePluginModule | None] = None
    last_dependency: ClassVar[EngineModule | None] = None

    def __init__(self, ctx, config: dict[str, Any] | None = None) -> None:
        self._ctx = ctx
        self.config = config or {}
        self.touched: list[int] = []

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        cls.last_dependency = build_ctx.require(lookup_mod.LookupModule)
        module = cls(
            build_ctx.engine_context,
            dict(build_ctx.config_for(cls)),
        )
        cls.last_created = module
        return module

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
    module_name = "fake_plugin_protocol"

    @server_module_handler("fake.echo")
    def echo(self, payload: bytes) -> bytes:
        return b"echo:" + payload

    @server_module_handler("fake.fail")
    def fail(self, payload: bytes) -> bytes:
        raise RuntimeError(payload.decode())


class _FakePluginTransportServiceModule(_FakePluginModule):
    module_name = "fake_plugin_transport"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.grpc_server: object | None = None
        self.zmq_server: object | None = None

    def register_grpc_services(self, server: object) -> None:
        self.grpc_server = server

    def register_zmq_services(self, server: object) -> None:
        self.zmq_server = server


class _FakeTransferReplacement(LMCacheDrivenTransferModule):
    last_created: ClassVar[_FakeTransferReplacement | None] = None

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        module = cls(build_ctx.engine_context)
        cls.last_created = module
        return module

    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        return None


class _FakeTransferDependent(DiscoverableModule):
    module_name = "fake_transfer_dependent"
    module_dependencies = (LMCacheDrivenTransferModule,)
    last_dependency: ClassVar[EngineModule | None] = None

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        cls.last_dependency = build_ctx.require(LMCacheDrivenTransferModule)
        return cls(build_ctx.engine_context)

    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        return None


class _FakeTransferNameCollision(_FakePluginModule):
    module_name = LMCacheDrivenTransferModule.module_name


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    module_type: type[DiscoverableModule],
    *,
    module_name: str = "fake_server_module_plugin",
) -> str:
    module = ModuleType(module_name)
    monkeypatch.setattr(module_type, "__module__", module_name)
    setattr(module, module_type.__name__, module_type)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def test_parse_server_module_specs_accepts_object_and_list() -> None:
    specs = parse_server_module_specs(
        [
            '{"module_path":"plugin.one","config":{"a":1}}',
            ('[{"module_path":"plugin.two"},{"module_path":"plugin.three"}]'),
        ]
    )

    assert [spec.module_path for spec in specs] == [
        "plugin.one",
        "plugin.two",
        "plugin.three",
    ]
    assert specs[0].config == {"a": 1}
    assert specs[1].config == {}
    assert specs[2].config == {}


def test_load_server_module_classes_returns_class_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _install_fake_module(monkeypatch, _FakePluginModule)
    configured = load_server_module_classes(
        [ExtServerModuleSpec(module_name, config={"mode": "test"})]
    )

    assert configured == [(_FakePluginModule, {"mode": "test"})]


def test_load_server_module_classes_rejects_duplicate_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _install_fake_module(monkeypatch, _FakePluginModule)
    spec = ExtServerModuleSpec(module_name)
    with pytest.raises(ValueError, match="configured more than once"):
        load_server_module_classes([spec, spec])


def test_load_server_module_classes_requires_discoverable_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "empty_server_module_plugin"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    with pytest.raises(TypeError, match="defines no concrete"):
        load_server_module_classes([ExtServerModuleSpec(module_name)])


def test_load_server_module_classes_ignores_imported_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "server_module_with_imported_dependency"
    module: Any = ModuleType(module_name)
    external_type = type(
        "ExternalModule",
        (_FakePluginModule,),
        {
            "__module__": module_name,
            "module_name": "external",
            "module_dependencies": (lookup_mod.LookupModule,),
        },
    )
    module.ExternalModule = external_type
    module.LookupModule = lookup_mod.LookupModule
    monkeypatch.setitem(sys.modules, module_name, module)

    configured = load_server_module_classes([ExtServerModuleSpec(module_name)])

    assert configured == [(external_type, {})]


def test_server_module_router_dispatches_extension_handlers() -> None:
    plugin_module = _FakePluginProtocolModule(MagicMock(name="ctx"))
    router = build_server_module_router(plugin_module.context, [plugin_module])
    assert router is not None

    registered = iter_request_handlers(router)
    assert [handler.operation for handler in registered] == [
        "server_module_affinity_call",
        "server_module_call",
    ]

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
    stub_module_resources,
) -> None:
    ctx = MagicMock(name="ctx")
    logger_info = MagicMock()
    monkeypatch.setattr(server_mod.logger, "info", logger_info)
    module_name = _install_fake_module(monkeypatch, _FakePluginModule)
    modules = server_mod._build_modules(
        ctx,
        MPServerConfig(
            worker_reap_timeout_seconds=0,
            server_modules=[ExtServerModuleSpec(module_name, config={"mode": "test"})],
        ),
        CoordinatorConfig(),
    )
    plugin_module = _FakePluginModule.last_created

    assert plugin_module is not None
    assert plugin_module.config == {"mode": "test"}
    assert plugin_module in modules
    assert plugin_module.context is ctx
    management = next(
        m for m in modules if isinstance(m, management_mod.ManagementModule)
    )
    assert management.ping(123)
    assert plugin_module.touched == [123]
    dependency = _FakePluginModule.last_dependency
    assert isinstance(dependency, lookup_mod.LookupModule)
    assert dependency in modules
    assert modules.index(dependency) < modules.index(plugin_module)
    logger_info.assert_called_once()
    log_format, module_count, module_details = logger_info.call_args.args
    log_message = log_format % (module_count, module_details)
    assert module_count == len(modules)
    assert all(
        isinstance(m, DiscoverableModule)
        and m.module_name + "(dependencies=[" in log_message
        for m in modules
    )
    assert "fake_plugin(dependencies=[lookup])" in log_message


def test_build_modules_prefers_same_named_external_subclass(
    monkeypatch: pytest.MonkeyPatch,
    stub_module_resources,
) -> None:
    ctx = MagicMock(name="ctx")
    replacement_module = _install_fake_module(
        monkeypatch,
        _FakeTransferReplacement,
        module_name="fake_transfer_replacement",
    )
    dependent_module = _install_fake_module(
        monkeypatch,
        _FakeTransferDependent,
        module_name="fake_transfer_dependent",
    )

    modules = server_mod._build_modules(
        ctx,
        MPServerConfig(
            worker_reap_timeout_seconds=0,
            server_modules=[
                ExtServerModuleSpec(dependent_module),
                ExtServerModuleSpec(replacement_module),
            ],
        ),
        CoordinatorConfig(),
    )

    replacement = _FakeTransferReplacement.last_created
    assert replacement is not None
    assert replacement in modules
    assert _FakeTransferDependent.last_dependency is replacement
    stub_module_resources[LMCacheDrivenTransferModule].assert_not_called()
    transfers = [m for m in modules if isinstance(m, LMCacheDrivenTransferModule)]
    assert transfers == [replacement]


def test_build_modules_rejects_same_named_external_non_subclass(
    monkeypatch: pytest.MonkeyPatch,
    stub_module_resources,
) -> None:
    module_name = _install_fake_module(monkeypatch, _FakeTransferNameCollision)

    with pytest.raises(ValueError, match="does not subclass"):
        server_mod._build_modules(
            MagicMock(name="ctx"),
            MPServerConfig(
                worker_reap_timeout_seconds=0,
                server_modules=[ExtServerModuleSpec(module_name)],
            ),
            CoordinatorConfig(),
        )
