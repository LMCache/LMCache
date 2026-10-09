# SPDX-License-Identifier: Apache-2.0
"""Tests for the MP server module composition contract."""

# Standard
from types import ModuleType
from unittest.mock import MagicMock
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import module_creator
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.ext_server_module import ExtServerModuleSpec
from lmcache.v1.multiprocess.modules.experimental import TRANSFER_QUERY


class _FakeLookup:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakeP2P:
    def __init__(self, *args, **kwargs) -> None:
        self.ctx = args[0]


class _FakeLMCacheDriven:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakeEngineDriven:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakeManagement:
    def __init__(self, ctx, **kwargs) -> None:
        self.ctx = ctx
        self.kwargs = kwargs


class _FakeBlend:
    def __init__(self, ctx, transfer_module, **kwargs) -> None:
        self.ctx = ctx


class _FakePluginLivenessTarget:
    """A plugin module that satisfies the instance-liveness contract."""

    @property
    def context(self):
        """Return the shared engine context."""
        return MagicMock(name="ctx")

    def report_status(self) -> dict:
        """Return no plugin-specific status."""
        return {}

    def close(self) -> None:
        """Release nothing."""
        return None

    def touch_instance(self, instance_id: int) -> None:
        """Accept a liveness refresh."""
        return None

    def reap_stale_instances(self, reap_timeout_s: float, grace_s: float) -> list[int]:
        """Reap nothing."""
        return []

    def tracked_instance_count(self) -> int:
        """Track no instances."""
        return 0

    def drop_instance_state(self, instance_id: int) -> None:
        """Mirror no state."""
        return None


@pytest.fixture
def stub_constructors(monkeypatch):
    """Stub constructors that need a live engine context.

    The transfer modules stay real classes because the creator isinstance-checks
    them to pick liveness targets and the lmcache-driven module.
    """
    monkeypatch.setattr(module_creator, "LookupModule", _FakeLookup)
    monkeypatch.setattr(module_creator, "P2PController", _FakeP2P)
    monkeypatch.setattr(
        module_creator, "LMCacheDrivenTransferModule", _FakeLMCacheDriven
    )
    monkeypatch.setattr(module_creator, "EngineDrivenTransferModule", _FakeEngineDriven)
    monkeypatch.setattr(module_creator, "ManagementModule", _FakeManagement)


@pytest.fixture
def stub_blend(monkeypatch):
    """Stub the blend module and its coordinator client."""
    monkeypatch.setattr(module_creator, "BlendModule", _FakeBlend)
    monkeypatch.setattr(
        module_creator, "BlendCoordinatorClient", MagicMock(name="coordinator")
    )


def _build(**config) -> list:
    return module_creator.build_modules(
        MagicMock(name="ctx"),
        MPServerConfig(**config),
        MagicMock(url="", event_reporting=False),
    )


def _types(modules: list) -> list[str]:
    return [type(module).__name__ for module in modules]


def _install_fake_factory(monkeypatch, factory) -> str:
    """Register a throwaway module in sys.modules; return its dotted path."""
    name = "fake_composition_plugin"
    module = ModuleType(name)
    module.build_server_modules = factory
    monkeypatch.setitem(sys.modules, name, module)
    return name


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "lmcache_driven",
            ["_FakeLookup", "_FakeP2P", "_FakeManagement", "_FakeLMCacheDriven"],
        ),
        (
            "engine_driven",
            ["_FakeLookup", "_FakeP2P", "_FakeManagement", "_FakeEngineDriven"],
        ),
        (
            "auto",
            [
                "_FakeLookup",
                "_FakeP2P",
                "_FakeManagement",
                "_FakeLMCacheDriven",
                "_FakeEngineDriven",
            ],
        ),
    ],
)
def test_transfer_mode_selects_modules_in_close_order(
    stub_constructors, mode: str, expected: list[str]
) -> None:
    """Check each --supported-transfer-mode value loads modules in close order."""
    assert _types(_build(supported_transfer_mode=mode)) == expected


def test_management_precedes_transfer_modules(stub_constructors) -> None:
    """Check close() stops the reaper before transfer modules clear state."""
    names = _types(_build(supported_transfer_mode="auto"))

    management_index = names.index("_FakeManagement")
    transfer_indexes = [i for i, name in enumerate(names) if "Driven" in name]
    assert management_index < min(transfer_indexes)


def test_experimental_modules_precede_blend_modules(
    stub_constructors, stub_blend
) -> None:
    """Check experimental modules close before blend, per the composition table."""
    names = _types(
        _build(
            engine_type="blend",
            supported_transfer_mode="lmcache_driven",
            enable=[TRANSFER_QUERY],
        )
    )

    assert names.index("QStoreModule") < names.index("_FakeBlend")


def test_blend_rejects_engine_driven_transfer_mode(
    stub_constructors, stub_blend
) -> None:
    """Check blend is rejected when no LMCache-driven module exists to wrap."""
    with pytest.raises(ValueError, match="blend engine requires"):
        _build(engine_type="blend", supported_transfer_mode="engine_driven")


def test_experimental_module_requires_lmcache_driven(stub_constructors) -> None:
    """Check --enable is rejected without an LMCache-driven transfer module."""
    with pytest.raises(ValueError, match="lmcache_driven"):
        _build(
            supported_transfer_mode="engine_driven",
            enable=[TRANSFER_QUERY],
        )


def test_management_is_not_exposed_to_plugin_factories(
    stub_constructors, monkeypatch
) -> None:
    """Check plugin factories see built-ins but not the management module.

    ManagementModule is constructed after plugin loading because it consumes
    plugin liveness targets, so exposing it would hand out a half-built object.
    """
    seen: list[list[str]] = []

    def factory(build_context):
        seen.append([type(m).__name__ for m in build_context.modules])
        return None

    module_name = _install_fake_factory(monkeypatch, factory)
    module_creator.build_server_components(
        MagicMock(name="ctx"),
        MPServerConfig(
            supported_transfer_mode="lmcache_driven",
            server_modules=[ExtServerModuleSpec(module_path=module_name)],
        ),
        MagicMock(url="", event_reporting=False),
    )

    assert seen == [["_FakeLookup", "_FakeP2P", "_FakeLMCacheDriven"]]


def test_plugin_liveness_target_reaches_management(
    stub_constructors, monkeypatch
) -> None:
    """Check a plugin exposing the liveness contract becomes a reaper target."""
    plugin_module = _FakePluginLivenessTarget()

    def factory(build_context):
        return plugin_module

    module_name = _install_fake_factory(monkeypatch, factory)
    components = module_creator.build_server_components(
        MagicMock(name="ctx"),
        MPServerConfig(
            supported_transfer_mode="lmcache_driven",
            server_modules=[ExtServerModuleSpec(module_path=module_name)],
        ),
        MagicMock(url="", event_reporting=False),
    )
    management = next(
        module for module in components.modules if isinstance(module, _FakeManagement)
    )

    assert plugin_module in management.kwargs["liveness_targets"]
