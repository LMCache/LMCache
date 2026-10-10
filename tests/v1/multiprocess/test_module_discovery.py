# SPDX-License-Identifier: Apache-2.0
"""Tests for engine-module discovery and the construction contract."""

# Standard
from types import ModuleType
from unittest.mock import MagicMock
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import module_creator
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.engine_module import (
    ModuleBuildContext,
    discover_modules,
)
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleSpec,
    server_module_handler,
)
from lmcache.v1.multiprocess.modules import engine_driven_transfer as ed_mod
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as ld_mod
from lmcache.v1.multiprocess.modules import lookup as lookup_mod
from lmcache.v1.multiprocess.modules import management as management_mod
from lmcache.v1.multiprocess.modules import p2p_controller as p2p_mod
from lmcache.v1.multiprocess.modules.blend import module as blend_mod
from lmcache.v1.multiprocess.modules.experimental import qstore as qstore_mod

# Every built-in module, with the class name the creator should discover and
# the order rank it declares. Adding a module to the tree means adding it here
# too, which is the point: the creator itself never names a module.
EXPECTED_MODULES = {
    "LookupModule": 10,
    "P2PController": 20,
    "ManagementModule": 30,
    "LMCacheDrivenTransferModule": 40,
    "EngineDrivenTransferModule": 41,
    "QStoreModule": 50,
    "BlendModule": 60,
}

_BUILTIN_MODULES = [
    (lookup_mod, "LookupModule"),
    (p2p_mod, "P2PController"),
    (ld_mod, "LMCacheDrivenTransferModule"),
    (ed_mod, "EngineDrivenTransferModule"),
    (qstore_mod, "QStoreModule"),
    (blend_mod, "BlendModule"),
    (management_mod, "ManagementModule"),
]


@pytest.fixture
def stub_constructors(monkeypatch):
    """Neutralize every built-in constructor; return captured kwargs by class.

    ``__init__`` is patched on the real classes rather than replacing them:
    discovery skips abstract classes, so a synthetic stand-in that drops the
    inherited ``create()`` would silently vanish from the composition.
    """
    captured: dict = {}

    def _capture(class_name: str):
        def __init__(self, *args, **kwargs):
            captured[class_name] = kwargs

        return __init__

    for module, class_name in _BUILTIN_MODULES:
        monkeypatch.setattr(
            getattr(module, class_name), "__init__", _capture(class_name)
        )
    return captured


def _config(**kwargs) -> MPServerConfig:
    return MPServerConfig(**kwargs)


def _coord() -> CoordinatorConfig:
    return MagicMock(url="", event_reporting=False)


def _build(**kwargs):
    return module_creator.build_server_components(
        MagicMock(name="ctx"), _config(**kwargs), _coord()
    )


def _names(modules) -> list[str]:
    return [type(m).__name__ for m in modules]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovery_finds_every_builtin_module() -> None:
    """Check the scan reports exactly the built-in module set."""
    assert {cls.__name__: cls.module_order for cls in discover_modules()} == (
        EXPECTED_MODULES
    )


def test_discovery_is_ordered_by_close_rank() -> None:
    """Check discovery returns modules sorted by close order."""
    ranks = [cls.module_order for cls in discover_modules()]

    assert ranks == sorted(ranks)


def test_every_discovered_module_declares_a_name() -> None:
    """Check each module carries the key used for cross-module lookups."""
    assert all(cls.module_name for cls in discover_modules())


def test_module_names_are_unique() -> None:
    """Check two modules cannot claim the same registry key."""
    names = [cls.module_name for cls in discover_modules()]

    assert len(names) == len(set(names))


def test_creator_does_not_import_concrete_modules() -> None:
    """Check the creator is module-agnostic.

    This is the property that lets a new module be added without touching the
    creator, so it is asserted rather than assumed.
    """
    # Standard
    from pathlib import Path
    import ast

    source = Path(module_creator.__file__).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    concrete = {
        "lmcache.v1.multiprocess.modules.lookup",
        "lmcache.v1.multiprocess.modules.p2p_controller",
        "lmcache.v1.multiprocess.modules.management",
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer",
        "lmcache.v1.multiprocess.modules.engine_driven_transfer",
        "lmcache.v1.multiprocess.modules.experimental.qstore",
        "lmcache.v1.multiprocess.modules.blend",
        "lmcache.v1.multiprocess.modules.blend.module",
    }

    assert not (imported & concrete)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "lmcache_driven",
            [
                "LookupModule",
                "P2PController",
                "ManagementModule",
                "LMCacheDrivenTransferModule",
            ],
        ),
        (
            "engine_driven",
            [
                "LookupModule",
                "P2PController",
                "ManagementModule",
                "EngineDrivenTransferModule",
            ],
        ),
        (
            "auto",
            [
                "LookupModule",
                "P2PController",
                "ManagementModule",
                "LMCacheDrivenTransferModule",
                "EngineDrivenTransferModule",
            ],
        ),
    ],
)
def test_transfer_mode_gates_and_orders_modules(
    stub_constructors, mode: str, expected: list[str]
) -> None:
    """Check --supported-transfer-mode selects the right modules, in order."""
    assert _names(_build(supported_transfer_mode=mode).modules) == expected


def test_management_closes_before_the_modules_it_reaps(
    stub_constructors,
) -> None:
    """Check close() stops the reaper before transfer modules clear state.

    The reaper is owned by the management module, so if it closed second the
    transfer modules could clear their instance state while the reaper was
    still scanning it.
    """
    names = _names(_build(supported_transfer_mode="auto").modules)

    assert names.index("ManagementModule") < names.index("LMCacheDrivenTransferModule")


def test_blend_loaded_only_for_the_blend_engine(stub_constructors) -> None:
    """Check --engine-type gates the blend module."""
    assert "BlendModule" not in _names(_build().modules)
    assert "BlendModule" in _names(_build(engine_type="blend").modules)


def test_blend_rejects_engine_driven_transfer_mode(
    stub_constructors,
) -> None:
    """Check blend is rejected with no LMCache-driven module to wrap."""
    with pytest.raises(ValueError, match="blend engine requires"):
        _build(engine_type="blend", supported_transfer_mode="engine_driven")


def test_qstore_loaded_only_when_enabled(stub_constructors) -> None:
    """Check --enable gates the experimental module."""
    assert "QStoreModule" not in _names(_build().modules)
    assert "QStoreModule" in _names(_build(enable=["transfer_query"]).modules)


def test_qstore_rejects_engine_driven_transfer_mode(
    stub_constructors,
) -> None:
    """Check --enable is rejected without an LMCache-driven module."""
    with pytest.raises(ValueError, match="lmcache_driven"):
        _build(enable=["transfer_query"], supported_transfer_mode="engine_driven")


def test_unknown_enable_is_rejected(stub_constructors) -> None:
    """Check a typo in --enable fails loudly instead of being ignored."""
    with pytest.raises(ValueError, match="Unknown --enable"):
        _build(enable=["no_such_feature"])


# ---------------------------------------------------------------------------
# Plugin interaction
# ---------------------------------------------------------------------------


def _install_factory(monkeypatch, factory, name="fake_plugin") -> str:
    """Register a throwaway out-of-tree module; return its dotted path."""
    module = ModuleType(name)
    module.build_server_modules = factory
    monkeypatch.setitem(sys.modules, name, module)
    return name


def test_plugin_sees_every_builtin_but_not_management(
    stub_constructors, monkeypatch
) -> None:
    """Check factories receive the built-ins, excluding the management module.

    Management is built after plugins (it consumes their liveness targets), so
    exposing it would hand out a half-built object.
    """
    seen: list[list[str]] = []

    def factory(build_context):
        seen.append([type(m).__name__ for m in build_context.modules])
        return None

    name = _install_factory(monkeypatch, factory)
    _build(server_modules=[ExtServerModuleSpec(module_path=name)])

    assert seen == [
        [
            "LookupModule",
            "P2PController",
            "LMCacheDrivenTransferModule",
        ]
    ]


def test_plugin_liveness_target_reaches_the_reaper(
    stub_constructors, monkeypatch
) -> None:
    """Check an out-of-tree liveness target is registered with management."""
    plugin = _LivenessPlugin()

    def factory(build_context):
        return plugin

    name = _install_factory(monkeypatch, factory)
    _build(server_modules=[ExtServerModuleSpec(module_path=name)])

    targets = stub_constructors["ManagementModule"]["liveness_targets"]
    assert plugin in targets


def test_builtin_liveness_targets_reach_the_reaper(stub_constructors) -> None:
    """Check transfer modules are collected as reaper targets.

    Lookup, P2P, and management do not track instances, so they must not
    appear -- the selection is structural, not positional.
    """
    _build(supported_transfer_mode="auto")

    targets = stub_constructors["ManagementModule"]["liveness_targets"]
    names = {type(t).__name__ for t in targets}
    assert names == {
        "LMCacheDrivenTransferModule",
        "EngineDrivenTransferModule",
    }


def test_router_is_appended_after_plugin_modules(
    stub_constructors, monkeypatch
) -> None:
    """Check the envelope router closes last so handlers are torn down first."""
    plugin = _ProtocolPlugin()

    def factory(build_context):
        return plugin

    name = _install_factory(monkeypatch, factory)
    modules = _build(server_modules=[ExtServerModuleSpec(module_path=name)]).modules

    assert type(modules[-1]).__name__ == "ExtServerModuleRouter"
    assert modules[-2] is plugin


def test_require_rejects_a_name_matching_the_wrong_type(
    stub_constructors, monkeypatch
) -> None:
    """Check require() verifies the module behind a name is the expected class.

    A renamed module would otherwise be handed to a caller expecting a
    different type, failing deep inside the module instead of here.
    """
    build_ctx = ModuleBuildContext(MagicMock(name="ctx"), _config(), _coord())
    build_ctx.register(_Other())

    with pytest.raises(ValueError, match="not a LookupModule"):
        build_ctx.require("other", lookup_mod.LookupModule)


def test_require_rejects_an_unbuilt_name(stub_constructors) -> None:
    """Check require() fails loudly when a dependency is missing."""
    build_ctx = ModuleBuildContext(MagicMock(name="ctx"), _config(), _coord())

    with pytest.raises(ValueError, match="is not built yet"):
        build_ctx.require("nope", lookup_mod.LookupModule)


class _Other:
    """A discoverable module that is not a LookupModule."""

    module_name = "other"
    module_order = 99

    @property
    def context(self):
        """Return a stand-in engine context."""
        return MagicMock(name="ctx")

    def report_status(self) -> dict:
        """Return no module-specific status."""
        return {}

    def close(self) -> None:
        """Release nothing."""
        return None


class _LivenessPlugin:
    """An out-of-tree module satisfying the instance-liveness contract."""

    @property
    def context(self):
        """Return a stand-in engine context."""
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


class _ProtocolPlugin:
    """An out-of-tree module exposing a namespaced extension handler."""

    @property
    def context(self):
        """Return a stand-in engine context."""
        return MagicMock(name="ctx")

    def report_status(self) -> dict:
        """Return no plugin-specific status."""
        return {}

    def close(self) -> None:
        """Release nothing."""
        return None

    @server_module_handler("fake.echo")
    def echo(self, payload: bytes) -> bytes:
        """Echo the payload back to the caller."""
        return b"echo:" + payload
