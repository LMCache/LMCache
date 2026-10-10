# SPDX-License-Identifier: Apache-2.0
"""Tests for engine-module discovery and the construction contract."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import ModuleBuildContext, discover_modules
from lmcache.v1.multiprocess.modules import engine_driven_transfer as ed_mod
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as ld_mod
from lmcache.v1.multiprocess.modules import lookup as lookup_mod
from lmcache.v1.multiprocess.modules import management as management_mod
from lmcache.v1.multiprocess.modules import p2p_controller as p2p_mod
from lmcache.v1.multiprocess.modules.blend import module as blend_mod
from lmcache.v1.multiprocess.modules.experimental import qstore as qstore_mod

# Every built-in module the builder discovers. Adding a module means adding
# it here too; the builder itself never names one.
EXPECTED_MODULE_NAMES = {
    "lookup",
    "p2p_controller",
    "management",
    "lmcache_driven_transfer",
    "engine_driven_transfer",
    "qstore",
    "blend",
}


@pytest.fixture
def stub_constructors(monkeypatch):
    """Patch each built-in's __init__; replacing the class would drop create()."""
    captured: dict = {}

    def _capture(class_name: str):
        def __init__(self, *args, **kwargs):
            captured[class_name] = kwargs

        return __init__

    for module, class_name in [
        (lookup_mod, "LookupModule"),
        (p2p_mod, "P2PController"),
        (ld_mod, "LMCacheDrivenTransferModule"),
        (ed_mod, "EngineDrivenTransferModule"),
        (qstore_mod, "QStoreModule"),
        (blend_mod, "BlendModule"),
        (management_mod, "ManagementModule"),
    ]:
        monkeypatch.setattr(
            getattr(module, class_name), "__init__", _capture(class_name)
        )
    return captured


def _build(**kwargs):
    """Assemble modules for the given MPServerConfig kwargs."""
    return server_mod._build_modules(
        MagicMock(name="ctx"),
        MPServerConfig(**kwargs),
        MagicMock(url="", event_reporting=False),
    )


def _names(modules) -> list[str]:
    return [type(m).__name__ for m in modules]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovery_finds_every_builtin_module() -> None:
    """Check the scan reports exactly the built-in set with valid deps."""
    found = {cls.module_name: cls for cls in discover_modules()}

    assert set(found) == EXPECTED_MODULE_NAMES
    # A module must not depend on itself, and every declared dependency must
    # name a real discoverable module (order_modules would also reject that).
    for cls in found.values():
        assert cls.module_name not in cls.module_dependencies
    # Modules that wrap / require the LMCache-driven transfer path declare it
    # so the builder constructs that module first.
    assert found["blend"].module_dependencies == ["lmcache_driven_transfer"]
    assert found["qstore"].module_dependencies == ["lmcache_driven_transfer"]


# ---------------------------------------------------------------------------
# Gating and ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected_present", "expected_absent"),
    [
        (
            "lmcache_driven",
            {
                "LookupModule",
                "P2PController",
                "ManagementModule",
                "LMCacheDrivenTransferModule",
            },
            {"EngineDrivenTransferModule", "QStoreModule", "BlendModule"},
        ),
        (
            "engine_driven",
            {
                "LookupModule",
                "P2PController",
                "ManagementModule",
                "EngineDrivenTransferModule",
            },
            {"LMCacheDrivenTransferModule", "QStoreModule", "BlendModule"},
        ),
        (
            "auto",
            {
                "LookupModule",
                "P2PController",
                "ManagementModule",
                "LMCacheDrivenTransferModule",
                "EngineDrivenTransferModule",
            },
            {"QStoreModule", "BlendModule"},
        ),
    ],
)
def test_transfer_mode_gates_modules(
    stub_constructors,
    mode: str,
    expected_present: set[str],
    expected_absent: set[str],
) -> None:
    """Check --supported-transfer-mode selects which modules are composed.

    The ordering itself is exercised by the close-order invariant tests; here
    we only assert which modules the mode admits.
    """
    names = set(_names(_build(supported_transfer_mode=mode)))
    assert expected_present <= names
    assert expected_absent & names == set()


def test_management_closes_before_the_modules_it_reaps(
    stub_constructors,
) -> None:
    """Check close() stops the reaper before transfer modules clear state."""
    names = _names(_build(supported_transfer_mode="auto"))
    assert names.index("ManagementModule") < names.index("LMCacheDrivenTransferModule")


def test_blend_gating(stub_constructors) -> None:
    """Check --engine-type gates blend, and blend needs an LMCache-driven path."""
    assert "BlendModule" not in _names(_build())
    assert "BlendModule" in _names(_build(engine_type="blend"))
    with pytest.raises(ValueError, match="blend engine requires"):
        _build(engine_type="blend", supported_transfer_mode="engine_driven")


def test_qstore_gating(stub_constructors) -> None:
    """Check --enable gates qstore, rejects bad modes, and rejects unknowns."""
    assert "QStoreModule" not in _names(_build())
    assert "QStoreModule" in _names(_build(enable=["transfer_query"]))
    with pytest.raises(ValueError, match="lmcache_driven"):
        _build(enable=["transfer_query"], supported_transfer_mode="engine_driven")
    with pytest.raises(ValueError, match="Unknown --enable"):
        _build(enable=["no_such_feature"])


# ---------------------------------------------------------------------------
# ModuleBuildContext.require
# ---------------------------------------------------------------------------


def test_require_errors() -> None:
    """Check require() fails loudly on a missing name or the wrong type."""
    build_ctx = ModuleBuildContext(
        MagicMock(name="ctx"),
        MPServerConfig(),
        MagicMock(url="", event_reporting=False),
    )
    build_ctx.register(_Other())

    with pytest.raises(ValueError, match="not a LookupModule"):
        build_ctx.require("other", lookup_mod.LookupModule)
    with pytest.raises(ValueError, match="is not built yet"):
        build_ctx.require("nope", lookup_mod.LookupModule)


class _Other:
    """A registered module that is not a LookupModule."""

    module_name = "other"

    @property
    def context(self):
        return MagicMock(name="ctx")

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        return None
