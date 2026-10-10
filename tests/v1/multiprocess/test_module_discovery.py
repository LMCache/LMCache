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

# Every built-in module and its close-order rank. Adding a module means adding
# it here too; the builder itself never names one.
EXPECTED_MODULES = {
    "LookupModule": 10,
    "P2PController": 20,
    "ManagementModule": 30,
    "LMCacheDrivenTransferModule": 40,
    "EngineDrivenTransferModule": 41,
    "QStoreModule": 50,
    "BlendModule": 60,
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


def test_discovery_finds_every_builtin_module_in_order() -> None:
    """Check the scan reports exactly the built-in set, in close order."""
    found = [(cls.__name__, cls.module_order) for cls in discover_modules()]

    assert dict(found) == EXPECTED_MODULES
    assert found == sorted(found, key=lambda item: item[1])
    names = [cls.module_name for cls in discover_modules()]
    assert all(names) and len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Gating and ordering
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
    """Check --supported-transfer-mode selects modules, in close order."""
    assert _names(_build(supported_transfer_mode=mode)) == expected


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
    module_order = 99

    @property
    def context(self):
        return MagicMock(name="ctx")

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        return None
