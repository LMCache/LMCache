# SPDX-License-Identifier: Apache-2.0
"""Module dependency and build-context contracts using synthetic modules."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
    order_modules,
)


class _Module(DiscoverableModule):
    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        return None


class _Dependency(_Module):
    module_name = "dependency"


class _Dependent(_Module):
    module_name = "dependent"
    module_dependencies = (_Dependency,)


class _Replacement(_Dependency):
    pass


def test_dependencies_and_lookup_survive_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_Dependency, "module_name", "renamed_dependency")
    assert order_modules([_Dependent, _Dependency]) == [_Dependency, _Dependent]
    build_ctx = ModuleBuildContext(MagicMock(), MPServerConfig(), CoordinatorConfig())
    with pytest.raises(ValueError, match="is not built yet"):
        build_ctx.require(_Dependency)
    module = MagicMock(spec=EngineModule)
    build_ctx.register(_Dependency, module)
    assert build_ctx.require(_Dependency) is module


def test_dependencies_and_lookup_resolve_same_named_subclass() -> None:
    assert order_modules([_Dependent, _Replacement]) == [
        _Replacement,
        _Dependent,
    ]
    build_ctx = ModuleBuildContext(MagicMock(), MPServerConfig(), CoordinatorConfig())
    module = MagicMock(spec=EngineModule)
    build_ctx.register(_Replacement, module)
    assert build_ctx.require(_Dependency) is module
    assert build_ctx.require(_Replacement) is module


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unnamed", "does not define module_name"),
        ("duplicate", "Duplicate module_name"),
        ("unknown", "not a discoverable module"),
        ("cycle", "Module dependency cycle"),
    ],
)
def test_dependency_graph_rejects_invalid_modules(
    case: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if case == "unnamed":
        monkeypatch.setattr(_Dependency, "module_name", "")
    elif case == "duplicate":
        monkeypatch.setattr(_Dependent, "module_name", _Dependency.module_name)
    elif case == "unknown":
        monkeypatch.setattr(_Dependent, "module_dependencies", (_Module,))
    else:
        monkeypatch.setattr(_Dependency, "module_dependencies", (_Dependent,))
    with pytest.raises(ValueError, match=message):
        order_modules([_Dependency, _Dependent])
