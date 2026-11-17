# SPDX-License-Identifier: Apache-2.0
"""Exercise module discovery against an isolated package, not built-in modules."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.engine_module import discover_modules

_MODULE_SOURCE = """
from lmcache.v1.multiprocess.engine_module import DiscoverableModule

class Plugin(DiscoverableModule):
    module_name = __name__
    @classmethod
    def create(cls, build_ctx):
        return None

class AbstractPlugin(DiscoverableModule):
    pass
"""


@pytest.fixture
def module_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    package = ModuleType("_discovery_modules")
    package.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, "lmcache.v1.multiprocess.modules", package)
    yield tmp_path
    for name in list(sys.modules):
        if name.startswith(package.__name__ + "."):
            sys.modules.pop(name)


def test_discovery_recurses_filters_and_deduplicates(module_package: Path) -> None:
    (module_package / "top.py").write_text(_MODULE_SOURCE)
    nested = module_package / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("from ..top import Plugin\n")
    (nested / "leaf.py").write_text(_MODULE_SOURCE)
    (module_package / "_private.py").write_text("raise RuntimeError('do not import')\n")

    found = discover_modules()

    assert sorted(cls.__module__ for cls in found) == [
        "_discovery_modules.nested.leaf",
        "_discovery_modules.top",
    ]
    assert all(cls.__name__ == "Plugin" for cls in found)


def test_discovery_propagates_import_failure(module_package: Path) -> None:
    (module_package / "broken.py").write_text("raise RuntimeError('broken module')\n")
    with pytest.raises(ImportError, match="broken.*broken module"):
        discover_modules()
