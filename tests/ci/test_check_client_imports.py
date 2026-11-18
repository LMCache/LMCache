# SPDX-License-Identifier: Apache-2.0
"""The lmcache-client import check, on a small hand-made wheel and checkout.

The checkout has a client module and a server module; the wheel ships only the
client one, whose source each case replaces with one kind of import.
"""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util
import zipfile

# Third Party
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/check_client_imports.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_lmcache_check_client_imports", _SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the import check from {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load_module().check

UNSATISFIED = [
    "from lmcache.v1 import server",
    "import lmcache.v1.server",
    "from . import server",
    "from .server import Server",
    "def f():\n    from lmcache.v1.server import Server",
]
SATISFIED = [
    "from lmcache.v1 import client, VERSION",
    "import lmcache.v1",
    "from lmcache.inprocess.integration import adapter",
    "from typing import TYPE_CHECKING\n"
    "if TYPE_CHECKING:\n"
    "    from lmcache.v1.server import Server",
    "try:\n    import lmcache.v1.server\nexcept ImportError:\n    pass",
]


def _run(tmp_path: Path, source: str) -> list[str]:
    repo = tmp_path / "repo"
    for module in ("__init__", "v1/__init__", "v1/client", "v1/server"):
        (repo / "lmcache" / f"{module}.py").parent.mkdir(parents=True, exist_ok=True)
        (repo / "lmcache" / f"{module}.py").write_text("")
    wheel = tmp_path / "c.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lmcache/__init__.py", "")
        archive.writestr("lmcache/v1/__init__.py", "VERSION = 1\n")
        archive.writestr("lmcache/v1/client.py", source)
    return check(wheel, repo)


@pytest.mark.parametrize("source", UNSATISFIED)
def test_a_module_the_wheel_leaves_out_fails(tmp_path: Path, source: str) -> None:
    errors = _run(tmp_path, source)
    assert errors and all("lmcache.v1.server" in e for e in errors), errors


@pytest.mark.parametrize("source", SATISFIED)
def test_shipped_external_typing_and_guarded_imports_pass(
    tmp_path: Path, source: str
) -> None:
    assert _run(tmp_path, source) == []
