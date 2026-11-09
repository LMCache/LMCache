# SPDX-License-Identifier: Apache-2.0
"""Import boundary of ``lmcache.v1.common``."""

# Standard
from collections.abc import Iterator
from pathlib import Path
import ast

# Third Party
import pytest

_REPO_ROOT = Path(__file__).parents[3]
_FORBIDDEN_PREFIXES = (
    "asyncio",
    "concurrent",
    "grpc",
    "lmcache.integration",
    "lmcache.lmcache_frontend",
    "lmcache.v1.client",
    "lmcache.v1.distributed",
    "lmcache.v1.memory_management",
    "lmcache.v1.memory_orchestrator",
    "lmcache.v1.mp_coordinator",
    "lmcache.v1.multiprocess",
    "lmcache.v1.platform",
    "lmcache.v1.server",
    "lmcache.v1.transport",
    "multiprocessing",
    "socket",
    "subprocess",
    "threading",
    "zmq",
)


def _resolve(name: str, level: int, package: str) -> str:
    if level == 0:
        return name
    base = package.rsplit(".", level - 1)[0]
    return f"{base}.{name}" if name else base


def _package_argument(call: ast.Call) -> str | None:
    values = call.args[1:2] + [kw.value for kw in call.keywords if kw.arg == "package"]
    match values:
        case [ast.Constant(value=str() as package)]:
            return package
    return None


def _imported_modules(source: str, package: str) -> Iterator[tuple[int, str]]:
    for node in ast.walk(ast.parse(source)):
        match node:
            case ast.Import(names=aliases):
                for alias in aliases:
                    yield node.lineno, alias.name
            case ast.ImportFrom(module=module, names=aliases, level=level):
                base = _resolve(module or "", level, package)
                yield node.lineno, base
                for alias in aliases:
                    yield node.lineno, f"{base}.{alias.name}"
            case ast.Call(
                func=ast.Name(id="__import__" | "import_module" as func)
                | ast.Attribute(attr="__import__" | "import_module" as func),
                args=[ast.Constant(value=str() as name), *_],
            ):
                relative = name.lstrip(".")
                level = len(name) - len(relative)
                anchor = _package_argument(node) if func == "import_module" else None
                yield node.lineno, _resolve(relative, level, anchor or package)


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _FORBIDDEN_PREFIXES
    )


def test_common_does_not_import_endpoint_implementations() -> None:
    """Static scan: covers lazy, TYPE_CHECKING, relative, parent-package and
    literal ``import_module``/``__import__`` imports, not transitive ones."""
    violations = [
        f"{path.relative_to(_REPO_ROOT)}:{lineno}:{module}"
        for path in sorted((_REPO_ROOT / "lmcache/v1/common").rglob("*.py"))
        for lineno, module in _imported_modules(
            path.read_text(), ".".join(path.parent.relative_to(_REPO_ROOT).parts)
        )
        if _is_forbidden(module)
    ]

    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "from lmcache.v1.memory_management import MemoryObj",
        'importlib.import_module(".server", "lmcache.v1.multiprocess")',
        'importlib.import_module(".server", package="lmcache.v1.multiprocess")',
    ],
)
def test_scan_rejects_forbidden_import(source: str) -> None:
    modules = [name for _, name in _imported_modules(source, "lmcache.v1.common")]
    assert any(map(_is_forbidden, modules)), modules


def test_scan_accepts_shared_import() -> None:
    source = 'import_module(".common", package="lmcache.v1")'
    modules = [name for _, name in _imported_modules(source, "lmcache.v1.common")]
    assert modules == ["lmcache.v1.common"]
    assert not any(map(_is_forbidden, modules))
