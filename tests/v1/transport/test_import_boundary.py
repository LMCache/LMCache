# SPDX-License-Identifier: Apache-2.0
"""Import boundary of ``lmcache.v1.transport``."""

# Standard
from collections.abc import Iterator
from pathlib import Path
import ast

_REPO_ROOT = Path(__file__).parents[3]
_FORBIDDEN_PREFIXES = (
    "asyncio",
    "concurrent",
    "grpc",
    "lmcache.integration",
    "lmcache.lmcache_frontend",
    "lmcache.v1.client",
    "lmcache.v1.distributed",
    "lmcache.v1.memory_orchestrator",
    "lmcache.v1.mp_coordinator",
    "lmcache.v1.multiprocess",
    "lmcache.v1.platform",
    "lmcache.v1.server",
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


def _imported_modules(path: Path) -> Iterator[tuple[int, str]]:
    package = ".".join(path.parent.relative_to(_REPO_ROOT).parts)
    for node in ast.walk(ast.parse(path.read_text())):
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
                func=ast.Name(id="__import__" | "import_module")
                | ast.Attribute(attr="__import__" | "import_module"),
                args=[ast.Constant(value=str() as name), *_],
            ):
                relative = name.lstrip(".")
                level = len(name) - len(relative)
                yield node.lineno, _resolve(relative, level, package)


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _FORBIDDEN_PREFIXES
    )


def test_transport_does_not_import_endpoint_implementations() -> None:
    """Static scan: covers lazy, TYPE_CHECKING, relative, parent-package and
    literal ``import_module``/``__import__`` imports, not transitive ones."""
    violations = [
        f"{path.relative_to(_REPO_ROOT)}:{lineno}:{module}"
        for path in sorted((_REPO_ROOT / "lmcache/v1/transport").rglob("*.py"))
        for lineno, module in _imported_modules(path)
        if _is_forbidden(module)
    ]

    assert violations == []
