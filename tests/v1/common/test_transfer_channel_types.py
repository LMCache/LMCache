# SPDX-License-Identifier: Apache-2.0
"""Compatibility and boundary tests for transfer-channel data types."""

# Standard
from collections.abc import Iterator
from pathlib import Path
import ast
import importlib

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.common.transfer_channel_types import (
    TransferChannelAddress,
    TransferChannelReadResult,
)

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


@pytest.mark.parametrize(
    "legacy_module",
    [
        "lmcache.v1.distributed.transfer_channel",
        "lmcache.v1.distributed.transfer_channel.api",
    ],
)
def test_legacy_paths_reexport_canonical_types(legacy_module: str) -> None:
    module = importlib.import_module(legacy_module)
    assert module.TransferChannelAddress is TransferChannelAddress
    assert module.TransferChannelReadResult is TransferChannelReadResult


def test_msgpack_wire_bytes_are_frozen() -> None:
    addresses = [TransferChannelAddress(offset=8, size=16)]
    result = TransferChannelReadResult(finished=True, succeeded_mask=[True, False])

    assert msgspec.msgpack.encode(addresses) == bytes.fromhex(
        "9182a66f666673657408a473697a6510"
    )
    assert msgspec.msgpack.encode(result) == bytes.fromhex(
        "82a866696e6973686564c3ae7375636365656465645f6d61736b92c3c2"
    )


def test_common_does_not_import_endpoint_implementations() -> None:
    """Static scan: covers lazy, TYPE_CHECKING, relative, parent-package and
    literal ``import_module``/``__import__`` imports, not transitive ones."""
    violations = [
        f"{path.relative_to(_REPO_ROOT)}:{lineno}:{module}"
        for path in sorted((_REPO_ROOT / "lmcache/v1/common").rglob("*.py"))
        for lineno, module in _imported_modules(path)
        if _is_forbidden(module)
    ]

    assert violations == []
