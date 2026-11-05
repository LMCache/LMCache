# SPDX-License-Identifier: Apache-2.0
"""Compatibility and boundary tests for server-module wire contracts."""

# Standard
from pathlib import Path
import ast

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest as LegacyServerModuleCallRequest,
)
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallResponse as LegacyServerModuleCallResponse,
)
from lmcache.v1.transport.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)


def test_legacy_path_reexports_canonical_contracts() -> None:
    assert LegacyServerModuleCallRequest is ServerModuleCallRequest
    assert LegacyServerModuleCallResponse is ServerModuleCallResponse


def test_server_module_msgpack_wire_bytes_are_frozen() -> None:
    request = ServerModuleCallRequest(method="fake.echo", payload=b"hello")
    response = ServerModuleCallResponse(success=False, payload=b"", error="boom")

    assert msgspec.msgpack.encode(request) == bytes.fromhex(
        "82a66d6574686f64a966616b652e6563686fa77061796c6f6164c40568656c6c6f"
    )
    assert msgspec.msgpack.encode(response) == bytes.fromhex(
        "83a773756363657373c2a77061796c6f6164c400a56572726f72a4626f6f6d"
    )


def test_shared_transport_does_not_import_endpoint_implementations() -> None:
    transport_root = Path(__file__).parents[3] / "lmcache/v1/transport"
    forbidden_prefixes = (
        "lmcache.v1.client",
        "lmcache.v1.multiprocess",
        "lmcache.v1.server",
    )
    violations: list[str] = []

    for path in transport_root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            else:
                continue
            for module in modules:
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{path.name}:{node.lineno}:{module}")

    assert violations == []
