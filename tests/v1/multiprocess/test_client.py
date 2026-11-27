# SPDX-License-Identifier: Apache-2.0
# Standard
from importlib.util import resolve_name
import ast

# First Party
from lmcache.v1.client.transport.grpc_impl.client import (
    GrpcMultiprocessClient,
)
from lmcache.v1.transport.grpc.descriptors import (
    client_method_name,
    iter_methods,
)
from lmcache.v1.transport.request_client import RequestClient
from lmcache.v1.transport.rpc import get_rpc_specs
from tests.import_utils import repo_path


def test_all_rpc_contracts_have_explicit_named_methods() -> None:
    contract_names = {
        name for name, value in RequestClient.__dict__.items() if callable(value)
    }
    expected_names = set(get_rpc_specs())

    assert expected_names <= contract_names
    grpc_client = GrpcMultiprocessClient(  # type: ignore[abstract]
        "grpc://127.0.0.1:1"
    )
    try:
        generated_names = {
            client_method_name(method.name) for _, method in iter_methods()
        }
        assert generated_names <= set(dir(grpc_client))
        assert all(callable(getattr(grpc_client, name)) for name in generated_names)
        assert generated_names <= GrpcMultiprocessClient.__dict__.keys()
        assert "submit_request" not in GrpcMultiprocessClient.__dict__
    finally:
        grpc_client.close()


# test-import: allow -- AST contract check intentionally scans checkout sources.
def test_business_callers_respect_request_transport_boundaries() -> None:
    """Worker clients use the factory, and the legacy listener stays server-side."""
    repo_root = repo_path()
    product = repo_root / "lmcache"
    client_transport = product / "v1/client/transport"
    server_transport = product / "v1/multiprocess/transport"
    implementation_tests = {
        repo_root / "tests/v1/multiprocess/test_mq.py",
        repo_root / "tests/v1/multiprocess/test_zmq_auth.py",
    }
    violations: list[str] = []
    for source_root in (product, repo_root / "tests"):
        for path in source_root.rglob("*.py"):
            forbidden_modules = []
            if (
                not path.is_relative_to(server_transport)
                and path not in implementation_tests
            ):
                forbidden_modules.append("lmcache.v1.multiprocess.transport.zmq_impl")
            if source_root == product and not path.is_relative_to(client_transport):
                forbidden_modules.append("lmcache.v1.client.transport.grpc_impl")
            if not forbidden_modules:
                continue
            package = ".".join(path.relative_to(repo_root).parent.parts)
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                    imports = [f"{module}.{alias.name}" for alias in node.names]
                elif isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                else:
                    continue
                if any(
                    name == module or name.startswith(f"{module}.")
                    for name in imports
                    for module in forbidden_modules
                ):
                    violations.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert violations == []
