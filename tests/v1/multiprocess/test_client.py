# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Any
import ast

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.grpc_impl.client import (
    GrpcMultiprocessClient,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    client_method_name,
    iter_methods,
)
from lmcache.v1.multiprocess.transport.zmq_impl import ZmqMultiprocessClient


class _RecordingMessageQueueClient(MessageQueueClient):
    """Record requests without opening a ZMQ socket."""

    def __init__(self) -> None:
        self.calls: list[tuple[RequestType, list[Any], Any | None]] = []
        self.closed = False

    def submit_request(
        self,
        request_type: RequestType,
        request_payloads: list[Any],
        response_cls: Any | None = None,
    ) -> MessagingFuture[Any]:
        future: MessagingFuture[Any] = MessagingFuture()
        self.calls.append((request_type, request_payloads, response_cls))
        future.set_result(request_type)
        return future

    def close(self) -> None:
        self.closed = True


def test_all_request_types_have_explicit_named_methods() -> None:
    contract_names = {
        name for name, value in RequestClient.__dict__.items() if callable(value)
    }
    expected_names = {name.lower() for name in RequestType.__members__}

    assert expected_names <= contract_names
    zmq_method_names = {
        name
        for name, value in ZmqMultiprocessClient.__dict__.items()
        if callable(value)
    }
    assert expected_names <= zmq_method_names
    assert "submit_request" not in ZmqMultiprocessClient.__dict__

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


def test_transport_clients_explicitly_inherit_shared_contract() -> None:
    assert RequestClient in ZmqMultiprocessClient.__bases__
    assert RequestClient in GrpcMultiprocessClient.__bases__


def test_zmq_client_explicitly_inherits_shared_contract() -> None:
    assert RequestClient in ZmqMultiprocessClient.__bases__


def test_only_zmq_transport_layer_submits_request_envelopes() -> None:
    """Business callers must use named methods instead of ZMQ envelopes."""
    repo_root = Path(__file__).parents[3]
    allowed = {
        repo_root / "lmcache/v1/multiprocess/mq.py",
        repo_root / "lmcache/v1/multiprocess/transport/zmq_impl/client.py",
    }
    violations: list[str] = []
    for path in (repo_root / "lmcache").rglob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "submit_request":
                violations.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert violations == []


def test_business_callers_create_clients_through_factory() -> None:
    """Transport implementations must not leak into business callers or tests."""
    repo_root = Path(__file__).parents[3]
    transport_root = repo_root / "lmcache/v1/multiprocess/transport"
    implementation_tests = {
        repo_root / "tests/v1/multiprocess/test_client.py",
        repo_root / "tests/v1/multiprocess/test_mq.py",
    }
    violations: list[str] = []
    for source_root in (repo_root / "lmcache", repo_root / "tests"):
        for path in source_root.rglob("*.py"):
            if path.is_relative_to(transport_root) or path in implementation_tests:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                imports_raw_client = (
                    node.module == "lmcache.v1.multiprocess.mq"
                    and any(alias.name == "MessageQueueClient" for alias in node.names)
                )
                if imports_raw_client or node.module.startswith(
                    "lmcache.v1.multiprocess.transport.zmq_impl"
                ):
                    violations.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert violations == []


def test_named_rpc_method_delegates_to_zmq_request_envelope() -> None:
    transport = _RecordingMessageQueueClient()
    client = ZmqMultiprocessClient(transport)

    future = client.lookup("key", 4)

    assert future.result(timeout=0) is RequestType.LOOKUP
    assert transport.calls == [
        (
            RequestType.LOOKUP,
            ["key", 4],
            get_response_class(RequestType.LOOKUP),
        )
    ]


def test_compatibility_alias_delegates_to_same_zmq_request_type() -> None:
    transport = _RecordingMessageQueueClient()
    client = ZmqMultiprocessClient(transport)

    client.cb_unregister_rope_v3(7)

    assert transport.calls == [
        (
            RequestType.CB_UNREGISTER_ROPE,
            [7],
            get_response_class(RequestType.CB_UNREGISTER_ROPE),
        )
    ]


def test_close_delegates_to_zmq_client() -> None:
    transport = _RecordingMessageQueueClient()
    client = ZmqMultiprocessClient(transport)

    client.close()

    assert transport.closed
