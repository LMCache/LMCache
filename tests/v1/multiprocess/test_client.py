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
    method_names = {
        name
        for name, value in ZmqMultiprocessClient.__dict__.items()
        if callable(value)
    }
    expected_names = {name.lower() for name in RequestType.__members__}

    assert expected_names <= contract_names
    assert expected_names <= method_names
    assert "__getattr__" not in ZmqMultiprocessClient.__dict__
    assert "submit_request" not in ZmqMultiprocessClient.__dict__


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
