# SPDX-License-Identifier: Apache-2.0
"""Authentication tests for the multiprocess ZMQ transport."""

# Standard
from typing import Any
import socket

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.security.api_key import APIKeyVerifier
from lmcache.v1.multiprocess.futures import MessagingFuture, MessagingStream
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    MessageQueueClient,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.transport.zmq_impl.wire import (
    AUTH_ENVELOPE_MAGIC,
    AUTH_ERROR_FRAME,
    encode_authentication_frame,
)


def _unused_tcp_url() -> str:
    """Reserve and release a local TCP port for one test server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


def _run_noop(
    accepted_keys: list[str], client_key: str | None
) -> tuple[str | BaseException, int]:
    """Run one real ZMQ request and return its outcome and dispatch count."""
    context = zmq.Context()
    calls = 0

    def noop() -> str:
        nonlocal calls
        calls += 1
        return "NOOP_OK"

    server = MessageQueueServer(
        _unused_tcp_url(), context, APIKeyVerifier(accepted_keys)
    )
    server.add_handler("noop", HandlerType.SERIALIZED, noop)
    server.start()
    endpoint = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    client = MessageQueueClient(endpoint, context, api_key=client_key)
    try:
        future: MessagingFuture[Any] | MessagingStream[Any] = client.submit_request(
            "noop", []
        )
        assert isinstance(future, MessagingFuture)
        try:
            outcome: str | BaseException = future.result(timeout=3)
        except BaseException as error:
            outcome = error
        return outcome, calls
    finally:
        client.close()
        server.close()
        context.term()


def test_zmq_authentication_accepts_each_configured_key() -> None:
    """Multiple accepted server keys support credential rotation."""
    assert _run_noop(["current", "next"], "current") == ("NOOP_OK", 1)
    assert _run_noop(["current", "next"], "next") == ("NOOP_OK", 1)


@pytest.mark.parametrize("client_key", [None, "wrong"])
def test_zmq_authentication_rejects_before_dispatch(client_key: str | None) -> None:
    """Missing and incorrect credentials never reach the RPC handler."""
    outcome, calls = _run_noop(["secret"], client_key)

    assert isinstance(outcome, PermissionError)
    assert calls == 0


@pytest.mark.parametrize("client_key", [None, "unused"])
def test_zmq_authentication_disabled_is_wire_compatible(
    client_key: str | None,
) -> None:
    """A new server without keys accepts legacy and enveloped requests."""
    assert _run_noop([], client_key) == ("NOOP_OK", 1)


@pytest.mark.parametrize(
    "auth_frames",
    [
        [],
        [encode_authentication_frame("wrong")],
        [AUTH_ENVELOPE_MAGIC + b"\xc1"],
        [
            AUTH_ENVELOPE_MAGIC
            + msgspec.msgpack.encode({"version": 2, "token": "secret"})
        ],
    ],
)
def test_invalid_authentication_precedes_operation_decoding(
    auth_frames: list[bytes],
) -> None:
    """Invalid credentials are rejected even if the RPC data cannot be decoded."""
    context = zmq.Context()
    server = MessageQueueServer(
        "tcp://127.0.0.1:0", context, APIKeyVerifier(["secret"])
    )
    client = context.socket(zmq.DEALER)
    client.setsockopt(zmq.RCVTIMEO, 3000)
    client.connect(server.socket.getsockopt_string(zmq.LAST_ENDPOINT))
    server.start()
    try:
        client.send_multipart([*auth_frames, b"opaque-uid", b"\xc1", b"\xc1"])
        assert client.recv_multipart() == [b"opaque-uid", AUTH_ERROR_FRAME]
    finally:
        client.close(linger=0)
        server.close()
        context.term()
