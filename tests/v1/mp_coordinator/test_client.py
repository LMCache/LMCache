# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CoordinatorClient over a fake transport."""

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.mp_coordinator.client import CoordinatorClient
from lmcache.v1.mp_coordinator.message import ErrorMsg, RegisterRetMsg
from lmcache.v1.mp_coordinator.transport import (
    ClientCommandHandler,
    ClientTransport,
    StopPredicate,
    TransportError,
)


class _FakeClientTransport(ClientTransport):
    """Fake client transport with scripted request behavior."""

    def __init__(self, reply: bytes | None = None, fail: bool = False) -> None:
        self._reply = reply
        self._fail = fail
        self.pushed: list[bytes] = []
        self.closed = False

    def request(self, payload: bytes, timeout_ms: int) -> bytes:
        if self._fail:
            raise TransportError("coordinator down")
        assert self._reply is not None
        return self._reply

    def push(self, payload: bytes) -> None:
        self.pushed.append(payload)

    def serve_commands(
        self, handler: ClientCommandHandler, should_stop: StopPredicate
    ) -> None:  # pragma: no cover - not exercised here
        return None

    def close(self) -> None:
        self.closed = True


def _client(transport: ClientTransport) -> CoordinatorClient:
    return CoordinatorClient(
        instance_id="i",
        transport=transport,
        control_port=9999,
        advertise_ip="127.0.0.1",
        heartbeat_interval=0.1,
        register_timeout_ms=300,
    )


def test_start_raises_when_registration_fails():
    client = _client(_FakeClientTransport(fail=True))
    with pytest.raises(RuntimeError):
        client.start()


def test_start_raises_on_unexpected_reply_type():
    # Coordinator returns an ErrorMsg-shaped payload instead of RegisterRetMsg.
    bad = msgspec.msgpack.encode(ErrorMsg(error="nope"))
    client = _client(_FakeClientTransport(reply=bad))
    with pytest.raises(RuntimeError):
        client.start()


def test_start_returns_register_reply():
    reply = msgspec.msgpack.encode(RegisterRetMsg())
    transport = _FakeClientTransport(reply=reply)
    client = _client(transport)
    ret = client.start()
    assert isinstance(ret, RegisterRetMsg)
    client.stop()
    assert transport.closed
    # stop() deregisters via a push.
    assert len(transport.pushed) == 1
