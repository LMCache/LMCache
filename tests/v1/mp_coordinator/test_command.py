# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CommandSender: per-instance serialization and socket repair.

Tests run via ``asyncio.run`` and use fake async sockets, so they need neither
the pytest-asyncio plugin nor real ZMQ I/O. A real (but unconnected) ZMQ
context is used where the repair path must rebuild a socket; connecting to an
unused loopback port is lazy in ZMQ, so no listener is required.
"""

# Standard
import asyncio
import socket as _socket

# Third Party
import pytest
import zmq
import zmq.asyncio

# First Party
from lmcache.v1.mp_coordinator.command import CommandSender
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstanceNode


def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _SlowSocket:
    """Fake async REQ socket that tracks how many exchanges overlap.

    A send increments the active count, the paired recv decrements it; both
    await, so without serialization two concurrent exchanges would overlap and
    push ``max_active`` above 1.
    """

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def send(self, payload: bytes) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)

    async def recv(self) -> bytes:
        await asyncio.sleep(0.01)
        self.active -= 1
        return b"ok"


class _FailingSocket:
    """Fake async socket whose send always raises a ZMQ error."""

    def __init__(self) -> None:
        self.closed = False

    async def send(self, payload: bytes) -> None:
        raise zmq.ZMQError("boom")

    async def recv(self) -> bytes:  # pragma: no cover - never reached
        return b""

    def setsockopt(self, *args: object) -> None:
        # Allow close_zmq_socket() to set LINGER on this fake.
        return None

    def close(self) -> None:
        self.closed = True


class _OkSocket:
    """Fake async socket that returns a fixed reply."""

    async def send(self, payload: bytes) -> None:
        return None

    async def recv(self) -> bytes:
        return b"ack"


def _node(instance_id: str, sock: object, control_port: int = 5000) -> MPInstanceNode:
    return MPInstanceNode(
        instance_id=instance_id,
        ip="127.0.0.1",
        control_port=control_port,
        command_socket=sock,  # type: ignore[arg-type]
        registration_time=0.0,
        last_heartbeat_time=0.0,
    )


def test_unicast_serializes_same_instance():
    registry = InstanceRegistry()
    sock = _SlowSocket()
    registry.register(_node("a", sock))
    # Context is unused on the success path.
    sender = CommandSender(registry, zmq.asyncio.Context.instance())

    async def run():
        await asyncio.gather(
            sender.unicast("a", b"1"),
            sender.unicast("a", b"2"),
        )

    asyncio.run(run())
    # Lock kept the two exchanges from overlapping on the one REQ socket.
    assert sock.max_active == 1


def test_unicast_rebuilds_socket_on_failure_then_recovers():
    async def run():
        ctx = zmq.asyncio.Context()
        try:
            registry = InstanceRegistry()
            bad = _FailingSocket()
            node = _node("a", bad, control_port=_free_port())
            registry.register(node)
            sender = CommandSender(registry, ctx)

            # The failing exchange raises, but the wedged socket is closed and
            # replaced so the channel is not permanently poisoned.
            with pytest.raises(zmq.ZMQError):
                await sender.unicast("a", b"x")
            assert bad.closed
            assert node.command_socket is not bad
        finally:
            ctx.destroy(linger=0)

    asyncio.run(run())


def test_unicast_unknown_instance_raises_keyerror():
    sender = CommandSender(InstanceRegistry(), zmq.asyncio.Context.instance())

    async def run():
        with pytest.raises(KeyError):
            await sender.unicast("missing", b"x")

    asyncio.run(run())


def test_broadcast_skips_failed_instances():
    async def run():
        ctx = zmq.asyncio.Context()
        try:
            registry = InstanceRegistry()
            registry.register(_node("ok", _OkSocket()))
            registry.register(_node("bad", _FailingSocket(), control_port=_free_port()))
            sender = CommandSender(registry, ctx)
            return await sender.broadcast(b"ping")
        finally:
            ctx.destroy(linger=0)

    replies = asyncio.run(run())
    assert replies == {"ok": b"ack"}
