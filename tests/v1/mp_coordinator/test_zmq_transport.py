# SPDX-License-Identifier: Apache-2.0
"""Tests for the ZMQ transport over real loopback sockets.

Each test stands up a real client control REP (via ``serve_commands``) and a
real coordinator transport, exercising the command channel end to end through
the public interface only -- no private attributes are touched.
"""

# Standard
import asyncio
import socket as _socket
import threading
import time

# Third Party
import pytest
import zmq
import zmq.asyncio

# First Party
from lmcache.v1.mp_coordinator.transport import (
    ReachInfo,
    TransportError,
    UnknownInstanceError,
)
from lmcache.v1.mp_coordinator.zmq_transport import (
    ZmqClientTransport,
    ZmqCoordinatorTransport,
)


def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ControlServer:
    """A real mp-side control REP server backed by ZmqClientTransport.

    Args:
        control_port: Port to bind the control REP socket on.
        handler: Maps a received command payload to a reply payload.
    """

    def __init__(self, control_port: int, handler) -> None:
        self._client = ZmqClientTransport(
            zmq.Context.instance(), "127.0.0.1:1", "127.0.0.1:1", control_port
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._client.serve_commands,
            args=(handler, self._stop.is_set),
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        time.sleep(0.2)  # let the REP socket bind

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _coordinator(
    ctx: zmq.asyncio.Context, command_timeout_ms: int = 30000
) -> ZmqCoordinatorTransport:
    return ZmqCoordinatorTransport(
        ctx,
        f"127.0.0.1:{_free_port()}",
        f"127.0.0.1:{_free_port()}",
        command_timeout_ms=command_timeout_ms,
    )


def test_command_round_trip_and_broadcast():
    control_port = _free_port()
    server = _ControlServer(control_port, lambda payload: b"ack")
    server.start()

    async def run():
        ctx = zmq.asyncio.Context()
        coord = _coordinator(ctx)
        try:
            coord.add_instance("a", ReachInfo("127.0.0.1", control_port))
            assert await coord.send_command("a", b"x") == b"ack"
            assert await coord.broadcast(b"y") == {"a": b"ack"}
        finally:
            coord.close()
            ctx.destroy(linger=0)

    try:
        asyncio.run(run())
    finally:
        server.stop()


def test_concurrent_send_to_same_instance_is_serialized():
    # Without the per-instance lock, two concurrent exchanges on one REQ socket
    # would violate the REQ state machine (EFSM) and one would raise. The lock
    # serializes them, so both succeed.
    control_port = _free_port()

    def slow(payload: bytes) -> bytes:
        time.sleep(0.05)
        return b"ack"

    server = _ControlServer(control_port, slow)
    server.start()

    async def run():
        ctx = zmq.asyncio.Context()
        coord = _coordinator(ctx)
        try:
            coord.add_instance("a", ReachInfo("127.0.0.1", control_port))
            r1, r2 = await asyncio.gather(
                coord.send_command("a", b"1"), coord.send_command("a", b"2")
            )
            assert r1 == b"ack" and r2 == b"ack"
        finally:
            coord.close()
            ctx.destroy(linger=0)

    try:
        asyncio.run(run())
    finally:
        server.stop()


def test_send_command_recovers_after_failure():
    # First command times out (nothing listening) -> TransportError. The socket
    # is rebuilt, so once a server comes up the next command succeeds rather
    # than staying wedged in the REQ state machine.
    control_port = _free_port()

    async def run():
        ctx = zmq.asyncio.Context()
        coord = _coordinator(ctx, command_timeout_ms=300)
        server = None
        try:
            coord.add_instance("a", ReachInfo("127.0.0.1", control_port))
            with pytest.raises(TransportError):
                await coord.send_command("a", b"x")

            server = _ControlServer(control_port, lambda payload: b"ack")
            server.start()
            assert await coord.send_command("a", b"x") == b"ack"
        finally:
            coord.close()
            ctx.destroy(linger=0)
            if server is not None:
                server.stop()

    asyncio.run(run())


def test_remove_while_command_waits_for_lock_raises_unknown():
    # One in-flight command holds the per-instance lock; a second queues on it.
    # Removing the instance must make the queued command raise
    # UnknownInstanceError (re-checked under the lock), never a raw KeyError
    # that would abort a broadcast.
    control_port = _free_port()

    def slow(payload: bytes) -> bytes:
        time.sleep(0.3)
        return b"ack"

    server = _ControlServer(control_port, slow)
    server.start()

    async def run():
        ctx = zmq.asyncio.Context()
        coord = _coordinator(ctx, command_timeout_ms=5000)
        try:
            coord.add_instance("a", ReachInfo("127.0.0.1", control_port))
            first = asyncio.create_task(coord.send_command("a", b"1"))
            await asyncio.sleep(0.1)  # let `first` take the lock and block in recv
            second = asyncio.create_task(coord.send_command("a", b"2"))
            await asyncio.sleep(0.05)  # let `second` queue on the lock
            coord.remove_instance("a")
            results = await asyncio.gather(first, second, return_exceptions=True)
            # The queued command surfaces the contracted exception, not KeyError.
            assert isinstance(results[1], UnknownInstanceError)
        finally:
            coord.close()
            ctx.destroy(linger=0)

    try:
        asyncio.run(run())
    finally:
        server.stop()


def test_send_command_unknown_instance_raises():
    async def run():
        ctx = zmq.asyncio.Context()
        coord = _coordinator(ctx)
        try:
            with pytest.raises(UnknownInstanceError):
                await coord.send_command("missing", b"x")
        finally:
            coord.close()
            ctx.destroy(linger=0)

    asyncio.run(run())


def test_broadcast_empty_when_no_instances():
    async def run():
        ctx = zmq.asyncio.Context()
        coord = _coordinator(ctx)
        try:
            assert await coord.broadcast(b"x") == {}
        finally:
            coord.close()
            ctx.destroy(linger=0)

    asyncio.run(run())
