# SPDX-License-Identifier: Apache-2.0
"""Tests for MPCoordinatorManager: dispatch table, transport, health.

The integration test starts the manager on a real event loop in loopback and
drives it with one or more CoordinatorClients. Client calls are blocking
(synchronous ZMQ), so they are offloaded to a thread executor to avoid blocking
the loop that must answer them.
"""

# Standard
import asyncio
import contextlib
import dataclasses
import socket as _socket

# Third Party
import msgspec
import pytest
import zmq
import zmq.asyncio

# First Party
from lmcache.v1.mp_coordinator.client import CoordinatorClient
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.manager import MPCoordinatorManager
from lmcache.v1.mp_coordinator.message import (
    CoordMsg,
    ErrorMsg,
    RegisterMsg,
    RegisterRetMsg,
    ReqMsg,
)


def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _config(**overrides) -> MPCoordinatorConfig:
    """Build a coordinator config bound to free loopback ports."""
    config = MPCoordinatorConfig(
        pull_url=f"127.0.0.1:{_free_port()}",
        reply_url=f"127.0.0.1:{_free_port()}",
        heartbeat_url=f"127.0.0.1:{_free_port()}",
        heartbeat_interval=0.2,
        instance_timeout=0.6,
        health_check_interval=0.0,
    )
    if overrides:
        config = dataclasses.replace(config, **overrides)
    return config


class _UnknownReq(ReqMsg):
    """A request type intentionally absent from the CoordMsg union."""

    value: int = 0


@contextlib.asynccontextmanager
async def _running_manager(config: MPCoordinatorConfig):
    """Start a coordinator on its own task and tear it down afterwards."""
    manager = MPCoordinatorManager(config)
    server = asyncio.create_task(manager.start_all())
    await asyncio.sleep(0.2)  # let sockets bind and the loop start
    try:
        yield manager
    finally:
        server.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server
        manager.close()


async def _request(reply_url: str, msg: object) -> object:
    """Send a request to the coordinator reply socket and decode the reply."""
    ctx = zmq.asyncio.Context.instance()
    socket = ctx.socket(zmq.REQ)  # type: ignore[attr-defined]
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.connect(f"tcp://{reply_url}")
    try:
        await socket.send(msgspec.msgpack.encode(msg))
        return msgspec.msgpack.decode(await socket.recv(), type=CoordMsg)
    finally:
        socket.close()


def test_register_request_is_handled():
    # Sending a RegisterMsg over the public reply socket yields a
    # RegisterRetMsg, proving the registration controller's handler is wired
    # into the dispatch table.
    async def run():
        config = _config()
        async with _running_manager(config):
            reply = await _request(
                config.reply_url,
                RegisterMsg(instance_id="a", ip="127.0.0.1", control_port=_free_port()),
            )
            assert isinstance(reply, RegisterRetMsg)

    asyncio.run(run())


def test_client_start_times_out_when_coordinator_down():
    # reply_url points at a free port with nothing listening: connect succeeds
    # lazily, send succeeds, recv times out -> start() must raise (not hang).
    client = CoordinatorClient(
        instance_id="x",
        reply_url=f"127.0.0.1:{_free_port()}",
        heartbeat_url="127.0.0.1:1",
        pull_url="127.0.0.1:1",
        control_port=_free_port(),
        advertise_ip="127.0.0.1",
        register_timeout_ms=300,
    )
    with pytest.raises(RuntimeError):
        client.start()


def test_unknown_request_returns_error():
    # A message whose tag is not in the CoordMsg union cannot be decoded by the
    # coordinator, which must answer with an ErrorMsg rather than crash.
    async def run():
        config = _config()
        async with _running_manager(config):
            reply = await _request(config.reply_url, _UnknownReq())
            assert isinstance(reply, ErrorMsg)

    asyncio.run(run())


def _make_client(config: MPCoordinatorConfig, instance_id: str) -> CoordinatorClient:
    """Build a CoordinatorClient targeting the given coordinator config."""
    return CoordinatorClient(
        instance_id=instance_id,
        reply_url=config.reply_url,
        heartbeat_url=config.heartbeat_url,
        pull_url=config.pull_url,
        control_port=_free_port(),
        heartbeat_interval=config.heartbeat_interval,
        advertise_ip="127.0.0.1",
    )


async def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll a predicate until true or timeout (seconds)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


def test_fleet_registration_broadcast_and_health_eviction():
    async def run():
        config = _config(health_check_interval=0.2)
        loop = asyncio.get_running_loop()
        async with _running_manager(config) as manager:
            client1 = _make_client(config, "i1")
            client2 = _make_client(config, "i2")
            try:
                # Register both clients (blocking start offloaded to threads).
                await loop.run_in_executor(None, client1.start)
                await loop.run_in_executor(None, client2.start)
                assert await _wait_for(
                    lambda: {n.instance_id for n in manager.registry.all_instances()}
                    == {"i1", "i2"}
                )

                # Push channel: broadcast reaches both control sockets, acked.
                replies = await manager.command_sender.broadcast(b"ping")
                assert replies == {"i1": b"ack", "i2": b"ack"}

                # Deregister via PUSH: client2 leaves.
                await loop.run_in_executor(None, client2.stop)
                assert await _wait_for(lambda: not manager.registry.contains("i2"))
                assert manager.registry.contains("i1")

                # Health eviction: a ghost registered over the public socket
                # that never heartbeats is evicted after the timeout.
                ghost = RegisterMsg(
                    instance_id="ghost", ip="127.0.0.1", control_port=_free_port()
                )
                reply = await _request(config.reply_url, ghost)
                assert isinstance(reply, RegisterRetMsg)
                assert manager.registry.contains("ghost")
                assert await _wait_for(
                    lambda: not manager.registry.contains("ghost"), timeout=3.0
                )

                # client1 still alive thanks to heartbeats.
                assert manager.registry.contains("i1")
            finally:
                await loop.run_in_executor(None, client1.stop)

    asyncio.run(run())
