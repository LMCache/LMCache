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
import pytest

# First Party
from lmcache.v1.mp_coordinator.client import CoordinatorClient
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.manager import MPCoordinatorManager
from lmcache.v1.mp_coordinator.message import (
    DeregisterMsg,
    ErrorMsg,
    HeartbeatMsg,
    RegisterMsg,
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
    """A request type intentionally absent from any dispatch table."""

    value: int = 0


def test_dispatch_table_built_from_controllers():
    manager = MPCoordinatorManager(_config())
    try:
        assert RegisterMsg in manager._req_dispatch
        assert HeartbeatMsg in manager._req_dispatch
        assert DeregisterMsg in manager._push_dispatch
    finally:
        manager.close()


def test_client_start_times_out_when_coordinator_down():
    # reply_url points at a free port with nothing listening: connect succeeds
    # lazily, send succeeds, recv times out -> start() must raise (not hang) and
    # release the already-bound control socket.
    control_port = _free_port()
    client = CoordinatorClient(
        instance_id="x",
        reply_url=f"127.0.0.1:{_free_port()}",
        heartbeat_url="127.0.0.1:1",
        pull_url="127.0.0.1:1",
        control_port=control_port,
        advertise_ip="127.0.0.1",
        register_timeout_ms=300,
    )
    with pytest.raises(RuntimeError):
        client.start()
    # Control socket was closed on failure, leaving a clean state.
    assert client._control_socket is None


def test_dispatch_unknown_request_returns_error():
    manager = MPCoordinatorManager(_config())
    try:
        ret = asyncio.run(manager._dispatch_req(_UnknownReq()))
        assert isinstance(ret, ErrorMsg)
    finally:
        manager.close()


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
        manager = MPCoordinatorManager(config)
        loop = asyncio.get_running_loop()
        server = asyncio.create_task(manager.start_all())
        await asyncio.sleep(0.2)  # let sockets bind and loop start

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

            # Push channel: broadcast reaches both control sockets and gets acks.
            replies = await manager.command_sender.broadcast(b"ping")
            assert replies == {"i1": b"ack", "i2": b"ack"}

            # Deregister via PUSH: client2 leaves.
            await loop.run_in_executor(None, client2.stop)
            assert await _wait_for(lambda: not manager.registry.contains("i2"))
            assert manager.registry.contains("i1")

            # Health eviction: a ghost that never heartbeats is evicted.
            ghost = RegisterMsg(
                instance_id="ghost",
                ip="127.0.0.1",
                control_port=_free_port(),
            )
            fut = asyncio.run_coroutine_threadsafe(
                manager._registration.register_instance(ghost), loop
            )
            await asyncio.wrap_future(fut)
            assert manager.registry.contains("ghost")
            assert await _wait_for(
                lambda: not manager.registry.contains("ghost"), timeout=3.0
            )

            # client1 still alive thanks to heartbeats.
            assert manager.registry.contains("i1")
        finally:
            await loop.run_in_executor(None, client1.stop)
            server.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server
            manager.close()

    asyncio.run(run())
