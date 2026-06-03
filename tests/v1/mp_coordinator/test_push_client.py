# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HttpPushClient using an httpx MockTransport (no real I/O)."""

# Standard
import asyncio

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.mp_coordinator.push_client import (
    HttpPushClient,
    PushError,
    UnknownInstanceError,
)
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstance


def _instance(instance_id: str, http_port: int) -> MPInstance:
    return MPInstance(
        instance_id=instance_id,
        ip="127.0.0.1",
        http_port=http_port,
        registration_time=0.0,
        last_heartbeat_time=0.0,
    )


def test_send_command_returns_reply():
    registry = InstanceRegistry()
    registry.register(_instance("i1", 8080))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/coordinator/command"
        return httpx.Response(200, json={"status": "ok"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as ac:
            client = HttpPushClient(registry, ac)
            assert await client.send_command("i1", {"type": "ping"}) == {"status": "ok"}

    asyncio.run(run())


def test_send_command_unknown_instance_raises():
    async def run():
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        async with httpx.AsyncClient(transport=transport) as ac:
            client = HttpPushClient(InstanceRegistry(), ac)
            with pytest.raises(UnknownInstanceError):
                await client.send_command("missing", {})

    asyncio.run(run())


def test_send_command_error_status_raises_push_error():
    registry = InstanceRegistry()
    registry.register(_instance("i1", 8080))

    async def run():
        transport = httpx.MockTransport(lambda r: httpx.Response(500, json={"e": "x"}))
        async with httpx.AsyncClient(transport=transport) as ac:
            client = HttpPushClient(registry, ac)
            with pytest.raises(PushError):
                await client.send_command("i1", {})

    asyncio.run(run())


def test_broadcast_skips_failed_instances():
    registry = InstanceRegistry()
    registry.register(_instance("ok", 8080))
    registry.register(_instance("bad", 8081))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 8081:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"status": "ok"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as ac:
            client = HttpPushClient(registry, ac)
            replies = await client.broadcast({"type": "ping"})
            assert replies == {"ok": {"status": "ok"}}

    asyncio.run(run())
