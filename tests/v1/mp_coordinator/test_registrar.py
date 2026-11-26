# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mp-side coordinator registration helpers.

Driven with an httpx MockTransport (no real I/O): register, the keep_registered
loop (heartbeat, re-register on 404, deregister on cancel), and resilience when
the coordinator is unreachable.
"""

# Standard
from unittest.mock import MagicMock
import asyncio
import contextlib
import json

# Third Party
import httpx

# First Party
from lmcache.v1.mp_coordinator.registrar import (
    bind_coordinator_chunk_size,
    keep_registered,
    register,
)

_BASE = "http://coord:9300"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_register_returns_assigned_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.url.path == "/instances"
            return httpx.Response(
                200, json={"instance_id": "mp-xyz", "re_registered": False}
            )
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            assigned = await register(
                client, _BASE, http_port=8080, advertise_ip="127.0.0.1"
            )
            assert assigned == "mp-xyz"

    asyncio.run(run())


def test_bind_coordinator_chunk_size(monkeypatch):
    response = httpx.Response(
        200,
        json={"chunk_size": 640},
        request=httpx.Request("PUT", f"{_BASE}/config/chunk-size"),
    )
    request = MagicMock(return_value=response)
    monkeypatch.setattr(httpx, "put", request)

    assert bind_coordinator_chunk_size(f"{_BASE}/", 640, timeout=3.0) == 640
    request.assert_called_once_with(
        f"{_BASE}/config/chunk-size",
        json={"chunk_size": 640},
        headers={},
        timeout=3.0,
    )


def test_bind_coordinator_chunk_size_uses_cli_key_file(tmp_path, monkeypatch):
    key_file = tmp_path / "key"
    key_file.write_text("fleet-secret\n", encoding="utf-8")
    for name in (
        "LMCACHE_API_KEY",
        "LMCACHE_API_KEY_FILE",
        "LMCACHE_HTTP_API_KEY",
        "LMCACHE_HTTP_API_KEY_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    response = httpx.Response(
        200,
        json={"chunk_size": 640},
        request=httpx.Request("PUT", f"{_BASE}/config/chunk-size"),
    )
    request = MagicMock(return_value=response)
    monkeypatch.setattr(httpx, "put", request)

    bind_coordinator_chunk_size(_BASE, 640, api_key_files=[str(key_file)])

    assert request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer fleet-secret"
    }


def test_register_forwards_p2p_and_mq_fields():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # Standard
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"instance_id": "i1", "re_registered": False})

    async def run():
        async with _client(handler) as client:
            await register(
                client,
                _BASE,
                http_port=8080,
                advertise_ip="10.0.0.1",
                instance_id="i1",
                p2p_advertised_url="10.0.0.1:7600",
                mq_port=5555,
                chunk_size=640,
            )

    asyncio.run(run())
    # The MQ server is reachable at the same advertised ip, so no separate
    # mq_ip is sent -- only the port.
    assert captured["ip"] == "10.0.0.1"
    assert captured["p2p_advertised_url"] == "10.0.0.1:7600"
    assert captured["mq_port"] == 5555
    assert captured["chunk_size"] == 640


def test_register_omits_mq_port_when_p2p_disabled():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # Standard
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"instance_id": "i1", "re_registered": False})

    async def run():
        async with _client(handler) as client:
            await register(client, _BASE, http_port=8080, advertise_ip="10.0.0.1")

    asyncio.run(run())
    assert captured["mq_port"] == 0


async def _run_loop_briefly(client: httpx.AsyncClient, **kwargs) -> None:
    """Run keep_registered for a few ticks, then cancel it.

    Supplies a no-op ``on_registered`` unless the case under test overrides
    it; the parameter is required, so only tests need the no-op.
    """
    kwargs.setdefault("on_registered", lambda: None)
    task = asyncio.create_task(
        keep_registered(
            client,
            _BASE,
            http_port=8080,
            advertise_ip="127.0.0.1",
            heartbeat_interval=0.03,
            **kwargs,
        )
    )
    await asyncio.sleep(0.12)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_keep_registered_heartbeats_then_deregisters():
    seen = {"heartbeats": 0, "delete_path": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            seen["heartbeats"] += 1
            return httpx.Response(200, json={"instance_id": "i1"})
        seen["delete_path"] = request.url.path  # DELETE on cancel
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(client, instance_id="i1")

    asyncio.run(run())
    assert seen["heartbeats"] >= 1
    assert seen["delete_path"] == "/instances/i1"


def test_keep_registered_reregisters_on_404():
    calls = {"register": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            calls["register"] += 1
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            return httpx.Response(404, json={"error": "unknown"})
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(client, instance_id="i1")

    asyncio.run(run())
    assert calls["register"] >= 2  # initial + at least one re-register after 404


def test_keep_registered_keeps_id_on_transient_heartbeat_failure():
    seen = {"register": 0, "delete_path": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen["register"] += 1
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            return httpx.Response(500, json={"error": "transient"})
        seen["delete_path"] = request.url.path
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(client, instance_id="i1")

    asyncio.run(run())
    # Transient 5xx must NOT spawn a duplicate registration, and the id is
    # kept so shutdown can still deregister.
    assert seen["register"] == 1
    assert seen["delete_path"] == "/instances/i1"


def test_keep_registered_survives_malformed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        # 200 with a schema-mismatched body -> RegisterResponse validation fails.
        return httpx.Response(200, json={"unexpected": "field"})

    async def run():
        async with _client(handler) as client:
            task = asyncio.create_task(
                keep_registered(
                    client,
                    _BASE,
                    http_port=8080,
                    advertise_ip="127.0.0.1",
                    heartbeat_interval=0.03,
                    on_registered=lambda: None,
                )
            )
            await asyncio.sleep(0.1)
            assert not task.done()  # validation error did not kill the task
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_keep_registered_survives_unreachable_coordinator():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("coordinator down")

    async def run():
        async with _client(handler) as client:
            task = asyncio.create_task(
                keep_registered(
                    client,
                    _BASE,
                    http_port=8080,
                    advertise_ip="127.0.0.1",
                    heartbeat_interval=0.03,
                    on_registered=lambda: None,
                )
            )
            await asyncio.sleep(0.1)
            assert not task.done()  # kept retrying, did not crash
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_on_registered_fires_for_every_registration():
    """The hook must fire on re-registration, not just the first one.

    A restarted coordinator keeps declarations in memory only, so without
    this the fleet view loses every capacity denominator until some
    unrelated topology change happens to resend one.
    """
    calls = {"register": 0, "hook": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            calls["register"] += 1
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            # Coordinator forgot us: forces a re-registration next tick.
            return httpx.Response(404, json={"error": "unknown"})
        return httpx.Response(204)

    def hook() -> None:
        calls["hook"] += 1

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(client, instance_id="i1", on_registered=hook)

    asyncio.run(run())
    assert calls["register"] >= 2, "expected a re-registration after the 404"
    assert calls["hook"] == calls["register"]


def test_keep_registered_refreshes_chunk_size_on_reregistration():
    seen: list[int | None] = []
    sizes = iter([None, 640])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            # Standard
            import json

            seen.append(json.loads(request.content)["chunk_size"])
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            return httpx.Response(404, json={"error": "unknown"})
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(
                client,
                instance_id="i1",
                chunk_size_provider=lambda: next(sizes, 640),
            )

    asyncio.run(run())
    assert seen[:2] == [None, 640]


def test_keep_registered_resends_bound_chunk_size_on_reregistration():
    """A restarted coordinator 404s the heartbeat and has lost its in-memory
    binding, so the re-registration must carry the size again."""
    seen: list[int | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen.append(json.loads(request.content)["chunk_size"])
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            return httpx.Response(404, json={"error": "unknown"})
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(
                client, instance_id="i1", chunk_size_provider=lambda: 640
            )

    asyncio.run(run())
    assert len(seen) >= 2
    assert set(seen) == {640}


def test_on_registered_is_required():
    """No default hook: a caller that republishes nothing says so."""
    # Standard
    import inspect

    parameter = inspect.signature(keep_registered).parameters["on_registered"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def _chunk_size_handler(seen: dict, register_status: int = 200, bind_status: int = 200):
    """Coordinator stub: records registrations, chunk-size binds and deletes."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen["register"].append(json.loads(request.content)["chunk_size"])
            return httpx.Response(
                register_status, json={"instance_id": "i1", "re_registered": False}
            )
        if request.url.path == "/config/chunk-size":
            seen["bind"].append(json.loads(request.content)["chunk_size"])
            return httpx.Response(bind_status, json={"chunk_size": 640})
        if request.method == "PUT":
            return httpx.Response(200, json={"instance_id": "i1"})
        seen["delete"] = request.url.path
        return httpx.Response(204)

    return handler


async def _run_loop_until_done(client: httpx.AsyncClient, **kwargs) -> bool:
    """Run keep_registered for a few ticks; return whether it stopped itself."""
    task = asyncio.create_task(
        keep_registered(
            client,
            _BASE,
            http_port=8080,
            advertise_ip="127.0.0.1",
            heartbeat_interval=0.03,
            instance_id="i1",
            on_registered=lambda: None,
            **kwargs,
        )
    )
    await asyncio.sleep(0.15)
    stopped = task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return stopped


def test_keep_registered_binds_chunk_size_finalized_after_registration():
    """Registered before negotiation, e.g. the coordinator was down when the
    size was finalized and came back with its state: the loop binds the size
    once."""
    seen: dict = {"register": [], "bind": [], "delete": None}
    sizes = iter([None])

    async def run() -> bool:
        async with _client(_chunk_size_handler(seen)) as client:
            return await _run_loop_until_done(
                client, chunk_size_provider=lambda: next(sizes, 640)
            )

    assert not asyncio.run(run())
    assert seen["register"] == [None]
    assert seen["bind"] == [640]


def test_keep_registered_does_not_rebind_size_sent_on_registration():
    seen: dict = {"register": [], "bind": [], "delete": None}

    async def run() -> bool:
        async with _client(_chunk_size_handler(seen)) as client:
            return await _run_loop_until_done(client, chunk_size_provider=lambda: 640)

    assert not asyncio.run(run())
    assert seen["register"] == [640]
    assert seen["bind"] == []


def test_keep_registered_leaves_coordinator_that_rejects_late_chunk_size():
    """A coordinator bound to another size during an outage rejects the late
    bind: the loop deregisters and stops instead of staying listed."""
    seen: dict = {"register": [], "bind": [], "delete": None}
    sizes = iter([None])

    async def run() -> bool:
        async with _client(_chunk_size_handler(seen, bind_status=409)) as client:
            return await _run_loop_until_done(
                client, chunk_size_provider=lambda: next(sizes, 640)
            )

    assert asyncio.run(run())
    assert seen["bind"] == [640]
    assert seen["delete"] == "/instances/i1"


def test_keep_registered_stops_when_registration_rejects_chunk_size():
    """A 409 on registration is not retried every tick."""
    seen: dict = {"register": [], "bind": [], "delete": None}

    async def run() -> bool:
        async with _client(_chunk_size_handler(seen, register_status=409)) as client:
            return await _run_loop_until_done(client, chunk_size_provider=lambda: 640)

    assert asyncio.run(run())
    assert seen["register"] == [640]
    assert seen["delete"] is None


def test_keep_registered_keeps_heartbeating_through_error_status():
    """Only a 409 ends the loop; a 5xx is retried on the next tick."""
    heartbeats = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            heartbeats.append(request.url.path)
            return httpx.Response(500, json={"error": "transient"})
        return httpx.Response(204)

    async def run() -> bool:
        async with _client(handler) as client:
            return await _run_loop_until_done(client)

    assert not asyncio.run(run())
    assert len(heartbeats) > 1
