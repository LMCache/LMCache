# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator warm-prefetch manager (submit + status proxy)."""

# Standard
from types import SimpleNamespace
import asyncio
import json as _json
import time

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.mp_coordinator.controllers import prefetch_manager
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.views.instance_registry import MPInstance


def _instance(instance_id: str, ip: str = "10.0.0.1", port: int = 8000) -> MPInstance:
    now = time.time()
    return MPInstance(
        instance_id=instance_id,
        ip=ip,
        http_port=port,
        registration_time=now,
        last_heartbeat_time=now,
    )


@pytest.mark.asyncio
async def test_submit_prefetch_posts_body_and_returns_reply():
    """submit_prefetch POSTs /cache/prefetches with the token body and returns the
    server's JSON reply verbatim."""
    mgr = PrefetchManager()
    target = _instance("mp-1", ip="10.0.0.7", port=8765)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["json"] = (request.read() or b"").decode()
        return httpx.Response(
            202, json={"request_id": "abc", "chunks": 2, "status": "submitted"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await mgr.submit_prefetch(
            target=target,
            http_client=client,
            model_name="m",
            world_size=2,
            token_ids=[1, 2, 3, 4],
            cache_salt="alice",
        )

    assert captured["method"] == "POST"
    assert captured["url"] == "http://10.0.0.7:8765/cache/prefetches"
    assert _json.loads(captured["json"]) == {
        "model_name": "m",
        "world_size": 2,
        "token_ids": [1, 2, 3, 4],
        "cache_salt": "alice",
    }
    assert result == {"request_id": "abc", "chunks": 2, "status": "submitted"}


@pytest.mark.asyncio
async def test_submit_prefetch_raises_on_http_error():
    """A non-2xx submit surfaces as an httpx error for the caller to map."""
    mgr = PrefetchManager()
    target = _instance("mp-1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPError):
            await mgr.submit_prefetch(
                target=target,
                http_client=client,
                model_name="m",
                world_size=1,
                token_ids=[1, 2],
                cache_salt="",
            )


@pytest.mark.asyncio
async def test_get_status_proxies_code_and_body():
    """get_status returns the server's (status_code, body) verbatim."""
    mgr = PrefetchManager()
    target = _instance("mp-1", ip="10.0.0.7", port=8765)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200, json={"status": "completed", "found_keys": 4, "total_keys": 4}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        code, body = await mgr.get_status(
            target=target, http_client=client, request_id="abc"
        )

    assert captured["url"] == "http://10.0.0.7:8765/cache/prefetches/abc"
    assert code == 200
    assert body == {"status": "completed", "found_keys": 4, "total_keys": 4}


@pytest.mark.asyncio
async def test_get_status_relays_404():
    """An unknown id on the server is relayed as a 404 (not raised)."""
    mgr = PrefetchManager()
    target = _instance("mp-1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "unknown"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        code, body = await mgr.get_status(
            target=target, http_client=client, request_id="x"
        )

    assert code == 404
    assert body == {"detail": "unknown"}


# -- Stats ---------------------------------------------------------------------


def _server(replies: dict[str, httpx.Response]):
    """A fake mp server answering submits and polls from ``replies``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return replies[request.method]

    return httpx.MockTransport(handler)


async def _submit(mgr: PrefetchManager, client: httpx.AsyncClient) -> dict:
    return await mgr.submit_prefetch(
        target=_instance("mp-1"),
        http_client=client,
        model_name="m",
        world_size=1,
        token_ids=[1, 2, 3, 4],
        cache_salt="",
    )


@pytest.mark.asyncio
async def test_stats_follow_a_prefetch_from_submit_to_completion():
    mgr = PrefetchManager()
    transport = _server(
        {
            "POST": httpx.Response(
                202, json={"request_id": "abc", "chunks": 2, "status": "submitted"}
            ),
            "GET": httpx.Response(
                200, json={"status": "completed", "found_keys": 3, "total_keys": 4}
            ),
        }
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await _submit(mgr, client)
        in_flight = mgr.stats().requests_in_flight
        await mgr.get_status(
            target=_instance("mp-1"), http_client=client, request_id="abc"
        )

    stats = mgr.stats()
    assert in_flight == {"mp-1": 1}
    assert stats.requests_in_flight == {}
    assert stats.requests_submitted["submitted"] == 1
    assert stats.chunks_requested_per_request.sum == 2
    assert stats.submit_duration_seconds.count == 1
    assert stats.status_polls["completed"] == 1
    assert stats.completion_duration_seconds.count == 1
    assert (stats.keys_requested, stats.keys_found) == (4, 3)


@pytest.mark.asyncio
async def test_stats_count_noop_failed_and_unknown_server_submits():
    mgr = PrefetchManager()
    noop = _server({"POST": httpx.Response(202, json={"chunks": 0, "status": "noop"})})
    failing = _server({"POST": httpx.Response(500, json={})})
    async with httpx.AsyncClient(transport=noop) as client:
        await _submit(mgr, client)
    async with httpx.AsyncClient(transport=failing) as client:
        with pytest.raises(httpx.HTTPError):
            await _submit(mgr, client)
    mgr.record_unknown_server()

    stats = mgr.stats()
    assert stats.requests_submitted == {
        "submitted": 0,
        "noop": 1,
        "unknown_server": 1,
        "server_error": 1,
    }
    assert stats.requests_in_flight == {}


@pytest.mark.asyncio
async def test_a_poll_the_server_does_not_know_stops_tracking_the_prefetch():
    mgr = PrefetchManager()
    transport = _server(
        {
            "POST": httpx.Response(
                202, json={"request_id": "abc", "chunks": 1, "status": "submitted"}
            ),
            "GET": httpx.Response(404, json={"detail": "unknown"}),
        }
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await _submit(mgr, client)
        await mgr.get_status(
            target=_instance("mp-1"), http_client=client, request_id="abc"
        )

    stats = mgr.stats()
    assert stats.status_polls["unknown_request"] == 1
    assert stats.requests_in_flight == {}
    assert stats.completion_duration_seconds.count == 0


@pytest.mark.asyncio
async def test_a_prefetch_never_polled_to_completion_is_counted_abandoned():
    mgr = PrefetchManager(abandon_after_seconds=0.01)
    transport = _server(
        {
            "POST": httpx.Response(
                202, json={"request_id": "abc", "chunks": 1, "status": "submitted"}
            ),
        }
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await _submit(mgr, client)
    await asyncio.sleep(0.05)

    stats = mgr.stats()
    assert stats.requests_abandoned == 1
    assert stats.requests_in_flight == {}


@pytest.mark.asyncio
async def test_a_submit_forgets_abandoned_prefetches_without_a_scrape(
    monkeypatch: pytest.MonkeyPatch,
):
    """With nothing scraping metrics, old prefetches must still be dropped,
    or the table grows without limit."""
    clock = [0.0]
    # Only the manager's clock: the event loop keeps the real one.
    monkeypatch.setattr(
        prefetch_manager, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    mgr = PrefetchManager(abandon_after_seconds=600.0)
    transport = _server(
        {
            "POST": httpx.Response(
                202, json={"request_id": "abc", "chunks": 1, "status": "submitted"}
            ),
        }
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await _submit(mgr, client)
        clock[0] = 700.0
        await _submit(mgr, client)  # forgets the first, now 700 s old
    clock[0] = 0.0  # so stats() itself forgets nothing more

    assert mgr.stats().requests_abandoned == 1


@pytest.mark.asyncio
async def test_a_submitted_reply_without_a_request_id_is_counted_not_tracked():
    mgr = PrefetchManager()
    transport = _server(
        {"POST": httpx.Response(202, json={"chunks": 1, "status": "submitted"})}
    )
    async with httpx.AsyncClient(transport=transport) as client:
        reply = await _submit(mgr, client)

    stats = mgr.stats()
    assert reply == {"chunks": 1, "status": "submitted"}
    assert stats.requests_submitted["submitted"] == 1
    assert stats.requests_in_flight == {}


@pytest.mark.asyncio
async def test_a_reply_that_is_not_json_is_counted_as_a_server_error():
    mgr = PrefetchManager()
    transport = _server({"POST": httpx.Response(202, text="not json")})
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError):
            await _submit(mgr, client)

    assert mgr.stats().requests_submitted["server_error"] == 1
