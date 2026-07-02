# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/cache/*`` REST API (warm-prefetch dispatch).

Quota writes, usage events, and status reads moved to the ``/quota`` group --
see ``test_quota_api.py``.
"""

# Standard
from dataclasses import asdict

# Third Party
from fastapi.testclient import TestClient
import httpx

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _client() -> TestClient:
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    return TestClient(create_app(config))


# -- Prefetch dispatch -------------------------------------------------------


def _prefetch_body(instance_id: str, salt: str = "alice") -> dict:
    return {
        "instance_id": instance_id,
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4],
        "cache_salt": salt,
    }


def _mock_mp_server() -> httpx.AsyncClient:
    """An outbound client that emulates the target MP server's prefetch API."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/cache/prefetches":
            return httpx.Response(
                202, json={"request_id": "abc", "chunks": 2, "status": "submitted"}
            )
        if request.method == "GET" and request.url.path == "/cache/prefetches/abc":
            return httpx.Response(
                200, json={"status": "completed", "found_keys": 2, "total_keys": 2}
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_prefetch_unknown_instance_returns_404():
    """Targeting an unregistered instance must 404 (before any dispatch)."""
    with _client() as client:
        resp = client.post("/cache/prefetches", json=_prefetch_body("does-not-exist"))
        assert resp.status_code == 404


def test_prefetch_submit_then_status_proxy():
    """A registered target: submit relays the server's request_id, and the
    status GET proxies the server's completion body."""
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        # Replace the lifespan's real outbound client with a mock MP server.
        client.app.state.ctx.outbound_client = _mock_mp_server()

        resp = client.post("/cache/prefetches", json=_prefetch_body("mp-1"))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "instance_id": "mp-1",
            "request_id": "abc",
            "chunks": 2,
            "status": "submitted",
        }

        status = client.get("/cache/prefetches/mp-1/abc")
        assert status.status_code == 200, status.text
        assert status.json() == {
            "status": "completed",
            "found_keys": 2,
            "total_keys": 2,
        }


def test_prefetch_status_unknown_instance_returns_404():
    """Status for an unregistered instance must 404."""
    with _client() as client:
        resp = client.get("/cache/prefetches/does-not-exist/abc")
        assert resp.status_code == 404


# -- Pin / unpin dispatch (L1 on the server, L2 on the coordinator) ----------

_PIN_KEY = ObjectKey(
    chunk_hash=bytes.fromhex("01"),
    model_name="m",
    kv_rank=0,
    cache_salt="alice",
)


def _pin_body(instance_id: str, salt: str = "alice", tier: str = "all") -> dict:
    return {
        "instance_id": instance_id,
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4],
        "cache_salt": salt,
        "tier": tier,
    }


def _mock_pin_server() -> httpx.AsyncClient:
    """An outbound client emulating the MP server's pin API. It pins L1 and
    returns the single ``_PIN_KEY`` encoded so the coordinator can pin L2."""
    encoded = asdict(_PIN_KEY.to_encoded_object_key())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cache/pins" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "requested": 1,
                    "pinned": 1,
                    "resolved_keys": [encoded],
                    "status": "pinned",
                },
            )
        if request.url.path == "/cache/pins" and request.method == "DELETE":
            return httpx.Response(
                200,
                json={
                    "requested": 1,
                    "unpinned": 1,
                    "resolved_keys": [encoded],
                    "status": "unpinned",
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_pin_unknown_instance_returns_404():
    """Targeting an unregistered instance must 404 (before any dispatch)."""
    with _client() as client:
        resp = client.post("/cache/pins", json=_pin_body("does-not-exist"))
        assert resp.status_code == 404


def test_pin_then_unpin_dispatch_and_track_l2():
    """Pin relays L1 counts and excludes the key from L2 eviction; unpin restores it."""
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = _mock_pin_server()

        # Track the key in the L2 eviction LRU with no quota (evict-all), so the
        # plan would evict it unless it is pinned.
        ctx = client.app.state.ctx
        ctx.usage_manager.record_stored(_PIN_KEY, 1000)
        ctx.eviction_manager.on_store(_PIN_KEY)
        assert ctx.eviction_manager.compute_eviction_plan()["alice"] == [_PIN_KEY]

        resp = client.post("/cache/pins", json=_pin_body("mp-1"))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "instance_id": "mp-1",
            "requested": 1,
            "affected": 1,
            "status": "pinned",
        }
        # Coordinator pinned L2: the key drops out of the eviction plan.
        assert ctx.eviction_manager.compute_eviction_plan() == {}

        resp = client.request("DELETE", "/cache/pins", json=_pin_body("mp-1"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "unpinned"
        # Unpin releases L2: the key is eligible again.
        assert ctx.eviction_manager.compute_eviction_plan()["alice"] == [_PIN_KEY]


def test_pin_l1_tier_does_not_track_l2():
    """tier=l1 must not touch the coordinator's L2 eviction set."""
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = _mock_pin_server()
        ctx = client.app.state.ctx
        ctx.usage_manager.record_stored(_PIN_KEY, 1000)
        ctx.eviction_manager.on_store(_PIN_KEY)
        assert ctx.eviction_manager.compute_eviction_plan()["alice"] == [_PIN_KEY]

        resp = client.post("/cache/pins", json=_pin_body("mp-1", tier="l1"))
        assert resp.status_code == 200, resp.text
        # L2 untouched: the key stays in the eviction plan.
        assert ctx.eviction_manager.compute_eviction_plan()["alice"] == [_PIN_KEY]


def test_pin_server_unreachable_returns_502():
    """A transport error talking to the MP server surfaces as 502."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        resp = client.post("/cache/pins", json=_pin_body("mp-1"))
        assert resp.status_code == 502
