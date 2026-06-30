# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/cache/*`` REST API (warm-prefetch dispatch).

Quota writes, usage events, and status reads moved to the ``/quota`` group --
see ``test_quota_api.py``.
"""

# Third Party
from fastapi.testclient import TestClient
import httpx

# First Party
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
