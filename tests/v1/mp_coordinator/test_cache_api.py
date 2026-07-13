# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/cache/*`` REST API (warm-prefetch dispatch).

Quota writes, usage events, and status reads moved to the ``/quota`` group --
see ``test_quota_api.py``.
"""

# Third Party
from fastapi.testclient import TestClient
import httpx

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.utils.cache_utils import resolve_object_keys


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


# -- Pin / unpin (coordinator-side L2 pin) -----------------------------------


def _pin_client() -> TestClient:
    """A coordinator with a small chunk_size so short token sequences resolve."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0, eviction_check_interval=0.0, chunk_size=4
    )
    return TestClient(create_app(config))


def _pin_body(salt: str = "alice") -> dict:
    return {
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "cache_salt": salt,
    }


def _resolve(ctx, salt: str = "alice") -> list[ObjectKey]:
    """Resolve the pin body's keys the same way the handler will."""
    keys, _ = resolve_object_keys(
        ctx.token_hasher, "m", 1, [1, 2, 3, 4, 5, 6, 7, 8], salt
    )
    return keys


def test_pin_then_unpin_tracks_l2_eviction():
    """Pin excludes the resolved keys from L2 eviction; unpin restores them."""
    with _pin_client() as client:
        ctx = client.app.state.ctx
        keys = _resolve(ctx)
        assert keys  # 2 chunks x world_size 1

        # Arm allowlist enforcement (unquota'd salts are exempt until the
        # default limit is set), then track the keys in the L2 eviction LRU
        # with no quota (evict-all), so the plan would evict them unless
        # pinned.
        assert (
            client.put("/quota/config", json={"default_limit_gb": 0}).status_code == 200
        )
        for k in keys:
            ctx.usage_manager.record_stored(k, 1000)
            ctx.eviction_manager.on_store(k)
        assert ctx.eviction_manager.compute_eviction_plan()["alice"]

        resp = client.post("/cache/pins", json=_pin_body())
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "requested": 2,
            "affected": len(keys),
            "status": "pinned",
        }
        # Pinned: the keys drop out of the eviction plan.
        assert ctx.eviction_manager.compute_eviction_plan() == {}

        resp = client.request("DELETE", "/cache/pins", json=_pin_body())
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "requested": 2,
            "affected": len(keys),
            "status": "unpinned",
        }
        # Unpinned: the keys are eligible for eviction again.
        assert ctx.eviction_manager.compute_eviction_plan()["alice"]


def test_pin_short_sequence_is_noop():
    """A sub-chunk sequence resolves to no keys (affected 0)."""
    with _pin_client() as client:
        body = {
            "model_name": "m",
            "world_size": 1,
            "token_ids": [1, 2],
            "cache_salt": "",
        }
        resp = client.post("/cache/pins", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"requested": 0, "affected": 0, "status": "pinned"}


def test_pin_invalid_cache_salt_returns_400():
    """An invalid cache_salt (forbidden char) is a 400."""
    with _pin_client() as client:
        resp = client.post("/cache/pins", json=_pin_body(salt="bad@salt"))
        assert resp.status_code == 400
