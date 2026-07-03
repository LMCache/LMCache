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

        # Track the keys in the L2 eviction LRU with no quota (evict-all), so the
        # plan would evict them unless pinned.
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


# -- Delete dispatch (L1 on the server, coordinator-managed pin-aware L2) -----

_DELETE_KEY = ObjectKey(
    chunk_hash=bytes.fromhex("01"),
    model_name="m",
    kv_rank=0,
    cache_salt="alice",
)


def _delete_body(
    instance_id: str, salt: str = "alice", tier: str = "all", force: bool = False
) -> dict:
    return {
        "instance_id": instance_id,
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4],
        "cache_salt": salt,
        "tier": tier,
        "force": force,
    }


def _mock_delete_server(l2_deletes: list) -> httpx.AsyncClient:
    """An outbound client emulating the MP server's delete API.

    ``POST /cache/delete`` deletes L1 and returns ``_DELETE_KEY`` (encoded);
    ``DELETE /cache/objects`` records the keys it was asked to remove into
    ``l2_deletes`` so tests can assert the coordinator's pin-aware filtering.
    """
    encoded = asdict(_DELETE_KEY.to_encoded_object_key())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cache/delete" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "requested": 1,
                    "deleted": 1,
                    "skipped": 0,
                    "resolved_keys": [encoded],
                    "status": "deleted",
                },
            )
        if request.url.path == "/cache/objects" and request.method == "DELETE":
            # Standard
            import json as _json

            body = _json.loads(request.content.decode())
            l2_deletes.append(body["keys"])
            return httpx.Response(
                200, json={"requested": len(body["keys"]), "ok": True}
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_delete_unknown_instance_returns_404():
    """Targeting an unregistered instance must 404 (before any dispatch)."""
    with _client() as client:
        resp = client.post("/cache/delete", json=_delete_body("does-not-exist"))
        assert resp.status_code == 404


def test_delete_all_tier_dispatches_l1_and_l2():
    """tier=all deletes L1 on the server and dispatches the resolved key to L2."""
    l2_deletes: list = []
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = _mock_delete_server(l2_deletes)

        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 200, resp.text
        # affected sums both tiers: 1 L1 key (node) + 1 L2 key (coordinator).
        assert resp.json() == {
            "instance_id": "mp-1",
            "requested": 1,
            "affected": 2,
            "skipped": 0,
            "status": "deleted",
        }
        # The (unpinned) resolved key was dispatched to the node's L2 delete.
        assert len(l2_deletes) == 1
        assert len(l2_deletes[0]) == 1


def test_delete_non_force_skips_l2_pinned_key():
    """Non-force delete must not remove an L2-pinned key at L2."""
    l2_deletes: list = []
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = _mock_delete_server(l2_deletes)
        ctx = client.app.state.ctx
        ctx.eviction_manager.pin([_DELETE_KEY])  # protect at L2

        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 200, resp.text
        # The pinned key was filtered out: no L2 delete dispatched.
        assert l2_deletes == []
        # ...the L2-pinned key is reported as skipped (not silently 0)...
        assert resp.json()["skipped"] == 1
        # ...and the pin survives (non-force does not drop it).
        assert ctx.eviction_manager.filter_unpinned([_DELETE_KEY]) == []


def test_delete_force_removes_and_drops_l2_pin():
    """Force delete removes even an L2-pinned key and purges the pin."""
    l2_deletes: list = []
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = _mock_delete_server(l2_deletes)
        ctx = client.app.state.ctx
        ctx.eviction_manager.pin([_DELETE_KEY])

        resp = client.post("/cache/delete", json=_delete_body("mp-1", force=True))
        assert resp.status_code == 200, resp.text
        # The pinned key was dispatched to L2 despite the pin...
        assert len(l2_deletes) == 1
        assert len(l2_deletes[0]) == 1
        # ...and the coordinator dropped the L2 pin.
        assert ctx.eviction_manager.filter_unpinned([_DELETE_KEY]) == [_DELETE_KEY]


def test_delete_l1_tier_does_not_touch_l2():
    """tier=l1 must not dispatch any L2 delete or touch the pin set."""
    l2_deletes: list = []
    with _client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.ctx.outbound_client = _mock_delete_server(l2_deletes)
        ctx = client.app.state.ctx
        ctx.eviction_manager.pin([_DELETE_KEY])

        resp = client.post("/cache/delete", json=_delete_body("mp-1", tier="l1"))
        assert resp.status_code == 200, resp.text
        assert l2_deletes == []
        assert ctx.eviction_manager.filter_unpinned([_DELETE_KEY]) == []


def test_delete_server_unreachable_returns_502():
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
        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 502
