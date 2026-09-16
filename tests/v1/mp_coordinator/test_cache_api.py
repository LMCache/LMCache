# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/cache/*`` REST API (warm-prefetch dispatch,
pins, delete, and move).

Quota writes, usage events, and status reads moved to the ``/quota`` group --
see ``test_quota_api.py``.
"""

# Standard
import json
import time

# Third Party
from fastapi.testclient import TestClient
import httpx

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.controllers.move_controller import MoveController
from lmcache.v1.mp_coordinator.http_apis.dependencies import CoordinatorContext
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys


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
        client.app.state.outbound_client = _mock_mp_server()

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
        eviction = ctx.controllers.get(FleetEvictionController)
        keys = _resolve(ctx)
        assert keys  # 2 chunks x world_size 1

        # Arm allowlist enforcement (unquota'd salts are exempt until the
        # default limit is set), then track the keys in the L2 eviction LRU
        # with no quota (evict-all), so the plan would evict them unless
        # pinned.
        assert (
            client.put("/quota/config", json={"default_limit_gb": 0}).status_code == 200
        )
        for seq, k in enumerate(keys, start=1):
            ctx.event_gate.ingest(
                CacheEventBatch(
                    instance_id="mp-1",
                    incarnation=1,
                    seq=seq,
                    event_type=CacheEventType.STORE,
                    tier=Tier.L2,
                    backend="fs",
                    entries=[
                        CacheEventEntry(key=k.to_encoded_object_key(), size_bytes=1000)
                    ],
                )
            )
        assert eviction.compute_eviction_plan()["alice"]

        resp = client.post("/cache/pins", json=_pin_body())
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "requested": 2,
            "affected": len(keys),
            "status": "pinned",
        }
        # Pinned: the keys drop out of the eviction plan.
        assert eviction.compute_eviction_plan() == {}

        resp = client.request("DELETE", "/cache/pins", json=_pin_body())
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "requested": 2,
            "affected": len(keys),
            "status": "unpinned",
        }
        # Unpinned: the keys are eligible for eviction again.
        assert eviction.compute_eviction_plan()["alice"]


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


def test_list_pins_empty_table():
    """No pins yet: an empty page with total 0, not a 404."""
    with _pin_client() as client:
        resp = client.get("/cache/pins")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"total": 0, "pins": []}


def test_list_pins_shows_pinned_keys_with_counts():
    """GET lists every key the POSTs resolved to, with the wire form of the
    key and a pin count that tracks repeated POSTs and DELETEs."""
    with _pin_client() as client:
        ctx = client.app.state.ctx
        keys = _resolve(ctx)
        assert client.post("/cache/pins", json=_pin_body()).status_code == 200
        assert client.post("/cache/pins", json=_pin_body()).status_code == 200

        resp = client.get("/cache/pins")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == len(keys)
        assert body["pins"] == [
            {
                "key": {
                    "chunk_hash_hex": k.chunk_hash.hex(),
                    "model_name": "m",
                    "kv_rank": k.kv_rank,
                    "object_group_id": 0,
                    "cache_salt": "alice",
                },
                "pin_count": 2,
            }
            for k in keys
        ]

        client.request("DELETE", "/cache/pins", json=_pin_body())
        counts = {p["pin_count"] for p in client.get("/cache/pins").json()["pins"]}
        assert counts == {1}

        client.request("DELETE", "/cache/pins", json=_pin_body())
        assert client.get("/cache/pins").json() == {"total": 0, "pins": []}


def test_list_pins_filters_by_cache_salt_and_model():
    """cache_salt and model_name narrow the listing; total follows the filter."""
    with _pin_client() as client:
        ctx = client.app.state.ctx
        alice = _resolve(ctx, "alice")
        bob = _resolve(ctx, "bob")
        client.post("/cache/pins", json=_pin_body("alice"))
        client.post("/cache/pins", json=_pin_body("bob"))

        body = client.get("/cache/pins", params={"cache_salt": "bob"}).json()
        assert body["total"] == len(bob)
        assert {p["key"]["cache_salt"] for p in body["pins"]} == {"bob"}

        body = client.get("/cache/pins", params={"model_name": "m"}).json()
        assert body["total"] == len(alice) + len(bob)

        body = client.get("/cache/pins", params={"model_name": "other"}).json()
        assert body == {"total": 0, "pins": []}


def test_list_pins_pages_and_validates_bounds():
    """offset/limit page the filtered set; out-of-range values are a 422."""
    with _pin_client() as client:
        ctx = client.app.state.ctx
        alice = _resolve(ctx, "alice")
        bob = _resolve(ctx, "bob")
        client.post("/cache/pins", json=_pin_body("alice"))
        client.post("/cache/pins", json=_pin_body("bob"))
        expected = [k.chunk_hash.hex() for k in alice + bob]

        first = client.get("/cache/pins", params={"offset": 0, "limit": 1}).json()
        second = client.get("/cache/pins", params={"offset": 1, "limit": 10}).json()
        assert first["total"] == second["total"] == len(expected)
        listed = [p["key"]["chunk_hash_hex"] for p in first["pins"] + second["pins"]]
        assert listed == expected

        assert client.get("/cache/pins", params={"offset": -1}).status_code == 422
        assert client.get("/cache/pins", params={"limit": 0}).status_code == 422
        assert client.get("/cache/pins", params={"limit": 10001}).status_code == 422


# -- Delete dispatch (coordinator resolves; key-addressed L1 + L2 to the node) --


def _delete_client() -> TestClient:
    """A coordinator with a small chunk_size so short token sequences resolve."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0, eviction_check_interval=0.0, chunk_size=4
    )
    return TestClient(create_app(config))


def _delete_body(
    instance_id: str, salt: str = "alice", tier: str = "all", force: bool = False
) -> dict:
    return {
        "instance_id": instance_id,
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "cache_salt": salt,
        "tier": tier,
        "force": force,
    }


def _resolve_delete(ctx, salt: str = "alice") -> list[ObjectKey]:
    """Resolve the delete body's keys the same way the handler will."""
    keys, _ = resolve_object_keys(
        ctx.token_hasher, "m", 1, [1, 2, 3, 4, 5, 6, 7, 8], salt
    )
    return keys


def _mock_delete_server(deletes: list) -> httpx.AsyncClient:
    """Emulate the node's unified key-addressed delete (``DELETE /cache/objects``).

    Records each request body ``{keys, tier, force}`` so tests can assert the
    coordinator's single-call dispatch and pin filtering, and reports the keys
    deleted: both tiers for ``all`` (n L1 + n L2), one tier otherwise.
    """
    # Standard
    import json as _json

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cache/objects" and request.method == "DELETE":
            body = _json.loads(request.content.decode())
            deletes.append(body)
            n = len(body["keys"])
            deleted = n * (2 if body.get("tier") == "all" else 1)
            return httpx.Response(
                200, json={"deleted": deleted, "skipped": 0, "ok": True}
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_delete_unknown_instance_returns_404():
    """Targeting an unregistered instance must 404 (before any dispatch)."""
    with _delete_client() as client:
        resp = client.post("/cache/delete", json=_delete_body("does-not-exist"))
        assert resp.status_code == 404


def test_delete_short_sequence_is_noop():
    """A sub-chunk sequence resolves to nothing and dispatches no delete."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.outbound_client = _mock_delete_server(deletes)

        body = _delete_body("mp-1")
        body["token_ids"] = [1, 2]  # shorter than one chunk (chunk_size=4)
        resp = client.post("/cache/delete", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "instance_id": "mp-1",
            "requested": 0,
            "affected": 0,
            "skipped": 0,
            "status": "noop",
        }
        assert deletes == []


def test_delete_invalid_cache_salt_returns_400():
    """A bad cache_salt fails resolution on the coordinator with a 400."""
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        resp = client.post("/cache/delete", json=_delete_body("mp-1", salt="bad@salt"))
        assert resp.status_code == 400


def test_delete_all_tier_single_call_both_tiers():
    """tier=all issues one DELETE /cache/objects that removes L1 and L2."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        ctx = client.app.state.ctx
        client.app.state.outbound_client = _mock_delete_server(deletes)
        n = len(_resolve_delete(ctx))
        assert n >= 1

        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["requested"] == 2  # 2 chunks (chunk_size=4, 8 tokens)
        assert body["affected"] == 2 * n  # n L1 + n L2
        assert body["skipped"] == 0
        # Exactly one node call, carrying tier=all and every resolved key.
        assert len(deletes) == 1
        assert deletes[0]["tier"] == "all"
        assert deletes[0]["force"] is False
        assert len(deletes[0]["keys"]) == n


def test_delete_non_force_holds_back_l2_pinned_key():
    """Non-force delete drops an L2-pinned key from the (single) delete set."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        ctx = client.app.state.ctx
        eviction = ctx.controllers.get(FleetEvictionController)
        client.app.state.outbound_client = _mock_delete_server(deletes)
        keys = _resolve_delete(ctx)
        eviction.pin([keys[0]])  # protect one key at L2

        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 200, resp.text
        # The pinned key is reported skipped and never dispatched (retained).
        assert resp.json()["skipped"] == 1
        assert len(deletes) == 1
        assert len(deletes[0]["keys"]) == len(keys) - 1
        # The pin survives (non-force does not drop it).
        assert eviction.filter_unpinned([keys[0]]) == []


def test_delete_force_removes_and_drops_l2_pin():
    """Force delete removes even an L2-pinned key and purges the pin."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        ctx = client.app.state.ctx
        eviction = ctx.controllers.get(FleetEvictionController)
        client.app.state.outbound_client = _mock_delete_server(deletes)
        keys = _resolve_delete(ctx)
        eviction.pin([keys[0]])

        resp = client.post("/cache/delete", json=_delete_body("mp-1", force=True))
        assert resp.status_code == 200, resp.text
        # Force dispatched every key despite the pin...
        assert len(deletes) == 1
        assert deletes[0]["force"] is True
        assert len(deletes[0]["keys"]) == len(keys)
        assert resp.json()["skipped"] == 0
        # ...and the coordinator dropped the L2 pin.
        assert eviction.filter_unpinned([keys[0]]) == [keys[0]]


def test_delete_l1_tier_ignores_l2_pins():
    """tier=l1 dispatches with tier=l1 and does not filter or drop L2 pins."""
    deletes: list = []
    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        ctx = client.app.state.ctx
        eviction = ctx.controllers.get(FleetEvictionController)
        client.app.state.outbound_client = _mock_delete_server(deletes)
        keys = _resolve_delete(ctx)
        eviction.pin([keys[0]])

        resp = client.post("/cache/delete", json=_delete_body("mp-1", tier="l1"))
        assert resp.status_code == 200, resp.text
        # tier=l1 does not consult L2 pins: every key is dispatched.
        assert len(deletes) == 1
        assert deletes[0]["tier"] == "l1"
        assert len(deletes[0]["keys"]) == len(keys)
        assert resp.json()["affected"] == len(keys)  # L1 only
        assert eviction.filter_unpinned([keys[0]]) == []  # pin untouched


def test_delete_server_unreachable_returns_502():
    """A transport error talking to the MP server surfaces as 502."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with _delete_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.app.state.outbound_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        resp = client.post("/cache/delete", json=_delete_body("mp-1"))
        assert resp.status_code == 502


# -- Move dispatch (target warm prefetch, then source L1 delete) --------------

_MOVE_SOURCE_IP = "10.0.0.1"
_MOVE_TARGET_IP = "10.0.0.2"


def _move_client() -> TestClient:
    """A coordinator with a small chunk_size and a fast move poll."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        chunk_size=4,
        extra_config={"move_poll_interval_s": 0.005},
    )
    return TestClient(create_app(config))


def _register_pair(client: TestClient) -> None:
    for instance_id, ip in (("mp-src", _MOVE_SOURCE_IP), ("mp-dst", _MOVE_TARGET_IP)):
        resp = client.post(
            "/instances", json={"instance_id": instance_id, "ip": ip, "http_port": 8080}
        )
        assert resp.status_code == 200, resp.text


def _move_body(
    source: str = "mp-src", target: str = "mp-dst", salt: str = "alice", **extra: object
) -> dict[str, object]:
    body: dict[str, object] = {
        "source_instance_id": source,
        "target_instance_id": target,
        "model_name": "m",
        "world_size": 1,
        "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "cache_salt": salt,
    }
    body.update(extra)
    return body


def _resolve_move(
    ctx: CoordinatorContext, salt: str = "alice", world_size: int = 1
) -> list[ObjectKey]:
    keys, _ = resolve_object_keys(
        ctx.token_hasher, "m", world_size, [1, 2, 3, 4, 5, 6, 7, 8], salt
    )
    return keys


def _mock_move_fleet(
    calls: dict[str, list[dict[str, object]]],
    missing: list[int],
    submit_status: int = 202,
    total: int = 2,
    submit_raw: bytes | None = None,
) -> httpx.AsyncClient:
    """Both MP servers behind one transport: the target accepts a prefetch and
    reports it complete with ``missing`` of ``total`` keys; the source
    records deletes. ``submit_raw`` replaces the submit reply body verbatim.

    ``calls`` collects the request bodies seen, under ``"submits"`` (target
    prefetch submits) and ``"deletes"`` (source deletes).
    """
    calls.setdefault("submits", [])
    calls.setdefault("deletes", [])

    def handler(request: httpx.Request) -> httpx.Response:
        host, path, method = request.url.host, request.url.path, request.method
        if host == _MOVE_TARGET_IP and method == "POST" and path == "/cache/prefetches":
            calls["submits"].append(json.loads(request.content.decode()))
            if submit_raw is not None:
                return httpx.Response(submit_status, content=submit_raw)
            return httpx.Response(
                submit_status,
                json={"request_id": "rid", "chunks": 2, "status": "submitted"},
            )
        if (
            host == _MOVE_TARGET_IP
            and method == "GET"
            and path == "/cache/prefetches/rid"
        ):
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "found_keys": total - len(missing),
                    "total_keys": total,
                    "missing_key_indices": missing,
                },
            )
        if host == _MOVE_SOURCE_IP and method == "DELETE" and path == "/cache/objects":
            body = json.loads(request.content.decode())
            calls["deletes"].append(body)
            return httpx.Response(
                200, json={"deleted": len(body["keys"]), "skipped": 0, "ok": True}
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _poll_move(
    client: TestClient, move_id: str, timeout: float = 3.0
) -> dict[str, object]:
    """Poll the move until it leaves ``pending``; return the terminal body."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/cache/moves/{move_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] != "pending":
            return body
        time.sleep(0.005)
    raise AssertionError("move did not settle in time")


def test_move_unknown_instance_returns_404():
    """Either end unregistered must 404, before any dispatch."""
    with _move_client() as client:
        client.post(
            "/instances",
            json={"instance_id": "mp-src", "ip": _MOVE_SOURCE_IP, "http_port": 8080},
        )
        assert client.post("/cache/moves", json=_move_body()).status_code == 404
        assert (
            client.post(
                "/cache/moves", json=_move_body(source="ghost", target="mp-src")
            ).status_code
            == 404
        )


def test_move_same_instance_returns_400():
    with _move_client() as client:
        _register_pair(client)
        resp = client.post("/cache/moves", json=_move_body(target="mp-src"))
        assert resp.status_code == 400


def test_move_unsupported_direction_returns_400():
    with _move_client() as client:
        _register_pair(client)
        resp = client.post("/cache/moves", json=_move_body(target_tier="l2"))
        assert resp.status_code == 400
        assert "l1" in resp.json()["detail"]


def test_move_invalid_cache_salt_returns_400():
    with _move_client() as client:
        _register_pair(client)
        resp = client.post("/cache/moves", json=_move_body(salt="bad@salt"))
        assert resp.status_code == 400


def test_move_short_sequence_is_noop():
    """A sub-chunk sequence resolves to nothing: no move, no outbound call."""
    calls: dict[str, list[dict[str, object]]] = {}
    with _move_client() as client:
        _register_pair(client)
        client.app.state.outbound_client = _mock_move_fleet(calls, missing=[])
        resp = client.post("/cache/moves", json=_move_body(token_ids=[1, 2]))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "move_id": "",
            "source_instance_id": "mp-src",
            "target_instance_id": "mp-dst",
            "requested": 0,
            "status": "noop",
        }
        assert calls["submits"] == []


def test_move_target_rejecting_submit_returns_502():
    calls: dict[str, list[dict[str, object]]] = {}
    with _move_client() as client:
        _register_pair(client)
        client.app.state.outbound_client = _mock_move_fleet(
            calls, missing=[], submit_status=503
        )
        resp = client.post("/cache/moves", json=_move_body())
        assert resp.status_code == 502
        assert client.get("/cache/moves/anything").status_code == 404


def test_move_target_answering_submit_with_non_json_returns_502():
    """A 200 from the target whose body is not a submit reply is an upstream
    error, not a coordinator crash, and records no move."""
    calls: dict[str, list[dict[str, object]]] = {}
    with _move_client() as client:
        _register_pair(client)
        client.app.state.outbound_client = _mock_move_fleet(
            calls, missing=[], submit_status=200, submit_raw=b"not json"
        )
        resp = client.post("/cache/moves", json=_move_body())
        assert resp.status_code == 502
        assert "unusable" in resp.json()["detail"]
        assert client.app.state.ctx.controllers.get(MoveController).in_flight == 0


def test_move_rejects_unknown_fields_before_dispatch():
    """A misspelt field is a 422, not a silently defaulted move: with
    ``keep_soruce`` ignored the source would have been deleted."""
    calls: dict[str, list[dict[str, object]]] = {}
    with _move_client() as client:
        _register_pair(client)
        client.app.state.outbound_client = _mock_move_fleet(calls, missing=[])
        resp = client.post("/cache/moves", json=_move_body(keep_soruce=True))
        assert resp.status_code == 422
        assert "keep_soruce" in resp.text
        assert calls["submits"] == []
        assert calls["deletes"] == []


def test_move_submits_then_status_reports_transfer_and_source_delete():
    """Submit relays a move_id; the coordinator drives the target's prefetch
    and deletes from the source exactly the keys the target loaded; the
    terminal status stays readable afterwards."""
    calls: dict[str, list[dict[str, object]]] = {}
    with _move_client() as client:
        _register_pair(client)
        ctx = client.app.state.ctx
        client.app.state.outbound_client = _mock_move_fleet(calls, missing=[1])
        keys = _resolve_move(ctx)
        assert len(keys) == 2  # 2 chunks (chunk_size=4, 8 tokens) x world_size 1

        resp = client.post("/cache/moves", json=_move_body())
        assert resp.status_code == 200, resp.text
        submitted = resp.json()
        assert submitted["status"] == "submitted"
        assert submitted["requested"] == 2
        move_id = submitted["move_id"]
        assert move_id

        body = _poll_move(client, move_id)
        assert body == {
            "move_id": move_id,
            "source_instance_id": "mp-src",
            "target_instance_id": "mp-dst",
            "status": "completed",
            "phase": "delete",
            "requested": 2,
            "loaded": 1,
            "missing": 1,
            "deleted": 1,
            "skipped": 0,
            "error": "",
        }
        # The target got the tokens verbatim; the source got one L1 delete
        # holding only the key the target loaded (position 0).
        assert calls["submits"] == [
            {
                "model_name": "m",
                "world_size": 1,
                "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "cache_salt": "alice",
            }
        ]
        assert len(calls["deletes"]) == 1
        assert calls["deletes"][0]["tier"] == "l1"
        assert calls["deletes"][0]["force"] is False
        assert [k["chunk_hash_hex"] for k in calls["deletes"][0]["keys"]] == [
            keys[0].chunk_hash.hex()
        ]
        # The terminal status is retained: a second read sees the same body.
        again = client.get(f"/cache/moves/{move_id}")
        assert again.status_code == 200
        assert again.json() == body


def test_move_world_size_two_counts_per_rank_keys():
    """``requested`` counts chunks; the other counts are per-rank keys in
    chunk-major order, so a sparse ``missing_key_indices`` maps to exactly
    the right keys on the source."""
    calls: dict[str, list[dict[str, object]]] = {}
    with _move_client() as client:
        _register_pair(client)
        ctx = client.app.state.ctx
        # 2 chunks x 2 ranks = 4 keys; the target loaded positions 0 and 3.
        client.app.state.outbound_client = _mock_move_fleet(
            calls, missing=[1, 2], total=4
        )
        keys = _resolve_move(ctx, world_size=2)
        assert len(keys) == 4

        resp = client.post("/cache/moves", json=_move_body(world_size=2))
        assert resp.status_code == 200, resp.text
        assert resp.json()["requested"] == 2
        body = _poll_move(client, resp.json()["move_id"])
        assert body["status"] == "completed"
        assert (body["requested"], body["loaded"], body["missing"]) == (2, 2, 2)
        assert body["deleted"] == 2
        assert [k["chunk_hash_hex"] for k in calls["deletes"][0]["keys"]] == [
            keys[0].chunk_hash.hex(),
            keys[3].chunk_hash.hex(),
        ]
        assert [k["kv_rank"] for k in calls["deletes"][0]["keys"]] == [
            keys[0].kv_rank,
            keys[3].kv_rank,
        ]


def test_move_keep_source_skips_the_source_delete():
    calls: dict[str, list[dict[str, object]]] = {}
    with _move_client() as client:
        _register_pair(client)
        client.app.state.outbound_client = _mock_move_fleet(calls, missing=[])
        move_id = client.post("/cache/moves", json=_move_body(keep_source=True)).json()[
            "move_id"
        ]
        body = _poll_move(client, move_id)
        assert body["status"] == "completed"
        assert (body["loaded"], body["deleted"]) == (2, 0)
        assert calls["deletes"] == []


def test_move_status_unknown_returns_404():
    with _move_client() as client:
        assert client.get("/cache/moves/does-not-exist").status_code == 404
