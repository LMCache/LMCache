# SPDX-License-Identifier: Apache-2.0
"""Tests for coordinator runtime-policy proxy and fleet fan-out APIs."""

# Standard
import json

# Third Party
from fastapi.testclient import TestClient
import httpx

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _client() -> TestClient:
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        enable_startup_resync=False,
    )
    return TestClient(create_app(config))


def _register(client: TestClient, *instance_ids: str) -> None:
    for index, instance_id in enumerate(instance_ids, start=1):
        response = client.post(
            "/instances",
            json={
                "instance_id": instance_id,
                "ip": "10.0.0.1",
                "http_port": 8000 + index,
            },
        )
        assert response.status_code == 200, response.text


def _update() -> dict:
    return {
        "store_policy": "lru",
        "l1_eviction": {"tunables": {"eviction_ratio": 0.2}},
    }


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_direct_policy_proxy_preserves_node_response() -> None:
    """Direct GET/PATCH/validate calls retain target status and payload."""
    calls: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, payload))
        if request.method == "GET":
            return httpx.Response(200, json={"version": 3, "store_policy": "lru"})
        if request.method == "POST":
            return httpx.Response(409, json={"error": "version_conflict"})
        return httpx.Response(200, json={"status": "updated", "version": 4})

    with _client() as client:
        _register(client, "node-a")
        client.app.state.ctx.outbound_client = _mock_client(handler)

        assert client.get("/instances/node-a/config/policies").json()["version"] == 3
        conflict = client.post(
            "/instances/node-a/config/policies/validate",
            json={"expected_version": 2, **_update()},
        )
        assert conflict.status_code == 409
        assert conflict.json() == {"error": "version_conflict"}
        patched = client.patch("/instances/node-a/config/policies", json=_update())
        assert patched.status_code == 200
        assert patched.json()["version"] == 4

    assert calls[0][0:2] == ("GET", "/config/policies")
    validation_body = calls[1][2]
    patch_body = calls[2][2]
    assert validation_body is not None
    assert patch_body is not None
    assert validation_body["expected_version"] == 2
    assert patch_body["store_policy"] == "lru"


def test_direct_policy_proxy_maps_unreachable_target_to_502() -> None:
    """A coordinator transport failure is distinct from a node response."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client() as client:
        _register(client, "node-a")
        client.app.state.ctx.outbound_client = _mock_client(handler)
        response = client.get("/instances/node-a/config/policies")
        assert response.status_code == 502
        assert "connection refused" in response.json()["detail"]


def test_direct_policy_proxy_unknown_instance_returns_404() -> None:
    with _client() as client:
        response = client.get("/instances/missing/config/policies")
        assert response.status_code == 404


def test_fleet_validate_is_all_target_and_does_not_apply() -> None:
    """Validation fans out to all nodes and never sends PATCH requests."""
    calls: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append((request.method, request.url.path, payload))
        return httpx.Response(
            200,
            json={
                "status": "valid",
                "version": {"node-a": 4, "node-b": 7}.get(
                    "node-a" if request.url.port == 8001 else "node-b", 4
                ),
            },
        )

    with _client() as client:
        _register(client, "node-a", "node-b")
        client.app.state.ctx.outbound_client = _mock_client(handler)
        response = client.post(
            "/fleet/config/policies/validate", json={"update": _update()}
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "valid"
        assert len(response.json()["results"]) == 2

    assert [method for method, _, _ in calls] == ["POST", "POST"]


def test_fleet_patch_uses_validation_barrier_and_per_node_versions() -> None:
    """A successful validation pass fences each concurrent apply call."""
    calls: list[tuple[str, str, dict]] = []
    versions = {8001: 4, 8002: 7}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append((request.method, request.url.path, payload))
        if request.method == "POST":
            return httpx.Response(
                200, json={"status": "valid", "version": versions[request.url.port]}
            )
        return httpx.Response(
            200, json={"status": "updated", "version": payload["expected_version"] + 1}
        )

    with _client() as client:
        _register(client, "node-a", "node-b")
        client.app.state.ctx.outbound_client = _mock_client(handler)
        response = client.patch("/fleet/config/policies", json={"update": _update()})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "updated"

    patch_bodies = {
        body["expected_version"] for method, _, body in calls if method == "PATCH"
    }
    assert patch_bodies == {4, 7}
    assert len([method for method, _, _ in calls if method == "PATCH"]) == 2


def test_fleet_patch_forwards_explicit_expected_versions() -> None:
    """Caller-provided versions override the versions observed at validation."""
    calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append((request.method, payload))
        if request.method == "POST":
            return httpx.Response(200, json={"status": "valid", "version": 20})
        return httpx.Response(200, json={"status": "updated", "version": 21})

    with _client() as client:
        _register(client, "node-a", "node-b")
        client.app.state.ctx.outbound_client = _mock_client(handler)
        response = client.patch(
            "/fleet/config/policies",
            json={
                "update": _update(),
                "expected_versions": {"node-a": 8, "node-b": 9},
            },
        )
        assert response.status_code == 200

    assert {
        payload["expected_version"] for method, payload in calls if method == "PATCH"
    } == {8, 9}


def test_fleet_validation_failure_prevents_all_apply_calls() -> None:
    """A rejected target stops the fleet before any PATCH is sent."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "POST" and request.url.port == 8002:
            return httpx.Response(409, json={"error": "version_conflict"})
        return httpx.Response(200, json={"status": "valid", "version": 1})

    with _client() as client:
        _register(client, "node-a", "node-b")
        client.app.state.ctx.outbound_client = _mock_client(handler)
        response = client.patch("/fleet/config/policies", json={"update": _update()})
        assert response.status_code == 409
        assert response.json()["status"] == "rejected"

    assert calls == ["POST", "POST"]


def test_fleet_apply_reports_partial_failure() -> None:
    """Apply failures after validation are visible per target."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"status": "valid", "version": 1})
        if request.url.port == 8002:
            raise httpx.ConnectError("node disappeared", request=request)
        return httpx.Response(200, json={"status": "updated", "version": 2})

    with _client() as client:
        _register(client, "node-a", "node-b")
        client.app.state.ctx.outbound_client = _mock_client(handler)
        response = client.patch("/fleet/config/policies", json={"update": _update()})
        assert response.status_code == 502
        assert response.json()["status"] == "partial"
        assert {item["status_code"] for item in response.json()["results"]} == {
            200,
            502,
        }


def test_fleet_without_registered_instances_returns_404() -> None:
    with _client() as client:
        response = client.post(
            "/fleet/config/policies/validate", json={"update": _update()}
        )
        assert response.status_code == 404
