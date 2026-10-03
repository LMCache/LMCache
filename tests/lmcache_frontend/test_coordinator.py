# SPDX-License-Identifier: Apache-2.0
"""Unit + integration tests for coordinator-sourced fleet membership.

These tests mock ``httpx.AsyncClient`` so the real network is never
touched. No coordinator or mp server process is required.
"""

# Standard
import asyncio

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.lmcache_frontend import app as fe


class _FakeResponse:
    """Fake ``httpx.Response`` with a preset JSON payload."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("boom")

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Fake ``httpx.AsyncClient`` that returns a preset JSON payload for
    any ``get()`` call."""

    def __init__(self, payload: dict | None = None, raises: bool = False):
        self._payload = payload or {}
        self._raises = raises

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, *args, **kwargs):
        if self._raises:
            raise RuntimeError("Connection is unreachable.")
        return _FakeResponse(self._payload)


def _patch_client(monkeypatch, payload=None, raises=False):
    """Patch ``httpx.AsyncClient`` with a fake client."""
    monkeypatch.setattr(
        fe.httpx, "AsyncClient", _FakeAsyncClient(payload=payload, raises=raises)
    )


_FAKE_HOST = "temp-coordinator"
_FAKE_PORT = str(8000)


# --------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------


def test_maps_instances_to_flat_node_list(monkeypatch):
    """Test ``fetch_nodes_from_coordinator`` maps coordinator payload to a
    flat list of mp-server nodes.

    Input: coordinator payload with 2 registered mp servers.

    Expected output:
    - A flat list of 2 nodes (no proxy wrapper).
    - All information (name, host, port) of each node is correctly mapped.
    """
    payload = {
        "instances": [
            {"instance_id": "instance_1", "ip": "10.0.0.1", "http_port": 8080},
            {"instance_id": "instance_2", "ip": "10.0.0.2", "http_port": 8081},
        ]
    }
    _patch_client(monkeypatch, payload=payload)

    # Fetch nodes from coordinators
    nodes = asyncio.run(
        fe.fetch_nodes_from_coordinator(f"http://{_FAKE_HOST}:{_FAKE_PORT}/")
    )

    # There should be a flat list of two nodes
    assert len(nodes) == 2

    # Check node data
    assert [node["name"] for node in nodes] == ["mp_instance_1", "mp_instance_2"]
    assert [node["host"] for node in nodes] == ["10.0.0.1", "10.0.0.2"]
    assert all(isinstance(node["port"], str) for node in nodes)  # port must be string
    assert [node["port"] for node in nodes] == ["8080", "8081"]


def test_empty_fleet_returns_empty_list(monkeypatch):
    """Test ``fetch_nodes_from_coordinator`` returns an empty list when the
    fleet is empty.

    Input: coordinator payload with no registered mp servers.

    Expected output: An empty list.
    """
    _patch_client(monkeypatch, payload={"instances": []})

    # Fetch nodes from coordinators
    nodes = asyncio.run(
        fe.fetch_nodes_from_coordinator(f"http://{_FAKE_HOST}:{_FAKE_PORT}/")
    )

    # There should be no nodes
    assert nodes == []


def test_unreachable_coordinator_returns_empty_list(monkeypatch):
    """A network failure degrades gracefully to an empty list, no raise."""
    _patch_client(monkeypatch, raises=True)

    # Fetch nodes from coordinators
    nodes = asyncio.run(
        fe.fetch_nodes_from_coordinator(f"http://{_FAKE_HOST}:{_FAKE_PORT}/")
    )

    # Expected: empty list, no raise
    assert nodes == []


def test_trailing_slash_is_normalized(monkeypatch):
    """A trailing slash in the URL does not produce a doubled path."""
    captured = {}

    # Patch the client to capture the URL
    class _FakeCapturingURLResponse(_FakeAsyncClient):
        """Fake client that captures the URL passed to ``get()``."""

        async def get(self, url, *args, **kwargs):
            captured["url"] = url
            return _FakeResponse({"instances": []})

    monkeypatch.setattr(fe.httpx, "AsyncClient", _FakeCapturingURLResponse())

    # Fetch nodes from coordinators
    asyncio.run(fe.fetch_nodes_from_coordinator("http://coord:9300/"))

    # Expected: the URL passed to get() should not have a doubled slash
    assert captured["url"] == "http://coord:9300/instances"


# --------------------------------------------------------------------------
# Integration tests
# --------------------------------------------------------------------------


@pytest.fixture
def client_with_fleet():
    """Fake frontend TestClient seeded with a coordinator-sourced fleet."""

    # Initialize a TestClient with a flat fleet of two mp servers
    fe._node_registry.replace(
        [
            {"name": "mp_instance_1", "host": "10.0.0.1", "port": "8080"},
            {"name": "mp_instance_2", "host": "10.0.0.2", "port": "8081"},
        ]
    )

    yield TestClient(fe.create_app())
    fe._node_registry.replace([])  # cleanup after test


def test_api_nodes_reflects_coordinator_membership(client_with_fleet):
    """Test that the ``/api/nodes`` endpoint reflects the
    coordinator-sourced fleet membership."""

    # Fetch the nodes from the frontend API
    response = client_with_fleet.get("/api/nodes")
    assert response.status_code == 200
    nodes = response.json()["nodes"]

    # Check the flat node list
    assert len(nodes) == 2
    assert [node["name"] for node in nodes] == ["mp_instance_1", "mp_instance_2"]
    assert [node["host"] for node in nodes] == ["10.0.0.1", "10.0.0.2"]
    assert [node["port"] for node in nodes] == ["8080", "8081"]


def test_ssrf_guard_rejects_unregistered_host(client_with_fleet):
    """Test that the SSRF guard rejects requests to unregistered hosts,
    even if the URL is valid."""
    # Unregister host
    target_host = "evil.example.com"
    target_port = 9999
    target_path = "/metrics"

    # Fetch metrics from an unregistered host
    response = client_with_fleet.get(f"/proxy/{target_host}/{target_port}{target_path}")
    assert response.status_code == 403
