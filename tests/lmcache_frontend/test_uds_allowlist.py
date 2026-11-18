# SPDX-License-Identifier: Apache-2.0
"""Unix domain socket authorization tests for the frontend proxy."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote
import argparse
import asyncio
import json
import os

# Third Party
from fastapi.testclient import TestClient
import httpx
import pytest

# First Party
from lmcache.lmcache_frontend import app as frontend

DOCKER = "/var/run/docker.sock"
DENIED = "is not configured for proxy access"


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset process-global frontend configuration around each test."""
    frontend.target_nodes.clear()
    monkeypatch.setattr(frontend, "args", None)
    yield
    frontend.target_nodes.clear()


def _serve(monkeypatch: pytest.MonkeyPatch, expected: str) -> None:
    """Answer UDS requests in memory, asserting the socket path used."""

    def make_transport(*, uds: str) -> httpx.MockTransport:
        assert uds == expected
        return httpx.MockTransport(
            lambda request: httpx.Response(200, text=f"ok:{request.url.path}")
        )

    monkeypatch.setattr(frontend.httpx, "AsyncHTTPTransport", make_transport)


def _forbid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if any UDS transport is created."""
    monkeypatch.setattr(
        frontend.httpx,
        "AsyncHTTPTransport",
        lambda **_kwargs: pytest.fail("unexpected socket transport"),
    )


def _direct(query_path: str) -> str:
    """Encode a socket path for the triple-decoded ``/proxy`` route."""
    return "/proxy/localhost/%s/status" % quote(
        quote(quote(query_path, safe=""), safe=""), safe=""
    )


def test_configured_socket_serves_both_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "/tmp/lmcache.sock"
    frontend.target_nodes.append({"name": "node", "host": "localhost", "port": path})
    _serve(monkeypatch, path)

    with TestClient(frontend.create_app(allowed_socket_paths={path})) as client:
        direct = client.get(_direct(path))
        named = client.get("/proxy2/node/status")

    assert (direct.status_code, named.status_code) == (200, 200)
    assert direct.text == "ok:/status"


@pytest.mark.parametrize(
    "url",
    [
        "/proxy/localhost/%25252Fvar%25252Frun%25252Fdocker.sock/info",
        "/proxy2/docker/info",
        "/proxy2/runtime/status",
    ],
)
def test_unconfigured_socket_is_denied(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    frontend.target_nodes.append(
        {"name": "docker", "host": "localhost", "port": DOCKER}
    )
    _forbid(monkeypatch)

    with TestClient(frontend.create_app()) as client:
        if url.endswith("runtime/status"):
            client.post(
                "/api/proxies",
                json={"name": "runtime", "host": "localhost", "port": "/tmp/x.sock"},
            )
        response = client.get(url)

    assert response.status_code == 403
    assert DENIED in response.json()["detail"]


def test_metrics_skips_unconfigured_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    frontend.target_nodes.append(
        {
            "name": "proxy",
            "host": "localhost",
            "port": "8000",
            "nodes": [{"name": "docker", "host": "localhost", "port": DOCKER}],
        }
    )
    _forbid(monkeypatch)

    with TestClient(frontend.create_app()) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert DENIED in response.text


def test_startup_sockets_are_trusted(tmp_path: Path) -> None:
    nodes = [
        {"name": "uds", "host": "localhost", "port": "/tmp/configured.sock"},
        {"name": "tcp", "host": "127.0.0.1", "port": "8080"},
    ]
    frontend.args = argparse.Namespace(
        nodes=json.dumps(nodes), config=None, host="127.0.0.1", port=8000
    )
    assert asyncio.run(frontend.initialize_nodes()) == frozenset(
        {"/tmp/configured.sock"}
    )

    tree = [
        {
            "name": "p",
            "port": "/tmp/p.sock",
            "nodes": [{"name": "c", "port": "/tmp/c.sock"}],
        }
    ]
    config = tmp_path / "nodes.json"
    config.write_text(json.dumps(tree), encoding="utf-8")
    frontend.args = argparse.Namespace(nodes=None, config=str(config))
    assert asyncio.run(frontend.initialize_nodes()) == frozenset(
        {"/tmp/p.sock", "/tmp/c.sock"}
    )


def test_supplier_discovery_grants_no_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend.args = argparse.Namespace(nodes=None, config=None)

    async def load_supplier(*_args, **_kwargs) -> bool:
        frontend.target_nodes.append({"name": "d", "port": "/tmp/new.sock"})
        return True

    monkeypatch.setattr(frontend, "load_nodes_from_supplier", load_supplier)

    assert (
        asyncio.run(frontend.initialize_nodes("http://supplier/nodes")) == frozenset()
    )


@pytest.mark.parametrize("outside", [False, True])
def test_directory_entry_gates_discovered_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outside: bool
) -> None:
    declared = tmp_path / "lmcache"
    declared.mkdir()
    path = str(tmp_path / "outside.sock" if outside else declared / "node.sock")
    frontend.target_nodes.append({"name": "node", "host": "localhost", "port": path})
    if outside:
        _forbid(monkeypatch)
    else:
        _serve(monkeypatch, path)

    with TestClient(
        frontend.create_app(allowed_socket_paths={str(declared)})
    ) as client:
        response = client.get("/proxy2/node/status")

    assert response.status_code == (403 if outside else 200)


@pytest.mark.parametrize("escape", ["parent", "symlink"])
def test_directory_entry_resists_escape(tmp_path: Path, escape: str) -> None:
    declared = tmp_path / "lmcache"
    declared.mkdir()
    outside = tmp_path / "outside.sock"
    outside.write_text("", encoding="utf-8")
    candidate = str(declared / "linked.sock")
    if escape == "parent":
        candidate = f"{declared}/../outside.sock"
    else:
        os.symlink(outside, candidate)

    assert not frontend._socket_allowed(candidate, {str(declared)})
