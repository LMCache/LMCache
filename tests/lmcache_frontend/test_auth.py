# SPDX-License-Identifier: Apache-2.0
"""Authentication tests for the LMCache frontend HTTP server."""

# Standard
from collections.abc import Iterator

# Third Party
from fastapi.testclient import TestClient
import httpx
import pytest

# First Party
from lmcache.lmcache_frontend import app as frontend
from lmcache.security.api_key import APIKeyVerifier


@pytest.fixture(autouse=True)
def reset_nodes() -> Iterator[None]:
    """Reset the process-global frontend node registry around each test."""
    frontend.target_nodes.clear()
    yield
    frontend.target_nodes.clear()


def test_health_and_static_assets_remain_public() -> None:
    """Health probes and browser assets do not require a token."""
    with TestClient(frontend.create_app(APIKeyVerifier(["secret"]))) as client:
        health = client.get("/health")
        static = client.get("/static/js/app.js")

    assert health.status_code == 200
    assert static.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}],
)
@pytest.mark.parametrize("path", ["/api/proxies", "/proxy2/node/status", "/future-api"])
def test_frontend_api_rejects_missing_or_wrong_key(
    headers: dict[str, str], path: str
) -> None:
    """Protected frontend APIs reject missing and incorrect credentials."""
    with TestClient(frontend.create_app(APIKeyVerifier(["secret"]))) as client:
        response = client.get(path, headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_metrics_remain_public_with_authenticated_backend(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    """Public scrapes still use the configured credential for backend requests."""

    def backend(request: httpx.Request) -> httpx.Response:
        assert request.headers.get_list("authorization") == ["Bearer backend-key"]
        return httpx.Response(200, text="cache_hits 7\n")

    async_client = httpx.AsyncClient

    def make_client(**kwargs: object) -> httpx.AsyncClient:
        return async_client(transport=httpx.MockTransport(backend))

    monkeypatch.setattr(frontend.httpx, "AsyncClient", make_client)
    frontend.target_nodes.append(
        {"nodes": [{"name": "node", "host": "127.0.0.1", "port": "8080"}]}
    )
    app = frontend.create_app(APIKeyVerifier(["secret"]), "Bearer backend-key")
    with TestClient(app) as client:
        response = client.get("/metrics", headers=headers)

    assert response.status_code == 200
    assert "cache_hits 7" in response.text


def test_frontend_api_accepts_configured_key() -> None:
    """A configured frontend token reaches the requested API handler."""
    with TestClient(frontend.create_app(APIKeyVerifier(["secret"]))) as client:
        response = client.get(
            "/api/proxies", headers={"Authorization": "Bearer secret"}
        )

    assert response.status_code == 200
    assert response.json() == {"proxies": []}


@pytest.mark.parametrize(
    ("frontend_keys", "backend_authorization", "expected_authorization"),
    [
        (["frontend-secret"], "Bearer backend-secret", ["Bearer backend-secret"]),
        (["frontend-secret"], None, []),
        ([], "Bearer backend-secret", ["Bearer backend-secret"]),
        ([], None, ["Bearer frontend-secret"]),
    ],
)
def test_proxy_does_not_forward_frontend_key(
    monkeypatch: pytest.MonkeyPatch,
    frontend_keys: list[str],
    backend_authorization: str | None,
    expected_authorization: list[str],
) -> None:
    """Backend credentials replace caller headers even when inbound auth is off."""
    captured_headers: list[str] = []

    def backend(request: httpx.Request) -> httpx.Response:
        captured_headers.extend(request.headers.get_list("authorization"))
        return httpx.Response(200, content=b"ok")

    async_client = httpx.AsyncClient

    def make_client(**kwargs: object) -> httpx.AsyncClient:
        return async_client(transport=httpx.MockTransport(backend))

    monkeypatch.setattr(frontend.httpx, "AsyncClient", make_client)
    frontend.target_nodes.append({"name": "node", "host": "127.0.0.1", "port": "8080"})
    app = frontend.create_app(
        APIKeyVerifier(frontend_keys),
        backend_authorization,
    )

    with TestClient(app) as client:
        response = client.get(
            "/proxy2/node/status",
            headers={"Authorization": "Bearer frontend-secret"},
        )

    assert response.status_code == 200
    assert captured_headers == expected_authorization


def test_frontend_remains_compatible_when_authentication_is_disabled() -> None:
    """An unconfigured frontend preserves its existing unauthenticated API."""
    with TestClient(frontend.create_app()) as client:
        response = client.get("/api/proxies")

    assert response.status_code == 200
