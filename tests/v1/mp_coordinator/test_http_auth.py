# SPDX-License-Identifier: Apache-2.0
"""Tests for MP coordinator bearer authentication (CVE-2026-107205).

The coordinator is the fleet control plane: when an API key is configured,
every route except the public liveness probe must present a valid bearer
token. When no key is configured the app stays open, preserving the no-key
default.
"""

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.http_auth import (
    CoordinatorAuthenticationConfig,
    CoordinatorAuthenticationMiddleware,
)


def _make_app(
    auth_config: CoordinatorAuthenticationConfig,
) -> FastAPI:
    app = FastAPI()
    app.state.http_authentication = auth_config
    app.add_middleware(CoordinatorAuthenticationMiddleware)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "healthy"}

    @app.get("/instances")
    def instances() -> dict[str, object]:
        return {"ok": True}

    return app


def _config() -> MPCoordinatorConfig:
    return MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )


def test_healthz_stays_public_when_auth_enabled() -> None:
    """The k8s liveness probe must not require a credential."""
    with TestClient(_make_app(CoordinatorAuthenticationConfig(["secret"]))) as client:
        response = client.get("/healthz")

    assert response.status_code == 200


def test_authentication_disabled_without_keys() -> None:
    """Existing deployments remain open when no key is configured."""
    with TestClient(_make_app(CoordinatorAuthenticationConfig())) as client:
        response = client.get("/instances")

    assert response.status_code == 200


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic secret", "Bearer wrong", "Bearer"],
)
def test_protected_route_rejects_invalid_credentials(
    authorization: str | None,
) -> None:
    """Fleet routes return a standard bearer challenge on failure."""
    headers = {} if authorization is None else {"Authorization": authorization}
    with TestClient(_make_app(CoordinatorAuthenticationConfig(["secret"]))) as client:
        response = client.get("/instances", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("api_key", ["current-key", "next-key"])
def test_each_configured_key_is_accepted(api_key: str) -> None:
    """Multiple keys support a zero-downtime rotation window."""
    with TestClient(
        _make_app(CoordinatorAuthenticationConfig(["current-key", "next-key"]))
    ) as client:
        response = client.get(
            "/instances", headers={"Authorization": f"Bearer {api_key}"}
        )

    assert response.status_code == 200


def test_cli_key_file_is_used_for_outbound_mp_requests(tmp_path) -> None:
    """The coordinator presents the configured CLI key to MP servers."""
    key_file = tmp_path / "coordinator-key"
    key_file.write_text("fleet-secret\n", encoding="utf-8")
    authentication = CoordinatorAuthenticationConfig.from_sources([key_file])

    assert authentication.client_auth_headers() == {
        "Authorization": "Bearer fleet-secret"
    }


def test_x_lmcache_api_key_header_is_accepted() -> None:
    """The alternate API-key header is accepted alongside Bearer."""
    with TestClient(_make_app(CoordinatorAuthenticationConfig(["secret"]))) as client:
        response = client.get("/instances", headers={"X-LMCache-API-Key": "secret"})

    assert response.status_code == 200


def test_real_app_blocks_fleet_routes_without_key_when_auth_enabled() -> None:
    """The assembled coordinator 401s unauthenticated fleet calls.

    This is the CVE-2026-107205 regression: previously /instances, /quota,
    /events, and /directory/keys were reachable by any network peer.
    """
    auth = CoordinatorAuthenticationConfig(["secret"])
    with TestClient(create_app(_config(), http_authentication=auth)) as client:
        assert client.get("/healthz").status_code == 200
        protected = client.get("/instances")

    assert protected.status_code == 401
    assert protected.headers["www-authenticate"] == "Bearer"


def test_real_app_allows_fleet_route_with_valid_key() -> None:
    """A valid bearer token reaches the assembled coordinator's fleet routes."""
    auth = CoordinatorAuthenticationConfig(["secret"])
    with TestClient(create_app(_config(), http_authentication=auth)) as client:
        response = client.get("/instances", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200


def test_real_app_without_key_keeps_routes_open() -> None:
    """A coordinator started with no key behaves exactly as before the fix."""
    with TestClient(create_app(_config())) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/instances").status_code == 200


# Standard
# ---------------------------------------------------------------------------
# End-to-end over a real TCP socket (a live uvicorn server, not ASGI transport)
# ---------------------------------------------------------------------------
import threading  # noqa: E402

# Third Party
import httpx  # noqa: E402
import uvicorn  # noqa: E402


def _serve_live(app, host: str = "127.0.0.1") -> tuple[uvicorn.Server, str]:
    """Boot *app* on an ephemeral port in a daemon thread.

    Returns:
        The running server and the base URL (``http://host:port``).
    """
    config = uvicorn.Config(app, host=host, port=0, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        threading.Event().wait(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, f"http://{host}:{port}"


def test_live_server_enforces_bearer_auth_end_to_end() -> None:
    """Drive the real coordinator over a live HTTP server.

    This is the CVE-2026-107205 end-to-end check: an unauthenticated peer
    hitting the running fleet control plane is refused, while the public
    probe and a holder of the key are served.
    """
    auth = CoordinatorAuthenticationConfig(["live-secret"])
    app = create_app(_config(), http_authentication=auth)
    server, base = _serve_live(app)
    try:
        with httpx.Client(base_url=base, timeout=5.0) as client:
            probe = client.get("/healthz")
            assert probe.status_code == 200

            unauthenticated = client.get("/instances")
            assert unauthenticated.status_code == 401
            assert unauthenticated.headers["www-authenticate"] == "Bearer"

            wrong = client.get("/instances", headers={"Authorization": "Bearer nope"})
            assert wrong.status_code == 401

            bearer = client.get(
                "/instances", headers={"Authorization": "Bearer live-secret"}
            )
            assert bearer.status_code == 200

            api_key_header = client.get(
                "/instances", headers={"X-LMCache-API-Key": "live-secret"}
            )
            assert api_key_header.status_code == 200
    finally:
        server.should_exit = True


def test_live_server_without_key_stays_open_end_to_end() -> None:
    """A live coordinator started with no key remains reachable, unauthenticated."""
    app = create_app(_config())
    server, base = _serve_live(app)
    try:
        with httpx.Client(base_url=base, timeout=5.0) as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/instances").status_code == 200
    finally:
        server.should_exit = True


def test_live_coordinator_uses_cli_key_for_outbound_mp_request(
    tmp_path,
) -> None:
    """A file-configured coordinator authenticates a proxied MP call."""
    key_file = tmp_path / "fleet-key"
    key_file.write_text("fleet-secret\n", encoding="utf-8")
    auth = CoordinatorAuthenticationConfig.from_sources([key_file])
    target_app = FastAPI()

    @target_app.post("/cache/prefetches")
    async def prefetch(request: Request) -> JSONResponse:
        if request.headers.get("authorization") != "Bearer fleet-secret":
            return JSONResponse(
                status_code=401,
                content={"detail": "coordinator did not send the configured key"},
            )
        return JSONResponse(
            content={"request_id": "probe-1", "chunks": 0, "status": "submitted"}
        )

    target_server, target_url = _serve_live(target_app)
    target_port = int(target_url.rsplit(":", maxsplit=1)[1])
    try:
        with TestClient(create_app(_config(), http_authentication=auth)) as client:
            headers = {"Authorization": "Bearer fleet-secret"}
            registration = client.post(
                "/instances",
                headers=headers,
                json={
                    "instance_id": "outbound-probe",
                    "ip": "127.0.0.1",
                    "http_port": target_port,
                },
            )
            assert registration.status_code == 200
            response = client.post(
                "/cache/prefetches",
                headers=headers,
                json={
                    "instance_id": "outbound-probe",
                    "model_name": "probe",
                    "world_size": 1,
                    "token_ids": [],
                },
            )
    finally:
        target_server.should_exit = True

    assert response.status_code == 200
    assert response.json()["status"] == "submitted"
