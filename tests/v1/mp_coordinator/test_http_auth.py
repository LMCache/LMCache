# SPDX-License-Identifier: Apache-2.0
"""Tests for MP coordinator bearer authentication (CVE-2026-107205).

The coordinator is the fleet control plane: when an API key is configured,
every route except the public liveness probe must present a valid bearer
token. When no key is configured the app stays open, preserving the no-key
default.
"""

# Third Party
from fastapi import FastAPI
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
    def healthz() -> dict[str, bool]:
        return {"status": "healthy"}

    @app.get("/instances")
    def instances() -> dict[str, bool]:
        return {"ok": True}

    return app


def _config() -> MPCoordinatorConfig:
    return MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )


def test_healthz_stays_public_when_auth_enabled() -> None:
    """The k8s liveness probe must not require a credential."""
    with TestClient(
        _make_app(CoordinatorAuthenticationConfig(["secret"]))
    ) as client:
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
    with TestClient(
        _make_app(CoordinatorAuthenticationConfig(["secret"]))
    ) as client:
        response = client.get("/instances", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("api_key", ["current-key", "next-key"])
def test_each_configured_key_is_accepted(api_key: str) -> None:
    """Multiple keys support a zero-downtime rotation window."""
    with TestClient(
        _make_app(
            CoordinatorAuthenticationConfig(["current-key", "next-key"])
        )
    ) as client:
        response = client.get(
            "/instances", headers={"Authorization": f"Bearer {api_key}"}
        )

    assert response.status_code == 200


def test_x_lmcache_api_key_header_is_accepted() -> None:
    """The alternate API-key header is accepted alongside Bearer."""
    with TestClient(
        _make_app(CoordinatorAuthenticationConfig(["secret"]))
    ) as client:
        response = client.get(
            "/instances", headers={"X-LMCache-API-Key": "secret"}
        )

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
        response = client.get(
            "/instances", headers={"Authorization": "Bearer secret"}
        )

    assert response.status_code == 200


def test_real_app_without_key_keeps_routes_open() -> None:
    """A coordinator started with no key behaves exactly as before the fix."""
    with TestClient(create_app(_config())) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/instances").status_code == 200
