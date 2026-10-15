# SPDX-License-Identifier: Apache-2.0
"""Tests for MP HTTP bearer authentication."""

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_auth import (
    HTTPAuthenticationConfig,
    HTTPAuthenticationMiddleware,
)


def _make_app(auth_config: HTTPAuthenticationConfig) -> FastAPI:
    app = FastAPI()
    app.state.http_authentication = auth_config
    app.add_middleware(HTTPAuthenticationMiddleware)

    @app.get("/")
    def root() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/healthcheck")
    def healthcheck() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/status")
    def status() -> dict[str, bool]:
        return {"ok": True}

    return app


@pytest.mark.parametrize("path", ["/", "/healthcheck"])
def test_public_routes_do_not_require_api_key(path: str) -> None:
    """Liveness and health probes remain public when auth is enabled."""
    with TestClient(_make_app(HTTPAuthenticationConfig(["secret"]))) as client:
        response = client.get(path)

    assert response.status_code == 200


def test_authentication_is_disabled_without_keys() -> None:
    """Existing deployments remain compatible when no key is configured."""
    with TestClient(_make_app(HTTPAuthenticationConfig())) as client:
        response = client.get("/status")

    assert response.status_code == 200


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic secret", "Bearer wrong", "Bearer"],
)
def test_protected_route_rejects_invalid_credentials(
    authorization: str | None,
) -> None:
    """Protected routes return a standard bearer challenge on failure."""
    headers = {} if authorization is None else {"Authorization": authorization}
    with TestClient(_make_app(HTTPAuthenticationConfig(["secret"]))) as client:
        response = client.get("/status", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("api_key", ["current-key", "next-key"])
def test_each_configured_api_key_is_accepted(api_key: str) -> None:
    """Multiple configured keys support a zero-downtime rotation window."""
    auth_config = HTTPAuthenticationConfig(["current-key", "next-key"])
    with TestClient(_make_app(auth_config)) as client:
        response = client.get("/status", headers={"Authorization": f"Bearer {api_key}"})

    assert response.status_code == 200


def test_key_file_contents_are_hashed_and_hidden(tmp_path) -> None:
    """Key files produce usable config without exposing secret material."""
    key_file = tmp_path / "api-key"
    key_file.write_text("top-secret\n", encoding="utf-8")

    auth_config = HTTPAuthenticationConfig.from_key_files([key_file])

    assert auth_config.enabled
    assert auth_config.accepts_authorization_header("Bearer top-secret")
    assert "top-secret" not in repr(auth_config)


@pytest.mark.parametrize("contents", ["", "first\nsecond\n"])
def test_invalid_key_file_is_rejected(tmp_path, contents: str) -> None:
    """Empty and multiline key files fail closed during startup."""
    key_file = tmp_path / "api-key"
    key_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="HTTP API key file"):
        HTTPAuthenticationConfig.from_key_files([key_file])
