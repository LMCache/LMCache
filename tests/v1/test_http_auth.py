# SPDX-License-Identifier: Apache-2.0
"""Tests for MP HTTP bearer authentication."""

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_auth import (
    HTTP_API_KEY_ENV_VAR,
    HTTP_API_KEY_FILE_ENV_VAR,
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


@pytest.mark.parametrize("api_key", ["current-key", "next-key"])
@pytest.mark.parametrize("header", ["X-LMCache-API-Key", "x-lmcache-api-key"])
def test_dedicated_header_accepts_each_configured_key(
    api_key: str, header: str
) -> None:
    """Proxy-compatible headers use the same accepted keys as Bearer auth."""
    auth_config = HTTPAuthenticationConfig(["current-key", "next-key"])
    with TestClient(_make_app(auth_config)) as client:
        response = client.get("/status", headers={header: api_key})

    assert response.status_code == 200


@pytest.mark.parametrize("api_key", ["", "wrong", "Bearer secret", "wrong-é"])
def test_dedicated_header_rejects_invalid_keys(api_key: str) -> None:
    """The dedicated header contains the raw key, not a Bearer value."""
    with TestClient(_make_app(HTTPAuthenticationConfig(["secret"]))) as client:
        response = client.get(
            "/status", headers={"X-LMCache-API-Key": api_key.encode("utf-8")}
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("authorization", "api_key", "expected_status"),
    [
        ("Bearer wrong", "secret", 200),
        ("Basic cluster-credential", "secret", 200),
        ("Bearer secret", "wrong", 200),
        ("Bearer secret", "secret", 200),
        ("Bearer wrong", "wrong", 401),
    ],
)
def test_both_headers_accept_either_valid_credential(
    authorization: str, api_key: str, expected_status: int
) -> None:
    """An unrelated proxy authorization value does not mask a valid API key."""
    with TestClient(_make_app(HTTPAuthenticationConfig(["secret"]))) as client:
        response = client.get(
            "/status",
            headers={"Authorization": authorization, "X-LMCache-API-Key": api_key},
        )

    assert response.status_code == expected_status


def test_key_file_contents_are_loaded_and_hidden(tmp_path) -> None:
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

    with pytest.raises(ValueError, match="API key file"):
        HTTPAuthenticationConfig.from_key_files([key_file])


def test_environment_key_and_file_are_combined_with_cli_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every supported source contributes an accepted key for rotation."""
    cli_key_file = tmp_path / "cli-key"
    cli_key_file.write_text("cli-secret\n", encoding="utf-8")
    environment_key_file = tmp_path / "environment-key"
    environment_key_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv(HTTP_API_KEY_FILE_ENV_VAR, str(environment_key_file))
    monkeypatch.setenv(HTTP_API_KEY_ENV_VAR, "environment-secret")

    auth_config = HTTPAuthenticationConfig.from_sources([cli_key_file])

    for api_key in ("cli-secret", "file-secret", "environment-secret"):
        assert auth_config.accepts_authorization_header(f"Bearer {api_key}")


@pytest.mark.parametrize(
    "environment_variable",
    [HTTP_API_KEY_ENV_VAR, HTTP_API_KEY_FILE_ENV_VAR],
)
def test_empty_environment_source_is_ignored(
    monkeypatch: pytest.MonkeyPatch, environment_variable: str
) -> None:
    """An empty environment value behaves like an unset source."""
    monkeypatch.setenv(environment_variable, "")

    auth_config = HTTPAuthenticationConfig.from_sources([])

    assert not auth_config.enabled
