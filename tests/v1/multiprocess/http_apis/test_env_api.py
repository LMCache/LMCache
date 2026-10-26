# SPDX-License-Identifier: Apache-2.0
"""Tests for the common environment endpoint."""

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_apis.common.env_api import router


def test_http_authentication_environment_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment endpoint never returns HTTP authentication secrets."""
    monkeypatch.setenv("LMCACHE_HTTP_API_KEY", "top-secret")
    monkeypatch.setenv("LMCACHE_API_KEY", "shared-secret")
    monkeypatch.setenv("LMCACHE_API_KEY_FILE", "/run/secrets/shared-key")
    monkeypatch.setenv("LMCACHE_HTTP_API_KEY_FILE", "/run/secrets/http-key")
    monkeypatch.setenv("LMCACHE_SAFE_TEST_VALUE", "visible")
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.get("/env")

    assert response.status_code == 200
    environment = response.json()
    assert environment["LMCACHE_HTTP_API_KEY"] == "<redacted>"
    assert environment["LMCACHE_API_KEY"] == "<redacted>"
    assert environment["LMCACHE_API_KEY_FILE"] == "/run/secrets/shared-key"
    assert environment["LMCACHE_HTTP_API_KEY_FILE"] == "/run/secrets/http-key"
    assert environment["LMCACHE_SAFE_TEST_VALUE"] == "visible"
