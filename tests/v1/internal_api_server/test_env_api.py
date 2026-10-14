# SPDX-License-Identifier: Apache-2.0
"""Tests for ``GET /env`` credential redaction."""

# Standard
import json

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.internal_api_server.common.env_api import router


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Serve ``/env`` with a known set of variables in the environment.

    Returns:
        The parsed ``/env`` response body.
    """
    for name, value in {
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "HF_TOKEN": "hf-token",
        "LMCACHE_HTTP_API_KEY": "lmcache-key",
        "PGPASSWORD": "pg-password",
        "GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
        "lowercase_api_key": "lower-secret",
        "LMCACHE_CHUNK_SIZE": "256",
    }.items():
        monkeypatch.setenv(name, value)
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/env")
    assert response.status_code == 200
    return json.loads(response.text)


@pytest.mark.parametrize(
    "name",
    [
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "LMCACHE_HTTP_API_KEY",
        "PGPASSWORD",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "lowercase_api_key",
    ],
)
def test_credential_values_are_redacted(env, name):
    """The name stays visible so operators can confirm it is set."""
    assert env[name] == "<redacted>"


def test_other_values_are_returned(env):
    assert env["LMCACHE_CHUNK_SIZE"] == "256"
