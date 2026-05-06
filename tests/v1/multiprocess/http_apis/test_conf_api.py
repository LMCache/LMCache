# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ``/conf`` endpoint exposed by conf_api.

Covers:
- Dataclass configs serialized with nested values.
- Plain-dict configs merged via ``make_json_safe``.
- Missing ``app.state.configs`` returns HTTP 503.
- Response body is valid JSON and is indented for readability.
"""

# Standard
from dataclasses import dataclass, field
from pathlib import Path
import json

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_apis.conf_api import router as conf_router


@dataclass
class _FakeMPConfig:
    host: str = "0.0.0.0"
    port: int = 9000
    path: Path = Path("/tmp/mp")


@dataclass
class _FakeStorageConfig:
    backend: str = "local"
    capacity: int = 1024
    tags: list = field(default_factory=lambda: ["a", "b"])


def _make_app(configs):
    app = FastAPI()
    app.include_router(conf_router)
    if configs is not None:
        app.state.configs = configs
    return app


class TestConfEndpoint:
    def test_dataclass_configs_serialized(self):
        app = _make_app(
            {
                "mp": _FakeMPConfig(host="1.2.3.4", port=8000),
                "storage": _FakeStorageConfig(backend="redis"),
            }
        )
        client = TestClient(app)
        resp = client.get("/conf")

        assert resp.status_code == 200
        body = resp.json()
        assert body["mp"] == {
            "host": "1.2.3.4",
            "port": 8000,
            "path": "/tmp/mp",
        }
        assert body["storage"]["backend"] == "redis"
        assert body["storage"]["capacity"] == 1024
        assert body["storage"]["tags"] == ["a", "b"]

    def test_plain_dict_config_merged(self):
        """Non-dataclass values still go through make_json_safe."""
        app = _make_app({"extra": {"k": Path("/v")}})
        client = TestClient(app)

        body = client.get("/conf").json()
        assert body == {"extra": {"k": "/v"}}

    def test_returns_503_when_configs_missing(self):
        """/conf returns 503 if app.state.configs is absent."""
        client = TestClient(_make_app(configs=None))
        resp = client.get("/conf")

        assert resp.status_code == 503
        assert resp.json() == {"error": "configs not initialized"}

    def test_response_is_indented_json(self):
        """Indented JSON renderer keeps the payload human-readable."""
        app = _make_app({"mp": _FakeMPConfig()})
        client = TestClient(app)

        raw = client.get("/conf").text
        # Indented output has newlines and 2-space indentation.
        assert "\n" in raw
        assert '  "mp"' in raw
        # And the body is still valid JSON.
        assert json.loads(raw)["mp"]["host"] == "0.0.0.0"

    def test_empty_configs_returns_empty_object(self):
        client = TestClient(_make_app(configs={}))
        resp = client.get("/conf")

        assert resp.status_code == 200
        assert resp.json() == {}


@pytest.mark.parametrize(
    "configs,expected_key",
    [
        ({"only": _FakeMPConfig()}, "only"),
        ({"a": _FakeMPConfig(), "b": _FakeStorageConfig()}, "b"),
    ],
)
def test_arbitrary_config_keys_round_trip(configs, expected_key):
    client = TestClient(_make_app(configs))
    body = client.get("/conf").json()
    assert expected_key in body
