# SPDX-License-Identifier: Apache-2.0
"""HTTP contract tests for auth, schema validation, and epochs."""

# Standard
from dataclasses import replace
from pathlib import Path

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.memory_coordinator.app import create_app

# Local
from .conftest import CAPACITY, TOKEN
from .conftest import config as _config
from .conftest import item

_ITEM = item(1).model_dump(mode="json")
_KEY = _ITEM["key"]
_HUGE = {"key": _KEY, "layout": {"shapes": [[CAPACITY * 2]], "dtypes": ["uint8"]}}


@pytest.fixture
def client(token_file: Path) -> TestClient:
    app = create_app(_config(token_file))
    return TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})


def _epoch(client: TestClient) -> str:
    return str(client.get("/v1/region").json()["region_epoch"])


def test_health_endpoints_require_no_auth(token_file: Path) -> None:
    with TestClient(create_app(_config(token_file))) as anonymous:
        for path, status in (
            ("/healthz", 200),
            ("/readyz", 200),
            ("/docs", 404),
            ("/openapi.json", 404),
        ):
            assert anonymous.get(path).status_code == status


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/region"),
        ("GET", "/v1/status"),
        ("POST", "/v1/writes/reserve"),
        ("POST", "/v1/writes/finish"),
        ("POST", "/v1/writes/abort"),
        ("POST", "/v1/lookup"),
    ],
)
def test_non_health_endpoints_reject_missing_and_wrong_tokens(
    token_file: Path,
    method: str,
    path: str,
) -> None:
    with TestClient(create_app(_config(token_file))) as client:
        assert client.request(method, path).status_code == 401
        assert (
            client.request(
                method, path, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 403
        )


@pytest.mark.parametrize(
    "body",
    [
        {"items": [_ITEM | {"payload": "AAAA"}]},
        {"items": [], "data": "AAAA"},
    ],
)
def test_schema_rejects_payload_fields(client: TestClient, body: dict) -> None:
    assert (
        client.post(
            "/v1/writes/reserve", json={"region_epoch": _epoch(client)} | body
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    ("path", "body", "status", "error"),
    [
        ("reserve", {"region_epoch": "stale", "items": []}, 409, "stale_epoch"),
        (
            "reserve",
            {"items": [_HUGE]},
            507,
            "out_of_space",
        ),
        (
            "finish",
            {"reservations": [{"key": _KEY, "token": "wrong"}]},
            409,
            "invalid_reservation",
        ),
    ],
)
def test_errors_map_to_http_status(
    client: TestClient,
    path: str,
    body: dict,
    status: int,
    error: str,
) -> None:
    response = client.post(
        f"/v1/writes/{path}",
        json={"region_epoch": _epoch(client)} | body,
    )
    assert response.status_code == status
    assert response.json()["error"] == error


@pytest.mark.parametrize(
    "contents", ["  \n", "two tokens", "tøken", "token\x00", "token\x7f"]
)
def test_invalid_token_file_fails_startup(tmp_path: Path, contents: str) -> None:
    empty = tmp_path / "token"
    empty.write_text(contents)
    with pytest.raises(ValueError, match="ASCII token"):
        create_app(_config(empty))


def test_startup_latch_survives_clean_shutdown(token_file: Path) -> None:
    config = _config(token_file)
    with TestClient(create_app(config)) as client:
        assert client.get("/readyz").status_code == 200
    state_file = Path(config.state_file)
    marker = state_file.read_bytes()
    assert marker and TOKEN.encode() not in marker
    with pytest.raises(RuntimeError, match="coordinated pool reset"):
        create_app(config)
    assert state_file.read_bytes() == marker


@pytest.mark.parametrize("state_file", ["", "relative.state"])
def test_startup_latch_requires_absolute_path(
    token_file: Path, state_file: str
) -> None:
    with pytest.raises(ValueError, match="absolute persistent path"):
        replace(_config(token_file), state_file=state_file)
