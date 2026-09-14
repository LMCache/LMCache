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
from lmcache.v1.memory_coordinator.config import MemoryCoordinatorConfig

_TOKEN = "test-memory-coordinator-token"
_CAPACITY = 64 * 1024
_ALIGNMENT = 4096


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.write_text(_TOKEN + "\n")
    return path


def _config(token_file: Path) -> MemoryCoordinatorConfig:
    return MemoryCoordinatorConfig(
        token_file=str(token_file),
        state_file=str(token_file.with_name("coordinator.state")),
        region_id="region",
        capacity_bytes=_CAPACITY,
        alignment_bytes=_ALIGNMENT,
        layout_id="layout",
    )


@pytest.fixture
def client(token_file: Path) -> TestClient:
    app = create_app(_config(token_file))
    return TestClient(app, headers={"Authorization": f"Bearer {_TOKEN}"})


def _key(seed: int) -> dict:
    return {
        "chunk_hash_hex": f"{seed:08x}",
        "model_name": "model",
        "kv_rank": 0,
        "object_group_id": 0,
        "cache_salt": "",
    }


def _layout() -> dict:
    return {"shapes": [[64]], "dtypes": ["float16"]}


def _epoch(client: TestClient) -> str:
    return str(client.get("/v1/region").json()["region_epoch"])


def test_health_endpoints_require_no_auth(token_file: Path) -> None:
    app = create_app(_config(token_file))
    anonymous = TestClient(app)
    assert anonymous.get("/healthz").status_code == 200
    assert anonymous.get("/readyz").status_code == 200
    assert anonymous.get("/docs").status_code == 404
    assert anonymous.get("/openapi.json").status_code == 404


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
    app = create_app(_config(token_file))
    anonymous = TestClient(app)
    assert anonymous.request(method, path).status_code == 401
    wrong = TestClient(app, headers={"Authorization": "Bearer wrong"})
    assert wrong.request(method, path).status_code == 403


def test_schema_rejects_payload_fields(client: TestClient) -> None:
    epoch = _epoch(client)
    smuggled = client.post(
        "/v1/writes/reserve",
        json={
            "region_epoch": epoch,
            "items": [
                {"key": _key(1), "layout": _layout(), "payload": "AAAA"},
            ],
        },
    )
    assert smuggled.status_code == 422
    top_level = client.post(
        "/v1/writes/reserve",
        json={"region_epoch": epoch, "items": [], "data": "AAAA"},
    )
    assert top_level.status_code == 422


def test_stale_epoch_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/writes/reserve",
        json={"region_epoch": "not-the-epoch", "items": []},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "stale_epoch"


def test_out_of_space_maps_to_507(client: TestClient) -> None:
    epoch = _epoch(client)
    huge = {"shapes": [[_CAPACITY * 2]], "dtypes": ["uint8"]}
    response = client.post(
        "/v1/writes/reserve",
        json={"region_epoch": epoch, "items": [{"key": _key(1), "layout": huge}]},
    )
    assert response.status_code == 507
    assert response.json()["error"] == "out_of_space"


def test_invalid_reservation_maps_to_409(client: TestClient) -> None:
    epoch = _epoch(client)
    response = client.post(
        "/v1/writes/finish",
        json={
            "region_epoch": epoch,
            "reservations": [
                {
                    "key": _key(1),
                    "token": "wrong",
                }
            ],
        },
    )
    assert response.status_code == 409
    assert response.json()["error"] == "invalid_reservation"


@pytest.mark.parametrize(
    "contents", ["  \n", "two tokens", "tøken", "token\x00", "token\x7f"]
)
def test_invalid_token_file_fails_startup(tmp_path: Path, contents: str) -> None:
    empty = tmp_path / "token"
    empty.write_text(contents)
    with pytest.raises(ValueError, match="ASCII token"):
        create_app(
            MemoryCoordinatorConfig(
                token_file=str(empty),
                state_file=str(tmp_path / "coordinator.state"),
                region_id="region",
                capacity_bytes=_CAPACITY,
                alignment_bytes=_ALIGNMENT,
                layout_id="layout",
            )
        )


def test_startup_latch_survives_clean_shutdown(token_file: Path) -> None:
    config = _config(token_file)
    with TestClient(create_app(config)) as client:
        assert client.get("/readyz").status_code == 200
    state_file = Path(config.state_file)
    marker = state_file.read_bytes()
    assert marker and _TOKEN.encode() not in marker
    with pytest.raises(RuntimeError, match="coordinated pool reset"):
        create_app(config)
    assert state_file.read_bytes() == marker


@pytest.mark.parametrize("state_file", ["", "relative.state"])
def test_startup_latch_requires_absolute_path(
    token_file: Path, state_file: str
) -> None:
    with pytest.raises(ValueError, match="absolute persistent path"):
        replace(_config(token_file), state_file=state_file)
