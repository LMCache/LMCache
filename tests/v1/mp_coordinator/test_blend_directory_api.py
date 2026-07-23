# SPDX-License-Identifier: Apache-2.0
"""Tests for the /blend fingerprint directory REST endpoints."""

# Standard
import base64

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.schemas import encode_tokens, encode_tokens_u64

CHUNK = 3
SCOPE = "model-a"


def _client() -> TestClient:
    config = MPCoordinatorConfig(health_check_interval=0.0, chunk_size=CHUNK)
    return TestClient(create_app(config))


def _range(prefix: str, tokens: list[int]) -> dict[str, object]:
    n_chunks = len(tokens) // CHUNK
    return {
        "model_scope": SCOPE,
        "tokens": tokens,
        "object_keys": [f"{prefix}{i}" for i in range(n_chunks)],
        "old_st_base": 0,
    }


def test_publish_then_match():
    with _client() as client:
        doc = [1, 2, 3, 4, 5, 6]
        resp = client.post("/blend/fingerprints", json={"ranges": [_range("K", doc)]})
        assert resp.status_code == 200
        assert resp.json() == {"inserted": 2}

        out = client.post(
            "/blend/match",
            json={"model_scope": SCOPE, "tokens_u64_b64": encode_tokens_u64(doc)},
        ).json()["matches"]
        assert [(m["object_key"], m["old_st"], m["cur_st"]) for m in out] == [
            ("K0", 0, 0),
            ("K1", 3, 3),
        ]


def test_match_miss_returns_empty():
    with _client() as client:
        out = client.post(
            "/blend/match",
            json={
                "model_scope": SCOPE,
                "tokens_u64_b64": encode_tokens_u64([7, 8, 9]),
            },
        ).json()
        assert out == {"matches": []}


def test_remove_evicts():
    with _client() as client:
        doc = [1, 2, 3, 4, 5, 6]
        client.post("/blend/fingerprints", json={"ranges": [_range("K", doc)]})
        resp = client.request(
            "DELETE", "/blend/fingerprints", json={"object_keys": ["K0", "K1"]}
        )
        assert resp.json() == {"removed": 2}
        out = client.post(
            "/blend/match",
            json={"model_scope": SCOPE, "tokens_u64_b64": encode_tokens_u64(doc)},
        ).json()
        assert out == {"matches": []}


def test_idempotent_publish():
    with _client() as client:
        doc = [1, 2, 3, 4, 5, 6]
        client.post("/blend/fingerprints", json={"ranges": [_range("K", doc)]})
        resp = client.post("/blend/fingerprints", json={"ranges": [_range("K", doc)]})
        assert resp.json() == {"inserted": 0}


def test_match_malformed_tokens_b64_returns_422():
    """A bad tokens_b64 is a client error (422), not an unhandled 500."""
    with _client() as client:
        # Not valid base64.
        resp = client.post(
            "/blend/match",
            json={"model_scope": SCOPE, "tokens_b64": "not!base64!"},
        )
        assert resp.status_code == 422

        # Valid base64 but not a whole number of uint64 tokens (4 bytes).
        resp = client.post(
            "/blend/match",
            json={
                "model_scope": SCOPE,
                "tokens_u64_b64": base64.b64encode(b"\x01\x02\x03\x04").decode("ascii"),
            },
        )
        assert resp.status_code == 422


def test_match_missing_token_payload_returns_422() -> None:
    """A missing token payload is a client error, not an unhandled 500."""
    with _client() as client:
        resp = client.post("/blend/match", json={"model_scope": SCOPE})
        assert resp.status_code == 422


def test_match_accepts_legacy_uint32_tokens_b64() -> None:
    """The old uint32 field remains readable for rolling upgrade safety."""
    with _client() as client:
        doc = [1, 2, 3]
        client.post("/blend/fingerprints", json={"ranges": [_range("K", doc)]})
        resp = client.post(
            "/blend/match",
            json={"model_scope": SCOPE, "tokens_b64": encode_tokens(doc)},
        )
        assert resp.status_code == 200
        assert resp.json()["matches"][0]["object_key"] == "K0"


def test_match_rejects_ambiguous_token_payloads() -> None:
    """Clients must not send both legacy and uint64 token payload fields."""
    with _client() as client:
        resp = client.post(
            "/blend/match",
            json={
                "model_scope": SCOPE,
                "tokens_b64": encode_tokens([1, 2, 3]),
                "tokens_u64_b64": encode_tokens_u64([1, 2, 3]),
            },
        )
        assert resp.status_code == 422
