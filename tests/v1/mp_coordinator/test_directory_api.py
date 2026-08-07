# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/directory`` REST API (cache-event
ingestion, placement lookup, and stats)."""

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _client() -> TestClient:
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    return TestClient(create_app(config))


def _key(h: str = "aa", model: str = "m", rank: int = 0, salt: str = "") -> dict:
    return {
        "chunk_hash_hex": h,
        "model_name": model,
        "kv_rank": rank,
        "cache_salt": salt,
    }


def _batch(
    instance_id: str = "node-a",
    incarnation: int = 1,
    seq: int = 1,
    event_type: str = "store",
    tier: str = "l1",
    backend: str = "dram",
    entries: list[dict] | None = None,
) -> dict:
    if entries is None:
        entries = [{"key": _key(), "size_bytes": 1024}]
    return {
        "instance_id": instance_id,
        "incarnation": incarnation,
        "seq": seq,
        "event_type": event_type,
        "tier": tier,
        "backend": backend,
        "entries": entries,
    }


def _post_events(client: TestClient, batches: list[dict]) -> dict:
    resp = client.post("/directory/events", json={"batches": batches})
    assert resp.status_code == 200
    return resp.json()


def _lookup(client: TestClient, keys: list[dict]) -> dict:
    resp = client.post("/directory/lookup", json={"keys": keys})
    assert resp.status_code == 200
    return resp.json()


# -- Events + lookup ---------------------------------------------------------


def test_store_events_then_lookup():
    with _client() as client:
        data = _post_events(client, [_batch()])
        assert data == {"applied": 1, "duplicates": 0, "stale": 0}

        result = _lookup(client, [_key()])["results"]
        assert len(result) == 1
        assert result[0]["key"]["chunk_hash_hex"] == "aa"
        [placement] = result[0]["placements"]
        assert placement["instance_id"] == "node-a"
        assert placement["incarnation"] == 1
        assert placement["tier"] == "l1"
        assert placement["backend"] == "dram"
        assert placement["size_bytes"] == 1024


def test_lookup_unknown_key_returns_empty_placements():
    with _client() as client:
        result = _lookup(client, [_key(h="ff")])["results"]
        assert result[0]["key"]["chunk_hash_hex"] == "ff"
        assert result[0]["placements"] == []


def test_delete_event_removes_placement():
    with _client() as client:
        _post_events(client, [_batch(seq=1)])
        _post_events(
            client,
            [_batch(seq=2, event_type="delete", entries=[{"key": _key()}])],
        )
        result = _lookup(client, [_key()])["results"]
        assert result[0]["placements"] == []


def test_duplicate_and_stale_batches_are_counted():
    with _client() as client:
        _post_events(client, [_batch(incarnation=2, seq=1)])
        data = _post_events(
            client,
            [
                _batch(incarnation=2, seq=1),  # replay -> duplicate
                _batch(incarnation=1, seq=9),  # pre-restart -> stale
                _batch(incarnation=2, seq=2),  # fresh -> applied
            ],
        )
        assert data == {"applied": 1, "duplicates": 1, "stale": 1}


# -- Token -> placement lookup -------------------------------------------------


def _lookup_tokens_body(n_tokens: int, model: str = "m") -> dict:
    return {
        "model_name": model,
        "world_size": 1,
        "token_ids": list(range(n_tokens)),
        "cache_salt": "",
    }


def test_lookup_tokens_short_sequence_resolves_no_chunks():
    with _client() as client:
        resp = client.post("/directory/lookup", json=_lookup_tokens_body(10))
        assert resp.status_code == 200
        assert resp.json() == {"chunks": 0, "results": []}


def test_lookup_tokens_roundtrip():
    with _client() as client:
        # Resolve one full chunk; nothing stored yet -> empty placements.
        first = client.post("/directory/lookup", json=_lookup_tokens_body(256)).json()
        assert first["chunks"] == 1
        assert len(first["results"]) == 1
        assert first["results"][0]["placements"] == []

        # Store the exact key the resolution produced, then look up again.
        key = first["results"][0]["key"]
        _post_events(client, [_batch(entries=[{"key": key, "size_bytes": 64}])])

        second = client.post("/directory/lookup", json=_lookup_tokens_body(256)).json()
        [placement] = second["results"][0]["placements"]
        assert placement["instance_id"] == "node-a"
        assert placement["size_bytes"] == 64


def test_lookup_tokens_invalid_model_name_is_400():
    with _client() as client:
        resp = client.post(
            "/directory/lookup", json=_lookup_tokens_body(256, model="a@b")
        )
        assert resp.status_code == 400


# -- Stats -------------------------------------------------------------------


def test_stats_reports_counts_and_gap_flag():
    with _client() as client:
        _post_events(client, [_batch(seq=1)])
        _post_events(client, [_batch(seq=5, entries=[{"key": _key(h="bb")}])])
        _post_events(
            client,
            [_batch(seq=6, tier="l2", backend="fs", entries=[{"key": _key(h="bb")}])],
        )

        data = client.get("/directory/stats").json()
        assert data["num_keys"] == 2
        assert data["num_placements"] == 3
        instance = data["instances"]["node-a"]
        assert instance["incarnation"] == 1
        assert instance["last_seq"] == 6
        assert instance["gap_detected"] is True
        assert instance["num_l1_keys"] == 2


# -- Token bindings ----------------------------------------------------------


def test_store_entry_with_tokens_populates_bindings():
    with _client() as client:
        entry = {"key": _key(), "size_bytes": 1024, "token_ids": [1, 2, 3]}
        data = _post_events(client, [_batch(entries=[entry])])
        assert data == {"applied": 1, "duplicates": 0, "stale": 0}

        key_directory = client.app.state.ctx.key_directory
        assert key_directory.get_token_ids([bytes.fromhex("aa")]) == [(1, 2, 3)]


# -- Listing + token ids -------------------------------------------------------


def test_list_keys_endpoint_filters_and_paginates():
    with _client() as client:
        _post_events(
            client,
            [
                _batch(
                    seq=1,
                    entries=[
                        {"key": _key(h="aa"), "size_bytes": 1, "token_ids": [1, 2]},
                        {"key": _key(h="bb"), "size_bytes": 2},
                    ],
                ),
                _batch(seq=2, tier="l2", backend="fs", entries=[{"key": _key(h="aa")}]),
            ],
        )

        data = client.get("/directory/keys").json()
        assert data["total"] == 2
        by_hash = {row["key"]["chunk_hash_hex"]: row for row in data["keys"]}
        assert by_hash["aa"]["num_tokens"] == 2
        assert len(by_hash["aa"]["placements"]) == 2
        assert by_hash["bb"]["num_tokens"] == 0

        l2 = client.get("/directory/keys", params={"tier": "l2"}).json()
        assert l2["total"] == 1
        assert [p["backend"] for p in l2["keys"][0]["placements"]] == ["fs"]

        page = client.get("/directory/keys", params={"offset": 1, "limit": 1}).json()
        assert page["total"] == 2
        assert len(page["keys"]) == 1

        assert client.get("/directory/keys", params={"offset": -1}).status_code == 422


def test_lookup_keys_form_returns_placements_and_tokens():
    with _client() as client:
        entry = {"key": _key(h="aa"), "size_bytes": 1, "token_ids": [1, 2, 3]}
        _post_events(client, [_batch(entries=[entry])])

        resp = client.post(
            "/directory/lookup", json={"keys": [_key(h="aa"), _key(h="ff")]}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks"] == 2
        results = body["results"]
        assert results[0]["token_ids"] == [1, 2, 3]
        assert len(results[0]["placements"]) == 1
        assert results[1]["token_ids"] == []  # unknown chunk
        assert results[1]["placements"] == []


def test_lookup_malformed_key_is_rejected():
    with _client() as client:
        resp = client.post("/directory/lookup", json={"keys": [_key(h="zz")]})
        assert resp.status_code == 422


def test_lookup_requires_exactly_one_form():
    with _client() as client:
        both = client.post(
            "/directory/lookup",
            json={"keys": [_key()], "token_ids": [1, 2], "model_name": "m"},
        )
        assert both.status_code == 422
        neither = client.post("/directory/lookup", json={})
        assert neither.status_code == 422
        no_model = client.post("/directory/lookup", json={"token_ids": [1, 2]})
        assert no_model.status_code == 422


# -- Request validation ------------------------------------------------------


def test_tier_all_is_rejected():
    with _client() as client:
        resp = client.post("/directory/events", json={"batches": [_batch(tier="all")]})
        assert resp.status_code == 422


def test_seq_zero_is_rejected():
    with _client() as client:
        resp = client.post("/directory/events", json={"batches": [_batch(seq=0)]})
        assert resp.status_code == 422


def test_malformed_key_hex_is_rejected():
    with _client() as client:
        resp = client.post(
            "/directory/events",
            json={"batches": [_batch(entries=[{"key": _key(h="zz")}])]},
        )
        assert resp.status_code == 422
