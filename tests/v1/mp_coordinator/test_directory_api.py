# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator ``/directory`` REST API (cache-event
ingestion, placement lookup, and stats)."""

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.schemas import encode_tokens


def _client() -> TestClient:
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    return TestClient(create_app(config))


def _blend_client() -> TestClient:
    """A coordinator with fragment (blend) lookup turned on."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        enable_blend_lookup=True,
    )
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
    resp = client.post("/events", json={"batches": batches})
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


def test_stats_reports_counts_and_per_instance_l1_keys():
    """Directory contents only — per-emitter stream state (incarnation,
    seq, gap flag) lives on the ingest gate and is not exposed here."""
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
        # Both keys got an L1 placement from node-a; the third batch added
        # an L2 placement, which the L1 index does not count.
        assert data["l1_keys_by_instance"]["node-a"] == 2
        assert "instances" not in data


# -- Token bindings ----------------------------------------------------------


def test_store_entry_with_tokens_populates_bindings():
    with _client() as client:
        entry = {
            "key": _key(),
            "size_bytes": 1024,
            "token_ids": [1, 2, 3],
            "token_offset": 256,
        }
        data = _post_events(client, [_batch(entries=[entry])])
        assert data == {"applied": 1, "duplicates": 0, "stale": 0}

        key_directory = client.app.state.ctx.key_directory
        assert key_directory.get_token_ids([bytes.fromhex("aa")]) == [(1, 2, 3)]


def test_lookup_returns_the_chunks_token_ids():
    """The normal lookup reports content only — positions are the blend
    lookup's concern."""
    with _client() as client:
        entry = {"key": _key(), "size_bytes": 1024, "token_ids": [1, 2, 3]}
        _post_events(client, [_batch(entries=[entry])])

        resp = client.post("/directory/lookup", json={"keys": [_key()]})
        assert resp.status_code == 200
        [result] = resp.json()["results"]
        assert result["token_ids"] == [1, 2, 3]
        assert "token_offset" not in result


# -- Blend (fragment) lookup ---------------------------------------------------


def _chunk_tokens(first: int, count: int) -> list[int]:
    return list(range(first, first + count))


def _blend_lookup(client: TestClient, tokens: list[int]) -> dict:
    resp = client.post(
        "/directory/blend-lookup", json={"tokens_b64": encode_tokens(tokens)}
    )
    assert resp.status_code == 200
    return resp.json()


def test_blend_lookup_finds_a_stored_chunk_mid_query():
    """The fragment form does not require a prefix: content is found at
    whatever offset it sits at in the query."""
    chunk_size = MPCoordinatorConfig().chunk_size
    content = _chunk_tokens(1000, chunk_size)
    with _blend_client() as client:
        entry = {
            "key": _key(),
            "size_bytes": 1024,
            "token_ids": content,
            "token_offset": 512,
        }
        _post_events(client, [_batch(entries=[entry])])

        data = _blend_lookup(client, [7, 8, 9] + content + [11, 12])

        assert data["matches"] == [{"chunk_hash": "aa", "old_st": 512, "cur_st": 3}]


def test_blend_lookup_without_a_match_is_empty():
    chunk_size = MPCoordinatorConfig().chunk_size
    with _blend_client() as client:
        data = _blend_lookup(client, _chunk_tokens(1, chunk_size + 5))
        assert data["matches"] == []


def test_blend_lookup_rejects_a_malformed_token_buffer():
    with _blend_client() as client:
        resp = client.post("/directory/blend-lookup", json={"tokens_b64": "not!b64"})
        assert resp.status_code == 422


def test_blend_lookup_stops_matching_a_deleted_chunk():
    chunk_size = MPCoordinatorConfig().chunk_size
    content = _chunk_tokens(1000, chunk_size)
    with _blend_client() as client:
        entry = {
            "key": _key(),
            "size_bytes": 1024,
            "token_ids": content,
            "token_offset": 0,
        }
        _post_events(client, [_batch(seq=1, entries=[entry])])
        assert _blend_lookup(client, content)["matches"]

        _post_events(
            client,
            [_batch(seq=2, event_type="delete", entries=[{"key": _key()}])],
        )
        assert _blend_lookup(client, content)["matches"] == []


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
        resp = client.post("/events", json={"batches": [_batch(tier="all")]})
        assert resp.status_code == 422


def test_seq_zero_is_rejected():
    with _client() as client:
        resp = client.post("/events", json={"batches": [_batch(seq=0)]})
        assert resp.status_code == 422


def test_malformed_key_hex_is_rejected():
    with _client() as client:
        resp = client.post(
            "/events",
            json={"batches": [_batch(entries=[{"key": _key(h="zz")}])]},
        )
        assert resp.status_code == 422


def test_stats_reports_blend_index_counts():
    """The e2e ladder needs a way to confirm the fleet index actually got
    populated from the event stream."""
    chunk_size = MPCoordinatorConfig().chunk_size
    content = _chunk_tokens(1000, chunk_size)
    with _blend_client() as client:
        assert client.get("/directory/stats").json()["blend"] == {
            "num_contents": 0,
            "num_chunks": 0,
            "table_size": 1024,
        }

        entry = {
            "key": _key(),
            "size_bytes": 1024,
            "token_ids": content,
            "token_offset": 0,
        }
        _post_events(client, [_batch(entries=[entry])])

        data = client.get("/directory/stats").json()["blend"]
        assert data["num_contents"] == 1
        assert data["num_chunks"] == 1
