# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for ``GET /requests/{instance_id}/{request_id}``:
cache events in over ``POST /events``, a request's tokens and chunk
placements out."""

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

# Raw hashes with non-UTF-8 bytes, so the hex wire form is exercised.
HASH_A = bytes([0xFF, 0x00, 0xA1])
HASH_B = bytes([0xFE, 0x01, 0xB2])
HASH_C = bytes([0xFD, 0x02, 0xC3])
TOKENS_A = [1, 2, 3, 4]
TOKENS_B = [5, 6, 7, 8]
TOKENS_C = [9, 10, 11, 12]


def _client(extra_config: dict | None = None) -> TestClient:
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        extra_config=extra_config or {},
    )
    return TestClient(create_app(config))


def _key(chunk_hash: bytes, rank: int = 0, salt: str = "") -> dict:
    return {
        "chunk_hash_hex": chunk_hash.hex(),
        "model_name": "m",
        "kv_rank": rank,
        "cache_salt": salt,
    }


def _store_entry(
    chunk_hash: bytes,
    tokens: list[int],
    offset: int,
    parent: bytes | None,
    request_ids: list[str],
    rank: int = 0,
    salt: str = "",
) -> dict:
    return {
        "key": _key(chunk_hash, rank, salt),
        "size_bytes": 1024,
        "token_ids": tokens,
        "token_offset": offset,
        "parent_hash_hex": parent.hex() if parent is not None else "",
        "request_ids": request_ids,
    }


def _batch(seq: int, event_type: str, entries: list[dict], tier: str = "l1") -> dict:
    return {
        "instance_id": "mp-0",
        "incarnation": 1,
        "seq": seq,
        "event_type": event_type,
        "tier": tier,
        "backend": "dram" if tier == "l1" else "fs",
        "entries": entries,
    }


def _post_events(client: TestClient, batches: list[dict]) -> None:
    resp = client.post("/events", json={"batches": batches})
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] == len(batches)


def _get_request(client: TestClient, request_id: str, instance_id: str = "mp-0"):
    return client.get(f"/requests/{instance_id}/{request_id}")


def _store_two_chunk_request(client: TestClient) -> None:
    """``req-1`` stores A then B, both at L1."""
    _post_events(
        client,
        [
            _batch(
                1,
                "store",
                [
                    _store_entry(HASH_A, TOKENS_A, 0, None, ["req-1"]),
                    _store_entry(HASH_B, TOKENS_B, 4, HASH_A, ["req-1"]),
                ],
            )
        ],
    )


def test_request_tokens_and_placements_are_recovered():
    with _client() as client:
        _store_two_chunk_request(client)

        resp = _get_request(client, "req-1")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["instance_id"] == "mp-0"
        assert body["request_id"] == "req-1"
        assert body["complete"] is True
        assert body["start_offset"] == 0
        assert body["token_ids"] == TOKENS_A + TOKENS_B
        assert [c["chunk_hash_hex"] for c in body["chunks"]] == [
            HASH_A.hex(),
            HASH_B.hex(),
        ]
        assert [c["token_offset"] for c in body["chunks"]] == [0, 4]
        assert [c["num_tokens"] for c in body["chunks"]] == [4, 4]
        for chunk in body["chunks"]:
            [key] = chunk["keys"]
            [placement] = key["placements"]
            assert placement["instance_id"] == "mp-0"
            assert placement["tier"] == "l1"


def test_every_rank_key_of_a_chunk_is_listed():
    with _client() as client:
        _post_events(
            client,
            [
                _batch(
                    1,
                    "store",
                    [
                        _store_entry(HASH_A, TOKENS_A, 0, None, ["req-1"], rank=0),
                        _store_entry(HASH_A, TOKENS_A, 0, None, ["req-1"], rank=1),
                    ],
                )
            ],
        )

        [chunk] = _get_request(client, "req-1").json()["chunks"]

        assert sorted(k["key"]["kv_rank"] for k in chunk["keys"]) == [0, 1]


def test_tokens_survive_eviction_with_empty_placements():
    with _client() as client:
        _store_two_chunk_request(client)
        _post_events(
            client,
            [_batch(2, "delete", [{"key": _key(HASH_A)}, {"key": _key(HASH_B)}])],
        )

        body = _get_request(client, "req-1").json()

        assert body["token_ids"] == TOKENS_A + TOKENS_B
        assert body["complete"] is True
        for chunk in body["chunks"]:
            [key] = chunk["keys"]
            assert key["placements"] == []


def test_prefix_stored_by_an_earlier_request_is_recovered():
    """``req-2`` hits A and B and stores only C; the parent chain still
    yields its full sequence."""
    with _client() as client:
        _store_two_chunk_request(client)
        _post_events(
            client,
            [
                _batch(
                    2, "store", [_store_entry(HASH_C, TOKENS_C, 8, HASH_B, ["req-2"])]
                )
            ],
        )

        body = _get_request(client, "req-2").json()

        assert body["complete"] is True
        assert body["token_ids"] == TOKENS_A + TOKENS_B + TOKENS_C


def test_tail_is_highest_offset_not_last_arrival():
    """An async L2 store of an earlier chunk arriving last must not cut
    the request short."""
    with _client() as client:
        _store_two_chunk_request(client)
        _post_events(
            client,
            [
                _batch(
                    2,
                    "store",
                    [_store_entry(HASH_A, TOKENS_A, 0, None, ["req-1"])],
                    tier="l2",
                )
            ],
        )

        body = _get_request(client, "req-1").json()

        assert body["token_ids"] == TOKENS_A + TOKENS_B


def test_a_chunk_submitted_by_two_requests_belongs_to_both():
    with _client() as client:
        _post_events(
            client,
            [
                _batch(
                    1,
                    "store",
                    [_store_entry(HASH_A, TOKENS_A, 0, None, ["req-1", "req-2"])],
                )
            ],
        )

        for request_id in ("req-1", "req-2"):
            assert _get_request(client, request_id).json()["token_ids"] == TOKENS_A


def test_missing_ancestor_reports_an_incomplete_sequence():
    """A chunk whose parent the index never saw starts the sequence at its
    own offset."""
    with _client() as client:
        _post_events(
            client,
            [
                _batch(
                    1, "store", [_store_entry(HASH_C, TOKENS_C, 8, HASH_B, ["req-3"])]
                )
            ],
        )

        body = _get_request(client, "req-3").json()

        assert body["complete"] is False
        assert body["start_offset"] == 8
        assert body["token_ids"] == TOKENS_C


def test_oldest_chunks_age_out_past_the_bound():
    with _client({"request_index_max_chunks": 2}) as client:
        _store_two_chunk_request(client)
        _post_events(
            client,
            [
                _batch(
                    2, "store", [_store_entry(HASH_C, TOKENS_C, 8, HASH_B, ["req-2"])]
                )
            ],
        )

        body = _get_request(client, "req-2").json()

        assert body["complete"] is False
        assert body["start_offset"] == 4
        assert body["token_ids"] == TOKENS_B + TOKENS_C


def test_a_shared_prefix_outlives_unrelated_stores():
    """Every request that extends a prefix keeps it fresh, so a popular
    system prompt is not the first chunk to age out."""
    with _client({"request_index_max_chunks": 3}) as client:
        hashes = [bytes([0xE0, i]) for i in range(4)]
        _post_events(
            client,
            [_batch(1, "store", [_store_entry(HASH_A, TOKENS_A, 0, None, ["req-0"])])]
            + [
                _batch(
                    seq,
                    "store",
                    [_store_entry(h, TOKENS_B, 4, HASH_A, [f"req-{seq}"])],
                )
                for seq, h in enumerate(hashes, start=2)
            ],
        )

        body = _get_request(client, "req-5").json()

        assert body["complete"] is True
        assert body["token_ids"] == TOKENS_A + TOKENS_B


def test_another_tenants_keys_of_the_same_tokens_are_not_listed():
    """Chunk hashes depend on tokens alone, so two salts storing the same
    prompt share a hash; each request sees only its own tenant's keys."""
    with _client() as client:
        _post_events(
            client,
            [
                _batch(
                    1,
                    "store",
                    [
                        _store_entry(HASH_A, TOKENS_A, 0, None, ["req-a"], salt="a"),
                        _store_entry(HASH_A, TOKENS_A, 0, None, ["req-b"], salt="b"),
                    ],
                )
            ],
        )

        for request_id, salt in (("req-a", "a"), ("req-b", "b")):
            [chunk] = _get_request(client, request_id).json()["chunks"]
            assert [k["key"]["cache_salt"] for k in chunk["keys"]] == [salt]


def test_request_ids_are_scoped_to_the_reporting_instance():
    with _client() as client:
        _store_two_chunk_request(client)

        assert _get_request(client, "req-1", instance_id="mp-1").status_code == 404


def test_unknown_request_is_404():
    with _client() as client:
        assert _get_request(client, "nope").status_code == 404


def test_store_without_request_ids_records_no_request():
    """Emitters predating the field send neither request ids nor parents."""
    with _client() as client:
        _post_events(
            client,
            [
                _batch(
                    1,
                    "store",
                    [{"key": _key(HASH_A), "size_bytes": 1, "token_ids": TOKENS_A}],
                )
            ],
        )

        assert _get_request(client, "req-1").status_code == 404
