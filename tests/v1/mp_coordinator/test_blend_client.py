# SPDX-License-Identifier: Apache-2.0
"""Tests for the blend-server side coordinator client.

The HTTP layer is replaced by an injected ``request_fn`` backed by a real
:class:`KeyDirectory`, so the queue/daemon/poll state machine is exercised
end-to-end against the actual fragment lookup without a network or server.
"""

# Standard
from collections.abc import Callable
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    BlendNamespace,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.blend_client import (
    PENDING,
    BlendCoordinatorClient,
)
from lmcache.v1.mp_coordinator.schemas import decode_tokens, encode_tokens
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory

CHUNK = 3
MODEL = "m"
WORLD_SIZE = 1
NS = BlendNamespace(model_name=MODEL, world_size=WORLD_SIZE)


def _directory_request(
    directory: KeyDirectory,
) -> Callable[[str, str, dict], dict]:
    """request_fn driving a real directory, mirroring the coordinator router."""

    def request(method: str, path: str, payload: dict) -> dict:
        if method == "POST" and path == "/directory/blend-lookup":
            namespace = BlendNamespace(
                model_name=payload["model_name"],
                cache_salt=payload["cache_salt"],
                world_size=payload["world_size"],
            )
            matches = directory.blend_match(
                decode_tokens(payload["tokens_b64"]), namespace
            )
            return {
                "matches": [
                    {
                        "chunk_hash": m.chunk_hash.hex(),
                        "old_st": m.old_st,
                        "cur_st": m.cur_st,
                    }
                    for m in matches
                ]
            }
        raise AssertionError(f"unexpected {method} {path}")

    return request


def _stored(directory: KeyDirectory, chunks: list[list[int]]) -> list[bytes]:
    """Feed ``chunks`` in as cache events; return their chunk hashes."""
    hashes = [bytes([index + 1]) * 4 for index in range(len(chunks))]
    for seq, (chunk_hash, tokens) in enumerate(zip(hashes, chunks, strict=True), 1):
        directory.consume(
            CacheEventBatch(
                instance_id="node-a",
                incarnation=1,
                seq=seq,
                event_type=CacheEventType.STORE,
                tier=Tier.L1,
                backend="dram",
                entries=[
                    CacheEventEntry(
                        key=ObjectKey(
                            chunk_hash=chunk_hash,
                            model_name=MODEL,
                            kv_rank=ObjectKey.ComputeKVRank(
                                world_size=WORLD_SIZE,
                                global_rank=0,
                                local_world_size=WORLD_SIZE,
                                local_rank=0,
                            ),
                        ).to_encoded_object_key(),
                        size_bytes=100,
                        token_ids=tokens,
                        token_offset=(seq - 1) * CHUNK,
                    )
                ],
            )
        )
    return hashes


def _directory() -> KeyDirectory:
    directory = KeyDirectory()
    directory.enable_blend_lookup(chunk_size=CHUNK, probe_stride=1)
    return directory


def _wait_match(client: BlendCoordinatorClient, rid: str, timeout: float = 2.0):
    end = time.time() + timeout
    while time.time() < end:
        v = client.poll_match(rid)
        if isinstance(v, list):
            return v
        time.sleep(0.005)
    return client.poll_match(rid)


def test_match_finds_chunks_the_event_stream_reported():
    directory = _directory()
    hashes = _stored(directory, [[1, 2, 3], [4, 5, 6]])
    client = BlendCoordinatorClient(request_fn=_directory_request(directory))
    try:
        client.submit_match("r1", [1, 2, 3, 4, 5, 6], NS)
        matches = _wait_match(client, "r1")
        assert isinstance(matches, list)
        # chunk_hash arrives as bytes, matching a local CBMatchResult.hash.
        assert [(x.chunk_hash, x.old_st, x.cur_st) for x in matches] == [
            (hashes[0], 0, 0),
            (hashes[1], 3, 3),
        ]
    finally:
        client.close()


def test_match_is_empty_when_nothing_is_cached():
    client = BlendCoordinatorClient(request_fn=_directory_request(_directory()))
    try:
        client.submit_match("r1", [1, 2, 3], NS)
        assert _wait_match(client, "r1") == []
    finally:
        client.close()


def test_poll_none_before_submit():
    client = BlendCoordinatorClient(request_fn=lambda mth, p, b: {})
    try:
        assert client.poll_match("never") is None
    finally:
        client.close()


def test_submit_is_idempotent():
    directory = _directory()
    _stored(directory, [[1, 2, 3]])
    client = BlendCoordinatorClient(request_fn=_directory_request(directory))
    try:
        client.submit_match("r1", [1, 2, 3], NS)
        client.submit_match("r1", [1, 2, 3], NS)  # no-op
        matches = _wait_match(client, "r1")
        assert isinstance(matches, list) and len(matches) == 1
    finally:
        client.close()


def test_match_error_degrades_to_empty():
    def boom(method: str, path: str, payload: dict) -> dict:
        raise RuntimeError("coordinator down")

    client = BlendCoordinatorClient(request_fn=boom)
    try:
        client.submit_match("r1", [1, 2, 3], NS)
        matches = _wait_match(client, "r1")
        assert matches == []  # failure -> local-only, never hangs
    finally:
        client.close()


def test_take_match_clears():
    directory = _directory()
    _stored(directory, [[1, 2, 3]])
    client = BlendCoordinatorClient(request_fn=_directory_request(directory))
    try:
        client.submit_match("r1", [1, 2, 3], NS)
        _wait_match(client, "r1")
        client.take_match("r1")
        assert client.poll_match("r1") is None
    finally:
        client.close()


def test_maybe_create():
    kwargs = {"timeout": 1.0, "match_concurrency": 8}
    assert BlendCoordinatorClient.maybe_create("", **kwargs) is None
    assert BlendCoordinatorClient.maybe_create("   ", **kwargs) is None
    assert BlendCoordinatorClient.maybe_create(None, **kwargs) is None

    client = BlendCoordinatorClient.maybe_create("http://coord:9300", **kwargs)
    assert client is not None
    assert client.match_budget_s == 1.0
    client.close()

    client = BlendCoordinatorClient.maybe_create(
        "http://coord:9300", timeout=1.5, match_concurrency=2
    )
    assert client is not None
    assert client.match_budget_s == 1.5
    client.close()


def test_maybe_create_rejects_bad_concurrency():
    with pytest.raises(ValueError):
        BlendCoordinatorClient.maybe_create(
            "http://coord:9300", timeout=1.0, match_concurrency=0
        )


def test_pending_sentinel_distinct():
    assert PENDING is not None and not isinstance(PENDING, list)


def test_match_carries_the_namespace_to_the_coordinator():
    """The server's own model, salt, and world size travel with the query
    so the coordinator can scope matches to keys this server can expand."""
    sent: list[dict] = []

    def capture(method: str, path: str, payload: dict) -> dict:
        sent.append(payload)
        return {"matches": []}

    client = BlendCoordinatorClient(request_fn=capture)
    try:
        namespace = BlendNamespace(
            model_name="llama", cache_salt="tenant-a", world_size=4
        )
        client.submit_match("r1", [1, 2, 3], namespace)
        _wait_match(client, "r1")
        assert sent == [
            {
                "tokens_b64": encode_tokens([1, 2, 3]),
                "model_name": "llama",
                "cache_salt": "tenant-a",
                "world_size": 4,
            }
        ]
    finally:
        client.close()


def test_another_namespace_gets_no_match_for_the_same_content():
    directory = _directory()
    _stored(directory, [[1, 2, 3]])
    client = BlendCoordinatorClient(request_fn=_directory_request(directory))
    try:
        client.submit_match("r1", [1, 2, 3], BlendNamespace(model_name="other"))
        assert _wait_match(client, "r1") == []
    finally:
        client.close()
