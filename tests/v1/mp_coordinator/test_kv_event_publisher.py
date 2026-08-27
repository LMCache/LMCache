# SPDX-License-Identifier: Apache-2.0
"""Tests for the engine-format (vLLM KV event) publisher: wire layout of
the msgpack payload against the positional field order llm-d's vLLM
adapter parses, golden fixtures, ZMQ framing / topic identity, fencing,
replay, and its mount behind the coordinator's ingest gate."""

# Standard
from pathlib import Path
import os
import struct
import time

# Third Party
from fastapi.testclient import TestClient
import msgspec
import pytest
import zmq

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.kv_event_publisher import (
    ZmqKVEventPublisher,
    emitter_id_for,
    encode_all_blocks_cleared,
    encode_batch,
    medium_for,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "kv_events"
_TS = 1700000000.5


def _hash(byte: int) -> bytes:
    """A 32-byte chunk hash (blake3-sized), like production keys carry."""
    return bytes([byte]) * 32


def _store_entry(
    byte: int, token_ids: list[int], parent: int = 0, model: str = "m"
) -> CacheEventEntry:
    key = ObjectKey(chunk_hash=_hash(byte), model_name=model, kv_rank=0)
    return CacheEventEntry(
        key=key.to_encoded_object_key(),
        size_bytes=1024,
        token_ids=token_ids,
        token_offset=0,
        parent_hash_hex=_hash(parent).hex() if parent else "",
    )


def _delete_entry(byte: int, model: str = "m") -> CacheEventEntry:
    key = ObjectKey(chunk_hash=_hash(byte), model_name=model, kv_rank=0)
    return CacheEventEntry(key=key.to_encoded_object_key())


def _batch(
    event_type: CacheEventType,
    entries: list[CacheEventEntry],
    tier: Tier = Tier.L1,
    backend: str = "dram",
    shared: bool = False,
    instance_id: str = "node:n1",
    seq: int = 1,
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=1,
        seq=seq,
        event_type=event_type,
        tier=tier,
        backend=backend,
        entries=entries,
        shared=shared,
        ts=_TS,
    )


def _decode(payload: bytes) -> list:
    return msgspec.msgpack.decode(payload)


# -- Wire layout ----------------------------------------------------------------


def test_store_at_offset_zero_has_no_parent():
    model, payload = encode_batch(
        _batch(CacheEventType.STORE, [_store_entry(1, [10, 11, 12, 13])])
    )
    assert model == "m"
    ts, events, dp_rank = _decode(payload)
    assert dp_rank is None
    assert ts == _TS
    # vLLM BlockStored positional layout, as read by llm-d's vllm_adapter.go:
    # [tag, block_hashes, parent_block_hash, token_ids, block_size, lora_id, medium]
    assert events == [
        ["BlockStored", [_hash(1)], None, [10, 11, 12, 13], 4, None, "lmcache-l1"]
    ]


def test_store_chain_carries_parent_hash():
    _, payload = encode_batch(
        _batch(
            CacheEventType.STORE,
            [_store_entry(1, [1, 2]), _store_entry(2, [3, 4], parent=1)],
        )
    )
    _, events, _ = _decode(payload)
    assert [e[0] for e in events] == ["BlockStored", "BlockStored"]
    assert events[0][2] is None
    assert events[1][1] == [_hash(2)] and events[1][2] == _hash(1)
    # No HMA fields (group_idx, ...) are ever appended.
    assert all(len(e) == 7 for e in events)


def test_delete_becomes_one_block_removed():
    _, payload = encode_batch(
        _batch(CacheEventType.DELETE, [_delete_entry(1), _delete_entry(2)])
    )
    _, events, _ = _decode(payload)
    # [tag, block_hashes, medium]
    assert events == [["BlockRemoved", [_hash(1), _hash(2)], "lmcache-l1"]]


def test_shared_l2_store_uses_backend_medium():
    _, payload = encode_batch(
        _batch(
            CacheEventType.STORE,
            [_store_entry(1, [1])],
            tier=Tier.L2,
            backend="FS",
            shared=True,
        )
    )
    _, events, _ = _decode(payload)
    assert events[0][6] == "lmcache-l2-fs"


def test_medium_is_stable_between_store_and_delete():
    assert medium_for(Tier.L1, "dram") == medium_for(Tier.L1, "cxl") == "lmcache-l1"
    assert medium_for(Tier.L2, "mooncake") == "lmcache-l2-mooncake"


def test_emitter_identity_is_instance_or_pool():
    private = _batch(CacheEventType.STORE, [_store_entry(1, [1])])
    assert emitter_id_for(private) == "node:n1"
    shared = _batch(
        CacheEventType.DELETE,
        [_delete_entry(1)],
        tier=Tier.L2,
        backend="FS",
        shared=True,
    )
    assert emitter_id_for(shared) == "pool:fs"


def test_all_blocks_cleared_layout():
    assert _decode(encode_all_blocks_cleared(_TS)) == [
        _TS,
        [["AllBlocksCleared"]],
        None,
    ]


def test_tokenless_store_is_skipped_but_delete_never_is():
    _, payload = encode_batch(_batch(CacheEventType.STORE, [_store_entry(1, [])]))
    assert payload == b""

    _, payload = encode_batch(
        _batch(CacheEventType.STORE, [_store_entry(1, []), _store_entry(2, [7])])
    )
    _, events, _ = _decode(payload)
    assert [e[1] for e in events] == [[_hash(2)]]

    _, payload = encode_batch(_batch(CacheEventType.DELETE, [_delete_entry(1)]))
    assert _decode(payload)[1][0][1] == [_hash(1)]


def test_non_placement_and_malformed_batches_rejected():
    with pytest.raises(ValueError):
        encode_batch(
            CacheEventBatch(
                instance_id="node-a",
                incarnation=1,
                seq=1,
                event_type=CacheEventType.ACCESS,
                tier=Tier.L1,
                backend="",
                entries=[_delete_entry(1)],
            )
        )
    with pytest.raises(ValueError):
        encode_batch(_batch(CacheEventType.DELETE, []))
    with pytest.raises(ValueError):
        encode_batch(
            CacheEventBatch(
                instance_id="node-a",
                incarnation=1,
                seq=1,
                event_type=CacheEventType.CONFIG,
                tier=Tier.L1,
                backend="dram",
            )
        )
    with pytest.raises(ValueError, match="several models"):
        encode_batch(
            _batch(
                CacheEventType.DELETE,
                [_delete_entry(1, model="a"), _delete_entry(2, model="b")],
            )
        )


# -- Golden fixtures (shared with llm-d's adapter tests) ------------------------


_GOLDEN = {
    "store_offset0": _batch(CacheEventType.STORE, [_store_entry(1, [1, 2, 3, 4])]),
    "store_offset256_parent": _batch(
        CacheEventType.STORE, [_store_entry(2, [5, 6, 7, 8], parent=1)]
    ),
    "delete": _batch(CacheEventType.DELETE, [_delete_entry(1), _delete_entry(2)]),
    "store_shared_l2": _batch(
        CacheEventType.STORE,
        [_store_entry(3, [9, 10, 11, 12])],
        tier=Tier.L2,
        backend="fs",
        shared=True,
    ),
}


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_golden_fixture(name):
    """The encoded bytes match the checked-in fixture. Set
    ``LMCACHE_UPDATE_FIXTURES=1`` to regenerate after an intended change."""
    _, payload = encode_batch(_GOLDEN[name])
    path = _FIXTURES / f"{name}.msgpack"
    if os.environ.get("LMCACHE_UPDATE_FIXTURES") == "1":
        path.write_bytes(payload)
    assert payload == path.read_bytes()


# -- ZMQ transport ---------------------------------------------------------------


def _subscriber(ctx: zmq.Context, endpoint: str, topic: bytes = b"kv@") -> zmq.Socket:
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVTIMEO, 5000)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, topic)
    time.sleep(0.3)  # PUB/SUB slow-joiner: let the subscription propagate
    return sub


@pytest.fixture
def zmq_ctx():
    ctx = zmq.Context()
    yield ctx
    ctx.term()


def test_consume_publishes_three_frame_messages_per_identity(zmq_ctx):
    publisher = ZmqKVEventPublisher("tcp://127.0.0.1:*")
    try:
        sub = _subscriber(zmq_ctx, publisher.endpoint)
        publisher.consume(_batch(CacheEventType.STORE, [_store_entry(1, [1, 2])]))
        # Ignored: not a placement.
        publisher.consume(
            CacheEventBatch(
                instance_id="node:n1",
                incarnation=1,
                seq=2,
                event_type=CacheEventType.ACCESS,
                tier=Tier.L1,
                backend="",
                entries=[_delete_entry(1)],
            )
        )
        publisher.consume(
            _batch(
                CacheEventType.STORE,
                [_store_entry(2, [3])],
                tier=Tier.L2,
                backend="fs",
                shared=True,
                instance_id="node:n2",
            )
        )
        publisher.consume(_batch(CacheEventType.DELETE, [_delete_entry(1)]))
        messages = [sub.recv_multipart() for _ in range(3)]
        sub.close()
    finally:
        publisher.close()

    # [topic, seq (8-byte big-endian, from 0), payload]; private placements
    # go out under the reporting instance, shared ones under the pool.
    assert [m[0] for m in messages] == [
        b"kv@node:n1@m",
        b"kv@pool:fs@m",
        b"kv@node:n1@m",
    ]
    assert [struct.unpack(">Q", m[1])[0] for m in messages] == [0, 1, 2]
    assert [_decode(m[2])[1][0][0] for m in messages] == [
        "BlockStored",
        "BlockStored",
        "BlockRemoved",
    ]


def test_fence_clears_every_model_of_the_instance_only(zmq_ctx):
    publisher = ZmqKVEventPublisher("tcp://127.0.0.1:*")
    try:
        sub = _subscriber(zmq_ctx, publisher.endpoint)
        publisher.consume(_batch(CacheEventType.STORE, [_store_entry(1, [1])]))
        publisher.consume(
            _batch(CacheEventType.STORE, [_store_entry(2, [2], model="m2")], seq=2)
        )
        publisher.consume(
            _batch(
                CacheEventType.STORE,
                [_store_entry(3, [3])],
                tier=Tier.L2,
                backend="fs",
                shared=True,
            )
        )
        publisher.consume(
            _batch(CacheEventType.STORE, [_store_entry(4, [4])], instance_id="node:n2")
        )
        for _ in range(4):
            sub.recv_multipart()
        publisher.fence_instance("node:n1")
        publisher.fence_instance("node:n1")  # forgotten: nothing more
        publisher.fence_instance("unknown")  # never seen: nothing
        cleared = [sub.recv_multipart() for _ in range(2)]
        sub.setsockopt(zmq.RCVTIMEO, 300)
        with pytest.raises(zmq.Again):
            sub.recv_multipart()
        sub.close()
    finally:
        publisher.close()

    # One AllBlocksCleared per model n1 published under; pool:fs and n2 untouched.
    assert sorted(m[0] for m in cleared) == [b"kv@node:n1@m", b"kv@node:n1@m2"]
    assert all(_decode(m[2])[1] == [["AllBlocksCleared"]] for m in cleared)


def test_replay_returns_buffered_messages_from_start_seq(zmq_ctx):
    publisher = ZmqKVEventPublisher(
        "tcp://127.0.0.1:*",
        replay_endpoint="tcp://127.0.0.1:*",
        replay_depth=8,
    )
    try:
        # No subscriber: PUB drops, but the ring still records every send.
        for _ in range(3):
            publisher.consume(_batch(CacheEventType.DELETE, [_delete_entry(1)]))

        dealer = zmq_ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.setsockopt(zmq.RCVTIMEO, 5000)
        dealer.connect(publisher.replay_endpoint)
        dealer.send_multipart([b"", struct.pack(">Q", 1)])
        replies = []
        while True:
            frames = dealer.recv_multipart()
            replies.append(frames)
            if frames[-1] == b"":
                break
        dealer.close()
    finally:
        publisher.close()

    *events, end = replies
    # Each replayed message is the live frame set behind an empty delimiter.
    assert [f[0] for f in events] == [b"", b""]
    assert [f[1] for f in events] == [b"kv@node:n1@m"] * 2
    assert [struct.unpack(">Q", f[2])[0] for f in events] == [1, 2]
    # End marker: empty topic, seq -1, empty payload.
    assert end == [b"", b"", struct.pack(">q", -1), b""]


def test_consume_after_close_is_dropped_not_raised():
    publisher = ZmqKVEventPublisher("tcp://127.0.0.1:*")
    publisher.close()
    publisher.consume(_batch(CacheEventType.DELETE, [_delete_entry(1)]))
    publisher.close()  # idempotent


@pytest.mark.parametrize("kwargs", [{"replay_depth": 0}, {"hwm": 0}])
def test_invalid_construction_rejected(kwargs):
    with pytest.raises(ValueError):
        ZmqKVEventPublisher("tcp://127.0.0.1:*", **kwargs)


# -- Mount behind the coordinator's ingest gate ----------------------------------


def _events_body(batch: CacheEventBatch) -> dict:
    return {
        "batches": [
            {
                "instance_id": batch.instance_id,
                "incarnation": batch.incarnation,
                "seq": batch.seq,
                "event_type": batch.event_type.value,
                "tier": batch.tier.value,
                "backend": batch.backend,
                "shared": batch.shared,
                "ts": batch.ts,
                "entries": [
                    {
                        "key": {
                            "chunk_hash_hex": e.key.chunk_hash_hex,
                            "model_name": e.key.model_name,
                            "kv_rank": e.key.kv_rank,
                        },
                        "size_bytes": e.size_bytes,
                        "token_ids": list(e.token_ids),
                        "token_offset": e.token_offset,
                        "parent_hash_hex": e.parent_hash_hex,
                    }
                    for e in batch.entries
                ],
            }
        ]
    }


def test_coordinator_publishes_admitted_events_and_fences_restarts(zmq_ctx):
    port = _free_port()
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        kv_events_endpoint=f"tcp://127.0.0.1:{port}",
    )
    with TestClient(create_app(config)) as client:
        sub = _subscriber(zmq_ctx, f"tcp://127.0.0.1:{port}")
        store = _batch(CacheEventType.STORE, [_store_entry(1, [1, 2], parent=0)])
        assert client.post("/events", json=_events_body(store)).json()["applied"] == 1
        # A replayed seq is dropped by the gate and never reaches the router.
        assert (
            client.post("/events", json=_events_body(store)).json()["duplicates"] == 1
        )
        restarted = CacheEventBatch(
            instance_id="node:n1",
            incarnation=2,
            seq=1,
            event_type=CacheEventType.DELETE,
            tier=Tier.L1,
            backend="dram",
            entries=[_delete_entry(1)],
            ts=_TS,
        )
        assert (
            client.post("/events", json=_events_body(restarted)).json()["applied"] == 1
        )
        messages = [sub.recv_multipart() for _ in range(3)]
        sub.setsockopt(zmq.RCVTIMEO, 300)
        with pytest.raises(zmq.Again):
            sub.recv_multipart()
        sub.close()

    assert [m[0] for m in messages] == [b"kv@node:n1@m"] * 3
    # Store, then the incarnation bump fences n1 (AllBlocksCleared) before
    # the new incarnation's first batch is published.
    assert [_decode(m[2])[1][0][0] for m in messages] == [
        "BlockStored",
        "AllBlocksCleared",
        "BlockRemoved",
    ]


def _free_port() -> int:
    # Standard
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
