# SPDX-License-Identifier: Apache-2.0

"""Tests for DynamoKvPublisher (store/evict -> Dynamo KV events coordinator)."""

# Standard
from collections.abc import Iterator
from typing import cast
import socket as _socket

# Third Party
import pytest

pytest.importorskip("vllm.distributed.kv_events", reason="vllm required")
pytest.importorskip("numba", reason="numba required by TokenHasher")

# Third Party
from vllm.distributed.kv_events import (  # noqa: E402
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
import msgspec  # noqa: E402
import zmq  # noqa: E402

# First Party
from lmcache.v1.mp_observability.dynamo.block_builder import build_blocks  # noqa: E402
from lmcache.v1.mp_observability.dynamo.emitter import (  # noqa: E402
    DynamoKvEventEmitter,
)
from lmcache.v1.mp_observability.dynamo.publisher import (  # noqa: E402
    DynamoKvPublisher,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher  # noqa: E402

DECODER = msgspec.msgpack.Decoder(type=KVEventBatch)
KV_BLOCK_SIZE = 16


def _free_port() -> int:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _recv_batch(sub: zmq.Socket, timeout_ms: int = 3000) -> KVEventBatch:
    """Receive one multipart message and return the decoded KVEventBatch."""
    if not sub.poll(timeout_ms):
        raise AssertionError("no message received before timeout")
    _topic, _seq, payload = sub.recv_multipart()
    return cast(KVEventBatch, DECODER.decode(payload))


@pytest.fixture
def transport() -> Iterator[tuple[DynamoKvEventEmitter, zmq.Socket, TokenHasher]]:
    """A bound emitter, a connected+subscribed SUB socket, and a real hasher.

    Tests build their own DynamoKvPublisher (so each can pick its chunk_size)
    using the yielded emitter. The slow-joiner warmup guarantees the first
    asserted event is not dropped.
    """
    bind = f"tcp://127.0.0.1:{_free_port()}"
    emitter = DynamoKvEventEmitter(bind)
    hasher = TokenHasher(chunk_size=KV_BLOCK_SIZE, hash_algorithm="blake3")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(bind)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    connected = False
    for _ in range(100):
        emitter.publish_removed([0])
        if sub.poll(50):
            connected = True
            break
    assert connected, "SUB never connected to PUB"
    while sub.poll(0):
        sub.recv_multipart()

    try:
        yield emitter, sub, hasher
    finally:
        sub.close(linger=0)
        ctx.term()
        emitter.close()


def test_on_store_basic_start_zero(
    transport: tuple[DynamoKvEventEmitter, zmq.Socket, TokenHasher],
) -> None:
    emitter, sub, hasher = transport
    publisher = DynamoKvPublisher(emitter, hasher, KV_BLOCK_SIZE, KV_BLOCK_SIZE)

    token_ids = list(range(KV_BLOCK_SIZE))  # exactly one block
    chunk_hashes = [b"chunk-0"]
    publisher.on_store(token_ids, 0, KV_BLOCK_SIZE, chunk_hashes)

    batch = _recv_batch(sub)
    assert len(batch.events) == 1
    event = batch.events[0]
    assert isinstance(event, BlockStored)
    assert event.parent_block_hash is None
    assert event.token_ids == token_ids
    assert event.block_size == KV_BLOCK_SIZE

    # Block hashes match a direct build_blocks over the same tokens.
    _, blocks = build_blocks(token_ids, KV_BLOCK_SIZE, hasher.none_hash, hasher)
    expected_hashes = [h for _, h in blocks]
    assert event.block_hashes == expected_hashes

    # Recording is observable: evicting the chunk returns those same hashes.
    publisher.on_evict(chunk_hashes)
    evict_batch = _recv_batch(sub)
    assert isinstance(evict_batch.events[0], BlockRemoved)
    assert evict_batch.events[0].block_hashes == expected_hashes


def test_on_store_with_prefix(
    transport: tuple[DynamoKvEventEmitter, zmq.Socket, TokenHasher],
) -> None:
    emitter, sub, hasher = transport
    publisher = DynamoKvPublisher(emitter, hasher, KV_BLOCK_SIZE, KV_BLOCK_SIZE)

    # 3 blocks total; only the last block [32, 48) is newly stored.
    token_ids = list(range(3 * KV_BLOCK_SIZE))
    start, end = 2 * KV_BLOCK_SIZE, 3 * KV_BLOCK_SIZE
    publisher.on_store(token_ids, start, end, [b"chunk-2"])

    batch = _recv_batch(sub)
    event = batch.events[0]
    assert isinstance(event, BlockStored)

    # Expected via build_blocks over the full prefix: parent is block[1]'s
    # hash; the new block is block[2].
    _, all_blocks = build_blocks(token_ids, KV_BLOCK_SIZE, hasher.none_hash, hasher)
    assert event.parent_block_hash == all_blocks[1][1]
    assert event.block_hashes == [all_blocks[2][1]]
    assert event.token_ids == token_ids[start:end]


def test_chunk_to_blocks_grouping(
    transport: tuple[DynamoKvEventEmitter, zmq.Socket, TokenHasher],
) -> None:
    emitter, sub, hasher = transport
    chunk_size = 2 * KV_BLOCK_SIZE  # 2 blocks per chunk
    publisher = DynamoKvPublisher(emitter, hasher, KV_BLOCK_SIZE, chunk_size)

    # 4 blocks across 2 chunks.
    token_ids = list(range(4 * KV_BLOCK_SIZE))
    publisher.on_store(token_ids, 0, 4 * KV_BLOCK_SIZE, [b"chunk-0", b"chunk-1"])
    _recv_batch(sub)  # the BlockStored

    _, all_blocks = build_blocks(token_ids, KV_BLOCK_SIZE, hasher.none_hash, hasher)
    all_hashes = [h for _, h in all_blocks]

    # Evict chunk-0 -> exactly its 2 blocks (the first segment).
    publisher.on_evict([b"chunk-0"])
    batch0 = _recv_batch(sub)
    assert isinstance(batch0.events[0], BlockRemoved)
    assert batch0.events[0].block_hashes == all_hashes[0:2]

    # Evict chunk-1 -> exactly its 2 blocks (the second segment).
    publisher.on_evict([b"chunk-1"])
    batch1 = _recv_batch(sub)
    assert batch1.events[0].block_hashes == all_hashes[2:4]


def test_on_evict_removes_and_clears(
    transport: tuple[DynamoKvEventEmitter, zmq.Socket, TokenHasher],
) -> None:
    emitter, sub, hasher = transport
    publisher = DynamoKvPublisher(emitter, hasher, KV_BLOCK_SIZE, KV_BLOCK_SIZE)

    token_ids = list(range(2 * KV_BLOCK_SIZE))
    chunk_hashes = [b"c0", b"c1"]
    publisher.on_store(token_ids, 0, 2 * KV_BLOCK_SIZE, chunk_hashes)
    stored = _recv_batch(sub)
    stored_hashes = stored.events[0].block_hashes

    publisher.on_evict(chunk_hashes)
    removed = _recv_batch(sub)
    assert isinstance(removed.events[0], BlockRemoved)
    assert removed.events[0].block_hashes == stored_hashes

    # A second evict finds nothing -> no event emitted.
    publisher.on_evict(chunk_hashes)
    assert not sub.poll(300)


def test_on_evict_unknown_is_noop(
    transport: tuple[DynamoKvEventEmitter, zmq.Socket, TokenHasher],
) -> None:
    emitter, sub, hasher = transport
    publisher = DynamoKvPublisher(emitter, hasher, KV_BLOCK_SIZE, KV_BLOCK_SIZE)

    publisher.on_evict([b"never-stored"])
    assert not sub.poll(300)


def test_chunk_size_not_multiple_raises(
    transport: tuple[DynamoKvEventEmitter, zmq.Socket, TokenHasher],
) -> None:
    emitter, _sub, hasher = transport
    with pytest.raises(ValueError, match="multiple"):
        DynamoKvPublisher(emitter, hasher, KV_BLOCK_SIZE, KV_BLOCK_SIZE + 1)
