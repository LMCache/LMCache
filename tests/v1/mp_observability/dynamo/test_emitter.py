# SPDX-License-Identifier: Apache-2.0

"""Tests for DynamoKvEventEmitter (vLLM-format KV events over ZMQ PUB)."""

# Standard
from collections.abc import Iterator
from typing import cast
import socket as _socket

# Third Party
import pytest

pytest.importorskip("vllm.distributed.kv_events", reason="vllm required")

# Third Party
from vllm.distributed.kv_events import (  # noqa: E402
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
import msgspec  # noqa: E402
import zmq  # noqa: E402

# First Party
from lmcache.v1.mp_observability.dynamo.emitter import (  # noqa: E402
    DynamoKvEventEmitter,
)

DECODER = msgspec.msgpack.Decoder(type=KVEventBatch)


def _free_port() -> int:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _recv_batch(
    sub: zmq.Socket, timeout_ms: int = 3000
) -> tuple[bytes, int, KVEventBatch]:
    """Receive one multipart message, returning (topic, seq, decoded batch)."""
    if not sub.poll(timeout_ms):
        raise AssertionError("no message received before timeout")
    topic, seq_bytes, payload = sub.recv_multipart()
    batch = cast(KVEventBatch, DECODER.decode(payload))
    return topic, int.from_bytes(seq_bytes, "big"), batch


@pytest.fixture
def emitter_and_sub() -> Iterator[tuple[DynamoKvEventEmitter, zmq.Socket]]:
    """A bound emitter plus a connected+subscribed SUB socket.

    Waits for the PUB/SUB connection to settle (slow-joiner) by retrying a
    warmup publish until the SUB actually receives something, so the first
    asserted message is never dropped.
    """
    bind = f"tcp://127.0.0.1:{_free_port()}"
    emitter = DynamoKvEventEmitter(bind)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(bind)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    # Slow-joiner warmup: publish until the SUB sees a message, then drain.
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
        yield emitter, sub
    finally:
        sub.close(linger=0)
        ctx.term()
        emitter.close()


def test_stored_round_trip(
    emitter_and_sub: tuple[DynamoKvEventEmitter, zmq.Socket],
) -> None:
    emitter, sub = emitter_and_sub
    emitter.publish_stored(
        token_ids=[10, 11, 12, 13],
        block_hashes=[111, 222],
        parent_hash=999,
        block_size=2,
    )
    _, _, batch = _recv_batch(sub)

    assert len(batch.events) == 1
    event = batch.events[0]
    assert isinstance(event, BlockStored)
    assert event.block_hashes == [111, 222]
    assert event.parent_block_hash == 999
    assert event.token_ids == [10, 11, 12, 13]
    assert event.block_size == 2


def test_removed_round_trip(
    emitter_and_sub: tuple[DynamoKvEventEmitter, zmq.Socket],
) -> None:
    emitter, sub = emitter_and_sub
    emitter.publish_removed([111, 222, 333])
    _, _, batch = _recv_batch(sub)

    assert len(batch.events) == 1
    event = batch.events[0]
    assert isinstance(event, BlockRemoved)
    assert event.block_hashes == [111, 222, 333]


def test_seq_increments(
    emitter_and_sub: tuple[DynamoKvEventEmitter, zmq.Socket],
) -> None:
    emitter, sub = emitter_and_sub
    emitter.publish_removed([1])
    emitter.publish_removed([2])

    _, seq0, _ = _recv_batch(sub)
    _, seq1, _ = _recv_batch(sub)
    assert seq1 == seq0 + 1


def test_stored_parent_none(
    emitter_and_sub: tuple[DynamoKvEventEmitter, zmq.Socket],
) -> None:
    emitter, sub = emitter_and_sub
    emitter.publish_stored(
        token_ids=[1, 2],
        block_hashes=[42],
        parent_hash=None,
        block_size=2,
    )
    _, _, batch = _recv_batch(sub)

    event = batch.events[0]
    assert isinstance(event, BlockStored)
    assert event.parent_block_hash is None
    assert event.block_hashes == [42]


def test_medium_and_dp_rank_are_encoded() -> None:
    """medium / data_parallel_rank passed to __init__ ride along on events."""
    bind = f"tcp://127.0.0.1:{_free_port()}"
    emitter = DynamoKvEventEmitter(bind, medium="GPU", data_parallel_rank=3)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(bind)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    for _ in range(100):
        emitter.publish_removed([0])
        if sub.poll(50):
            break
    while sub.poll(0):
        sub.recv_multipart()

    try:
        emitter.publish_stored(
            token_ids=[1, 2],
            block_hashes=[7],
            parent_hash=None,
            block_size=2,
        )
        _, _, stored_batch = _recv_batch(sub)
        assert stored_batch.data_parallel_rank == 3
        stored_event = stored_batch.events[0]
        assert isinstance(stored_event, BlockStored)
        assert stored_event.medium == "GPU"

        emitter.publish_removed([7])
        _, _, removed_batch = _recv_batch(sub)
        assert removed_batch.data_parallel_rank == 3
        removed_event = removed_batch.events[0]
        assert isinstance(removed_event, BlockRemoved)
        assert removed_event.medium == "GPU"
    finally:
        sub.close(linger=0)
        ctx.term()
        emitter.close()
