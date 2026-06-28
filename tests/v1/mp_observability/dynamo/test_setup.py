# SPDX-License-Identifier: Apache-2.0

"""Tests for setup_dynamo_kv_publishing (the Part D wiring factory).

Uses a fake context (SimpleNamespace), a real EventBus, and a real ZMQ PUB
bound to a test address with a SUB peer, so the whole chain the factory builds
is verified without standing up a server.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast
import socket as _socket

# Third Party
import pytest

pytest.importorskip("vllm.distributed.kv_events", reason="vllm required")
pytest.importorskip("numba", reason="numba required by TokenHasher")

# Third Party
from vllm.distributed.kv_events import BlockRemoved, KVEventBatch  # noqa: E402
import msgspec  # noqa: E402
import zmq  # noqa: E402

# First Party
from lmcache.v1.mp_observability.config import ObservabilityConfig  # noqa: E402
from lmcache.v1.mp_observability.dynamo.block_builder import build_blocks  # noqa: E402
from lmcache.v1.mp_observability.dynamo.setup import (  # noqa: E402
    setup_dynamo_kv_publishing,
)
from lmcache.v1.mp_observability.event import Event, EventType  # noqa: E402
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig  # noqa: E402
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
    if not sub.poll(timeout_ms):
        raise AssertionError("no message received before timeout")
    _topic, _seq, payload = sub.recv_multipart()
    return cast(KVEventBatch, DECODER.decode(payload))


def _make_ctx() -> SimpleNamespace:
    """Fake MPCacheServerContext: real hasher + chunk_size + writable slot."""
    return SimpleNamespace(
        token_hasher=TokenHasher(chunk_size=KV_BLOCK_SIZE, hash_algorithm="blake3"),
        chunk_size=KV_BLOCK_SIZE,
        dynamo_kv_publisher=None,
    )


@pytest.fixture
def bus() -> Iterator[EventBus]:
    b = EventBus(EventBusConfig(enabled=True))
    b.start()
    try:
        yield b
    finally:
        b.stop()


def test_disabled_returns_none_and_touches_nothing(bus: EventBus) -> None:
    ctx = _make_ctx()
    obs_config = ObservabilityConfig(enable_dynamo_kv_events=False)

    # Fake SimpleNamespace context feeds the real signature on purpose.
    result = setup_dynamo_kv_publishing(ctx, obs_config, bus)  # type: ignore[arg-type]

    assert result is None
    assert ctx.dynamo_kv_publisher is None
    assert not bus.has_subscribers(EventType.L1_KEYS_EVICTED)


def test_enabled_builds_chain_and_evict_round_trips(bus: EventBus) -> None:
    ctx = _make_ctx()
    obs_config = ObservabilityConfig(
        enable_dynamo_kv_events=True,
        dynamo_zmq_bind=f"tcp://127.0.0.1:{_free_port()}",
        dynamo_kv_block_size=KV_BLOCK_SIZE,
        dynamo_medium="GPU",
        dynamo_dp_rank=0,
    )

    publisher = setup_dynamo_kv_publishing(
        ctx,  # type: ignore[arg-type]
        obs_config,
        bus,
    )

    # Factory installed everything.
    assert publisher is not None
    assert ctx.dynamo_kv_publisher is publisher
    assert bus.has_subscribers(EventType.L1_KEYS_EVICTED)

    # Connect a SUB peer to the publisher's bound PUB and settle slow-joiner
    # by recording+publishing a throwaway store until the SUB sees traffic.
    sub_ctx = zmq.Context()
    sub = sub_ctx.socket(zmq.SUB)
    sub.connect(obs_config.dynamo_zmq_bind)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    connected = False
    for _ in range(100):
        publisher.on_store(list(range(KV_BLOCK_SIZE)), 0, KV_BLOCK_SIZE, [b"warm"])
        if sub.poll(50):
            connected = True
            break
    assert connected, "SUB never connected to PUB"
    while sub.poll(0):
        sub.recv_multipart()

    try:
        # Record a real chunk->blocks mapping, then drain the BlockStored.
        token_ids = list(range(2 * KV_BLOCK_SIZE))
        chunk_hashes = [b"c0", b"c1"]
        publisher.on_store(token_ids, 0, 2 * KV_BLOCK_SIZE, chunk_hashes)
        _recv_batch(sub)  # the BlockStored

        # Drive eviction through the bus + the factory-registered subscriber.
        bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                session_id="req-1",
                metadata={
                    "keys": [SimpleNamespace(chunk_hash=ch) for ch in chunk_hashes]
                },
            )
        )
        batch = _recv_batch(sub)
    finally:
        sub.close(linger=0)
        sub_ctx.term()

    event = batch.events[0]
    assert isinstance(event, BlockRemoved)
    _, blocks = build_blocks(
        token_ids, KV_BLOCK_SIZE, ctx.token_hasher.none_hash, ctx.token_hasher
    )
    assert event.block_hashes == [h for _, h in blocks]
