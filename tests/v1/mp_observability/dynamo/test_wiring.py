# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Dynamo KV wiring (store hook + evict subscriber).

These exercise the bypass paths end to end without a GPU or a real server:
synthetic keys/object-keys feed the store hook, and a real EventBus drives the
evict subscriber. The transport is a real ZMQ PUB bound to a test address with
a SUB peer asserting the received events.
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
from lmcache.v1.mp_observability.dynamo.evict_subscriber import (  # noqa: E402
    DynamoEvictSubscriber,
)
from lmcache.v1.mp_observability.dynamo.publisher import (  # noqa: E402
    DynamoKvPublisher,
)
from lmcache.v1.mp_observability.dynamo.store_hook import (  # noqa: E402
    publish_store_from_key,
)
from lmcache.v1.mp_observability.event import Event, EventType  # noqa: E402
from lmcache.v1.mp_observability.event_bus import (  # noqa: E402
    EventBus,
    EventBusConfig,
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
    if not sub.poll(timeout_ms):
        raise AssertionError("no message received before timeout")
    _topic, _seq, payload = sub.recv_multipart()
    return cast(KVEventBatch, DECODER.decode(payload))


def _make_key(token_ids: list[int], start: int, end: int) -> SimpleNamespace:
    """A synthetic stand-in for IPCCacheServerKey (only the fields used)."""
    return SimpleNamespace(token_ids=tuple(token_ids), start=start, end=end)


def _make_object_keys(chunk_hashes: list[bytes]) -> list[SimpleNamespace]:
    """Synthetic object keys carrying only ``chunk_hash``."""
    return [SimpleNamespace(chunk_hash=ch) for ch in chunk_hashes]


@pytest.fixture
def wiring() -> Iterator[tuple[DynamoKvPublisher, zmq.Socket, TokenHasher]]:
    """A real publisher (bound emitter) + SUB peer + hasher, slow-joiner warmed."""
    bind = f"tcp://127.0.0.1:{_free_port()}"
    emitter = DynamoKvEventEmitter(bind)
    hasher = TokenHasher(chunk_size=KV_BLOCK_SIZE, hash_algorithm="blake3")
    publisher = DynamoKvPublisher(emitter, hasher, KV_BLOCK_SIZE, KV_BLOCK_SIZE)

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
        yield publisher, sub, hasher
    finally:
        sub.close(linger=0)
        ctx.term()
        publisher.close()


def test_store_hook_publishes_block_stored(
    wiring: tuple[DynamoKvPublisher, zmq.Socket, TokenHasher],
) -> None:
    publisher, sub, hasher = wiring
    token_ids = list(range(2 * KV_BLOCK_SIZE))  # 2 blocks / 2 chunks
    key = _make_key(token_ids, 0, 2 * KV_BLOCK_SIZE)
    object_keys = _make_object_keys([b"c0", b"c1"])

    # Synthetic key/object-keys feed the real signature on purpose.
    publish_store_from_key(publisher, key, object_keys)  # type: ignore[arg-type]

    batch = _recv_batch(sub)
    event = batch.events[0]
    assert isinstance(event, BlockStored)
    _, blocks = build_blocks(token_ids, KV_BLOCK_SIZE, hasher.none_hash, hasher)
    assert event.block_hashes == [h for _, h in blocks]
    assert event.token_ids == token_ids
    assert event.parent_block_hash is None
    assert event.block_size == KV_BLOCK_SIZE


def test_evict_via_event_bus_publishes_block_removed(
    wiring: tuple[DynamoKvPublisher, zmq.Socket, TokenHasher],
) -> None:
    publisher, sub, hasher = wiring

    # Store first so the registry has something to evict.
    token_ids = list(range(2 * KV_BLOCK_SIZE))
    chunk_hashes = [b"c0", b"c1"]
    publish_store_from_key(
        publisher,
        _make_key(token_ids, 0, 2 * KV_BLOCK_SIZE),  # type: ignore[arg-type]
        _make_object_keys(chunk_hashes),  # type: ignore[arg-type]
    )
    stored = _recv_batch(sub)
    stored_hashes = stored.events[0].block_hashes

    # Drive eviction through a real EventBus + subscriber.
    bus = EventBus(EventBusConfig(enabled=True))
    bus.register_subscriber(DynamoEvictSubscriber(publisher))
    bus.start()
    try:
        bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                session_id="req-1",
                metadata={"keys": _make_object_keys(chunk_hashes)},
            )
        )
        batch = _recv_batch(sub)
    finally:
        bus.stop()

    event = batch.events[0]
    assert isinstance(event, BlockRemoved)
    assert event.block_hashes == stored_hashes


def test_store_hook_swallows_publisher_errors(
    wiring: tuple[DynamoKvPublisher, zmq.Socket, TokenHasher],
) -> None:
    _publisher, sub, _hasher = wiring

    class _BoomPublisher:
        def on_store(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("boom")

    key = _make_key([0, 1, 2], 0, 3)
    # Must not raise -- the store path is unaffected by a publish failure.
    publish_store_from_key(
        _BoomPublisher(),  # type: ignore[arg-type]
        key,  # type: ignore[arg-type]
        _make_object_keys([b"c0"]),  # type: ignore[arg-type]
    )
    # And nothing was emitted on the wire.
    assert not sub.poll(300)


def test_disabled_publisher_is_noop(
    wiring: tuple[DynamoKvPublisher, zmq.Socket, TokenHasher],
) -> None:
    _publisher, sub, _hasher = wiring
    key = _make_key(list(range(KV_BLOCK_SIZE)), 0, KV_BLOCK_SIZE)

    # publisher=None: pure no-op, no event published.
    publish_store_from_key(
        None,
        key,  # type: ignore[arg-type]
        _make_object_keys([b"c0"]),  # type: ignore[arg-type]
    )
    assert not sub.poll(300)
