# SPDX-License-Identifier: Apache-2.0
"""Ordering, invalidation and ownership at the subscriber transport boundary."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import Mock
import queue
import subprocess
import sys
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.utils import CacheStoreEvent
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import (
    KVEventBatch,
    KVEventProfile,
    KVEventRecord,
    NativeBlockHash,
)
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.kv_event_subscriber import (
    KVEventLedger,
    KVEventReset,
    MPKVEventSubscriber,
)
from lmcache.v1.multiprocess.transport.base import RequestClient

PROFILE = KVEventProfile("native-test", 2)


@pytest.mark.parametrize("hashes", [(10, 11, 12), (b"shared", b"first", b"sibling")])
def test_ledger_preserves_shared_blocks_and_slices_native_descriptors(
    hashes: tuple[NativeBlockHash, NativeBlockHash, NativeBlockHash],
) -> None:
    """Each placement is announced once and retained until its last chunk leaves."""
    shared, first_hash, sibling_hash = hashes
    first = KVEventRecord(
        1,
        "stored",
        "CPU",
        "model",
        (shared, first_hash),
        None,
        (1, 2, 3, 4),
        2,
        PROFILE.profile_id,
        2,
    )
    sibling = replace(
        first, seq=2, block_hashes=(shared, sibling_hash), token_ids=(1, 2, 5, 6)
    )
    ledger = KVEventLedger()
    (stored,) = ledger.apply(first)
    assert isinstance(stored, CacheStoreEvent)
    assert stored.block_hashes == [shared, first_hash]
    assert stored.token_ids == [1, 2, 3, 4]
    assert ledger.apply(replace(first, seq=3)) == []
    (stored,) = ledger.apply(sibling)
    assert isinstance(stored, CacheStoreEvent)
    assert stored.block_hashes == [sibling_hash]
    assert stored.parent_block_hash == shared
    assert stored.token_ids == [5, 6] and stored.block_size == 2
    removed_first = replace(first, seq=4, kind="removed", token_ids=())
    (removed,) = ledger.apply(removed_first)
    assert removed.block_hashes == [first_hash]
    assert ledger.apply(removed_first) == []
    (stored,) = ledger.apply(replace(first, seq=5))
    assert isinstance(stored, CacheStoreEvent)
    assert stored.block_hashes == [first_hash] and stored.parent_block_hash == shared
    (removed,) = ledger.apply(replace(sibling, seq=6, kind="removed", token_ids=()))
    assert removed.block_hashes == [sibling_hash]
    (removed,) = ledger.apply(replace(removed_first, seq=7))
    assert removed.block_hashes == [shared, first_hash]
    assert ledger.reset() == []


def test_ledger_ignores_unseen_chunk_removals_after_truncated_replay() -> None:
    """A removal without its old STORE cannot withdraw a sibling's shared block."""
    first = KVEventRecord(
        1,
        "stored",
        "CPU",
        "model",
        (10, 11),
        None,
        (1, 2, 3, 4),
        2,
        PROFILE.profile_id,
        2,
    )
    unseen = replace(first, block_hashes=(10, 12), kind="removed", token_ids=())
    ledger = KVEventLedger()
    ledger.apply(first)
    assert ledger.apply(unseen) == []
    (removed,) = ledger.apply(replace(first, kind="removed", token_ids=()))
    assert removed.block_hashes == [10, 11]


def test_ledger_reset_withdraws_shared_blocks_once_and_discards_contributors() -> None:
    """Loss clears both advertisements and chunk references before replay."""
    first = KVEventRecord(
        1,
        "stored",
        "CPU",
        "model",
        (10, 11),
        None,
        (1, 2, 3, 4),
        2,
        PROFILE.profile_id,
        2,
    )
    sibling = replace(first, seq=2, block_hashes=(10, 12), token_ids=(1, 2, 5, 6))
    ledger = KVEventLedger()
    ledger.apply(first)
    ledger.apply(sibling)
    (removed,) = ledger.reset()
    assert removed.block_hashes == [10, 11, 12] and removed.medium == "CPU"
    assert ledger.reset() == []
    (stored,) = ledger.apply(first)
    assert stored.block_hashes == [10, 11]
    assert ledger.apply(replace(sibling, kind="removed", token_ids=())) == []
    (removed,) = ledger.apply(replace(first, kind="removed", token_ids=()))
    assert removed.block_hashes == [10, 11]


def batch(
    seq: int,
    *,
    incarnation: int = 1,
    next_cursor: int | None = None,
    lost: bool = False,
    enabled: bool = True,
    events: list[KVEventRecord] | None = None,
    schema_version: int = 1,
) -> KVEventBatch:
    """Build a native batch with explicit overrides for stream-boundary tests."""
    record = KVEventRecord(
        seq, "stored", "CPU", "model", (10,), None, (1, 2), 2, PROFILE.profile_id, 2
    )
    return KVEventBatch(
        enabled,
        incarnation,
        seq if next_cursor is None else next_cursor,
        lost,
        [record] if events is None else events,
        schema_version,
    )


class Source:
    """A cancellable stream; send waits until reception has processed the item."""

    def __init__(self) -> None:
        self.inbox: queue.Queue[
            tuple[KVEventBatch | Exception | None, threading.Event]
        ] = queue.Queue()
        self.consumed = threading.Event()
        self.closed = threading.Event()
        self.stream = MessagingStream(self.read, self.cancel)

    def read(self, closed: threading.Event) -> KVEventBatch:
        self.consumed.set()
        value, self.consumed = self.inbox.get(timeout=5)
        if closed.is_set():
            raise StopIteration
        if isinstance(value, Exception):
            raise value
        assert value is not None
        return value

    def cancel(self) -> None:
        self.closed.set()
        self.consumed.set()
        self.inbox.put((None, threading.Event()))

    def send(self, value: KVEventBatch | Exception) -> None:
        consumed = threading.Event()
        self.inbox.put((value, consumed))
        assert consumed.wait(5), "receiver did not process the item"


@pytest.fixture
def connected() -> Iterator[tuple[MPKVEventSubscriber, Mock, Source]]:
    source = Source()
    client = Mock(spec=RequestClient)
    client.subscribe_kv_events.return_value = source.stream
    subscriber = MPKVEventSubscriber(
        client, 42, "model", world_size=2, profile=PROFILE, max_pending_events=4
    )
    subscriber.start()
    yield subscriber, client, source
    subscriber.close()
    subscriber.close()
    assert source.closed.is_set()
    client.close.assert_not_called()


def test_order_deduplication_and_filtered_gaps(
    connected: tuple[MPKVEventSubscriber, Mock, Source],
) -> None:
    subscriber, client, source = connected
    source.send(batch(1, next_cursor=3))
    removed = replace(batch(5).events[0], kind="removed", token_ids=())
    source.send(batch(5, next_cursor=6, events=[removed]))
    source.send(batch(8))
    source.send(batch(8))
    kinds: list[str] = []
    for update in subscriber.drain():
        assert isinstance(update, KVEventBatch)
        kinds.extend(event.kind for event in update.events)
    assert kinds == ["stored", "removed", "stored"]
    assert subscriber.drain() == []
    client.subscribe_kv_events.assert_called_once_with(42, "model", 0, 3, 2, PROFILE)


@pytest.mark.parametrize("reason", ["server_restart", "events_lost", "queue_overflow"])
def test_reset_precedes_recovered_data(
    connected: tuple[MPKVEventSubscriber, Mock, Source], reason: str
) -> None:
    subscriber, _, source = connected
    source.send(batch(100))
    subscriber.drain()
    if reason == "queue_overflow":
        for seq in range(101, 106):
            source.send(batch(seq))
        expected = 105
    else:
        source.send(
            batch(
                1,
                incarnation=2 if reason == "server_restart" else 1,
                lost=reason == "events_lost",
            )
        )
        expected = 1
    reset, recovered = subscriber.drain()
    assert reset == KVEventReset(reason)
    assert isinstance(recovered, KVEventBatch)
    assert recovered.events[0].seq == expected


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("profile mismatch"),
        batch(1, enabled=False),
        batch(2, next_cursor=1),
        batch(9),
    ],
)
def test_terminal_error_is_visible_at_drain_and_startup(
    connected: tuple[MPKVEventSubscriber, Mock, Source],
    failure: KVEventBatch | Exception,
) -> None:
    subscriber, client, source = connected
    source.send(batch(10))
    subscriber.drain()
    source.send(failure)
    assert source.closed.wait(5)
    (reset,) = subscriber.drain()
    assert isinstance(reset, KVEventReset)
    assert reset.reason == "protocol_error" and reset.error
    with pytest.raises(RuntimeError):
        subscriber.start(timeout=0.1)
    assert client.subscribe_kv_events.call_count == 1


def test_startup_timeout_closes_owned_stream(
    connected: tuple[MPKVEventSubscriber, Mock, Source],
) -> None:
    subscriber, _, source = connected
    with pytest.raises(LMCacheTimeoutError, match="negotiation timed out"):
        subscriber.start(timeout=0.05)
    assert source.closed.is_set()


def test_import_does_not_load_engines() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lmcache.v1.multiprocess.kv_event_subscriber; "
            "assert not {'vllm', 'sglang'}.intersection(sys.modules)",
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("schema_version", [0, 2])
def test_unknown_stream_schema_is_terminal(
    connected: tuple[MPKVEventSubscriber, Mock, Source], schema_version: int
) -> None:
    subscriber, _, source = connected
    source.send(batch(1, schema_version=schema_version))
    (reset,) = subscriber.drain()
    assert isinstance(reset, KVEventReset)
    assert reset.reason == "protocol_error"
    assert reset.error is not None
    assert "schema mismatch" in reset.error


@pytest.mark.parametrize(
    "changes",
    [
        {"block_hashes": (10, 11), "token_ids": (1, 2)},
        {"token_ids": (1, -1)},
        {"block_hashes": [10, 11]},
        {"group_idx": 1},
        {"kv_cache_spec_kind": "sliding_window"},
    ],
)
def test_malformed_native_descriptor_never_reaches_ledger(
    connected: tuple[MPKVEventSubscriber, Mock, Source], changes: dict[str, object]
) -> None:
    subscriber, _, source = connected
    value = batch(1)
    malformed = replace(value.events[0], **changes)  # type: ignore[arg-type]
    source.send(replace(value, events=[malformed]))
    (reset,) = subscriber.drain()
    assert isinstance(reset, KVEventReset)
    assert reset.reason == "protocol_error" and reset.error


def test_close_before_start_does_not_borrow_transport_resources() -> None:
    client = Mock(spec=RequestClient)
    subscriber = MPKVEventSubscriber(client, 42, "model", world_size=2, profile=PROFILE)
    subscriber.close()
    subscriber.close()
    with pytest.raises(RuntimeError, match="closed"):
        subscriber.start()
    client.subscribe_kv_events.assert_not_called()
    client.close.assert_not_called()


def test_close_cancels_a_stream_returned_after_shutdown_started() -> None:
    client = Mock(spec=RequestClient)
    source = Source()
    entered, release = threading.Event(), threading.Event()

    def subscribe(*args: object) -> MessagingStream[KVEventBatch]:
        entered.set()
        assert release.wait(5)
        return source.stream

    client.subscribe_kv_events.side_effect = subscribe
    subscriber = MPKVEventSubscriber(client, 42, "model", world_size=2, profile=PROFILE)
    closer = threading.Thread(target=subscriber.close)
    try:
        subscriber.start()
        assert entered.wait(5)
        closer.start()
        deadline = time.monotonic() + 5
        while True:
            try:
                subscriber.start()
            except RuntimeError:
                break
            assert time.monotonic() < deadline
            time.sleep(0.001)
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        subscriber.close()
    assert not closer.is_alive() and source.closed.is_set()
    assert subscriber.drain() == []
    client.close.assert_not_called()
