# SPDX-License-Identifier: Apache-2.0
"""Subscriber lifetime and ordering through the public stream/client boundary."""

# Standard
from dataclasses import replace
from unittest.mock import MagicMock
import queue
import subprocess
import sys
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import KVEventBatch, KVEventRecord
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.kv_event_subscriber import (
    KVEventStreamClosed,
    MPKVEventSubscriber,
)
from lmcache.v1.multiprocess.transport.base import RequestClient


def test_import_does_not_load_inference_engines() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from lmcache.v1.multiprocess.kv_event_subscriber "
            "import MPKVEventSubscriber; "
            "assert not any(name.split('.')[0] in ('vllm', 'sglang') "
            "for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


class EventChannel:
    """Cancellable transport double with observable receiver and delivery queues."""

    def __init__(self) -> None:
        self.inbox: queue.Queue[KVEventBatch | Exception | None] = queue.Queue()
        self.updates: queue.Queue[list[KVEventBatch | KVEventStreamClosed]] = (
            queue.Queue()
        )
        self.receivers: list[threading.Thread] = []
        self.attempts: list[float] = []
        self.cancelled = threading.Event()
        self.subscribed = threading.Event()
        self.reading = threading.Event()
        self.client = MagicMock(spec=RequestClient)
        self.client.subscribe_kv_events.side_effect = self.subscribe

    def subscribe(self, *args: object) -> MessagingStream[KVEventBatch]:
        """Record the receiving thread and return a blocking cancellable stream."""
        self.receivers.append(threading.current_thread())
        self.attempts.append(time.monotonic())
        self.subscribed.set()
        return MessagingStream(self.read, self.cancel)

    def read(self, closed: threading.Event) -> KVEventBatch:
        """Read a batch, propagate an injected failure, or finish on cancellation."""
        self.reading.set()
        value = self.inbox.get(timeout=5)
        if value is None or closed.is_set():
            raise StopIteration
        if isinstance(value, Exception):
            raise value
        return value

    def cancel(self) -> None:
        """Wake a blocked read and record stream cancellation."""
        self.cancelled.set()
        self.inbox.put(None)

    def collect(self, subscriber: MPKVEventSubscriber) -> None:
        """Drain on the receiver thread, as required by the notification contract."""
        self.updates.put(subscriber.drain())


def test_batches_keep_payload_and_order() -> None:
    channel = EventChannel()
    subscriber = MPKVEventSubscriber(channel.client, 41, "model", channel.collect)
    stored = KVEventRecord(1, "stored", "CPU", "model", [b"chunk"], None, [1, 2])
    batches = [
        KVEventBatch(True, 7, 1, False, [stored]),
        KVEventBatch(True, 7, 12, False, [replace(stored, kind="removed")]),
        KVEventBatch(True, 8, 1, True, [stored]),
    ]
    try:
        subscriber.start()
        subscriber.start()
        for batch in batches:
            channel.inbox.put(batch)
        assert [channel.updates.get(timeout=5) for _ in batches] == [
            [batch] for batch in batches
        ]
        assert subscriber.drain() == []
        channel.client.subscribe_kv_events.assert_called_once_with(41, "model", 0, 1024)
    finally:
        subscriber.close()
    subscriber.close()
    assert channel.cancelled.is_set()
    assert all(not thread.is_alive() for thread in channel.receivers)
    channel.client.close.assert_not_called()
    channel.client.ping.assert_not_called()


@pytest.mark.parametrize("failure", [None, ConnectionError("disconnected")])
def test_stream_end_and_disconnect_replay_after_interruptible_delay(
    failure: Exception | None,
) -> None:
    channel = EventChannel()
    # Each attempt gets its own stream, so cancellation cannot end the next one.
    second = EventChannel()
    channel.client.subscribe_kv_events.side_effect = [
        channel.subscribe(),
        second.subscribe(),
    ]
    subscriber = MPKVEventSubscriber(channel.client, 41, "model", channel.collect)
    try:
        subscriber.start()
        start = time.monotonic()
        channel.inbox.put(failure)
        assert channel.updates.get(timeout=5) == [KVEventStreamClosed()]
        second.inbox.put(KVEventBatch(True, 7, 0, False, []))
        assert channel.updates.get(timeout=5) == [KVEventBatch(True, 7, 0, False, [])]
        assert time.monotonic() - start >= 0.9
        assert channel.client.subscribe_kv_events.call_count == 2
        assert all(
            call.args == (41, "model", 0, 1024)
            for call in channel.client.subscribe_kv_events.call_args_list
        )
    finally:
        subscriber.close()
    assert second.cancelled.is_set()
    channel.client.close.assert_not_called()


def test_subscription_failure_backoff_is_cancelled_by_close() -> None:
    channel = EventChannel()
    channel.client.subscribe_kv_events.side_effect = ConnectionError("unavailable")
    subscriber = MPKVEventSubscriber(channel.client, 41, "model", channel.collect)
    subscriber.start()
    try:
        assert channel.updates.get(timeout=5) == [KVEventStreamClosed()]
    finally:
        start = time.monotonic()
        subscriber.close()
    assert time.monotonic() - start < 0.8
    channel.client.subscribe_kv_events.assert_called_once()
    channel.client.close.assert_not_called()


@pytest.mark.parametrize("failure", ["disabled", "read", "subscribe"])
def test_disabled_and_terminal_streams_do_not_restart(failure: str) -> None:
    channel = EventChannel()
    receivers: list[threading.Thread] = []

    def collect(subscriber: MPKVEventSubscriber) -> None:
        receivers.append(threading.current_thread())
        channel.collect(subscriber)

    subscriber = MPKVEventSubscriber(channel.client, 41, "model", collect)
    expected: KVEventBatch | KVEventStreamClosed
    if failure == "disabled":
        expected = KVEventBatch(False, 7, 0, False, [])
        channel.inbox.put(expected)
    else:
        error = RuntimeError("invalid channel")
        expected = KVEventStreamClosed(error=str(error))
        if failure == "subscribe":
            channel.client.subscribe_kv_events.side_effect = error
        else:
            channel.inbox.put(error)
    try:
        subscriber.start()
        assert channel.updates.get(timeout=5) == [expected]
        receivers[0].join(timeout=5)
        assert not receivers[0].is_alive()
        subscriber.start()
        assert subscriber.drain() == []
    finally:
        subscriber.close()
    channel.client.subscribe_kv_events.assert_called_once()
    channel.client.close.assert_not_called()


def test_close_before_start_prevents_subscription() -> None:
    channel = EventChannel()
    subscriber = MPKVEventSubscriber(channel.client, 41, "model", channel.collect)
    subscriber.close()
    subscriber.close()
    with pytest.raises(RuntimeError, match="closed"):
        subscriber.start()
    channel.client.subscribe_kv_events.assert_not_called()
    channel.client.close.assert_not_called()


def test_batch_callback_failure_delivers_terminal_update() -> None:
    channel = EventChannel()

    def collect(subscriber: MPKVEventSubscriber) -> None:
        updates = subscriber.drain()
        if isinstance(updates[0], KVEventBatch):
            raise RuntimeError("batch rejected")
        channel.updates.put(updates)

    subscriber = MPKVEventSubscriber(channel.client, 41, "model", collect)
    try:
        subscriber.start()
        channel.inbox.put(KVEventBatch(True, 7, 0, False, []))
        assert channel.updates.get(timeout=5) == [
            KVEventStreamClosed(error="batch rejected")
        ]
    finally:
        subscriber.close()
    assert channel.cancelled.is_set()
    assert not channel.receivers[0].is_alive()
    channel.client.subscribe_kv_events.assert_called_once()
    channel.client.close.assert_not_called()


def test_close_cancels_blocked_read_and_joins_receiver() -> None:
    channel = EventChannel()
    subscriber = MPKVEventSubscriber(channel.client, 41, "model", channel.collect)
    subscriber.start()
    try:
        assert channel.reading.wait(5)
    finally:
        subscriber.close()
    assert channel.cancelled.is_set()
    assert not channel.receivers[0].is_alive()
    assert channel.updates.empty()
    channel.client.close.assert_not_called()


def _wait_for_close(subscriber: MPKVEventSubscriber) -> None:
    """Observe shutdown through start rejection without accessing internal state."""
    deadline = time.monotonic() + 5
    while True:
        try:
            subscriber.start()
        except RuntimeError:
            return
        assert time.monotonic() < deadline, "close did not stop reception"
        time.sleep(0.001)


def test_close_cancels_stream_returned_by_inflight_subscription() -> None:
    channel = EventChannel()
    entered = threading.Event()
    release = threading.Event()

    def subscribe(*args: object) -> MessagingStream[KVEventBatch]:
        entered.set()
        assert release.wait(5)
        return channel.subscribe(*args)

    channel.client.subscribe_kv_events.side_effect = subscribe
    subscriber = MPKVEventSubscriber(channel.client, 41, "model", channel.collect)
    subscriber.start()
    closer = threading.Thread(target=subscriber.close)
    try:
        assert entered.wait(5)
        closer.start()
        _wait_for_close(subscriber)
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        subscriber.close()
    assert not closer.is_alive()
    assert channel.cancelled.is_set()
    assert all(not thread.is_alive() for thread in channel.receivers)
    assert channel.updates.empty()
    channel.client.close.assert_not_called()


def test_close_waits_for_inflight_callback() -> None:
    channel = EventChannel()
    entered = threading.Event()
    release = threading.Event()

    def collect(subscriber: MPKVEventSubscriber) -> None:
        entered.set()
        assert release.wait(5)
        channel.collect(subscriber)

    subscriber = MPKVEventSubscriber(channel.client, 41, "model", collect)
    closer = threading.Thread(target=subscriber.close)
    batch = KVEventBatch(True, 7, 0, False, [])
    try:
        subscriber.start()
        channel.inbox.put(batch)
        assert entered.wait(5)
        closer.start()
        _wait_for_close(subscriber)
        assert closer.is_alive()
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        subscriber.close()
    assert not closer.is_alive()
    assert channel.updates.get(timeout=5) == [batch]
    assert channel.updates.empty()
    assert not channel.receivers[0].is_alive()
    channel.client.close.assert_not_called()


@pytest.mark.parametrize("failure", [ConnectionError("gap"), RuntimeError("invalid")])
def test_failure_callback_must_not_raise(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel = EventChannel()
    errors: queue.Queue[BaseException | None] = queue.Queue()

    def record_failure(args: threading.ExceptHookArgs) -> None:
        errors.put(args.exc_value)

    def collect(subscriber: MPKVEventSubscriber) -> None:
        channel.collect(subscriber)
        raise RuntimeError("closure handler failed")

    monkeypatch.setattr(threading, "excepthook", record_failure)
    subscriber = MPKVEventSubscriber(channel.client, 41, "model", collect)
    try:
        subscriber.start()
        channel.inbox.put(failure)
        assert str(errors.get(timeout=5)) == "closure handler failed"
        channel.receivers[0].join(timeout=5)
        assert not channel.receivers[0].is_alive()
        expected_error = None if isinstance(failure, ConnectionError) else str(failure)
        assert channel.updates.get(timeout=5) == [
            KVEventStreamClosed(error=expected_error)
        ]
        assert channel.updates.empty()
    finally:
        subscriber.close()
    channel.client.subscribe_kv_events.assert_called_once()
    channel.client.close.assert_not_called()
