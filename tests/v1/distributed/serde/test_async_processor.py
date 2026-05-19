# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for AsyncSerdeProcessor.

Verifies the async, notifier-based contract:
- submit returns a task id immediately.
- The corresponding notifier fd is signaled on completion.
- query_result returns the bool outcome, None before completion and after
  being consumed (non-idempotent).
- Failing serialize/deserialize produces result=False and still signals fd.
- Serialize and deserialize use distinct event fds.
"""

# Standard
from typing import Callable, Optional
import select
import time

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor,
    Deserializer,
    Serializer,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
from lmcache.v1.mp_observability.subscribers.metrics.serde import SerdeMetricsSubscriber
from lmcache.v1.platform import consume_fd
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    read_counters,
)


class _FakeSerializer(Serializer):
    def __init__(self, transform: Optional[Callable[[int], None]] = None) -> None:
        self._transform = transform
        self.calls = 0

    def serialize(self, src, dst) -> int:  # type: ignore[no-untyped-def]
        if self._transform is not None:
            self._transform(self.calls)
        self.calls += 1
        return dst.get_size() if hasattr(dst, "get_size") else 1

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        return 1


class _FakeDeserializer(Deserializer):
    def __init__(self, transform: Optional[Callable[[int], None]] = None) -> None:
        self._transform = transform
        self.calls = 0

    def deserialize(self, src, dst) -> None:  # type: ignore[no-untyped-def]
        if self._transform is not None:
            self._transform(self.calls)
        self.calls += 1


class _SizedObject:
    def __init__(self, size: int) -> None:
        self._size = size

    def get_size(self) -> int:
        return self._size


def _wait_for_fd(fd: int, timeout_s: float = 2.0) -> bool:
    """Wait until ``fd`` is readable or timeout. Drains the pending signal."""
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining_ms = int(max(0, (deadline - time.monotonic()) * 1000))
        if poller.poll(remaining_ms):
            try:
                consume_fd(fd)
            except OSError:
                pass
            return True
    return False


def test_serialize_and_deserialize_fds_are_distinct() -> None:
    processor = AsyncSerdeProcessor(_FakeSerializer(), _FakeDeserializer())
    try:
        assert (
            processor.get_serialize_event_fd() != processor.get_deserialize_event_fd()
        )
    finally:
        processor.close()


def test_serialize_signals_fd_and_result_is_true() -> None:
    serializer = _FakeSerializer()
    processor = AsyncSerdeProcessor(serializer, _FakeDeserializer())
    try:
        task_id = processor.submit_serialize([object()], [object()])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        assert processor.query_serialize_result(task_id) is True
        # Non-idempotent: second query returns None.
        assert processor.query_serialize_result(task_id) is None
        assert serializer.calls == 1
    finally:
        processor.close()


def test_deserialize_signals_fd_and_result_is_true() -> None:
    deserializer = _FakeDeserializer()
    processor = AsyncSerdeProcessor(_FakeSerializer(), deserializer)
    try:
        task_id = processor.submit_deserialize([object()], [object()])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_deserialize_event_fd()), "fd never signaled"
        assert processor.query_deserialize_result(task_id) is True
        assert processor.query_deserialize_result(task_id) is None
        assert deserializer.calls == 1
    finally:
        processor.close()


def test_serialize_failure_reports_false() -> None:
    """If the sync serializer raises, query_serialize_result returns False."""

    def _boom(_i: int) -> None:
        raise RuntimeError("serialize failed")

    processor = AsyncSerdeProcessor(_FakeSerializer(_boom), _FakeDeserializer())
    try:
        task_id = processor.submit_serialize([object()], [object()])  # type: ignore[list-item]
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        assert processor.query_serialize_result(task_id) is False
    finally:
        processor.close()


def test_query_returns_none_before_completion() -> None:
    """Querying an unknown/not-yet-completed task yields None, not an error."""
    processor = AsyncSerdeProcessor(_FakeSerializer(), _FakeDeserializer())
    try:
        # No task submitted with id 42 — should be None, not raise.
        assert processor.query_serialize_result(42) is None
        assert processor.query_deserialize_result(42) is None
    finally:
        processor.close()


def test_estimate_serialized_size_delegates_to_serializer() -> None:
    serializer = _FakeSerializer()
    processor = AsyncSerdeProcessor(serializer, _FakeDeserializer())
    try:
        layout = MemoryLayoutDesc(shapes=[], dtypes=[])
        assert processor.estimate_serialized_size(layout) == 1
    finally:
        processor.close()


def test_serde_event_session_ids_are_unique_per_processor() -> None:
    """Concurrent processors must not collide in start/end correlation keys."""
    bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
    session_ids: list[str] = []

    def _record_session_id(event: Event) -> None:
        session_ids.append(event.session_id)

    bus.subscribe(EventType.CB_SERDE_ENCODE_START, _record_session_id)
    processor_a = AsyncSerdeProcessor(
        _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
    )
    processor_b = AsyncSerdeProcessor(
        _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
    )
    try:
        bus.start()
        task_a = processor_a.submit_serialize(
            [_SizedObject(4096)],
            [_SizedObject(2048)],  # type: ignore[list-item]
        )
        task_b = processor_b.submit_serialize(
            [_SizedObject(4096)],
            [_SizedObject(2048)],  # type: ignore[list-item]
        )
        assert _wait_for_fd(processor_a.get_serialize_event_fd()), "fd A never signaled"
        assert _wait_for_fd(processor_b.get_serialize_event_fd()), "fd B never signaled"
        assert processor_a.query_serialize_result(task_a) is True
        assert processor_b.query_serialize_result(task_b) is True
        bus.stop()
        assert len(session_ids) == 2
        assert len(set(session_ids)) == 2
    finally:
        processor_a.close()
        processor_b.close()
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


def test_serialize_emits_cb_serde_encode_metrics() -> None:
    """AsyncSerdeProcessor must publish encode serde events on completion."""
    bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
    bus.register_subscriber(SerdeMetricsSubscriber())
    before = read_counters()
    processor = AsyncSerdeProcessor(
        _FakeSerializer(), _FakeDeserializer(), serde_type="fp8"
    )
    try:
        bus.start()
        task_id = processor.submit_serialize(
            [_SizedObject(4096)],
            [_SizedObject(2048)],  # type: ignore[list-item]
        )
        assert _wait_for_fd(processor.get_serialize_event_fd()), "fd never signaled"
        assert processor.query_serialize_result(task_id) is True
        bus.stop()
        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 4096
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 2048
    finally:
        processor.close()
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


def test_deserialize_failure_emits_cb_serde_decode_failure_metric() -> None:
    """Serde decode failures must be visible as lmcache_blend.serde_failures."""

    def _boom(_i: int) -> None:
        raise RuntimeError("deserialize failed")

    bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=100))
    bus.register_subscriber(SerdeMetricsSubscriber())
    before = read_counters()
    processor = AsyncSerdeProcessor(
        _FakeSerializer(), _FakeDeserializer(_boom), serde_type="cachegen"
    )
    try:
        bus.start()
        task_id = processor.submit_deserialize(
            [_SizedObject(2048)],
            [_SizedObject(4096)],  # type: ignore[list-item]
        )
        assert _wait_for_fd(processor.get_deserialize_event_fd()), "fd never signaled"
        assert processor.query_deserialize_result(task_id) is False
        bus.stop()
        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_failures", 0) >= 1
    finally:
        processor.close()
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))
