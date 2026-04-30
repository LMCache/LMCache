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
from lmcache.v1.platform import consume_fd


class _FakeSerializer(Serializer):
    def __init__(self, transform: Optional[Callable[[int], None]] = None) -> None:
        self._transform = transform
        self.calls = 0

    def serialize(self, src, dst) -> int:  # type: ignore[no-untyped-def]
        if self._transform is not None:
            self._transform(self.calls)
        self.calls += 1
        return 1

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
