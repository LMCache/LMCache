# SPDX-License-Identifier: Apache-2.0

"""Focused serde wire-path tests through AsyncSerdeProcessor + EventBus."""

# Standard
import select
import time

# First Party
from lmcache.v1.distributed.serde import AsyncSerdeProcessor
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
from lmcache.v1.mp_observability.subscribers.metrics.serde import SerdeMetricsSubscriber
from lmcache.v1.platform import consume_fd
from tests.v1.distributed.serde.test_async_processor import (
    _FakeDeserializer,
    _FakeSerializer,
    _SizedObject,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    read_counters,
)


def _wait_for_fd(fd: int, timeout_s: float = 2.0) -> bool:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if poller.poll(int(max(0, (deadline - time.monotonic()) * 1000))):
            try:
                consume_fd(fd)
            except OSError:
                pass
            return True
    return False


def test_encode_wire_path_emits_serde_metrics() -> None:
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
        assert _wait_for_fd(processor.get_serialize_event_fd())
        assert processor.query_serialize_result(task_id) is True
        time.sleep(0.15)
        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_bytes_in", 0) >= 4096
        assert delta.get("lmcache_blend.serde_bytes_out", 0) >= 2048
    finally:
        processor.close()
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


def test_decode_failure_wire_path_emits_failure_metric() -> None:
    def _boom(_i: int) -> None:
        raise RuntimeError("decode failed")

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
        assert _wait_for_fd(processor.get_deserialize_event_fd())
        assert processor.query_deserialize_result(task_id) is False
        time.sleep(0.15)
        delta = counter_delta(before, read_counters())
        assert delta.get("lmcache_blend.serde_failures", 0) >= 1
    finally:
        processor.close()
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))
