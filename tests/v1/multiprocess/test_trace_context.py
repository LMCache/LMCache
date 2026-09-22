# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the real MQ, worker and event-subscriber propagation path."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any
import asyncio
import threading

# Third Party
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState
import msgspec
import pytest
import zmq

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.propagation import (
    capture_trace_context,
    extract_trace_context,
    run_with_trace_context,
)
from lmcache.v1.mp_observability.subscribers.tracing.mp_server import (
    MPServerTracingSubscriber,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.rpc import get_rpc_spec
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    BlockingRequestHandler,
    MessageQueueClient,
    MessageQueueServer,
    SyncRequestHandler,
    msgspec_encode,
)
import lmcache.v1.mp_observability.subscribers.tracing.mp_server as tracing_module


def parent_span(number: int, sampled: bool = True) -> NonRecordingSpan:
    """Return a deterministic W3C parent, including its sampling decision."""
    return NonRecordingSpan(
        SpanContext(
            number,
            number + 100,
            False,
            TraceFlags(1 if sampled else 0),
            TraceState([("vendor", "value")]),
        )
    )


def key() -> IPCCacheServerKey:
    """Create a CPU-only cache request key."""
    return IPCCacheServerKey.from_token_ids("model", 1, 0, [1, 2], request_id="r")


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable propagation independently from exporter configuration."""
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")


def test_disabled_drops_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LMCACHE_MP_TRACE_CONTEXT", raising=False)
    with trace.use_span(parent_span(1)):
        assert capture_trace_context() == {}
        assert Event(EventType.MP_REQUEST_START).trace_context == {}
        assert (
            not trace.get_current_span(extract_trace_context({}))
            .get_span_context()
            .is_valid
        )


@pytest.mark.usefixtures("enabled")
def test_baggage_not_transmitted_and_disabled_receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = context.attach(baggage.set_baggage("prompt", "private payload"))
    try:
        with trace.use_span(parent_span(1)):
            carrier = capture_trace_context()
        assert set(carrier) == {"traceparent", "tracestate"}
        monkeypatch.delenv("LMCACHE_MP_TRACE_CONTEXT")
        remote = trace.get_current_span(extract_trace_context(carrier))
        assert not remote.get_span_context().is_valid
    finally:
        context.detach(token)


@pytest.mark.usefixtures("enabled")
def test_key_wire_compatibility_and_identity() -> None:
    original = key()
    with trace.use_span(parent_span(1)):
        propagated = replace(original, trace_context=capture_trace_context())
    assert original == propagated
    assert hash(original) == hash(propagated)
    assert "traceparent" not in repr(propagated)
    wire = msgspec.msgpack.decode(msgspec.msgpack.encode(propagated))
    wire.pop("trace_context")
    assert (
        msgspec.msgpack.decode(
            msgspec.msgpack.encode(wire), type=IPCCacheServerKey
        ).trace_context
        is None
    )

    @dataclass
    class LegacyKey:
        model_name: str
        world_size: int
        worker_id: int | None
        token_ids: tuple[int, ...]
        start: int
        end: int
        request_id: str

    legacy = msgspec.msgpack.decode(msgspec.msgpack.encode(propagated), type=LegacyKey)
    assert legacy.token_ids == original.token_ids


@pytest.mark.usefixtures("enabled")
@pytest.mark.parametrize(
    "carrier", [{}, {"traceparent": "bad"}, {"traceparent": "x" * 513}]
)
def test_invalid_carrier_isolated(carrier: dict[str, str]) -> None:
    with trace.use_span(parent_span(1)):
        result = run_with_trace_context(
            carrier, lambda: trace.get_current_span().get_span_context()
        )
        assert not result.is_valid
        assert trace.get_current_span().get_span_context().trace_id == 1


@pytest.mark.usefixtures("enabled")
@pytest.mark.parametrize("error", [ValueError, asyncio.CancelledError])
def test_exception_and_cancellation_restore_context(error: type[BaseException]) -> None:
    def fail() -> None:
        raise error()

    with trace.use_span(parent_span(1)):
        with pytest.raises(error):
            run_with_trace_context({}, fail)
        assert trace.get_current_span().get_span_context().trace_id == 1


@pytest.mark.usefixtures("enabled")
def test_real_handlers_and_concurrent_worker_isolation() -> None:
    def observe(request: IPCCacheServerKey) -> int:
        return trace.get_current_span().get_span_context().trace_id

    blocking = BlockingRequestHandler([IPCCacheServerKey], int, observe)
    sync = SyncRequestHandler([IPCCacheServerKey], int, observe)
    with ThreadPoolExecutor(max_workers=2) as pool:
        blocking.executor = pool
        futures = []
        for number in range(1, 21):
            with trace.use_span(parent_span(number)):
                request = replace(key(), trace_context=capture_trace_context())
            payload = [msgspec_encode(request, IPCCacheServerKey)]
            assert sync(payload) == number
            futures.append(blocking(payload))
        assert [future.result(5) for future in futures] == list(range(1, 21))
        assert blocking([msgspec_encode(key(), IPCCacheServerKey)]).result(5) == 0


@pytest.mark.usefixtures("enabled")
@pytest.mark.parametrize("sampled", [True, False])
def test_real_zmq_worker_event_bus_parentage(
    monkeypatch: pytest.MonkeyPatch, sampled: bool
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        tracing_module, "_tracer", provider.get_tracer("lmcache_mp.server")
    )
    bus = EventBus(EventBusConfig())
    bus.register_subscriber(MPServerTracingSubscriber())
    completed = threading.Event()
    bus.subscribe(EventType.MP_REQUEST_END, lambda event: completed.set())

    def lookup(request: IPCCacheServerKey, tp_size: int) -> None:
        bus.publish(Event(EventType.MP_REQUEST_START, session_id=request.request_id))
        bus.publish(Event(EventType.MP_STORE_START, session_id=request.request_id))
        bus.publish(Event(EventType.MP_STORE_END, session_id=request.request_id))
        bus.publish(Event(EventType.MP_REQUEST_END, session_id=request.request_id))

    ctx = zmq.Context()
    server = MessageQueueServer("tcp://127.0.0.1:*", ctx)
    server.add_blocking_handler(get_rpc_spec("lookup"), lookup)
    server.add_normal_thread_pool(["lookup"], max_workers=2)
    bus.start()
    server.start()
    client = MessageQueueClient(server.socket.getsockopt_string(zmq.LAST_ENDPOINT), ctx)
    request = key()
    try:
        parent = parent_span(99, sampled)
        with trace.use_span(parent):
            future: Any = client.submit_request("lookup", [request, 1])
        assert future.result(5) is None
        assert completed.wait(5)
        spans = exporter.get_finished_spans()
        assert request.trace_context is None
        if sampled:
            root = next(span for span in spans if span.name == "request")
            child = next(span for span in spans if span.name == "mp.store")
            assert root.context.trace_id == 99
            assert root.parent.span_id == parent.get_span_context().span_id
            assert root.context.trace_state.get("vendor") == "value"
            assert child.parent.span_id == root.context.span_id
            assert not {"traceparent", "tracestate", "token_ids"} & set(root.attributes)
        else:
            assert spans == ()
    finally:
        client.close()
        server.close()
        bus.stop()
        ctx.term()
        provider.shutdown()
