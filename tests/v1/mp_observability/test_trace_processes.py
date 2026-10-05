# SPDX-License-Identifier: Apache-2.0
"""Trace propagation through real transports between two spawned OS processes."""

# Standard
from multiprocessing.connection import Connection
import multiprocessing
import os

# Third Party
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing.mp_server import (
    MPServerTracingSubscriber,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_url,
    start_lookup_request_server,
)


def run_server(
    connection: Connection, transport: RequestTransport, endpoint: str
) -> None:
    """Run a CPU transport/event server; export only numeric span relationships."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    bus = EventBus(EventBusConfig())
    bus.register_subscriber(MPServerTracingSubscriber())

    def finished(event: Event) -> None:
        connection.send(
            (
                os.getpid(),
                [
                    (
                        span.name,
                        span.context.trace_id,
                        span.context.span_id,
                        span.parent.span_id if span.parent else None,
                    )
                    for span in exporter.get_finished_spans()
                ],
            )
        )

    bus.subscribe(EventType.MP_REQUEST_END, finished)

    class Module:
        @request_handler(HandlerType.BLOCKING)
        def lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
            bus.publish(Event(EventType.MP_REQUEST_START, session_id=key.request_id))
            bus.publish(Event(EventType.MP_STORE_START, session_id=key.request_id))
            bus.publish(Event(EventType.MP_STORE_END, session_id=key.request_id))

        @request_handler(HandlerType.BLOCKING)
        def end_session(self, request_id: str) -> None:
            bus.publish(Event(EventType.MP_REQUEST_END, session_id=request_id))

    server = start_lookup_request_server(transport, endpoint, Module())
    try:
        bus.start()
        connection.send(endpoint)
        connection.recv()
    finally:
        server.close()
        bus.stop()
        provider.shutdown()
        connection.close()


@pytest.mark.parametrize("transport", REQUEST_TRANSPORTS)
@pytest.mark.parametrize("sampled", [True, False])
def test_two_process_parent_and_keyless_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    transport: RequestTransport,
    sampled: bool,
    unused_tcp_port: int,
) -> None:
    """A later keyless END closes the original parented request in another PID."""
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
    mp = multiprocessing.get_context("spawn")
    parent_conn, child_conn = mp.Pipe()
    endpoint = request_server_url(transport, unused_tcp_port)
    process = mp.Process(target=run_server, args=(child_conn, transport, endpoint))
    process.start()
    child_conn.close()
    client: RequestClient | None = None
    try:
        assert parent_conn.poll(20), "child server did not initialize"
        assert parent_conn.recv() == endpoint
        client = RequestClientFactory.create(endpoint)
        request = IPCCacheServerKey.from_token_ids("model", 1, 0, [1], request_id="r")
        span = NonRecordingSpan(SpanContext(123, 456, False, TraceFlags(int(sampled))))
        with trace.use_span(span):
            future = client.lookup(request, 1)
        assert future.result(10) is None
        end = client.end_session("r")
        assert end.result(10) is None
        assert parent_conn.poll(10)
        pid, spans = parent_conn.recv()
        assert pid != os.getpid()
        assert pid == process.pid
        if sampled:
            root = next(span for span in spans if span[0] == "request")
            child = next(span for span in spans if span[0] == "mp.store")
            assert root[1] == 123 and root[3] == 456
            assert child[1] == 123 and child[3] == root[2]
        else:
            assert not spans
    finally:
        if client is not None:
            client.close()
        if process.is_alive():
            parent_conn.send("stop")
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent_conn.close()
    assert process.exitcode == 0
