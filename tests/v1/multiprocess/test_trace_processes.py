# SPDX-License-Identifier: Apache-2.0
"""Trace propagation through real transports between two spawned OS processes."""

# Standard
from multiprocessing.connection import Connection
from typing import Any
import multiprocessing
import os

# Third Party
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
import pytest
import zmq

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing.mp_server import (
    MPServerTracingSubscriber,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.rpc import get_rpc_spec
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    MessageQueueClient,
    MessageQueueServer,
)


def run_server(connection: Connection, transport: str) -> None:
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

    module = Module()
    ctx = zmq.Context()
    server: Any
    if transport == "zmq":
        server = MessageQueueServer("tcp://127.0.0.1:*", ctx)
        for operation in ("lookup", "end_session"):
            server.add_blocking_handler(
                get_rpc_spec(operation), getattr(module, operation)
            )
        server.add_normal_thread_pool(["lookup", "end_session"], max_workers=2)
        endpoint = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    else:
        # First Party
        from lmcache.v1.multiprocess.transport.grpc_impl.server import (
            GrpcMultiprocessServer,
        )

        server = GrpcMultiprocessServer("grpc://127.0.0.1:0", 2, 1, 4)
        server.add_modules([module])
        endpoint = f"grpc://127.0.0.1:{server.bound_port}"
    try:
        bus.start()
        server.start()
        connection.send(endpoint)
        connection.recv()
    finally:
        server.close()
        bus.stop()
        provider.shutdown()
        ctx.term()
        connection.close()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
@pytest.mark.parametrize("sampled", [True, False])
def test_two_process_parent_and_keyless_lifecycle(
    monkeypatch: pytest.MonkeyPatch, transport: str, sampled: bool
) -> None:
    """A later keyless END closes the original parented request in another PID."""
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
    mp = multiprocessing.get_context("spawn")
    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=run_server, args=(child_conn, transport))
    process.start()
    child_conn.close()
    client: Any = None
    ctx = zmq.Context()
    try:
        assert parent_conn.poll(20), "child server did not initialize"
        endpoint = parent_conn.recv()
        if transport == "zmq":
            client = MessageQueueClient(endpoint, ctx)
        else:
            # First Party
            from lmcache.v1.multiprocess.transport.grpc_impl.client import (
                GrpcMultiprocessClient,
            )

            client = GrpcMultiprocessClient(endpoint)
        request = IPCCacheServerKey.from_token_ids("model", 1, 0, [1], request_id="r")
        span = NonRecordingSpan(SpanContext(123, 456, False, TraceFlags(int(sampled))))
        with trace.use_span(span):
            future = (
                client.submit_request("lookup", [request, 1])
                if transport == "zmq"
                else client.lookup(request, 1)
            )
        assert future.result(10) is None
        end = (
            client.submit_request("end_session", ["r"])
            if transport == "zmq"
            else client.end_session("r")
        )
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
        ctx.term()
    assert process.exitcode == 0
