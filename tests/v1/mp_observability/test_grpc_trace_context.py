# SPDX-License-Identifier: Apache-2.0
"""W3C metadata on keyless gRPC calls, including worker reuse after failure."""

# Standard
from queue import Queue

# Third Party
from opentelemetry import trace
import grpc
import pytest

# First Party
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.grpc_impl.server import GrpcMultiprocessServer
from tests.v1.mp_observability.test_trace_context import parent_span
from tests.v1.multiprocess.transport_test_utils import (
    request_server_url,
)


@pytest.mark.parametrize(
    "handler_type,affinity",
    [
        (HandlerType.SYNC, False),
        (HandlerType.BLOCKING, False),
        (HandlerType.BLOCKING, True),
    ],
    ids=["sync", "pool", "affinity"],
)
def test_keyless_parent_and_failure_reuse(
    monkeypatch: pytest.MonkeyPatch,
    unused_tcp_port: int,
    handler_type: HandlerType,
    affinity: bool,
) -> None:
    """Metadata reaches keyless handlers and failures leave no stale parent."""
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    observed: Queue[tuple[int, bool]] = Queue()

    class Module:
        @request_handler(handler_type, requires_client_affinity=affinity)
        def end_session(self, request_id: str) -> None:
            """Record the caller parent, then optionally fail the call."""
            parent = trace.get_current_span().get_span_context()
            observed.put((parent.trace_id, parent.trace_flags.sampled))
            if request_id == "fail":
                raise ValueError("test handler failure")

    endpoint = request_server_url("grpc", unused_tcp_port)
    server = GrpcMultiprocessServer(endpoint, 1, 1, 1)
    server.add_modules([Module()])
    server.start()
    client = RequestClientFactory.create(endpoint)
    try:
        for number, sampled in [(10, True), (11, False)]:
            with trace.use_span(parent_span(number, sampled)):
                assert client.end_session("ok").result(5) is None
            assert observed.get(timeout=5) == (number, sampled)
        with trace.use_span(parent_span(12)):
            with pytest.raises(grpc.RpcError):
                client.end_session("fail").result(5)
        assert observed.get(timeout=5) == (12, True)
        assert client.end_session("ok").result(5) is None
        assert observed.get(timeout=5) == (0, False)
    finally:
        client.close()
        server.close()
