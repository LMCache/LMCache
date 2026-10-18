# SPDX-License-Identifier: Apache-2.0
"""Tests that a failed request reaches the caller instead of timing out."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

# Test helpers
from tests.v1.multiprocess.test_mq import (
    create_cache_key,
)
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_url,
    start_lookup_request_server,
)

# Anything approaching this budget is the hang these tests exist to catch.
_RESULT_TIMEOUT_S = 10.0


class _RaisingHandler:
    """Fail every FREE_LOOKUP_LOCKS request."""

    @request_handler(HandlerType.BLOCKING)
    def free_lookup_locks(self, key: IPCCacheServerKey, tp_size: int) -> None:
        raise RuntimeError("handler blew up")


@pytest.mark.parametrize("request_transport", REQUEST_TRANSPORTS)
def test_handler_exception_reaches_the_caller(
    request_transport: RequestTransport,
) -> None:
    """A raising handler fails the caller's future rather than going quiet."""
    server_url = request_server_url(
        request_transport,
        15580 if request_transport == "zmq" else 15581,
    )
    server = start_lookup_request_server(
        request_transport, server_url, _RaisingHandler()
    )
    client = RequestClientFactory.create(server_url)
    try:
        started = time.monotonic()
        with pytest.raises(Exception) as excinfo:
            client.free_lookup_locks(create_cache_key(0), 1).result(
                timeout=_RESULT_TIMEOUT_S
            )
        elapsed = time.monotonic() - started

        assert not isinstance(excinfo.value, LMCacheTimeoutError), (
            "the caller timed out instead of being told the handler failed"
        )
        assert elapsed < _RESULT_TIMEOUT_S / 2
    finally:
        client.close()
        server.close()


def test_unregistered_operation_reaches_the_caller() -> None:
    """An operation with no handler fails the caller rather than going quiet.

    ZMQ only: gRPC rejects unknown methods at the framework level.
    """
    server_url = request_server_url("zmq", 15582)
    server = start_lookup_request_server("zmq", server_url, object())
    client = RequestClientFactory.create(server_url)
    try:
        started = time.monotonic()
        with pytest.raises(Exception) as excinfo:
            client.free_lookup_locks(create_cache_key(0), 1).result(
                timeout=_RESULT_TIMEOUT_S
            )
        elapsed = time.monotonic() - started

        assert not isinstance(excinfo.value, LMCacheTimeoutError)
        assert elapsed < _RESULT_TIMEOUT_S / 2
    finally:
        client.close()
        server.close()
