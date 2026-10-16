# SPDX-License-Identifier: Apache-2.0
"""Request snapshot delivery through the public MP transport and event bus."""

# Standard
from queue import Queue
from typing import cast
from unittest.mock import MagicMock
import socket

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus
from lmcache.v1.multiprocess.custom_types import RequestSnapshot, RequestSnapshotEntry
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

# Test helpers
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)


def _management(bus: EventBus) -> ManagementModule:
    """Build the reporting module with an isolated event bus and no cache engine."""
    context = MagicMock(spec=MPCacheServerContext)
    context.event_bus = bus
    return ManagementModule(context)


@pytest.mark.parametrize("request_transport", REQUEST_TRANSPORTS)
@pytest.mark.parametrize(
    "requests",
    [
        (),
        (
            RequestSnapshotEntry("a", (11, 12), -2, 12.5, "WAITING"),
            RequestSnapshotEntry("b", (21,), 4, 13.25, "PREEMPTED"),
        ),
    ],
    ids=["empty", "active"],
)
def test_request_snapshot_reaches_subscriber(
    request_transport: RequestTransport,
    requests: tuple[RequestSnapshotEntry, ...],
) -> None:
    """Both transports preserve copied fields and deliver empty observations."""
    bus = EventBus()
    received: Queue[Event] = Queue()
    bus.subscribe(EventType.MP_REQUEST_SNAPSHOT, received.put)
    module = _management(bus)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = cast(int, sock.getsockname()[1])
    url = request_server_url(request_transport, port)
    server = create_request_server(
        [module], request_server_config(request_transport, url)
    )
    bus.start()
    server.start()
    client = RequestClientFactory.create(url)
    try:
        snapshot = RequestSnapshot("scheduler-a", 3, "model", "priority", requests)
        assert client.report_request_snapshot(snapshot).result(timeout=5) is None
        event = received.get(timeout=5)
        assert event.event_type is EventType.MP_REQUEST_SNAPSHOT
        assert event.metadata == {"snapshot": snapshot}
    finally:
        client.close()
        server.close()
        bus.stop()
        module.close()


def test_request_snapshot_without_subscribers_is_not_queued() -> None:
    """Reporting has no event queue cost when no consumer is registered."""
    bus = EventBus()
    module = _management(bus)

    module.report_request_snapshot(
        RequestSnapshot("scheduler-a", 3, "model", "priority", ())
    )

    assert bus.queue_depth() == 0
