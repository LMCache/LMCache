# SPDX-License-Identifier: Apache-2.0
"""Shutdown ordering for the multiprocess cache server."""

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.server import close_cache_server


def test_close_cache_server_stops_bus_last() -> None:
    """The event bus must outlive the server and engine teardown."""
    calls: list[str] = []

    class _Server:
        def close(self) -> None:
            calls.append("server")

    class _Engine:
        def close(self) -> None:
            calls.append("engine")

    class _Bus:
        def stop(self) -> None:
            calls.append("bus")

    close_cache_server(_Server(), _Engine(), _Bus())  # type: ignore[arg-type]

    assert calls == ["server", "engine", "bus"]


def test_close_cache_server_flushes_engine_teardown_events() -> None:
    """A real EventBus must deliver an event published by engine.close()."""
    received: list[Event] = []

    class _Engine:
        def close(self) -> None:
            bus.publish(Event(event_type=EventType.MP_STORE_END))

    class _Server:
        def close(self) -> None:
            pass

    bus = EventBus(EventBusConfig(enabled=True))
    bus.subscribe(EventType.MP_STORE_END, received.append)
    bus.start()

    close_cache_server(_Server(), _Engine(), bus)  # type: ignore[arg-type]

    assert len(received) == 1
    assert received[0].event_type == EventType.MP_STORE_END


def test_close_cache_server_stops_bus_when_engine_close_raises() -> None:
    """The bus must be stopped even when engine teardown fails."""
    calls: list[str] = []

    class _Server:
        def close(self) -> None:
            calls.append("server")

    class _Engine:
        def close(self) -> None:
            calls.append("engine")
            raise RuntimeError("engine close failed")

    class _Bus:
        def stop(self) -> None:
            calls.append("bus")

    try:
        close_cache_server(_Server(), _Engine(), _Bus())  # type: ignore[arg-type]
    except RuntimeError:
        pass

    assert calls == ["server", "engine", "bus"]
