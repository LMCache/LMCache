# SPDX-License-Identifier: Apache-2.0
"""Shutdown ordering for the multiprocess cache server."""

# First Party
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
