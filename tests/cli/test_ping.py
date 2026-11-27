# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache ping``."""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import urlparse
import argparse
import io
import json
import socket
import warnings

# Third Party
import pytest

# First Party
from lmcache.cli.commands import ping as ping_command
from lmcache.cli.commands.ping import (
    PingCommand,
    ping,
)

# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal handler that serves a canned response."""

    response_body: bytes = b""
    response_code: int = 200

    def do_GET(self):
        self.send_response(self.response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format, *args):
        pass  # suppress stderr noise


def _make_handler(code: int, body: bytes = b""):
    return type("_H", (_MockHandler,), {"response_code": code, "response_body": body})


def _start_server(code: int, body: bytes = b"") -> tuple[HTTPServer, int]:
    """Start a local HTTP server and return ``(server, port)``."""
    server = HTTPServer(("127.0.0.1", 0), _make_handler(code, body))
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()
    return server, port


# ---------------------------------------------------------------------------
# ping() helper tests
# ---------------------------------------------------------------------------


class TestPing:
    def test_success(self):
        server, port = _start_server(200)
        try:
            status, rtt_ms, error = ping(f"http://127.0.0.1:{port}/health")
            assert status == "OK"
            assert rtt_ms > 0
            assert error is None
        finally:
            server.server_close()

    def test_503(self):
        body = json.dumps(
            {"status": "unhealthy", "reason": "engine not initialized"}
        ).encode()
        server, port = _start_server(503, body)
        try:
            status, rtt_ms, error = ping(f"http://127.0.0.1:{port}/health")
            assert status == "FAIL"
            assert rtt_ms > 0
            assert "503" in error
        finally:
            server.server_close()

    def test_connection_refused(self):
        status, rtt_ms, error = ping("http://127.0.0.1:19999/health")
        assert status == "FAIL"
        assert rtt_ms >= 0
        assert "Cannot connect" in error


# ---------------------------------------------------------------------------
# PingCommand end-to-end tests (real HTTP server)
# ---------------------------------------------------------------------------


class TestPingCommandKvcache:
    def test_ok(self):
        body = json.dumps({"status": "healthy"}).encode()
        server, port = _start_server(200, body)
        try:
            cmd = PingCommand()

            class FakeArgs:
                target = "kvcache"
                url = f"http://127.0.0.1:{port}"
                format = "json"
                output = None

            buf = io.StringIO()
            # Standard
            from unittest.mock import patch

            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())
            assert output["title"] == "Ping KV Cache"
            m = output["metrics"]
            assert m["status"] == "OK"
            assert m["round_trip_time_ms"] > 0
        finally:
            server.server_close()

    def test_503_exits_1(self):
        body = json.dumps(
            {"status": "unhealthy", "reason": "engine not initialized"}
        ).encode()
        server, port = _start_server(503, body)
        try:
            cmd = PingCommand()

            class FakeArgs:
                target = "kvcache"
                url = f"http://127.0.0.1:{port}"
                format = "json"
                output = None

            with pytest.raises(SystemExit) as exc_info:
                cmd.execute(FakeArgs())
            assert exc_info.value.code == 1
        finally:
            server.server_close()

    def test_connection_refused_exits_1(self):
        cmd = PingCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://127.0.0.1:19999"
            format = "json"
            output = None

        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(FakeArgs())
        assert exc_info.value.code == 1


class TestPingCommandEngine:
    def test_ok(self):
        server, port = _start_server(200)
        try:
            cmd = PingCommand()

            class FakeArgs:
                target = "engine"
                url = f"http://127.0.0.1:{port}"
                format = "json"
                output = None

            buf = io.StringIO()
            # Standard
            from unittest.mock import patch

            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())
            assert output["title"] == "Ping Engine"
            m = output["metrics"]
            assert m["status"] == "OK"
            assert m["round_trip_time_ms"] > 0
        finally:
            server.server_close()

    def test_503_exits_1(self):
        server, port = _start_server(503)
        try:
            cmd = PingCommand()

            class FakeArgs:
                target = "engine"
                url = f"http://127.0.0.1:{port}"
                format = "json"
                output = None

            with pytest.raises(SystemExit) as exc_info:
                cmd.execute(FakeArgs())
            assert exc_info.value.code == 1
        finally:
            server.server_close()


# ---------------------------------------------------------------------------
# Default URL resolution
# ---------------------------------------------------------------------------


def _port_is_listening(host: str, port: int) -> bool:
    """Return True when something accepts connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _busy_port_hint(url: str) -> str:
    """Explain a busy default port when a default-URL test fails.

    Returns an empty string when the port is free, so callers can use the
    result directly as an assertion message.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    if not _port_is_listening(host, port):
        return ""
    return (
        f"\nNote: something is already listening on {host}:{port}. These tests no "
        f"longer depend on the port being free, so this is not the cause of a "
        f"failure here - but if you are debugging one, stop the process using "
        f"port {port} first (e.g. `pkill -f 'lmcache server'`, or `lsof -i "
        f":{port}` to find it) and re-run."
    )


class TestDefaultUrls:
    """``url=None`` resolves to the documented default.

    The resolution is asserted on the URL the command actually probes, so the
    test no longer depends on nothing listening on that port: on a machine
    that already runs an LMCache server, ``ping`` used to succeed and the
    expected ``SystemExit`` never happened.
    """

    @staticmethod
    def _args(target: str) -> argparse.Namespace:
        return argparse.Namespace(target=target, url=None, format="json", output=None)

    @staticmethod
    def _resolved_url(monkeypatch, target: str) -> str:
        probed: list[str] = []

        def fake_ping(url: str, timeout: int = 10):
            probed.append(url)
            return ("FAIL", 0.0, "connection refused")

        monkeypatch.setattr(ping_command, "ping", fake_ping)

        with pytest.raises(SystemExit) as exc_info:
            PingCommand().execute(TestDefaultUrls._args(target))

        assert exc_info.value.code != 0
        assert len(probed) == 1
        hint = _busy_port_hint(probed[0])
        if hint:
            warnings.warn(hint, stacklevel=2)
        return probed[0]

    def test_kvcache_default(self, monkeypatch):
        """url=None resolves to localhost:8080/healthcheck."""
        resolved = self._resolved_url(monkeypatch, "kvcache")
        assert resolved == "http://localhost:8080/healthcheck", _busy_port_hint(
            resolved
        )

    def test_engine_default(self, monkeypatch):
        """url=None resolves to localhost:8000/health."""
        resolved = self._resolved_url(monkeypatch, "engine")
        assert resolved == "http://localhost:8000/health", _busy_port_hint(resolved)


class TestBusyPortHint:
    """The hint explains how to free the default port when one is occupied."""

    def test_free_port_has_no_hint(self):
        # Port 1 on loopback is not served; nothing listens there.
        assert _busy_port_hint("http://localhost:1/health") == ""

    def test_busy_port_hint_names_the_port_and_the_command(self):
        # Standard
        import socket as _socket

        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]

            hint = _busy_port_hint(f"http://localhost:{port}/health")

        assert str(port) in hint
        assert "already listening" in hint
        assert "lsof" in hint
