# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache conf`` CLI subcommand.

Covers:
- Fetching ``/conf`` from a running server via ``--url`` (mock HTTP).
- Persisting fetched JSON via ``--output``.
- Error handling for connection refused and HTTP 5xx.
"""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch
import argparse
import io
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.conf import ConfCommand, _resolve_conf_endpoint

# ---------------------------------------------------------------------------
# Mock HTTP handler — mirrors the pattern used in test_ping.py
# ---------------------------------------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal handler that serves a canned response on any GET."""

    response_body: bytes = b""
    response_code: int = 200

    def do_GET(self):
        self.send_response(self.response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format, *args):
        pass  # silence stderr noise in test output


def _make_handler(code: int, body: bytes = b""):
    return type("_H", (_MockHandler,), {"response_code": code, "response_body": body})


def _start_server(code: int, body: bytes = b"") -> tuple[HTTPServer, int]:
    """Start a local one-shot HTTP server; return ``(server, port)``."""
    server = HTTPServer(("127.0.0.1", 0), _make_handler(code, body))
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()
    return server, port


def _fake_args(url: str | None = None, output: str | None = None) -> argparse.Namespace:
    """Build a namespace with the fields ``ConfCommand.execute`` reads."""
    return argparse.Namespace(url=url, output=output)


# ---------------------------------------------------------------------------
# --url source
# ---------------------------------------------------------------------------


class TestResolveConfEndpoint:
    def test_default_url(self) -> None:
        assert _resolve_conf_endpoint(None) == "http://localhost:8080/conf"

    def test_port_shorthand(self) -> None:
        assert _resolve_conf_endpoint("8080") == "http://localhost:8080/conf"

    def test_base_url(self) -> None:
        assert _resolve_conf_endpoint("http://localhost:8080") == (
            "http://localhost:8080/conf"
        )

    def test_conf_endpoint_url(self) -> None:
        assert _resolve_conf_endpoint("http://localhost:8080/conf") == (
            "http://localhost:8080/conf"
        )


class TestConfCommandFromUrl:
    """``lmcache conf --url URL`` fetches and pretty-prints ``/conf``."""

    def test_fetches_and_prints(self) -> None:
        """Successful 200 response is parsed and printed."""
        body = json.dumps({"http": {"port": 8080}, "mp": {"port": 5555}}).encode()
        server, port = _start_server(200, body)
        try:
            cmd = ConfCommand()
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(_fake_args(url=f"http://127.0.0.1:{port}"))

            printed = json.loads(buf.getvalue())
            assert printed["http"]["port"] == 8080
            assert printed["mp"]["port"] == 5555
        finally:
            server.server_close()

    def test_fetches_prints_and_writes_file(self, tmp_path: Path) -> None:
        """--output persists the JSON fetched from --url."""
        body = json.dumps({"z": 1, "a": 2}).encode()
        server, port = _start_server(200, body)
        output_path = tmp_path / "conf.json"
        try:
            cmd = ConfCommand()
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(
                    _fake_args(
                        url=f"http://127.0.0.1:{port}",
                        output=str(output_path),
                    )
                )

            assert json.loads(buf.getvalue()) == {"a": 2, "z": 1}
            assert json.loads(output_path.read_text()) == {"a": 2, "z": 1}
        finally:
            server.server_close()

    def test_http_error_exits_1(self) -> None:
        """5xx response causes SystemExit(1)."""
        server, port = _start_server(503, b'{"error": "not ready"}')
        try:
            cmd = ConfCommand()
            with pytest.raises(SystemExit) as exc:
                cmd.execute(_fake_args(url=f"http://127.0.0.1:{port}"))
            assert exc.value.code == 1
        finally:
            server.server_close()

    def test_connection_refused_exits_1(self) -> None:
        """Unreachable server causes SystemExit(1)."""
        cmd = ConfCommand()
        with pytest.raises(SystemExit) as exc:
            cmd.execute(_fake_args(url="http://127.0.0.1:19999"))
        assert exc.value.code == 1

    def test_default_url_is_localhost_8080(self) -> None:
        """url=None resolves to http://localhost:8080/conf."""
        captured: dict[str, str] = {}

        def _fake_fetch(url: str, timeout: int = 10) -> str:
            captured["url"] = url
            return "{}"

        cmd = ConfCommand()
        with patch(
            "lmcache.cli.commands.conf._fetch_from_url", side_effect=_fake_fetch
        ):
            cmd.execute(_fake_args(url=None))

        assert captured["url"] == "http://localhost:8080/conf"

    def test_port_shorthand_uses_localhost(self) -> None:
        """--url 8080 resolves to http://localhost:8080/conf."""
        captured: dict[str, str] = {}

        def _fake_fetch(url: str, timeout: int = 10) -> str:
            captured["url"] = url
            return "{}"

        cmd = ConfCommand()
        with patch(
            "lmcache.cli.commands.conf._fetch_from_url", side_effect=_fake_fetch
        ):
            cmd.execute(_fake_args(url="8080"))

        assert captured["url"] == "http://localhost:8080/conf"
