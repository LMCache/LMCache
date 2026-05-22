# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache conf`` CLI subcommand.

Covers:
- Reading a JSON configuration file via ``--file``.
- Fetching ``/conf`` from a running server via ``--url`` (mock HTTP).
- Error handling for connection refused, HTTP 5xx, and unreadable files.
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
from lmcache.cli.commands.conf import ConfCommand

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


def _fake_args(url: str | None = None, file: str | None = None) -> argparse.Namespace:
    """Build a namespace with the fields ``ConfCommand.execute`` reads."""
    return argparse.Namespace(url=url, file=file)


# ---------------------------------------------------------------------------
# --file source
# ---------------------------------------------------------------------------


class TestConfCommandFromFile:
    """``lmcache conf --file PATH`` reads and pretty-prints a JSON file."""

    def test_reads_and_prints_json(self, tmp_path: Path) -> None:
        """Valid JSON file is pretty-printed to stdout."""
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps({"http": {"port": 8080}}))

        cmd = ConfCommand()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cmd.execute(_fake_args(file=str(dump)))

        printed = json.loads(buf.getvalue())
        assert printed == {"http": {"port": 8080}}

    def test_indented_output(self, tmp_path: Path) -> None:
        """Output is indented for readability."""
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps({"k": "v"}))

        cmd = ConfCommand()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cmd.execute(_fake_args(file=str(dump)))

        assert "\n" in buf.getvalue()
        assert "  " in buf.getvalue()

    def test_keys_are_sorted(self, tmp_path: Path) -> None:
        """Output keys are sorted for stable diffs."""
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps({"z": 1, "a": 2}))

        cmd = ConfCommand()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cmd.execute(_fake_args(file=str(dump)))

        out = buf.getvalue()
        assert out.index('"a"') < out.index('"z"')

    def test_nonexistent_file_exits_1(self, tmp_path: Path) -> None:
        """Missing file causes SystemExit(1)."""
        cmd = ConfCommand()
        with pytest.raises(SystemExit) as exc:
            cmd.execute(_fake_args(file=str(tmp_path / "missing.json")))
        assert exc.value.code == 1

    def test_non_json_file_is_printed_verbatim(self, tmp_path: Path) -> None:
        """If the file is not valid JSON, contents are printed unmodified."""
        dump = tmp_path / "dump.json"
        dump.write_text("not json at all")

        cmd = ConfCommand()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cmd.execute(_fake_args(file=str(dump)))

        assert "not json at all" in buf.getvalue()


# ---------------------------------------------------------------------------
# --url source
# ---------------------------------------------------------------------------


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
