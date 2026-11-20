# SPDX-License-Identifier: Apache-2.0
"""`lmcache kvcache delete` driven through the real CLI.

The unit tests patch the HTTP helper and hand-build a ``Namespace``, so the
parser, the mutually exclusive group and the exit codes are never exercised.
These tests call ``main([...])`` against a local server instead, which is what
a user runs.
"""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
import json
import threading

# Third Party
import pytest

# First Party
from lmcache.cli.main import main

_KEY = {
    "chunk_hash_hex": "00" * 32,
    "model_name": "org/model",
    "kv_rank": 0,
}


class _Recorder:
    """Collects the requests the CLI makes."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response: tuple[int, dict[str, Any]] = (200, {"ok": True, "deleted": 1})


@pytest.fixture
def server():
    """A local HTTP server that records one DELETE and answers with the stub."""
    recorder = _Recorder()

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            recorder.requests.append({"path": self.path, "body": body})
            status, payload = recorder.response
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_DELETE(self) -> None:  # noqa: N802 - http.server naming
            self._handle()

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            self._handle()

        def log_message(self, *args: object) -> None:
            """Silence the default stderr logging."""

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", recorder
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def _run(argv: list[str]) -> None:
    main(argv)


class TestParserAndRequest:
    def test_one_key_is_sent_with_the_default_tier(self, server) -> None:
        url, recorder = server

        _run(["kvcache", "delete", "--url", url, "--key", json.dumps(_KEY)])

        assert len(recorder.requests) == 1
        sent = recorder.requests[0]
        assert sent["path"] == "/cache/objects"
        assert sent["body"]["tier"] == "l1"
        assert sent["body"]["keys"] == [_KEY]

    def test_tier_and_force_reach_the_server(self, server) -> None:
        url, recorder = server

        _run(
            [
                "kvcache",
                "delete",
                "--url",
                url,
                "--key",
                json.dumps(_KEY),
                "--tier",
                "l2",
                "--adapter",
                "fs",
                "--force",
            ]
        )

        body = recorder.requests[0]["body"]
        assert body["tier"] == "l2"
        assert body["adapter"] == "fs"
        assert body["force"] is True

    def test_no_confirmation_prompt_is_needed(self, server) -> None:
        """`clear` and `quota delete` do not ask, so `delete` must not either."""
        url, recorder = server

        # A TTY check or an input() would hang or fail here; reaching the
        # server proves the command runs without one.
        _run(["kvcache", "delete", "--url", url, "--key", json.dumps(_KEY)])

        assert len(recorder.requests) == 1


class TestParserRejections:
    def test_missing_key_source_is_a_usage_error(self, server) -> None:
        url, _ = server

        with pytest.raises(SystemExit) as exc:
            _run(["kvcache", "delete", "--url", url])

        assert exc.value.code == 2

    def test_key_and_keys_file_are_mutually_exclusive(self, server, tmp_path) -> None:
        url, _ = server
        path = tmp_path / "keys.json"
        path.write_text(json.dumps([_KEY]), encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            _run(
                [
                    "kvcache",
                    "delete",
                    "--url",
                    url,
                    "--key",
                    json.dumps(_KEY),
                    "--keys-file",
                    str(path),
                ]
            )

        assert exc.value.code == 2

    def test_unknown_tier_is_a_usage_error(self, server) -> None:
        url, _ = server

        with pytest.raises(SystemExit) as exc:
            _run(
                [
                    "kvcache",
                    "delete",
                    "--url",
                    url,
                    "--key",
                    json.dumps(_KEY),
                    "--tier",
                    "l3",
                ]
            )

        assert exc.value.code == 2


class TestExitCodes:
    def test_server_rejection_exits_one(self, server) -> None:
        url, recorder = server
        recorder.response = (200, {"ok": False, "error": "adapter rejected"})

        with pytest.raises(SystemExit) as exc:
            _run(["kvcache", "delete", "--url", url, "--key", json.dumps(_KEY)])

        assert exc.value.code == 1

    def test_fastapi_validation_error_exits_one(self, server) -> None:
        """FastAPI's 422 body carries `detail`; the exit code still says failed."""
        url, recorder = server
        recorder.response = (422, {"detail": "chunk_hash_hex is not hex"})

        with pytest.raises(SystemExit) as exc:
            _run(["kvcache", "delete", "--url", url, "--key", json.dumps(_KEY)])

        assert exc.value.code == 1


class TestErrorMessage:
    """The message choice is pure, so it does not depend on log capture."""

    @staticmethod
    def _message(body: dict[str, object], default: str = "HTTP Error 422") -> str:
        # First Party
        from lmcache.cli.commands.kvcache import _error_message

        return _error_message(body, default)

    def test_detail_from_a_validation_error_is_used(self) -> None:
        assert self._message({"detail": "chunk_hash_hex is not hex"}) == (
            "chunk_hash_hex is not hex"
        )

    def test_message_wins_over_detail(self) -> None:
        assert self._message({"message": "adapter rejected", "detail": "x"}) == (
            "adapter rejected"
        )

    def test_error_field_is_accepted(self) -> None:
        assert self._message({"error": "no L2 adapter"}) == "no L2 adapter"

    def test_structured_detail_is_serialized(self) -> None:
        """FastAPI's 422 carries a list; the reason is in it."""
        assert json.loads(self._message({"detail": [{"loc": ["body"]}]})) == [
            {"loc": ["body"]}
        ]

    def test_an_empty_body_falls_back_to_the_default(self) -> None:
        assert self._message({}) == "HTTP Error 422"


class TestTheReasonIsReported:
    """The CLI has to log what the helper picked, not just pick it."""

    @staticmethod
    def _logged(caplog_handler: list[str]) -> str:
        return "\n".join(caplog_handler)

    @pytest.fixture
    def records(self):
        # Standard
        import logging

        # First Party
        from lmcache.cli.commands import kvcache

        collected: list[str] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                collected.append(record.getMessage())

        handler = _Collect()
        kvcache.logger.addHandler(handler)
        try:
            yield collected
        finally:
            kvcache.logger.removeHandler(handler)

    def test_a_string_detail_reaches_the_output(self, server, records) -> None:
        url, recorder = server
        recorder.response = (422, {"detail": "chunk_hash_hex is not hex"})

        with pytest.raises(SystemExit):
            _run(["kvcache", "delete", "--url", url, "--key", json.dumps(_KEY)])

        assert "chunk_hash_hex is not hex" in self._logged(records)

    def test_a_structured_detail_reaches_the_output(self, server, records) -> None:
        url, recorder = server
        recorder.response = (
            422,
            {
                "detail": [
                    {"loc": ["body", "key", "chunk_hash_hex"], "msg": "Field required"}
                ]
            },
        )

        with pytest.raises(SystemExit):
            _run(["kvcache", "delete", "--url", url, "--key", json.dumps(_KEY)])

        logged = self._logged(records)
        assert "chunk_hash_hex" in logged
        assert "Field required" in logged


class TestDocumentedInvocations:
    """The documented command lines have to parse and run as written.

    `-q` belongs to the `kvcache` parser, so it comes before the sub-command;
    writing it after the sub-command fails with `unrecognized arguments`.
    """

    def test_quiet_before_the_subcommand_runs_the_delete(self, server, capsys) -> None:
        url, recorder = server

        _run(["kvcache", "-q", "delete", "--url", url, "--key", json.dumps(_KEY)])

        assert len(recorder.requests) == 1
        assert capsys.readouterr().out == ""

    def test_quiet_before_the_subcommand_runs_the_clear(self, server, capsys) -> None:
        url, _ = server

        _run(["kvcache", "-q", "clear", "--url", url])

        assert capsys.readouterr().out == ""
