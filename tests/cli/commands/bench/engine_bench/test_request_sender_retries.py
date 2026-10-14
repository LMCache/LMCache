# SPDX-License-Identifier: Apache-2.0
"""Verify that benchmark requests do not hide errors behind SDK retries."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import Mock
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender


@contextmanager
def _error_server(status: int) -> Iterator[tuple[str, list[str]]]:
    attempts: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            attempts.append(self.path)
            body = json.dumps({"error": {"message": "Engine unavailable"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            # Keep the failing regression fast if SDK retries are enabled.
            self.send_header("retry-after-ms", "1")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", attempts
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
@pytest.mark.parametrize("completions_mode", [False, True])
async def test_retryable_error_is_attempted_once(
    status: int, completions_mode: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheduled request must produce one HTTP attempt and one failed result."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    callback = Mock()
    with _error_server(status) as (url, attempts):
        sender = RequestSender(
            url,
            "test-model",
            completions_mode=completions_mode,
            on_finished=[callback],
        )
        try:
            result = await sender.send_request(
                "request-1", [{"role": "user", "content": "Hello"}]
            )
        finally:
            await sender.close()

    endpoint = "/v1/completions" if completions_mode else "/v1/chat/completions"
    assert attempts == [endpoint]
    assert not result.successful
    assert str(status) in result.error
    callback.assert_called_once_with(result, "")
