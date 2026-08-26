# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache query engine`` HTTP fallback behavior."""

# Standard
from email.message import Message
from io import BytesIO
from typing import Any
from unittest.mock import patch
import json
import urllib.error
import urllib.request

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query._request import Request


def _sse_lines(text: str, *, chat: bool) -> list[bytes]:
    payload = (
        {"choices": [{"delta": {"content": text}}]}
        if chat
        else {"choices": [{"text": text}]}
    )
    usage = {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    return [
        f"data: {json.dumps(payload)}\n".encode(),
        f"data: {json.dumps(usage)}\n".encode(),
        b"data: [DONE]\n",
    ]


class _FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    def read(self) -> bytes:
        return b"".join(self._lines)


def _http_error(url: str, code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "error", Message(), BytesIO(body))


def _is_completions(url: str) -> bool:
    return url.rstrip("/").endswith("/completions") and "/chat/completions" not in url


def _is_chat(url: str) -> bool:
    return "/chat/completions" in url


def _request() -> Request:
    return Request(
        "http://engine:8000/v1",
        "test-model",
        max_tokens=8,
        timeout=5.0,
    )


def _patch_urlopen(side_effect: Any) -> Any:
    return patch(
        "lmcache.cli.commands.query._request.urllib.request.urlopen",
        side_effect=side_effect,
    )


class TestQueryEngineCompletionsFallback:
    def test_http_404_falls_back_to_chat(self) -> None:
        calls: list[str] = []

        def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
            url = req.full_url
            calls.append(url)
            if _is_completions(url):
                raise _http_error(url, 404, b"Not Found")
            assert _is_chat(url)
            return _FakeResponse(_sse_lines("from-chat", chat=True))

        with _patch_urlopen(fake_urlopen):
            answer, _metrics = _request().send_request("hello")

        assert answer == "from-chat"
        assert any(_is_completions(url) for url in calls)
        assert any(_is_chat(url) for url in calls)

    def test_http_405_falls_back_to_chat(self) -> None:
        def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
            url = req.full_url
            if _is_completions(url):
                raise _http_error(url, 405, b"Method Not Allowed")
            return _FakeResponse(_sse_lines("ok", chat=True))

        with _patch_urlopen(fake_urlopen):
            answer, _metrics = _request().send_request("hello")
        assert answer == "ok"

    def test_empty_completions_response_falls_back_to_chat(self) -> None:
        def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
            url = req.full_url
            if _is_completions(url):
                return _FakeResponse([b"data: [DONE]\n"])
            return _FakeResponse(_sse_lines("recovered", chat=True))

        with _patch_urlopen(fake_urlopen):
            answer, _metrics = _request().send_request("hello")
        assert answer == "recovered"

    def test_json_404_body_still_falls_back_to_chat(self) -> None:
        """OpenAI-style 404 JSON must still be treated as unsupported, not a
        generic API error, so the chat endpoint is tried."""
        body = json.dumps({"error": {"message": "Not Found", "code": "not_found"}})

        def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
            url = req.full_url
            if _is_completions(url):
                raise _http_error(url, 404, body.encode())
            return _FakeResponse(_sse_lines("from-chat", chat=True))

        with _patch_urlopen(fake_urlopen):
            answer, _metrics = _request().send_request("hello")
        assert answer == "from-chat"

    @pytest.mark.parametrize("code", [401, 429, 500])
    def test_non_compatibility_http_errors_do_not_retry_chat(self, code: int) -> None:
        calls: list[str] = []

        def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
            url = req.full_url
            calls.append(url)
            raise _http_error(url, code, b'{"error":{"message":"nope"}}')

        with _patch_urlopen(fake_urlopen), pytest.raises(RuntimeError, match="HTTP"):
            _request().send_request("hello")

        assert calls
        assert all(_is_completions(url) for url in calls)
        assert not any(_is_chat(url) for url in calls)

    def test_timeout_does_not_retry_chat(self) -> None:
        calls: list[str] = []

        def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
            calls.append(req.full_url)
            raise urllib.error.URLError("timed out")

        with (
            _patch_urlopen(fake_urlopen),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            _request().send_request("hello")

        assert calls == ["http://engine:8000/v1/completions"]

    def test_completions_only_never_retries_chat(self) -> None:
        calls: list[str] = []

        def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
            calls.append(req.full_url)
            raise _http_error(req.full_url, 404, b"Not Found")

        req = Request(
            "http://engine:8000/v1",
            "test-model",
            max_tokens=8,
            timeout=5.0,
            completions_only=True,
        )
        with (
            _patch_urlopen(fake_urlopen),
            pytest.raises(RuntimeError, match="HTTP 404"),
        ):
            req.send_request("hello")

        assert calls == ["http://engine:8000/v1/completions"]
