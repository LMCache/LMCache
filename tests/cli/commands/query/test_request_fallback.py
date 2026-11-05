# SPDX-License-Identifier: Apache-2.0
"""Endpoint fallback of ``Request.send_request`` on completions failures."""

# Standard
from io import BytesIO
from unittest.mock import MagicMock, patch
import json
import urllib.error
import urllib.request

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query._request import Request

_URLOPEN = "lmcache.cli.commands.query._request.urllib.request.urlopen"


def _chat_stream(text: str) -> BytesIO:
    chunks = [
        {"choices": [{"delta": {"content": text}}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return BytesIO((body + "data: [DONE]\n\n").encode())


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://engine.test/v1/completions",
        code,
        "error",
        None,  # type: ignore[arg-type]
        BytesIO(body.encode()),
    )


def _completions_first_request() -> Request:
    return Request(
        "http://engine.test",
        "model",
        max_tokens=8,
        timeout=5.0,
        completions_only=False,
        chat_first=False,
    )


def _requested_paths(urlopen: MagicMock) -> list[str]:
    paths = []
    for call in urlopen.call_args_list:
        req = call.args[0]
        assert isinstance(req, urllib.request.Request)
        paths.append(req.full_url.removeprefix("http://engine.test/v1/"))
    return paths


@pytest.mark.parametrize(
    "error",
    [
        _http_error(401, '{"error": {"message": "invalid api key"}}'),
        _http_error(429, "rate limited"),
        _http_error(500, "internal server error"),
        urllib.error.URLError("timed out"),
    ],
    ids=["auth", "rate-limit", "server", "transport"],
)
def test_request_failure_is_reported_without_chat_retry(error: Exception) -> None:
    """Failures unrelated to endpoint support surface after a single request."""
    with patch(_URLOPEN, side_effect=[error]) as urlopen:
        with pytest.raises(RuntimeError):
            _completions_first_request().send_request("prompt")

    assert _requested_paths(urlopen) == ["completions"]


@pytest.mark.parametrize(
    "error",
    [
        _http_error(404, "Not Found"),
        _http_error(405, "Method Not Allowed"),
        _http_error(404, '{"error": {"message": "route not found"}}'),
    ],
    ids=["404", "405", "404-openai-json"],
)
def test_unsupported_completions_endpoint_falls_back_to_chat(
    error: Exception,
) -> None:
    """HTTP 404/405 from /v1/completions retries via /v1/chat/completions."""
    with patch(_URLOPEN, side_effect=[error, _chat_stream("hi")]) as urlopen:
        answer, _ = _completions_first_request().send_request("prompt")

    assert answer == "hi"
    assert _requested_paths(urlopen) == ["completions", "chat/completions"]


def test_empty_completion_response_falls_back_to_chat() -> None:
    """An empty completions stream keeps its existing chat fallback."""
    responses = [BytesIO(b"data: [DONE]\n\n"), _chat_stream("hi")]
    with patch(_URLOPEN, side_effect=responses) as urlopen:
        answer, _ = _completions_first_request().send_request("prompt")

    assert answer == "hi"
    assert _requested_paths(urlopen) == ["completions", "chat/completions"]
