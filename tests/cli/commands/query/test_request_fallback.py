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


def _completion_stream(text: str) -> BytesIO:
    chunk = json.dumps({"choices": [{"text": text}]})
    return BytesIO(f"data: {chunk}\n\ndata: [DONE]\n\n".encode())


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


def _chat_first_request() -> Request:
    return Request(
        "http://engine.test",
        "model",
        max_tokens=8,
        timeout=5.0,
        completions_only=False,
        chat_first=True,
    )


def _requested_paths(urlopen: MagicMock) -> list[str]:
    paths = []
    for call in urlopen.call_args_list:
        req = call.args[0]
        assert isinstance(req, urllib.request.Request)
        paths.append(req.full_url.removeprefix("http://engine.test/v1/"))
    return paths


@pytest.mark.parametrize(
    "outcome",
    [
        _http_error(401, '{"error": {"message": "invalid api key"}}'),
        _http_error(429, "rate limited"),
        _http_error(500, "internal server error"),
        urllib.error.URLError("timed out"),
        _http_error(500, '{"error": {"message": "Empty response from engine"}}'),
        BytesIO(
            b'data: {"error": {"message": "No completion output from engine"}}\n\n'
        ),
    ],
    ids=[
        "auth",
        "rate-limit",
        "server",
        "transport",
        "server-500-empty-text",
        "stream-error-empty-text",
    ],
)
def test_request_failure_is_reported_without_chat_retry(
    outcome: Exception | BytesIO,
) -> None:
    """Failures unrelated to endpoint support surface after a single request.

    Server-sent error text that resembles an empty stream must not count as one.
    """
    with patch(_URLOPEN, side_effect=[outcome]) as urlopen:
        with pytest.raises(RuntimeError):
            _completions_first_request().send_request("prompt")

    assert _requested_paths(urlopen) == ["completions"]


@pytest.mark.parametrize(
    "error",
    [
        _http_error(404, "Not Found"),
        _http_error(405, "Method Not Allowed"),
        _http_error(404, '{"error": {"message": "route not found"}}'),
        _http_error(
            501,
            '{"error": {"message": "The model does not support Completions API",'
            ' "type": "NotImplementedError", "code": 501}}',
        ),
    ],
    ids=["404", "405", "404-openai-json", "501-openai-json"],
)
def test_unsupported_completions_endpoint_falls_back_to_chat(
    error: Exception,
) -> None:
    """HTTP 404/405/501 from /v1/completions retries via chat completions."""
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


def test_empty_completion_then_missing_chat_template_reports_chat_error() -> None:
    """A base model (empty completion, no chat template) gets the chat error."""
    responses = [
        BytesIO(b"data: [DONE]\n\n"),
        _http_error(400, '{"error": {"message": "No chat template found"}}'),
    ]
    with patch(_URLOPEN, side_effect=responses) as urlopen:
        with pytest.raises(RuntimeError, match="No chat template found") as exc_info:
            _completions_first_request().send_request("prompt")

    assert "; then " not in str(exc_info.value)
    assert _requested_paths(urlopen) == ["completions", "chat/completions"]


@pytest.mark.parametrize(
    "outcome",
    [
        _http_error(500, '{"error": {"message": "Could not load chat template"}}'),
        _http_error(401, '{"error": {"message": "key may not use chat template"}}'),
        _http_error(429, '{"error": {"message": "chat template quota exceeded"}}'),
        BytesIO(
            b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
            b'data: {"error": {"message": "chat template render failed"}}\n\n'
        ),
        _http_error(
            400,
            '{"error": {"message": "Input must be an object",'
            ' "type": "BadRequestError", "param": "chat_template_kwargs",'
            ' "code": 400}}',
        ),
        _http_error(
            400,
            '{"message": "Input must be an object", "param": "chat_template_kwargs"}',
        ),
        _http_error(
            400,
            '{"error": {"message": "The chat template rejected the request:'
            ' roles must alternate", "type": "BadRequestError", "code": 400}}',
        ),
    ],
    ids=[
        "server-500",
        "auth-401",
        "rate-limit-429",
        "error-after-token",
        "400-template-field-name",
        "400-template-field-name-flat",
        "400-configured-template-rejects",
    ],
)
def test_chat_first_failure_mentioning_template_is_not_retried(
    outcome: Exception | BytesIO,
) -> None:
    """Chat failures other than a missing-template 400 surface after one request.

    Their text or field names may mention a chat template; only a 400 whose
    error message says no template exists means the model has none.
    """
    with patch(_URLOPEN, side_effect=[outcome]) as urlopen:
        with pytest.raises(RuntimeError, match="chat template|Input must be an object"):
            _chat_first_request().send_request("prompt")

    assert _requested_paths(urlopen) == ["chat/completions"]


@pytest.mark.parametrize(
    "message",
    [
        "As of transformers v4.44, default chat template is no longer allowed,"
        " so you must provide a chat template if the tokenizer does not define one.",
        "Cannot use chat template functions because tokenizer.chat_template is"
        " not set and no template argument was passed!",
    ],
    ids=["vllm", "transformers"],
)
def test_chat_first_missing_template_falls_back_to_completions(message: str) -> None:
    """An HTTP 400 saying the model has no chat template retries via completions."""
    body = {"error": {"message": message, "type": "BadRequestError", "code": 400}}
    missing = _http_error(400, json.dumps(body))
    with patch(_URLOPEN, side_effect=[missing, _completion_stream("hi")]) as urlopen:
        answer, _ = _chat_first_request().send_request("prompt")

    assert answer == "hi"
    assert _requested_paths(urlopen) == ["chat/completions", "completions"]
