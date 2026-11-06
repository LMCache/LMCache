# SPDX-License-Identifier: Apache-2.0
"""Public query-request metrics and endpoint fallback behavior."""

# Standard
from email.message import Message
from io import BytesIO
from unittest.mock import patch
import json
import urllib.error

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query._request import Request

_BASE = "http://engine.test/v1"
_COMPLETIONS = f"{_BASE}/completions"
_CHAT = f"{_BASE}/chat/completions"


def _http_error(status: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        _COMPLETIONS, status, "request failed", Message(), BytesIO(body.encode())
    )


def _response(*, chat: bool = True) -> BytesIO:
    choice: dict[str, object] = (
        {"delta": {"content": "answer"}} if chat else {"text": "answer"}
    )
    return BytesIO(
        (f"data: {json.dumps({'choices': [choice]})}\n\ndata: [DONE]\n").encode()
    )


def _request(*, completions_only: bool = False, chat_first: bool = False) -> Request:
    return Request(
        _BASE,
        "model",
        8,
        5.0,
        completions_only=completions_only,
        chat_first=chat_first,
    )


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize(
    "usage,text,expected_tokens",
    [
        ({"prompt_tokens": 10, "completion_tokens": 0}, "", 0),
        ({"prompt_tokens": 10, "completion_tokens": 2}, "Hello", 2),
        ({"prompt_tokens": 10}, "Hello", 128),
        ({"prompt_tokens": 10, "completion_tokens": None}, "Hello", 128),
        (None, "Hello", 128),
    ],
)
def test_send_request_preserves_reported_completion_count(
    chat: bool,
    usage: dict[str, int | None] | None,
    text: str,
    expected_tokens: int,
) -> None:
    """A reported zero is a token count; only an absent count uses the cap."""
    choice: dict[str, object] = {"delta": {"content": text}} if chat else {"text": text}
    chunks: list[dict[str, object]] = [{"choices": [choice]}]
    if usage is not None:
        chunks.append({"choices": [], "usage": usage})
    response = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    response += "data: [DONE]\n\n"
    request = Request(
        "http://engine.test",
        "model",
        max_tokens=128,
        timeout=5.0,
        completions_only=not chat,
        chat_first=chat,
    )

    with patch(
        "lmcache.cli.commands.query._request.urllib.request.urlopen",
        return_value=BytesIO(response.encode()),
    ):
        answer, metrics = request.send_request("prompt")

    assert answer == text
    assert metrics["output_tokens"] == ("Output tokens", expected_tokens)
    assert metrics["prompt_tokens"][1] == (10 if usage is not None else 0)
    if expected_tokens == 0:
        assert metrics["tpot_ms_per_token"][1] == 0.0
        assert metrics["throughput_tokens_per_s"][1] == 0.0


@pytest.mark.parametrize("status", [404, 405])
@pytest.mark.parametrize(
    "body",
    ["unsupported", '{"error":{"message":"unsupported","type":"NotFound"}}'],
)
def test_unsupported_http_endpoint_retries_chat(status: int, body: str) -> None:
    """Plain and OpenAI JSON errors retain their actual HTTP status."""
    with patch(
        "urllib.request.urlopen", side_effect=[_http_error(status, body), _response()]
    ) as urlopen:
        answer, _ = _request().send_request("prompt")
    assert answer == "answer"
    assert [call.args[0].full_url for call in urlopen.call_args_list] == [
        _COMPLETIONS,
        _CHAT,
    ]


@pytest.mark.parametrize(
    "response",
    [b"", b"data: [DONE]\n", b"unrecognized response\n"],
)
def test_empty_completion_retries_chat(response: bytes) -> None:
    """Both existing empty-response conditions still allow chat fallback."""
    with patch(
        "urllib.request.urlopen", side_effect=[BytesIO(response), _response()]
    ) as urlopen:
        answer, _ = _request().send_request("prompt")
    assert answer == "answer"
    assert [call.args[0].full_url for call in urlopen.call_args_list] == [
        _COMPLETIONS,
        _CHAT,
    ]


@pytest.mark.parametrize("status", [401, 429, 500])
@pytest.mark.parametrize(
    "message",
    [
        "request denied",
        "POST failed (HTTP 404)",
        "POST failed (HTTP 405)",
        "Empty response from engine (no SSE chunks parsed).",
        "No completion output from engine",
    ],
)
@pytest.mark.parametrize("json_body", [False, True])
def test_http_failure_never_retries_chat(
    status: int,
    message: str,
    json_body: bool,
) -> None:
    """Server-provided text cannot override an actual non-fallback status."""
    body = json.dumps({"error": {"message": message}}) if json_body else message
    error = _http_error(status, body)
    with patch("urllib.request.urlopen", side_effect=error) as urlopen:
        with pytest.raises(RuntimeError, match=f"HTTP {status}") as caught:
            _request().send_request("prompt")
    assert message in str(caught.value)
    assert caught.value.__cause__ is error
    urlopen.assert_called_once()
    assert urlopen.call_args.args[0].full_url == _COMPLETIONS


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("connection refused"),
        urllib.error.URLError("timed out"),
        urllib.error.URLError("Empty response from engine (HTTP 404)"),
        TimeoutError("timed out"),
    ],
)
def test_transport_failure_never_retries_chat(error: Exception) -> None:
    """Transport errors and direct socket timeouts issue just one request."""
    expected = TimeoutError if isinstance(error, TimeoutError) else RuntimeError
    with patch("urllib.request.urlopen", side_effect=error) as urlopen:
        with pytest.raises(expected):
            _request().send_request("prompt")
    urlopen.assert_called_once()
    assert urlopen.call_args.args[0].full_url == _COMPLETIONS


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize(
    "message", ["server failed", "Empty response from engine (HTTP 404)"]
)
def test_sse_error_never_retries_chat(partial: bool, message: str) -> None:
    """An SSE error is not a missing endpoint, even after output was consumed."""
    output = 'data: {"choices":[{"text":"partial"}]}\n' if partial else ""
    output += f"data: {json.dumps({'error': {'message': message}})}\n"
    with patch(
        "urllib.request.urlopen", return_value=BytesIO(output.encode())
    ) as urlopen:
        with pytest.raises(RuntimeError, match="engine|server"):
            _request().send_request("prompt")
    urlopen.assert_called_once()


@pytest.mark.parametrize(
    "error", [urllib.error.URLError("connection lost"), TimeoutError("timed out")]
)
def test_stream_read_failure_never_retries_chat(error: Exception) -> None:
    """A network failure after the first completion chunk must not repeat work."""
    with patch("urllib.request.urlopen") as urlopen:
        response = urlopen.return_value.__enter__.return_value
        response.readline.side_effect = [
            b'data: {"choices":[{"text":"partial"}]}\n',
            error,
        ]
        expected = TimeoutError if isinstance(error, TimeoutError) else RuntimeError
        with pytest.raises(expected):
            _request().send_request("prompt")
    urlopen.assert_called_once()
    assert response.readline.call_count == 2


@pytest.mark.parametrize("status", [404, 405])
@pytest.mark.parametrize("chat_first", [False, True])
def test_completions_only_never_retries(status: int, chat_first: bool) -> None:
    """--completions takes priority over --chat-first and disables fallback."""
    with patch(
        "urllib.request.urlopen",
        side_effect=_http_error(status, '{"error":"unsupported"}'),
    ) as urlopen:
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            _request(completions_only=True, chat_first=chat_first).send_request(
                "prompt"
            )
    urlopen.assert_called_once()
    assert urlopen.call_args.args[0].full_url == _COMPLETIONS


def test_completions_only_empty_response_never_retries() -> None:
    """An empty stream also respects --completions."""
    with patch("urllib.request.urlopen", return_value=BytesIO()) as urlopen:
        with pytest.raises(RuntimeError, match="Empty response"):
            _request(completions_only=True).send_request("prompt")
    urlopen.assert_called_once()


@pytest.mark.parametrize("json_body", [False, True])
def test_chat_first_missing_template_retries_completions(json_body: bool) -> None:
    """Missing-template HTTP errors retain the chat-first compatibility path."""
    message = "must provide a chat template"
    body = json.dumps({"error": {"message": message}}) if json_body else message
    with patch(
        "urllib.request.urlopen",
        side_effect=[_http_error(400, body), _response(chat=False)],
    ) as urlopen:
        answer, _ = _request(chat_first=True).send_request("prompt")
    assert answer == "answer"
    assert [call.args[0].full_url for call in urlopen.call_args_list] == [
        _CHAT,
        _COMPLETIONS,
    ]


@pytest.mark.parametrize("status", [400, 401, 404, 405, 429, 500])
def test_chat_first_other_errors_never_retry(status: int) -> None:
    """Chat-first only falls back when the chat template is missing."""
    with patch(
        "urllib.request.urlopen", side_effect=_http_error(status, '{"error":"failed"}')
    ) as urlopen:
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            _request(chat_first=True).send_request("prompt")
    urlopen.assert_called_once()
    assert urlopen.call_args.args[0].full_url == _CHAT


def test_both_endpoints_fail_keeps_errors_and_chain() -> None:
    """A failed compatibility retry reports both failures with a useful cause."""
    with patch(
        "urllib.request.urlopen",
        side_effect=[
            _http_error(404, "unsupported"),
            _http_error(500, "server failed"),
        ],
    ) as urlopen:
        with pytest.raises(RuntimeError, match="HTTP 404.*|HTTP 500") as caught:
            _request().send_request("prompt")
    assert "unsupported" in str(caught.value)
    assert "server failed" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert urlopen.call_count == 2


def test_empty_completion_missing_chat_template_keeps_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The empty-completion/base-model diagnostic remains available."""
    with patch(
        "urllib.request.urlopen",
        side_effect=[BytesIO(), _http_error(400, '{"error":"missing chat template"}')],
    ) as urlopen:
        with pytest.raises(RuntimeError, match="missing chat template"):
            _request().send_request("prompt")
    assert "--completions" in capsys.readouterr().err
    assert urlopen.call_count == 2


def test_chat_first_sse_missing_template_retries_completions() -> None:
    """The existing chat-first fallback also accepts streamed API errors."""
    with patch(
        "urllib.request.urlopen",
        side_effect=[
            BytesIO(b'data: {"error":{"message":"missing chat template"}}\n'),
            _response(chat=False),
        ],
    ) as urlopen:
        answer, _ = _request(chat_first=True).send_request("prompt")
    assert answer == "answer"
    assert [call.args[0].full_url for call in urlopen.call_args_list] == [
        _CHAT,
        _COMPLETIONS,
    ]
