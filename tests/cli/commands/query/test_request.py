# SPDX-License-Identifier: Apache-2.0
"""Public query-request token metrics for OpenAI-compatible streams."""

# Standard
from io import BytesIO
from unittest.mock import patch
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query._request import Request


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


class _TimedStream:
    """An SSE body where each line arrives 100 ms after the previous one."""

    def __init__(self, lines: list[str], clock: list[float]) -> None:
        self._lines = [line.encode() for line in lines]
        self._clock = clock

    def __enter__(self) -> "_TimedStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def readline(self) -> bytes:
        self._clock[0] += 0.1
        return self._lines.pop(0) if self._lines else b""


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning": "Let me think"},
        {"reasoning_content": "Let me think"},
        {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]},
    ],
)
def test_chat_ttft_counts_reasoning_and_tool_call_tokens(
    delta: dict[str, object],
) -> None:
    chunks = [
        {"choices": [{"delta": delta}]},
        {"choices": [{"delta": delta}]},
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ]
    lines = [f"data: {json.dumps(chunk)}\n" for chunk in chunks] + ["data: [DONE]\n"]
    clock = [0.0]
    request = Request(
        "http://engine.test",
        "model",
        max_tokens=128,
        timeout=5.0,
        completions_only=False,
        chat_first=True,
    )

    with (
        patch(
            "lmcache.cli.commands.query._request.urllib.request.urlopen",
            return_value=_TimedStream(lines, clock),
        ),
        patch("lmcache.cli.commands.query._request.time.time", lambda: clock[0]),
    ):
        answer, metrics = request.send_request("prompt")

    assert answer == "Hi"
    assert metrics["ttft_ms"][1] == pytest.approx(100.0)
