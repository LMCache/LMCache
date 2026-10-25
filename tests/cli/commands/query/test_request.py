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
