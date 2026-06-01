# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the disagg-proxy completion→chat stream chunk conversion.

Regression coverage for LMCache#3475: the example PD-disaggregation proxy
crashed with ``IndexError`` when the decoder streamed a chunk carrying an
empty ``choices`` list (e.g. the trailing usage/metadata frame emitted when
``stream_options.include_usage`` is set). The conversion helper now returns
``None`` for those frames (the caller skips them) instead of indexing
``choices[0]``. Empty-choices frames are dropped rather than forwarded
because this PD proxy miscounts the decoder subrequest's ``usage`` (the
prefill token is both appended to the decoder prompt and emitted to the
client as content), matching the pre-existing behavior where the converted
chat chunk never carried ``usage``.

The proxy lives under ``examples/`` and is a runnable script (guarded by
``if __name__ == "__main__"``), so the test inserts its directory on
``sys.path`` and imports the module to exercise the pure helper directly --
no server, sockets, or GPU required.
"""

# Standard
from pathlib import Path
import sys

# Third Party
import pytest

# Make the example script importable (it is not a package).
_PROXY_DIR = Path(__file__).resolve().parents[2] / "examples" / "disagg_prefill"
sys.path.insert(0, str(_PROXY_DIR))

# Third Party
from disagg_proxy_server import (  # noqa: E402
    completion_chunk_to_chat_chunk,
)


def _base_completion_chunk(choices: list, usage=None) -> dict:
    """Build a minimal ``text_completion`` stream chunk for tests."""
    chunk = {
        "id": "cmpl-abc",
        "object": "text_completion",
        "created": 1234567890,
        "model": "Qwen3-8B",
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def test_single_choice_is_converted_to_chat_delta() -> None:
    """A normal token chunk maps ``choices[0].text`` to ``delta.content``."""
    src = _base_completion_chunk(
        [
            {
                "index": 0,
                "text": "hello",
                "logprobs": {"tokens": ["hello"]},
                "finish_reason": None,
            }
        ]
    )

    out = completion_chunk_to_chat_chunk(src)

    assert out["object"] == "chat.completion.chunk"
    assert out["id"] == "cmpl-abc"
    assert out["created"] == 1234567890
    assert out["model"] == "Qwen3-8B"
    assert len(out["choices"]) == 1
    choice = out["choices"][0]
    assert choice["index"] == 0
    assert choice["delta"] == {"content": "hello"}
    assert choice["logprobs"] == {"tokens": ["hello"]}
    assert choice["finish_reason"] is None


def test_empty_choices_returns_none_instead_of_raising() -> None:
    """The #3475 crash case: a chunk with ``choices: []`` must not raise.

    Before the fix, ``choices[0]`` raised ``IndexError`` and aborted the
    whole streamed response. The helper now returns ``None`` so the caller
    skips the frame.
    """
    src = _base_completion_chunk([])

    assert completion_chunk_to_chat_chunk(src) is None


def test_empty_choices_with_usage_is_still_dropped() -> None:
    """The trailing usage/metadata frame (empty choices + usage) is dropped.

    Forwarding it would propagate the decoder subrequest's ``usage``, which
    is miscounted for the client request in this PD proxy. Returning
    ``None`` matches the pre-existing behavior of never emitting ``usage``.
    """
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    src = _base_completion_chunk([], usage=usage)

    assert completion_chunk_to_chat_chunk(src) is None


def test_missing_choices_key_returns_none() -> None:
    """A chunk lacking the ``choices`` key entirely returns ``None``.

    ``.get("choices") or []`` guards both the empty-list and missing-key
    shapes, so neither raises.
    """
    src = {
        "id": "cmpl-abc",
        "object": "text_completion",
        "created": 1234567890,
        "model": "Qwen3-8B",
    }

    assert completion_chunk_to_chat_chunk(src) is None


def test_usage_is_never_forwarded_on_token_chunks() -> None:
    """A token chunk that carries ``usage`` still converts without ``usage``.

    The proxy intentionally never surfaces the decoder subrequest's usage
    (it is miscounted for the client request), so even a non-empty chunk
    that happens to include ``usage`` drops it on conversion.
    """
    src = _base_completion_chunk(
        [{"index": 0, "text": "x", "finish_reason": None}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    out = completion_chunk_to_chat_chunk(src)

    assert out is not None
    assert "usage" not in out


def test_first_choice_used_when_multiple_present() -> None:
    """Only the first choice is mapped (the proxy is single-choice)."""
    src = _base_completion_chunk(
        [
            {"index": 0, "text": "first", "finish_reason": None},
            {"index": 1, "text": "second", "finish_reason": None},
        ]
    )

    out = completion_chunk_to_chat_chunk(src)

    assert len(out["choices"]) == 1
    assert out["choices"][0]["delta"] == {"content": "first"}


def test_missing_logprobs_and_finish_reason_default_to_none() -> None:
    """Optional per-choice fields fall back to ``None`` via ``.get``."""
    src = _base_completion_chunk([{"index": 0, "text": "x"}])

    out = completion_chunk_to_chat_chunk(src)

    choice = out["choices"][0]
    assert choice["logprobs"] is None
    assert choice["finish_reason"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
