# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCacheRequestStream generation history.
Covers the output_tokens contract across generate(), update() and modify_kv(),
using a scripted engine and an in-memory KV context instead of a server.
"""

# Standard
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.context import LMCacheSDKContext
from lmcache.sdk.request import (
    LMCacheRequestStream,
    LMCacheRequestStreamError,
    TokenEvent,
)

CHUNK_SIZE = 4
PROMPT = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]


class _ScriptedEngine:
    """post_completion that yields a fixed token list per call.

    Each call consumes the next script entry, truncated to max_tokens, and
    records the prompt it was sent. A token id of -1 raises instead of
    yielding, to simulate an engine failing mid-stream.
    """

    def __init__(self, script: list[list[int]]) -> None:
        self.script = script
        self.prompts: list[list[int]] = []

    def __call__(
        self,
        prompt_token_ids: list[int],
        sampling_params: dict[str, Any],
        cache_salt: str,
    ) -> Iterator[TokenEvent]:
        self.prompts.append(list(prompt_token_ids))
        tokens = self.script[len(self.prompts) - 1]
        for token_id in tokens[: sampling_params.get("max_tokens", 1)]:
            if token_id == -1:
                raise RuntimeError("engine failed mid-stream")
            yield TokenEvent(token_id=token_id, text=f"<{token_id}>")


class _FakeKVContext:
    """In-memory KV context that, like the real one, retrieves only the
    chunk-aligned prefix of the requested tokens."""

    kind = LMCacheSDKCacheKind.KV
    chunk_size = CHUNK_SIZE

    def lookup(self, tokens: Sequence[int], cache_salt: str = "") -> int:
        return (len(tokens) // CHUNK_SIZE) * CHUNK_SIZE

    def retrieve(
        self, tokens: Sequence[int], cache_salt: str = "", start_token_id: int = 0
    ) -> tuple[torch.Tensor, None]:
        aligned = (len(tokens) // CHUNK_SIZE) * CHUNK_SIZE
        return torch.zeros(2, 1, aligned - start_token_id, 1), None

    def store(
        self, kv: torch.Tensor, tokens: Sequence[int], cache_salt: str = ""
    ) -> bool:
        return True


def _make_stream(engine: _ScriptedEngine) -> LMCacheRequestStream:
    ctx = cast(LMCacheSDKContext, _FakeKVContext())
    return LMCacheRequestStream([ctx], engine, prompt_token_ids=PROMPT)


def _drop_first_chunk(
    tensors: Mapping[LMCacheSDKCacheKind, torch.Tensor], tokens: Sequence[int]
) -> tuple[torch.Tensor, Sequence[int]]:
    kv = tensors[LMCacheSDKCacheKind.KV]
    return kv[:, :, CHUNK_SIZE:], list(tokens[CHUNK_SIZE:])


def test_output_tokens_excludes_prompt() -> None:
    """Check that output_tokens holds only generated ids, never the prompt."""
    stream = _make_stream(_ScriptedEngine([[30, 31]]))

    stream.generate({"max_tokens": 2})

    assert stream.output_tokens == [30, 31]
    assert len(stream.output_tokens) == stream.decoded_tokens


def test_output_tokens_accumulate_across_generate_calls() -> None:
    """Check that ids from every generate() are concatenated in order and
    caller-supplied suffix tokens are not counted as output."""
    stream = _make_stream(_ScriptedEngine([[30], [40, 41]]))

    stream.generate({"max_tokens": 1})
    stream.generate({"max_tokens": 2}, suffix_tokens=[7, 8])

    assert stream.output_tokens == [30, 40, 41]
    assert stream.decoded_tokens == 3


def test_output_tokens_empty_when_nothing_generated() -> None:
    """Check that a pass yielding no tokens leaves output_tokens empty."""
    stream = _make_stream(_ScriptedEngine([[]]))

    stream.generate({"max_tokens": 5})

    assert stream.output_tokens == []
    assert stream.decoded_tokens == 0


def test_output_tokens_unaffected_by_modify_kv() -> None:
    """Check output_tokens through the documented prefill -> modify_kv ->
    decode flow, where modify_kv truncates the live sequence and moves the
    prefill token into suffix_tokens."""
    stream = _make_stream(_ScriptedEngine([[500], [600, 601, 602]]))

    stream.generate({"max_tokens": 1})
    assert stream.output_tokens == [500]

    stream.modify_kv(_drop_first_chunk)
    assert stream.output_tokens == [500]

    stream.generate({"max_tokens": 3})
    assert stream.output_tokens == [500, 600, 601, 602]
    assert stream.output_text == "<500><600><601><602>"
    assert len(stream.output_tokens) == stream.decoded_tokens


def test_output_tokens_unaffected_by_update() -> None:
    """Check that installing an edited sequence via update() keeps the
    generated history."""
    stream = _make_stream(_ScriptedEngine([[30, 31]]))
    stream.generate({"max_tokens": 2})

    stream.update(LMCacheSDKCacheKind.KV, torch.zeros(2, 1, 3, 1), [9, 8, 7])

    assert stream.tokens == [9, 8, 7]
    assert stream.output_tokens == [30, 31]


def test_output_tokens_keep_tokens_from_failed_generate() -> None:
    """Check that tokens yielded before a mid-stream failure stay in
    output_tokens, consistent with decoded_tokens and output_text."""
    stream = _make_stream(_ScriptedEngine([[30, -1]]))

    with pytest.raises(LMCacheRequestStreamError):
        stream.generate({"max_tokens": 2})

    assert stream.output_tokens == [30]
    assert stream.decoded_tokens == 1
    assert stream.output_text == "<30>"


def test_mutating_output_tokens_does_not_change_stream() -> None:
    """Check that output_tokens returns a copy: editing it changes neither
    the generated history nor the next prompt."""
    engine = _ScriptedEngine([[30, 31], [40]])
    stream = _make_stream(engine)
    stream.generate({"max_tokens": 2})

    stream.output_tokens.clear()
    stream.generate({"max_tokens": 1})

    assert stream.output_tokens == [30, 31, 40]
    assert engine.prompts[1] == PROMPT + [30, 31]


def test_mutating_suffix_tokens_does_not_change_next_prompt() -> None:
    """Check that suffix_tokens returns a copy: editing it does not change
    the uncached tail that the next generate() replays."""
    engine = _ScriptedEngine([[500], [600]])
    stream = _make_stream(engine)
    stream.generate({"max_tokens": 1})
    stream.modify_kv(_drop_first_chunk)
    suffix = stream.suffix_tokens

    stream.suffix_tokens.clear()
    stream.generate({"max_tokens": 1})

    assert suffix == [108, 109, 500]
    assert engine.prompts[1] == [104, 105, 106, 107, 108, 109, 500]
