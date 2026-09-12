# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCacheRequestStream output-token accounting.

These exercise the ``output_tokens`` contract: the property must return only
the generated tokens, never the prompt or a KV-edited live sequence, and must
stay consistent with ``decoded_tokens``.
"""

# First Party
from lmcache.sdk.request import LMCacheRequestStream, TokenEvent


def _two_token_completion(prompt_token_ids, sampling_params, cache_salt):
    yield TokenEvent(token_id=30, text="x")
    yield TokenEvent(token_id=31, text="y")


def _make_stream(post_completion, prompt_token_ids):
    return LMCacheRequestStream(
        contexts=[],
        post_completion=post_completion,
        prompt_token_ids=prompt_token_ids,
    )


def test_output_tokens_excludes_prompt():
    stream = _make_stream(_two_token_completion, [10, 20])
    stream.generate({"max_tokens": 2})

    assert stream.output_tokens == [30, 31]
    assert len(stream.output_tokens) == stream.decoded_tokens


def test_output_tokens_accumulate_across_generate_calls():
    stream = _make_stream(_two_token_completion, [10, 20])
    stream.generate({"max_tokens": 2})
    stream.generate({"max_tokens": 2})

    assert stream.output_tokens == [30, 31, 30, 31]
    assert len(stream.output_tokens) == stream.decoded_tokens


def test_output_tokens_empty_when_no_generation():
    stream = _make_stream(lambda *_: iter(()), [1, 2, 3])
    stream.generate({"max_tokens": 5})

    assert stream.output_tokens == []
    assert stream.decoded_tokens == 0


def test_output_tokens_survive_a_replaced_live_sequence():
    # update()/modify_kv() rebind self.tokens to the edited sequence; the
    # generated-token history must not be affected by that.
    stream = _make_stream(_two_token_completion, [10, 20])
    stream.generate({"max_tokens": 2})
    stream.tokens = [99, 98, 97]

    assert stream.output_tokens == [30, 31]
