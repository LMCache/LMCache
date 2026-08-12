# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared tokenizer helpers."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.tokenizers import (
    build_single_token_pool,
    try_load_tokenizer,
)

# Local
from .fake_tokenizer import FakeTokenizer, make_fake_tokenizer


class TestBuildSingleTokenPool:
    def test_returns_requested_size(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 50)
        assert len(pool.words) == 50

    def test_all_entries_are_single_tokens(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 50)
        for word in pool.words:
            assert len(fake_tokenizer.encode(f" {word}")) == 1

    def test_all_unique(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 100)
        assert len(set(pool.words)) == 100

    def test_rejects_non_letter_entries(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 100)
        for word in pool.words:
            assert word.isalpha() and word.islower()

    def test_rejects_words_failing_round_trip(self) -> None:
        """A vocabulary entry that re-encodes to two tokens is unusable."""
        tokenizer = make_fake_tokenizer(20, split_words=("waaaaword",))
        pool = build_single_token_pool(tokenizer, 19)
        assert "waaaaword" not in pool.words

    def test_deterministic(self, fake_tokenizer: FakeTokenizer) -> None:
        assert build_single_token_pool(fake_tokenizer, 30, seed=7) == (
            build_single_token_pool(fake_tokenizer, 30, seed=7)
        )
        assert build_single_token_pool(fake_tokenizer, 30, seed=7).words == (
            build_single_token_pool(fake_tokenizer, 30, seed=7).words
        )

    def test_different_seeds_differ(self, fake_tokenizer: FakeTokenizer) -> None:
        assert build_single_token_pool(fake_tokenizer, 30, seed=1) != (
            build_single_token_pool(fake_tokenizer, 30, seed=2)
        )

    def test_raises_when_pool_too_small(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match="stay a\\s+single token"):
            build_single_token_pool(fake_tokenizer, 10_000)

    def test_supports_sentencepiece_vocabularies(self) -> None:
        """SentencePiece keys must be read as keys, not via ``decode()``.

        ``decode()`` drops the ``▁`` word-start marker, so a decode-based
        filter finds no candidates at all on these models.
        """
        tokenizer = make_fake_tokenizer(100, marker="▁")
        pool = build_single_token_pool(tokenizer, 50)
        assert len(pool.words) == 50
        assert len(tokenizer.encode(pool.join(pool.words))) == 50

    def test_falls_back_to_separator_convention(self) -> None:
        """A tokenizer may charge for an explicit leading space (Phi-3).

        Prefixing every word then costs two tokens each, so the pool must
        separate the words instead and open the text with a letter.
        """
        tokenizer = make_fake_tokenizer(100, marker="▁", space_is_token=True)
        pool = build_single_token_pool(tokenizer, 50)
        assert pool.leading_space is False
        assert len(tokenizer.encode(pool.join(pool.words))) == 50

    def test_raises_cleanly_when_no_candidates(self) -> None:
        """No usable entries must give our error, not one from the tokenizer."""
        tokenizer = FakeTokenizer([], extra_tokens=("\n\n", " Capitalized"))
        with pytest.raises(ValueError, match="only 0 words"):
            build_single_token_pool(tokenizer, 10)


class TestTokenPoolJoin:
    def test_token_count_equals_word_count(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 40)
        assert len(fake_tokenizer.encode(pool.join(pool.words))) == 40

    def test_every_word_carries_its_leading_space(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        pool = build_single_token_pool(fake_tokenizer, 2)
        assert pool.leading_space is True
        assert pool.join(["alpha", "beta"]) == " alpha beta"

    def test_separator_convention_omits_the_leading_space(self) -> None:
        tokenizer = make_fake_tokenizer(100, marker="▁", space_is_token=True)
        pool = build_single_token_pool(tokenizer, 50)
        assert pool.join(["alpha", "beta"]) == "alpha beta"

    def test_empty_input(self, fake_tokenizer: FakeTokenizer) -> None:
        pool = build_single_token_pool(fake_tokenizer, 2)
        assert pool.join([]) == ""


class TestTryLoadTokenizer:
    def test_none_model_returns_none(self) -> None:
        assert try_load_tokenizer(None) is None

    def test_unloadable_model_returns_none(self) -> None:
        assert try_load_tokenizer("/nonexistent/path/to/no/such/tokenizer") is None
