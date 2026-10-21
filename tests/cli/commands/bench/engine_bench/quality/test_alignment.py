# SPDX-License-Identifier: Apache-2.0
"""Tests for chunk alignment of prompt blocks."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality import alignment
from lmcache.cli.commands.bench.engine_bench.quality.alignment import ChunkAligner

# Local
from ..fake_tokenizer import make_fake_tokenizer

_CHUNK = 32


@pytest.fixture
def aligner(monkeypatch) -> ChunkAligner:
    # The fake tokenizer's vocabulary is smaller than the default pool.
    monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)
    return ChunkAligner(make_fake_tokenizer(), "fake-model", _CHUNK, seed=7)


def _prefix_tokens(aligner: ChunkAligner) -> int:
    """Tokens the fake chat template places ahead of the user content."""
    return aligner.token_length("user\n")


class TestChunkAligner:
    def test_rejects_non_positive_alignment(self, monkeypatch) -> None:
        monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)
        with pytest.raises(ValueError, match="align_tokens"):
            ChunkAligner(make_fake_tokenizer(), "fake-model", 0, seed=7)

    def test_passage_occupies_whole_chunks(self, aligner) -> None:
        block = aligner.passage_block("alpha beta gamma")
        assert aligner.token_length(block) % _CHUNK == 0

    def test_passage_keeps_its_text_first(self, aligner) -> None:
        assert aligner.passage_block("alpha beta").startswith("alpha beta")

    def test_passage_padding_is_deterministic(self, aligner, monkeypatch) -> None:
        """Two runs must build byte-identical prompts."""
        again = ChunkAligner(make_fake_tokenizer(), "fake-model", _CHUNK, seed=7)
        assert aligner.passage_block("alpha") == again.passage_block("alpha")

    def test_distinct_passages_get_distinct_filler(self, aligner) -> None:
        """Shared filler would make two passages' chunks collide."""
        first = aligner.passage_block("alpha")[len("alpha") :]
        second = aligner.passage_block("beta")[len("beta") :]
        assert first != second

    def test_system_block_ends_on_a_chunk_boundary(self, aligner) -> None:
        block = aligner.system_block("Answer the question.")
        total = _prefix_tokens(aligner) + aligner.token_length(block)
        assert total % _CHUNK == 0
