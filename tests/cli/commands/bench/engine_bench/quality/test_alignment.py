# SPDX-License-Identifier: Apache-2.0
"""Tests for chunk alignment of prompt blocks."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality import alignment
from lmcache.cli.commands.bench.engine_bench.quality.alignment import (
    BLOCK_END,
    ChunkAligner,
)
from lmcache.cli.commands.bench.engine_bench.quality.prompts import (
    QA_SYSTEM_PROMPT,
    compose_qa_messages,
    compose_store_messages,
)

# Local
from ..fake_tokenizer import make_fake_tokenizer

_CHUNK = 32


@pytest.fixture
def aligner(monkeypatch) -> ChunkAligner:
    # The fake tokenizer's vocabulary is smaller than the default pool.
    monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)
    return ChunkAligner(make_fake_tokenizer(), "fake-model", _CHUNK, seed=7)


# Passages that open with letters, which a block ending in a filler word
# would glue onto ("wordVienna"), and one with surrounding whitespace.
_PASSAGES = ["Vienna is Austrian.", "alpha beta gamma", "  delta epsilon\n"]


def _prefix_tokens(aligner: ChunkAligner) -> int:
    """Tokens the fake chat template places ahead of the user content."""
    return aligner.token_length("user\n")


def _assert_passages_on_chunk_boundaries(
    aligner: ChunkAligner,
    messages: list[dict[str, str]],
    system: str,
    passages: list[str],
) -> None:
    """Assert each passage keeps its own tokens, at a chunk boundary, in context.

    The aligner sizes blocks by encoding each alone; that only holds if the
    prompt encodes to the same tokens at the same offsets.  The fake chat
    template's suffix is left out: it opens with a newline, where real
    templates open theirs with a special token that cannot merge.
    """
    tokenizer = make_fake_tokenizer()
    full = tokenizer.encode("user\n" + messages[0]["content"])
    start = _prefix_tokens(aligner)
    for index, block in enumerate([system, *passages]):
        ids = tokenizer.encode(block)
        assert full[start : start + len(ids)] == ids, f"re-tokenized: {block!r}"
        start += len(ids)
        if index < len(passages):
            assert start % _CHUNK == 0, f"passage {index} starts off-boundary"


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

    def test_passage_one_token_short_of_a_boundary_is_aligned(self, aligner) -> None:
        """One filler word plus its block end costs two tokens, not one."""
        passage = next(
            p
            for p in ("x" * n for n in range(1, 2 * _CHUNK))
            if aligner.token_length(p + BLOCK_END) % _CHUNK == _CHUNK - 1
        )
        block = aligner.passage_block(passage)
        assert aligner.token_length(block) % _CHUNK == 0

    def test_system_block_ends_on_a_chunk_boundary(self, aligner) -> None:
        block = aligner.system_block("Answer the question.")
        total = _prefix_tokens(aligner) + aligner.token_length(block)
        assert total % _CHUNK == 0

    def test_blocks_start_on_chunk_boundaries_in_the_measured_prompt(
        self, aligner
    ) -> None:
        system = aligner.system_block(QA_SYSTEM_PROMPT)
        passages = [aligner.passage_block(p) for p in _PASSAGES]
        messages = compose_qa_messages(system, passages, "Where is Vienna?")
        _assert_passages_on_chunk_boundaries(aligner, messages, system, passages)

    def test_blocks_start_on_chunk_boundaries_in_the_store_prompt(
        self, aligner
    ) -> None:
        system = aligner.system_block(QA_SYSTEM_PROMPT)
        passages = [aligner.passage_block(p) for p in _PASSAGES]
        messages = compose_store_messages(system, passages)
        _assert_passages_on_chunk_boundaries(aligner, messages, system, passages)
