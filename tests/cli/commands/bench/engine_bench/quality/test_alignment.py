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
    STORE_PROMPT_END,
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

    def test_trimming_template_keeps_store_and_measured_blocks_identical(
        self, monkeypatch
    ) -> None:
        """A template that trims content must not drop the last BLOCK_END.

        Llama 3 Instruct trims user content; if a store prompt ended with
        its last passage, the trailing newline would be dropped there but
        kept before the question, and that passage's last chunk would differ.
        """
        monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)
        aligner = ChunkAligner(
            make_fake_tokenizer(trim_content=True), "fake-model", _CHUNK, seed=7
        )
        system = aligner.system_block(QA_SYSTEM_PROMPT)
        passages = [aligner.passage_block(p) for p in _PASSAGES]
        prompts = [
            compose_store_messages(system, passages, STORE_PROMPT_END),
            compose_qa_messages(system, passages, "Where is Vienna?"),
        ]
        store, measured = (aligner.place_blocks(m, passages)[1] for m in prompts)
        assert [p.problem for p in store] == [""] * len(passages)
        assert [p.problem for p in measured] == [""] * len(passages)
        assert [p.tokens for p in store] == [p.tokens for p in measured]


def _framed(filler: alignment.Filler, seed: int = 7) -> ChunkAligner:
    return ChunkAligner(
        make_fake_tokenizer(bpe_pieces=True), "fake-model", _CHUNK, seed, filler=filler
    )


class TestFramedFillers:
    """The dots / dashes fillers: a labelled frame around the passage."""

    @pytest.mark.parametrize("filler", [alignment.Filler.DOTS, alignment.Filler.DASHES])
    def test_passage_occupies_whole_chunks(self, filler) -> None:
        aligner = _framed(filler)
        for text in ("Alpha beta gamma.", "Rome is in Italy.", "x" * 80):
            assert aligner.token_length(aligner.passage_block(text)) % _CHUNK == 0

    def test_frame_labels_the_passage_with_its_id(self) -> None:
        text = "Paris is in France."
        ident = alignment.block_id(text)
        block = _framed(alignment.Filler.DOTS).passage_block(text)
        assert block.startswith(f"Passage {ident}:\n{text}\n\n")
        assert block.endswith(f"\nend of passage {ident}")

    def test_block_ends_with_a_digit(self) -> None:
        """A final digit never merges with the next block's first word."""
        for text in ("a", "b", "Rome is in Italy.", "Oslo"):
            ident = alignment.block_id(text)
            assert len(ident) == 7 and ident[-1].isdigit()

    @pytest.mark.parametrize(
        ("filler", "unit"),
        [(alignment.Filler.DOTS, " ."), (alignment.Filler.DASHES, " -")],
    )
    def test_padding_repeats_the_unit(self, filler, unit) -> None:
        aligner = _framed(filler)
        block = aligner.passage_block("Rome is in Italy.")
        pad = aligner.pad_tokens("Rome is in Italy.")
        assert pad > 0
        assert unit * pad in block

    def test_padding_does_not_depend_on_the_seed(self) -> None:
        first = _framed(alignment.Filler.DOTS, seed=1).passage_block("Rome.")
        second = _framed(alignment.Filler.DOTS, seed=2).passage_block("Rome.")
        assert first == second

    def test_system_block_ends_on_a_chunk_boundary(self) -> None:
        aligner = _framed(alignment.Filler.DOTS)
        block = aligner.system_block("Answer the question.")
        assert block.endswith("\nend of instructions 0")
        total = _prefix_tokens(aligner) + aligner.token_length(block)
        assert total % _CHUNK == 0
        assert aligner.system_pad_tokens > 0

    def test_words_is_the_default_and_unchanged(self, aligner) -> None:
        assert aligner.filler is alignment.Filler.WORDS
        assert aligner.passage_block("alpha beta").startswith("alpha beta\n")


class TestPlaceBlocks:
    """The in-context check that each block lands on chunk boundaries."""

    @staticmethod
    def _messages(*parts: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": "".join(parts)}]

    def test_aligned_blocks_have_no_problem(self) -> None:
        aligner = _framed(alignment.Filler.DOTS)
        system = aligner.system_block("Answer the question.")
        blocks = [aligner.passage_block(t) for t in ("Rome is in Italy.", "Oslo.")]
        total, placements = aligner.place_blocks(
            self._messages(system, *blocks, "\n\nQuestion: Capital?"), blocks
        )
        assert [p.problem for p in placements] == ["", ""]
        assert placements[0].start_token % _CHUNK == 0
        assert total > sum(len(p.tokens) for p in placements)

    def test_a_shifted_block_is_reported(self) -> None:
        aligner = _framed(alignment.Filler.DOTS)
        system = aligner.system_block("Answer the question.")
        block = aligner.passage_block("Rome is in Italy.")
        _, (placement,) = aligner.place_blocks(
            self._messages(system, " extra", block), [block]
        )
        assert "chunk boundary" in placement.problem

    def test_a_merged_boundary_is_reported(self) -> None:
        """A newline before the block merges with its leading newline into
        one "\n\n" token, which then straddles the block's start."""
        aligner = _framed(alignment.Filler.DOTS)
        _, (placement,) = aligner.place_blocks(self._messages("a\n", "\nb"), ["\nb"])
        assert "straddles its start" in placement.problem

    def test_a_block_the_template_changed_is_reported(self, monkeypatch) -> None:
        aligner = _framed(alignment.Filler.DOTS)
        tokenizer = aligner._tokenizer
        monkeypatch.setattr(
            tokenizer,
            "apply_chat_template",
            lambda messages, **_: "user\n" + messages[0]["content"].strip(),
        )
        _, (placement,) = aligner.place_blocks(
            self._messages("Rome is in Italy.  "), ["Rome is in Italy.  "]
        )
        assert "not found verbatim" in placement.problem
