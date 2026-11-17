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
