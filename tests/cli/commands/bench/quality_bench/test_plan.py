# SPDX-License-Identifier: Apache-2.0
"""Tests for ``bench quality`` prompt plans and reuse manifests."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality import alignment
from lmcache.cli.commands.bench.engine_bench.quality.alignment import (
    ChunkAligner,
    Filler,
)
from lmcache.cli.commands.bench.engine_bench.quality.dataset import Sample
from lmcache.cli.commands.bench.engine_bench.quality.prompts import (
    ANSWER_TAG_PROMPT,
)
from lmcache.cli.commands.bench.quality_bench.core.config import ReuseScenario
from lmcache.cli.commands.bench.quality_bench.core.plan import (
    EvidenceSource,
    PlanAlignmentError,
    QualityPlan,
    StoreOrigin,
    build_plan,
)

# Local
from ..engine_bench.fake_tokenizer import make_fake_tokenizer
from .conftest import CHUNK
from .fakes import samples


def _plan(aligner, scenario: ReuseScenario, seed: int = 11):
    return build_plan(samples(), aligner, scenario, CHUNK, seed)


class TestIndependentScenario:
    def test_each_distinct_passage_is_stored_once(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        # Seven passages across three questions, one shared.
        assert len(plan.store_requests) == 6

    def test_store_request_holds_one_aligned_passage(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        content = plan.store_requests[0].messages[0]["content"]
        assert "Paris is in France." in content
        assert "Rome is in Italy." not in content

    def test_measured_prompt_keeps_dataset_order(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        content = plan.samples[0].messages[0]["content"]
        assert content.index("Paris") < content.index("Rome")
        assert content.rstrip().endswith("Answer:")

    def test_shared_passage_points_at_one_store_request(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        rome_first = plan.samples[0].passages[1]
        rome_second = plan.samples[1].passages[0]
        assert rome_first.passage_id == rome_second.passage_id
        assert rome_first.store_request_id == rome_second.store_request_id
        assert rome_first.origin is StoreOrigin.ALONE


class TestReorderedScenario:
    def test_each_question_stores_its_passages_together(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.REORDERED)
        assert len(plan.store_requests) == 3
        content = plan.store_requests[2].messages[0]["content"]
        assert content.index("Bern") < content.index("Vienna") < content.index("Prague")

    def test_every_multi_passage_question_is_reordered(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.REORDERED)
        for sample_plan in plan.samples:
            stored = [p.stored_position for p in sample_plan.passages]
            assert stored != sorted(stored)
            assert sorted(stored) == list(range(len(stored)))

    def test_order_is_deterministic_per_seed(self, aligner) -> None:
        first = _plan(aligner, ReuseScenario.REORDERED, seed=5)
        again = _plan(aligner, ReuseScenario.REORDERED, seed=5)
        assert first.prompts_digest == again.prompts_digest

    def test_provenance_records_both_positions(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.REORDERED)
        for passage in plan.samples[2].passages:
            assert passage.origin is StoreOrigin.WITH_SAMPLE_PASSAGES
            assert passage.store_request_id == plan.store_requests[2].request_id
        assert [p.measured_position for p in plan.samples[2].passages] == [0, 1, 2]


class TestEvidenceAndManifest:
    def test_dataset_marked_evidence_is_used(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.REORDERED)
        swiss = plan.samples[2]
        assert swiss.evidence_source is EvidenceSource.DATASET
        [evidence] = [p for p in swiss.passages if p.is_evidence]
        assert evidence.stored_position == 0

    def test_unmarked_evidence_falls_back_to_the_answer_string(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        capital = plan.samples[0]
        assert capital.evidence_source is EvidenceSource.ANSWER_STRING
        assert [p.is_evidence for p in capital.passages] == [True, False]

    def test_manifest_entry_schema(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        entry = plan.samples[0].manifest_entry(plan.scenario, plan.chunk_size)
        assert entry["qid"] == "s0"
        assert entry["scenario"] == "independent"
        assert entry["chunk_size"] == CHUNK
        assert entry["prompt_hash"] == plan.samples[0].prompt_hash
        assert entry["evidence_passage_ids"] == [plan.samples[0].passages[0].passage_id]
        assert {"origin", "store_request_id", "is_evidence"} <= set(
            entry["passages"][0]  # type: ignore[index]
        )

    def test_scenarios_build_different_prompts(self, aligner) -> None:
        independent = _plan(aligner, ReuseScenario.INDEPENDENT)
        reordered = _plan(aligner, ReuseScenario.REORDERED)
        assert independent.prompts_digest != reordered.prompts_digest

    def test_empty_samples_rejected(self, aligner) -> None:
        with pytest.raises(ValueError, match="samples"):
            build_plan([], aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)


class TestMeasuredEarlier:
    """Passages an earlier measured question also held, which the candidate arm
    might reuse from that question's cached prompt."""

    @pytest.mark.parametrize("scenario", list(ReuseScenario))
    def test_recurring_passage_names_the_earlier_question(
        self, aligner, scenario
    ) -> None:
        plan = _plan(aligner, scenario)
        # Passages are in measured order, which reordered permutes.
        first = {p.passage_id for p in plan.samples[0].passages}
        [rome_again] = [p for p in plan.samples[1].passages if p.passage_id in first]
        assert rome_again.measured_earlier_in == ("q0",)
        assert all(not p.measured_earlier_in for p in plan.samples[0].passages)

    def test_share_counts_every_measured_passage(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        # One recurrence (Rome) among seven measured passages.
        assert plan.measured_earlier_share() == pytest.approx(1 / 7)

    def test_every_earlier_question_is_listed_in_order(self, aligner) -> None:
        shared = "Shared passage."
        trio = [
            Sample(f"s{i}", [shared, f"Own {i}."], f"Q{i}?", ["x"]) for i in range(3)
        ]
        plan = build_plan(trio, aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)
        assert plan.samples[2].passages[0].measured_earlier_in == ("q0", "q1")

    def test_repeat_within_one_question_is_not_an_earlier_question(
        self, aligner
    ) -> None:
        doubled = [
            Sample("a", ["Same.", "Same."], "Q?", ["x"]),
            Sample("b", ["Same."], "Q?", ["x"]),
        ]
        plan = build_plan(doubled, aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)
        assert [p.measured_earlier_in for p in plan.samples[0].passages] == [(), ()]
        assert plan.samples[1].passages[0].measured_earlier_in == ("q0",)

    def test_no_recurrence_reads_zero(self, aligner) -> None:
        distinct = [Sample(f"s{i}", [f"Only {i}."], "Q?", ["x"]) for i in range(3)]
        plan = build_plan(distinct, aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)
        assert plan.measured_earlier_share() == 0.0

    def test_manifest_records_it(self, aligner) -> None:
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        entry = plan.samples[1].manifest_entry(plan.scenario, plan.chunk_size)
        passages = entry["passages"]
        assert isinstance(passages, list)
        assert [p["measured_earlier_in"] for p in passages] == [["q0"], []]


def _filler_aligner(filler: Filler) -> ChunkAligner:
    return ChunkAligner(
        make_fake_tokenizer(bpe_pieces=True), "fake-model", CHUNK, 3, filler=filler
    )


class TestFillers:
    @pytest.fixture(autouse=True)
    def small_word_pool(self, monkeypatch) -> None:
        """The fake tokenizer's vocabulary is smaller than the default pool."""
        monkeypatch.setattr(alignment, "_FILLER_VOCAB_SIZE", 200)

    @pytest.mark.parametrize("filler", list(Filler))
    def test_every_filler_asks_for_answer_tags(self, filler) -> None:
        """The prompt does not change with --filler; only the padding does."""
        plan = _plan(_filler_aligner(filler), ReuseScenario.INDEPENDENT)
        prompt = plan.samples[0].prompt_text
        assert prompt.startswith(ANSWER_TAG_PROMPT.system)
        assert prompt.endswith(
            ANSWER_TAG_PROMPT.question_template.format(question="Capital?")
        )
        assert "final_answer" not in prompt

    def test_framed_fillers_label_each_passage(self) -> None:
        plan = _plan(_filler_aligner(Filler.DOTS), ReuseScenario.INDEPENDENT)
        prompt = plan.samples[0].prompt_text
        assert prompt.count("Passage ") == len(plan.samples[0].passages)

    @pytest.mark.parametrize("filler", list(Filler))
    @pytest.mark.parametrize(
        "scenario", [ReuseScenario.INDEPENDENT, ReuseScenario.REORDERED]
    )
    def test_every_filler_passes_the_placement_check(self, filler, scenario) -> None:
        plan = _plan(_filler_aligner(filler), scenario)
        assert plan.padding is not None
        assert plan.padding.filler == filler.value
        assert 0 < plan.padding.share < 1

    def test_manifest_records_padding_per_passage(self) -> None:
        aligner = _filler_aligner(Filler.DOTS)
        plan = _plan(aligner, ReuseScenario.INDEPENDENT)
        entry = plan.samples[0].manifest_entry(plan.scenario, plan.chunk_size)
        passages = entry["passages"]
        assert passages[0]["pad_tokens"] == aligner.pad_tokens(  # type: ignore[index]
            "Paris is in France."
        )

    def test_padding_round_trips(self) -> None:
        plan = _plan(_filler_aligner(Filler.DASHES), ReuseScenario.REORDERED)
        assert QualityPlan.from_record(plan.to_record()) == plan

    def test_plans_without_padding_still_load(self, aligner) -> None:
        """Plans saved before padding was recorded."""
        record = _plan(aligner, ReuseScenario.INDEPENDENT).to_record()
        del record["padding"]
        for sample in record["samples"]:  # type: ignore[union-attr]
            for passage in sample["passages"]:
                del passage["pad_tokens"]
        plan = QualityPlan.from_record(record)
        assert plan.padding is None
        assert plan.samples[0].passages[0].pad_tokens == 0

    def test_a_misplaced_passage_refuses_the_plan(self, monkeypatch) -> None:
        """A chat prefix one token off shifts every passage off the boundary."""
        aligner = _filler_aligner(Filler.DOTS)
        real = aligner._chat_prefix_tokens
        monkeypatch.setattr(aligner, "_chat_prefix_tokens", lambda: real() + 1)
        with pytest.raises(PlanAlignmentError, match="chunk boundary") as exc:
            _plan(aligner, ReuseScenario.INDEPENDENT)
        assert exc.value.problems
        assert "--filler dots" in str(exc.value)


def _contexts(count: int) -> list[Sample]:
    """*count* questions with distinct three-passage contexts."""
    return [
        Sample(
            f"c{i}",
            [f"City {i} lies on river {i}.", f"River {i} is long.", f"Mayor {i}."],
            f"Where is city {i}?",
            [f"river {i}"],
        )
        for i in range(count)
    ]


def _context_text(aligner: ChunkAligner, sample: Sample) -> str:
    return "".join(aligner.framed(p) for p in sample.documents)


class TestConcatenated:
    @pytest.fixture
    def framer(self) -> ChunkAligner:
        return _filler_aligner(Filler.DOTS)

    @staticmethod
    def _build(framer, samples, unit_samples=None, seed=5, min_unit_tokens=200):
        return build_plan(
            samples,
            framer,
            ReuseScenario.CONCATENATED,
            CHUNK,
            seed,
            unit_samples=unit_samples,
            min_unit_tokens=min_unit_tokens,
        )

    @staticmethod
    def _unit_text(plan, sample_plan) -> str:
        unit = sample_plan.passages[0].store_request_id
        [request] = [r for r in plan.store_requests if r.request_id == unit]
        return request.messages[0]["content"]

    def test_units_reach_the_minimum_size(self, framer) -> None:
        plan = self._build(framer, _contexts(8))
        sizes = [
            framer.token_length(r.messages[0]["content"]) for r in plan.store_requests
        ]
        assert len(sizes) > 1
        assert all(size >= 200 for size in sizes[:-1])

    def test_each_context_is_stored_whole_without_the_system_prompt(
        self, framer
    ) -> None:
        samples = _contexts(6)
        plan = self._build(framer, samples)
        for request in plan.store_requests:
            content = request.messages[0]["content"]
            assert content.startswith("Passage ")
            assert ANSWER_TAG_PROMPT.system not in content
        for sample, sample_plan in zip(samples, plan.samples, strict=True):
            assert _context_text(framer, sample) in self._unit_text(plan, sample_plan)

    def test_measured_prompt_is_unpadded(self, framer) -> None:
        sample = _contexts(1)[0]
        plan = self._build(framer, [sample])
        assert plan.samples[0].prompt_text == (
            ANSWER_TAG_PROMPT.system
            + _context_text(framer, sample)
            + ANSWER_TAG_PROMPT.question_template.format(question=sample.question)
        )
        assert plan.padding is not None and plan.padding.pad_tokens == 0

    def test_units_do_not_depend_on_how_many_questions_are_measured(
        self, framer
    ) -> None:
        everything = _contexts(8)
        full = self._build(framer, everything)
        first_two = self._build(framer, everything[:2], unit_samples=everything)
        for index in range(2):
            assert self._unit_text(first_two, first_two.samples[index]) == (
                self._unit_text(full, full.samples[index])
            )
            assert (
                first_two.samples[index].unit_position
                == full.samples[index].unit_position
            )
        assert len(first_two.store_requests) <= 2

    def test_the_shuffle_is_seeded(self, framer) -> None:
        samples = _contexts(8)
        once = self._build(framer, samples, seed=5)
        again = self._build(framer, samples, seed=5)
        other = self._build(framer, samples, seed=6)
        assert once.prompts_digest == again.prompts_digest
        assert once.prompts_digest != other.prompts_digest

    def test_a_duplicate_context_is_stored_once(self, framer) -> None:
        samples = _contexts(4)
        twin = Sample("twin", samples[1].documents, "Same context?", ["x"])
        plan = self._build(framer, [*samples, twin])
        first, copy = plan.samples[1], plan.samples[4]
        # Whichever the shuffle placed first owns the context; the other
        # names it.
        pairs = {(first.duplicate_of, copy.duplicate_of)}
        assert pairs <= {(None, "c1"), ("twin", None)}
        assert self._unit_text(plan, first) == self._unit_text(plan, copy)
        stored = "".join(r.messages[0]["content"] for r in plan.store_requests)
        assert stored.count(_context_text(framer, samples[1])) == 1

    def test_one_context_per_unit_at_zero(self, framer) -> None:
        plan = self._build(framer, _contexts(4), min_unit_tokens=0)
        assert len(plan.store_requests) == 4
        assert [s.unit_position for s in plan.samples] == [0, 0, 0, 0]

    def test_reuse_is_expected_inside_each_context(self, framer) -> None:
        plan = self._build(framer, _contexts(6))
        for sample_plan in plan.samples:
            expected = sample_plan.expected_reused_tokens
            assert expected is not None and expected >= CHUNK
        assert plan.expected_reused_tokens() == sum(
            s.expected_reused_tokens or 0 for s in plan.samples
        )

    def test_manifest_records_the_unit(self, framer) -> None:
        plan = self._build(framer, _contexts(6))
        entry = plan.samples[3].manifest_entry(plan.scenario, plan.chunk_size)
        assert entry["unit_position"] == plan.samples[3].unit_position
        assert entry["duplicate_of"] is None
        passages = entry["passages"]
        assert isinstance(passages, list)
        assert {p["origin"] for p in passages} == {"in_unit"}
        offset = passages[0]["stored_position"]
        assert [p["stored_position"] for p in passages] == [
            offset + i for i in range(len(passages))
        ]

    def test_round_trips(self, framer) -> None:
        plan = self._build(framer, _contexts(5))
        assert QualityPlan.from_record(plan.to_record()) == plan

    def test_measured_samples_must_lead_the_unit_samples(self, framer) -> None:
        samples = _contexts(3)
        with pytest.raises(ValueError, match="prefix"):
            self._build(framer, samples[1:], unit_samples=samples)


class TestExpectedReuse:
    def test_independent_expects_the_padded_blocks(self) -> None:
        plan = _plan(_filler_aligner(Filler.DOTS), ReuseScenario.INDEPENDENT)
        for sample_plan in plan.samples:
            assert sample_plan.expected_reused_tokens
            assert sample_plan.expected_reused_tokens % CHUNK == 0

    def test_older_plans_have_no_expectation(self, aligner) -> None:
        record = _plan(aligner, ReuseScenario.INDEPENDENT).to_record()
        for sample in record["samples"]:  # type: ignore[union-attr]
            del sample["expected_reused_tokens"]
        assert QualityPlan.from_record(record).expected_reused_tokens() is None
