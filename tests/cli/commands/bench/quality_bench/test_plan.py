# SPDX-License-Identifier: Apache-2.0
"""Tests for ``bench quality`` prompt plans and reuse manifests."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.dataset import Sample
from lmcache.cli.commands.bench.quality_bench.core.config import ReuseScenario
from lmcache.cli.commands.bench.quality_bench.core.plan import (
    EvidenceSource,
    StoreOrigin,
    build_plan,
)

# Local
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
