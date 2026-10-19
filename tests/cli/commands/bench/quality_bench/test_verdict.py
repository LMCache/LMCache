# SPDX-License-Identifier: Apache-2.0
"""Tests for per-question rows and the verdict."""

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.dataset import Sample
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import BlendCounters
from lmcache.cli.commands.bench.quality_bench.core.config import (
    ArmRole,
    ReuseScenario,
    VerdictMetric,
)
from lmcache.cli.commands.bench.quality_bench.core.plan import build_plan
from lmcache.cli.commands.bench.quality_bench.core.runner import ArmAnswer, ArmRun
from lmcache.cli.commands.bench.quality_bench.core.verdict import (
    ENGAGEMENT_WINDOW,
    MAX_UNSCORED_SHARE,
    LossTag,
    ScoreTable,
    Verdict,
    build_rows,
    check_engagement,
    decide,
    lexical_scores,
    run_problems,
    score,
)

# Local
from .conftest import CHUNK

_N = 40


def _samples(n: int = _N) -> list[Sample]:
    return [
        Sample(
            f"s{i}",
            [f"Fact {i} is alpha{i}.", f"Other {i} is beta{i}."],
            f"What is fact {i}?",
            [f"alpha{i}"],
        )
        for i in range(n)
    ]


def _answer(sample: Sample, text: str, share: float = 0.0) -> ArmAnswer:
    return ArmAnswer(
        sample_id=sample.sample_id,
        request_id="q",
        successful=True,
        response=f"<final_answer>{text}</final_answer>" if text else "",
        answer=text,
        ttft=0.1,
        num_output_tokens=4,
        counters=BlendCounters(requested_tokens=100, non_prefix_hit_tokens=100 * share),
    )


def _failed(sample: Sample) -> ArmAnswer:
    return ArmAnswer(
        sample_id=sample.sample_id,
        request_id="q",
        successful=False,
        response="",
        answer="",
        ttft=-1.0,
        num_output_tokens=0,
    )


def _run(
    role: ArmRole,
    answers: list[ArmAnswer],
    run_index: int = 0,
    measured: bool = False,
    store_failures: int = 0,
) -> ArmRun:
    total = BlendCounters()
    for a in answers:
        total = total + a.counters
    return ArmRun(
        role=role,
        run_index=run_index,
        answers=answers,
        counters=total,
        engagement_measured=measured,
        store_failures=store_failures,
    )


def _correct(samples: list[Sample]) -> list[ArmAnswer]:
    return [_answer(s, s.answers[0], share=0.8) for s in samples]


def _with_losses(samples: list[Sample], lost: int) -> list[ArmAnswer]:
    answers = _correct(samples)
    for i in range(lost):
        answers[i] = _answer(samples[i], "not provided", share=0.8)
    return answers


@pytest.fixture
def plan(aligner):
    return build_plan(_samples(), aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)


def _decide(plan, blend_answers, measured=True, tolerance=0.05, references=2):
    samples = [s.sample for s in plan.samples]
    runs = [
        _run(ArmRole.REFERENCE, _correct(samples), run_index=k)
        for k in range(references)
    ]
    runs.append(_run(ArmRole.BLEND, blend_answers, measured=measured))
    return decide(plan, runs, VerdictMetric.MATCH, tolerance, 0.3), runs


class TestDecide:
    def test_identical_answers_pass(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        verdict, _ = _decide(plan, _correct(samples))
        assert verdict.verdict is Verdict.PASS
        assert verdict.blend.delta == 0.0
        assert verdict.reference_flip_rate == 0.0

    def test_small_loss_within_tolerance_passes(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        verdict, _ = _decide(plan, _with_losses(samples, 1))
        assert verdict.blend.delta == pytest.approx(-1 / _N)
        assert verdict.verdict is Verdict.PASS

    def test_large_significant_loss_fails(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        verdict, _ = _decide(plan, _with_losses(samples, 20))
        assert verdict.verdict is Verdict.FAIL

    def test_large_significant_gain_fails(self, plan) -> None:
        """The check is two-sided: blend far above the reference fails too."""
        samples = [s.sample for s in plan.samples]
        runs = [
            _run(ArmRole.REFERENCE, _with_losses(samples, 20), run_index=k)
            for k in range(2)
        ]
        runs.append(_run(ArmRole.BLEND, _correct(samples), measured=True))
        verdict = decide(plan, runs, VerdictMetric.MATCH, 0.05, 0.3)
        assert verdict.blend.delta == pytest.approx(20 / _N)
        assert verdict.verdict is Verdict.FAIL

    def test_loss_over_tolerance_but_noisy_is_underpowered(self, aligner) -> None:
        few = build_plan(_samples(4), aligner, ReuseScenario.INDEPENDENT, CHUNK, 1)
        samples = [s.sample for s in few.samples]
        verdict, _ = _decide(few, _with_losses(samples, 1))
        assert verdict.verdict is Verdict.UNDERPOWERED
        assert verdict.samples_needed > len(samples)

    def test_unengaged_blend_is_invalid(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        idle = [_answer(s, s.answers[0], share=0.0) for s in samples]
        verdict, _ = _decide(plan, idle)
        assert verdict.verdict is Verdict.INVALID
        assert not verdict.engagement.engaged
        assert len(verdict.invalid_reasons) == 1
        assert "blend share" in verdict.invalid_reasons[0]

    def test_valid_verdict_has_no_invalid_reasons(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        verdict, _ = _decide(plan, _correct(samples))
        assert verdict.invalid_reasons == []

    @pytest.mark.parametrize("role", [ArmRole.REFERENCE, ArmRole.BLEND])
    def test_arm_with_no_successful_request_is_invalid(self, plan, role) -> None:
        """An unreachable arm scores 0 everywhere."""
        samples = [s.sample for s in plan.samples]
        down = [_failed(s) for s in samples]
        runs = [
            _run(
                ArmRole.REFERENCE,
                down if role is ArmRole.REFERENCE else _correct(samples),
            ),
            _run(
                ArmRole.BLEND,
                down if role is ArmRole.BLEND else _correct(samples),
            ),
        ]
        verdict = decide(plan, runs, VerdictMetric.MATCH, 0.05, 0.3)
        assert verdict.verdict is Verdict.INVALID
        label = "reference_1" if role is ArmRole.REFERENCE else "blend"
        assert verdict.invalid_reasons == [f"{label}: no request succeeded"]

    def test_every_arm_down_is_invalid_not_pass(self, plan) -> None:
        """Both arms scoring 0 must not read as identical answers."""
        samples = [s.sample for s in plan.samples]
        runs = [
            _run(ArmRole.REFERENCE, [_failed(s) for s in samples]),
            _run(ArmRole.BLEND, [_failed(s) for s in samples]),
        ]
        verdict = decide(plan, runs, VerdictMetric.MATCH, 0.05, 0.3)
        assert verdict.blend.delta == 0.0
        assert verdict.verdict is Verdict.INVALID
        assert len(verdict.invalid_reasons) == 2

    def test_some_failed_requests_do_not_invalidate(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        blend = _correct(samples)
        blend[0] = _failed(samples[0])
        verdict, _ = _decide(plan, blend)
        assert verdict.verdict is Verdict.PASS
        assert verdict.invalid_reasons == []

    def test_store_failures_are_invalid(self, plan) -> None:
        """A failed store means the arm did not measure the chosen scenario."""
        samples = [s.sample for s in plan.samples]
        runs = [
            _run(ArmRole.REFERENCE, _correct(samples)),
            _run(ArmRole.BLEND, _correct(samples), store_failures=3),
        ]
        verdict = decide(plan, runs, VerdictMetric.MATCH, 0.05, 0.3)
        assert verdict.verdict is Verdict.INVALID
        assert verdict.invalid_reasons == ["blend: 3 store requests failed"]

    def test_run_problems_judge_one_run_on_its_own(self, plan) -> None:
        """What `bench quality run` reports before any other arm exists."""
        samples = [s.sample for s in plan.samples]
        assert run_problems(_run(ArmRole.BLEND, _correct(samples))) == []
        down = _run(ArmRole.REFERENCE, [_failed(s) for s in samples], store_failures=2)
        assert run_problems(down) == [
            "reference_1: no request succeeded",
            "reference_1: 2 store requests failed",
        ]

    def test_unmeasured_engagement_is_reported_not_invalid(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        idle = [_answer(s, s.answers[0], share=0.0) for s in samples]
        verdict, _ = _decide(plan, idle, measured=False)
        assert verdict.verdict is Verdict.PASS
        assert not verdict.engagement.measured

    def test_one_reference_run_has_no_flip_rate(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        verdict, _ = _decide(plan, _correct(samples), references=1)
        assert verdict.reference_flip_rate == -1.0

    def test_second_reference_is_compared_too(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        verdict, _ = _decide(plan, _correct(samples))
        assert [c.label for c in verdict.others] == ["reference_2"]

    def test_misaligned_answers_are_rejected(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        with pytest.raises(ValueError, match="do not match"):
            _decide(plan, _correct(samples)[1:])


class TestEngagement:
    def test_collapse_mid_run_is_caught_by_a_window(self, plan) -> None:
        """Blending that stops partway through the run is flagged as not engaged."""
        samples = [s.sample for s in plan.samples]
        answers = _correct(samples)
        for i in range(_N - ENGAGEMENT_WINDOW, _N):
            answers[i] = _answer(samples[i], samples[i].answers[0], share=0.0)
        check = check_engagement(_run(ArmRole.BLEND, answers, measured=True), 0.3)
        assert check.blend_share == pytest.approx(0.6)
        assert check.min_window_share == 0.0
        assert not check.engaged

    def test_segmented_prefix_tokens_are_reported(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        gapped = BlendCounters(requested_tokens=100, segmented_prefix_hit_tokens=40)
        answers = [_answer(s, s.answers[0]) for s in samples]
        run = ArmRun(
            role=ArmRole.BLEND,
            run_index=0,
            answers=answers,
            counters=gapped,
            engagement_measured=True,
            store_failures=0,
        )
        check = check_engagement(run, 0.3)
        assert check.segmented_prefix_hit_tokens == 40.0
        assert check.blend_share == 0.0

    def test_low_share_from_l2_gaps_points_at_storage(self, plan) -> None:
        """An INVALID from L2 load failures should not be read as blending off."""
        samples = [s.sample for s in plan.samples]
        gapped = BlendCounters(requested_tokens=100, segmented_prefix_hit_tokens=80)
        answers = [_answer(s, s.answers[0]) for s in samples]
        runs = [
            _run(ArmRole.REFERENCE, _correct(samples)),
            ArmRun(
                role=ArmRole.BLEND,
                run_index=0,
                answers=answers,
                counters=gapped,
                engagement_measured=True,
                store_failures=0,
            ),
        ]
        verdict = decide(plan, runs, VerdictMetric.MATCH, 0.05, 0.3)
        assert verdict.verdict is Verdict.INVALID
        [reason] = verdict.invalid_reasons
        assert "80 tokens loaded past L2 prefix gaps" in reason

    def test_plain_low_share_has_no_storage_hint(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        idle = [_answer(s, s.answers[0], share=0.0) for s in samples]
        verdict, _ = _decide(plan, idle)
        assert "L2" not in verdict.invalid_reasons[0]


class TestBuildRows:
    def test_rows_tag_losses(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        blend = _correct(samples)
        blend[0] = _answer(samples[0], "not provided", 0.8)
        blend[1] = _answer(samples[1], "", 0.8)
        blend[2] = _answer(samples[2], "beta2", 0.8)
        blend[3] = _answer(samples[3], "gamma", 0.8)
        _, runs = _decide(plan, blend)
        rows = build_rows(plan, runs, VerdictMetric.MATCH)
        assert [r["loss_tag"] for r in rows[:5]] == [
            LossTag.ABSTENTION.value,
            LossTag.EMPTY.value,
            LossTag.WRONG_VALUE.value,
            LossTag.WORDING.value,
            LossTag.NONE.value,
        ]

    def test_blend_higher_is_its_own_tag(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        references = [
            _run(ArmRole.REFERENCE, _with_losses(samples, 1), run_index=0),
        ]
        runs = [*references, _run(ArmRole.BLEND, _correct(samples), measured=True)]
        rows = build_rows(plan, runs, VerdictMetric.MATCH)
        assert rows[0]["loss_tag"] == LossTag.BLEND_HIGHER.value

    def test_row_schema(self, plan) -> None:
        samples = [s.sample for s in plan.samples]
        _, runs = _decide(plan, _correct(samples))
        row = build_rows(plan, runs, VerdictMetric.MATCH)[0]
        assert row["qid"] == "s0"
        scores, evidence = row["scores"], row["evidence"]
        assert isinstance(scores, dict) and isinstance(evidence, list)
        assert set(scores) == {"reference_1", "reference_2", "blend"}
        assert row["reference_flipped"] is False
        assert row["blend_share"] == pytest.approx(0.8)
        assert evidence[0]["origin"] == "alone"


class TestPrecomputedScores:
    """Scores supplied by the caller like the LLM judge."""

    def _runs(self, plan):
        samples = [s.sample for s in plan.samples]
        return [
            _run(ArmRole.REFERENCE, _correct(samples)),
            _run(ArmRole.BLEND, _correct(samples), measured=True),
        ]

    def test_judge_scores_drive_the_verdict(self, plan) -> None:
        runs = self._runs(plan)
        scores: ScoreTable = {
            runs[0].label: [1.0] * _N,
            runs[1].label: [0.5] * _N,
        }
        verdict = decide(plan, runs, VerdictMetric.JUDGE, 0.05, 0.3, scores=scores)
        assert verdict.verdict is Verdict.FAIL
        assert verdict.blend.delta == pytest.approx(-0.5)

    def test_unscored_questions_are_left_out(self, plan) -> None:
        runs = self._runs(plan)
        reference: list[float | None] = [1.0] * _N
        blend: list[float | None] = [1.0] * _N
        blend[0] = None
        blend[1] = None
        scores: ScoreTable = {runs[0].label: list(reference), runs[1].label: blend}
        verdict = decide(plan, runs, VerdictMetric.JUDGE, 0.05, 0.3, scores=scores)
        assert verdict.unscored == 2
        # 2 of 40 is exactly MAX_UNSCORED_SHARE: still a valid verdict.
        assert verdict.verdict is Verdict.PASS
        assert verdict.invalid_reasons == []

    def test_too_many_unscored_questions_are_invalid(self, plan) -> None:
        runs = self._runs(plan)
        blend: list[float | None] = [1.0] * _N
        blend[:3] = [None, None, None]
        scores: ScoreTable = {runs[0].label: [1.0] * _N, runs[1].label: blend}
        verdict = decide(plan, runs, VerdictMetric.JUDGE, 0.05, 0.3, scores=scores)
        assert 3 > MAX_UNSCORED_SHARE * _N
        assert verdict.verdict is Verdict.INVALID
        assert verdict.invalid_reasons == [
            f"3 of {_N} questions unscored (more than 5%)"
        ]

    def test_nothing_scored_is_an_error(self, plan) -> None:
        runs = self._runs(plan)
        scores: ScoreTable = {runs[0].label: [None] * _N, runs[1].label: [1.0] * _N}
        with pytest.raises(ValueError, match="no question"):
            decide(plan, runs, VerdictMetric.JUDGE, 0.05, 0.3, scores=scores)

    def test_judge_metric_needs_precomputed_scores(self, plan) -> None:
        runs = self._runs(plan)
        with pytest.raises(ValueError):
            decide(plan, runs, VerdictMetric.JUDGE, 0.05, 0.3)
        with pytest.raises(ValueError):
            score(runs[0].answers[0], ["x"], VerdictMetric.JUDGE)

    def test_judge_rows_carry_match_as_a_cross_check(self, plan) -> None:
        runs = self._runs(plan)
        scores: ScoreTable = {runs[0].label: [0.75] * _N, runs[1].label: [None] * _N}
        rows = build_rows(plan, runs, VerdictMetric.JUDGE, scores=scores)
        assert rows[0]["scores"] == {runs[0].label: 0.75, runs[1].label: None}
        assert rows[0]["match"] == {runs[0].label: 1.0, runs[1].label: 1.0}
        assert rows[0]["loss_tag"] is None

    def test_lexical_scores_match_score(self, plan) -> None:
        runs = self._runs(plan)
        table = lexical_scores(plan, runs, VerdictMetric.MATCH)
        assert table[runs[0].label] == [1.0] * _N
