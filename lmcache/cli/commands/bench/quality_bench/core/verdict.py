# SPDX-License-Identifier: Apache-2.0
"""Per-question rows and the verdict for ``lmcache bench quality``.

The verdict acts as a gatekeeper for quality checking. That is, a gap passes
on tolerance, fails only when it also clears its own error bar, and is
otherwise UNDERPOWERED with the sample count that would settle it. Additionally,
a run that is unfit to compare (e.g. an unreachable arm, or a candidate
whose captured blend metrics show it did not blend) is INVALID. The
reference is run twice so its flip rate is reported as the noise floor.
"""

# Standard
from dataclasses import dataclass, field
import enum
import math
import statistics

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.scoring import (
    answer_match,
    best_f1,
    is_abstention,
    normalize_answer,
)
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import BlendCounters
from lmcache.cli.commands.bench.quality_bench.core.config import ArmRole, VerdictMetric
from lmcache.cli.commands.bench.quality_bench.core.plan import QualityPlan, SamplePlan
from lmcache.cli.commands.bench.quality_bench.core.runner import ArmAnswer, ArmRun

# Per run label, one score per question in plan order. ``None`` marks a
# question the scorer could not score (an LLM judge that gave no rating).
ScoreTable = dict[str, list[float | None]]

# Questions per engagement window. Engagement that collapses mid-run (e.g.
# the fingerprint index filling up) shows in a window long before it moves
# the run-wide share.
ENGAGEMENT_WINDOW = 10

# Largest share of questions that may go unscored before the verdict is
# INVALID. A healthy judge leaves almost nothing unrated after its retries,
# so more than this means the judge is malfunctioning. Unscored questions
# are dropped from every arm alike, which costs only power while the gaps are
# random. Past this share, they are likely to track the answers (e.g. long
# candidate answers failing to parse), which would bias the comparison.
MAX_UNSCORED_SHARE = 0.05


class Verdict(enum.Enum):
    """Outcome of comparing the candidate arm with the reference."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNDERPOWERED = "UNDERPOWERED"
    INVALID = "INVALID"


class LossTag(enum.Enum):
    """How the candidate arm's answer differs from the reference's."""

    NONE = "none"
    """Same score."""

    CANDIDATE_HIGHER = "candidate_higher"
    """The candidate arm scored higher."""

    ABSTENTION = "abstention"
    """The candidate arm declined to answer ("the passages do not contain X")."""

    EMPTY = "empty"
    """No complete answer: failed request, or reasoning ran out of tokens."""

    WRONG_VALUE = "wrong_value"
    """A wrong answer that appears elsewhere in the prompt, e.g. the
    neighboring line of a table."""

    WORDING = "wording"
    """Any other lower-scoring answer."""


@dataclass(frozen=True)
class EngagementCheck:
    """Whether the blend server actually blended during the measured pass.

    Attributes:
        measured: Whether counters were read at all.
        blend_share: Share of looked-up tokens served by blending, run-wide.
        min_window_share: Lowest share over consecutive windows of
            :data:`ENGAGEMENT_WINDOW` questions. Equals ``blend_share`` when
            the run is shorter than one window.
        stale_chunks: Chunks that matched but were no longer stored.
        segmented_prefix_hit_tokens: Prefix tokens loaded after an L2 load
            gap (see :class:`BlendCounters`) -- not blending, so outside
            ``blend_share``.
        fingerprints_registered: Chunks indexed during the pass.
        engaged: Whether both shares reach the minimum. ``True`` when not
            measured, so an unmeasured run is reported (not invalidated).
    """

    measured: bool
    blend_share: float
    min_window_share: float
    stale_chunks: float
    segmented_prefix_hit_tokens: float
    fingerprints_registered: float
    engaged: bool


@dataclass(frozen=True)
class ArmComparison:
    """One arm's score against the first reference run.

    Attributes:
        label: The arm run's label.
        mean_score: Mean per-question score.
        delta: ``mean_score`` minus the reference's, paired by question.
        standard_error: Standard error of the paired per-question difference.
            If fewer than two questions, then ``inf``.
    """

    label: str
    mean_score: float
    delta: float
    standard_error: float


@dataclass(frozen=True)
class QualityVerdict:
    """The run's conclusion.

    Attributes:
        verdict: PASS, FAIL, UNDERPOWERED or INVALID.
        metric: The score compared.
        tolerance: Acceptable score difference, either direction, in score
            units.
        candidate: The candidate arm against the reference.
        others: Every other arm run (later reference runs, control) against the
            first reference run.
        reference_flip_rate: Share of questions whose score changed between
            the first two reference runs. ``-1.0`` with one reference run.
        samples_needed: For UNDERPOWERED, questions needed for the observed
            gap to clear two standard errors. Otherwise, ``0``.
        engagement: The candidate arm's engagement check.
        unscored: Questions left out of every comparison because some run
            had no score for them.
        invalid_reasons: Why the verdict is INVALID. Otherwise, empty.
    """

    verdict: Verdict
    metric: VerdictMetric
    tolerance: float
    candidate: ArmComparison
    others: list[ArmComparison]
    reference_flip_rate: float
    samples_needed: int
    engagement: EngagementCheck
    unscored: int = 0
    invalid_reasons: list[str] = field(default_factory=list)


def score(answer: ArmAnswer, references: list[str], metric: VerdictMetric) -> float:
    """Score one answer. An unparsed answer scores ``0.0``.

    Args:
        answer: The arm's outcome for the question.
        references: Gold answers.
        metric: Which score to compute.

    Returns:
        ``1.0``/``0.0`` for ``MATCH``, the best token F1 for ``F1``.

    Raises:
        ValueError: For ``JUDGE``, which needs the judge (see
            ``judge.AnswerAccuracyJudge.score_runs``).
    """
    if metric is VerdictMetric.JUDGE:
        raise ValueError("judge scores come from the judge, not score()")
    if not answer.parsed:
        return 0.0
    if metric is VerdictMetric.MATCH:
        return float(answer_match(answer.answer, references))
    return best_f1(answer.answer, references)


def lexical_scores(
    plan: QualityPlan, runs: list[ArmRun], metric: VerdictMetric
) -> ScoreTable:
    """Score every run with a judge-free metric.

    Args:
        plan: The plan the runs measured.
        runs: The arm runs, each answering every plan sample in order.
        metric: ``MATCH`` or ``F1``.

    Returns:
        Per run label, one score per question.

    Raises:
        ValueError: For ``JUDGE``.
    """
    golds = [s.sample.answers for s in plan.samples]
    table: ScoreTable = {}
    for run in runs:
        table[run.label] = [
            score(a, g, metric) for a, g in zip(run.answers, golds, strict=True)
        ]
    return table


def loss_tag(
    candidate: ArmAnswer,
    candidate_score: float,
    reference_score: float,
    sample_plan: SamplePlan,
) -> LossTag:
    """Classify how *candidate* differs from the reference on one question.

    Args:
        candidate: The candidate arm's outcome.
        candidate_score: Its score.
        reference_score: The first reference run's score.
        sample_plan: The question's plan, for locating the answer in the
            prompt.

    Returns:
        The tag.
    """
    if candidate_score > reference_score:
        return LossTag.CANDIDATE_HIGHER
    if candidate_score == reference_score:
        return LossTag.NONE
    if not candidate.parsed:
        return LossTag.EMPTY
    if is_abstention(candidate.answer):
        return LossTag.ABSTENTION
    found = normalize_answer(candidate.answer)
    if found and f" {found} " in f" {normalize_answer(sample_plan.prompt_text)} ":
        return LossTag.WRONG_VALUE
    return LossTag.WORDING


def run_problems(run: ArmRun) -> list[str]:
    """List what makes one arm run unfit to compare, on its own.

    Each problem makes the verdict INVALID, whatever the other runs show,
    so ``bench quality run`` can report it as soon as the pass finishes.

    Args:
        run: One arm run.

    Returns:
        One reason per problem: no successful request (an unreachable
        engine scores like a wrong one), or failed store requests (the arm
        did not cache what the scenario says). Empty when the run is fit.
    """
    problems: list[str] = []
    if not any(answer.successful for answer in run.answers):
        problems.append(f"{run.label}: no request succeeded")
    if run.store_failures:
        problems.append(f"{run.label}: {run.store_failures} store requests failed")
    return problems


def check_engagement(candidate: ArmRun, min_blend_share: float) -> EngagementCheck:
    """Decide whether a candidate blended enough for its scores to count.

    Uses the blend metrics captured during the candidate pass; a candidate
    run without them is reported as unmeasured, never invalidated.

    Args:
        candidate: The candidate arm's run.
        min_blend_share: Required share, run-wide and per window.

    Returns:
        The check.
    """
    if not candidate.engagement_measured:
        return EngagementCheck(
            measured=False,
            blend_share=0.0,
            min_window_share=0.0,
            stale_chunks=0.0,
            segmented_prefix_hit_tokens=0.0,
            fingerprints_registered=0.0,
            engaged=True,
        )
    share = candidate.counters.blend_share
    windows: list[float] = []
    for start in range(
        0, len(candidate.answers) - ENGAGEMENT_WINDOW + 1, ENGAGEMENT_WINDOW
    ):
        total = BlendCounters()
        for answer in candidate.answers[start : start + ENGAGEMENT_WINDOW]:
            total = total + answer.counters
        windows.append(total.blend_share)
    min_window = min(windows) if windows else share
    return EngagementCheck(
        measured=True,
        blend_share=share,
        min_window_share=min_window,
        stale_chunks=candidate.counters.stale_chunks,
        segmented_prefix_hit_tokens=candidate.counters.segmented_prefix_hit_tokens,
        fingerprints_registered=candidate.counters.fingerprints_registered,
        engaged=share >= min_blend_share and min_window >= min_blend_share,
    )


def decide(
    plan: QualityPlan,
    runs: list[ArmRun],
    metric: VerdictMetric,
    tolerance: float,
    min_blend_share: float,
    scores: ScoreTable | None = None,
) -> QualityVerdict:
    """Compare every arm with the first reference run and reach a verdict.

    Args:
        plan: The plan every run measured.
        runs: Every arm run. Must include at least one reference run and
            exactly one candidate run, each answering every plan sample in plan
            order.
        metric: Score to compare.
        tolerance: Acceptable candidate score difference, either direction, in
            score units.
        min_blend_share: Engagement minimum for a valid verdict.
        scores: Precomputed scores per run label (required for ``JUDGE``).
            Computed from *metric* when omitted. Questions any run has no
            score for are left out of every comparison.

    Returns:
        The verdict. It is INVALID, whatever the scores, when the candidate arm
        was measured (by captured blend metrics) not to blend enough, when an
        arm run had no successful
        request (an unreachable engine scores like a wrong one), when a
        store request failed (the arm did not measure the chosen scenario),
        or when more than :data:`MAX_UNSCORED_SHARE` of the questions are
        unscored.

    Raises:
        ValueError: If *runs* lacks a reference or candidate run, a run's
            answers do not line up with the plan, *scores* lacks a run, or
            no question is scored by every run.
    """
    references = [r for r in runs if r.role is ArmRole.REFERENCE]
    candidates = [r for r in runs if r.role is ArmRole.CANDIDATE]
    if not references or len(candidates) != 1:
        raise ValueError("runs need at least one reference run and one candidate run")
    for run in runs:
        if [a.sample_id for a in run.answers] != [
            s.sample.sample_id for s in plan.samples
        ]:
            raise ValueError(f"{run.label} answers do not match the plan samples")

    table = scores if scores is not None else lexical_scores(plan, runs, metric)
    complete = _complete_scores(table, runs)
    unscored = len(plan.samples) - len(complete[runs[0].label])
    reference_scores = complete[references[0].label]
    candidate = candidates[0]
    candidate_comparison = _compare(
        candidate.label, complete[candidate.label], reference_scores
    )
    others = [
        _compare(run.label, complete[run.label], reference_scores)
        for run in runs
        if run is not references[0] and run is not candidate
    ]

    flip_rate = -1.0
    if len(references) > 1:
        second = complete[references[1].label]
        flips = sum(1 for a, b in zip(reference_scores, second, strict=True) if a != b)
        flip_rate = flips / len(reference_scores)

    engagement = check_engagement(candidate, min_blend_share)
    invalid_reasons = _invalid_reasons(
        runs, engagement, min_blend_share, unscored, len(plan.samples)
    )
    # Two-sided: reusing KV should not improve answers, so a large gain flags
    # the setup like a large loss does.
    gap = abs(candidate_comparison.delta)
    significant = gap > 2 * candidate_comparison.standard_error
    samples_needed = 0
    if invalid_reasons:
        verdict = Verdict.INVALID
    elif gap <= tolerance:
        verdict = Verdict.PASS
    elif significant:
        verdict = Verdict.FAIL
    else:
        verdict = Verdict.UNDERPOWERED
        differences = [
            b - r
            for b, r in zip(complete[candidate.label], reference_scores, strict=True)
        ]
        spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
        samples_needed = math.ceil((2 * spread / gap) ** 2) if spread > 0 else 0

    return QualityVerdict(
        verdict=verdict,
        metric=metric,
        tolerance=tolerance,
        candidate=candidate_comparison,
        others=others,
        reference_flip_rate=flip_rate,
        samples_needed=samples_needed,
        engagement=engagement,
        unscored=unscored,
        invalid_reasons=invalid_reasons,
    )


def build_rows(
    plan: QualityPlan,
    runs: list[ArmRun],
    metric: VerdictMetric,
    scores: ScoreTable | None = None,
) -> list[dict[str, object]]:
    """Return one row per question, joining every run with the manifest.

    The rows are the hand-off to whoever investigates a loss. That is,
    each names the lost question, every arm's answer and score, how the
    candidate answer failed, and which passages held the answer and how they
    were cached.

    Args:
        plan: The plan every run measured.
        runs: Every arm run, as passed to :func:`decide`.
        metric: Score reported per arm.
        scores: Precomputed scores per run label, as passed to
            :func:`decide`. For ``JUDGE`` the rows also carry the ``match``
            score, as a cross-check of the judge.

    Returns:
        Rows with keys ``qid``, ``gold``, ``scores`` (label -> score, or
        ``None`` where unscored), ``match`` (label -> 0/1, only for
        ``JUDGE``),
        ``answers`` (label -> extracted answer), ``loss_tag`` (candidate versus
        the first reference run), ``reference_flipped`` (bool, or ``None``
        with one reference run), ``blend_share`` (this question's share, or
        ``None`` when unmeasured) and ``evidence`` (the manifest's evidence
        passages with their origin and positions).

    Raises:
        ValueError: If *runs* lacks a reference or candidate run.
    """
    references = [r for r in runs if r.role is ArmRole.REFERENCE]
    candidates = [r for r in runs if r.role is ArmRole.CANDIDATE]
    if not references or len(candidates) != 1:
        raise ValueError("runs need at least one reference run and one candidate run")
    candidate = candidates[0]
    table = scores if scores is not None else lexical_scores(plan, runs, metric)
    matches = (
        lexical_scores(plan, runs, VerdictMetric.MATCH)
        if metric is VerdictMetric.JUDGE
        else None
    )

    rows: list[dict[str, object]] = []
    for index, sample_plan in enumerate(plan.samples):
        golds = sample_plan.sample.answers
        row_scores = {run.label: table[run.label][index] for run in runs}
        reference_score = row_scores[references[0].label]
        candidate_score = row_scores[candidate.label]
        flipped: bool | None = None
        if len(references) > 1 and reference_score is not None:
            second = row_scores[references[1].label]
            flipped = None if second is None else second != reference_score
        candidate_answer = candidate.answers[index]
        tag: str | None = None
        if reference_score is not None and candidate_score is not None:
            tag = loss_tag(
                candidate_answer, candidate_score, reference_score, sample_plan
            ).value
        row: dict[str, object] = {
            "qid": sample_plan.sample.sample_id,
            "gold": golds,
            "scores": row_scores,
        }
        if matches is not None:
            row["match"] = {run.label: matches[run.label][index] for run in runs}
        rows.append(
            {
                **row,
                "answers": {run.label: run.answers[index].answer for run in runs},
                "loss_tag": tag,
                "reference_flipped": flipped,
                "blend_share": (
                    candidate_answer.counters.blend_share
                    if candidate.engagement_measured
                    else None
                ),
                "evidence": [
                    {
                        "passage_id": p.passage_id,
                        "origin": p.origin.value,
                        "stored_position": p.stored_position,
                        "measured_position": p.measured_position,
                    }
                    for p in sample_plan.passages
                    if p.is_evidence
                ],
            }
        )
    return rows


def _complete_scores(table: ScoreTable, runs: list[ArmRun]) -> dict[str, list[float]]:
    """Keep only the questions every run has a score for.

    Raises:
        ValueError: If *table* lacks a run, or no question is fully scored.
    """
    for run in runs:
        if run.label not in table:
            raise ValueError(f"no scores for {run.label}")
    columns = [table[run.label] for run in runs]
    keep = [
        index
        for index in range(len(columns[0]))
        if all(column[index] is not None for column in columns)
    ]
    if not keep:
        raise ValueError("no question has a score from every run")
    complete: dict[str, list[float]] = {}
    for run in runs:
        picked = (table[run.label][index] for index in keep)
        complete[run.label] = [value for value in picked if value is not None]
    return complete


def _invalid_reasons(
    runs: list[ArmRun],
    engagement: EngagementCheck,
    min_blend_share: float,
    unscored: int,
    num_questions: int,
) -> list[str]:
    """List what makes the runs unfit to compare. If fit, then empty."""
    reasons: list[str] = []
    if unscored > MAX_UNSCORED_SHARE * num_questions:
        reasons.append(
            f"{unscored} of {num_questions} questions unscored (more than "
            f"{MAX_UNSCORED_SHARE:.0%})"
        )
    for run in runs:
        reasons.extend(run_problems(run))
    if not engagement.engaged:
        reason = (
            f"blend share {engagement.blend_share:.3f} (min window "
            f"{engagement.min_window_share:.3f}) is below {min_blend_share}"
        )
        if engagement.segmented_prefix_hit_tokens:
            # Not blending, but not a fallback to recompute either. Point at
            # storage before anyone debugs the blend path.
            reason += (
                f", with {engagement.segmented_prefix_hit_tokens:.0f} tokens "
                f"loaded past L2 prefix gaps (check L2 health)"
            )
        reasons.append(reason)
    return reasons


def _compare(
    label: str, scores: list[float], reference_scores: list[float]
) -> ArmComparison:
    """Compare one run's scores with the reference's, paired by question."""
    differences = [s - r for s, r in zip(scores, reference_scores, strict=True)]
    error = (
        statistics.stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else math.inf
    )
    return ArmComparison(
        label=label,
        mean_score=sum(scores) / len(scores),
        delta=sum(differences) / len(differences),
        standard_error=error,
    )
