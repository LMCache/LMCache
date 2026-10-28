# SPDX-License-Identifier: Apache-2.0
"""Results files and the terminal summary for ``lmcache bench quality``."""

# Standard
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.scoring import is_abstention
from lmcache.cli.commands.bench.quality_bench.core.config import (
    ArmRole,
    QualityAnalyzeConfig,
    VerdictMetric,
)
from lmcache.cli.commands.bench.quality_bench.core.judge import JUDGE_METRIC
from lmcache.cli.commands.bench.quality_bench.core.plan import QualityPlan
from lmcache.cli.commands.bench.quality_bench.core.run_files import (
    ROWS_FILE,
    SUMMARY_FILE,
    ArmRecord,
    PlanRecord,
    write_json,
    write_jsonl,
)
from lmcache.cli.commands.bench.quality_bench.core.runner import ArmRun
from lmcache.cli.commands.bench.quality_bench.core.verdict import (
    ArmComparison,
    QualityVerdict,
    ReuseReport,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)


def run_fingerprint(plan_record: PlanRecord, config: QualityAnalyzeConfig) -> str:
    """Digest of everything that determines the prompts and their scoring.

    Two summaries are comparable only when their fingerprints match.

    Args:
        plan_record: The results directory's plan.
        config: The analysis's config.

    Returns:
        A 16-hex-digit digest.
    """
    material = json.dumps(
        {
            "config": {**plan_record.settings, **config.scoring_settings()},
            "prompts": plan_record.plan.prompts_digest,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def arm_counts(run: ArmRun) -> dict[str, int]:
    """Count the failure modes obscured by a mean score.

    Args:
        run: One arm run.

    Returns:
        ``answered`` (complete answers), ``empty`` (failed or cut off) and
        ``abstained`` (complete answers that decline to answer).
    """
    parsed = [a for a in run.answers if a.parsed]
    return {
        "answered": len(parsed),
        "empty": len(run.answers) - len(parsed),
        "abstained": sum(1 for a in parsed if is_abstention(a.answer)),
    }


def write_results(
    config: QualityAnalyzeConfig,
    plan_record: PlanRecord,
    arms: list[ArmRecord],
    verdict: QualityVerdict,
    rows: list[dict[str, object]],
) -> str:
    """Write the scored results into the results directory.

    Files, under ``config.results_dir``, next to what ``run`` saved:

    * ``rows.jsonl`` — the per-question comparison rows.
    * ``summary.json`` — fingerprint, settings, each arm's provenance,
      verdict (with any reasons it is INVALID), engagement, per-arm scores
      and failure counts.

    Both are written atomically and as strict JSON (an undefined standard
    error is ``null``), so any JSON parser reads them.

    Args:
        config: The analysis's config.
        plan_record: The results directory's plan.
        arms: Every arm pass, as analyzed.
        verdict: The verdict.
        rows: The rows from ``build_rows``.

    Returns:
        Path of ``summary.json``.
    """
    plan = plan_record.plan
    runs = [arm.run for arm in arms]
    candidate = next(arm for arm in arms if arm.run.role is ArmRole.CANDIDATE)
    write_jsonl(os.path.join(config.results_dir, ROWS_FILE), rows)

    summary = {
        "run_fingerprint": run_fingerprint(plan_record, config),
        "plan_fingerprint": plan_record.fingerprint,
        "settings": {
            **dict(plan_record.settings),
            **dict(config.scoring_settings()),
            "reference_repeats": sum(
                1 for run in runs if run.role is ArmRole.REFERENCE
            ),
            "tolerance": config.tolerance,
            "min_blend_share": config.min_blend_share,
            "blend_metrics_url": candidate.blend_metrics_url,
        },
        "arms": {arm.run.label: arm.metadata() for arm in arms},
        "num_questions": len(plan.samples),
        "unscored_questions": verdict.unscored,
        "judge": (
            {"model": config.judge.model, "metric": JUDGE_METRIC}
            if config.verdict_metric is VerdictMetric.JUDGE and config.judge
            else None
        ),
        "evidence_sources": plan.evidence_coverage(),
        "measured_earlier_share": plan.measured_earlier_share(),
        "padding": _padding_record(plan),
        "reuse": _reuse_record(verdict.reuse),
        "num_store_requests": len(plan.store_requests),
        "verdict": verdict.verdict.value,
        "invalid_reasons": verdict.invalid_reasons,
        "samples_needed": verdict.samples_needed,
        "reference_flip_rate": verdict.reference_flip_rate,
        "candidate": _comparison_record(verdict.candidate),
        "others": [_comparison_record(c) for c in verdict.others],
        "engagement": asdict(verdict.engagement),
        "arm_counts": {run.label: arm_counts(run) for run in runs},
        "store_failures": {run.label: run.store_failures for run in runs},
        "loss_tags": dict(
            Counter(str(row["loss_tag"]) for row in rows if row["loss_tag"])
        ),
    }
    summary_path = os.path.join(config.results_dir, SUMMARY_FILE)
    write_json(summary_path, summary)
    logger.info("Wrote quality results to %s", config.results_dir)
    return summary_path


def format_summary(
    plan: QualityPlan, runs: list[ArmRun], verdict: QualityVerdict
) -> str:
    """Render the verdict and per-arm scores for the terminal.

    Args:
        plan: The run's plan.
        runs: Every arm run.
        verdict: The verdict.

    Returns:
        Multi-line text.
    """
    comparisons: dict[str, ArmComparison] = {
        c.label: c for c in [verdict.candidate, *verdict.others]
    }
    lines = [
        f"bench quality: {len(plan.samples)} questions, "
        f"scenario={plan.scenario.value}, metric={verdict.metric.value}",
        f"{'arm':<14}{'score':>8}{'delta':>9}{'2*SE':>8}{'empty':>7}{'abstain':>9}",
    ]
    for run in runs:
        counts = arm_counts(run)
        comparison = comparisons.get(run.label)
        if comparison is None:
            # The first reference run, which every delta is measured from.
            reference_mean = verdict.candidate.mean_score - verdict.candidate.delta
            score_text, delta_text, error_text = f"{reference_mean:.3f}", "", ""
        else:
            score_text = f"{comparison.mean_score:.3f}"
            delta_text = f"{comparison.delta:+.3f}"
            error_text = f"{2 * comparison.standard_error:.3f}"
        lines.append(
            f"{run.label:<14}{score_text:>8}{delta_text:>9}{error_text:>8}"
            f"{counts['empty']:>7}{counts['abstained']:>9}"
        )
    if verdict.reference_flip_rate >= 0:
        lines.append(f"reference flip rate: {verdict.reference_flip_rate:.3f}")
    engagement = verdict.engagement
    if engagement.measured:
        lines.append(
            f"blend share: {engagement.blend_share:.3f} "
            f"(min window {engagement.min_window_share:.3f}), "
            f"stale chunks: {engagement.stale_chunks:.0f}, "
            f"fingerprints registered: {engagement.fingerprints_registered:.0f}"
        )
        if engagement.segmented_prefix_hit_tokens:
            lines.append(
                f"segmented-prefix tokens: "
                f"{engagement.segmented_prefix_hit_tokens:.0f} (loaded past L2 "
                f"prefix gaps, not counted as blending)"
            )
    else:
        lines.append("blend share: not captured (no --capture-blend-metrics)")
    if plan.padding is not None:
        lines.append(
            f"padding: --filler {plan.padding.filler}, "
            f"{plan.padding.share:.1%} of measured prompt tokens"
        )
    reuse = verdict.reuse
    if reuse is not None:
        expected = (
            f"expected {reuse.expected_tokens}"
            if reuse.expected_tokens is not None
            else "expected unknown"
        )
        ratio = reuse.observed_over_expected
        lines.append(
            f"reuse: {expected} tokens, observed {reuse.observed_tokens:.0f} "
            f"(prefix {reuse.observed_prefix_tokens:.0f}, blended "
            f"{reuse.observed_blended_tokens:.0f})"
            + (f" = {ratio:.2f} of expected" if ratio is not None else "")
        )
        for group in reuse.groups:
            lines.append(
                f"  {group.kind.value:<8} {group.questions:>4} questions  "
                f"delta {group.delta:+.3f}  2*SE {2 * group.standard_error:.3f}"
            )
    coverage = plan.evidence_coverage()
    lines.append(
        "evidence passages identified: "
        + ", ".join(f"{source} {count}" for source, count in coverage.items())
    )
    earlier_share = plan.measured_earlier_share()
    if earlier_share:
        lines.append(
            f"passages also in an earlier question: {earlier_share:.1%} (a "
            f"caching candidate may reuse them from that question's prompt. "
            f"See measured_earlier_in in the manifest)"
        )
    if verdict.unscored:
        lines.append(
            f"unscored: {verdict.unscored} questions left out (no judge rating "
            f"for some arm)"
        )
    if verdict.metric is not VerdictMetric.JUDGE:
        lines.append(
            f"note: {verdict.metric.value} is a judge-free screen. It can "
            f"understate a loss. Use --judge-url for numbers to report."
        )
    verdict_line = f"verdict: {verdict.verdict.value}"
    if verdict.samples_needed:
        verdict_line += f" (about {verdict.samples_needed} questions would settle it)"
    if verdict.invalid_reasons:
        verdict_line += f" ({'; '.join(verdict.invalid_reasons)})"
    lines.append(verdict_line)
    return "\n".join(lines)


def _reuse_record(reuse: ReuseReport | None) -> dict[str, object] | None:
    """Return the reuse report as JSON, or ``None`` when unmeasured."""
    if reuse is None:
        return None
    return {
        "expected_tokens": reuse.expected_tokens,
        "observed_tokens": reuse.observed_tokens,
        "observed_prefix_tokens": reuse.observed_prefix_tokens,
        "observed_blended_tokens": reuse.observed_blended_tokens,
        "observed_over_expected": reuse.observed_over_expected,
        "groups": [
            {
                "kind": group.kind.value,
                "questions": group.questions,
                "delta": group.delta,
                "standard_error": (
                    None if math.isinf(group.standard_error) else group.standard_error
                ),
            }
            for group in reuse.groups
        ],
    }


def _padding_record(plan: QualityPlan) -> dict[str, object] | None:
    """Return the plan's filler statistics, or ``None`` for older plans."""
    if plan.padding is None:
        return None
    return {
        "filler": plan.padding.filler,
        "pad_tokens": plan.padding.pad_tokens,
        "prompt_tokens": plan.padding.prompt_tokens,
        "share": plan.padding.share,
    }


def _comparison_record(comparison: ArmComparison) -> dict[str, object]:
    """Return *comparison* for ``summary.json``, as strict JSON.

    ``standard_error`` is infinite with fewer than two scored questions;
    JSON has no infinity, so it is written as ``null``.
    """
    record = asdict(comparison)
    if not math.isfinite(comparison.standard_error):
        record["standard_error"] = None
    return record
