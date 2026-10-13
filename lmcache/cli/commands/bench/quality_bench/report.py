# SPDX-License-Identifier: Apache-2.0
"""Results files and the terminal summary for ``lmcache bench quality``."""

# Standard
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.scoring import is_abstention
from lmcache.cli.commands.bench.quality_bench.config import (
    QualityBenchConfig,
    VerdictMetric,
)
from lmcache.cli.commands.bench.quality_bench.judge import JUDGE_METRIC
from lmcache.cli.commands.bench.quality_bench.plan import QualityPlan
from lmcache.cli.commands.bench.quality_bench.runner import ArmRun
from lmcache.cli.commands.bench.quality_bench.verdict import (
    ArmComparison,
    QualityVerdict,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)

MANIFEST_FILE = "manifest.jsonl"
ROWS_FILE = "rows.jsonl"
SUMMARY_FILE = "summary.json"

# Name of each run's subdirectory of --output-dir: when the run started.
RUN_DIR_FORMAT = "%Y%m%d-%H%M%S"


def run_fingerprint(config: QualityBenchConfig, plan: QualityPlan) -> str:
    """Digest of everything that determines the prompts and their scoring.

    Two result directories are comparable only when their fingerprints
    match.

    Args:
        config: The run's config.
        plan: The run's plan.

    Returns:
        A 16-hex-digit digest.
    """
    material = json.dumps(
        {"config": dict(config.fingerprint_material()), "prompts": plan.prompts_digest},
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


def create_run_dir(output_dir: str, started: datetime) -> str:
    """Create the subdirectory of *output_dir* that one run writes into.

    Every run gets its own directory, so runs into the same ``--output-dir``
    never mix files: a rerun with fewer arms cannot leave a stale
    ``answers_apc.jsonl`` behind, and a run that aborts while judging cannot
    leave its answers next to an earlier run's summary. Only the judge
    cache is shared, directly under *output_dir*.

    Args:
        output_dir: The ``--output-dir`` root -- created if missing.
        started: When the run started. Names the directory
            (``YYYYmmdd-HHMMSS``).

    Returns:
        Path of the new directory. A run that starts in the same second as
        an existing directory gets a ``-2``, ``-3``, … suffix.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, started.strftime(RUN_DIR_FORMAT))
    path, suffix = base, 1
    while True:
        try:
            # mkdir claims the name atomically, even against a concurrent run.
            os.mkdir(path)
            return path
        except FileExistsError:
            suffix += 1
            path = f"{base}-{suffix}"


def write_answers(run_dir: str, plan: QualityPlan, runs: list[ArmRun]) -> None:
    """Write what the arms produced, before any scoring.

    Written first so that a judge failure during scoring does not lose the
    arm runs, which are the expensive part. Files, under *run_dir*:

    * ``manifest.jsonl`` — one reuse-manifest record per question.
    * ``answers_<label>.jsonl`` — one record per question per arm run.

    Args:
        run_dir: The run's directory, from :func:`create_run_dir`.
        plan: The run's plan.
        runs: Every arm run.
    """
    _write_jsonl(
        os.path.join(run_dir, MANIFEST_FILE),
        [s.manifest_entry(plan.scenario, plan.chunk_size) for s in plan.samples],
    )
    for run in runs:
        _write_jsonl(
            os.path.join(run_dir, f"answers_{run.label}.jsonl"),
            [
                {
                    "qid": a.sample_id,
                    "request_id": a.request_id,
                    "successful": a.successful,
                    "answer": a.answer,
                    "response": a.response,
                    "ttft": round(a.ttft, 4),
                    "num_output_tokens": a.num_output_tokens,
                    "counters": asdict(a.counters) if run.engagement_measured else None,
                }
                for a in run.answers
            ],
        )
    logger.info("Wrote manifest and answers to %s", run_dir)


def write_results(
    run_dir: str,
    config: QualityBenchConfig,
    plan: QualityPlan,
    runs: list[ArmRun],
    verdict: QualityVerdict,
    rows: list[dict[str, object]],
) -> str:
    """Write the scored results, next to :func:`write_answers`'s files.

    Files, under *run_dir*:

    * ``rows.jsonl`` — the per-question comparison rows.
    * ``summary.json`` — fingerprint, settings, verdict (with any reasons it
      is INVALID), engagement, per-arm scores and failure counts.

    Args:
        run_dir: The run's directory, from :func:`create_run_dir`.
        config: The run's config.
        plan: The run's plan.
        runs: Every arm run.
        verdict: The verdict.
        rows: The rows from ``build_rows``.

    Returns:
        Path of ``summary.json``.
    """
    _write_jsonl(os.path.join(run_dir, ROWS_FILE), rows)

    summary = {
        "run_fingerprint": run_fingerprint(config, plan),
        "settings": {
            **dict(config.fingerprint_material()),
            "arms": {arm.role.value: arm.url for arm in config.arms},
            "reference_repeats": config.reference_repeats,
            "verdict_metric": config.verdict_metric.value,
            "tolerance": config.tolerance,
            "min_blend_share": config.min_blend_share,
            "blend_metrics_url": config.blend_metrics_url,
        },
        "num_questions": len(plan.samples),
        "unscored_questions": verdict.unscored,
        "judge": (
            {"model": config.judge.model, "metric": JUDGE_METRIC}
            if config.verdict_metric is VerdictMetric.JUDGE and config.judge
            else None
        ),
        "evidence_sources": plan.evidence_coverage(),
        "measured_earlier_share": plan.measured_earlier_share(),
        "num_store_requests": len(plan.store_requests),
        "verdict": verdict.verdict.value,
        "invalid_reasons": verdict.invalid_reasons,
        "samples_needed": verdict.samples_needed,
        "reference_flip_rate": verdict.reference_flip_rate,
        "blend": asdict(verdict.blend),
        "others": [asdict(c) for c in verdict.others],
        "engagement": asdict(verdict.engagement),
        "arm_counts": {run.label: arm_counts(run) for run in runs},
        "store_failures": {run.label: run.store_failures for run in runs},
        "loss_tags": dict(
            Counter(str(row["loss_tag"]) for row in rows if row["loss_tag"])
        ),
    }
    summary_path = os.path.join(run_dir, SUMMARY_FILE)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.info("Wrote quality results to %s", run_dir)
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
        c.label: c for c in [verdict.blend, *verdict.others]
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
            reference_mean = verdict.blend.mean_score - verdict.blend.delta
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
        lines.append("blend share: not measured (no --blend-metrics-url)")
    coverage = plan.evidence_coverage()
    lines.append(
        "evidence passages identified: "
        + ", ".join(f"{source} {count}" for source, count in coverage.items())
    )
    earlier_share = plan.measured_earlier_share()
    if earlier_share:
        lines.append(
            f"passages also in an earlier question: {earlier_share:.1%} (the "
            f"blend arm may reuse them from that question's prompt. See "
            f"measured_earlier_in in the manifest)"
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


def _write_jsonl(path: str, records: list[dict[str, object]]) -> None:
    """Write *records* to *path*, one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
