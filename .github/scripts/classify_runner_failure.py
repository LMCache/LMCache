# SPDX-License-Identifier: Apache-2.0
"""Classify a failed GitHub Actions job as a runner-infrastructure failure.

The recovery workflow for `CPU Device Tests` must only rerun a job when the
evidence says GitHub lost the runner, never when a test or an install step
failed. That decision lives here, away from the workflow, so it can be tested
with recorded API payloads instead of trusting a shell `if` in YAML.

The input is a snapshot assembled by the workflow from three read-only GitHub
API calls:

- ``run``: one entry from ``GET /repos/{owner}/{repo}/actions/runs/{id}``
- ``jobs``: entries from ``GET /repos/{owner}/{repo}/actions/runs/{id}/jobs``
- ``annotations``: ``{job_id: [...]}`` from
  ``GET /repos/{owner}/{repo}/check-runs/{job_id}/annotations``

The output is a :class:`Decision` per failed job plus the ids that may be
rerun. Unknown evidence is never a rerun.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import argparse
import json
import sys

#: Annotation text GitHub emits when the hosted runner disappears.
LOST_RUNNER_SIGNAL = "The hosted runner lost communication with the server"

#: Additional infrastructure-only signals. Each entry must be specific to the
#: runner or the GitHub service, never to a command that ran in the job.
DEFAULT_SIGNALS: tuple[str, ...] = (
    LOST_RUNNER_SIGNAL,
    "The runner has received a shutdown signal",
    "The self-hosted runner lost communication with the server",
    "The operation was canceled",
)

#: Step names whose start means LMCache work happened; a rerun is unsafe once
#: one of them has begun. Matched case-insensitively as substrings, and kept
#: specific so that environment preparation (ffmpeg, Python) does not count.
EXECUTING_STEP_MARKERS: tuple[str, ...] = (
    "install lmcache",
    "install vllm",
    "install pygrpc",
    "server bench",
    "vllm e2e",
    "unit tests",
    "collect verify",
    "prepare model",
    "cache huggingface",
    "download model",
)

#: Annotation text that indicates a command produced a verdict. Seeing one of
#: these next to an infrastructure signal makes the evidence mixed.
VERDICT_MARKERS: tuple[str, ...] = (
    "Process completed with exit code",
    "FAILED ",
    "AssertionError",
    "Traceback (most recent call last)",
    "No space left on device",
    "fatal: ",
    "error: ",
)

#: Steps that run before any LMCache work. A job that dies in one of them is
#: consistent with a lost runner; a job that dies later is not.
PREPARATORY_STEP_MARKERS: tuple[str, ...] = (
    "set up job",
    "harden runner",
    "checkout",
    "free disk space",
    "setup python",
    "install ffmpeg",
)

#: Steps GitHub appends for cleanup and log upload. They may run (and even
#: succeed) after the runner is gone, so they never count as work.
IGNORED_STEP_MARKERS: tuple[str, ...] = (
    "post ",
    "upload logs on failure",
    "pre ",
)

#: Reason codes, so the workflow summary and the tests name the same thing.
REASON_RERUN = "lost_runner"
REASON_NOT_PULL_REQUEST = "not_a_pull_request"
REASON_ALREADY_RETRIED = "already_retried"
REASON_JOB_NOT_FAILED = "job_not_failed"
REASON_WORK_STARTED = "work_started"
REASON_FAILED_STEP_NOT_PREPARATORY = "failed_step_not_preparatory"
REASON_UNKNOWN_EVIDENCE = "unknown_evidence"
REASON_MIXED_EVIDENCE = "mixed_evidence"


@dataclass(frozen=True)
class JobDecision:
    """What to do with one failed job."""

    job_id: int
    job_name: str
    rerun: bool
    reason: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    """The classification of one workflow run."""

    rerun_job_ids: tuple[int, ...]
    jobs: tuple[JobDecision, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form for the workflow."""
        return {
            "rerun_job_ids": list(self.rerun_job_ids),
            "notes": list(self.notes),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "rerun": job.rerun,
                    "reason": job.reason,
                    "evidence": list(job.evidence),
                }
                for job in self.jobs
            ],
        }


def _step_started(step: Mapping[str, Any]) -> bool:
    """Return True when a step did more than get skipped."""
    if step.get("status") not in ("completed",):
        return step.get("status") in ("in_progress",)
    return step.get("conclusion") not in ("skipped", "cancelled", None)


def _is_preparatory(step_name: str) -> bool:
    lowered = step_name.lower()
    return any(marker in lowered for marker in PREPARATORY_STEP_MARKERS)


def _is_ignored(step_name: str) -> bool:
    lowered = step_name.lower()
    return any(lowered.startswith(marker) for marker in IGNORED_STEP_MARKERS)


def _failed_step(steps: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the step the job failed in, if the API reports one."""
    for step in steps:
        if step.get("conclusion") == "failure":
            return step
    return None


def _work_started(steps: Sequence[Mapping[str, Any]]) -> tuple[bool, tuple[str, ...]]:
    """Detect whether LMCache work began before the job died.

    Only the markers in :data:`EXECUTING_STEP_MARKERS` count: a job that got as
    far as installing LMCache, benching a server or running a vLLM e2e leg has
    produced evidence a rerun would hide. Cleanup and log-upload steps are
    ignored, because GitHub runs them after the job already failed.
    """
    started: list[str] = []
    for step in steps:
        name = str(step.get("name", ""))
        if _is_ignored(name) or not _step_started(step):
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in EXECUTING_STEP_MARKERS):
            started.append(name)
    return bool(started), tuple(started)


def _annotation_texts(annotations: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    texts: list[str] = []
    for annotation in annotations:
        message = annotation.get("message")
        if isinstance(message, str) and message.strip():
            texts.append(message.strip())
    return tuple(texts)


def _match_signals(texts: Sequence[str], signals: Sequence[str]) -> tuple[str, ...]:
    matched = [signal for signal in signals if any(signal in t for t in texts)]
    return tuple(matched)


def _match_verdicts(texts: Sequence[str]) -> tuple[str, ...]:
    hits: list[str] = []
    for text in texts:
        for marker in VERDICT_MARKERS:
            if marker in text:
                hits.append(text[:200])
                break
    return tuple(hits)


def classify_job(
    job: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
    *,
    attempt: int,
    is_pull_request: bool,
    signals: Sequence[str] = DEFAULT_SIGNALS,
) -> JobDecision:
    """Decide whether one failed job may be rerun automatically."""
    job_id = int(job.get("id", 0))
    job_name = str(job.get("name", ""))
    texts = _annotation_texts(annotations)

    if job.get("conclusion") != "failure":
        return JobDecision(job_id, job_name, False, REASON_JOB_NOT_FAILED, texts)
    if not is_pull_request:
        return JobDecision(job_id, job_name, False, REASON_NOT_PULL_REQUEST, texts)
    if attempt > 1:
        return JobDecision(job_id, job_name, False, REASON_ALREADY_RETRIED, texts)

    steps = job.get("steps") or []
    failed_step = _failed_step(steps)
    if failed_step is not None:
        failed_name = str(failed_step.get("name", ""))
        if not _is_preparatory(failed_name):
            return JobDecision(
                job_id,
                job_name,
                False,
                REASON_FAILED_STEP_NOT_PREPARATORY,
                (failed_name,),
            )

    started, started_steps = _work_started(steps)
    if started:
        return JobDecision(
            job_id, job_name, False, REASON_WORK_STARTED, started_steps[:5]
        )

    matched = _match_signals(texts, signals)
    if not matched:
        return JobDecision(job_id, job_name, False, REASON_UNKNOWN_EVIDENCE, texts)

    verdicts = _match_verdicts(texts)
    if verdicts:
        return JobDecision(job_id, job_name, False, REASON_MIXED_EVIDENCE, verdicts[:5])

    return JobDecision(job_id, job_name, True, REASON_RERUN, matched)


def classify(
    run: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    signals: Sequence[str] = DEFAULT_SIGNALS,
) -> Decision:
    """Classify a whole workflow run.

    Args:
        run: The workflow-run payload.
        jobs: Jobs of that run.
        annotations: Check-run annotations keyed by job id.
        signals: Infrastructure signals that may justify a rerun.

    Returns:
        The decision, with one entry per failed job.
    """
    attempt = int(run.get("run_attempt") or 1)
    event = str(run.get("event") or "")
    is_pull_request = event == "pull_request"

    notes: list[str] = []
    if not is_pull_request:
        notes.append(f"event is {event!r}, not 'pull_request'")
    if attempt > 1:
        notes.append(f"run_attempt is {attempt}, automatic rerun happens once")

    decisions: list[JobDecision] = []
    for job in jobs:
        job_id = int(job.get("id", 0))
        decision = classify_job(
            job,
            annotations.get(job_id, ()),
            attempt=attempt,
            is_pull_request=is_pull_request,
            signals=signals,
        )
        decisions.append(decision)

    rerun_ids = tuple(d.job_id for d in decisions if d.rerun)
    if len(rerun_ids) > 1:
        notes.append(f"{len(rerun_ids)} jobs classified as rerunnable")
    return Decision(rerun_job_ids=rerun_ids, jobs=tuple(decisions), notes=tuple(notes))


def render_summary(decision: Decision) -> str:
    """Render a markdown job summary for the workflow."""
    lines = ["## CPU Device Tests rerun classification", ""]
    if decision.rerun_job_ids:
        lines.append(
            "Rerun requested for: "
            + ", ".join(f"`{job_id}`" for job_id in decision.rerun_job_ids)
        )
    else:
        lines.append("No rerun requested.")
    lines.append("")
    lines.append("| Job | Decision | Reason | Evidence |")
    lines.append("| --- | --- | --- | --- |")
    for job in decision.jobs:
        decision_text = "rerun" if job.rerun else "skip"
        evidence = "<br>".join(_escape_cell(item) for item in job.evidence[:3])
        if len(job.evidence) > 3:
            evidence += "<br>…"
        lines.append(
            f"| `{job.job_id}` {_escape_cell(job.job_name)} | {decision_text} | "
            f"`{job.reason}` | {evidence} |"
        )
    for note in decision.notes:
        lines.append("")
        lines.append(f"> {note}")
    lines.append("")
    return "\n".join(lines)


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the classifier over snapshot files.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 when classification succeeded.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="workflow run JSON")
    parser.add_argument("--jobs", required=True, type=Path, help="run jobs JSON")
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="JSON object mapping job id to its check-run annotations",
    )
    parser.add_argument("--decision-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)

    run = _load_json(args.run)
    jobs = _load_json(args.jobs)
    raw_annotations = _load_json(args.annotations)
    annotations = {
        int(job_id): value for job_id, value in dict(raw_annotations).items()
    }

    decision = classify(run, jobs, annotations)

    if args.decision_out is not None:
        args.decision_out.write_text(
            json.dumps(decision.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    if args.summary_out is not None:
        args.summary_out.write_text(render_summary(decision), encoding="utf-8")

    print(json.dumps(decision.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
