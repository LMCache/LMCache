# SPDX-License-Identifier: Apache-2.0
"""Fixture tests for the CPU Device Tests rerun classifier.

Every case is a recorded payload: no network, no GitHub token. The positive
fixture mirrors the lost-runner failure linked from the issue (a job that died
in Checkout with every later step skipped); the negatives cover the ways a
failure can look similar but must never be retried.
"""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util
import sys

# Third Party
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/classify_runner_failure.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_lmcache_classify_runner_failure", _SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the classifier from {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotation resolution needs the module registered while it runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


classifier = _load_module()

RUN_PR = {
    "id": 42901580858,
    "name": "CPU Device Tests",
    "event": "pull_request",
    "run_attempt": 1,
    "conclusion": "failure",
    "head_sha": "5afe369cce0882fe202f6dd2d315f278854c765a",
}

LOST_RUNNER_ANNOTATION = {
    "path": ".github",
    "annotation_level": "failure",
    "message": (
        "The hosted runner lost communication with the server. Anything in your "
        "workflow that terminates the runner process, starves it for CPU/Memory, "
        "or blocks its network access can cause this error."
    ),
}

# The step list of the job linked from the issue, trimmed to the interesting
# entries: it died in Checkout, every LMCache step was skipped, and GitHub still
# marks "Upload logs on failure" and "Post Checkout" as run afterwards.
LOST_RUNNER_STEPS = [
    {"name": "Set up job", "status": "completed", "conclusion": "success"},
    {"name": "Pre Harden Runner", "status": "completed", "conclusion": "success"},
    {"name": "Harden Runner", "status": "completed", "conclusion": "success"},
    {"name": "Checkout", "status": "completed", "conclusion": "failure"},
    {"name": "Setup Python 3.12", "status": "completed", "conclusion": "skipped"},
    {"name": "Install FFmpeg (Linux)", "status": "completed", "conclusion": "skipped"},
    {
        "name": "Install vLLM CPU (prebuilt nightly from PyPI)",
        "status": "completed",
        "conclusion": "skipped",
    },
    {
        "name": "Install lmcache (CPU-only, no vLLM)",
        "status": "completed",
        "conclusion": "skipped",
    },
    {
        "name": "Server bench — LMCache-driven",
        "status": "completed",
        "conclusion": "skipped",
    },
    {
        "name": "vLLM e2e — LMCache-driven",
        "status": "completed",
        "conclusion": "skipped",
    },
    {"name": "Upload logs on failure", "status": "completed", "conclusion": "success"},
    {"name": "Post Checkout", "status": "completed", "conclusion": "success"},
    {"name": "Post Harden Runner", "status": "in_progress", "conclusion": None},
]

LOST_RUNNER_JOB = {
    "id": 130132997421,
    "name": "CPU device (ubuntu-22.04 / opt-125m / grpc)",
    "conclusion": "failure",
    "steps": LOST_RUNNER_STEPS,
}

TEST_FAILURE_JOB = {
    "id": 130132997999,
    "name": "CPU device (ubuntu-22.04 / opt-125m / zmq)",
    "conclusion": "failure",
    "steps": [
        {"name": "Set up job", "status": "completed", "conclusion": "success"},
        {"name": "Checkout", "status": "completed", "conclusion": "success"},
        {"name": "Setup Python 3.12", "status": "completed", "conclusion": "success"},
        {
            "name": "Install lmcache (CPU-only, no vLLM)",
            "status": "completed",
            "conclusion": "success",
        },
        {
            "name": "vLLM e2e — LMCache-driven",
            "status": "completed",
            "conclusion": "failure",
        },
    ],
}


def _run(jobs, annotations, run=None):
    """Classify with the fixtures, defaulting to the linked PR run."""
    return classifier.classify(run or RUN_PR, jobs, annotations)


class TestPositive:
    def test_lost_runner_at_checkout_is_rerun(self):
        decision = _run(
            [LOST_RUNNER_JOB],
            {LOST_RUNNER_JOB["id"]: [LOST_RUNNER_ANNOTATION]},
        )

        assert decision.rerun_job_ids == (LOST_RUNNER_JOB["id"],)
        (job,) = decision.jobs
        assert job.rerun is True
        assert job.reason == classifier.REASON_RERUN
        assert any("lost communication" in item for item in job.evidence)

    def test_only_the_failed_job_is_selected(self):
        healthy = {
            "id": 2,
            "name": "CPU device (ubuntu-22.04 / opt-125m / zmq)",
            "conclusion": "success",
            "steps": [
                {"name": "Set up job", "status": "completed", "conclusion": "success"},
                {"name": "Checkout", "status": "completed", "conclusion": "success"},
            ],
        }
        decision = _run(
            [LOST_RUNNER_JOB, healthy],
            {LOST_RUNNER_JOB["id"]: [LOST_RUNNER_ANNOTATION]},
        )

        assert decision.rerun_job_ids == (LOST_RUNNER_JOB["id"],)
        reasons = {job.job_id: job.reason for job in decision.jobs}
        assert reasons[healthy["id"]] == classifier.REASON_JOB_NOT_FAILED


class TestNegatives:
    def test_real_checkout_error_is_not_rerun(self):
        job = dict(LOST_RUNNER_JOB)
        annotations = {
            job["id"]: [
                {
                    "annotation_level": "failure",
                    "message": "fatal: could not read Username for 'https://github.com'",
                }
            ]
        }

        decision = _run([job], annotations)

        assert decision.rerun_job_ids == ()
        assert decision.jobs[0].reason == classifier.REASON_UNKNOWN_EVIDENCE

    def test_test_failure_is_not_rerun(self):
        annotations = {
            TEST_FAILURE_JOB["id"]: [
                {
                    "annotation_level": "failure",
                    "message": (
                        "FAILED tests/cli/test_ping.py::TestDefaultUrls - "
                        "AssertionError"
                    ),
                }
            ]
        }

        decision = _run([TEST_FAILURE_JOB], annotations)

        assert decision.rerun_job_ids == ()
        assert decision.jobs[0].reason == classifier.REASON_FAILED_STEP_NOT_PREPARATORY
        assert "vLLM e2e" in decision.jobs[0].evidence[0]

    def test_work_started_before_a_preparatory_failure_is_not_rerun(self):
        """Checkout failed, but LMCache had already been installed once."""
        steps = [dict(step) for step in LOST_RUNNER_STEPS]
        steps[7] = {
            "name": "Install lmcache (CPU-only, no vLLM)",
            "status": "completed",
            "conclusion": "success",
        }
        job = dict(LOST_RUNNER_JOB, steps=steps)

        decision = _run([job], {job["id"]: [LOST_RUNNER_ANNOTATION]})

        assert decision.rerun_job_ids == ()
        assert decision.jobs[0].reason == classifier.REASON_WORK_STARTED

    def test_second_attempt_is_never_rerun(self):
        run = dict(RUN_PR, run_attempt=2)
        decision = _run(
            [LOST_RUNNER_JOB],
            {LOST_RUNNER_JOB["id"]: [LOST_RUNNER_ANNOTATION]},
            run=run,
        )

        assert decision.rerun_job_ids == ()
        assert decision.jobs[0].reason == classifier.REASON_ALREADY_RETRIED
        assert any("run_attempt" in note for note in decision.notes)

    def test_mixed_evidence_is_not_rerun(self):
        annotations = {
            LOST_RUNNER_JOB["id"]: [
                LOST_RUNNER_ANNOTATION,
                {
                    "annotation_level": "failure",
                    "message": "Process completed with exit code 1.",
                },
            ]
        }

        decision = _run([LOST_RUNNER_JOB], annotations)

        assert decision.rerun_job_ids == ()
        assert decision.jobs[0].reason == classifier.REASON_MIXED_EVIDENCE

    def test_non_pull_request_run_is_not_rerun(self):
        run = dict(RUN_PR, event="push")
        decision = _run(
            [LOST_RUNNER_JOB],
            {LOST_RUNNER_JOB["id"]: [LOST_RUNNER_ANNOTATION]},
            run=run,
        )

        assert decision.rerun_job_ids == ()
        assert decision.jobs[0].reason == classifier.REASON_NOT_PULL_REQUEST

    def test_missing_annotations_is_not_rerun(self):
        decision = _run([LOST_RUNNER_JOB], {})

        assert decision.rerun_job_ids == ()
        assert decision.jobs[0].reason == classifier.REASON_UNKNOWN_EVIDENCE


class TestSummary:
    def test_summary_mentions_the_decision_and_the_job(self):
        decision = _run(
            [LOST_RUNNER_JOB],
            {LOST_RUNNER_JOB["id"]: [LOST_RUNNER_ANNOTATION]},
        )

        summary = classifier.render_summary(decision)

        assert "Rerun requested" in summary
        assert str(LOST_RUNNER_JOB["id"]) in summary
        assert classifier.REASON_RERUN in summary

    def test_summary_says_no_rerun_when_skipping(self):
        decision = _run([LOST_RUNNER_JOB], {})

        summary = classifier.render_summary(decision)

        assert "No rerun requested." in summary


class TestCli:
    def test_writes_decision_and_summary(self, tmp_path, capsys):
        # Standard
        import json

        run_path = tmp_path / "run.json"
        jobs_path = tmp_path / "jobs.json"
        annotations_path = tmp_path / "annotations.json"
        decision_path = tmp_path / "decision.json"
        summary_path = tmp_path / "summary.md"

        run_path.write_text(json.dumps(RUN_PR), encoding="utf-8")
        jobs_path.write_text(json.dumps([LOST_RUNNER_JOB]), encoding="utf-8")
        annotations_path.write_text(
            json.dumps({str(LOST_RUNNER_JOB["id"]): [LOST_RUNNER_ANNOTATION]}),
            encoding="utf-8",
        )

        exit_code = classifier.main(
            [
                "--run",
                str(run_path),
                "--jobs",
                str(jobs_path),
                "--annotations",
                str(annotations_path),
                "--decision-out",
                str(decision_path),
                "--summary-out",
                str(summary_path),
            ]
        )

        assert exit_code == 0
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        assert decision["rerun_job_ids"] == [LOST_RUNNER_JOB["id"]]
        assert "Rerun requested" in summary_path.read_text(encoding="utf-8")
        assert str(LOST_RUNNER_JOB["id"]) in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
