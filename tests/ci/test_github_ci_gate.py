# SPDX-License-Identifier: Apache-2.0
"""Tests for the k3-unit-tests gate on the GitHub Layer 1 / Layer 2 checks.

Run without loading the torch-dependent tests/conftest.py:
    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_github_ci_gate.py
"""

# Standard
from pathlib import Path
from types import ModuleType
from typing import Any
import importlib.util
import os
import shutil
import subprocess
import urllib.request

# Third Party
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / ".buildkite/k3_tests/common_scripts"
GATE = SCRIPTS / "github_ci_gate.py"
WRAPPER = SCRIPTS / "run-gpu-with-l2-watch.sh"


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    """Load the gate script, which lives outside any package."""
    spec = importlib.util.spec_from_file_location("github_ci_gate", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(
    name: str, status: str = "completed", conclusion: str | None = "success"
) -> dict[str, Any]:
    """Build a GitHub Actions job."""
    return {"name": name, "status": status, "conclusion": conclusion}


def _run(
    created_at: str, status: str = "completed", conclusion: str | None = "success"
) -> dict[str, Any]:
    """Build a GitHub Actions workflow run."""
    return {
        "created_at": created_at,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/run/{created_at}",
    }


DRAFT_JOBS = [_job("Check code quality", conclusion="skipped"), _job("Test / t")]
READY_JOBS = [_job("Check code quality"), _job("Test / t")]


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/LMCache/LMCache.git",
        "https://github.com/LMCache/LMCache",
        "git@github.com:LMCache/LMCache.git",
    ],
)
def test_repo_slug(gate: ModuleType, url: str) -> None:
    """HTTPS and SSH clone URLs resolve to the owner/name slug."""
    assert gate.repo_slug(url) == "LMCache/LMCache"


def test_repo_slug_rejects_non_github(gate: ModuleType) -> None:
    """A non-GitHub URL is an error rather than a wrong API path."""
    with pytest.raises(ValueError):
        gate.repo_slug("https://gitlab.com/LMCache/LMCache.git")


@pytest.mark.parametrize(
    ("jobs_by_run", "expected"),
    [
        ({}, None),
        # A draft run (pre-commit skipped) never gates the GPU
        ({"t1": DRAFT_JOBS}, None),
        # Ready again on the same commit: the newest ready run wins
        ({"t1": READY_JOBS, "t2": DRAFT_JOBS, "t3": READY_JOBS}, "t3"),
        ({"t1": READY_JOBS, "t2": DRAFT_JOBS}, "t1"),
    ],
)
def test_newest_ready_run(
    gate: ModuleType, jobs_by_run: dict[str, list[Any]], expected: str | None
) -> None:
    """Only ready-for-review runs count, newest first."""
    runs = [_run(created_at) for created_at in jobs_by_run]
    found = gate.newest_ready_run(runs, lambda r: jobs_by_run[r["created_at"]])
    assert (found[0]["created_at"] if found else None) == expected


@pytest.mark.parametrize(
    ("gate_job", "expected"),
    [
        (_job("Check code quality", "in_progress", None), "wait"),
        (_job("Check code quality"), "pass"),
        (_job("Check code quality", conclusion="failure"), "fail"),
        (_job("Check code quality", conclusion="cancelled"), "fail"),
    ],
)
def test_l1_verdict(gate: ModuleType, gate_job: dict[str, Any], expected: str) -> None:
    """The GPU starts only after pre-commit succeeded."""
    assert gate.l1_verdict([gate_job]) == expected


@pytest.mark.parametrize(
    ("run", "jobs", "expected"),
    [
        # A failed Layer 2 job stops the GPU before the rest of the run ends
        (
            _run("t", "in_progress", None),
            [_job("Test / test: 3.12"), _job("CPU device / x", conclusion="failure")],
            "fail",
        ),
        # A new push or going back to draft cancels Layer 2
        (
            _run("t", "in_progress", None),
            [_job("Test / test: 3.11", conclusion="cancelled")],
            "fail",
        ),
        (
            _run("t", "in_progress", None),
            [_job("Test / test: 3.12"), _job("CPU device / x", "in_progress", None)],
            "wait",
        ),
        (_run("t"), [_job("Test / test: 3.12"), _job("CPU device / x")], "pass"),
    ],
)
def test_l2_verdict(
    gate: ModuleType, run: dict[str, Any], jobs: list[Any], expected: str
) -> None:
    """Layer 2 fails on its first failed job and passes with the whole run."""
    assert gate.l2_verdict(run, jobs) == expected


def test_non_pr_build_needs_no_github(
    gate: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dev and scheduled builds pass both modes without calling GitHub."""

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("non-PR builds must not call the GitHub API")

    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert gate.main(["github_ci_gate.py", "wait-l1"]) == 0
    assert gate.main(["github_ci_gate.py", "watch-l2"]) == 0


def test_unknown_mode_is_a_usage_error(gate: ModuleType) -> None:
    """A typo in the pipeline must not silently pass the gate."""
    assert gate.main(["github_ci_gate.py", "wait-l2"]) == 2


@pytest.mark.skipif(
    shutil.which("setsid") is None or shutil.which("bash") is None,
    reason="needs bash and setsid (Linux)",
)
@pytest.mark.parametrize(
    ("watch_exit", "watch_delay", "test_exit", "test_delay", "expected_exit"),
    [
        # Layer 2 fails while the GPU tests run: stop them, fail the step
        (1, 0.2, 0, 30, 1),
        # Layer 2 passes: the GPU tests' own result counts
        (0, 0.2, 3, 3, 3),
        # The GPU tests finish first: their result counts, the watcher is stopped
        (1, 30, 0, 0.2, 0),
    ],
)
def test_run_gpu_with_l2_watch(
    tmp_path: Path,
    watch_exit: int,
    watch_delay: float,
    test_exit: int,
    test_delay: float,
    expected_exit: int,
) -> None:
    """The wrapper ends the GPU tests early only when Layer 2 fails."""
    wrapper = tmp_path / WRAPPER.name
    shutil.copy(WRAPPER, wrapper)
    (tmp_path / GATE.name).write_text(
        f"import sys, time\ntime.sleep({watch_delay})\nsys.exit({watch_exit})\n",
        encoding="utf-8",
    )
    finished = tmp_path / "tests-finished"
    tests = tmp_path / "tests.sh"
    tests.write_text(
        f"sleep {test_delay}\ntouch {finished}\nexit {test_exit}\n", encoding="utf-8"
    )

    proc = subprocess.run(
        ["bash", str(wrapper), "bash", str(tests)],
        env={**os.environ, "BUILDKITE_PULL_REQUEST": "42"},
        timeout=20,
        check=False,
    )

    assert proc.returncode == expected_exit
    assert finished.exists() is (expected_exit != 1)
