# SPDX-License-Identifier: Apache-2.0
"""Tests for the k3-unit-tests step that waits for GitHub Layer 2.

Run without loading the torch-dependent tests/conftest.py:
    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_wait_for_github_l2.py
"""

# Standard
from pathlib import Path
from types import ModuleType
from typing import Any
import importlib.util
import urllib.request

# Third Party
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".buildkite/k3_tests/common_scripts/wait_for_github_l2.py"
)


@pytest.fixture(scope="module")
def waiter() -> ModuleType:
    """Load the wait script, which lives outside any package."""
    spec = importlib.util.spec_from_file_location("wait_for_github_l2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(
    created_at: str,
    ready: bool,
    status: str = "completed",
    conclusion: str | None = "success",
) -> dict[str, Any]:
    """Build a workflow run; ``ready`` marks a run where pre-commit ran."""
    return {
        "created_at": created_at,
        "ready": ready,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/run/{created_at}",
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/LMCache/LMCache.git",
        "https://github.com/LMCache/LMCache",
        "git@github.com:LMCache/LMCache.git",
    ],
)
def test_repo_slug(waiter: ModuleType, url: str) -> None:
    """HTTPS and SSH clone URLs resolve to the owner/name slug."""
    assert waiter.repo_slug(url) == "LMCache/LMCache"


def test_repo_slug_rejects_non_github(waiter: ModuleType) -> None:
    """A non-GitHub URL is an error rather than a wrong API path."""
    with pytest.raises(ValueError):
        waiter.repo_slug("https://gitlab.com/LMCache/LMCache.git")


@pytest.mark.parametrize(
    ("runs", "expected", "expected_created_at"),
    [
        # No run yet (GitHub has not created the ready-for-review run)
        ([], "wait", None),
        # A draft run (pre-commit skipped) passing must not release the GPU
        ([_run("2026-11-18T01:00:00Z", ready=False)], "wait", None),
        (
            [_run("2026-11-18T01:00:00Z", ready=True, status="in_progress")],
            "wait",
            "2026-11-18T01:00:00Z",
        ),
        ([_run("2026-11-18T01:00:00Z", ready=True)], "pass", "2026-11-18T01:00:00Z"),
        (
            [_run("2026-11-18T01:00:00Z", ready=True, conclusion="failure")],
            "fail",
            "2026-11-18T01:00:00Z",
        ),
        # Converting to draft cancels the ready run
        (
            [
                _run("2026-11-18T01:00:00Z", ready=True, conclusion="cancelled"),
                _run("2026-11-18T01:05:00Z", ready=False),
            ],
            "fail",
            "2026-11-18T01:00:00Z",
        ),
        # Ready again on the same commit: the newest ready run decides
        (
            [
                _run("2026-11-18T01:00:00Z", ready=True, conclusion="cancelled"),
                _run("2026-11-18T01:10:00Z", ready=True),
            ],
            "pass",
            "2026-11-18T01:10:00Z",
        ),
    ],
)
def test_decide(
    waiter: ModuleType,
    runs: list[dict[str, Any]],
    expected: str,
    expected_created_at: str | None,
) -> None:
    """Only the newest ready-for-review run gates the GPU steps."""
    verdict, run = waiter.decide(runs, lambda r: r["ready"])
    assert verdict == expected
    assert (run["created_at"] if run else None) == expected_created_at


def test_non_pr_build_skips_the_wait(
    waiter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dev and scheduled builds start the GPU steps without calling GitHub."""

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("non-PR builds must not call the GitHub API")

    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert waiter.main() == 0
