# SPDX-License-Identifier: Apache-2.0
"""Tie a PR's Buildkite GPU steps to the GitHub Layer 1 and Layer 2 checks.

Layer 1 (pre-commit) and Layer 2 (test.yml, cpu_device.yml) run in the GitHub
Actions "Code Quality" workflow (code_quality_checks.yml) once a PR is ready
for review. Two modes poll the newest ready-for-review run of the PR's head
commit:

``wait-l1``
    Exit 0 once Layer 1 passed, 1 if it failed or the run was cancelled. Runs
    before the GPU steps, so the GPU starts only for code that passed Layer 1
    and, for first-time fork contributors, only after a maintainer approved
    the GitHub run.
``watch-l2``
    Runs next to the GPU tests. Exit 0 once Layer 2 passed, 1 as soon as a
    Layer 2 job fails or is cancelled, so run-gpu-with-l2-watch.sh can stop
    the GPU tests early.

Non-PR builds exit 0 straight away. GitHub API errors are retried, so a
network glitch never fails or stops the GPU tests.

Environment: BUILDKITE_PULL_REQUEST, BUILDKITE_COMMIT, BUILDKITE_REPO, and
GITHUB_API_TOKEN. The token only lifts the API rate limit, which the cluster
otherwise shares. PR code runs in the same pods, so it must have no scopes;
never pass the repository write token from buildkite-git-creds here.

Usage: python github_ci_gate.py {wait-l1|watch-l2}
"""

# Standard
from typing import Any, Callable, Optional
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://api.github.com"
WORKFLOW_FILE = "code_quality_checks.yml"
# pre-commit is skipped on draft PRs, so a run where it ran is a ready run.
GATE_JOB = "Check code quality"
# Jobs of the reusable workflows code_quality_checks.yml calls as Layer 2.
L2_JOB_PREFIXES = ("Test / ", "CPU device / ")
FAILED = frozenset({"failure", "cancelled", "timed_out", "startup_failure"})
POLL_SECONDS = 60

WAIT_L1 = "wait-l1"
WATCH_L2 = "watch-l2"

PASS = "pass"
FAIL = "fail"
WAIT = "wait"

Job = dict[str, Any]
Run = dict[str, Any]


def repo_slug(buildkite_repo: str) -> str:
    """Return ``owner/name`` from a GitHub clone URL.

    Args:
        buildkite_repo: ``BUILDKITE_REPO``, e.g.
            ``https://github.com/LMCache/LMCache.git`` or
            ``git@github.com:LMCache/LMCache.git``.

    Returns:
        The repository slug, e.g. ``LMCache/LMCache``.

    Raises:
        ValueError: If the URL is not a GitHub repository URL.
    """
    match = re.search(r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?/?$", buildkite_repo)
    if match is None:
        raise ValueError(f"not a GitHub repository URL: {buildkite_repo}")
    return match.group(1)


def is_ready_run(jobs: list[Job]) -> bool:
    """Return whether a run's jobs belong to a ready-for-review run.

    Args:
        jobs: The run's jobs from the GitHub API.

    Returns:
        True when the ``GATE_JOB`` (pre-commit) was not skipped.
    """
    return any(
        job["name"] == GATE_JOB and job["conclusion"] != "skipped" for job in jobs
    )


def newest_ready_run(
    runs: list[Run], jobs_of: Callable[[Run], list[Job]]
) -> Optional[tuple[Run, list[Job]]]:
    """Return the newest ready-for-review run and its jobs.

    Args:
        runs: Code Quality runs for the head commit, in any order.
        jobs_of: Fetches a run's jobs.

    Returns:
        ``(run, jobs)``, or None while there is no ready-for-review run yet,
        e.g. before GitHub created it or while a fork run awaits approval.
    """
    for run in sorted(runs, key=lambda r: r["created_at"], reverse=True):
        jobs = jobs_of(run)
        if is_ready_run(jobs):
            return run, jobs
    return None


def l1_verdict(jobs: list[Job]) -> str:
    """Return PASS, FAIL or WAIT for Layer 1 of a ready-for-review run."""
    gate = next(job for job in jobs if job["name"] == GATE_JOB)
    if gate["status"] != "completed":
        return WAIT
    return PASS if gate["conclusion"] == "success" else FAIL


def l2_verdict(run: Run, jobs: list[Job]) -> str:
    """Return PASS, FAIL or WAIT for Layer 2 of a ready-for-review run.

    A Layer 2 job that failed or was cancelled (a new push, or the PR going
    back to draft) fails Layer 2 before the rest of the run finishes.
    """
    if any(
        job["name"].startswith(L2_JOB_PREFIXES) and job["conclusion"] in FAILED
        for job in jobs
    ):
        return FAIL
    if run["status"] != "completed":
        return WAIT
    return PASS if run["conclusion"] == "success" else FAIL


def _get_json(url: str, token: str) -> Any:
    """GET ``url`` from the GitHub API and decode the JSON body."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main(argv: list[str]) -> int:
    """Poll the commit's Code Quality runs until the mode has a verdict.

    Args:
        argv: ``[program, mode]`` with mode ``wait-l1`` or ``watch-l2``.

    Returns:
        0 when the layer passed or this is not a PR build, 1 when it failed,
        2 on a usage error.
    """
    if len(argv) != 2 or argv[1] not in (WAIT_L1, WATCH_L2):
        print(f"usage: {argv[0]} {{{WAIT_L1}|{WATCH_L2}}}", file=sys.stderr)
        return 2
    mode = argv[1]
    layer = "Layer 1" if mode == WAIT_L1 else "Layer 2"

    pull_request = os.environ.get("BUILDKITE_PULL_REQUEST", "false")
    if pull_request == "false":
        print("Not a pull request build; no GitHub layer to follow.")
        return 0

    repo = repo_slug(os.environ["BUILDKITE_REPO"])
    commit = os.environ["BUILDKITE_COMMIT"]
    token = os.environ.get("GITHUB_API_TOKEN", "")
    runs_url = (
        f"{API}/repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
        f"?head_sha={commit}&event=pull_request&per_page=20"
    )

    def jobs_of(run: Run) -> list[Job]:
        return _get_json(f"{run['jobs_url']}?per_page=100", token)["jobs"]

    print(f"Following {layer} of PR #{pull_request} at {commit} ({repo})")
    while True:
        try:
            found = newest_ready_run(
                _get_json(runs_url, token)["workflow_runs"], jobs_of
            )
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"GitHub API error, retrying: {error}")
            time.sleep(POLL_SECONDS)
            continue

        if found is None:
            print("Still waiting: no ready-for-review run yet")
        else:
            run, jobs = found
            verdict = l1_verdict(jobs) if mode == WAIT_L1 else l2_verdict(run, jobs)
            if verdict == PASS:
                print(f"{layer} passed: {run['html_url']}")
                return 0
            if verdict == FAIL:
                print(f"{layer} did not pass: {run['html_url']}")
                return 1
            print(f"Still waiting: {run['html_url']}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
