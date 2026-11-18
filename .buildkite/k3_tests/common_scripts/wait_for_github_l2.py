# SPDX-License-Identifier: Apache-2.0
"""Hold a PR's Buildkite GPU steps until the GitHub Layer 2 checks pass.

Layer 1 (pre-commit) and Layer 2 (test.yml, cpu_device.yml) run in the GitHub
Actions "Code Quality" workflow (code_quality_checks.yml) once a PR is ready
for review. This step polls that workflow for the PR's head commit and exits
0 when the newest ready-for-review run succeeded, or 1 when it failed or was
cancelled. The GPU steps sit behind a ``wait``, so they run once per commit
and only after Layer 2 passed, for same-repository and fork PRs alike.
Non-PR builds pass straight through.

Environment: BUILDKITE_PULL_REQUEST, BUILDKITE_COMMIT, BUILDKITE_REPO, and
GITHUB_TOKEN (the API rate limit without a token is shared by the cluster).

Usage: python wait_for_github_l2.py
"""

# Standard
from typing import Any, Callable, Optional
import json
import os
import re
import sys
import time
import urllib.request

API = "https://api.github.com"
WORKFLOW_FILE = "code_quality_checks.yml"
# pre-commit is skipped on draft PRs, so a run where it ran is a ready run.
GATE_JOB = "Check code quality"
POLL_SECONDS = 60

PASS = "pass"
FAIL = "fail"
WAIT = "wait"


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


def decide(
    runs: list[dict[str, Any]],
    gate_job_ran: Callable[[dict[str, Any]], bool],
) -> tuple[str, Optional[dict[str, Any]]]:
    """Decide from the commit's Code Quality runs whether the GPU steps may run.

    Args:
        runs: Workflow runs for the head commit, in any order.
        gate_job_ran: Returns whether ``GATE_JOB`` ran (was not skipped) in a
            run, i.e. whether the run is a ready-for-review run.

    Returns:
        ``(PASS | FAIL | WAIT, run)``, where ``run`` is the newest
        ready-for-review run, or ``None`` while there is none yet.
    """
    for run in sorted(runs, key=lambda r: r["created_at"], reverse=True):
        if not gate_job_ran(run):
            continue
        if run["status"] != "completed":
            return WAIT, run
        return (PASS if run["conclusion"] == "success" else FAIL), run
    return WAIT, None


def _get_json(url: str, token: str) -> Any:
    """GET ``url`` from the GitHub API and decode the JSON body."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    """Poll until Layer 2 has a result for this build's commit.

    Returns:
        0 when Layer 2 passed or this is not a PR build, 1 when it failed.
    """
    pull_request = os.environ.get("BUILDKITE_PULL_REQUEST", "false")
    if pull_request == "false":
        print("Not a pull request build; nothing to wait for.")
        return 0

    repo = repo_slug(os.environ["BUILDKITE_REPO"])
    commit = os.environ["BUILDKITE_COMMIT"]
    token = os.environ.get("GITHUB_TOKEN", "")
    runs_url = (
        f"{API}/repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
        f"?head_sha={commit}&event=pull_request&per_page=20"
    )

    def gate_job_ran(run: dict[str, Any]) -> bool:
        jobs = _get_json(f"{run['jobs_url']}?per_page=100", token)["jobs"]
        return any(
            job["name"] == GATE_JOB and job["conclusion"] != "skipped"
            for job in jobs
        )

    print(f"Waiting for Layer 2 of PR #{pull_request} at {commit} ({repo})")
    while True:
        verdict, run = decide(_get_json(runs_url, token)["workflow_runs"], gate_job_ran)
        if verdict == PASS:
            print(f"Layer 2 passed: {run['html_url']}")
            return 0
        if verdict == FAIL:
            print(
                f"Layer 2 did not pass ({run['conclusion']}): {run['html_url']}\n"
                "GPU tests were not run. Fix the GitHub checks and push again."
            )
            return 1
        state = run["html_url"] if run else "no ready-for-review run yet"
        print(f"Still waiting: {state}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
