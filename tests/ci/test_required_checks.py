# SPDX-License-Identifier: Apache-2.0
"""Exercise CI admission through its shell interface, without GPUs or GitHub.

Run with ``pytest --confcutdir=tests/ci tests/ci/test_required_checks.py``.
The HTTP fixture serves real curl requests; wrapper tests use temporary Git
history and record uploads instead of starting Buildkite jobs.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import threading

# Third Party
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMMON = Path(".buildkite/k3_tests/common_scripts")
GATE = COMMON / "wait-for-required-checks.sh"
UPLOAD = COMMON / "upload-pipeline.sh"
SHA = "a" * 40
NEW_SHA = "b" * 40
PR_PATH = "/repos/test/repo/pulls/7"
CHECKS_PATH = (
    f"/repos/test/repo/commits/{SHA}/check-runs?filter=all&per_page=100&page=1"
)
STATUSES_PATH = f"/repos/test/repo/commits/{SHA}/statuses?per_page=100&page=1"


def response(body: object, status: int = 200) -> Response:
    """Encode a JSON-serializable body and HTTP status for the API fixture."""
    return Response(status, json.dumps(body))


def checks(
    quality: str = "success", dco: str = "success"
) -> dict[str, list[dict[str, object]]]:
    """Build Check Run records; ``pending`` represents an unfinished run."""
    return {
        "check_runs": [
            {
                "id": number,
                "name": name,
                "status": "in_progress" if conclusion == "pending" else "completed",
                "conclusion": None if conclusion == "pending" else conclusion,
                "check_suite": {"id": 1},
            }
            for number, name, conclusion in (
                (1, "Check code quality", quality),
                (2, "DCO", dco),
                (3, "Detect changes", "success"),
            )
        ]
    }


@pytest.fixture
def api() -> Iterator[GitHubAPI]:
    """Serve queued responses, repeating the last response for each route."""
    fixture = GitHubAPI(
        routes={
            PR_PATH: [response({"state": "open", "head": {"sha": SHA}})],
            CHECKS_PATH: [response(checks())],
            STATUSES_PATH: [response([])],
        },
        requests=[],
        environment={},
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Return the next configured response and record the request."""
            fixture.requests.append(self.path)
            replies = fixture.routes.get(self.path, [response({}, 404)])
            reply = replies.pop(0) if len(replies) > 1 else replies[0]
            self.send_response(reply.status)
            self.end_headers()
            self.wfile.write(reply.body.encode())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Keep expected HTTP failures out of pytest output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    fixture.environment = {
        **os.environ,
        "GITHUB_API_URL": f"http://127.0.0.1:{server.server_port}",
        "GITHUB_TOKEN": "ci-test-token",
        # CI/developer proxy settings must not redirect the local HTTP fixture.
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "REQUIRED_CHECKS_TIMEOUT_SECONDS": "5",
        "REQUIRED_CHECKS_POLL_SECONDS": "1",
    }
    thread.start()
    try:
        yield fixture
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def run_gate(api: GitHubAPI) -> subprocess.CompletedProcess[str]:
    """Run the production gate against the local API and capture its result."""
    return subprocess.run(
        ["bash", str(ROOT / GATE), "test/repo", "7", SHA],
        env=api.environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("quality", ["success", "neutral", "skipped"])
def test_admits_allowed_quality_results(api: GitHubAPI, quality: str) -> None:
    """Only the declared Code Quality exceptions accompany a successful DCO."""
    api.routes[CHECKS_PATH] = [response(checks(quality=quality))]
    result = run_gate(api)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fast checks passed" in result.stdout
    assert api.requests.count(PR_PATH) == 2


@pytest.mark.parametrize("name", ["quality", "dco"])
@pytest.mark.parametrize(
    "conclusion", ["failure", "cancelled", "timed_out", "action_required", "stale"]
)
def test_rejects_terminal_failures(api: GitHubAPI, name: str, conclusion: str) -> None:
    """A failed prerequisite cannot admit downstream work."""
    api.routes[CHECKS_PATH] = [response(checks(**{name: conclusion}))]
    result = run_gate(api)
    assert result.returncode != 0
    assert "refusing admission" in result.stderr


@pytest.mark.parametrize("conclusion", ["neutral", "skipped"])
def test_dco_must_succeed(api: GitHubAPI, conclusion: str) -> None:
    """DCO does not inherit Code Quality's skipped/neutral exceptions."""
    api.routes[CHECKS_PATH] = [response(checks(dco=conclusion))]
    assert run_gate(api).returncode != 0


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped", "pending"])
def test_skipped_quality_requires_successful_path_selection(
    api: GitHubAPI, conclusion: str
) -> None:
    """A failed prerequisite must not turn skipped Code Quality into admission."""
    records = checks(quality="skipped")
    selection = records["check_runs"][2]
    selection["conclusion"] = None if conclusion == "pending" else conclusion
    selection["status"] = "in_progress" if conclusion == "pending" else "completed"
    api.routes[CHECKS_PATH] = [response(records)]
    api.environment["REQUIRED_CHECKS_TIMEOUT_SECONDS"] = "1"
    assert run_gate(api).returncode != 0


def test_skipped_quality_uses_its_own_check_suite(api: GitHubAPI) -> None:
    """Another workflow's path-selection success cannot admit skipped quality."""
    records = checks(quality="skipped")
    records["check_runs"][2]["check_suite"] = {"id": 2}
    api.routes[CHECKS_PATH] = [response(records)]
    api.environment["REQUIRED_CHECKS_TIMEOUT_SECONDS"] = "1"
    assert run_gate(api).returncode != 0


def test_skipped_quality_without_suite_cannot_pass(api: GitHubAPI) -> None:
    """Malformed suite identity cannot associate unrelated path checks."""
    records = checks(quality="skipped")
    del records["check_runs"][0]["check_suite"]
    api.routes[CHECKS_PATH] = [response(records)]
    assert run_gate(api).returncode != 0


def test_waits_for_latest_run_without_legacy_fallback(api: GitHubAPI) -> None:
    """A pending rerun takes precedence over earlier check/status successes."""
    pending = checks(dco="pending")
    pending["check_runs"][1]["id"] = 3
    pending["check_runs"].append(checks()["check_runs"][1])
    api.routes[CHECKS_PATH] = [response(pending), response(checks())]
    api.routes[STATUSES_PATH] = [
        response([{"id": 1, "context": "DCO", "state": "success"}])
    ]
    result = run_gate(api)
    assert result.returncode == 0, result.stdout + result.stderr
    assert api.requests.count(CHECKS_PATH) == 2
    assert STATUSES_PATH not in api.requests


@pytest.mark.parametrize("state", ["success", "failure", "error"])
def test_legacy_dco_status(api: GitHubAPI, state: str) -> None:
    """Legacy statuses are used only when there is no DCO Check Run."""
    quality = checks()["check_runs"][:1]
    api.routes[CHECKS_PATH] = [response({"check_runs": quality})]
    api.routes[STATUSES_PATH] = [
        response([{"id": 1, "context": "DCO", "state": state}])
    ]
    assert (run_gate(api).returncode == 0) == (state == "success")


@pytest.mark.parametrize("pending", ["quality", "dco", "missing"])
def test_wait_is_bounded(api: GitHubAPI, pending: str) -> None:
    """Both missing checks and unfinished checks eventually reject admission."""
    api.environment["REQUIRED_CHECKS_TIMEOUT_SECONDS"] = "1"
    api.routes[CHECKS_PATH] = [
        response(
            {"check_runs": []}
            if pending == "missing"
            else checks(**{pending: "pending"})
        )
    ]
    result = run_gate(api)
    assert result.returncode != 0
    assert "timed out" in result.stderr or "deadline exceeded" in result.stderr


@pytest.mark.parametrize("during_wait", [False, True])
def test_rejects_changed_head(api: GitHubAPI, during_wait: bool) -> None:
    """A changed head is rejected both during waiting and just before admission."""
    api.routes[PR_PATH].append(response({"state": "open", "head": {"sha": NEW_SHA}}))
    if during_wait:
        api.routes[CHECKS_PATH] = [response(checks(dco="pending"))]
    result = run_gate(api)
    assert result.returncode != 0
    assert "stale build" in result.stderr
    assert not any(NEW_SHA in path for path in api.requests)


def test_reads_later_check_pages(api: GitHubAPI) -> None:
    """A required check on page two is not mistaken for a missing check."""
    api.routes[CHECKS_PATH] = [
        response({"check_runs": [{"id": i, "name": "unrelated"} for i in range(100)]})
    ]
    api.routes[CHECKS_PATH.replace("&page=1", "&page=2")] = [response(checks())]
    result = run_gate(api)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reads_later_status_pages(api: GitHubAPI) -> None:
    """Legacy DCO statuses are paginated just like Check Runs."""
    api.routes[CHECKS_PATH] = [response({"check_runs": checks()["check_runs"][:1]})]
    api.routes[STATUSES_PATH] = [
        response([{"id": i, "context": "unrelated"} for i in range(100)])
    ]
    api.routes[STATUSES_PATH.replace("&page=1", "&page=2")] = [
        response([{"id": 101, "context": "DCO", "state": "success"}])
    ]
    result = run_gate(api)
    assert result.returncode == 0, result.stdout + result.stderr


def test_retries_are_bounded(api: GitHubAPI) -> None:
    """A persistent server failure rejects admission after three requests."""
    api.routes[CHECKS_PATH] = [response({}, 503)]
    result = run_gate(api)
    assert result.returncode != 0
    assert api.requests.count(CHECKS_PATH) == 3


def test_closed_pr_is_not_admitted(api: GitHubAPI) -> None:
    """Closing a PR while a build is queued invalidates admission."""
    api.routes[PR_PATH] = [response({"state": "closed", "head": {"sha": SHA}})]
    assert run_gate(api).returncode != 0
    assert CHECKS_PATH not in api.requests


def test_retries_transient_api_failure(api: GitHubAPI) -> None:
    """A temporary server error is retried without admitting an unknown result."""
    api.routes[CHECKS_PATH].insert(0, response({}, 503))
    result = run_gate(api)
    assert result.returncode == 0, result.stdout + result.stderr
    assert api.requests.count(CHECKS_PATH) == 2


@pytest.mark.parametrize("reply", [(403, "{}"), (200, "not JSON"), (200, "{}")])
def test_api_failure_cannot_pass(api: GitHubAPI, reply: tuple[int, str]) -> None:
    """Permission errors and malformed responses fail closed."""
    api.routes[CHECKS_PATH] = [Response(*reply)]
    assert run_gate(api).returncode != 0


def git(repo: Path, *arguments: str) -> str:
    """Run Git in an isolated test repository and return standard output."""
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path, api: GitHubAPI) -> Checkout:
    """Create diverged base/PR history with an untrusted replacement gate."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "dev")
    git(repo, "config", "user.name", "CI test")
    git(repo, "config", "user.email", "ci@example.invalid")
    shutil.copytree(ROOT / COMMON, repo / COMMON)
    (repo / "README.md").write_text("base\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "trusted base")
    git(repo, "remote", "add", "origin", str(repo))
    git(repo, "checkout", "-b", "pr")
    (repo / "lmcache").mkdir()
    (repo / "lmcache/example.py").write_text("value = 1\n")
    (repo / GATE).write_text("touch untrusted-gate-ran\nexit 0\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "PR with modified gate")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "dev")
    (repo / "README.md").write_text("base advanced\n")
    git(repo, "commit", "-am", "advance base")
    git(repo, "checkout", "pr")

    # Redirect only the pipeline's GitHub fetch to the local base repository.
    # All revision resolution, merging, filtering and git-show remain real.
    binaries = tmp_path / "bin"
    binaries.mkdir()
    real_git = shutil.which("git")
    if real_git is None:
        pytest.fail("Git is required for wrapper tests")
    git_shim = binaries / "git"
    git_shim.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        f"local_repo = {str(repo)!r}\n"
        "args = [local_repo if arg == 'https://github.com/test/repo.git'\n"
        "        else arg for arg in args]\n"
        f"os.execv({real_git!r}, [{real_git!r}, *args])\n"
    )
    git_shim.chmod(0o755)
    upload_log = tmp_path / "uploads"
    agent = binaries / "buildkite-agent"
    agent.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$UPLOAD_LOG"\n')
    agent.chmod(0o755)
    environment = {
        **api.environment,
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "UPLOAD_LOG": str(upload_log),
        "BUILDKITE_REPO": "https://github.com/test/repo.git",
        "BUILDKITE_COMMIT": head,
        "BUILDKITE_PULL_REQUEST": "7",
        "BUILDKITE_PULL_REQUEST_BASE_BRANCH": "dev",
        "BUILDKITE_PULL_REQUEST_LABELS": "force-ci",
        "LMCACHE_PR_BASE_MERGE": "auto",
        "XPU_CANDIDATE_VALIDATION": "0",
    }
    api.routes[PR_PATH] = [response({"state": "open", "head": {"sha": head}})]
    api.routes[CHECKS_PATH.replace(SHA, head)] = api.routes[CHECKS_PATH]
    return Checkout(repo, head, environment, upload_log)


def run_upload(
    checkout: Checkout, pipeline: str = ".buildkite/k3_tests/unit/pipeline.yml"
) -> subprocess.CompletedProcess[str]:
    """Execute the real upload wrapper with only the agent command replaced."""
    return subprocess.run(
        ["bash", str(checkout.repo / UPLOAD), pipeline],
        cwd=checkout.repo,
        env=checkout.environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("conclusion", ["success", "failure"])
@pytest.mark.parametrize(
    "repository_url",
    ["https://github.com/test/repo.git", "git@github.com:test/repo.git"],
)
def test_wrapper_uses_trusted_gate_and_original_sha(
    api: GitHubAPI, checkout: Checkout, conclusion: str, repository_url: str
) -> None:
    """The PR cannot replace the gate, and a base merge cannot change its SHA."""
    api.routes[CHECKS_PATH.replace(SHA, checkout.head)] = [
        response(checks(dco=conclusion))
    ]
    checkout.environment["BUILDKITE_REPO"] = repository_url
    result = run_upload(checkout)
    assert (result.returncode == 0) == (conclusion == "success"), result.stderr
    assert checkout.upload_log.exists() == (conclusion == "success")
    assert not (checkout.repo / "untrusted-gate-ran").exists()
    assert git(checkout.repo, "rev-parse", "HEAD") != checkout.head


def test_wrapper_rejects_a_newer_pr_head(api: GitHubAPI, checkout: Checkout) -> None:
    """An old queued build cannot use the newer head's passing checks."""
    api.routes[PR_PATH] = [response({"state": "open", "head": {"sha": NEW_SHA}})]
    result = run_upload(checkout)
    assert result.returncode != 0
    assert "stale build" in result.stderr
    assert not checkout.upload_log.exists()


def test_wrapper_requires_gate_in_trusted_base(checkout: Checkout) -> None:
    """Bootstrap never falls back to executing a gate supplied by the PR."""
    git(checkout.repo, "checkout", "dev")
    git(checkout.repo, "rm", str(GATE))
    git(checkout.repo, "commit", "-m", "base without gate")
    git(checkout.repo, "checkout", "pr")
    # Avoid an unrelated modify/delete merge conflict in this bootstrap case.
    checkout.environment["LMCACHE_PR_BASE_MERGE"] = "never"
    result = run_upload(checkout)
    assert result.returncode != 0
    assert "trusted base has no fast-check gate" in result.stderr
    assert not checkout.upload_log.exists()
    assert not (checkout.repo / "untrusted-gate-ran").exists()


def test_non_pr_build_does_not_query_checks(api: GitHubAPI, checkout: Checkout) -> None:
    """Push/nightly builds retain their existing upload behavior."""
    checkout.environment["BUILDKITE_PULL_REQUEST"] = "false"
    result = run_upload(checkout)
    assert result.returncode == 0, result.stderr
    assert "pipeline upload" in checkout.upload_log.read_text()
    assert api.requests == []


def test_expensive_label_skip_does_not_wait(api: GitHubAPI, checkout: Checkout) -> None:
    """An existing label skip finishes without a gate or downstream upload."""
    checkout.environment["BUILDKITE_PULL_REQUEST_LABELS"] = "skip expensive ci"
    result = run_upload(checkout, ".buildkite/k3_tests/integration/pipeline.yml")
    assert result.returncode == 0, result.stderr
    assert "pipeline upload" not in checkout.upload_log.read_text()
    assert api.requests == []


def test_docs_only_build_skips_without_querying_checks(
    api: GitHubAPI, checkout: Checkout
) -> None:
    """The existing path skip remains ahead of admission and GPU upload."""
    git(checkout.repo, "checkout", "-b", "docs", "dev")
    (checkout.repo / "README.md").write_text("docs-only PR\n")
    git(checkout.repo, "commit", "-am", "docs-only change")
    checkout.environment["BUILDKITE_COMMIT"] = git(checkout.repo, "rev-parse", "HEAD")
    checkout.environment["BUILDKITE_PULL_REQUEST_LABELS"] = ""
    result = run_upload(checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pipeline upload" not in checkout.upload_log.read_text()
    assert api.requests == []


@pytest.mark.parametrize("workflow", ["pr_ci.yml", "pr_full_build.yml"])
def test_all_downstream_calls_are_gated_and_summarized(workflow: str) -> None:
    """New workflow calls must participate in admission and final reporting."""
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    jobs = document["jobs"]
    assert set(jobs["summary"]["needs"]) == set(jobs) - {"summary"}
    assert "always()" in jobs["summary"]["if"]
    for name, job in jobs.items():
        if "uses" in job and name != "gate":
            needs = job["needs"]
            assert "gate" in (needs if isinstance(needs, list) else [needs]), name


@pytest.mark.skipif(
    not os.environ.get("CI_PICOMATCH_PATH"),
    reason="Set CI_PICOMATCH_PATH to picomatch@2.3.1 to test path selection",
)
@pytest.mark.parametrize(
    ("filename", "selected"),
    [
        ("lmcache/cli/main.py", ["build_cli"]),
        ("lmcache/cli/commands/server.py", ["build_cli"]),
        ("lmcache/cli/README.md", []),
        (".github/workflows/test.yml", ["actionlint"]),
        (".github/workflows/example.yaml", ["actionlint"]),
        (".github/actions/example.yml", ["actionlint"]),
        (".github/workflows/actionlint.dockerfile", ["actionlint"]),
        (".github/actions/build-artifacts/action.yml", ["build_cpu", "build_cli"]),
        ("csrc/deep/kernel.cu", ["build_cpu"]),
        ("requirements/build.txt", ["build_cpu"]),
        ("pyproject_cli.toml", ["build_cli"]),
        ("README.md", []),
    ],
)
def test_pr_path_selection(filename: str, selected: list[str]) -> None:
    """Preserve native path-filter coverage using dorny's actual glob engine."""
    document = yaml.safe_load((ROOT / ".github/workflows/pr_ci.yml").read_text())
    filters = yaml.safe_load(document["jobs"]["changes"]["steps"][0]["with"]["filters"])
    result = subprocess.run(
        [
            "node",
            "-e",
            "const picomatch = require(process.env.CI_PICOMATCH_PATH);"
            "const {filters, filename} = JSON.parse(require('fs').readFileSync(0));"
            "const selected = Object.entries(filters).filter(([, patterns]) =>"
            "  picomatch(patterns, {dot: true})(filename)).map(([name]) => name);"
            "process.stdout.write(JSON.stringify(selected));",
        ],
        input=json.dumps({"filters": filters, "filename": filename}),
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert json.loads(result.stdout) == selected


def run_summary(
    workflow: str, results: dict[str, object], requested: bool = False
) -> subprocess.CompletedProcess[str]:
    """Execute the summary's actual shell predicate with simulated job results."""
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    script = document["jobs"]["summary"]["steps"][0]["run"]
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env={
            **os.environ,
            "NEEDS": json.dumps(results),
            "REQUESTED": json.dumps(requested),
        },
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_pr_summary_requires_gate_success(result: str) -> None:
    """Skipped dependents cannot turn a failed/cancelled gate into green CI."""
    results = {
        name: {"result": "success"}
        for name in ["gate", "changes", "test", "cpu", "operator"]
    }
    results["gate"]["result"] = result
    optional = ["actionlint", "azure", "aerospike", "build-cpu", "build-cli", "codeql"]
    all_results: dict[str, object] = {
        **results,
        **{name: {"result": "skipped"} for name in optional},
    }
    all_results["changes"] = {
        "result": "success",
        "outputs": dict.fromkeys(optional, "false"),
    }
    assert (run_summary("pr_ci.yml", all_results).returncode == 0) == (
        result == "success"
    )


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_selected_artifact_build_must_succeed(result: str) -> None:
    """Path-selected CPU/CLI builds are required even without the full label."""
    results: dict[str, object] = {
        name: {"result": "success"} for name in ["gate", "test", "cpu", "operator"]
    }
    optional = ["actionlint", "azure", "aerospike", "build-cpu", "build-cli", "codeql"]
    outputs = dict.fromkeys(optional, "false")
    outputs.update({"build-cpu": "true", "build-cli": "true"})
    results.update({name: {"result": "skipped"} for name in optional})
    results.update({"build-cpu": {"result": result}, "build-cli": {"result": result}})
    results["changes"] = {"result": "success", "outputs": outputs}
    assert (run_summary("pr_ci.yml", results).returncode == 0) == (result == "success")


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_full_summary_distinguishes_unrequested_builds(
    requested: bool, result: str
) -> None:
    """Only an unrequested full build may have skipped gate/build jobs."""
    document = yaml.safe_load(
        (ROOT / ".github/workflows/pr_full_build.yml").read_text()
    )
    results: dict[str, object] = {
        name: {"result": result} for name in document["jobs"]["summary"]["needs"]
    }
    expected = "success" if requested else "skipped"
    assert (run_summary("pr_full_build.yml", results, requested).returncode == 0) == (
        result == expected
    )


@pytest.mark.parametrize(
    ("workflow", "failed_job"),
    [("pr_ci.yml", "test"), ("pr_full_build.yml", "build-main")],
)
def test_summary_rejects_one_failed_downstream_job(
    workflow: str, failed_job: str
) -> None:
    """A single failed task must fail the summary even if every other task passes."""
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    results: dict[str, object] = {
        name: {"result": "success"} for name in document["jobs"]["summary"]["needs"]
    }
    if workflow == "pr_ci.yml":
        results["changes"] = {
            "result": "success",
            "outputs": dict.fromkeys(
                [
                    "actionlint",
                    "azure",
                    "aerospike",
                    "build-cpu",
                    "build-cli",
                    "codeql",
                ],
                "true",
            ),
        }
    assert run_summary(workflow, results, requested=True).returncode == 0
    results[failed_job] = {"result": "failure"}
    assert run_summary(workflow, results, requested=True).returncode != 0


@dataclass
class Response:
    """One HTTP response used by the local GitHub fixture."""

    status: int
    body: str


@dataclass
class GitHubAPI:
    """API routes, recorded requests, and the environment passed to the gate."""

    routes: dict[str, list[Response]]
    requests: list[str]
    environment: dict[str, str]


@dataclass
class Checkout:
    """Isolated Buildkite checkout and the immutable commit it was created for."""

    repo: Path
    head: str
    environment: dict[str, str]
    upload_log: Path
