# SPDX-License-Identifier: Apache-2.0
"""Tests for the weekly full build's pass/fail gate.

Run without loading the torch-dependent tests/conftest.py:
    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_weekly_full_build.py
"""

# Standard
from pathlib import Path
from types import ModuleType
from typing import Any
import importlib.util
import json
import os
import re
import subprocess

# Third Party
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github/scripts/weekly_full_build.py"
MANIFEST = REPO / ".buildkite/k3_tests/weekly/manifest.json"
MP_PIPELINE = REPO / ".buildkite/k3_tests/multiprocess/pipeline.yml"
UPLOAD = REPO / ".buildkite/k3_tests/common_scripts/upload-pipeline.sh"


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    """Load the gate script, which lives outside any package."""
    spec = importlib.util.spec_from_file_location("weekly_full_build", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())


def _job(name: str, state: str = "passed", **fields: Any) -> dict[str, Any]:
    return {"type": "script", "name": name, "state": state, "id": name, **fields}


def _label_for(pattern: str) -> str:
    """Turn a required-label pattern into a label it matches."""
    label = (
        pattern.lstrip("^")
        .rstrip("$")
        .replace("(?!preemption )[a-z_]+", "lmcache_driven")
    )
    if label.startswith(":"):
        return label
    if pattern.startswith("^") and pattern.endswith("$"):
        return re.sub(r"\\([()+])", r"\1", label)
    return f":compression: vllm{label}zmq"


def _passing_builds(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Builds in which every required job passed."""
    builds = {}
    for number, spec in enumerate(manifest["pipelines"], start=1):
        jobs = [_job(":pipeline: Upload pipeline")]
        jobs += [_job(_label_for(p)) for p in spec["required"]]
        builds[spec["slug"]] = {
            "state": "passed",
            "number": number,
            "web_url": f"https://buildkite.com/lmcache/{spec['slug']}/builds/{number}",
            "jobs": jobs,
        }
    return builds


def _kinds(report: dict[str, Any]) -> set[str]:
    return {p["kind"] for entry in report["pipelines"] for p in entry["problems"]}


def test_complete_passing_run_passes(gate: ModuleType, manifest: dict) -> None:
    report = gate.evaluate(manifest, _passing_builds(manifest), {})
    assert report["passed"], report


@pytest.mark.parametrize(
    ("change", "kind"),
    [
        ({"state": "failed", "exit_status": 1}, "failed"),
        ({"state": "timed_out"}, "failed"),
        ({"soft_failed": True}, "soft_failed"),
        ({"state": "broken"}, "not_run"),
    ],
)
def test_required_job_that_did_not_pass_fails_the_run(
    gate: ModuleType, manifest: dict, change: dict, kind: str
) -> None:
    builds = _passing_builds(manifest)
    builds["k3-multiprocess-test"]["jobs"][1].update(change)
    report = gate.evaluate(manifest, builds, {})
    assert not report["passed"]
    assert kind in _kinds(report)


@pytest.mark.parametrize(
    ("slug", "jobs", "context"),
    [
        ("k3-unit-tests", [_job(":pipeline: Upload pipeline")], "path-filter-skip"),
        (
            "k3-multiprocess-test",
            [
                _job(":fast_forward: Skip expensive multiprocess CI"),
                _job(":pipeline: Upload pipeline", "broken"),
            ],
            "skip expensive ci",
        ),
    ],
)
def test_skipped_suite_fails_even_when_the_build_is_green(
    gate: ModuleType,
    manifest: dict,
    slug: str,
    jobs: list[dict[str, Any]],
    context: str,
) -> None:
    builds = _passing_builds(manifest)
    builds[slug]["jobs"] = jobs
    annotations = {slug: [{"context": context, "body_html": "Skipped"}]}
    report = gate.evaluate(manifest, builds, annotations)
    assert not report["passed"]
    assert {"missing", "suite_skipped"} <= _kinds(report)


def test_unsupported_workload_skip_fails_unless_expected(
    gate: ModuleType, manifest: dict
) -> None:
    annotations = {
        "k3-multiprocess-test": [
            {
                "context": "weekly-skip-123",
                "body_html": "<p>Skipped as unsupported: "
                "<code>:compression: sglang / sleep_mode / zmq</code></p>",
            }
        ]
    }
    builds = _passing_builds(manifest)
    assert not gate.evaluate(manifest, builds, annotations)["passed"]

    expected = dict(manifest, expected_unsupported=["sglang / sleep_mode /"])
    assert gate.evaluate(expected, builds, annotations)["passed"]


def test_manifest_allows_expected_sglang_unsupported_skips(
    gate: ModuleType, manifest: dict
) -> None:
    builds = _passing_builds(manifest)
    annotations = {
        "k3-multiprocess-test": [
            {
                "context": f"weekly-skip-{i}",
                "body_html": (
                    "<p>Skipped as unsupported: "
                    f"<code>:compression: sglang / {workload} / grpc</code></p>"
                ),
            }
            for i, workload in enumerate(
                [
                    "deadlock",
                    "long_doc_qa_l2",
                    "lm_eval_preemption / lmcache_driven",
                    "lm_eval_preemption / engine_driven",
                ]
            )
        ]
    }
    assert gate.evaluate(manifest, builds, annotations)["passed"]


@pytest.mark.parametrize(
    "label",
    [
        ":test_tube: Verify SGLang nightly / zmq",
        ":pushpin: Pin verified vLLM nightly",
        ":fast_forward: Skip expensive multiprocess CI",
    ],
)
def test_allowed_not_run_steps_may_stay_broken(
    gate: ModuleType, manifest: dict, label: str
) -> None:
    builds = _passing_builds(manifest)
    builds["k3-multiprocess-test"]["jobs"].append(_job(label, "broken"))
    assert gate.evaluate(manifest, builds, {})["passed"]


def test_pass_after_retry_counts_but_is_reported(
    gate: ModuleType, manifest: dict
) -> None:
    builds = _passing_builds(manifest)
    flaky = builds["k3-multiprocess-test"]["jobs"][1]
    builds["k3-multiprocess-test"]["jobs"].append(
        dict(flaky, state="failed", retried=True, id="first-attempt")
    )
    report = gate.evaluate(manifest, builds, {})
    assert report["passed"]
    retried = {label for p in report["pipelines"] for label in p["retried"]}
    assert retried == {flaky["name"]}


@pytest.mark.parametrize("build_state", ["failed", "canceled", "running", None])
def test_build_that_did_not_pass_fails_the_run(
    gate: ModuleType, manifest: dict, build_state: str | None
) -> None:
    builds = _passing_builds(manifest)
    if build_state is None:
        del builds["k3-blend-test"]
    else:
        builds["k3-blend-test"]["state"] = build_state
    report = gate.evaluate(manifest, builds, {})
    assert not report["passed"]
    assert "build" in _kinds(report)


@pytest.fixture
def docs_only_push(tmp_path: Path) -> dict[str, Any]:
    """A git checkout whose last commit only touches docs, plus a fake agent.

    The fake ``buildkite-agent`` records uploaded pipeline files in
    ``uploads.txt`` instead of talking to Buildkite.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    (repo / "code.py").write_text("x = 1\n")
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "code"], check=True)
    (repo / "README.md").write_text("docs\n")
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "docs"], check=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    agent = bin_dir / "buildkite-agent"
    uploads = tmp_path / "uploads.txt"
    agent.write_text(
        f'#!/usr/bin/env bash\n[[ "$1 $2" == "pipeline upload" ]] '
        f'&& echo "$3" >> {uploads}\nexit 0\n'
    )
    agent.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "BUILDKITE_PULL_REQUEST": "false",
    }
    return {"repo": repo, "env": env, "uploads": uploads}


def _upload(checkout: dict[str, Any], **env: str) -> list[str]:
    """Run upload-pipeline.sh for the unit suite; return uploaded files."""
    subprocess.run(
        ["bash", str(UPLOAD), ".buildkite/k3_tests/unit/pipeline.yml"],
        cwd=checkout["repo"],
        env={**checkout["env"], **env},
        check=True,
        capture_output=True,
    )
    uploads = checkout["uploads"]
    return uploads.read_text().split() if uploads.exists() else []


def test_weekly_build_runs_suites_the_path_filter_would_skip(
    docs_only_push: dict[str, Any],
) -> None:
    assert _upload(docs_only_push) == []
    assert _upload(docs_only_push, CI_WEEKLY_FULL="1") == [
        ".buildkite/k3_tests/unit/pipeline.yml"
    ]


def test_push_build_of_the_weekly_branch_uploads_nothing(
    docs_only_push: dict[str, Any],
) -> None:
    # A code change would normally run; the push of ci-weekly-full must not.
    (docs_only_push["repo"] / "code.py").write_text("x = 2\n")
    git = ["git", "-C", str(docs_only_push["repo"]), "-c", "user.name=t"]
    subprocess.run(git + ["-c", "user.email=t@t", "commit", "-qam", "code"], check=True)
    assert _upload(docs_only_push, BUILDKITE_BRANCH="ci-weekly-full") == []
    assert _upload(docs_only_push, BUILDKITE_BRANCH="dev") == [
        ".buildkite/k3_tests/unit/pipeline.yml"
    ]


def test_every_multiprocess_workload_is_required(manifest: dict) -> None:
    """A workload added to pipeline.yml must also be added to the manifest."""
    workloads = set(re.findall(r"run\.sh ([a-z0-9_]+)", MP_PIPELINE.read_text()))
    spec = next(p for p in manifest["pipelines"] if p["slug"] == "k3-multiprocess-test")
    required = " ".join(spec["required"])
    # lm_eval runs as lm_eval_<mode>; the pin canaries reuse vllm_bench/lm_eval.
    unlisted = {w for w in workloads if w not in required and f"{w}_" not in required}
    assert not unlisted, f"add these workloads to {MANIFEST.name}: {sorted(unlisted)}"


@pytest.mark.parametrize(
    "manifest_rel",
    [
        ".buildkite/k3_tests/weekly/manifest-xpu.json",
        ".buildkite/k3_tests/weekly/manifest-amd.json",
        ".buildkite/k3_tests/weekly/manifest-rbln.json",
    ],
)
def test_complete_vendor_passing_run_passes(
    gate: ModuleType, manifest_rel: str
) -> None:
    vendor_manifest = json.loads((REPO / manifest_rel).read_text(encoding="utf-8"))
    report = gate.evaluate(vendor_manifest, _passing_builds(vendor_manifest), {})
    assert report["passed"], report


def _expand_step_labels(pipeline: dict[str, Any]) -> list[str]:
    """Expand Buildkite pipeline step/group labels across matrix combinations."""
    # Standard
    import itertools

    labels: list[str] = []
    for item in pipeline["steps"]:
        steps = item.get("steps", [item]) if "steps" in item else [item]
        for step in steps:
            raw = step["label"]
            setup = step.get("matrix", {}).get("setup", {})
            if not setup:
                labels.append(raw)
                continue
            keys = list(setup.keys())
            for combo in itertools.product(*(setup[k] for k in keys)):
                lbl = raw
                for k, v in zip(keys, combo, strict=True):
                    lbl = lbl.replace(f"{{{{matrix.{k}}}}}", str(v))
                labels.append(lbl)
    return labels


@pytest.mark.parametrize(
    ("manifest_rel", "slug", "pipeline_rel"),
    [
        (
            ".buildkite/k3_tests/weekly/manifest-xpu.json",
            "unit-tests-xpu",
            ".buildkite/k3_tests/xpu/unittests/pipeline.yml",
        ),
        (
            ".buildkite/k3_tests/weekly/manifest-xpu.json",
            "xpu-mp-test",
            ".buildkite/k3_tests/xpu/multiprocess/pipeline.yml",
        ),
        (
            ".buildkite/k3_tests/weekly/manifest-amd.json",
            "amd-mp-test",
            ".buildkite/k3_tests/amd/pipeline.yml",
        ),
        (
            ".buildkite/k3_tests/weekly/manifest-rbln.json",
            "rbln-mp-test",
            ".buildkite/k3_tests/rbln/pipeline.yml",
        ),
    ],
)
def test_every_vendor_job_is_required(
    manifest_rel: str, slug: str, pipeline_rel: str
) -> None:
    """Every expanded job in a vendor pipeline must match its weekly manifest."""
    # Third Party
    import yaml

    vendor_manifest = json.loads((REPO / manifest_rel).read_text(encoding="utf-8"))
    pipeline = yaml.safe_load((REPO / pipeline_rel).read_text(encoding="utf-8"))
    labels = _expand_step_labels(pipeline)

    spec = next(p for p in vendor_manifest["pipelines"] if p["slug"] == slug)
    patterns = [re.compile(p) for p in spec["required"]]
    unmatched = [lbl for lbl in labels if not any(p.search(lbl) for p in patterns)]
    unused = [p.pattern for p in patterns if not any(p.search(lbl) for lbl in labels)]
    assert not unmatched, f"add these jobs to {manifest_rel} ({slug}): {unmatched}"
    assert not unused, f"unused patterns in {manifest_rel} ({slug}): {unused}"


def test_fetch_annotations_paginates(
    gate: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = {
        1: [{"id": i} for i in range(100)],
        2: [{"id": 100}, {"id": 101}],
    }
    seen_urls: list[str] = []

    def fake_api_request(url: str, token: str) -> list[dict[str, int]]:
        seen_urls.append(url)
        page = int(url.rsplit("page=", 1)[1])
        return pages[page]

    monkeypatch.setattr(gate, "api_request", fake_api_request)
    items = gate.fetch_annotations("https://api.buildkite.com/v2/b/1", "tok")
    assert len(items) == 102
    assert seen_urls == [
        "https://api.buildkite.com/v2/b/1/annotations?per_page=100&page=1",
        "https://api.buildkite.com/v2/b/1/annotations?per_page=100&page=2",
    ]


def test_preemption_does_not_satisfy_ordinary_lm_eval(
    gate: ModuleType, manifest: dict
) -> None:
    builds = _passing_builds(manifest)
    builds["k3-multiprocess-test"]["jobs"] = [
        j
        for j in builds["k3-multiprocess-test"]["jobs"]
        if "lm_eval_lmcache_driven" not in j["name"]
    ]
    report = gate.evaluate(manifest, builds, {})
    assert not report["passed"]
    assert "missing" in _kinds(report)


def test_api_request_retries_read_timeout(
    gate: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    class FakeResp:
        def __enter__(self) -> "FakeResp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TimeoutError("timed out")
            return b'{"ok": true}'

    monkeypatch.setattr(gate, "urlopen", lambda req, timeout=60: FakeResp())
    monkeypatch.setattr(gate.time, "sleep", lambda _: None)
    assert gate.api_request("https://example.test", "tok") == {"ok": True}
    assert attempts == 3
