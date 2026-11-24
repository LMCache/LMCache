# SPDX-License-Identifier: Apache-2.0
"""Tests for the k3-multiprocess PR-smoke and heavy tier split.

Run without loading the torch-dependent tests/conftest.py:
    python -m pytest -q --confcutdir=tests/ci tests/ci/test_multiprocess_split.py
"""

# Standard
from pathlib import Path
from typing import Any
import os
import shutil
import subprocess

# Third Party
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MP_DIR = REPO_ROOT / ".buildkite" / "k3_tests" / "multiprocess"
MANIFEST_PATH = MP_DIR / "split_manifest.yaml"
SMOKE_PIPELINE_PATH = MP_DIR / "smoke-pipeline.yml"
HEAVY_PIPELINE_PATH = MP_DIR / "pipeline.yml"
UPLOAD_SCRIPT = (
    REPO_ROOT / ".buildkite" / "k3_tests" / "common_scripts" / "upload-pipeline.sh"
)
PATH_FILTER_SCRIPT = (
    REPO_ROOT / ".buildkite" / "k3_tests" / "common_scripts" / "path-filter.sh"
)

REQUIRED_MANIFEST_FIELDS = {
    "workload",
    "gpu_requirement",
    "tier",
    "trigger",
    "coverage_objective",
    "acceptance_check",
}


def _collect_steps(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten top-level and grouped Buildkite steps."""
    collected: list[dict[str, Any]] = []
    for item in doc.get("steps", []):
        if isinstance(item, dict) and "steps" in item:
            collected.extend(item["steps"])
        elif isinstance(item, dict):
            collected.append(item)
    return collected


def test_split_manifest_schema_and_tiers() -> None:
    """Every manifest workload defines all required fields and a valid tier."""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert set(manifest["tiers"].keys()) == {"pr-smoke", "heavy"}

    workloads = manifest["workloads"]
    assert len(workloads) >= 18
    for entry in workloads:
        assert REQUIRED_MANIFEST_FIELDS.issubset(entry.keys())
        assert entry["tier"] in {"pr-smoke", "heavy"}

    pr_smoke_names = {w["workload"] for w in workloads if w["tier"] == "pr-smoke"}
    assert pr_smoke_names == {
        "cpu_e2e_validation",
        "cache_stats",
        "lazy_offload",
        "http_api",
        "restart_recovery",
        "lm_eval",
    }


def test_smoke_pipeline_only_contains_0_or_1_gpu_pr_smoke() -> None:
    """smoke-pipeline.yml must stay fast (0/1-GPU) and independent of heavy."""
    smoke = yaml.safe_load(SMOKE_PIPELINE_PATH.read_text(encoding="utf-8"))
    steps = _collect_steps(smoke)
    command_steps = [s for s in steps if "command" in s]
    assert len(command_steps) == 6

    trigger_steps = [s for s in steps if "trigger" in s]
    assert len(trigger_steps) == 0

    text = SMOKE_PIPELINE_PATH.read_text(encoding="utf-8")
    assert '"nvidia.com/gpu": "2"' not in text
    assert '"nvidia.com/gpu": "4"' not in text
    assert "sglang" not in text

    commands = " ".join(str(s.get("command", "")) for s in command_steps)
    for expected in (
        "run.sh lm_eval",
        "run.sh restart_recovery",
        "run.sh cache_stats",
        "run.sh lazy_offload",
        "run.sh http_api",
        "run-cpu-e2e-validation.sh",
    ):
        assert expected in commands


def test_heavy_pipeline_keeps_common_matrix() -> None:
    """pipeline.yml preserves *common-matrix / *common-inference-engines."""
    heavy = yaml.safe_load(HEAVY_PIPELINE_PATH.read_text(encoding="utf-8"))
    steps = _collect_steps(heavy)

    by_label = {str(s.get("label", "")): s for s in steps}
    for workload in ("long_doc_qa_l2", "deadlock", "restart_recovery"):
        matching = [s for lbl, s in by_label.items() if f"/ {workload} /" in lbl]
        assert len(matching) == 1
        assert matching[0]["matrix"]["setup"]["inference_engine"] == [
            "vllm",
            "sglang",
        ]

    preempt_steps = [
        s for lbl, s in by_label.items() if "/ lm_eval_preemption /" in lbl
    ]
    assert len(preempt_steps) == 1
    assert preempt_steps[0]["matrix"]["setup"]["inference_engine"] == [
        "vllm",
        "sglang",
    ]


def _find_bash() -> str | None:
    """Locate bash (Git Bash or system bash) for shell gating tests."""
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        shutil.which("bash"),
    ):
        if candidate and Path(candidate).exists():
            # Skip WSL stub if no distro is installed.
            if "system32\\bash.exe" in candidate.lower():
                continue
            return candidate
    return None


@pytest.mark.parametrize(
    ("env_overrides", "changed_files", "expected_skip"),
    [
        # Ordinary PR touching lmcache code -> skip heavy
        (
            {"BUILDKITE_PULL_REQUEST": "42", "BUILDKITE_PULL_REQUEST_LABELS": ""},
            "lmcache/v1/cache_engine.py",
            True,
        ),
        # PR with force-ci -> run heavy
        (
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "force-ci",
            },
            "lmcache/v1/cache_engine.py",
            False,
        ),
        # Rollback flag MP_FULL_ON_PR=true -> run heavy
        (
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "",
                "MP_FULL_ON_PR": "true",
            },
            "lmcache/v1/cache_engine.py",
            False,
        ),
        # Non-PR (dev push) -> run heavy
        (
            {"BUILDKITE_PULL_REQUEST": "false", "BUILDKITE_PULL_REQUEST_LABELS": ""},
            "lmcache/v1/cache_engine.py",
            False,
        ),
        # Scheduled nightly -> run heavy
        (
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_SOURCE": "schedule",
            },
            "lmcache/v1/cache_engine.py",
            False,
        ),
        # PR touching shared CI dependency -> run heavy
        (
            {"BUILDKITE_PULL_REQUEST": "42", "BUILDKITE_PULL_REQUEST_LABELS": ""},
            ".buildkite/k3_tests/common_scripts/path-filter.sh",
            False,
        ),
        (
            {"BUILDKITE_PULL_REQUEST": "42", "BUILDKITE_PULL_REQUEST_LABELS": ""},
            "pyproject.toml",
            False,
        ),
    ],
)
def test_should_skip_multiprocess_heavy_rules(
    tmp_path: Path,
    env_overrides: dict[str, str],
    changed_files: str,
    expected_skip: bool,
) -> None:
    """Verify heavy multiprocess gating across PR, label, rollback, and CI edits."""
    bash = _find_bash()
    if bash is None:
        pytest.skip("Git Bash / bash not available")
    assert bash is not None

    clean_pf = tmp_path / "path-filter.sh"
    clean_pf.write_text(
        PATH_FILTER_SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n"),
        encoding="utf-8",
    )
    script = f"""
source "{clean_pf.as_posix()}"
_path_filter_get_changed_files() {{ printf '%s\\n' "{changed_files}"; }}
should_skip_multiprocess_heavy ".buildkite/k3_tests/multiprocess/pipeline.yml"
"""
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run([bash, "-c", script], env=env, check=False)
    assert (proc.returncode == 0) is expected_skip


@pytest.mark.parametrize(
    ("labels", "source", "pipeline_rel", "expected_uploaded"),
    [
        # full: PR-smoke runs, heavy is bypassed
        (
            "full",
            "webhook",
            ".buildkite/k3_tests/multiprocess/smoke-pipeline.yml",
            True,
        ),
        (
            "full",
            "webhook",
            ".buildkite/k3_tests/multiprocess/pipeline.yml",
            False,
        ),
        # skip expensive ci skips both smoke and heavy
        (
            "full,skip expensive ci",
            "webhook",
            ".buildkite/k3_tests/multiprocess/smoke-pipeline.yml",
            False,
        ),
        (
            "full,skip expensive ci",
            "webhook",
            ".buildkite/k3_tests/multiprocess/pipeline.yml",
            False,
        ),
        # force-ci: both smoke and heavy run in parallel from webhook
        (
            "full,force-ci",
            "webhook",
            ".buildkite/k3_tests/multiprocess/smoke-pipeline.yml",
            True,
        ),
        (
            "full,force-ci",
            "webhook",
            ".buildkite/k3_tests/multiprocess/pipeline.yml",
            True,
        ),
    ],
)
def test_upload_pipeline_parallel_labels(
    tmp_path: Path,
    labels: str,
    source: str,
    pipeline_rel: str,
    expected_uploaded: bool,
) -> None:
    """Verify 3-label parallel execution: full, force-ci, and skip expensive ci."""
    _run_upload_pipeline(
        tmp_path,
        labels=labels,
        source=source,
        pipeline_rel=pipeline_rel,
        changed_file="lmcache/v1/cache_engine.py",
        expected_uploaded=expected_uploaded,
    )


def _run_upload_pipeline(
    tmp_path: Path,
    *,
    labels: str,
    source: str,
    pipeline_rel: str,
    changed_file: str,
    expected_uploaded: bool,
) -> None:
    """Run upload-pipeline.sh with a fake git/buildkite-agent; check the upload."""
    bash = _find_bash()
    if bash is None:
        pytest.skip("Git Bash / bash not available")
    assert bash is not None

    scripts_dir = tmp_path / "common_scripts"
    scripts_dir.mkdir(parents=True)
    for src in (
        UPLOAD_SCRIPT,
        PATH_FILTER_SCRIPT,
        UPLOAD_SCRIPT.parent / "helpers.sh",
    ):
        (scripts_dir / src.name).write_text(
            src.read_text(encoding="utf-8").replace("\r\n", "\n"),
            encoding="utf-8",
        )

    marker = tmp_path / "uploaded.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_agent = fake_bin / "buildkite-agent"
    fake_agent.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "${{1:-}}" == "pipeline" ]]; then touch "{marker.as_posix()}"; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_agent.chmod(0o755)
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "diff" ]]; then\n'
        f'  printf "{changed_file}\\n"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    fake_bin_str = fake_bin.as_posix()
    script = f"""
FB=$(cygpath -u "{fake_bin_str}" 2>/dev/null || printf '%s' "{fake_bin_str}")
export PATH="$FB:$PATH"
bash "{(scripts_dir / "upload-pipeline.sh").as_posix()}" "{pipeline_rel}"
"""
    env = os.environ.copy()
    env.update(
        {
            "BUILDKITE_PULL_REQUEST": "42",
            "BUILDKITE_PULL_REQUEST_LABELS": labels,
            "BUILDKITE_SOURCE": source,
            "BUILDKITE_PULL_REQUEST_BASE_BRANCH": "dev",
        }
    )
    proc = subprocess.run([bash, "-c", script], env=env, check=False)
    assert proc.returncode == 0
    assert marker.exists() is expected_uploaded


def test_smoke_skips_unrelated_paths_and_docs_dont_trigger_heavy(
    tmp_path: Path,
) -> None:
    """Smoke uses multiprocess path-filter; .md edits don't trigger heavy."""
    bash = _find_bash()
    if bash is None:
        pytest.skip("Git Bash / bash not available")
    assert bash is not None

    clean_pf = tmp_path / "path-filter.sh"
    clean_pf.write_text(
        PATH_FILTER_SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n"),
        encoding="utf-8",
    )
    script = f"""
source "{clean_pf.as_posix()}"
_path_filter_get_changed_files() {{ printf 'examples/foo.py\\n'; }}
should_skip_ci ".buildkite/k3_tests/multiprocess/smoke-pipeline.yml"
r1=$?
_path_filter_get_changed_files() {{ printf '.buildkite/k3_tests/README.md\\n'; }}
has_shared_ci_dependency_change
r2=$?
exit $((r1 == 0 && r2 == 1 ? 0 : 1))
"""
    env = os.environ.copy()
    env.pop("CI_WEEKLY_FULL", None)
    env.update({"BUILDKITE_PULL_REQUEST": "42", "BUILDKITE_PULL_REQUEST_LABELS": ""})
    proc = subprocess.run([bash, "-c", script], env=env, check=False)
    assert proc.returncode == 0


@pytest.mark.parametrize(
    ("changed_file", "expected_rc"),
    [
        ("lmcache/integration/sglang/adapter.py", 1),
        ("lmcache/v1/common/kv_format/detectors/sglang.py", 1),
        ("benchmarks/long_doc_qa/long_doc_qa.py", 1),
        # Ordinary production code stays on the smoke tier (heavy bypassed).
        ("lmcache/v1/cache_engine.py", 0),
        ("lmcache/v1/common/kv_format/detectors/vllm.py", 0),
        ("benchmarks/other_bench/run.py", 0),
    ],
)
def test_heavy_only_paths_keep_heavy_enabled(
    tmp_path: Path, changed_file: str, expected_rc: int
) -> None:
    """SGLang and long_doc_qa edits must not bypass heavy on ordinary PRs."""
    bash = _find_bash()
    if bash is None:
        pytest.skip("Git Bash / bash not available")
    assert bash is not None

    clean_pf = tmp_path / "path-filter.sh"
    clean_pf.write_text(
        PATH_FILTER_SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n"),
        encoding="utf-8",
    )
    script = f"""
source "{clean_pf.as_posix()}"
_path_filter_get_changed_files() {{
    printf '%s\\n' '{changed_file}'
}}
should_skip_multiprocess_heavy ".buildkite/k3_tests/multiprocess/pipeline.yml"
"""
    env = os.environ.copy()
    env.pop("CI_WEEKLY_FULL", None)
    env.update({"BUILDKITE_PULL_REQUEST": "42", "BUILDKITE_PULL_REQUEST_LABELS": ""})
    proc = subprocess.run([bash, "-c", script], env=env, check=False)
    assert proc.returncode == expected_rc


def test_mp_labeled_sglang_only_pr_uploads_heavy_without_force_ci(
    tmp_path: Path,
) -> None:
    """End to end: `mp` PR touching only lmcache/integration/sglang/** uploads heavy."""
    _run_upload_pipeline(
        tmp_path,
        labels="mp",
        source="webhook",
        pipeline_rel=".buildkite/k3_tests/multiprocess/pipeline.yml",
        changed_file="lmcache/integration/sglang/adapter.py",
        expected_uploaded=True,
    )


def test_mp_labeled_non_sglang_pr_still_bypasses_heavy(tmp_path: Path) -> None:
    """Control: the same `mp` PR on ordinary code keeps bypassing heavy."""
    _run_upload_pipeline(
        tmp_path,
        labels="mp",
        source="webhook",
        pipeline_rel=".buildkite/k3_tests/multiprocess/pipeline.yml",
        changed_file="lmcache/v1/cache_engine.py",
        expected_uploaded=False,
    )
