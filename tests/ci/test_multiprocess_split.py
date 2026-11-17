# SPDX-License-Identifier: Apache-2.0
"""Tests for LM-1131 k3-multiprocess PR-smoke and heavy tier split.

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
    """smoke-pipeline.yml must stay fast (0/1-GPU) and cover only pr-smoke."""
    smoke = yaml.safe_load(SMOKE_PIPELINE_PATH.read_text(encoding="utf-8"))
    steps = _collect_steps(smoke)
    command_steps = [s for s in steps if "command" in s]
    assert len(command_steps) == 6

    trigger_steps = [s for s in steps if "trigger" in s]
    assert len(trigger_steps) == 1
    assert trigger_steps[0]["trigger"] == "k3-multiprocess-test"
    assert "force-ci" in str(trigger_steps[0].get("if", ""))

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


def test_heavy_pipeline_prunes_noop_skip_sglang_variants() -> None:
    """Known noop_skip sglang matrices are pruned from pipeline.yml."""
    heavy = yaml.safe_load(HEAVY_PIPELINE_PATH.read_text(encoding="utf-8"))
    steps = _collect_steps(heavy)

    by_label = {str(s.get("label", "")): s for s in steps}
    for workload in ("long_doc_qa_l2", "deadlock"):
        matching = [s for lbl, s in by_label.items() if f"/ {workload} /" in lbl]
        assert len(matching) == 1
        assert matching[0]["matrix"]["setup"]["inference_engine"] == ["vllm"]

    restart_steps = [s for lbl, s in by_label.items() if "/ restart_recovery /" in lbl]
    assert len(restart_steps) == 1
    assert restart_steps[0]["matrix"]["setup"]["inference_engine"] == ["vllm"]

    preempt_steps = [
        s for lbl, s in by_label.items() if "/ lm_eval_preemption /" in lbl
    ]
    assert len(preempt_steps) == 1
    assert preempt_steps[0]["matrix"]["setup"]["inference_engine"] == ["vllm"]


def _find_bash() -> str | None:
    """Locate bash (Git Bash or system bash) for shell gating tests."""
    for candidate in (
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
    ):
        if candidate and Path(candidate).exists():
            # Skip WSL stub if no distro is installed.
            if "System32\\bash.exe" in candidate:
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
        # PR with mp-full -> run heavy
        (
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "mp-full",
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
    ("labels", "pipeline_rel", "expected_uploaded"),
    [
        # mp-full bypasses smoke-pipeline.yml (exits 0) and runs heavy pipeline.yml
        ("full,mp-full", ".buildkite/k3_tests/multiprocess/smoke-pipeline.yml", False),
        ("full,mp-full", ".buildkite/k3_tests/multiprocess/pipeline.yml", True),
        # skip expensive ci skips both smoke and heavy
        (
            "full,skip expensive ci",
            ".buildkite/k3_tests/multiprocess/smoke-pipeline.yml",
            False,
        ),
        (
            "full,skip expensive ci",
            ".buildkite/k3_tests/multiprocess/pipeline.yml",
            False,
        ),
    ],
)
def test_upload_pipeline_mp_full_and_skip_expensive_ci(
    tmp_path: Path,
    labels: str,
    pipeline_rel: str,
    expected_uploaded: bool,
) -> None:
    """Verify mp-full bypasses smoke-only and skip expensive ci skips both."""
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
        f'#!/usr/bin/env bash\nif [[ "${{1:-}}" == "pipeline" ]]; then touch "{marker.as_posix()}"; fi\nexit 0\n',
        encoding="utf-8",
    )

    script = f"""
export PATH="{fake_bin.as_posix()}:$PATH"
merge_pr_base_branch() {{ :; }}
_path_filter_get_changed_files() {{ printf 'lmcache/v1/cache_engine.py\\n'; }}
bash "{(scripts_dir / "upload-pipeline.sh").as_posix()}" "{pipeline_rel}"
"""
    env = os.environ.copy()
    env.update(
        {
            "BUILDKITE_PULL_REQUEST": "42",
            "BUILDKITE_PULL_REQUEST_LABELS": labels,
            "BUILDKITE_PULL_REQUEST_BASE_BRANCH": "dev",
        }
    )
    proc = subprocess.run([bash, "-c", script], env=env, check=False)
    assert proc.returncode == 0
    assert marker.exists() is expected_uploaded
