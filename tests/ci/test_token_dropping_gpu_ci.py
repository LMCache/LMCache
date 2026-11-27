# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the isolated token-dropping GPU pipeline."""

# Standard
from pathlib import Path
import os
import shutil
import subprocess
import sys

# Third Party
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / ".buildkite/k3_tests/token-dropping"


def _find_bash() -> str:
    bash = next(
        (
            candidate
            for candidate in (
                r"C:\Program Files\Git\bin\bash.exe",
                shutil.which("bash"),
            )
            if candidate
            and Path(candidate).exists()
            and "system32\\bash.exe" not in candidate.lower()
        ),
        None,
    )
    if bash is None:
        pytest.skip("bash not available")
    return bash


@pytest.mark.parametrize(
    ("changes", "labels", "expected_upload"),
    [
        ("lmcache/v1/cache_engine.py", "", False),
        ("lmcache/v1/cache_engine.py", "token dropping,force-ci", False),
        ("lmcache/integration/vllm/experimental/token_drop_adapter.py", "", None),
        ("lmcache/integration/vllm/experimental/token_drop.py", "", None),
        ("lmcache/integration/vllm/experimental/token_drop_worker.py", "", True),
        (
            "lmcache/integration/vllm/experimental/token_drop_kv.py",
            "token dropping",
            True,
        ),
        ("tests/v1/test_token_drop_gpu_views.py", "skip expensive ci", True),
        ("tests/v1/test_token_drop_worker.py", "token dropping,force-ci", True),
        (".buildkite/k3_tests/token-dropping/run.sh", "", True),
        ("docs/notes.md\ntests/v1/test_token_drop_worker.py", "", True),
        ("", "", True),
        (None, "", True),
    ],
)
@pytest.mark.parametrize("has_worker", [False, True])
def test_gpu_upload_scope(tmp_path, changes, labels, expected_upload, has_worker):
    bash = _find_bash()
    suite = tmp_path / ".buildkite/k3_tests/token-dropping"
    suite.mkdir(parents=True)
    common = suite.parent / "common_scripts"
    if has_worker:
        worker = tmp_path / "lmcache/integration/vllm/experimental/token_drop_worker.py"
        worker.parent.mkdir(parents=True)
        worker.touch()
    if expected_upload is None:
        expected_upload = has_worker
    common.mkdir()
    (suite / "upload.sh").write_text(
        (SUITE / "upload.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    (common / "path-filter.sh").write_text(
        "_path_filter_get_changed_files() {\n"
        '  if [[ "$CHANGE_ERROR" == 1 ]]; then return 1; fi\n'
        '  printf "%s\\n" "$CHANGES"\n'
        "}\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_agent = tmp_path / "buildkite-agent"
    fake_agent.write_text(
        '#!/usr/bin/env bash\necho "UPLOADED: $*"\n',
        encoding="utf-8",
        newline="\n",
    )
    fake_agent.chmod(0o755)
    result = subprocess.run(
        [
            bash,
            "-c",
            'FB=$(cygpath -u "$1" 2>/dev/null || printf "%s" "$1"); '
            'export PATH="$FB:$PATH"; bash "$2"',
            "test",
            tmp_path.as_posix(),
            (suite / "upload.sh").as_posix(),
        ],
        env={
            **os.environ,
            "CHANGES": changes or "",
            "CHANGE_ERROR": "1" if changes is None else "0",
            "BUILDKITE_PULL_REQUEST_LABELS": labels,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert ("UPLOADED: pipeline upload " in result.stdout) is expected_upload


@pytest.mark.parametrize(
    ("cases", "expected_pass"),
    [
        (
            '<testcase classname="tests.v1.test_token_drop_gpu_views" />'
            '<testcase classname="tests.v1.test_token_drop_worker" />',
            True,
        ),
        ("", False),
        ('<testcase classname="tests.v1.test_token_drop_gpu_views" />', False),
        (
            '<testcase classname="tests.v1.test_token_drop_gpu_views">'
            "<skipped /></testcase>"
            '<testcase classname="tests.v1.test_token_drop_worker" />',
            False,
        ),
        (
            '<testcase classname="tests.v1.test_token_drop_gpu_views" />'
            '<testcase classname="tests.v1.test_token_drop_worker">'
            "<failure /></testcase>",
            False,
        ),
        (
            '<testcase classname="tests.v1.test_token_drop_gpu_views" />'
            '<testcase classname="tests.v1.test_token_drop_worker">'
            "<error /></testcase>",
            False,
        ),
    ],
)
def test_gpu_result_gate(tmp_path, cases, expected_pass):
    report = tmp_path / "junit.xml"
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    result = subprocess.run(
        [
            sys.executable,
            str(SUITE / "verify-results.py"),
            str(report),
            "tests/v1/test_token_drop_gpu_views.py",
            "tests/v1/test_token_drop_worker.py",
        ],
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) is expected_pass


@pytest.mark.parametrize("has_lane", [False, True])
def test_bootstrap_handles_prs_predating_lane(tmp_path, has_lane):
    bash = _find_bash()
    if has_lane:
        script = tmp_path / ".buildkite/k3_tests/token-dropping/upload.sh"
        script.parent.mkdir(parents=True)
        script.write_text('echo "LANE STARTED"\n', encoding="utf-8", newline="\n")
    config = yaml.safe_load(
        (SUITE / "buildkite-pipeline.yml").read_text(encoding="utf-8")
    )
    result = subprocess.run(
        [bash, "-c", config["steps"][0]["command"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ("LANE STARTED" in result.stdout) is has_lane
    assert ("no GPU allocated" in result.stdout) is not has_lane
