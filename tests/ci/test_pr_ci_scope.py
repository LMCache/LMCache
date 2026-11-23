# SPDX-License-Identifier: Apache-2.0
"""Contract tests for .github/scripts/pr-ci-scope.sh."""

# Standard
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

# Third Party
import pytest

# First Party
from tests.import_utils import repo_path

SCRIPT = repo_path(".github/scripts/pr-ci-scope.sh")


def _find_bash() -> str | None:
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        shutil.which("bash"),
    ):
        if (
            candidate
            and Path(candidate).exists()
            and "system32" not in candidate.lower()
        ):
            return candidate
    return None


def _run_scope(tmp_path: Path, pr_body: str) -> dict[str, object]:
    bash = _find_bash()
    if bash is None:
        pytest.skip("Git Bash / bash not available")
    assert bash is not None
    clean_script = tmp_path / "pr-ci-scope.sh"
    clean_script.write_text(
        SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n"),
        encoding="utf-8",
    )
    out_file = tmp_path / "github_output.txt"
    env = {
        **os.environ,
        "PR_BODY": pr_body,
        "GITHUB_OUTPUT": out_file.as_posix(),
    }
    if shutil.which("jq") is None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        jq_shim = bin_dir / "jq"
        jq_shim.write_text(
            f'#!/usr/bin/env bash\n"{Path(sys.executable).as_posix()}" -c '
            '"import json, sys; '
            "print(json.dumps([l.rstrip('\\r\\n') for l in sys.stdin]))\"\n",
            encoding="utf-8",
        )
        jq_shim.chmod(0o755)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run([bash, clean_script.as_posix()], env=env, check=True)
    result: dict[str, object] = {}
    for line in out_file.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        result[key] = value
    return {
        "python_versions": json.loads(str(result["python_versions"])),
        "cpu_os": json.loads(str(result["cpu_os"])),
        "rust": result["rust"] == "true",
    }


def test_empty_body_falls_back_to_full_matrix(tmp_path: Path) -> None:
    assert _run_scope(tmp_path, "") == {
        "python_versions": ["3.11", "3.12", "3.13", "3.14"],
        "cpu_os": ["ubuntu-22.04", "macos-latest"],
        "rust": False,
    }


def test_checked_subsets_and_rust(tmp_path: Path) -> None:
    body = """
- [x] `py3.12`
- [X] `py3.13`
* [x] `ubuntu-22.04`
- [x] `rust`
"""
    assert _run_scope(tmp_path, body) == {
        "python_versions": ["3.12", "3.13"],
        "cpu_os": ["ubuntu-22.04"],
        "rust": True,
    }


def test_crlf_and_unrelated_text_ignored(tmp_path: Path) -> None:
    body = (
        "Summary: fixes $(rm -rf /)\r\n"
        "- [x] this PR contains unit tests\r\n"
        "- [x] `py3.14` extra note\r\n"
        "- [ ] `macos-latest`\r\n"
    )
    assert _run_scope(tmp_path, body) == {
        "python_versions": ["3.14"],
        "cpu_os": ["ubuntu-22.04", "macos-latest"],
        "rust": False,
    }
