# SPDX-License-Identifier: Apache-2.0
"""CI routing checks runnable without loading the LMCache pytest fixtures."""

# Standard
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

# Third Party
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".buildkite/k3_tests/common_scripts"


def run_shell(script: str, **variables: str) -> subprocess.CompletedProcess[str]:
    """Run real routing scripts with an isolated CI environment."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("BUILDKITE_", "LMCACHE_CI_", "LMCACHE_PR_BASE_MERGE"))
    }
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=env | variables,
        capture_output=True,
        text=True,
        timeout=10,
    )


class TestCIMode(unittest.TestCase):
    def test_selection_and_overrides(self) -> None:
        """Only an unambiguous selector excludes the opposite runtime."""
        cases = [
            ({}, "all"),
            ({"LMCACHE_CI_MODE": "all"}, "all"),
            ({"LMCACHE_CI_MODE": "mp"}, "mp"),
            ({"LMCACHE_CI_MODE": "inprocess"}, "inprocess"),
            ({"LMCACHE_CI_MODE": "typo"}, "all"),
            ({"BUILDKITE_PULL_REQUEST_LABELS": "full,ci-mp"}, "mp"),
            ({"BUILDKITE_PULL_REQUEST_LABELS": "ci-inprocess,full"}, "inprocess"),
            ({"BUILDKITE_PULL_REQUEST_LABELS": "ci-mp,ci-inprocess"}, "all"),
            ({"BUILDKITE_PULL_REQUEST_LABELS": "not-ci-mp,ci-inprocess-extra"}, "all"),
            (
                {"LMCACHE_CI_MODE": "all", "BUILDKITE_PULL_REQUEST_LABELS": "ci-mp"},
                "all",
            ),
            (
                {
                    "LMCACHE_CI_MODE": "mp",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ci-inprocess",
                },
                "mp",
            ),
            (
                {"LMCACHE_CI_MODE": "mp", "BUILDKITE_PULL_REQUEST_LABELS": "force-ci"},
                "all",
            ),
            ({"LMCACHE_CI_MODE": "inprocess", "BUILDKITE_SOURCE": "schedule"}, "all"),
        ]
        for variables, selected in cases:
            for suite in ("mp", "inprocess", "shared", "unknown"):
                with self.subTest(variables=variables, suite=suite):
                    result = run_shell(
                        f'source "{SCRIPTS}/ci-mode.sh"\n'
                        f"if should_skip_ci_mode {suite}; "
                        "then echo skip; else echo run; fi",
                        **variables,
                    )
                    skip = (
                        selected != "all"
                        and suite in {"mp", "inprocess"}
                        and suite != selected
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), "skip" if skip else "run")

    def test_k3_pipeline_routing(self) -> None:
        """Audit runtime pipelines while retaining unit, mixed and unknown suites."""
        suites = {
            "integration": "inprocess",
            "correctness": "inprocess",
            "comprehensive": "inprocess",
            "multiprocess": "mp",
            "sglang": "mp",
            "blend": "mp",
            "xpu/multiprocess": "mp",
            "unit": "shared",
            "xpu/unittests": "shared",
            "amd": "shared",
            "musa": "shared",
            "future": "shared",
        }
        for suite, runtime in suites.items():
            for selected in ("all", "mp", "inprocess"):
                with self.subTest(suite=suite, selected=selected):
                    result = run_shell(
                        f'source "{SCRIPTS}/path-filter.sh"\n'
                        "_path_filter_get_changed_files() { echo lmcache/utils.py; }\n"
                        f"if should_skip_ci .buildkite/k3_tests/{suite}/pipeline.yml; "
                        "then echo skip; else echo run; fi",
                        LMCACHE_CI_MODE=selected,
                    )
                    skip = (
                        selected != "all"
                        and runtime != "shared"
                        and runtime != selected
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), "skip" if skip else "run")

    def test_path_filtering_is_retained(self) -> None:
        """Mode eligibility retains path filtering and its full-run overrides."""
        cases = [
            ("docs/index.rst", {}, "skip"),
            ("tests/test_utils.py", {}, "skip"),
            ("lmcache/utils.py", {}, "run"),
            ("", {}, "run"),
            ("docs/index.rst", {"BUILDKITE_PULL_REQUEST_LABELS": "force-ci"}, "run"),
            ("docs/index.rst", {"BUILDKITE_SOURCE": "schedule"}, "run"),
        ]
        for changed_file, overrides, expected in cases:
            with self.subTest(changed_file=changed_file, overrides=overrides):
                result = run_shell(
                    f'source "{SCRIPTS}/path-filter.sh"\n'
                    '_path_filter_get_changed_files() { echo "$CHANGED_FILE"; }\n'
                    "if should_skip_ci .buildkite/k3_tests/multiprocess/pipeline.yml; "
                    "then echo skip; else echo run; fi",
                    CHANGED_FILE=changed_file,
                    LMCACHE_CI_MODE="mp",
                    **overrides,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_direct_pipeline_guards(self) -> None:
        """Older entry points skip before installing or launching anything."""
        pipelines = {
            ".buildkite/pipelines/comprehensive-tests.yml": "inprocess",
            ".buildkite/pipelines/multiprocessing-test.steps.yml": "mp",
            ".buildkite/vllm-integration-tests.yml": "inprocess",
            ".buildkite/correctness/pipeline.correctness.steps.yml": "inprocess",
            ".buildkite/correctness/pipeline.mmlu.yml": "inprocess",
            ".buildkite/k3_tests/amd/pipeline.yml": "mp",
        }
        for path, runtime in pipelines.items():
            pipeline = yaml.safe_load((ROOT / path).read_text())
            step = pipeline["steps"][-1] if "/amd/" in path else pipeline["steps"][0]
            command = step.get("command", step.get("commands"))
            if isinstance(command, list):
                command = "\n".join(command)
            guard = f"if should_skip_ci_mode {runtime}; then exit 0; fi"
            self.assertIn(guard, command)
            prefix = command.split(guard)[0] + guard + "\necho continued"
            for mode in ("mp", "inprocess"):
                with self.subTest(path=path, mode=mode):
                    result = run_shell(prefix, LMCACHE_CI_MODE=mode)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        result.stdout.strip(), "continued" if mode == runtime else ""
                    )

    def test_upload_skips_and_failure_propagation(self) -> None:
        """Retain admission/base-merge gates and propagate upload failures."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            agent = path / "buildkite-agent"
            agent.write_text('#!/bin/sh\necho "$*"\n[ "$1" != pipeline ] || exit 7\n')
            git = path / "git"
            git.write_text(
                '#!/bin/sh\n[ "$1" != fetch ] || exit "${FETCH_EXIT:-0}"\n'
                '[ "$1" != diff ] || echo lmcache/utils.py\n'
            )
            agent.chmod(0o755)
            git.chmod(0o755)
            cases = [
                ("inprocess", {}, 0, False),
                ("mp", {}, 7, True),
                ("mp", {"BUILDKITE_PULL_REQUEST_LABELS": "good first issue"}, 0, False),
                (
                    "inprocess",
                    {"BUILDKITE_PULL_REQUEST_LABELS": "good first issue,force-ci"},
                    7,
                    True,
                ),
                ("mp", {"FETCH_EXIT": "9"}, 9, False),
            ]
            for mode, overrides, code, uploaded in cases:
                with self.subTest(mode=mode, overrides=overrides):
                    result = run_shell(
                        f'bash "{SCRIPTS}/upload-pipeline.sh" '
                        ".buildkite/k3_tests/multiprocess/pipeline.yml",
                        PATH=f"{directory}:{os.environ['PATH']}",
                        LMCACHE_CI_MODE=mode,
                        BUILDKITE_PULL_REQUEST="5303",
                        **overrides,
                    )
                    self.assertEqual(result.returncode, code, result.stderr)
                    self.assertEqual("pipeline upload" in result.stdout, uploaded)


if __name__ == "__main__":
    unittest.main()
