# SPDX-License-Identifier: Apache-2.0
"""Exercise the Buildkite upload entry point with local Git repos and a fake agent.

Run with Python 3.10+ and Git; LMCache, pytest, and GPU dependencies are not needed.
"""

# Standard
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

UPLOAD_SCRIPT = Path(__file__).resolve().parents[1] / "upload-pipeline.sh"
SUITES = ("integration", "multiprocess")
TEST_FILE = "tests/v1/test_mp_connector_kv_roles.py"


class UploadPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.git("init", "--quiet", "--initial-branch=dev")
        self.git("config", "user.name", "CI filter test")
        self.git("config", "user.email", "ci-filter@example.com")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "diff.renames", "true")
        self.change_files(TEST_FILE, "lmcache/runtime.py")
        self.git(
            "clone", "--quiet", "--bare", str(self.repo), str(self.root / "origin")
        )
        self.git("remote", "add", "origin", str(self.root / "origin"))
        self.git("fetch", "--quiet", "origin")
        self.git("checkout", "--quiet", "-b", "test-pr")

        self.agent_log = self.root / "agent.log"
        agent = self.root / "buildkite-agent"
        agent.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$*" >> "$AGENT_LOG"\n'
            'if [[ "$1" == annotate ]]; then\n'
            '    exit "${ANNOTATE_EXIT_CODE:-0}"\n'
            "fi\n"
            'exit "${UPLOAD_EXIT_CODE:-0}"\n'
        )
        agent.chmod(0o755)

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, text=True, stderr=subprocess.STDOUT
        ).strip()

    def change_files(self, *paths: str) -> None:
        for name in paths:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as stream:
                stream.write("# fixture change\n")
        self.git("add", "--all")
        self.git("commit", "--quiet", "-m", "Fixture change")

    def assert_upload(
        self,
        suite: str,
        *,
        upload: bool | None,
        exit_code: int = 0,
        **overrides: str,
    ) -> None:
        pipeline = f".buildkite/k3_tests/{suite}/pipeline.yml"
        # Keep inherited Buildkite settings from affecting the fixture build.
        env = {k: v for k, v in os.environ.items() if not k.startswith("BUILDKITE_")}
        env.update(
            PATH=f"{self.root}{os.pathsep}{env['PATH']}",
            AGENT_LOG=str(self.agent_log),
            ANNOTATE_EXIT_CODE="0",
            UPLOAD_EXIT_CODE="0",
            BUILDKITE_PULL_REQUEST="5295",
            BUILDKITE_PULL_REQUEST_BASE_BRANCH="dev",
            BUILDKITE_PULL_REQUEST_LABELS="full",
            BUILDKITE_SOURCE="webhook",
            LMCACHE_PR_BASE_MERGE="auto",
        )
        env.update(overrides)
        self.agent_log.write_text("")
        result = subprocess.run(
            ["bash", str(UPLOAD_SCRIPT), pipeline],
            cwd=self.repo,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
        calls = self.agent_log.read_text().splitlines()
        if upload is None:
            self.assertEqual(calls, [])
        elif upload:
            self.assertEqual(calls, [f"pipeline upload {pipeline}"])
        else:
            self.assertEqual(len(calls), 1, calls)
            self.assertTrue(calls[0].startswith("annotate --style success "), calls)
            self.assertIn("--context path-filter-skip", calls[0])

    def test_tests_only_reports_success_without_uploading(self) -> None:
        self.change_files(TEST_FILE)
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=False)

    def test_tests_and_docs_report_success_without_uploading(self) -> None:
        self.change_files(TEST_FILE, "docs/test-guide.md")
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=False)

    def test_base_branch_updates_do_not_make_test_only_pr_relevant(self) -> None:
        self.change_files(TEST_FILE)
        self.git("checkout", "--quiet", "dev")
        self.change_files("lmcache/runtime.py")
        self.git("push", "--quiet", "origin", "dev")
        self.git("checkout", "--quiet", "test-pr")
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=False)

    def test_tests_only_still_uploads_unit_tests(self) -> None:
        self.change_files(TEST_FILE)
        self.assert_upload("unit", upload=True)

    def test_production_changes_upload_both_suites(self) -> None:
        self.change_files(TEST_FILE, "lmcache/runtime.py")
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=True)

    def test_shared_harness_changes_upload_both_suites(self) -> None:
        self.change_files(".buildkite/k3_harness/setup-env.sh")
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=True)

    def test_dependency_changes_upload_both_suites(self) -> None:
        self.change_files(TEST_FILE, "requirements/common.txt")
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=True)

    def test_deleting_tests_reports_success_without_uploading(self) -> None:
        self.git("rm", TEST_FILE)
        self.git("commit", "--quiet", "-m", "Delete test")
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=False)

    def test_suite_script_changes_upload_own_suite(self) -> None:
        self.change_files(".buildkite/k3_tests/multiprocess/run.sh")
        self.assert_upload("multiprocess", upload=True)
        self.assert_upload("integration", upload=False)

    def test_force_ci_overrides_test_only_skip(self) -> None:
        self.change_files(TEST_FILE)
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(
                    suite, upload=True, BUILDKITE_PULL_REQUEST_LABELS="full,force-ci"
                )

    def test_scheduled_builds_still_upload(self) -> None:
        self.change_files(TEST_FILE)
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=True, BUILDKITE_SOURCE="schedule")

    def test_missing_base_runs_tests(self) -> None:
        self.change_files(TEST_FILE)
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(
                    suite,
                    upload=True,
                    BUILDKITE_PULL_REQUEST_BASE_BRANCH="missing",
                    LMCACHE_PR_BASE_MERGE="never",
                )

    def test_base_merge_failure_does_not_report_successful_skip(self) -> None:
        self.change_files(TEST_FILE)
        self.assert_upload(
            "integration",
            upload=None,
            exit_code=128,
            BUILDKITE_PULL_REQUEST_BASE_BRANCH="missing",
        )

    def test_empty_diff_runs_tests(self) -> None:
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assert_upload(suite, upload=True)

    def test_annotation_failure_does_not_fail_skipped_build(self) -> None:
        self.change_files(TEST_FILE)
        self.assert_upload("integration", upload=False, ANNOTATE_EXIT_CODE="7")

    def test_upload_failure_fails_build(self) -> None:
        self.change_files("lmcache/runtime.py")
        self.assert_upload(
            "integration", upload=True, exit_code=23, UPLOAD_EXIT_CODE="23"
        )

    def test_moving_production_code_into_tests_still_uploads(self) -> None:
        self.git("mv", "lmcache/runtime.py", "tests/moved_runtime.py")
        self.git("commit", "--quiet", "-m", "Move production code into tests")
        for source in ("pr", "push"):
            for suite in SUITES:
                with self.subTest(source=source, suite=suite):
                    self.assert_upload(
                        suite,
                        upload=True,
                        BUILDKITE_PULL_REQUEST="5295" if source == "pr" else "false",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
