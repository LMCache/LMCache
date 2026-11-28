# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from unittest.mock import patch
import importlib.util
import sys
import unittest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / ".buildkite"
    / "scripts"
    / "fast_fail_cancel.py"
)
_SPEC = importlib.util.spec_from_file_location("fast_fail_cancel", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
fast_fail_cancel = importlib.util.module_from_spec(_SPEC)
sys.modules["fast_fail_cancel"] = fast_fail_cancel
_SPEC.loader.exec_module(fast_fail_cancel)


class TestFastFailCancel(unittest.TestCase):
    def setUp(self):
        self.commit = "a1b2c3d4e5f67890123456789012345678901234"
        self.repo = "LMCache/LMCache"
        self.org = "lmcache"

    @patch("fast_fail_cancel._http_request")
    def test_cancel_github_actions_dry_run_filters_pr_event_and_self(self, mock_http):
        mock_http.side_effect = [
            (
                200,
                {
                    "workflow_runs": [
                        {
                            "id": 1001,
                            "name": "Code Quality",
                            "event": "pull_request",
                            "head_branch": "feature-1",
                            "pull_requests": [{"number": 12}],
                            "status": "in_progress",
                            "html_url": "https://github.com/run/1001",
                        },
                        {
                            "id": 1002,
                            "name": "Aerospike integration",
                            "event": "pull_request",
                            "head_branch": "feature-1",
                            "pull_requests": [{"number": 12}],
                            "status": "in_progress",
                            "html_url": "https://github.com/run/1002",
                        },
                        {
                            "id": 1003,
                            "name": "CPU Device Rerun",
                            "event": "workflow_dispatch",
                            "head_branch": "dev",
                            "status": "in_progress",
                            "html_url": "https://github.com/run/1003",
                        },
                        {
                            "id": 1004,
                            "name": "CodeQL Advanced",
                            "event": "pull_request",
                            "head_branch": "feature-1",
                            "pull_requests": [{"number": 99}],
                            "status": "in_progress",
                            "html_url": "https://github.com/run/1004",
                        },
                    ]
                },
            ),
            (200, {"workflow_runs": []}),
        ]

        cancelled = fast_fail_cancel.cancel_github_actions(
            commit_sha=self.commit,
            repo=self.repo,
            token="fake_gh_token",
            current_run_id="1001",
            pr_number="12",
            dry_run=True,
        )

        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["id"], "1002")
        for call_args in mock_http.call_args_list:
            self.assertNotEqual(call_args[1].get("method"), "POST")

    @patch("fast_fail_cancel._http_request")
    def test_cancel_github_actions_live(self, mock_http):
        mock_http.side_effect = [
            (
                200,
                {
                    "workflow_runs": [
                        {
                            "id": 1002,
                            "name": "CodeQL Advanced",
                            "event": "pull_request",
                            "head_branch": "feature-1",
                            "status": "in_progress",
                            "html_url": "https://github.com/run/1002",
                        }
                    ]
                },
            ),
            (200, {"workflow_runs": []}),
            (202, {}),
        ]

        cancelled = fast_fail_cancel.cancel_github_actions(
            commit_sha=self.commit,
            repo=self.repo,
            token="fake_gh_token",
            current_run_id=None,
            dry_run=False,
        )

        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["id"], "1002")
        post_calls = [
            c for c in mock_http.call_args_list if c[1].get("method") == "POST"
        ]
        self.assertEqual(len(post_calls), 1)
        self.assertIn("1002/cancel", post_calls[0][0][0])

    @patch("fast_fail_cancel._http_request")
    def test_cancel_buildkite_builds_includes_failing_and_filters_pr(self, mock_http):
        mock_http.return_value = (
            200,
            [
                {
                    "id": "bk-uuid-1",
                    "number": 101,
                    "pipeline": {"slug": "k3-unit-tests"},
                    "branch": "feature-1",
                    "state": "running",
                    "pull_request": {"id": "12"},
                    "web_url": "https://buildkite.com/101",
                },
                {
                    "id": "bk-uuid-2",
                    "number": 102,
                    "pipeline": {"slug": "k3-multiprocess-smoke"},
                    "branch": "feature-1",
                    "state": "failing",
                    "pull_request": {"id": "12"},
                    "web_url": "https://buildkite.com/102",
                },
                {
                    "id": "bk-uuid-3",
                    "number": 103,
                    "pipeline": {"slug": "k3-multiprocess-test"},
                    "branch": "dev",
                    "state": "running",
                    "pull_request": None,
                    "web_url": "https://buildkite.com/103",
                },
                {
                    "id": "bk-uuid-4",
                    "number": 104,
                    "pipeline": {"slug": "k3-multiprocess-smoke"},
                    "branch": "feature-other",
                    "state": "creating",
                    "pull_request": {"id": "99"},
                    "web_url": "https://buildkite.com/104",
                },
            ],
        )

        cancelled = fast_fail_cancel.cancel_buildkite_builds(
            commit_sha=self.commit,
            org=self.org,
            token="fake_bk_token",
            current_build_id="bk-uuid-1",
            pr_number="12",
            dry_run=True,
        )

        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["number"], 102)
        self.assertEqual(cancelled[0]["pipeline"], "k3-multiprocess-smoke")

    @patch("fast_fail_cancel.cancel_github_actions")
    @patch("fast_fail_cancel.cancel_buildkite_builds")
    def test_main_cli_execution(self, mock_cancel_bk, mock_cancel_gha):
        mock_cancel_gha.return_value = [{"id": "1001"}]
        mock_cancel_bk.return_value = [{"number": 101}]

        test_args = [
            "fast_fail_cancel.py",
            "--commit",
            self.commit,
            "--pr-number",
            "12",
            "--github-token",
            "test_gh",
            "--buildkite-token",
            "test_bk",
            "--dry-run",
        ]
        with patch.object(sys, "argv", test_args):
            exit_code = fast_fail_cancel.main()
            self.assertEqual(exit_code, 0)
            mock_cancel_gha.assert_called_once()
            mock_cancel_bk.assert_called_once()


if __name__ == "__main__":
    unittest.main()
