# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache quota`` CLI command."""

# Standard
from unittest.mock import MagicMock, patch
import argparse
import json
import sys

# Inject a fake openai module so that the auto-discovery of
# lmcache.cli.commands (which imports bench -> openai) does not fail
# in environments where openai is not installed.
if "openai" not in sys.modules:
    _fake_openai = MagicMock()
    sys.modules["openai"] = _fake_openai

# Third Party
import pytest

# First Party
from lmcache.cli.commands.quota import QuotaCommand
from lmcache.cli.commands.quota._helpers import (
    DEFAULT_SALT_SENTINEL,
    escape_salt,
    normalize_url,
    unescape_salt,
)


@pytest.fixture
def cmd() -> QuotaCommand:
    return QuotaCommand()


@pytest.fixture
def parser(cmd: QuotaCommand) -> argparse.ArgumentParser:
    """An ArgumentParser with QuotaCommand registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cmd.register(sub)
    return p


class TestHelpers:
    def test_normalize_url(self) -> None:
        assert normalize_url("localhost:8080") == "http://localhost:8080"
        assert normalize_url("https://host:443/") == "https://host:443"

    def test_escape_unescape_salt(self) -> None:
        assert escape_salt("") == DEFAULT_SALT_SENTINEL
        assert escape_salt("tenant1") == "tenant1"
        assert unescape_salt(DEFAULT_SALT_SENTINEL) == ""
        assert unescape_salt("tenant1") == "tenant1"


class TestQuotaCommandMetadata:
    def test_name_and_help(self, cmd: QuotaCommand) -> None:
        assert cmd.name() == "quota"
        assert "quota" in cmd.help().lower()


class TestQuotaCommandExecute:
    @pytest.mark.parametrize(
        "response",
        [
            {
                "cache_salt": "tenant1",
                "limit_gb": 2.5,
                "current_usage_gb": 0.125,
                "exists": True,
            },
            {
                "cache_salt": "tenant1",
                "quota_limit_gb": 2.5,
                "usage_gb": 0.125,
                "quota_exists": True,
            },
        ],
    )
    @patch("lmcache.cli.commands.quota.get_command.http_request")
    def test_get_response_schemas(
        self,
        mock_http: MagicMock,
        parser: argparse.ArgumentParser,
        cmd: QuotaCommand,
        capsys: pytest.CaptureFixture[str],
        response: dict[str, object],
    ) -> None:
        """Server and coordinator responses expose the same CLI metrics."""
        mock_http.return_value = response
        args = parser.parse_args(["quota", "get", "tenant1", "--format", "json"])
        cmd.execute(args)

        assert json.loads(capsys.readouterr().out)["metrics"] == {
            "cache_salt": "tenant1",
            "limit_gb": 2.5,
            "current_usage_gb": 0.125,
            "exists": True,
        }

    @pytest.mark.parametrize("usage_gb", [0.0, 0.125])
    @patch("lmcache.cli.commands.quota.get_command.http_request")
    def test_get_coordinator_without_explicit_quota(
        self,
        mock_http: MagicMock,
        parser: argparse.ArgumentParser,
        cmd: QuotaCommand,
        capsys: pytest.CaptureFixture[str],
        usage_gb: float,
    ) -> None:
        """An absent explicit quota retains zero and false from the API."""
        mock_http.return_value = {
            "cache_salt": "tenant1",
            "quota_limit_gb": 0.0,
            "usage_gb": usage_gb,
            "quota_exists": False,
        }
        args = parser.parse_args(["quota", "get", "tenant1", "--format", "json"])
        cmd.execute(args)

        assert json.loads(capsys.readouterr().out)["metrics"] == {
            "cache_salt": "tenant1",
            "limit_gb": 0.0,
            "current_usage_gb": usage_gb,
            "exists": False,
        }

    @pytest.mark.parametrize(
        "response",
        [
            {"users": {"tenant1": {"limit_gb": 2.5, "current_usage_gb": 0.125}}},
            {
                "total_gb": 0.625,
                "by_cache_salt": [
                    {
                        "cache_salt": "tenant1",
                        "quota_limit_gb": 2.5,
                        "usage_gb": 0.125,
                        "quota_exists": True,
                    },
                    {
                        "cache_salt": "usage-only",
                        "quota_limit_gb": 0.0,
                        "usage_gb": 0.5,
                        "quota_exists": False,
                    },
                ],
            },
        ],
    )
    @patch("lmcache.cli.commands.quota.list_command.http_request")
    def test_list_response_schemas(
        self,
        mock_http: MagicMock,
        parser: argparse.ArgumentParser,
        cmd: QuotaCommand,
        capsys: pytest.CaptureFixture[str],
        response: dict[str, object],
    ) -> None:
        """List registered quotas for either endpoint, excluding usage-only rows."""
        mock_http.return_value = response
        args = parser.parse_args(["quota", "list", "--format", "json"])
        cmd.execute(args)

        assert json.loads(capsys.readouterr().out)["metrics"]["quotas"] == [
            {
                "cache_salt": "tenant1",
                "limit_gb": 2.5,
                "current_usage_gb": 0.125,
            }
        ]

    @pytest.mark.parametrize(
        "entries",
        [
            [],
            [
                {
                    "cache_salt": "usage-only",
                    "quota_limit_gb": 0.0,
                    "usage_gb": 0.5,
                    "quota_exists": False,
                }
            ],
        ],
    )
    @patch("lmcache.cli.commands.quota.list_command.http_request")
    def test_list_coordinator_without_explicit_quotas(
        self,
        mock_http: MagicMock,
        parser: argparse.ArgumentParser,
        cmd: QuotaCommand,
        capsys: pytest.CaptureFixture[str],
        entries: list[dict[str, object]],
    ) -> None:
        """An empty quota registry stays empty even when the API reports usage."""
        mock_http.return_value = {"total_gb": 0.5, "by_cache_salt": entries}
        args = parser.parse_args(["quota", "list", "--format", "json"])
        cmd.execute(args)

        assert json.loads(capsys.readouterr().out)["metrics"] == {
            "info": "No quotas configured"
        }

    @patch("lmcache.cli.commands.quota.list_command.http_request")
    def test_list_coordinator_explicit_zero_quota(
        self,
        mock_http: MagicMock,
        parser: argparse.ArgumentParser,
        cmd: QuotaCommand,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A registered zero limit stays listed even when usage is also zero."""
        mock_http.return_value = {
            "total_gb": 0.0,
            "by_cache_salt": [
                {
                    "cache_salt": "tenant1",
                    "quota_limit_gb": 0.0,
                    "usage_gb": 0.0,
                    "quota_exists": True,
                }
            ],
        }
        args = parser.parse_args(["quota", "list", "--format", "json"])
        cmd.execute(args)

        assert json.loads(capsys.readouterr().out)["metrics"]["quotas"] == [
            {"cache_salt": "tenant1", "limit_gb": 0.0, "current_usage_gb": 0.0}
        ]

    @patch("lmcache.cli.commands.quota.set_command.http_request")
    def test_set(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {
            "cache_salt": "tenant1",
            "limit_gb": 10.5,
            "status": "ok",
        }
        args = parser.parse_args(["quota", "set", "tenant1", "--limit-gb", "10.5"])
        cmd.execute(args)

        mock_http.assert_called_once_with(
            "PUT",
            "http://localhost:8080/quota/tenant1",
            data={"limit_gb": 10.5},
        )
        out = capsys.readouterr().out
        assert "Quota Set" in out and "tenant1" in out

    @patch("lmcache.cli.commands.quota.get_command.http_request")
    def test_get(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {
            "cache_salt": "tenant1",
            "limit_gb": 10.5,
            "current_usage_gb": 3.27,
            "exists": True,
        }
        args = parser.parse_args(["quota", "get", "tenant1"])
        cmd.execute(args)

        mock_http.assert_called_once_with("GET", "http://localhost:8080/quota/tenant1")
        out = capsys.readouterr().out
        assert "Quota Info" in out and "3.27" in out

    @patch("lmcache.cli.commands.quota.list_command.http_request")
    def test_list(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {
            "users": {
                "tenant1": {"limit_gb": 10.5, "current_usage_gb": 3.27},
                "_default": {"limit_gb": 5.0, "current_usage_gb": 1.82},
            }
        }
        args = parser.parse_args(["quota", "list"])
        cmd.execute(args)

        mock_http.assert_called_once_with("GET", "http://localhost:8080/quota")
        out = capsys.readouterr().out
        assert "tenant1" in out and "_default" in out

    @patch("lmcache.cli.commands.quota.delete_command.http_request")
    def test_delete(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {"cache_salt": "tenant1", "status": "removed"}
        args = parser.parse_args(["quota", "delete", "tenant1"])
        cmd.execute(args)

        mock_http.assert_called_once_with(
            "DELETE",
            "http://localhost:8080/quota/tenant1",
        )
        out = capsys.readouterr().out
        assert "Quota Delete" in out and "removed" in out

    @patch("lmcache.cli.commands.quota.set_command.http_request")
    def test_quiet_suppresses_output(
        self,
        mock_http,
        cmd,
        parser,
        capsys,
    ) -> None:
        mock_http.return_value = {"cache_salt": "t1", "limit_gb": 1.0, "status": "ok"}
        args = parser.parse_args(["quota", "set", "t1", "--limit-gb", "1", "-q"])
        cmd.execute(args)
        assert capsys.readouterr().out == ""
