# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache server`` CLI command."""

# Standard
from pathlib import Path
from unittest.mock import patch
import argparse
import importlib.util
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.cli.commands.server import ServerCommand


@pytest.fixture
def cmd():
    return ServerCommand()


@pytest.fixture
def parser(cmd):
    """An ArgumentParser with ServerCommand's arguments registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    cmd.register(sub)
    return p


@pytest.mark.skipif(
    importlib.util.find_spec("lmcache.lmcache_native") is None,
    reason="Fresh CLI process requires the native extension",
)
def test_server_help_with_incompatible_bigtable(tmp_path: Path) -> None:
    """A broken Bigtable dependency does not prevent full server help output."""
    bigtable_package = tmp_path / "google" / "cloud" / "bigtable"
    bigtable_package.mkdir(parents=True)
    bigtable_package.joinpath("__init__.py").write_text(
        'raise TypeError("Descriptors cannot be created directly")\n'
    )
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-m", "lmcache.cli.main", "server", "--help"],
        cwd=repo_root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(repo_root))),
            "LMCACHE_TRACK_USAGE": "false",
            "LMCACHE_DISABLE_BANNER": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "TypeError: Descriptors cannot be created directly" in result.stderr
    assert "bigtable_l2_adapter" in result.stderr
    assert "--l2-adapter" in result.stdout
    assert "--http-port" in result.stdout
    assert "mock" in result.stdout


class TestServerCommandMetadata:
    def test_name(self, cmd):
        assert cmd.name() == "server"

    def test_help(self, cmd):
        assert "server" in cmd.help().lower()


class TestServerCommandArguments:
    def test_registers_subcommand(self, parser):
        """The 'server' subcommand should be parseable."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert hasattr(args, "func")

    def test_mp_server_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--host",
                "0.0.0.0",
                "--port",
                "6666",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 6666

    def test_http_frontend_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--http-host",
                "127.0.0.1",
                "--http-port",
                "9000",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.http_host == "127.0.0.1"
        assert args.http_port == 9000

    def test_http_api_key_files_are_repeatable(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--http-api-key-file",
                "/run/secrets/current-key",
                "--http-api-key-file",
                "/run/secrets/next-key",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.http_api_key_file == [
            "/run/secrets/current-key",
            "/run/secrets/next-key",
        ]

    def test_prometheus_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--prometheus-port",
                "9999",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.prometheus_port == 9999

    def test_default_values(self, parser):
        """Required args only — everything else should get defaults."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.host == "localhost"
        assert args.port == 5555
        assert args.http_host == "127.0.0.1"
        assert args.http_port == 8080


class TestServerCommandExecute:
    def test_func_bound_to_execute(self, cmd, parser):
        """parser.parse_args should bind func to ServerCommand.execute."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.func == cmd.execute

    def test_execute_calls_run_http_server(self, parser):
        """execute() should call run_http_server with parsed configs."""
        http_server = pytest.importorskip("lmcache.v1.multiprocess.http_server")
        with patch.object(http_server, "run_http_server") as mock_run:
            args = parser.parse_args(
                [
                    "server",
                    "--l1-size-gb",
                    "4",
                    "--eviction-policy",
                    "LRU",
                ]
            )
            cmd = ServerCommand()
            cmd.execute(args)

            mock_run.assert_called_once()
            kwargs = mock_run.call_args.kwargs
            assert "http_config" in kwargs
            assert "mp_config" in kwargs
            assert "storage_manager_config" in kwargs
            assert "obs_config" in kwargs
            assert "http_authentication" in kwargs
