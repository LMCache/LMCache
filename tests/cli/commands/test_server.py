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
from lmcache.cli.commands import base, server
from lmcache.cli.commands.server import ServerCommand
from lmcache.v1.multiprocess.config import parse_args_to_p2p_config
from lmcache.v1.transport.grpc import _proto_gen
from tests.import_utils import lmcache_subprocess_env
import lmcache


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
    import_root = Path(lmcache.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "lmcache.cli.main", "server", "--help"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(tmp_path), str(import_root), os.environ.get("PYTHONPATH", ""))
            ),
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

    def test_p2p_write_args_reach_the_p2p_config(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
                "--p2p-advertise-url",
                "10.0.0.1:8700",
                "--enable-p2p-write",
                "--p2p-write-l1-usage-threshold",
                "0.5",
                "--p2p-store-timeout",
                "12",
            ]
        )
        config = parse_args_to_p2p_config(args)
        assert config.enable_write is True
        assert config.write_l1_usage_threshold == 0.5
        assert config.store_timeout == 12.0

    def test_p2p_writes_are_off_by_default(self, parser):
        args = parser.parse_args(
            ["server", "--l1-size-gb", "4", "--eviction-policy", "LRU"]
        )
        config = parse_args_to_p2p_config(args)
        assert config.enable_write is False
        assert config.write_l1_usage_threshold == 0.7
        assert config.store_timeout == 30.0

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

    @pytest.mark.parametrize("option", ["--api-key-file", "--http-api-key-file"])
    def test_api_key_files_are_repeatable(self, parser, option):
        args = parser.parse_args(
            [
                "server",
                option,
                "/run/secrets/current-key",
                option,
                "/run/secrets/next-key",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.api_key_file == [
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

    def test_ungenerated_bindings_are_reported_at_registration(
        self, tmp_path: Path
    ) -> None:
        """A build that skipped the gRPC generator names it, not the CLI package.

        The generator output is hidden by pointing its package at an empty
        directory. LMCache loggers do not propagate, so the warning is read
        from the module logger rather than ``caplog``.
        """
        with (
            patch.dict(sys.modules, {"lmcache.v1.distributed.config": None}),
            patch.object(_proto_gen, "__path__", [str(tmp_path)]),
            patch.object(server.logger, "warning") as mock_warning,
        ):
            ServerCommand().add_arguments(argparse.ArgumentParser())

        logged = " ".join(map(str, mock_warning.call_args.args))
        assert "_generate.py" in logged
        assert "lightweight" not in logged


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

    def test_execute_explains_ungenerated_bindings(
        self,
        parser: argparse.ArgumentParser,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A build that skipped the gRPC generator points at the generator."""
        args = parser.parse_args(["server"])
        with (
            patch.dict(sys.modules, {"lmcache.v1.distributed.config": None}),
            patch.object(_proto_gen, "__path__", [str(tmp_path)]),
            pytest.raises(SystemExit) as excinfo,
        ):
            ServerCommand().execute(args)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "_generate.py" in err
        assert "requirements/proto.txt" in err
        assert "CUDA extensions" not in err

    @pytest.mark.skipif(
        importlib.util.find_spec("lmcache.v1.transport.grpc._proto_gen.p2p_service_pb2")
        is None,
        reason="Requires the generated gRPC bindings",
    )
    def test_execute_reports_the_import_error_when_bindings_exist(
        self,
        parser: argparse.ArgumentParser,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A complete install shows the failing import, not a fix-it hint."""
        args = parser.parse_args(["server"])
        with (
            patch.dict(sys.modules, {"lmcache.v1.distributed.config": None}),
            pytest.raises(SystemExit) as excinfo,
        ):
            ServerCommand().execute(args)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "lmcache.v1.distributed.config" in err
        assert "pip install lmcache" in err
        assert "_generate.py" not in err

    def test_execute_asks_for_the_full_package_outside_a_checkout(
        self,
        parser: argparse.ArgumentParser,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An installed wheel without bindings gets the reinstall advice only.

        The package is relocated to a site-packages-like directory with no
        ``requirements/proto.txt`` next to it, and the generator output is
        hidden as in the bindings-missing cases.
        """
        args = parser.parse_args(["server"])
        no_bindings = tmp_path / "no_bindings"
        no_bindings.mkdir()
        installed = (
            tmp_path / "site-packages" / "lmcache" / "cli" / "commands" / "base.py"
        )
        with (
            patch.dict(sys.modules, {"lmcache.v1.distributed.config": None}),
            patch.object(_proto_gen, "__path__", [str(no_bindings)]),
            patch.object(base, "__file__", str(installed)),
            pytest.raises(SystemExit) as excinfo,
        ):
            ServerCommand().execute(args)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "pip install lmcache" in err
        assert "_generate.py" not in err

    def test_cli_names_the_generator_when_bindings_are_missing(
        self, tmp_path: Path
    ) -> None:
        """``lmcache server`` through the real entry point names the generator.

        The generated files are hidden by pointing both ``_proto_gen``
        packages at an empty directory before the CLI runs.
        """
        empty = tmp_path / "no_bindings"
        empty.mkdir()
        script = (
            "import lmcache.v1.transport.grpc._proto_gen as transport\n"
            "import lmcache.v1.memory_orchestrator._proto_gen as orchestrator\n"
            f"transport.__path__ = [{str(empty)!r}]\n"
            f"orchestrator.__path__ = [{str(empty)!r}]\n"
            "from lmcache.cli.main import main\n"
            "main(['server'])\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env={
                **lmcache_subprocess_env(),
                "LMCACHE_TRACK_USAGE": "false",
                "LMCACHE_DISABLE_BANNER": "1",
            },
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 1, result.stderr
        assert "_generate.py" in result.stderr
        assert "requirements/proto.txt" in result.stderr
        assert "CUDA extensions" not in result.stderr
