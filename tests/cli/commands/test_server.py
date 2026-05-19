# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache server`` CLI command."""

# Standard
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import os

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
        assert args.http_host == "0.0.0.0"
        assert args.http_port == 8080

    def test_config_file_can_seed_required_storage_args(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--config-file",
                "/tmp/lmcache-server.yaml",
            ]
        )
        assert args.config_file == "/tmp/lmcache-server.yaml"
        assert args.l1_size_gb is None
        assert args.eviction_policy is None


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

    @patch("lmcache.v1.multiprocess.http_server.run_http_server")
    def test_execute_calls_run_http_server(self, mock_run, parser):
        """execute() should call run_http_server with parsed configs."""
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

    @patch("lmcache.v1.multiprocess.http_server.run_http_server")
    def test_execute_uses_config_file_for_storage_defaults(
        self,
        mock_run,
        parser,
        tmp_path,
    ):
        """Config-file values seed required server startup fields."""
        disk_path = tmp_path / "l2"
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            "\n".join(
                [
                    "chunk_size: 128",
                    "max_local_cpu_size: 2.5",
                    "cache_policy: LRU",
                    f"local_disk: {disk_path}",
                ]
            )
        )
        args = parser.parse_args(["server", "--config-file", str(config_file)])
        cmd = ServerCommand()

        cmd.execute(args)

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        mp_config = kwargs["mp_config"]
        storage_config = kwargs["storage_manager_config"]
        assert mp_config.chunk_size == 128
        assert storage_config.l1_manager_config.memory_config.size_in_bytes == int(
            2.5 * (1 << 30)
        )
        assert storage_config.eviction_config.eviction_policy == "LRU"
        adapter = storage_config.l2_adapter_config.adapters[0]
        assert adapter.base_path == str(disk_path)

    @patch("lmcache.v1.multiprocess.http_server.run_http_server")
    def test_execute_uses_config_file_env_for_storage_defaults(
        self,
        mock_run,
        parser,
        tmp_path,
        monkeypatch,
    ):
        """LMCACHE_CONFIG_FILE seeds required Python server startup fields."""
        disk_path = tmp_path / "env-file-l2"
        config_file = tmp_path / "server-from-env.yaml"
        config_file.write_text(
            "\n".join(
                [
                    "chunk_size: 160",
                    "max_local_cpu_size: 4.5",
                    "cache_policy: LRU",
                    f"local_disk: {disk_path}",
                ]
            )
        )
        monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(config_file))
        args = parser.parse_args(["server"])
        cmd = ServerCommand()

        cmd.execute(args)

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        mp_config = kwargs["mp_config"]
        storage_config = kwargs["storage_manager_config"]
        assert mp_config.chunk_size == 160
        assert storage_config.l1_manager_config.memory_config.size_in_bytes == int(
            4.5 * (1 << 30)
        )
        assert storage_config.eviction_config.eviction_policy == "LRU"
        adapter = storage_config.l2_adapter_config.adapters[0]
        assert adapter.base_path == str(disk_path)

    @patch("lmcache.v1.multiprocess.http_server.run_http_server")
    def test_execute_config_file_precedes_engine_env(
        self,
        mock_run,
        parser,
        tmp_path,
        monkeypatch,
    ):
        """Config-file startup values win over supported LMCache env vars."""
        file_disk_path = tmp_path / "file-l2"
        env_disk_path = tmp_path / "env-l2"
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            "\n".join(
                [
                    "chunk_size: 128",
                    "max_local_cpu_size: 2.5",
                    "cache_policy: LRU",
                    f"local_disk: {file_disk_path}",
                ]
            )
        )
        monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "96")
        monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "3.5")
        monkeypatch.setenv("LMCACHE_CACHE_POLICY", "noop")
        monkeypatch.setenv("LMCACHE_LOCAL_DISK", str(env_disk_path))
        args = parser.parse_args(["server", "--config-file", str(config_file)])
        cmd = ServerCommand()

        cmd.execute(args)

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        mp_config = kwargs["mp_config"]
        storage_config = kwargs["storage_manager_config"]
        assert mp_config.chunk_size == 128
        assert storage_config.l1_manager_config.memory_config.size_in_bytes == int(
            2.5 * (1 << 30)
        )
        assert storage_config.eviction_config.eviction_policy == "LRU"
        adapter = storage_config.l2_adapter_config.adapters[0]
        assert adapter.base_path == str(file_disk_path)

    @patch("lmcache.v1.multiprocess.http_server.run_http_server")
    def test_execute_uses_env_for_storage_defaults(
        self,
        mock_run,
        parser,
        tmp_path,
        monkeypatch,
    ):
        """Supported LMCache env vars seed Python server startup fields."""
        disk_path = tmp_path / "env-l2"
        monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "96")
        monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "3.5")
        monkeypatch.setenv("LMCACHE_CACHE_POLICY", "LRU")
        monkeypatch.setenv("LMCACHE_LOCAL_DISK", str(disk_path))
        args = parser.parse_args(["server"])
        cmd = ServerCommand()

        cmd.execute(args)

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        mp_config = kwargs["mp_config"]
        storage_config = kwargs["storage_manager_config"]
        assert mp_config.chunk_size == 96
        assert storage_config.l1_manager_config.memory_config.size_in_bytes == int(
            3.5 * (1 << 30)
        )
        assert storage_config.eviction_config.eviction_policy == "LRU"
        adapter = storage_config.l2_adapter_config.adapters[0]
        assert adapter.base_path == str(disk_path)

    @patch("lmcache.v1.multiprocess.native_launcher.run_native_server")
    def test_execute_native_uses_config_file_before_launch(
        self,
        mock_run_native,
        parser,
        tmp_path,
    ):
        """Native launch sees supported config-file values on parsed args."""
        disk_path = tmp_path / "native-l2"
        config_file = tmp_path / "native-server.yaml"
        config_file.write_text(
            "\n".join(
                [
                    "chunk_size: 64",
                    "max_local_cpu_size: 1.25",
                    "cache_policy: LRU",
                    f"local_disk: {disk_path}",
                ]
            )
        )
        args = parser.parse_args(
            ["server", "--native", "--config-file", str(config_file)]
        )
        cmd = ServerCommand()

        cmd.execute(args)

        mock_run_native.assert_called_once()
        native_args = mock_run_native.call_args.args[0]
        assert native_args.chunk_size == 64
        assert native_args.l1_size_gb == 1.25
        assert native_args.eviction_policy == "LRU"
        assert json.loads(native_args.l2_adapter[0]) == {
            "base_path": str(disk_path),
            "type": "fs",
        }

    @patch("lmcache.v1.multiprocess.native_launcher.run_native_server")
    def test_execute_native_uses_config_file_env_before_launch(
        self,
        mock_run_native,
        parser,
        tmp_path,
        monkeypatch,
    ):
        """Native launch sees LMCACHE_CONFIG_FILE values on parsed args."""
        disk_path = tmp_path / "native-env-file-l2"
        config_file = tmp_path / "native-server-from-env.yaml"
        config_file.write_text(
            "\n".join(
                [
                    "chunk_size: 192",
                    "max_local_cpu_size: 2.25",
                    "cache_policy: LRU",
                    f"local_disk: {disk_path}",
                ]
            )
        )
        monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(config_file))
        args = parser.parse_args(["server", "--native"])
        cmd = ServerCommand()

        cmd.execute(args)

        mock_run_native.assert_called_once()
        native_args = mock_run_native.call_args.args[0]
        assert native_args.chunk_size == 192
        assert native_args.l1_size_gb == 2.25
        assert native_args.eviction_policy == "LRU"
        assert json.loads(native_args.l2_adapter[0]) == {
            "base_path": str(disk_path),
            "type": "fs",
        }

    @patch("lmcache.v1.multiprocess.native_launcher.run_native_server")
    def test_execute_native_config_file_precedes_engine_env(
        self,
        mock_run_native,
        parser,
        tmp_path,
        monkeypatch,
    ):
        """Native launch ignores engine env overrides when config file exists."""
        file_disk_path = tmp_path / "native-file-l2"
        env_disk_path = tmp_path / "native-env-l2"
        config_file = tmp_path / "native-server.yaml"
        config_file.write_text(
            "\n".join(
                [
                    "chunk_size: 64",
                    "max_local_cpu_size: 1.25",
                    "cache_policy: LRU",
                    f"local_disk: {file_disk_path}",
                ]
            )
        )
        monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "80")
        monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "1.75")
        monkeypatch.setenv("LMCACHE_CACHE_POLICY", "noop")
        monkeypatch.setenv("LMCACHE_LOCAL_DISK", str(env_disk_path))
        monkeypatch.setenv("LMCACHE_REMOTE_URL", "redis://localhost:6379")
        args = parser.parse_args(
            ["server", "--native", "--config-file", str(config_file)]
        )
        cmd = ServerCommand()

        cmd.execute(args)

        mock_run_native.assert_called_once()
        native_args = mock_run_native.call_args.args[0]
        assert native_args.chunk_size == 64
        assert native_args.l1_size_gb == 1.25
        assert native_args.eviction_policy == "LRU"
        assert json.loads(native_args.l2_adapter[0]) == {
            "base_path": str(file_disk_path),
            "type": "fs",
        }

    @patch("lmcache.v1.multiprocess.native_launcher.run_native_server")
    def test_execute_native_uses_env_before_launch(
        self,
        mock_run_native,
        parser,
        tmp_path,
        monkeypatch,
    ):
        """Native launch sees supported LMCache env values on parsed args."""
        disk_path = tmp_path / "native-env-l2"
        monkeypatch.setenv("LMCACHE_MP_NATIVE", "1")
        monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "80")
        monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "1.75")
        monkeypatch.setenv("LMCACHE_CACHE_POLICY", "LRU")
        monkeypatch.setenv("LMCACHE_LOCAL_DISK", str(disk_path))
        args = parser.parse_args(["server"])
        cmd = ServerCommand()

        cmd.execute(args)

        mock_run_native.assert_called_once()
        native_args = mock_run_native.call_args.args[0]
        assert native_args.chunk_size == 80
        assert native_args.l1_size_gb == 1.75
        assert native_args.eviction_policy == "LRU"
        assert json.loads(native_args.l2_adapter[0]) == {
            "base_path": str(disk_path),
            "type": "fs",
        }

    @patch("lmcache.v1.multiprocess.native_launcher.run_native_server")
    def test_execute_native_rejects_unsupported_config_file_mode(
        self,
        mock_run_native,
        parser,
        tmp_path,
        capsys,
    ):
        """Native launch fails before exec for unsupported config-file modes."""
        config_file = tmp_path / "native-unsupported.yaml"
        config_file.write_text(
            "\n".join(
                [
                    "max_local_cpu_size: 1",
                    "cache_policy: LRU",
                    "remote_url: redis://localhost:6379",
                ]
            )
        )
        args = parser.parse_args(
            ["server", "--native", "--config-file", str(config_file)]
        )
        cmd = ServerCommand()

        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(args)

        assert exc_info.value.code == 2
        mock_run_native.assert_not_called()
        assert "remote storage" in capsys.readouterr().err


class TestNativeLauncher:
    def test_native_argv_defaults_to_no_cuda(self, parser, monkeypatch):
        # First Party
        from lmcache.v1.multiprocess import native_launcher

        enable_cuda_values: list[bool] = []

        def fake_ensure_native_binary(*, enable_cuda: bool = False) -> Path:
            enable_cuda_values.append(enable_cuda)
            return Path("/tmp/lmcache-mp-server-native")

        monkeypatch.setattr(
            native_launcher, "ensure_native_binary", fake_ensure_native_binary
        )
        args = parser.parse_args(
            ["server", "--native", "--l1-size-gb", "1", "--eviction-policy", "LRU"]
        )

        native_launcher.native_argv_from_args(args)

        assert enable_cuda_values == [False]

    def test_native_argv_uses_cuda_for_explicit_cuda_request(
        self, parser, monkeypatch
    ):
        # First Party
        from lmcache.v1.multiprocess import native_launcher

        enable_cuda_values: list[bool] = []

        def fake_ensure_native_binary(*, enable_cuda: bool = False) -> Path:
            enable_cuda_values.append(enable_cuda)
            return Path("/tmp/lmcache-mp-server-native")

        monkeypatch.setattr(
            native_launcher, "ensure_native_binary", fake_ensure_native_binary
        )
        args = parser.parse_args(
            [
                "server",
                "--native-cuda",
                "--l1-size-gb",
                "1",
                "--eviction-policy",
                "LRU",
            ]
        )

        native_launcher.native_argv_from_args(args)

        assert enable_cuda_values == [True]

    def test_native_argv_uses_cuda_for_explicit_cuda_binary_env(
        self, parser, monkeypatch
    ):
        # First Party
        from lmcache.v1.multiprocess import native_launcher

        enable_cuda_values: list[bool] = []
        monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA_BINARY", "/tmp/native-cuda")

        def fake_ensure_native_binary(*, enable_cuda: bool = False) -> Path:
            enable_cuda_values.append(enable_cuda)
            return Path("/tmp/native-cuda")

        monkeypatch.setattr(
            native_launcher, "ensure_native_binary", fake_ensure_native_binary
        )
        args = parser.parse_args(
            ["server", "--native", "--l1-size-gb", "1", "--eviction-policy", "LRU"]
        )

        native_launcher.native_argv_from_args(args)

        assert enable_cuda_values == [True]

    def test_native_argv_rejects_zero_chunk_size(self, parser, monkeypatch):
        # First Party
        from lmcache.v1.multiprocess import native_launcher

        monkeypatch.setattr(
            native_launcher,
            "ensure_native_binary",
            lambda *, enable_cuda=False: Path("/tmp/lmcache-mp-server-native"),
        )
        args = parser.parse_args(
            [
                "server",
                "--native",
                "--l1-size-gb",
                "1",
                "--eviction-policy",
                "LRU",
                "--chunk-size",
                "0",
            ]
        )

        with pytest.raises(ValueError, match="positive --chunk-size"):
            native_launcher.native_argv_from_args(args)

    def test_native_argv_maps_native_disk_path_to_binary_flag(
        self, parser, monkeypatch, tmp_path
    ):
        # First Party
        from lmcache.v1.multiprocess import native_launcher

        monkeypatch.setattr(
            native_launcher,
            "ensure_native_binary",
            lambda *, enable_cuda=False: Path("/tmp/lmcache-mp-server-native"),
        )
        args = parser.parse_args(
            [
                "server",
                "--native",
                "--l1-size-gb",
                "1",
                "--eviction-policy",
                "LRU",
                "--native-disk-path",
                str(tmp_path),
            ]
        )

        argv = native_launcher.native_argv_from_args(args)

        assert "--native-disk-path" not in argv
        assert argv[argv.index("--cxx-disk-path") + 1] == str(tmp_path)

    def test_run_native_server_exec_env_handles_missing_commit_id(
        self, parser, monkeypatch
    ):
        # First Party
        from lmcache import utils
        from lmcache.v1.multiprocess import native_launcher

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            native_launcher,
            "native_argv_from_args",
            lambda args: ["/tmp/lmcache-mp-server-native", "--help"],
        )
        monkeypatch.setattr(utils, "VERSION", None)
        monkeypatch.setattr(utils, "COMMIT_ID", None)

        def fake_execve(path, argv, env):
            captured["path"] = path
            captured["argv"] = argv
            captured["env"] = env
            raise SystemExit(0)

        monkeypatch.setattr(native_launcher.os, "execve", fake_execve)
        args = parser.parse_args(
            ["server", "--native", "--l1-size-gb", "1", "--eviction-policy", "LRU"]
        )

        with pytest.raises(SystemExit) as exc_info:
            native_launcher.run_native_server(args)

        assert exc_info.value.code == 0
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["LMCACHE_NATIVE_VERSION"] == ""
        assert env["LMCACHE_NATIVE_COMMIT_ID"] == ""

    def test_cuda_fallback_does_not_use_no_cuda_path_binary(
        self, tmp_path, monkeypatch
    ):
        # First Party
        from lmcache.v1.multiprocess import native_launcher

        monkeypatch.setattr(
            native_launcher, "native_source_dir", lambda: tmp_path / "missing"
        )
        monkeypatch.setattr(
            native_launcher,
            "packaged_native_binary_path",
            lambda *, enable_cuda=False: tmp_path / "missing-packaged",
        )
        monkeypatch.setattr(
            native_launcher.shutil,
            "which",
            lambda name: "/usr/bin/lmcache-mp-server-native"
            if name == "lmcache-mp-server-native"
            else None,
        )

        assert (
            native_launcher.ensure_native_binary(enable_cuda=False)
            == Path("/usr/bin/lmcache-mp-server-native")
        )
        with pytest.raises(FileNotFoundError, match="lmcache-mp-server-native-cuda"):
            native_launcher.ensure_native_binary(enable_cuda=True)

    def test_source_freshness_includes_cuda_and_cmake(self, tmp_path):
        # First Party
        from lmcache.v1.multiprocess import native_launcher

        source_dir = tmp_path / "native"
        source_dir.mkdir()
        binary = tmp_path / "lmcache-mp-server-native"
        binary.write_text("binary")
        cuda_source = source_dir / "native_transfer_kernel.cu"
        cuda_source.write_text("kernel")
        os.utime(binary, (1, 1))
        os.utime(cuda_source, (2, 2))

        assert native_launcher._source_is_newer(binary, source_dir)

        os.utime(cuda_source, (1, 1))
        cmake = source_dir / "CMakeLists.txt"
        cmake.write_text("cmake")
        os.utime(cmake, (3, 3))

        assert native_launcher._source_is_newer(binary, source_dir)

    @patch("lmcache.v1.multiprocess.native_launcher.run_native_server")
    def test_execute_native_rejects_unsupported_env_mode(
        self,
        mock_run_native,
        parser,
        monkeypatch,
        capsys,
    ):
        """Native launch fails before exec for unsupported LMCache env modes."""
        monkeypatch.setenv("LMCACHE_MP_NATIVE", "1")
        monkeypatch.setenv("LMCACHE_REMOTE_URL", "redis://localhost:6379")
        args = parser.parse_args(
            ["server", "--l1-size-gb", "1", "--eviction-policy", "LRU"]
        )
        cmd = ServerCommand()

        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(args)

        assert exc_info.value.code == 2
        mock_run_native.assert_not_called()
        assert "remote storage" in capsys.readouterr().err
