# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import subprocess

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import mp_server_launcher as launcher_mod
from lmcache.integration.vllm.mp_server_launcher import (
    MPServerAutostartConfig,
    MPServerLauncher,
    maybe_autostart_mp_server,
    shutdown_mp_server_launcher,
)


class FakeProcess:
    """Minimal process double for subprocess lifecycle tests."""

    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        """Return the configured process return code."""
        return self.returncode

    def terminate(self) -> None:
        """Record termination and mark the process as exited."""
        self.terminate_called = True
        self.returncode = 0

    def kill(self) -> None:
        """Record forced termination and mark the process as exited."""
        self.kill_called = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        """Record wait calls and return the process return code."""
        self.wait_calls.append(timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakeResponse:
    """Minimal urlopen response double for health checks."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body

    def __enter__(self) -> "FakeResponse":
        """Return the response for context-manager use."""
        return self

    def __exit__(self, *args: object) -> None:
        """No-op context-manager cleanup."""
        return None

    def read(self) -> bytes:
        """Return the configured response body."""
        return self.body


def test_config_defaults_to_disabled_without_validating_remote_host() -> None:
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={
            "lmcache.mp.autostart.server_args": ["invalid"],
            "lmcache.mp.autostart.wait_timeout": "invalid",
        },
        server_host="tcp://192.168.1.10",
        server_port=5555,
    )

    assert not config.enabled
    assert config.server_args == ()


def test_config_ignores_non_mapping_extra_config() -> None:
    config = MPServerAutostartConfig.from_extra_config(
        extra_config=object(),
        server_host="tcp://localhost",
        server_port=5555,
    )

    assert not config.enabled


def test_config_parses_enabled_string_and_default_health_url() -> None:
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": "true"},
        server_host="tcp://localhost",
        server_port="5555",
    )

    assert config.enabled
    assert config.host == "localhost"
    assert config.port == 5555
    assert config.health_url == "http://127.0.0.1:8080/healthcheck"


def test_config_parses_server_args_and_http_port() -> None:
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={
            "lmcache.mp.autostart": True,
            "lmcache.mp.autostart.server_args": (
                "--http-port 18080 --l1-size-gb 20"
            ),
        },
        server_host="127.0.0.1",
        server_port=5555,
    )

    assert config.server_args == ("--http-port", "18080", "--l1-size-gb", "20")
    assert config.health_url == "http://127.0.0.1:18080/healthcheck"


@pytest.mark.parametrize(
    ("server_host", "expected_host"),
    [
        ("::1", "::1"),
        ("[::1]:5555", "::1"),
        ("tcp://::1", "::1"),
        ("tcp://[::1]:5555", "::1"),
    ],
)
def test_config_accepts_ipv6_loopback_hosts(
    server_host: str,
    expected_host: str,
) -> None:
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": True},
        server_host=server_host,
        server_port=5555,
    )

    assert config.host == expected_host


def test_config_rejects_remote_host_when_enabled() -> None:
    with pytest.raises(ValueError, match="only supports local hosts"):
        MPServerAutostartConfig.from_extra_config(
            extra_config={"lmcache.mp.autostart": True},
            server_host="tcp://192.168.1.10",
            server_port=5555,
        )


def test_maybe_autostart_only_starts_rank_zero(monkeypatch) -> None:
    instances = []

    class FakeLauncher:
        def __init__(self, config: MPServerAutostartConfig) -> None:
            self.config = config
            self.started = False
            instances.append(self)

        def start(self) -> None:
            self.started = True

    monkeypatch.setattr(launcher_mod, "MPServerLauncher", FakeLauncher)

    worker_rank_zero = maybe_autostart_mp_server(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
        rank=0,
    )
    worker_rank_one = maybe_autostart_mp_server(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
        rank=1,
    )

    assert worker_rank_zero is instances[0]
    assert instances[0].started
    assert worker_rank_one is None
    assert len(instances) == 1


def test_shutdown_mp_server_launcher_ignores_none() -> None:
    shutdown_mp_server_launcher(None)


def test_shutdown_mp_server_launcher_shutdowns_owned_launcher() -> None:
    launcher = MagicMock()

    shutdown_mp_server_launcher(launcher)

    launcher.shutdown.assert_called_once_with()


def test_launcher_skips_start_when_server_already_healthy(monkeypatch) -> None:
    popen_mock = MagicMock()
    monkeypatch.setattr(launcher_mod, "is_mp_server_healthy", lambda _: True)
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", popen_mock)
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
    )

    MPServerLauncher(config).start()

    popen_mock.assert_not_called()


def test_launcher_starts_command_and_waits_until_healthy(monkeypatch) -> None:
    health_results = iter([False, True])
    process = FakeProcess()
    popen_mock = MagicMock(return_value=process)
    monkeypatch.setattr(
        launcher_mod, "is_mp_server_healthy", lambda _: next(health_results)
    )
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", popen_mock)
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
    )

    MPServerLauncher(config).start()

    popen_mock.assert_called_once()
    command = popen_mock.call_args.args[0]
    assert command[:3] == [
        launcher_mod.sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
    ]
    assert command[3:] == ["--host", "localhost", "--port", "5555"]


def test_launcher_timeout_terminates_process(monkeypatch) -> None:
    process = FakeProcess()
    popen_mock = MagicMock(return_value=process)
    monotonic_values = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(launcher_mod, "is_mp_server_healthy", lambda _: False)
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", popen_mock)
    monkeypatch.setattr(launcher_mod.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(launcher_mod.time, "sleep", lambda _: None)
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={
            "lmcache.mp.autostart": True,
            "lmcache.mp.autostart.wait_timeout": 0.1,
        },
        server_host="tcp://localhost",
        server_port=5555,
    )

    with pytest.raises(ConnectionError, match="did not become healthy"):
        MPServerLauncher(config).start()

    assert process.terminate_called
    assert not process.kill_called


def test_launcher_early_exit_raises_connection_error(monkeypatch) -> None:
    process = FakeProcess(returncode=2)
    popen_mock = MagicMock(return_value=process)
    monkeypatch.setattr(launcher_mod, "is_mp_server_healthy", lambda _: False)
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", popen_mock)
    monkeypatch.setattr(launcher_mod.time, "monotonic", lambda: 0.0)
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
    )

    with pytest.raises(ConnectionError, match="exited before becoming healthy"):
        MPServerLauncher(config).start()


def test_health_probe_requires_healthy_json(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher_mod,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(200, b'{"status": "healthy"}'),
    )

    assert launcher_mod.is_mp_server_healthy("http://127.0.0.1:8080/healthcheck")


def test_health_probe_returns_false_for_non_healthy_json(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher_mod,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(200, b'{"status": "starting"}'),
    )

    assert not launcher_mod.is_mp_server_healthy(
        "http://127.0.0.1:8080/healthcheck"
    )


def test_shutdown_kills_process_after_terminate_timeout(monkeypatch) -> None:
    class StubbornProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls.append(timeout)
            if not self.kill_called:
                raise subprocess.TimeoutExpired(cmd="server", timeout=timeout)
            self.returncode = 0
            return self.returncode

    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
    )
    health_results = iter([False, True])
    monkeypatch.setattr(
        launcher_mod, "is_mp_server_healthy", lambda _: next(health_results)
    )
    process = StubbornProcess()
    monkeypatch.setattr(
        launcher_mod.subprocess, "Popen", MagicMock(return_value=process)
    )
    launcher = MPServerLauncher(config)

    launcher.start()
    launcher.shutdown()

    assert process.terminate_called
    assert process.kill_called
