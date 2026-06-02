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


class FakeFuture:
    """Minimal future double for ZMQ health checks."""

    def __init__(
        self,
        result: bool | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result_value = result
        self.error = error
        self.timeout: float | None = None

    def result(self, timeout: float | None = None) -> bool | None:
        """Return the configured result or raise the configured error."""
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        return self.result_value


class FakeMessageQueueClient:
    """Minimal message queue client double for health checks."""

    instances: list["FakeMessageQueueClient"] = []
    future = FakeFuture(True)

    def __init__(self, server_url: str, context: object) -> None:
        self.server_url = server_url
        self.context = context
        self.closed = False
        self.requests: list[tuple[object, list[object], object]] = []
        self.instances.append(self)

    def submit_request(
        self,
        request_type: object,
        request_payloads: list[object],
        response_cls: object,
    ) -> FakeFuture:
        """Record the request and return the configured fake future."""
        self.requests.append((request_type, request_payloads, response_cls))
        return self.future

    def close(self) -> None:
        """Record that the client was closed."""
        self.closed = True


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


def test_config_parses_enabled_string() -> None:
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": "true"},
        server_host="tcp://localhost",
        server_port="5555",
    )

    assert config.enabled
    assert config.host == "localhost"
    assert config.port == 5555


def test_config_parses_server_args() -> None:
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={
            "lmcache.mp.autostart": True,
            "lmcache.mp.autostart.server_args": ("--http-port 18080 --l1-size-gb 20"),
        },
        server_host="127.0.0.1",
        server_port=5555,
    )

    assert config.server_args == ("--http-port", "18080", "--l1-size-gb", "20")


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


@pytest.mark.parametrize(
    ("extra_config", "server_port", "match"),
    [
        (
            {"lmcache.mp.autostart": "maybe"},
            5555,
            "must be a boolean",
        ),
        (
            {
                "lmcache.mp.autostart": True,
                "lmcache.mp.autostart.server_args": ["invalid"],
            },
            5555,
            "must be a string",
        ),
        (
            {
                "lmcache.mp.autostart": True,
                "lmcache.mp.autostart.wait_timeout": "invalid",
            },
            5555,
            "must be a number",
        ),
        (
            {
                "lmcache.mp.autostart": True,
                "lmcache.mp.autostart.wait_timeout": 0,
            },
            5555,
            "must be positive",
        ),
        (
            {"lmcache.mp.autostart": True},
            "invalid",
            "must be an integer",
        ),
        (
            {"lmcache.mp.autostart": True},
            0,
            "must be positive",
        ),
    ],
)
def test_config_rejects_invalid_enabled_values(
    extra_config: dict[str, object],
    server_port: int | str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MPServerAutostartConfig.from_extra_config(
            extra_config=extra_config,
            server_host="tcp://localhost",
            server_port=server_port,
        )


def test_maybe_autostart_starts_when_enabled(monkeypatch) -> None:
    instances = []

    class FakeLauncher:
        def __init__(self, config: MPServerAutostartConfig) -> None:
            self.config = config
            self.started = False
            self.server_url = None
            self.zmq_context = None
            instances.append(self)

        def start(self, server_url: str, zmq_context: object) -> None:
            self.started = True
            self.server_url = server_url
            self.zmq_context = zmq_context

    monkeypatch.setattr(launcher_mod, "MPServerLauncher", FakeLauncher)
    zmq_context = MagicMock()

    launcher = maybe_autostart_mp_server(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
        server_url="tcp://localhost:5555",
        zmq_context=zmq_context,
    )

    assert launcher is instances[0]
    assert instances[0].started
    assert instances[0].server_url == "tcp://localhost:5555"
    assert instances[0].zmq_context is zmq_context
    assert len(instances) == 1


def test_shutdown_mp_server_launcher_ignores_none() -> None:
    shutdown_mp_server_launcher(None)


def test_shutdown_mp_server_launcher_shutdowns_owned_launcher() -> None:
    launcher = MagicMock()

    shutdown_mp_server_launcher(launcher)

    launcher.shutdown.assert_called_once_with()


def test_launcher_skips_start_when_server_already_healthy(monkeypatch) -> None:
    popen_mock = MagicMock()
    monkeypatch.setattr(
        launcher_mod,
        "is_mp_server_healthy",
        lambda server_url, zmq_context: True,
    )
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", popen_mock)
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
    )

    MPServerLauncher(config).start("tcp://localhost:5555", MagicMock())

    popen_mock.assert_not_called()


def test_launcher_starts_command_and_waits_until_healthy(monkeypatch) -> None:
    health_results = iter([False, True])
    process = FakeProcess()
    popen_mock = MagicMock(return_value=process)
    monkeypatch.setattr(
        launcher_mod,
        "is_mp_server_healthy",
        lambda server_url, zmq_context: next(health_results),
    )
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", popen_mock)
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
    )

    MPServerLauncher(config).start("tcp://localhost:5555", MagicMock())

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
    monkeypatch.setattr(
        launcher_mod,
        "is_mp_server_healthy",
        lambda server_url, zmq_context: False,
    )
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
        MPServerLauncher(config).start("tcp://localhost:5555", MagicMock())

    assert process.terminate_called
    assert not process.kill_called


def test_launcher_early_exit_raises_connection_error(monkeypatch) -> None:
    process = FakeProcess(returncode=2)
    popen_mock = MagicMock(return_value=process)
    monkeypatch.setattr(
        launcher_mod,
        "is_mp_server_healthy",
        lambda server_url, zmq_context: False,
    )
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", popen_mock)
    monkeypatch.setattr(launcher_mod.time, "monotonic", lambda: 0.0)
    config = MPServerAutostartConfig.from_extra_config(
        extra_config={"lmcache.mp.autostart": True},
        server_host="tcp://localhost",
        server_port=5555,
    )

    with pytest.raises(ConnectionError, match="exited before becoming healthy"):
        MPServerLauncher(config).start("tcp://localhost:5555", MagicMock())


def test_health_probe_sends_zmq_ping_and_closes_client(monkeypatch) -> None:
    FakeMessageQueueClient.instances = []
    FakeMessageQueueClient.future = FakeFuture(True)
    monkeypatch.setattr(
        launcher_mod,
        "_create_message_queue_client",
        FakeMessageQueueClient,
    )
    monkeypatch.setattr(
        launcher_mod,
        "_submit_ping",
        lambda client: client.submit_request("PING", [], bool),
    )
    zmq_context = MagicMock()

    assert launcher_mod.is_mp_server_healthy(
        "tcp://localhost:5555",
        zmq_context,
        timeout=0.25,
    )

    client = FakeMessageQueueClient.instances[0]
    assert client.server_url == "tcp://localhost:5555"
    assert client.context is zmq_context
    assert client.closed
    assert client.requests == [("PING", [], bool)]
    assert FakeMessageQueueClient.future.timeout == 0.25


def test_health_probe_returns_false_on_zmq_ping_timeout(monkeypatch) -> None:
    FakeMessageQueueClient.instances = []
    FakeMessageQueueClient.future = FakeFuture(error=TimeoutError())
    monkeypatch.setattr(
        launcher_mod,
        "_create_message_queue_client",
        FakeMessageQueueClient,
    )
    monkeypatch.setattr(
        launcher_mod,
        "_submit_ping",
        lambda client: client.submit_request("PING", [], bool),
    )

    assert not launcher_mod.is_mp_server_healthy(
        "tcp://localhost:5555",
        MagicMock(),
    )
    assert FakeMessageQueueClient.instances[0].closed


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
        launcher_mod,
        "is_mp_server_healthy",
        lambda server_url, zmq_context: next(health_results),
    )
    process = StubbornProcess()
    monkeypatch.setattr(
        launcher_mod.subprocess, "Popen", MagicMock(return_value=process)
    )
    launcher = MPServerLauncher(config)

    launcher.start("tcp://localhost:5555", MagicMock())
    launcher.shutdown()

    assert process.terminate_called
    assert process.kill_called
