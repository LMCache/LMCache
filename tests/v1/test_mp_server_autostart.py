# SPDX-License-Identifier: Apache-2.0

# Standard
from types import ModuleType
from unittest.mock import MagicMock
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import mp_server_launcher as launcher_mod
from lmcache.integration.vllm.mp_server_launcher import (
    MPServerAutostartConfig,
    MPServerLauncher,
    maybe_start_mp_server_from_url,
    wait_for_mp_server_from_url,
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


class FakeRequestType:
    """Minimal request type double for health checks."""

    PING = "PING"


def fake_get_response_class(request_type: object) -> type[bool]:
    """Return the fake response class for PING requests."""
    assert request_type == FakeRequestType.PING
    return bool


def fake_module(name: str, attrs: dict[str, object]) -> ModuleType:
    """Build a fake module for lazy-import health probe tests.

    Args:
        name: Fully qualified module name to expose in ``sys.modules``.
        attrs: Attribute names and values the fake module should provide.

    Returns:
        A module object with the requested attributes.
    """
    module = ModuleType(name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    return module


def patch_mp_health_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the modules loaded lazily by ``is_mp_server_healthy``.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to update ``sys.modules``.
    """
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.multiprocess.mq",
        fake_module(
            "lmcache.v1.multiprocess.mq",
            {"MessageQueueClient": FakeMessageQueueClient},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.multiprocess.protocol",
        fake_module(
            "lmcache.v1.multiprocess.protocol",
            {
                "RequestType": FakeRequestType,
                "get_response_class": fake_get_response_class,
            },
        ),
    )


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
    "server_args",
    [
        "--host 0.0.0.0",
        "--host=0.0.0.0",
        "--hos 0.0.0.0",
        "--port 6000",
        "--port=6000",
        "--por=6000",
        "--http-host 0.0.0.0",
        "--http-host=0.0.0.0",
        "--http-ho=0.0.0.0",
    ],
)
def test_config_rejects_endpoint_server_args(server_args: str) -> None:
    with pytest.raises(ValueError, match="cannot override"):
        MPServerAutostartConfig.from_extra_config(
            extra_config={
                "lmcache.mp.autostart": True,
                "lmcache.mp.autostart.server_args": server_args,
            },
            server_host="127.0.0.1",
            server_port=5555,
        )


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
            {"lmcache.mp.autostart": 1},
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


def test_maybe_start_mp_server_from_url_starts_when_enabled(monkeypatch) -> None:
    instances = []

    class FakeLauncher:
        def __init__(self, config: MPServerAutostartConfig) -> None:
            self.config = config
            self.started = False
            self.waited = False
            self.server_url: str | None = None
            self.zmq_context: object | None = None
            instances.append(self)

        def start(self, server_url: str, zmq_context: object) -> None:
            self.started = True
            self.server_url = server_url
            self.zmq_context = zmq_context

        def wait_until_healthy(self, server_url: str, zmq_context: object) -> None:
            self.waited = True
            self.server_url = server_url
            self.zmq_context = zmq_context

    monkeypatch.setattr(launcher_mod, "MPServerLauncher", FakeLauncher)
    zmq_context = MagicMock()

    launcher = maybe_start_mp_server_from_url(
        extra_config={"lmcache.mp.autostart": True},
        server_url="tcp://localhost:5555",
        zmq_context=zmq_context,
    )

    assert launcher is instances[0]
    assert instances[0].config.host == "localhost"
    assert instances[0].config.port == 5555
    assert instances[0].started
    assert not instances[0].waited
    assert instances[0].server_url == "tcp://localhost:5555"
    assert instances[0].zmq_context is zmq_context
    assert len(instances) == 1


def test_wait_for_mp_server_from_url_waits_when_enabled(monkeypatch) -> None:
    instances = []

    class FakeLauncher:
        def __init__(self, config: MPServerAutostartConfig) -> None:
            self.config = config
            self.started = False
            self.waited = False
            instances.append(self)

        def start(self, _server_url: str, _zmq_context: object) -> None:
            self.started = True

        def wait_until_healthy(self, server_url: str, zmq_context: object) -> None:
            self.waited = True
            self.server_url = server_url
            self.zmq_context = zmq_context

    monkeypatch.setattr(launcher_mod, "MPServerLauncher", FakeLauncher)

    wait_for_mp_server_from_url(
        extra_config={"lmcache.mp.autostart": True},
        server_url="tcp://localhost:5555",
        zmq_context=MagicMock(),
    )

    assert len(instances) == 1
    assert instances[0].config.host == "localhost"
    assert instances[0].config.port == 5555
    assert not instances[0].started
    assert instances[0].waited
    assert instances[0].server_url == "tcp://localhost:5555"


def test_maybe_start_mp_server_from_url_ignores_invalid_url_when_disabled() -> None:
    launcher = maybe_start_mp_server_from_url(
        extra_config={"lmcache.mp.autostart": False},
        server_url="not a valid url",
        zmq_context=MagicMock(),
    )

    assert launcher is None


def test_wait_for_mp_server_from_url_ignores_invalid_url_when_disabled() -> None:
    wait_for_mp_server_from_url(
        extra_config={"lmcache.mp.autostart": False},
        server_url="not a valid url",
        zmq_context=MagicMock(),
    )


def test_maybe_start_mp_server_from_url_parses_local_server_url(monkeypatch) -> None:
    instances = []

    class FakeLauncher:
        def __init__(self, config: MPServerAutostartConfig) -> None:
            self.config = config
            instances.append(self)

        def start(self, server_url: str, zmq_context: object) -> None:
            self.server_url = server_url
            self.zmq_context = zmq_context
            return None

        def wait_until_healthy(self, server_url: str, zmq_context: object) -> None:
            self.server_url = server_url
            self.zmq_context = zmq_context
            return None

    monkeypatch.setattr(launcher_mod, "MPServerLauncher", FakeLauncher)
    zmq_context = MagicMock()

    launcher = maybe_start_mp_server_from_url(
        extra_config={"lmcache.mp.autostart": True},
        server_url="tcp://localhost:5555",
        zmq_context=zmq_context,
    )

    assert launcher is instances[0]
    assert instances[0].config.host == "localhost"
    assert instances[0].config.port == 5555


def test_maybe_start_mp_server_from_url_prefers_configured_endpoint(
    monkeypatch,
) -> None:
    instances = []

    class FakeLauncher:
        def __init__(self, config: MPServerAutostartConfig) -> None:
            self.config = config
            instances.append(self)

        def start(self, server_url: str, zmq_context: object) -> None:
            self.server_url = server_url
            self.zmq_context = zmq_context
            return None

        def wait_until_healthy(self, server_url: str, zmq_context: object) -> None:
            self.server_url = server_url
            self.zmq_context = zmq_context
            return None

    monkeypatch.setattr(launcher_mod, "MPServerLauncher", FakeLauncher)
    extra_config = {
        "lmcache.mp.autostart": True,
        "lmcache.mp.host": "::1",
        "lmcache.mp.port": 5555,
    }
    zmq_context = MagicMock()

    launcher = maybe_start_mp_server_from_url(
        extra_config=extra_config,
        server_url="tcp://::1:5555",
        zmq_context=zmq_context,
    )

    assert launcher is instances[0]
    assert instances[0].config.host == "::1"
    assert instances[0].config.port == 5555


def test_maybe_start_mp_server_from_url_rejects_missing_port_when_enabled() -> None:
    with pytest.raises(ValueError, match="Invalid LMCache MP server URL"):
        maybe_start_mp_server_from_url(
            extra_config={"lmcache.mp.autostart": True},
            server_url="tcp://localhost",
            zmq_context=MagicMock(),
        )


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
    assert command[3:] == [
        "--host",
        "localhost",
        "--port",
        "5555",
        "--http-host",
        "localhost",
    ]


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
    patch_mp_health_modules(monkeypatch)
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
    patch_mp_health_modules(monkeypatch)

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
                timeout_value = 0.0 if timeout is None else timeout
                raise subprocess.TimeoutExpired(cmd="server", timeout=timeout_value)
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
