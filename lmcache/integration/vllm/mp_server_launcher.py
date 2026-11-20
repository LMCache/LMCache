# SPDX-License-Identifier: Apache-2.0
"""Helpers for auto-starting the LMCache multiprocess server."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.parse import urlparse
import math
import shlex
import subprocess
import sys
import time

# Third Party
import zmq

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_AUTOSTART_KEY = "lmcache.mp.autostart"
_SERVER_ARGS_KEY = "lmcache.mp.autostart.server_args"
_WAIT_TIMEOUT_KEY = "lmcache.mp.autostart.wait_timeout"

_DEFAULT_WAIT_TIMEOUT = 90.0
_LOCAL_HOSTS = {"localhost", "127.0.0.1"}
_DISALLOWED_SERVER_ARGS = {"--host", "--port", "--http-host"}
_PING_TIMEOUT_SECONDS = 1.0
_POLL_INTERVAL_SECONDS = 0.5
_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def _load_mp_health_dependencies() -> "_RequestClientFactory":
    """Load MQ health probe dependencies when a probe is executed.

    The MQ modules may require torch through transitive imports, while most of
    this module only parses config or builds process commands. Keeping these
    imports lazy lets config-only callers import this launcher without torch.

    Returns:
        The transport client factory used to send a PING request.
    """
    # First Party
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

    return cast(_RequestClientFactory, RequestClientFactory)


def _build_autostart_config_from_url(
    *,
    extra_config: object | None,
    server_url: str,
) -> "MPServerAutostartConfig":
    """Build auto-start configuration from connector config and ZMQ URL.

    Args:
        extra_config: vLLM ``kv_connector_extra_config`` mapping.
        server_url: ZMQ URL used by the connector to reach the MP server.

    Returns:
        Parsed ``MPServerAutostartConfig``.

    Raises:
        ValueError: If an auto-start configuration value or server URL is
            invalid.
    """
    if not is_mp_server_autostart_enabled(extra_config):
        return MPServerAutostartConfig.from_extra_config(
            extra_config=extra_config,
            server_host="",
            server_port=0,
        )

    try:
        parsed = urlparse(server_url if "://" in server_url else f"tcp://{server_url}")
        if parsed.netloc.count(":") > 1:
            raise ValueError(
                "IPv6 is not supported by MP auto-start; use localhost or 127.0.0.1"
            )
        server_host, server_port = parsed.hostname, parsed.port
        if (
            parsed.scheme != "tcp"
            or server_host is None
            or server_port is None
            or parsed.username is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Expected a local TCP host:port endpoint")
    except ValueError as exc:
        raise ValueError(
            f"Invalid LMCache MP server URL: {server_url!r}: {exc}"
        ) from exc

    return MPServerAutostartConfig.from_extra_config(
        extra_config=extra_config,
        server_host=server_host,
        server_port=server_port,
    )


def _get_extra_config_value(
    extra_config: object | None,
    key: str,
    default: object | None = None,
) -> object | None:
    if not isinstance(extra_config, Mapping):
        return default
    return extra_config.get(key, default)


def _parse_bool(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError(
        f"{_AUTOSTART_KEY} must be a boolean or boolean string, got {value!r}"
    )


def is_mp_server_autostart_enabled(extra_config: object | None) -> bool:
    """Return whether connector extra config enables MP server auto-start.

    Args:
        extra_config: vLLM extra-config mapping, or None.

    Returns:
        The parsed boolean; missing keys or non-mappings disable auto-start.

    Raises:
        ValueError: If the value is not a boolean or recognized boolean string.
    """
    return _parse_bool(_get_extra_config_value(extra_config, _AUTOSTART_KEY))


def _parse_port(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        port = value
    elif isinstance(value, str):
        try:
            port = int(value)
        except ValueError as exc:
            raise ValueError(
                f"LMCache MP server port must be an integer: {value!r}"
            ) from exc
    else:
        raise ValueError(f"LMCache MP server port must be an integer: {value!r}")
    if port <= 0:
        raise ValueError(f"LMCache MP server port must be positive: {value!r}")
    if port > 65535:
        raise ValueError(f"LMCache MP server port must be at most 65535: {value!r}")
    return port


def _parse_server_args(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        raise ValueError(f"{_SERVER_ARGS_KEY} must be a string, got {value!r}")
    server_args = tuple(shlex.split(value))
    for arg in server_args:
        option = arg.split("=", 1)[0]
        for disallowed_arg in _DISALLOWED_SERVER_ARGS:
            if option == disallowed_arg or disallowed_arg.startswith(option):
                raise ValueError(
                    f"{_SERVER_ARGS_KEY} cannot override {disallowed_arg}; "
                    "configure the MP endpoint with lmcache.mp.server_urls "
                    "or lmcache.mp.host/port"
                )
    return server_args


def _parse_wait_timeout(value: object | None) -> float:
    if isinstance(value, (float, int, str)):
        try:
            wait_timeout = float(value)
        except ValueError as exc:
            raise ValueError(
                f"{_WAIT_TIMEOUT_KEY} must be a number, got {value!r}"
            ) from exc
    else:
        raise ValueError(f"{_WAIT_TIMEOUT_KEY} must be a number, got {value!r}")
    if not math.isfinite(wait_timeout) or wait_timeout <= 0:
        raise ValueError(
            f"{_WAIT_TIMEOUT_KEY} must be positive and finite, got {value!r}"
        )
    return wait_timeout


def _normalize_local_host(server_host: str) -> str:
    host_to_parse = server_host.strip()
    if host_to_parse.removeprefix("tcp://").count(":") > 1:
        raise ValueError(
            "IPv6 is not supported by MP auto-start; use localhost or 127.0.0.1"
        )
    if host_to_parse in _LOCAL_HOSTS:
        return host_to_parse

    if host_to_parse.startswith("tcp://"):
        host_without_scheme = host_to_parse[len("tcp://") :]
        if host_without_scheme in _LOCAL_HOSTS:
            return host_without_scheme

    if "://" not in host_to_parse:
        host_to_parse = f"tcp://{host_to_parse}"

    try:
        host = urlparse(host_to_parse).hostname
    except ValueError as exc:
        raise ValueError(f"Invalid LMCache MP server host: {server_host!r}") from exc

    if host not in _LOCAL_HOSTS:
        raise ValueError(
            "LMCache MP auto-start only supports local hosts "
            f"{sorted(_LOCAL_HOSTS)}, got {server_host!r}"
        )
    return host


class _MessagingFuture(Protocol):
    def result(self, timeout: float | None = None) -> object:
        """Return the completed result, or raise if unavailable."""


class _RequestClient(Protocol):
    def ping(self, instance_id: int | None) -> _MessagingFuture:
        """Probe the server, optionally checking a registered instance."""

    def close(self) -> None:
        """Close the MQ client."""


class _RequestClientFactory(Protocol):
    def create(self, server_url: str, *, context: zmq.Context) -> _RequestClient:
        """Create a transport client for the endpoint."""


@dataclass(frozen=True)
class MPServerAutostartConfig:
    """Configuration for an auto-started LMCache multiprocess server."""

    enabled: bool
    host: str
    port: int
    wait_timeout: float
    server_args: tuple[str, ...]

    @classmethod
    def from_extra_config(
        cls,
        extra_config: object | None,
        server_host: str,
        server_port: int | str,
    ) -> "MPServerAutostartConfig":
        """Build auto-start configuration from vLLM connector extra config.

        Args:
            extra_config: vLLM ``kv_connector_extra_config`` mapping.
            server_host: Local IPv4 host (localhost or 127.0.0.1), optionally
                prefixed with tcp://, from the resolved connector endpoint.
            server_port: Port from the connector's resolved server endpoint.

        Returns:
            Parsed ``MPServerAutostartConfig``.

        Raises:
            ValueError: If a configured value is invalid, including an IPv6 or
                non-local host when auto-start is enabled.
        """
        enabled = is_mp_server_autostart_enabled(extra_config)
        if not enabled:
            return cls(
                enabled=False,
                host="",
                port=0,
                wait_timeout=_DEFAULT_WAIT_TIMEOUT,
                server_args=(),
            )

        server_args = _parse_server_args(
            _get_extra_config_value(extra_config, _SERVER_ARGS_KEY, "")
        )
        wait_timeout = _parse_wait_timeout(
            _get_extra_config_value(
                extra_config, _WAIT_TIMEOUT_KEY, _DEFAULT_WAIT_TIMEOUT
            )
        )
        host = _normalize_local_host(server_host)
        port = _parse_port(server_port)

        return cls(
            enabled=True,
            host=host,
            port=port,
            wait_timeout=wait_timeout,
            server_args=server_args,
        )

    def command(self) -> list[str]:
        """Return the command used to start the HTTP MP server.

        Returns:
            Command arguments suitable for ``subprocess.Popen``.
        """
        return [
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.http_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--http-host",
            self.host,
            *self.server_args,
        ]


class MPServerLauncher:
    """Starts a local LMCache MP server and cleans up failed startup attempts."""

    def __init__(self, config: MPServerAutostartConfig) -> None:
        """Initialize the launcher.

        Args:
            config: Parsed auto-start configuration.
        """
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None

    def start(self, server_url: str, zmq_context: zmq.Context) -> None:
        """Start the MP server if it is not already reachable over ZMQ.

        Args:
            server_url: ZMQ URL used by the connector to reach the MP server.
            zmq_context: ZMQ context used for health probing.

        Returns:
            None.

        Raises:
            ConnectionError: If the server process exits early or does not become
                healthy before the configured timeout.
        """
        if not self.config.enabled:
            return

        if is_mp_server_healthy(server_url, zmq_context):
            logger.info(
                "LMCache MP server is already healthy at %s; skipping auto-start",
                server_url,
            )
            return

        command = self.config.command()
        logger.info("Auto-starting LMCache MP server with command: %s", command)
        self._process = subprocess.Popen(command)
        try:
            self._wait_until_healthy(
                server_url,
                zmq_context,
                require_owned_process=True,
            )
        except Exception:
            self.shutdown()
            raise

    def wait_until_healthy(self, server_url: str, zmq_context: zmq.Context) -> None:
        """Wait for the MP server to become reachable over ZMQ.

        Args:
            server_url: ZMQ URL used by the connector to reach the MP server.
            zmq_context: ZMQ context used for health probing.

        Returns:
            None.

        Raises:
            ConnectionError: If the server does not become healthy before the
                configured timeout.
        """
        if not self.config.enabled:
            return

        if is_mp_server_healthy(server_url, zmq_context):
            logger.info("LMCache MP server is already healthy at %s", server_url)
            return

        logger.info("Waiting for LMCache MP server to become healthy at %s", server_url)
        self._wait_until_healthy(
            server_url,
            zmq_context,
            require_owned_process=False,
        )

    def shutdown(self) -> None:
        """Terminate a process started by this launcher, if it is still running.

        Used for startup failure cleanup; reusing an already healthy server
        gives this launcher no process to terminate. Adapter shutdown does not
        call this method, but vLLM may terminate the child through its own
        process-tree cleanup. Server survival after vLLM exit is not guaranteed.

        Returns:
            None.

        Raises:
            OSError: If signaling or waiting for the owned process fails.
            subprocess.TimeoutExpired: If the process does not exit within the
                timeout after being killed.
        """
        if self._process is None:
            return

        process = self._process
        self._process = None
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)

    def _wait_until_healthy(
        self,
        server_url: str,
        zmq_context: zmq.Context,
        *,
        require_owned_process: bool,
    ) -> None:
        deadline = time.monotonic() + self.config.wait_timeout
        while time.monotonic() < deadline:
            if is_mp_server_healthy(server_url, zmq_context):
                logger.info(
                    "LMCache MP server became healthy at %s",
                    server_url,
                )
                return

            if require_owned_process:
                process = self._process
                if process is None:
                    raise ConnectionError(
                        "Auto-started LMCache MP server is not active."
                    )
                return_code = process.poll()
                if return_code is not None:
                    raise ConnectionError(
                        "Auto-started LMCache MP server exited before becoming "
                        f"healthy. returncode={return_code}, "
                        f"server_url={server_url}, "
                        f"command={self.config.command()}"
                    )
            time.sleep(_POLL_INTERVAL_SECONDS)

        message = (
            "LMCache MP server did not become healthy within "
            f"{self.config.wait_timeout}s. server_url={server_url}"
        )
        if require_owned_process:
            message += f", command={self.config.command()}"
        raise ConnectionError(message)


def is_mp_server_healthy(
    server_url: str,
    zmq_context: zmq.Context,
    timeout: float = _PING_TIMEOUT_SECONDS,
) -> bool:
    """Return whether the MP server responds to a ZMQ PING request.

    Args:
        server_url: ZMQ URL of the LMCache MP server.
        zmq_context: ZMQ context used to create a temporary MQ client.
        timeout: Maximum seconds to wait for the PING response.

    Returns:
        ``True`` if the server returns a successful PING response, otherwise
        ``False``.
    """
    client: _RequestClient | None = None
    try:
        factory = _load_mp_health_dependencies()
        client = factory.create(server_url, context=zmq_context)
        future = client.ping(None)
        return bool(future.result(timeout=timeout))
    except Exception:
        logger.debug("LMCache MP server ZMQ PING failed", exc_info=True)
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Failed to close LMCache MP health client", exc_info=True)


def maybe_start_mp_server_from_url(
    *,
    extra_config: object | None,
    server_url: str,
    zmq_context: zmq.Context,
) -> MPServerLauncher | None:
    """Start the connector's local LMCache MP server when auto-start is enabled.

    Args:
        extra_config: vLLM ``kv_connector_extra_config`` mapping.
        server_url: ZMQ URL used by the connector to reach the MP server.
        zmq_context: ZMQ context used for health probing.

    Returns:
        A launcher for the enabled auto-start configuration, or ``None`` when
        auto-start is disabled. The launcher owns a process only when it had to
        start one.

    Raises:
        ValueError: If an auto-start configuration value or server URL is
            invalid.
        ConnectionError: If auto-start is enabled and the server does not
            become reachable before the configured timeout.
    """
    config = _build_autostart_config_from_url(
        extra_config=extra_config,
        server_url=server_url,
    )
    if not config.enabled:
        return None

    launcher = MPServerLauncher(config)
    launcher.start(server_url=server_url, zmq_context=zmq_context)
    return launcher


def wait_for_mp_server_from_url(
    *,
    extra_config: object | None,
    server_url: str,
    zmq_context: zmq.Context,
) -> None:
    """Wait for the connector's local LMCache MP server when auto-start is enabled.

    Args:
        extra_config: vLLM ``kv_connector_extra_config`` mapping.
        server_url: ZMQ URL used by the connector to reach the MP server.
        zmq_context: ZMQ context used for health probing.

    Returns:
        None.

    Raises:
        ValueError: If an auto-start configuration value or server URL is
            invalid.
        ConnectionError: If auto-start is enabled and the server does not
            become reachable before the configured timeout.
    """
    config = _build_autostart_config_from_url(
        extra_config=extra_config,
        server_url=server_url,
    )
    if not config.enabled:
        return

    MPServerLauncher(config).wait_until_healthy(
        server_url=server_url,
        zmq_context=zmq_context,
    )
