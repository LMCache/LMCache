# SPDX-License-Identifier: Apache-2.0
"""Helpers for auto-starting the LMCache multiprocess server."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse
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
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_PING_TIMEOUT_SECONDS = 1.0
_POLL_INTERVAL_SECONDS = 0.5
_SHUTDOWN_TIMEOUT_SECONDS = 10.0


class _MessagingFuture(Protocol):
    def result(self, timeout: float | None = None) -> object:
        """Return the completed result, or raise if unavailable."""


class _MessageQueueClient(Protocol):
    def submit_request(
        self,
        request_type: object,
        request_payloads: list[object],
        response_cls: object | None = None,
    ) -> _MessagingFuture:
        """Submit an MQ request and return its future."""

    def close(self) -> None:
        """Close the MQ client."""


def _create_message_queue_client(
    server_url: str,
    zmq_context: zmq.Context,
) -> _MessageQueueClient:
    # Keep the multiprocess stack deferred so config-only tests can import
    # this launcher without requiring the full torch/vLLM runtime.
    # First Party
    from lmcache.v1.multiprocess.mq import MessageQueueClient

    return MessageQueueClient(server_url, zmq_context)


def _submit_ping(client: _MessageQueueClient) -> _MessagingFuture:
    # Keep the multiprocess stack deferred so config-only tests can import
    # this launcher without requiring the full torch/vLLM runtime.
    # First Party
    from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

    return client.submit_request(
        RequestType.PING,
        [],
        get_response_class(RequestType.PING),
    )


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
    client: _MessageQueueClient | None = None
    try:
        client = _create_message_queue_client(server_url, zmq_context)
        future = _submit_ping(client)
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


def maybe_autostart_mp_server(
    *,
    extra_config: object | None,
    server_host: str,
    server_port: int | str,
    server_url: str,
    zmq_context: zmq.Context,
) -> "MPServerLauncher | None":
    """Start the LMCache MP server when configured.

    Args:
        extra_config: vLLM ``kv_connector_extra_config`` mapping.
        server_host: LMCache MP server host from ``lmcache.mp.host``.
        server_port: LMCache MP server port from ``lmcache.mp.port``.
        server_url: ZMQ URL used by the connector to reach the MP server.
        zmq_context: ZMQ context used for health probing.

    Returns:
        The launcher that owns the server process, or ``None`` when no process
        was started by this connector.

    Raises:
        ValueError: If an auto-start configuration value is invalid.
        ConnectionError: If auto-start is enabled and the server does not
            become reachable before the configured timeout.
    """
    config = MPServerAutostartConfig.from_extra_config(
        extra_config=extra_config,
        server_host=server_host,
        server_port=server_port,
    )
    if not config.enabled:
        return None

    launcher = MPServerLauncher(config)
    launcher.start(server_url=server_url, zmq_context=zmq_context)
    return launcher


def shutdown_mp_server_launcher(launcher: "MPServerLauncher | None") -> None:
    """Shutdown an owned MP server launcher when it exists.

    Args:
        launcher: Launcher instance to shut down, or ``None``.
    """
    if launcher is not None:
        launcher.shutdown()


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
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError(
        f"{_AUTOSTART_KEY} must be a boolean or boolean string, got {value!r}"
    )


def _parse_port(value: object) -> int:
    if isinstance(value, int):
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
    return port


def _parse_server_args(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        raise ValueError(f"{_SERVER_ARGS_KEY} must be a string, got {value!r}")
    return tuple(shlex.split(value))


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
    if wait_timeout <= 0:
        raise ValueError(f"{_WAIT_TIMEOUT_KEY} must be positive, got {value!r}")
    return wait_timeout


def _normalize_local_host(server_host: str) -> str:
    host_to_parse = server_host.strip()
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
            server_host: LMCache MP server host from ``lmcache.mp.host``.
            server_port: LMCache MP server port from ``lmcache.mp.port``.

        Returns:
            Parsed ``MPServerAutostartConfig``.

        Raises:
            ValueError: If a configured value is invalid.
        """
        enabled = _parse_bool(_get_extra_config_value(extra_config, _AUTOSTART_KEY))
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
        """Return the command used to start the HTTP MP server."""
        return [
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.http_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.server_args,
        ]


class MPServerLauncher:
    """Owns the lifecycle of an auto-started LMCache MP server process."""

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
            self._wait_until_healthy(server_url, zmq_context)
        except Exception:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        """Terminate the auto-started MP server process, if one is owned."""
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

    def _wait_until_healthy(self, server_url: str, zmq_context: zmq.Context) -> None:
        deadline = time.monotonic() + self.config.wait_timeout
        while time.monotonic() < deadline:
            if is_mp_server_healthy(server_url, zmq_context):
                logger.info(
                    "LMCache MP server became healthy at %s",
                    server_url,
                )
                return

            process = self._process
            if process is None:
                raise ConnectionError("Auto-started LMCache MP server is not active.")
            return_code = process.poll()
            if return_code is not None:
                raise ConnectionError(
                    "Auto-started LMCache MP server exited before becoming "
                    f"healthy. returncode={return_code}, "
                    f"server_url={server_url}, "
                    f"command={self.config.command()}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)

        raise ConnectionError(
            "Auto-started LMCache MP server did not become healthy within "
            f"{self.config.wait_timeout}s. server_url={server_url}, "
            f"command={self.config.command()}"
        )
