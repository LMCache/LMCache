# SPDX-License-Identifier: Apache-2.0
"""Helpers for auto-starting the LMCache multiprocess server."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast
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
_HOST_KEY = "lmcache.mp.host"
_PORT_KEY = "lmcache.mp.port"
_SERVER_ARGS_KEY = "lmcache.mp.autostart.server_args"
_WAIT_TIMEOUT_KEY = "lmcache.mp.autostart.wait_timeout"

_DEFAULT_WAIT_TIMEOUT = 90.0
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_DISALLOWED_SERVER_ARGS = {"--host", "--port", "--http-host"}
_PING_TIMEOUT_SECONDS = 1.0
_POLL_INTERVAL_SECONDS = 0.5
_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def _load_mp_health_dependencies() -> tuple[
    "_MessageQueueClientFactory",
    "_RequestTypeNamespace",
    Callable[[object], object | None],
]:
    """Load MQ health probe dependencies when a probe is executed.

    The MQ modules may require torch through transitive imports, while most of
    this module only parses config or builds process commands. Keeping these
    imports lazy lets config-only callers import this launcher without torch.

    Returns:
        The MQ client factory, request type namespace, and response-class
        resolver used to send a PING request.
    """
    # First Party
    from lmcache.v1.multiprocess.mq import MessageQueueClient
    from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

    return (
        cast(_MessageQueueClientFactory, MessageQueueClient),
        cast(_RequestTypeNamespace, RequestType),
        cast(Callable[[object], object | None], get_response_class),
    )


def _create_message_queue_client(
    factory: _MessageQueueClientFactory,
    server_url: str,
    zmq_context: zmq.Context,
) -> _MessageQueueClient:
    """Create an MQ client for MP server health probing.

    Args:
        factory: Lazy-loaded ``MessageQueueClient`` factory.
        server_url: ZMQ URL of the LMCache MP server.
        zmq_context: ZMQ context used to create the client.

    Returns:
        A message queue client connected to ``server_url``.
    """
    return factory(server_url, zmq_context)


def _submit_ping(
    client: _MessageQueueClient,
    request_type: _RequestTypeNamespace,
    response_class_getter: Callable[[object], object | None],
) -> _MessagingFuture:
    """Submit an MP server PING health probe through the MQ client.

    Args:
        client: MQ client used to submit the request.
        request_type: Lazy-loaded request type namespace.
        response_class_getter: Lazy-loaded response-class resolver.

    Returns:
        A messaging future for the PING response.
    """
    ping_request_type = request_type.PING
    return client.submit_request(
        ping_request_type,
        [],
        response_class_getter(ping_request_type),
    )


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
    if not _parse_bool(_get_extra_config_value(extra_config, _AUTOSTART_KEY)):
        return MPServerAutostartConfig.from_extra_config(
            extra_config=extra_config,
            server_host="",
            server_port=0,
        )

    server_host = _get_extra_config_value(extra_config, _HOST_KEY)
    server_port_value = _get_extra_config_value(extra_config, _PORT_KEY)
    if server_port_value is not None and not isinstance(server_port_value, (int, str)):
        raise ValueError(
            f"LMCache MP server port must be an integer: {server_port_value!r}"
        )
    server_port = server_port_value
    if server_host is None or server_port is None:
        parsed = urlparse(server_url if "://" in server_url else f"tcp://{server_url}")
        if parsed.hostname is None or parsed.port is None:
            raise ValueError(f"Invalid LMCache MP server URL: {server_url!r}")
        server_host = parsed.hostname if server_host is None else server_host
        server_port = parsed.port if server_port is None else server_port

    return MPServerAutostartConfig.from_extra_config(
        extra_config=extra_config,
        server_host=str(server_host),
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
    server_args = tuple(shlex.split(value))
    for arg in server_args:
        option = arg.split("=", 1)[0]
        for disallowed_arg in _DISALLOWED_SERVER_ARGS:
            if option == disallowed_arg or disallowed_arg.startswith(option):
                raise ValueError(
                    f"{_SERVER_ARGS_KEY} cannot override {disallowed_arg}; "
                    "configure the MP endpoint with lmcache.mp.host/port"
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


class _MessageQueueClientFactory(Protocol):
    def __call__(self, server_url: str, context: zmq.Context) -> _MessageQueueClient:
        """Create a message queue client."""


class _RequestTypeNamespace(Protocol):
    PING: object


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
        """Terminate the auto-started MP server process.

        This is used for startup failure cleanup. Normal vLLM shutdown does not
        call it because other vLLM instances may share the same MP server.

        Returns:
            None.
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
    client: _MessageQueueClient | None = None
    try:
        (
            message_queue_client_factory,
            request_type,
            response_class_getter,
        ) = _load_mp_health_dependencies()
        client = _create_message_queue_client(
            message_queue_client_factory,
            server_url,
            zmq_context,
        )
        future = _submit_ping(client, request_type, response_class_getter)
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
