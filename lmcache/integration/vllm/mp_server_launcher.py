# SPDX-License-Identifier: Apache-2.0
"""Helpers for auto-starting the LMCache multiprocess server."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen
import json
import shlex
import subprocess
import sys
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_AUTOSTART_KEY = "lmcache.mp.autostart"
_HEALTH_URL_KEY = "lmcache.mp.autostart.health_url"
_SERVER_ARGS_KEY = "lmcache.mp.autostart.server_args"
_WAIT_TIMEOUT_KEY = "lmcache.mp.autostart.wait_timeout"

_DEFAULT_HTTP_PORT = 8080
_DEFAULT_WAIT_TIMEOUT = 90.0
_HEALTHCHECK_PATH = "/healthcheck"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_POLL_INTERVAL_SECONDS = 0.5
_REQUEST_TIMEOUT_SECONDS = 2.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def is_mp_server_healthy(health_url: str) -> bool:
    """Return whether the MP server health endpoint reports healthy.

    Args:
        health_url: URL of the MP server health check endpoint.

    Returns:
        ``True`` if the endpoint returns HTTP 200 with
        ``{"status": "healthy"}``, otherwise ``False``.
    """
    try:
        with urlopen(health_url, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read().decode("utf-8"))
            return data.get("status") == "healthy"
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def maybe_autostart_mp_server(
    *,
    extra_config: object | None,
    server_host: str,
    server_port: int | str,
    rank: int,
) -> "MPServerLauncher | None":
    """Start the LMCache MP server for worker rank 0 when configured.

    Args:
        extra_config: vLLM ``kv_connector_extra_config`` mapping.
        server_host: LMCache MP server host from ``lmcache.mp.host``.
        server_port: LMCache MP server port from ``lmcache.mp.port``.
        rank: vLLM global rank.

    Returns:
        The launcher that owns the server process, or ``None`` when no process
        was started by this connector.
    """
    config = MPServerAutostartConfig.from_extra_config(
        extra_config=extra_config,
        server_host=server_host,
        server_port=server_port,
    )
    if not config.enabled:
        return None
    if rank != 0:
        logger.info("LMCache MP auto-start is enabled only for worker rank 0")
        return None

    launcher = MPServerLauncher(config)
    launcher.start()
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


def _get_http_port(server_args: tuple[str, ...]) -> int:
    http_port = _DEFAULT_HTTP_PORT
    index = 0
    while index < len(server_args):
        arg = server_args[index]
        if arg == "--http-port" and index + 1 < len(server_args):
            http_port = _parse_port(server_args[index + 1])
            index += 2
            continue
        if arg.startswith("--http-port="):
            http_port = _parse_port(arg.split("=", maxsplit=1)[1])
        index += 1
    return http_port


@dataclass(frozen=True)
class MPServerAutostartConfig:
    """Configuration for an auto-started LMCache multiprocess server."""

    enabled: bool
    host: str
    port: int
    wait_timeout: float
    health_url: str
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
                health_url="",
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
        health_url = _get_extra_config_value(extra_config, _HEALTH_URL_KEY)

        host = _normalize_local_host(server_host)
        port = _parse_port(server_port)
        if health_url is None or str(health_url).strip() == "":
            http_port = _get_http_port(server_args)
            health_url = f"http://127.0.0.1:{http_port}{_HEALTHCHECK_PATH}"

        return cls(
            enabled=True,
            host=host,
            port=port,
            wait_timeout=wait_timeout,
            health_url=str(health_url),
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

    def start(self) -> None:
        """Start the MP server if the health endpoint is not already healthy.

        Raises:
            ConnectionError: If the server process exits early or does not become
                healthy before the configured timeout.
        """
        if not self.config.enabled:
            return

        if is_mp_server_healthy(self.config.health_url):
            logger.info(
                "LMCache MP server is already healthy at %s; skipping auto-start",
                self.config.health_url,
            )
            return

        command = self.config.command()
        logger.info("Auto-starting LMCache MP server with command: %s", command)
        self._process = subprocess.Popen(command)
        try:
            self._wait_until_healthy()
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

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + self.config.wait_timeout
        while time.monotonic() < deadline:
            if is_mp_server_healthy(self.config.health_url):
                logger.info(
                    "LMCache MP server became healthy at %s",
                    self.config.health_url,
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
                    f"health_url={self.config.health_url}, "
                    f"command={self.config.command()}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)

        raise ConnectionError(
            "Auto-started LMCache MP server did not become healthy within "
            f"{self.config.wait_timeout}s. health_url={self.config.health_url}, "
            f"command={self.config.command()}"
        )
