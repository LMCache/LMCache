# SPDX-License-Identifier: Apache-2.0
"""Configuration for the mp coordinator process.

A small, explicit, frozen dataclass with environment-variable loading. The
engine-wide ``create_config_class`` helper is intentionally not used here: it
injects engine semantics (e.g. an ``lmcache_instance_id``) that are meaningless
for a standalone coordinator process.
"""

# Standard
from dataclasses import dataclass
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_ENV_PREFIX = "LMCACHE_MP_COORDINATOR_"


@dataclass(frozen=True)
class MPCoordinatorConfig:
    """Socket addresses and timing parameters for the coordinator.

    Attributes:
        pull_url: Bind address (``host:port``) for the fire-and-forget (PULL)
            channel receiving deregistrations from mp servers.
        reply_url: Bind address for the request/reply (ROUTER) channel handling
            register and heartbeat.
        heartbeat_interval: Seconds between heartbeats expected from mp servers.
        instance_timeout: Seconds without a heartbeat after which an instance is
            considered dead and evicted.
        health_check_interval: Seconds between health-check sweeps. A value of
            ``0`` disables the health-check thread.
    """

    pull_url: str = "0.0.0.0:9300"
    reply_url: str = "0.0.0.0:9301"
    heartbeat_interval: float = 5.0
    instance_timeout: float = 30.0
    health_check_interval: float = 10.0

    def __post_init__(self) -> None:
        """Validate timing parameters.

        Raises:
            ValueError: If any timing parameter is negative, or if the
                instance timeout is not larger than the heartbeat interval.
        """
        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if self.instance_timeout <= 0:
            raise ValueError("instance_timeout must be positive")
        if self.health_check_interval < 0:
            raise ValueError("health_check_interval must be non-negative")
        if self.instance_timeout <= self.heartbeat_interval:
            raise ValueError(
                "instance_timeout must exceed heartbeat_interval so a single "
                "missed heartbeat does not evict a live instance"
            )

    @classmethod
    def from_env(cls) -> "MPCoordinatorConfig":
        """Build a config from ``LMCACHE_MP_COORDINATOR_*`` environment variables.

        Unset variables fall back to the dataclass defaults.

        Returns:
            A validated configuration instance.
        """

        def _str(name: str, default: str) -> str:
            return os.getenv(f"{_ENV_PREFIX}{name}", default)

        def _float(name: str, default: float) -> float:
            raw = os.getenv(f"{_ENV_PREFIX}{name}")
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.warning(
                    "Invalid %s%s=%r; using default %s",
                    _ENV_PREFIX,
                    name,
                    raw,
                    default,
                )
                return default

        return cls(
            pull_url=_str("PULL_URL", cls.pull_url),
            reply_url=_str("REPLY_URL", cls.reply_url),
            heartbeat_interval=_float("HEARTBEAT_INTERVAL", cls.heartbeat_interval),
            instance_timeout=_float("INSTANCE_TIMEOUT", cls.instance_timeout),
            health_check_interval=_float(
                "HEALTH_CHECK_INTERVAL", cls.health_check_interval
            ),
        )
