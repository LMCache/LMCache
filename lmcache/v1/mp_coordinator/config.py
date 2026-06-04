# SPDX-License-Identifier: Apache-2.0
"""Configuration for the mp coordinator process.

A small, explicit, frozen dataclass with environment-variable loading
(``LMCACHE_MP_COORDINATOR_*``).
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
    """HTTP bind address and timing parameters for the coordinator.

    Attributes:
        host: Host the coordinator's HTTP server binds to.
        port: Port the coordinator's HTTP server binds to.
        instance_timeout: Seconds without a heartbeat after which an instance is
            considered dead and evicted. Set this comfortably above the mp
            servers' own heartbeat cadence (which they choose).
        health_check_interval: Seconds between health-check sweeps. A value of
            ``0`` disables the health-check loop.
    """

    host: str = "0.0.0.0"
    port: int = 9300
    instance_timeout: float = 30.0
    health_check_interval: float = 10.0

    def __post_init__(self) -> None:
        """Validate timing parameters.

        Raises:
            ValueError: If a timing parameter is non-positive/negative.
        """
        if self.instance_timeout <= 0:
            raise ValueError("instance_timeout must be positive")
        if self.health_check_interval < 0:
            raise ValueError("health_check_interval must be non-negative")

    @classmethod
    def from_env(cls) -> "MPCoordinatorConfig":
        """Build a config from ``LMCACHE_MP_COORDINATOR_*`` environment variables.

        Unset variables fall back to the dataclass defaults.

        Returns:
            A validated configuration instance.
        """

        def _str(name: str, default: str) -> str:
            return os.getenv(f"{_ENV_PREFIX}{name}", default)

        def _num(name: str, default: float, cast) -> float:
            raw = os.getenv(f"{_ENV_PREFIX}{name}")
            if raw is None:
                return default
            try:
                return cast(raw)
            except ValueError:
                logger.warning(
                    "Invalid %s%s=%r; using default %s", _ENV_PREFIX, name, raw, default
                )
                return default

        return cls(
            host=_str("HOST", cls.host),
            port=int(_num("PORT", cls.port, int)),
            instance_timeout=_num("INSTANCE_TIMEOUT", cls.instance_timeout, float),
            health_check_interval=_num(
                "HEALTH_CHECK_INTERVAL", cls.health_check_interval, float
            ),
        )
