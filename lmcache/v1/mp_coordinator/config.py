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
        eviction_check_interval: Seconds between eviction sweeps. A value of
            ``0`` disables the eviction loop.
        eviction_ratio: Fraction of tracked keys (by count) to evict per
            cycle (0.0 to 1.0).
        trigger_watermark: Eviction fires when usage reaches this fraction
            of the quota (0.0 to 1.0).
        enable_startup_resync: When ``True`` the coordinator runs a
            one-shot L2 resync on startup that paginates an MP server's
            ``GET /l2/keys`` and backfills usage + eviction trackers.
            Set to ``False`` to skip — useful in tests and in
            deployments that start the coordinator before any MP
            servers exist and don't care about pre-existing L2 state.
        resync_poll_interval: Seconds between registry checks while
            waiting for the first MP server to register before startup
            resync.
        resync_max_wait: Maximum seconds the startup resync waits for
            an MP server before giving up. Beyond this, the coordinator
            keeps running with empty trackers until normal usage events
            fill them in.
        resync_page_size: ``page_size`` to request from ``GET /l2/keys``
            during resync. Larger values reduce RTT count; the server
            clamps to its own ceiling.
    """

    host: str = "0.0.0.0"
    port: int = 9300
    instance_timeout: float = 30.0
    health_check_interval: float = 10.0
    eviction_check_interval: float = 5.0
    eviction_ratio: float = 0.2
    trigger_watermark: float = 1.0
    enable_startup_resync: bool = True
    resync_poll_interval: float = 1.0
    resync_max_wait: float = 60.0
    resync_page_size: int = 1000

    def __post_init__(self) -> None:
        """Validate timing parameters.

        Raises:
            ValueError: If a timing parameter is non-positive/negative.
        """
        if self.instance_timeout <= 0:
            raise ValueError("instance_timeout must be positive")
        if self.health_check_interval < 0:
            raise ValueError("health_check_interval must be non-negative")
        if self.eviction_check_interval < 0:
            raise ValueError("eviction_check_interval must be non-negative")
        if not 0.0 <= self.eviction_ratio <= 1.0:
            raise ValueError("eviction_ratio must be between 0.0 and 1.0")
        if not 0.0 < self.trigger_watermark <= 1.0:
            raise ValueError(
                "trigger_watermark must be between 0.0 (exclusive) and 1.0"
            )
        if self.resync_poll_interval <= 0:
            raise ValueError("resync_poll_interval must be positive")
        if self.resync_max_wait < 0:
            raise ValueError("resync_max_wait must be non-negative")
        if self.resync_page_size <= 0:
            raise ValueError("resync_page_size must be positive")

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

        def _bool(name: str, default: bool) -> bool:
            raw = os.getenv(f"{_ENV_PREFIX}{name}")
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            host=_str("HOST", cls.host),
            port=int(_num("PORT", cls.port, int)),
            instance_timeout=_num("INSTANCE_TIMEOUT", cls.instance_timeout, float),
            health_check_interval=_num(
                "HEALTH_CHECK_INTERVAL", cls.health_check_interval, float
            ),
            eviction_check_interval=_num(
                "EVICTION_CHECK_INTERVAL", cls.eviction_check_interval, float
            ),
            eviction_ratio=_num("EVICTION_RATIO", cls.eviction_ratio, float),
            trigger_watermark=_num("TRIGGER_WATERMARK", cls.trigger_watermark, float),
            enable_startup_resync=_bool(
                "ENABLE_STARTUP_RESYNC", cls.enable_startup_resync
            ),
            resync_poll_interval=_num(
                "RESYNC_POLL_INTERVAL", cls.resync_poll_interval, float
            ),
            resync_max_wait=_num("RESYNC_MAX_WAIT", cls.resync_max_wait, float),
            resync_page_size=int(_num("RESYNC_PAGE_SIZE", cls.resync_page_size, int)),
        )
