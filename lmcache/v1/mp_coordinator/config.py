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
        blend_chunk_size: Tokens per chunk for the global CacheBlend directory
            (the match unit). Must equal the LMCache chunk size the blend servers
            use, so the coordinator chunks published/queried tokens the same way.
        blend_probe_stride: Positions between match probes. With partial-fill
            reuse any offset is usable, so ``1`` (probe every offset) gives full
            recall; raise only to trade recall for coordinator CPU.
    """

    host: str = "0.0.0.0"
    port: int = 9300
    instance_timeout: float = 30.0
    health_check_interval: float = 10.0
    eviction_check_interval: float = 5.0
    eviction_ratio: float = 0.2
    trigger_watermark: float = 1.0
    blend_chunk_size: int = 256
    blend_probe_stride: int = 1

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
        if self.blend_chunk_size < 1:
            raise ValueError("blend_chunk_size must be positive")
        if self.blend_probe_stride < 1:
            raise ValueError("blend_probe_stride must be positive")

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
            eviction_check_interval=_num(
                "EVICTION_CHECK_INTERVAL", cls.eviction_check_interval, float
            ),
            eviction_ratio=_num("EVICTION_RATIO", cls.eviction_ratio, float),
            trigger_watermark=_num("TRIGGER_WATERMARK", cls.trigger_watermark, float),
            blend_chunk_size=int(_num("BLEND_CHUNK_SIZE", cls.blend_chunk_size, int)),
            blend_probe_stride=int(
                _num("BLEND_PROBE_STRIDE", cls.blend_probe_stride, int)
            ),
        )
