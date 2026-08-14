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
        chunk_size: Tokens per KV chunk. The single fleet chunk size: it is the
            CacheBlend match unit *and* resolves a pin request's ``token_ids`` to
            object keys. Must equal the MP servers' ``--chunk-size`` or blend
            matches and resolved pin keys will not line up with what was stored.
        hash_algorithm: Token hash algorithm for pin key resolution. Must equal
            the MP servers' ``--hash-algorithm`` (default ``blake3``, which is
            self-contained; other algorithms require vLLM importable in the
            coordinator process).
        enable_blend_lookup: When ``True``, index stored chunk content so
            ``POST /directory/blend-lookup`` can serve fleet CacheBlend
            reuse. Off by default; a fleet without CacheBlend then hashes
            no content.
        blend_probe_stride: Positions between match probes; ``1`` gives full
            recall. Ignored unless ``enable_blend_lookup`` is set.
        timeout_keep_alive: Seconds the HTTP server keeps idle connections
            open before closing them. Must be greater than the heartbeat
            interval of MP servers to avoid race-condition disconnects.
    """

    host: str = "0.0.0.0"
    port: int = 9300
    instance_timeout: float = 30.0
    health_check_interval: float = 10.0
    eviction_check_interval: float = 5.0
    eviction_ratio: float = 0.2
    trigger_watermark: float = 1.0
    chunk_size: int = 256
    hash_algorithm: str = "blake3"
    enable_blend_lookup: bool = False
    blend_probe_stride: int = 1
    timeout_keep_alive: int = 10

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
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not self.hash_algorithm:
            raise ValueError("hash_algorithm must be a non-empty string")
        if self.blend_probe_stride < 1:
            raise ValueError("blend_probe_stride must be positive")
        if self.timeout_keep_alive <= 0:
            raise ValueError("timeout_keep_alive must be positive")

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
            chunk_size=int(_num("CHUNK_SIZE", cls.chunk_size, int)),
            hash_algorithm=_str("HASH_ALGORITHM", cls.hash_algorithm),
            enable_blend_lookup=_bool("ENABLE_BLEND_LOOKUP", cls.enable_blend_lookup),
            blend_probe_stride=int(
                _num("BLEND_PROBE_STRIDE", cls.blend_probe_stride, int)
            ),
            timeout_keep_alive=int(
                _num("TIMEOUT_KEEP_ALIVE", cls.timeout_keep_alive, int)
            ),
        )
