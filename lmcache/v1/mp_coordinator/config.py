# SPDX-License-Identifier: Apache-2.0
"""Configuration for the mp coordinator process.

A small, explicit, frozen dataclass. Values come from ``lmcache coordinator``
CLI flags (see :mod:`lmcache.cli.commands.coordinator`); unset flags leave the
defaults below.
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field


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
        checkpoint_path: File the coordinator's derived state is
            checkpointed to. Empty disables checkpointing, and the
            coordinator starts cold after every restart.
        checkpoint_interval: Seconds between checkpoint writes; ``0``
            writes only on a clean stop. Ignored without a path.
        metadata_path: File the operator-set state (L2 pins and
            per-``cache_salt`` quotas) is stored in. Empty disables it,
            and that state is lost on restart.
        timeout_keep_alive: Seconds the HTTP server keeps idle connections
            open before closing them. Must be greater than the heartbeat
            interval of MP servers to avoid race-condition disconnects.
        extra_config: Settings the core config does not name, keyed by
            whatever reads them. Discovery means a new view or controller
            is one file; without this, giving it a setting would still
            mean editing this class, the CLI, and the docs.
        metrics_enabled: Whether to initialize OpenTelemetry metrics.
        otlp_endpoint: OTLP gRPC endpoint for metrics push mode. When unset,
            metrics use Prometheus pull mode on the coordinator HTTP port.
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
    checkpoint_path: str = ""
    checkpoint_interval: float = 60.0
    metadata_path: str = ""
    extra_config: Mapping[str, object] = field(default_factory=dict)
    timeout_keep_alive: int = 10
    metrics_enabled: bool = True
    otlp_endpoint: str | None = None

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
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.timeout_keep_alive <= 0:
            raise ValueError("timeout_keep_alive must be positive")
