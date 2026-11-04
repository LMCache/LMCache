# SPDX-License-Identifier: Apache-2.0
"""Wire schema for the single-process (InProcess) usage-telemetry messages.

The single source of truth for the whole wire schema lives in
:mod:`lmcache.usage_telemetry.messages`; this module only holds the
messages that the InProcess engine can send, so that MP-side code never
pulls InProcess types through the shared schema module.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.usage_telemetry.messages import UsageMessage

if TYPE_CHECKING:
    # First Party
    from lmcache.inprocess.v1.config import LMCacheEngineConfig
    from lmcache.inprocess.v1.metadata import LMCacheMetadata


@dataclass
class EngineMessage(UsageMessage):
    """Configuration snapshot of a single-process LMCacheEngine.

    Sent once at engine startup (single-process integrations only).
    """

    chunksize: int
    local_device: str
    max_local_cache_size: int
    remote_url: str | None
    remote_serde: str | None
    pipelined_backend: bool
    save_decode_cache: bool
    enable_blending: bool
    blend_recompute_ratio: float
    blend_min_tokens: int
    model_name: str
    world_size: int
    worker_id: int
    kv_dtype: torch.dtype
    kv_shape: tuple[int, int, int, int, int]

    @classmethod
    def from_config(
        cls, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> EngineMessage:
        """Build the message from the engine configuration and metadata.

        Args:
            config: The engine configuration to snapshot.
            metadata: The engine metadata (model, world size, kv layout).

        Returns:
            The populated message.
        """
        return cls(
            chunksize=config.chunk_size,
            local_device="cpu" if config.local_cpu else torch_device_type,
            max_local_cache_size=int(config.max_local_cpu_size),
            remote_url=config.remote_url,
            remote_serde=config.remote_serde,
            pipelined_backend=False,
            save_decode_cache=config.save_decode_cache,
            enable_blending=config.enable_blending,
            blend_recompute_ratio=0.15,
            blend_min_tokens=config.blend_min_tokens,
            model_name=metadata.model_name,
            world_size=metadata.world_size,
            worker_id=metadata.worker_id,
            kv_dtype=metadata.kv_dtype,
            kv_shape=metadata.kv_shape,
        )


@dataclass
class MetadataMessage(UsageMessage):
    """Process start time and uptime at send time.

    Sent once at engine startup (single-process integrations only).
    """

    start_time: str
    duration: float


@dataclass
class CacheLifespanMessage(UsageMessage):
    """Cache-entry lifespan histogram flushed with each continuous report.

    Single-process mode only; measures the store-to-reuse gap per reused
    chunk.
    """

    ENDPOINT: ClassVar[str] = "cache-lifespan"

    cache_lifespan_histogram: dict[float, int]
    """Bucket lower bound (minutes) to sample count. Numeric keys become
    strings on the wire (JSON object keys)."""
    sequence_number: int
    uptime_seconds: float
    """Seconds since the reporting process started."""
