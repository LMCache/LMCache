# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral memory usage snapshots for distributed storage tiers."""

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Literal

# First Party
from lmcache.v1.distributed.tiers import Tier

StorageTier = Literal[Tier.L1, Tier.L2]


class StorageHealth(str, Enum):
    """Health state reported while collecting a storage usage snapshot."""

    OK = "ok"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StorageTierUsageSnapshot:
    """Point-in-time usage and health data for one storage tier."""

    tier: StorageTier
    adapter_id: int | None
    backend_type: str | None
    used_bytes: int | None
    capacity_bytes: int | None
    trigger_watermark: float | None
    eviction_policy: str | None
    health: StorageHealth
    collection_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StorageUsageSnapshot:
    """Point-in-time L1 and per-adapter L2 memory usage data."""

    l1: StorageTierUsageSnapshot
    l2: tuple[StorageTierUsageSnapshot, ...]
