# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import List


@dataclass
class StorageManagerStats:
    # StorageManager-level metrics
    # These measure the full round-trip at the StorageManager API boundary
    # Interval counters (incremental, reset each logging period)
    interval_sm_read_requests: int = 0
    """Number of submit_prefetch_task calls in this interval"""

    interval_sm_read_hit_keys: int = 0
    """Keys that were cache hits (succeeded) during SM read prefetch"""

    interval_sm_read_miss_keys: int = 0
    """Keys that were cache misses (failed) during SM read prefetch"""

    interval_sm_write_requests: int = 0
    """Number of reserve_write calls in this interval"""

    interval_sm_write_success_keys: int = 0
    """Keys successfully allocated for write (memory reserved)"""

    interval_sm_write_failed_keys: int = 0
    """Keys that failed allocation (OOM or write conflict)"""

    # L1Manager-level metrics
    # These measure the L1 in-memory cache tier specifically
    # Interval counters
    interval_l1_read_keys: int = 0
    """Keys reserved for read on L1 in this interval"""

    interval_l1_write_keys: int = 0
    """Keys reserved for write on L1 in this interval"""

    interval_l1_evicted_keys: int = 0
    """Keys deleted from L1 by the eviction controller in this interval"""

    # Latency distributions (per-batch, in seconds)
    # Safe to use FIFO deque because L1Manager serializes all callbacks under its lock
    l1_read_latency: List[float] = field(default_factory=list)
    """Per-batch L1 read latency: on_l1_keys_reserved_read → on_l1_keys_read_finished"""

    l1_write_latency: List[float] = field(default_factory=list)
    """
    Per-batch L1 write latency: on_l1_keys_reserved_write → on_l1_keys_write_finished
    """
