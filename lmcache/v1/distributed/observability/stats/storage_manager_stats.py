# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass


@dataclass
class StorageManagerStats:
    # StorageManager-level metrics
    # These measure the full round-trip at the StorageManager API boundary
    # Interval counters (incremental, reset each logging period)
    interval_sm_read_requests: int = 0
    """Number of submit_prefetch_task calls in this interval"""

    interval_sm_read_succeed_keys: int = 0
    """Keys that were cache hits (succeeded) during SM read prefetch"""

    interval_sm_read_failed_keys: int = 0
    """Keys that were cache misses (failed) during SM read prefetch"""

    interval_sm_write_requests: int = 0
    """Number of reserve_write calls in this interval"""

    interval_sm_write_succeed_keys: int = 0
    """Keys successfully allocated for write (memory reserved)"""

    interval_sm_write_failed_keys: int = 0
    """Keys that failed allocation (OOM or write conflict)"""
