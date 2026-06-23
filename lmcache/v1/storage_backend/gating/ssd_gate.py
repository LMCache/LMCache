# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Optional, TypedDict
import threading

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.gating.base_gate import BaseStorageGate
from lmcache.v1.storage_backend.gating.write_veto import WriteVetoReason


class SsdStorageGateStats(TypedDict):
    """Aggregated storage-gate counters for diagnostics and tests."""

    lookup_tracked_count: int
    read_tracked_count: int
    write_tracked_count: int
    delete_tracked_count: int
    total_lookup_count: int
    total_read_count: int
    total_write_count: int
    total_delete_count: int


class SsdStorageGate(BaseStorageGate):
    """
    Wear-oriented gate for local disk: optional minimum chunk size and read
    count before admitting writes.

    Internal counters (lookup / read / write / delete) are updated only via
    ``record_*``; ``on_*`` and ``explain_write_veto`` only read them. A
    per-instance lock serializes counter updates and reads so the gate is safe
    to use from multiple threads even if a backend forgets to hold its own lock.

    ``_lookup_counts`` may be cleared when it grows past
    ``max_tracked_chunk_hashes`` to limit memory (read/write/delete maps are
    still bounded by chunk lifecycle + deletes).

    ``record_write(..., new_admission=True)`` increments the per-chunk write
    counter only on first admission; metadata refresh uses
    ``new_admission=False`` so write-count metrics stay meaningful.
    """

    _default_max_tracked_chunk_hashes = 12_500_000

    def __init__(
        self,
        *,
        min_size_bytes: int = 0,
        min_read_count_before_write: int = 0,
        max_tracked_chunk_hashes: Optional[int] = None,
    ) -> None:
        self._min_size_bytes = min_size_bytes
        self._min_read_count_before_write = min_read_count_before_write
        self._max_tracked = (
            max_tracked_chunk_hashes
            if max_tracked_chunk_hashes is not None
            else self._default_max_tracked_chunk_hashes
        )
        self._lock = threading.Lock()
        self._lookup_counts: Dict[Any, int] = {}
        self._read_counts: Dict[Any, int] = {}
        self._write_counts: Dict[Any, int] = {}
        self._delete_counts: Dict[Any, int] = {}

    @staticmethod
    def _chunk_hash(key: CacheEngineKey) -> Any:
        return key.chunk_hash

    def on_lookup(self, key: CacheEngineKey) -> bool:  # noqa: ARG002
        return True

    def on_read(self, key: CacheEngineKey) -> bool:  # noqa: ARG002
        return True

    def on_delete(self, key: CacheEngineKey) -> bool:  # noqa: ARG002
        return True

    def explain_write_veto(
        self,
        key: CacheEngineKey,
        size_bytes: int,
    ) -> Optional[WriteVetoReason]:
        with self._lock:
            if self._min_size_bytes > 0 and size_bytes < self._min_size_bytes:
                return WriteVetoReason.LENGTH
            h = self._chunk_hash(key)
            reads = self._read_counts.get(h, 0)
            if (
                self._min_read_count_before_write > 0
                and reads < self._min_read_count_before_write
            ):
                return WriteVetoReason.FREQUENCY
            return None

    def record_lookup(self, key: CacheEngineKey) -> None:
        with self._lock:
            if len(self._lookup_counts) >= self._max_tracked:
                self._lookup_counts.clear()
            h = self._chunk_hash(key)
            self._lookup_counts[h] = self._lookup_counts.get(h, 0) + 1

    def record_read(self, key: CacheEngineKey) -> None:
        with self._lock:
            h = self._chunk_hash(key)
            self._read_counts[h] = self._read_counts.get(h, 0) + 1

    def record_write(self, key: CacheEngineKey, *, new_admission: bool = True) -> None:
        with self._lock:
            h = self._chunk_hash(key)
            if new_admission:
                self._write_counts[h] = self._write_counts.get(h, 0) + 1
            self._read_counts[h] = 0

    def record_delete(self, key: CacheEngineKey) -> None:
        with self._lock:
            h = self._chunk_hash(key)
            self._lookup_counts.pop(h, None)
            self._read_counts.pop(h, None)
            self._write_counts.pop(h, None)
            self._delete_counts[h] = self._delete_counts.get(h, 0) + 1

    def get_stats(self) -> SsdStorageGateStats:
        """
        Return aggregate gate counters without exposing per-chunk internals.

        Returns:
            Aggregated counts for tracked chunk hashes and completed lookup,
            read, write, and delete records.
        """
        with self._lock:
            return {
                "lookup_tracked_count": len(self._lookup_counts),
                "read_tracked_count": len(self._read_counts),
                "write_tracked_count": len(self._write_counts),
                "delete_tracked_count": len(self._delete_counts),
                "total_lookup_count": sum(self._lookup_counts.values()),
                "total_read_count": sum(self._read_counts.values()),
                "total_write_count": sum(self._write_counts.values()),
                "total_delete_count": sum(self._delete_counts.values()),
            }
