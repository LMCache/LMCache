# SPDX-License-Identifier: Apache-2.0

"""L1 metrics subscriber — OTel counters and chunk lifecycle histograms."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any
import random
import time

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


@dataclass
class _L1ChunkState:
    """Per-chunk lifecycle state in the shadow map."""

    alloc_time: float
    last_access_time: float


class L1MetricsSubscriber(EventSubscriber):
    """Maintains OTel counters and chunk lifecycle histograms for L1Manager.

    Counters:
    - ``lmcache_mp.l1_read_keys``  — keys read from L1
    - ``lmcache_mp.l1_write_keys`` — keys written to L1
    - ``lmcache_mp.l1_evicted_keys`` — keys evicted from L1

    Histograms (chunk lifecycle):
    - ``lmcache_mp.l1_chunk_lifetime_seconds`` — allocation to eviction
    - ``lmcache_mp.l1_chunk_idle_before_evict_seconds`` — last access to eviction
    - ``lmcache_mp.l1_chunk_reuse_gap_seconds`` — gap between consecutive touches
    - ``lmcache_mp.l1_chunk_evict_reuse_gap_seconds`` — eviction to next reuse (capped at ``max_evict_reuse_wait``)

    Parameters:
        sample_rate: Fraction of chunks to track for lifecycle histograms
            (0, 1.0].  Default 0.01 (1%).  Counters always count all events.
        max_evict_reuse_wait: Maximum seconds to track an evicted chunk
            waiting for reuse.  If not reused within this window the gap
            is reported as ``max_evict_reuse_wait`` and the entry is
            discarded.  Default 300 s (5 min).
    """

    def __init__(
        self,
        sample_rate: float = 0.01,
        max_evict_reuse_wait: float = 300.0,
    ) -> None:
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self._sample_rate = sample_rate
        self._max_evict_reuse_wait = max_evict_reuse_wait
        meter = metrics.get_meter("lmcache.l1")
        self._read_counter = meter.create_counter(
            "lmcache_mp.l1_read_keys",
            description="Total keys read from L1",
        )
        self._write_counter = meter.create_counter(
            "lmcache_mp.l1_write_keys",
            description="Total keys written to L1",
        )
        self._evicted_counter = meter.create_counter(
            "lmcache_mp.l1_evicted_keys",
            description="Total keys evicted from L1",
        )
        self._lifetime_hist = meter.create_histogram(
            "lmcache_mp.l1_chunk_lifetime_seconds",
            description=(
                "Histogram of L1 chunk lifetime from allocation to eviction (seconds)."
            ),
            unit="s",
        )
        self._idle_hist = meter.create_histogram(
            "lmcache_mp.l1_chunk_idle_before_evict_seconds",
            description=("Histogram of idle time before L1 chunk eviction (seconds)."),
            unit="s",
        )
        self._reuse_gap_hist = meter.create_histogram(
            "lmcache_mp.l1_chunk_reuse_gap_seconds",
            description=(
                "Histogram of time gaps between consecutive "
                "touches (write or read) of the same L1 chunk (seconds)."
            ),
            unit="s",
        )
        self._evict_reuse_gap_hist = meter.create_histogram(
            "lmcache_mp.l1_chunk_evict_reuse_gap_seconds",
            description=(
                "Histogram of time from L1 chunk eviction to "
                "next reuse.  Capped at max_evict_reuse_wait."
            ),
            unit="s",
        )

        # Shadow map: key -> chunk lifecycle state (live chunks).
        self._shadow: dict[Any, _L1ChunkState] = {}
        # Evicted map: key -> eviction timestamp (waiting for reuse).
        self._evicted_at: dict[Any, float] = {}
        # Keys we decided to sample (much smaller than tracking the
        # non-sampled set when sample_rate is low).
        self._sampled: set[Any] = set()

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L1_READ_FINISHED: self._on_read_finished,
            EventType.L1_WRITE_FINISHED: self._on_write_finished,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED: self._on_write_finished,
            EventType.L1_KEYS_EVICTED: self._on_evicted,
        }

    def _on_read_finished(self, event: Event) -> None:
        keys = event.metadata["keys"]
        self._read_counter.add(len(keys))
        now = event.timestamp or time.time()
        for key in keys:
            state = self._shadow.get(key)
            if state is not None:
                self._reuse_gap_hist.record(now - state.last_access_time)
                state.last_access_time = now

    def _on_write_finished(self, event: Event) -> None:
        keys = event.metadata["keys"]
        self._write_counter.add(len(keys))
        now = event.timestamp or time.time()
        for key in keys:
            # Check if this is a reuse of an evicted chunk.
            evict_time = self._evicted_at.pop(key, None)
            if evict_time is not None:
                gap = min(now - evict_time, self._max_evict_reuse_wait)
                self._evict_reuse_gap_hist.record(gap)

            state = self._shadow.get(key)
            if state is not None:
                # Re-write of existing chunk counts as a touch.
                self._reuse_gap_hist.record(now - state.last_access_time)
                self._shadow[key] = _L1ChunkState(
                    alloc_time=now,
                    last_access_time=now,
                )
            else:
                # Not currently tracked — re-roll for non-sampled keys
                # (cheaper than storing all skipped keys).
                if key not in self._sampled and not self._should_sample():
                    continue
                self._sampled.add(key)
                self._shadow[key] = _L1ChunkState(
                    alloc_time=now,
                    last_access_time=now,
                )
        self._sweep_stale_evictions(now)

    def _on_evicted(self, event: Event) -> None:
        keys = event.metadata["keys"]
        self._evicted_counter.add(len(keys))
        now = event.timestamp or time.time()
        for key in keys:
            state = self._shadow.pop(key, None)
            if state is not None:
                self._lifetime_hist.record(now - state.alloc_time)
                self._idle_hist.record(now - state.last_access_time)
                # Start tracking eviction-to-reuse gap (only for sampled).
                self._evicted_at[key] = now
        self._sweep_stale_evictions(now)

    def _should_sample(self) -> bool:
        return random.random() < self._sample_rate

    def _sweep_stale_evictions(self, now: float) -> None:
        """Report T and discard evicted entries older than max_evict_reuse_wait."""
        stale = [
            key
            for key, evict_time in self._evicted_at.items()
            if now - evict_time >= self._max_evict_reuse_wait
        ]
        for key in stale:
            self._evicted_at.pop(key, None)
            self._evict_reuse_gap_hist.record(self._max_evict_reuse_wait)
