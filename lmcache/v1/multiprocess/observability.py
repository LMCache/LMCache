# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Dict, Optional
import os
import threading

# Third Party
import prometheus_client

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class MPServerStats:
    """
    Statistics for the MP Server, aligned with MixedMemoryAllocator design.

    The allocator uses an "explicit free list" where free memory is tracked
    as a sorted list of "holes" (free blocks). Allocated regions are the
    gaps between these holes.
    """

    # === Request Counters ===
    store_requests: int
    lookup_requests: int
    retrieve_requests: int

    # === Token Counters ===
    stored_tokens: int
    lookup_tokens: int
    lookup_hit_tokens: int
    lookup_hit_rate: float

    # === Basic Memory Stats ===
    total_memory_bytes: int
    used_memory_bytes: int
    free_memory_bytes: int
    memory_utilization: float
    align_bytes: int

    # === Allocation Tracking ===
    num_active_allocations: int
    num_allocated_regions: int

    # === Cache Key Counts ===
    committed_keys_count: int
    reserved_keys_count: int
    locked_keys_count: int

    # === Eviction Stats ===
    eviction_count: int
    evicted_keys_count: int

    # === KV Cache Operation Duration ===
    store_time_total_ms: float  # Total store time in ms
    retrieve_time_total_ms: float  # Total retrieve time in ms
    store_duration_avg_ms: float  # Average store duration per operation
    retrieve_duration_avg_ms: float  # Average retrieve duration per operation
    store_duration_last_ms: float  # Last store operation duration
    retrieve_duration_last_ms: float  # Last retrieve operation duration

    # === Hole (Free Block) Statistics ===
    # In explicit-list allocators, "holes" are the free blocks
    num_holes: int  # Number of free blocks in explicit list
    largest_hole_bytes: int  # Largest contiguous free block
    smallest_hole_bytes: int  # Smallest free block
    avg_hole_bytes: int  # Average hole size
    median_hole_bytes: int  # Median hole size
    std_hole_bytes: int  # Standard deviation of hole sizes

    # === Fragmentation Metrics ===
    external_fragmentation: float  # 1 - (largest/total_free), 0=good, 1=bad
    hole_scatter_index: float  # (holes-1)/allocations, lower=better
    allocation_efficiency: float  # usable_holes/total_free, 1=all usable
    unusable_bytes: int  # Bytes in holes < align_bytes
    unusable_hole_count: int  # Count of holes < align_bytes
    compaction_benefit_bytes: int  # Bytes we could reclaim by compaction
    non_coalesced_pairs: int  # Should be 0 if allocator is healthy

    # === Hole Size Distribution ===
    # Aligned with typical KV cache chunk sizes (~4MB per chunk)
    holes_unusable: int  # < align_bytes (completely unusable)
    holes_tiny: int  # < 1MB (< 1 chunk)
    holes_small: int  # 1-4MB (1 chunk)
    holes_medium: int  # 4-16MB (1-4 chunks)
    holes_large: int  # 16-64MB (4-16 chunks)
    holes_xlarge: int  # 64-256MB (16-64 chunks)
    holes_huge: int  # > 256MB (64+ chunks)


class MPStatsCollector:
    """
    Collects statistics for the MP Server.
    Thread-safe singleton aligned with MixedMemoryAllocator design.
    """

    _instance: Optional["MPStatsCollector"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._data_lock = threading.Lock()

        # Request counters
        self.store_requests = 0
        self.lookup_requests = 0
        self.retrieve_requests = 0

        # Token counters
        self.stored_tokens = 0
        self.lookup_tokens = 0
        self.lookup_hit_tokens = 0

        # Eviction stats
        self.eviction_count = 0
        self.evicted_keys_count = 0

        # KV Cache operation duration tracking
        self.store_time_total_ms = 0.0
        self.retrieve_time_total_ms = 0.0
        self.store_duration_last_ms = 0.0
        self.retrieve_duration_last_ms = 0.0
        # For calculating average duration per interval
        self._interval_store_count = 0
        self._interval_store_time_ms = 0.0
        self._interval_retrieve_count = 0
        self._interval_retrieve_time_ms = 0.0

        # Basic memory stats
        self.total_memory_bytes = 0
        self.used_memory_bytes = 0
        self.free_memory_bytes = 0
        self.align_bytes = 4096

        # Allocation tracking
        self.num_active_allocations = 0
        self.num_allocated_regions = 0

        # Cache key counts
        self.committed_keys_count = 0
        self.reserved_keys_count = 0
        self.locked_keys_count = 0

        # Hole (free block) statistics
        self.num_holes = 0
        self.largest_hole_bytes = 0
        self.smallest_hole_bytes = 0
        self.avg_hole_bytes = 0
        self.median_hole_bytes = 0
        self.std_hole_bytes = 0

        # Fragmentation metrics
        self.external_fragmentation = 0.0
        self.hole_scatter_index = 0.0
        self.allocation_efficiency = 1.0
        self.unusable_bytes = 0
        self.unusable_hole_count = 0
        self.compaction_benefit_bytes = 0
        self.non_coalesced_pairs = 0

        # Hole size distribution
        self.hole_distribution: Dict[str, int] = {
            "unusable_lt_align": 0,
            "tiny_lt_1mb": 0,
            "small_1mb_4mb": 0,
            "medium_4mb_16mb": 0,
            "large_16mb_64mb": 0,
            "xlarge_64mb_256mb": 0,
            "huge_gt_256mb": 0,
        }

    @classmethod
    def get_instance(cls) -> "MPStatsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def on_store(self, num_tokens: int, elapsed_ms: float) -> None:
        """Track a completed store operation with timing."""
        with self._data_lock:
            self.store_requests += 1
            self.stored_tokens += num_tokens
            self.store_time_total_ms += elapsed_ms
            self.store_duration_last_ms = elapsed_ms
            self._interval_store_count += 1
            self._interval_store_time_ms += elapsed_ms

    def on_lookup(self, num_tokens: int, hit_tokens: int) -> None:
        with self._data_lock:
            self.lookup_requests += 1
            self.lookup_tokens += num_tokens
            self.lookup_hit_tokens += hit_tokens

    def on_retrieve(self, num_tokens: int, elapsed_ms: float) -> None:
        """Track a completed retrieve operation with timing."""
        with self._data_lock:
            self.retrieve_requests += 1
            self.retrieve_time_total_ms += elapsed_ms
            self.retrieve_duration_last_ms = elapsed_ms
            self._interval_retrieve_count += 1
            self._interval_retrieve_time_ms += elapsed_ms

    def on_eviction(self, evicted_keys: int) -> None:
        with self._data_lock:
            self.eviction_count += 1
            self.evicted_keys_count += evicted_keys

    def update_memory_stats(self, stats: dict) -> None:
        with self._data_lock:
            # Basic memory stats
            self.total_memory_bytes = stats.get("total_bytes", 0)
            self.used_memory_bytes = stats.get("used_bytes", 0)
            self.free_memory_bytes = stats.get("free_bytes", 0)
            self.align_bytes = stats.get("align_bytes", 4096)

            # Allocation tracking
            self.num_active_allocations = stats.get("num_active_allocations", 0)
            self.num_allocated_regions = stats.get("num_allocated_regions", 0)

            # Cache key counts
            self.committed_keys_count = stats.get("committed_keys_count", 0)
            self.reserved_keys_count = stats.get("reserved_keys_count", 0)
            self.locked_keys_count = stats.get("locked_keys_count", 0)

            # Hole (free block) statistics
            self.num_holes = stats.get("num_holes", 0)
            self.largest_hole_bytes = stats.get("largest_hole_bytes", 0)
            self.smallest_hole_bytes = stats.get("smallest_hole_bytes", 0)
            self.avg_hole_bytes = stats.get("avg_hole_bytes", 0)
            self.median_hole_bytes = stats.get("median_hole_bytes", 0)
            self.std_hole_bytes = stats.get("std_hole_bytes", 0)

            # Fragmentation metrics
            self.external_fragmentation = stats.get("external_fragmentation", 0.0)
            self.hole_scatter_index = stats.get("hole_scatter_index", 0.0)
            self.allocation_efficiency = stats.get("allocation_efficiency", 1.0)
            self.unusable_bytes = stats.get("unusable_bytes", 0)
            self.unusable_hole_count = stats.get("unusable_hole_count", 0)
            self.compaction_benefit_bytes = stats.get("compaction_benefit_bytes", 0)
            self.non_coalesced_pairs = stats.get("non_coalesced_pairs", 0)

            # Hole size distribution
            if "hole_distribution" in stats and stats["hole_distribution"]:
                self.hole_distribution = stats["hole_distribution"]

    def get_stats(self) -> MPServerStats:
        with self._data_lock:
            # Calculate derived metrics
            memory_utilization = (
                self.used_memory_bytes / self.total_memory_bytes
                if self.total_memory_bytes > 0
                else 0.0
            )
            lookup_hit_rate = (
                self.lookup_hit_tokens / self.lookup_tokens
                if self.lookup_tokens > 0
                else 0.0
            )

            # Calculate average duration per operation in this interval
            store_duration_avg_ms = (
                self._interval_store_time_ms / self._interval_store_count
                if self._interval_store_count > 0
                else 0.0
            )
            retrieve_duration_avg_ms = (
                self._interval_retrieve_time_ms / self._interval_retrieve_count
                if self._interval_retrieve_count > 0
                else 0.0
            )

            # Reset interval counters after reading
            self._interval_store_count = 0
            self._interval_store_time_ms = 0.0
            self._interval_retrieve_count = 0
            self._interval_retrieve_time_ms = 0.0

            return MPServerStats(
                # Request counters
                store_requests=self.store_requests,
                lookup_requests=self.lookup_requests,
                retrieve_requests=self.retrieve_requests,
                # Token counters
                stored_tokens=self.stored_tokens,
                lookup_tokens=self.lookup_tokens,
                lookup_hit_tokens=self.lookup_hit_tokens,
                lookup_hit_rate=lookup_hit_rate,
                # Basic memory stats
                total_memory_bytes=self.total_memory_bytes,
                used_memory_bytes=self.used_memory_bytes,
                free_memory_bytes=self.free_memory_bytes,
                memory_utilization=memory_utilization,
                align_bytes=self.align_bytes,
                # Allocation tracking
                num_active_allocations=self.num_active_allocations,
                num_allocated_regions=self.num_allocated_regions,
                # Cache key counts
                committed_keys_count=self.committed_keys_count,
                reserved_keys_count=self.reserved_keys_count,
                locked_keys_count=self.locked_keys_count,
                # Eviction stats
                eviction_count=self.eviction_count,
                evicted_keys_count=self.evicted_keys_count,
                # KV Cache operation duration
                store_time_total_ms=self.store_time_total_ms,
                retrieve_time_total_ms=self.retrieve_time_total_ms,
                store_duration_avg_ms=store_duration_avg_ms,
                retrieve_duration_avg_ms=retrieve_duration_avg_ms,
                store_duration_last_ms=self.store_duration_last_ms,
                retrieve_duration_last_ms=self.retrieve_duration_last_ms,
                # Hole statistics
                num_holes=self.num_holes,
                largest_hole_bytes=self.largest_hole_bytes,
                smallest_hole_bytes=self.smallest_hole_bytes,
                avg_hole_bytes=self.avg_hole_bytes,
                median_hole_bytes=self.median_hole_bytes,
                std_hole_bytes=self.std_hole_bytes,
                # Fragmentation metrics
                external_fragmentation=self.external_fragmentation,
                hole_scatter_index=self.hole_scatter_index,
                allocation_efficiency=self.allocation_efficiency,
                unusable_bytes=self.unusable_bytes,
                unusable_hole_count=self.unusable_hole_count,
                compaction_benefit_bytes=self.compaction_benefit_bytes,
                non_coalesced_pairs=self.non_coalesced_pairs,
                # Hole size distribution
                holes_unusable=self.hole_distribution.get("unusable_lt_align", 0),
                holes_tiny=self.hole_distribution.get("tiny_lt_1mb", 0),
                holes_small=self.hole_distribution.get("small_1mb_4mb", 0),
                holes_medium=self.hole_distribution.get("medium_4mb_16mb", 0),
                holes_large=self.hole_distribution.get("large_16mb_64mb", 0),
                holes_xlarge=self.hole_distribution.get("xlarge_64mb_256mb", 0),
                holes_huge=self.hole_distribution.get("huge_gt_256mb", 0),
            )


class MPPrometheusExporter:
    """
    Exports MP Server metrics to Prometheus.
    Aligned with MixedMemoryAllocator's explicit free list design.
    """

    def __init__(self, host: str, port: int, chunk_size: int):
        # Ensure PROMETHEUS_MULTIPROC_DIR is set
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            default_dir = "/tmp/lmcache_mp_prometheus"
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = default_dir
            if not os.path.exists(default_dir):
                os.makedirs(default_dir, exist_ok=True)

        self.labels = {
            "host": host,
            "port": str(port),
            "chunk_size": str(chunk_size),
        }
        labelnames = list(self.labels.keys())

        # === Request Counters ===
        self.counter_store_requests = prometheus_client.Counter(
            "lmcache_mp:store_requests", "Total store requests", labelnames
        )
        self.counter_lookup_requests = prometheus_client.Counter(
            "lmcache_mp:lookup_requests", "Total lookup requests", labelnames
        )
        self.counter_retrieve_requests = prometheus_client.Counter(
            "lmcache_mp:retrieve_requests", "Total retrieve requests", labelnames
        )

        # === Token Counters ===
        self.counter_stored_tokens = prometheus_client.Counter(
            "lmcache_mp:stored_tokens", "Total tokens stored", labelnames
        )
        self.counter_lookup_tokens = prometheus_client.Counter(
            "lmcache_mp:lookup_tokens", "Total tokens looked up", labelnames
        )
        self.counter_lookup_hit_tokens = prometheus_client.Counter(
            "lmcache_mp:lookup_hit_tokens", "Total tokens hit in lookup", labelnames
        )

        # === Eviction Counters ===
        self.counter_eviction_count = prometheus_client.Counter(
            "lmcache_mp:eviction_count", "Total eviction operations", labelnames
        )
        self.counter_evicted_keys = prometheus_client.Counter(
            "lmcache_mp:evicted_keys", "Total keys evicted", labelnames
        )

        # === KV Cache Operation Duration ===
        self.gauge_store_duration_avg = prometheus_client.Gauge(
            "lmcache_mp:store_duration_avg_ms",
            "Average store operation duration (ms)",
            labelnames,
        )
        self.gauge_retrieve_duration_avg = prometheus_client.Gauge(
            "lmcache_mp:retrieve_duration_avg_ms",
            "Average retrieve operation duration (ms)",
            labelnames,
        )
        self.gauge_store_duration_last = prometheus_client.Gauge(
            "lmcache_mp:store_duration_last_ms",
            "Last store operation duration (ms)",
            labelnames,
        )
        self.gauge_retrieve_duration_last = prometheus_client.Gauge(
            "lmcache_mp:retrieve_duration_last_ms",
            "Last retrieve operation duration (ms)",
            labelnames,
        )
        self.gauge_store_time_total = prometheus_client.Gauge(
            "lmcache_mp:store_time_total_ms",
            "Total cumulative store time (ms)",
            labelnames,
        )
        self.gauge_retrieve_time_total = prometheus_client.Gauge(
            "lmcache_mp:retrieve_time_total_ms",
            "Total cumulative retrieve time (ms)",
            labelnames,
        )

        # === Hit Rate ===
        self.gauge_lookup_hit_rate = prometheus_client.Gauge(
            "lmcache_mp:lookup_hit_rate", "Lookup hit rate (0.0 - 1.0)", labelnames
        )

        # === Memory Stats ===
        self.gauge_total_memory = prometheus_client.Gauge(
            "lmcache_mp:total_memory_bytes", "Total memory (bytes)", labelnames
        )
        self.gauge_used_memory = prometheus_client.Gauge(
            "lmcache_mp:used_memory_bytes", "Used memory (bytes)", labelnames
        )
        self.gauge_free_memory = prometheus_client.Gauge(
            "lmcache_mp:free_memory_bytes", "Free memory (bytes)", labelnames
        )
        self.gauge_memory_utilization = prometheus_client.Gauge(
            "lmcache_mp:memory_utilization", "Memory utilization (0.0-1.0)", labelnames
        )

        # === Allocation Tracking ===
        self.gauge_num_active_allocations = prometheus_client.Gauge(
            "lmcache_mp:num_active_allocations", "Active allocations", labelnames
        )
        self.gauge_num_allocated_regions = prometheus_client.Gauge(
            "lmcache_mp:num_allocated_regions", "Allocated regions", labelnames
        )

        # === Cache Key Counts ===
        self.gauge_committed_keys = prometheus_client.Gauge(
            "lmcache_mp:committed_keys_count", "Committed cache keys", labelnames
        )
        self.gauge_reserved_keys = prometheus_client.Gauge(
            "lmcache_mp:reserved_keys_count", "Reserved cache keys", labelnames
        )
        self.gauge_locked_keys = prometheus_client.Gauge(
            "lmcache_mp:locked_keys_count", "Locked cache keys", labelnames
        )

        # === Hole (Free Block) Statistics ===
        self.gauge_num_holes = prometheus_client.Gauge(
            "lmcache_mp:num_holes", "Number of holes (free blocks)", labelnames
        )
        self.gauge_largest_hole = prometheus_client.Gauge(
            "lmcache_mp:largest_hole_bytes", "Largest hole size (bytes)", labelnames
        )
        self.gauge_smallest_hole = prometheus_client.Gauge(
            "lmcache_mp:smallest_hole_bytes", "Smallest hole size (bytes)", labelnames
        )
        self.gauge_avg_hole = prometheus_client.Gauge(
            "lmcache_mp:avg_hole_bytes", "Average hole size (bytes)", labelnames
        )
        self.gauge_median_hole = prometheus_client.Gauge(
            "lmcache_mp:median_hole_bytes", "Median hole size (bytes)", labelnames
        )

        # === Fragmentation Metrics ===
        self.gauge_external_fragmentation = prometheus_client.Gauge(
            "lmcache_mp:external_fragmentation",
            "External fragmentation: 1-(largest/total). 0=good, 1=bad",
            labelnames,
        )
        self.gauge_hole_scatter_index = prometheus_client.Gauge(
            "lmcache_mp:hole_scatter_index",
            "Hole scatter: (holes-1)/allocations. Lower=better",
            labelnames,
        )
        self.gauge_allocation_efficiency = prometheus_client.Gauge(
            "lmcache_mp:allocation_efficiency",
            "Allocatable portion of free memory (0.0-1.0)",
            labelnames,
        )
        self.gauge_unusable_bytes = prometheus_client.Gauge(
            "lmcache_mp:unusable_bytes", "Bytes in unusable holes", labelnames
        )
        self.gauge_unusable_hole_count = prometheus_client.Gauge(
            "lmcache_mp:unusable_hole_count", "Count of unusable holes", labelnames
        )
        self.gauge_compaction_benefit = prometheus_client.Gauge(
            "lmcache_mp:compaction_benefit_bytes",
            "Potential gain from compaction",
            labelnames,
        )
        self.gauge_non_coalesced_pairs = prometheus_client.Gauge(
            "lmcache_mp:non_coalesced_pairs",
            "Adjacent holes not coalesced (should be 0)",
            labelnames,
        )

        # === Hole Size Distribution ===
        self.gauge_holes_unusable = prometheus_client.Gauge(
            "lmcache_mp:holes_unusable", "Holes < align_bytes (unusable)", labelnames
        )
        self.gauge_holes_tiny = prometheus_client.Gauge(
            "lmcache_mp:holes_tiny", "Holes < 1MB", labelnames
        )
        self.gauge_holes_small = prometheus_client.Gauge(
            "lmcache_mp:holes_small", "Holes 1-4MB (~1 chunk)", labelnames
        )
        self.gauge_holes_medium = prometheus_client.Gauge(
            "lmcache_mp:holes_medium", "Holes 4-16MB (1-4 chunks)", labelnames
        )
        self.gauge_holes_large = prometheus_client.Gauge(
            "lmcache_mp:holes_large", "Holes 16-64MB (4-16 chunks)", labelnames
        )
        self.gauge_holes_xlarge = prometheus_client.Gauge(
            "lmcache_mp:holes_xlarge", "Holes 64-256MB (16-64 chunks)", labelnames
        )
        self.gauge_holes_huge = prometheus_client.Gauge(
            "lmcache_mp:holes_huge", "Holes > 256MB (64+ chunks)", labelnames
        )

        self._prev_stats: Optional[MPServerStats] = None

    def export(self, stats: MPServerStats) -> None:
        """Export stats to Prometheus metrics."""
        labels = self.labels

        # === Counter increments (delta from previous) ===
        if self._prev_stats:
            self.counter_store_requests.labels(**labels).inc(
                max(0, stats.store_requests - self._prev_stats.store_requests)
            )
            self.counter_lookup_requests.labels(**labels).inc(
                max(0, stats.lookup_requests - self._prev_stats.lookup_requests)
            )
            self.counter_retrieve_requests.labels(**labels).inc(
                max(0, stats.retrieve_requests - self._prev_stats.retrieve_requests)
            )
            self.counter_stored_tokens.labels(**labels).inc(
                max(0, stats.stored_tokens - self._prev_stats.stored_tokens)
            )
            self.counter_lookup_tokens.labels(**labels).inc(
                max(0, stats.lookup_tokens - self._prev_stats.lookup_tokens)
            )
            self.counter_lookup_hit_tokens.labels(**labels).inc(
                max(0, stats.lookup_hit_tokens - self._prev_stats.lookup_hit_tokens)
            )
            self.counter_eviction_count.labels(**labels).inc(
                max(0, stats.eviction_count - self._prev_stats.eviction_count)
            )
            self.counter_evicted_keys.labels(**labels).inc(
                max(0, stats.evicted_keys_count - self._prev_stats.evicted_keys_count)
            )
        else:
            self.counter_store_requests.labels(**labels).inc(stats.store_requests)
            self.counter_lookup_requests.labels(**labels).inc(stats.lookup_requests)
            self.counter_retrieve_requests.labels(**labels).inc(stats.retrieve_requests)
            self.counter_stored_tokens.labels(**labels).inc(stats.stored_tokens)
            self.counter_lookup_tokens.labels(**labels).inc(stats.lookup_tokens)
            self.counter_lookup_hit_tokens.labels(**labels).inc(stats.lookup_hit_tokens)
            self.counter_eviction_count.labels(**labels).inc(stats.eviction_count)
            self.counter_evicted_keys.labels(**labels).inc(stats.evicted_keys_count)

        # === Hit Rate ===
        self.gauge_lookup_hit_rate.labels(**labels).set(stats.lookup_hit_rate)

        # === KV Cache Operation Duration ===
        self.gauge_store_duration_avg.labels(**labels).set(stats.store_duration_avg_ms)
        self.gauge_retrieve_duration_avg.labels(**labels).set(
            stats.retrieve_duration_avg_ms
        )
        self.gauge_store_duration_last.labels(**labels).set(
            stats.store_duration_last_ms
        )
        self.gauge_retrieve_duration_last.labels(**labels).set(
            stats.retrieve_duration_last_ms
        )
        self.gauge_store_time_total.labels(**labels).set(stats.store_time_total_ms)
        self.gauge_retrieve_time_total.labels(**labels).set(
            stats.retrieve_time_total_ms
        )

        # === Memory Stats ===
        self.gauge_total_memory.labels(**labels).set(stats.total_memory_bytes)
        self.gauge_used_memory.labels(**labels).set(stats.used_memory_bytes)
        self.gauge_free_memory.labels(**labels).set(stats.free_memory_bytes)
        self.gauge_memory_utilization.labels(**labels).set(stats.memory_utilization)

        # === Allocation Tracking ===
        self.gauge_num_active_allocations.labels(**labels).set(
            stats.num_active_allocations
        )
        self.gauge_num_allocated_regions.labels(**labels).set(
            stats.num_allocated_regions
        )

        # === Cache Key Counts ===
        self.gauge_committed_keys.labels(**labels).set(stats.committed_keys_count)
        self.gauge_reserved_keys.labels(**labels).set(stats.reserved_keys_count)
        self.gauge_locked_keys.labels(**labels).set(stats.locked_keys_count)

        # === Hole Statistics ===
        self.gauge_num_holes.labels(**labels).set(stats.num_holes)
        self.gauge_largest_hole.labels(**labels).set(stats.largest_hole_bytes)
        self.gauge_smallest_hole.labels(**labels).set(stats.smallest_hole_bytes)
        self.gauge_avg_hole.labels(**labels).set(stats.avg_hole_bytes)
        self.gauge_median_hole.labels(**labels).set(stats.median_hole_bytes)

        # === Fragmentation Metrics ===
        self.gauge_external_fragmentation.labels(**labels).set(
            stats.external_fragmentation
        )
        self.gauge_hole_scatter_index.labels(**labels).set(stats.hole_scatter_index)
        self.gauge_allocation_efficiency.labels(**labels).set(
            stats.allocation_efficiency
        )
        self.gauge_unusable_bytes.labels(**labels).set(stats.unusable_bytes)
        self.gauge_unusable_hole_count.labels(**labels).set(stats.unusable_hole_count)
        self.gauge_compaction_benefit.labels(**labels).set(
            stats.compaction_benefit_bytes
        )
        self.gauge_non_coalesced_pairs.labels(**labels).set(stats.non_coalesced_pairs)

        # === Hole Size Distribution ===
        self.gauge_holes_unusable.labels(**labels).set(stats.holes_unusable)
        self.gauge_holes_tiny.labels(**labels).set(stats.holes_tiny)
        self.gauge_holes_small.labels(**labels).set(stats.holes_small)
        self.gauge_holes_medium.labels(**labels).set(stats.holes_medium)
        self.gauge_holes_large.labels(**labels).set(stats.holes_large)
        self.gauge_holes_xlarge.labels(**labels).set(stats.holes_xlarge)
        self.gauge_holes_huge.labels(**labels).set(stats.holes_huge)

        self._prev_stats = stats


class MPMetricsLogger:
    """
    Background thread that periodically collects and exports metrics.
    """

    def __init__(
        self,
        host: str,
        port: int,
        chunk_size: int,
        log_interval: int = 5,
    ):
        self.collector = MPStatsCollector.get_instance()
        self.exporter = MPPrometheusExporter(host, port, chunk_size)
        self.log_interval = log_interval
        self.is_running = True
        self.shutdown_event = threading.Event()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("MPMetricsLogger started with interval %d seconds", log_interval)

    def _worker(self) -> None:
        while self.is_running:
            try:
                stats = self.collector.get_stats()
                self.exporter.export(stats)
            except Exception as e:
                logger.warning("Error exporting MP metrics: %s", str(e))
            self.shutdown_event.wait(self.log_interval)

    def shutdown(self) -> None:
        self.is_running = False
        self.shutdown_event.set()
        self.thread.join(timeout=2.0)
        logger.info("MPMetricsLogger shutdown complete")
