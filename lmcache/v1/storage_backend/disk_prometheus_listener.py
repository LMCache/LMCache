# SPDX-License-Identifier: Apache-2.0
# Standard
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger

logger = init_logger(__name__)


class DiskPrometheusListener:
    """
    Collects local-disk SSD wear metrics and registers Prometheus gauges.

    All counter updates go through ``on_*`` methods so the backend stays
    decoupled from Prometheus wiring.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._disk_write_ops = 0
        self._disk_write_bytes = 0
        self._disk_remove_ops = 0
        self._disk_evict_bytes = 0
        self._disk_gated_count = 0
        self._disk_gated_by_length_count = 0
        self._disk_gated_by_frequency_count = 0

    def register(self) -> None:
        """
        Register gauges with PrometheusLogger when the singleton exists.

        Scraped values read the current counters held by this listener.
        """
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            return
        prometheus_logger.disk_write_ops.set_function(lambda: self._disk_write_ops)
        prometheus_logger.disk_write_bytes.set_function(lambda: self._disk_write_bytes)
        prometheus_logger.disk_remove_ops.set_function(lambda: self._disk_remove_ops)
        prometheus_logger.disk_evict_bytes.set_function(lambda: self._disk_evict_bytes)
        prometheus_logger.disk_gated_count.set_function(lambda: self._disk_gated_count)
        prometheus_logger.disk_gated_by_length_count.set_function(
            lambda: self._disk_gated_by_length_count
        )
        prometheus_logger.disk_gated_by_frequency_count.set_function(
            lambda: self._disk_gated_by_frequency_count
        )
        prometheus_logger.disk_write_avg_size_bytes.set_function(
            lambda: (
                self._disk_write_bytes // self._disk_write_ops
                if self._disk_write_ops > 0
                else 0
            )
        )

    def on_write(self, size_bytes: int) -> None:
        """Record one completed disk write of ``size_bytes``."""
        with self._lock:
            self._disk_write_ops += 1
            self._disk_write_bytes += size_bytes

    def on_remove(self, evict_size_bytes: int) -> None:
        """Record one disk remove (evict) of ``evict_size_bytes``."""
        with self._lock:
            self._disk_remove_ops += 1
            self._disk_evict_bytes += evict_size_bytes

    def on_gated_by_length(self) -> None:
        """Record a put gated by length-based policy."""
        with self._lock:
            self._disk_gated_count += 1
            self._disk_gated_by_length_count += 1

    def on_gated_by_frequency(self) -> None:
        """Record a put gated by frequency-based policy."""
        with self._lock:
            self._disk_gated_count += 1
            self._disk_gated_by_frequency_count += 1

    def log_summary(self) -> None:
        """Emit current counters at INFO (e.g. when the disk backend closes)."""
        with self._lock:
            wops = self._disk_write_ops
            wbytes = self._disk_write_bytes
            rops = self._disk_remove_ops
            ebytes = self._disk_evict_bytes
            gated = self._disk_gated_count
            glen = self._disk_gated_by_length_count
            gfreq = self._disk_gated_by_frequency_count
        avg_size = wbytes // wops if wops > 0 else 0
        logger.info(
            "Disk metrics: disk_write_ops=%s, disk_write_bytes=%s, "
            "disk_write_avg_size_bytes=%s, disk_remove_ops=%s, disk_evict_bytes=%s, "
            "disk_gated_count=%s (by_length=%s, by_frequency=%s)",
            wops,
            wbytes,
            avg_size,
            rops,
            ebytes,
            gated,
            glen,
            gfreq,
        )
