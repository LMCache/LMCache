# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

# Third Party
import prometheus_client

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class PrometheusLogger(ABC):
    """Abstract base class for per-component Prometheus loggers.

    Provides shared helper utilities for creating and logging to Prometheus
    metrics. Concrete subclasses define their own metric attributes and
    implement ``log_prometheus(stats)``.
    """

    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def __init__(
        self,
        labels: Dict[str, str],
        config: Optional[Any] = None,
    ) -> None:
        self.labels = labels
        self.config = config
        self._counters: List[prometheus_client.Counter] = []
        self._histograms: List[prometheus_client.Histogram] = []
        self._gauges: List[prometheus_client.Gauge] = []

    def create_gauge(
        self,
        name: str,
        documentation: str,
        labelnames: List[str],
    ) -> prometheus_client.Gauge:
        """Create a Gauge and register it for unregister()."""
        gauge = self._gauge_cls(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
        )
        self._gauges.append(gauge)
        return gauge

    def create_counter(
        self,
        name: str,
        documentation: str,
        labelnames: List[str],
    ) -> prometheus_client.Counter:
        """Create a Counter and register it for reset_counters()."""
        counter = self._counter_cls(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
        )
        self._counters.append(counter)
        return counter

    def create_histogram(
        self,
        name: str,
        documentation: str,
        labelnames: List[str],
        buckets: Sequence[float],
    ) -> prometheus_client.Histogram:
        """Create a Histogram and register it for reset_histograms().

        If the ``config`` object's extra_config contains a key
        ``histogram_bucket_<short_name>`` (where ``<short_name>`` is the
        metric name after the first ``:`` separator), the value will be used
        as the bucket list, overriding the default *buckets* argument.
        """
        short_name = name.split(":", 1)[-1]
        config_key = "histogram_bucket_%s" % short_name
        custom = (
            self.config.get_extra_config_value(config_key)
            if self.config is not None
            else None
        )
        if custom is not None:
            buckets = custom
            logger.info(
                "Using custom buckets for histogram %s from extra_config key '%s'",
                name,
                config_key,
            )
        histogram = self._histogram_cls(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
            buckets=buckets,
        )
        self._histograms.append(histogram)
        return histogram

    def log_gauge(self, gauge: Any, data: Union[int, float]) -> None:
        """Set a gauge metric using the instance labels."""
        if self.labels:
            gauge.labels(**self.labels).set(data)
        else:
            gauge.set(data)

    def log_counter(self, counter: Any, data: Union[int, float]) -> None:
        """Increment a counter metric; negative values are silently ignored."""
        if data < 0:
            return
        if self.labels:
            counter.labels(**self.labels).inc(data)
        else:
            counter.inc(data)

    def log_histogram(
        self, histogram: Any, data: Union[List[int], List[float]]
    ) -> None:
        """Observe a list of values on a histogram metric."""
        if self.labels:
            for value in data:
                histogram.labels(**self.labels).observe(value)
        else:
            for value in data:
                histogram.observe(value)

    @abstractmethod
    def log_prometheus(self) -> None:
        """Log accumulated stats to Prometheus and reset internal counters."""
        ...

    def reset_counters(self) -> None:
        """Reset all tracked Counter metrics and re-initialise with labels."""
        for counter in self._counters:
            counter.clear()
            counter.labels(**self.labels)

    def reset_histograms(self) -> None:
        """Reset all tracked Histogram metrics and re-initialise with labels."""
        for histogram in self._histograms:
            histogram.clear()
            histogram.labels(**self.labels)

    def unregister(self) -> None:
        """Unregister all tracked metrics from the Prometheus registry.

        Called on shutdown so that re-instantiation (e.g. in tests) does not
        raise ``ValueError: Duplicated timeseries in CollectorRegistry``.
        """
        for metric in (*self._counters, *self._histograms, *self._gauges):
            try:
                prometheus_client.REGISTRY.unregister(metric)
            except Exception:
                pass
