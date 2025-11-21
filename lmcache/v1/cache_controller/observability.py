# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
from prometheus_client import REGISTRY
import prometheus_client

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class PrometheusLogger:
    """
    Prometheus logger for cache controller metrics.
    Provides dynamic metrics for monitoring KV pool and worker registration.
    """

    _gauge_cls = prometheus_client.Gauge

    def __init__(self, labels: dict):
        self.labels = labels
        labelnames = list(self.labels.keys())

        # Dynamic metrics for cache controller
        self._init_dynamic_metrics(labelnames)

    def _init_dynamic_metrics(self, labelnames):
        """
        Initialize dynamic metrics that will be updated by lambda functions.
        """
        # KV Pool metrics
        self.kv_pool_keys_count = self._gauge_cls(
            name="lmcache:cache_controller_kv_pool_keys_count",
            documentation="The number of keys in the KV pool",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)

        # Registration Controller metrics
        self.registered_workers_count = self._gauge_cls(
            name="lmcache:cache_controller_registered_workers_count",
            documentation="The total number of registered workers",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)

    _instance = None

    @staticmethod
    def GetOrCreate(
        labels: dict,
    ) -> "PrometheusLogger":
        if PrometheusLogger._instance is None:
            PrometheusLogger._instance = PrometheusLogger(labels)
        if PrometheusLogger._instance.labels != labels:
            logger.error(
                "CacheControllerPrometheusLogger instance already created with "
                "different metadata. This should not happen except in test"
            )
        return PrometheusLogger._instance

    @staticmethod
    def GetInstance() -> "PrometheusLogger":
        assert PrometheusLogger._instance is not None, (
            "CacheControllerPrometheusLogger instance not created yet"
        )
        return PrometheusLogger._instance

    @staticmethod
    def GetInstanceOrNone() -> Optional["PrometheusLogger"]:
        """
        Returns the singleton instance of CacheControllerPrometheusLogger if it exists,
        otherwise returns None.
        """
        return PrometheusLogger._instance

    @staticmethod
    def DestroyInstance():
        PrometheusLogger._instance = None

    @staticmethod
    def unregister_all_metrics():
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass
