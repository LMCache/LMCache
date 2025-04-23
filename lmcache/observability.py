import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Union

import prometheus_client

from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import thread_safe

logger = init_logger(__name__)


@dataclass
class LMCacheStats:
    # Counter (will accumulate over time)
    num_retrieve_requests: int
    num_store_requests: int

    num_requested_tokens: int
    num_hit_tokens: int

    # Real time value measurements (will be reset after each log)
    cache_hit_rate: float

    local_cache_usage_bytes: int  # Size of the used local cache in bytes
    remote_cache_usage_bytes: int  # Size of the used remote cache in bytes
    local_storage_usage_bytes: int  # Size of the used local storage in bytes

    # Distribution measurements
    time_to_retrieve: List[float]
    time_to_store: List[float]
    retrieve_speed: List[float]  # Tokens per second
    store_speed: List[float]  # Tokens per second


@dataclass
class RetrieveRequestStats:
    num_tokens: int
    local_hit_tokens: int
    remote_hit_tokens: int  # Not used for now
    start_time: float
    end_time: float

    def time_to_retrieve(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def retrieve_speed(self):
        if self.time_to_retrieve() == 0:
            return 0
        return (self.local_hit_tokens + self.remote_hit_tokens) /\
                self.time_to_retrieve()


@dataclass
class StoreRequestStats:
    num_tokens: int
    start_time: float
    end_time: float

    def time_to_store(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def store_speed(self):
        if self.time_to_store() == 0:
            return 0
        return self.num_tokens / self.time_to_store()


class LMCStatsMonitor:

    def __init__(self):
        # Accumulated stats over time
        self.num_retrieve_requests = 0
        self.num_store_requests = 0

        self.num_requested_tokens = 0
        self.num_hit_tokens = 0

        # Interval metrics that will be reset after each log
        self.interval_requested_tokens = 0
        self.interval_hit_tokens = 0

        self.local_cache_usage_bytes = 0
        self.remote_cache_usage_bytes = 0
        self.local_storage_usage_bytes = 0

        self.retrieve_requests: Dict[int, RetrieveRequestStats] = {}
        self.store_requests: Dict[int, StoreRequestStats] = {}

        self.retrieve_request_id = 0
        self.store_request_id = 0

    @thread_safe
    def on_retrieve_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in 
        on_retrieve_finished
        """
        curr_time = time.time()
        retrieve_stats = RetrieveRequestStats(num_tokens=num_tokens,
                                              local_hit_tokens=0,
                                              remote_hit_tokens=0,
                                              start_time=curr_time,
                                              end_time=0)
        self.interval_requested_tokens += num_tokens
        self.num_requested_tokens += num_tokens
        self.num_retrieve_requests += 1
        self.retrieve_requests[self.retrieve_request_id] = retrieve_stats
        self.retrieve_request_id += 1
        return self.retrieve_request_id - 1

    @thread_safe
    def on_retrieve_finished(self, request_id: int, retrieved_tokens: int):
        curr_time = time.time()
        assert request_id in self.retrieve_requests
        retrieve_stats = self.retrieve_requests[request_id]
        retrieve_stats.local_hit_tokens = retrieved_tokens
        retrieve_stats.end_time = curr_time
        self.interval_hit_tokens += retrieved_tokens
        self.num_hit_tokens += retrieved_tokens

    @thread_safe
    def on_store_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in on_store_finished
        """
        curr_time = time.time()
        store_stats = StoreRequestStats(num_tokens=num_tokens,
                                        start_time=curr_time,
                                        end_time=0)
        self.num_store_requests += 1
        self.store_requests[self.store_request_id] = store_stats
        self.store_request_id += 1
        return self.store_request_id - 1

    @thread_safe
    def on_store_finished(self, request_id: int):
        curr_time = time.time()
        assert request_id in self.store_requests
        store_stats = self.store_requests[request_id]
        store_stats.end_time = curr_time

    @thread_safe
    def update_local_cache_usage(self, usage: int):
        self.local_cache_usage_bytes = usage

    @thread_safe
    def update_remote_cache_usage(self, usage: int):
        self.remote_cache_usage_bytes = usage

    @thread_safe
    def update_local_storage_usage(self, usage: int):
        self.local_storage_usage_bytes = usage

    @thread_safe
    def _clear(self):
        """
        Clear all the distribution stats 
        """
        self.interval_requested_tokens = 0
        self.interval_hit_tokens = 0

        new_retrieve_requests = {}
        for request_id, retrieve_stats in self.retrieve_requests.items():
            if retrieve_stats.end_time == 0:
                new_retrieve_requests[request_id] = retrieve_stats
        self.retrieve_requests = new_retrieve_requests

        new_store_requests = {}
        for request_id, store_stats in self.store_requests.items():
            if store_stats.end_time == 0:
                new_store_requests[request_id] = store_stats
        self.store_requests = new_store_requests

    @thread_safe
    def get_stats_and_clear(self) -> LMCacheStats:
        """
        This function should be called with by prometheus adapter with 
        a specific interval.
        The function will return the latest states between the current 
        call and the previous call.
        """
        cache_hit_rate = 0 if self.interval_requested_tokens == 0 else \
                self.interval_hit_tokens / self.interval_requested_tokens

        def filter_out_invalid(stats: List[float]):
            return [x for x in stats if x != 0]

        time_to_retrieve = filter_out_invalid([
            stats.time_to_retrieve()
            for stats in self.retrieve_requests.values()
        ])

        time_to_store = filter_out_invalid(
            [stats.time_to_store() for stats in self.store_requests.values()])

        retrieve_speed = filter_out_invalid([
            stats.retrieve_speed()
            for stats in self.retrieve_requests.values()
        ])

        store_speed = filter_out_invalid(
            [stats.store_speed() for stats in self.store_requests.values()])

        ret = LMCacheStats(
            num_retrieve_requests=self.num_retrieve_requests,
            num_store_requests=self.num_store_requests,
            num_requested_tokens=self.num_requested_tokens,
            num_hit_tokens=self.num_hit_tokens,
            cache_hit_rate=cache_hit_rate,
            local_cache_usage_bytes=self.local_cache_usage_bytes,
            remote_cache_usage_bytes=self.remote_cache_usage_bytes,
            local_storage_usage_bytes=self.local_storage_usage_bytes,
            time_to_retrieve=time_to_retrieve,
            time_to_store=time_to_store,
            retrieve_speed=retrieve_speed,
            store_speed=store_speed,
        )
        self._clear()
        return ret

    _instance = None

    @staticmethod
    def GetOrCreate() -> "LMCStatsMonitor":
        if LMCStatsMonitor._instance is None:
            LMCStatsMonitor._instance = LMCStatsMonitor()
        return LMCStatsMonitor._instance

    @staticmethod
    def DestroyInstance():
        LMCStatsMonitor._instance = None


class PrometheusLogger:
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def __init__(self, metadata: LMCacheEngineMetadata):
        self.metadata = metadata

        self.labels = self._metadata_to_labels(metadata)
        labelnames = list(self.labels.keys())

        self.counter_num_retrieve_requests = self._counter_cls(
            name="lmcache:num_retrieve_requests",
            documentation="Total number of retrieve requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_store_requests = self._counter_cls(
            name="lmcache:num_store_requests",
            documentation="Total number of store requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_requested_tokens = self._counter_cls(
            name="lmcache:num_requested_tokens",
            documentation="Total number of tokens requested from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_hit_tokens = self._counter_cls(
            name="lmcache:num_hit_tokens",
            documentation="Total number of tokens hit in lmcache",
            labelnames=labelnames,
        )

        self.gauge_cache_hit_rate = self._gauge_cls(
            name="lmcache:cache_hit_rate",
            documentation="Cache hit rate of lmcache since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent")

        self.gauge_local_cache_usage = self._gauge_cls(
            name="lmcache:local_cache_usage",
            documentation="Local cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum")

        self.gauge_remote_cache_usage = self._gauge_cls(
            name="lmcache:remote_cache_usage",
            documentation="Remote cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum")

        self.gauge_local_storage_usage = self._gauge_cls(
            name="lmcache:local_storage_usage",
            documentation="Local storage usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum")

        time_to_retrieve_buckets = [
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
            1.0, 2.5, 5.0, 7.5, 10.0
        ]
        self.histogram_time_to_retrieve = self._histogram_cls(
            name="lmcache:time_to_retrieve",
            documentation="Time to retrieve from lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_retrieve_buckets,
        )

        time_to_store_buckets = [
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
            1.0, 2.5, 5.0, 7.5, 10.0
        ]
        self.histogram_time_to_store = self._histogram_cls(
            name="lmcache:time_to_store",
            documentation="Time to store to lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_store_buckets,
        )

        retrieve_speed_buckets = [
            1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
            32768, 65536
        ]
        self.histogram_retrieve_speed = self._histogram_cls(
            name="lmcache:retrieve_speed",
            documentation="Retrieve speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=retrieve_speed_buckets,
        )

        store_speed_buckets = [
            1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
            32768, 65536
        ]
        self.histogram_store_speed = self._histogram_cls(
            name="lmcache:store_speed",
            documentation="Store speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=store_speed_buckets,
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        # Prevent ValueError from negative increment
        if data < 0:
            return
        counter.labels(**self.labels).inc(data)

    def _log_histogram(self, histogram, data: Union[List[int],
                                                    List[float]]) -> None:
        # Convenience function for logging to histogram.
        for value in data:
            histogram.labels(**self.labels).observe(value)

    def log_prometheus(self, stats: LMCacheStats):
        self._log_counter(self.counter_num_retrieve_requests,
                          stats.num_retrieve_requests)
        self._log_counter(self.counter_num_store_requests,
                          stats.num_store_requests)

        self._log_counter(self.counter_num_requested_tokens,
                          stats.num_requested_tokens)
        self._log_counter(self.counter_num_hit_tokens, stats.num_hit_tokens)

        self._log_gauge(self.gauge_cache_hit_rate, stats.cache_hit_rate)

        self._log_gauge(self.gauge_local_cache_usage,
                        stats.local_cache_usage_bytes)

        self._log_gauge(self.gauge_remote_cache_usage,
                        stats.remote_cache_usage_bytes)

        self._log_gauge(self.gauge_local_storage_usage,
                        stats.local_storage_usage_bytes)

        self._log_histogram(self.histogram_time_to_retrieve,
                            stats.time_to_retrieve)

        self._log_histogram(self.histogram_time_to_store, stats.time_to_store)

        self._log_histogram(self.histogram_retrieve_speed,
                            stats.retrieve_speed)

        self._log_histogram(self.histogram_store_speed, stats.store_speed)

    @staticmethod
    def _metadata_to_labels(metadata: LMCacheEngineMetadata):
        return {
            "model_name": metadata.model_name,
            "worker_id": metadata.worker_id
        }

    _instance = None

    @staticmethod
    def GetOrCreate(metadata: LMCacheEngineMetadata) -> "PrometheusLogger":
        if PrometheusLogger._instance is None:
            PrometheusLogger._instance = PrometheusLogger(metadata)
        #assert PrometheusLogger._instance.metadata == metadata, \
        #    "PrometheusLogger instance already created with different metadata"
        if PrometheusLogger._instance.metadata != metadata:
            logger.error("PrometheusLogger instance already created with"
                         "different metadata. This should not happen except "
                         "in test")
        return PrometheusLogger._instance

    @staticmethod
    def GetInstance() -> "PrometheusLogger":
        assert PrometheusLogger._instance is not None, \
            "PrometheusLogger instance not created yet"
        return PrometheusLogger._instance


class LMCacheStatsLogger:

    def __init__(self, metadata: LMCacheEngineMetadata, log_interval: int):
        self.metadata = metadata
        self.log_interval = log_interval
        self.monitor = LMCStatsMonitor.GetOrCreate()
        self.prometheus_logger = PrometheusLogger.GetOrCreate(metadata)
        self.is_running = True

        self.thread = threading.Thread(target=self.log_worker, daemon=True)
        self.thread.start()

    def log_worker(self):
        while self.is_running:
            stats = self.monitor.get_stats_and_clear()
            self.prometheus_logger.log_prometheus(stats)
            time.sleep(self.log_interval)

    def shutdown(self):
        self.is_running = False
        self.thread.join()
