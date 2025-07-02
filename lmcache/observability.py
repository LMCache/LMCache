# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from dataclasses import dataclass
from typing import Dict, List, Union
import os
import threading
import time

# Third Party
import prometheus_client

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import thread_safe

logger = init_logger(__name__)


@dataclass
class LMCacheStats:
    # Counter (Note that these are incremental values,
    # which will accumulate over time in Counter)
    interval_retrieve_requests: int
    interval_store_requests: int
    interval_requested_tokens: int
    interval_hit_tokens: int

    interval_remote_read_requests: int
    interval_remote_read_bytes: int
    interval_remote_write_requests: int
    interval_remote_write_bytes: int

    interval_remote_time_to_get: List[float]
    interval_remote_time_to_put: List[float]
    interval_remote_time_to_get_sync: List[float]

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
        return (
            self.local_hit_tokens + self.remote_hit_tokens
        ) / self.time_to_retrieve()


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
        # Interval metrics that will be reset after each log
        # Accumulate incremental values in the Prometheus Counter
        self.interval_retrieve_requests = 0
        self.interval_store_requests = 0
        self.interval_requested_tokens = 0
        self.interval_hit_tokens = 0

        # remote backends read/write metrics
        self.interval_remote_read_requests = 0
        self.interval_remote_read_bytes = 0
        self.interval_remote_write_requests = 0
        self.interval_remote_write_bytes = 0

        # remote backends get/put cost time metrics
        self.interval_remote_time_to_get: List[float] = []
        self.interval_remote_time_to_put: List[float] = []
        # the time of get value from remote backends synchronously,
        # which includes rpc and schedule time
        self.interval_remote_time_to_get_sync: List[float] = []

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
        retrieve_stats = RetrieveRequestStats(
            num_tokens=num_tokens,
            local_hit_tokens=0,
            remote_hit_tokens=0,
            start_time=curr_time,
            end_time=0,
        )
        self.interval_requested_tokens += num_tokens
        self.interval_retrieve_requests += 1
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

    @thread_safe
    def on_store_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in on_store_finished
        """
        curr_time = time.time()
        store_stats = StoreRequestStats(
            num_tokens=num_tokens, start_time=curr_time, end_time=0
        )
        self.interval_store_requests += 1
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
    def update_interval_remote_read_metrics(self, read_bytes: int):
        self.interval_remote_read_requests += 1
        self.interval_remote_read_bytes += read_bytes

    @thread_safe
    def update_interval_remote_write_metrics(self, write_bytes: int):
        self.interval_remote_write_requests += 1
        self.interval_remote_write_bytes += write_bytes

    @thread_safe
    def update_interval_remote_time_to_get(self, get_time: float):
        self.interval_remote_time_to_get.append(get_time)

    @thread_safe
    def update_interval_remote_time_to_put(self, put_time: float):
        self.interval_remote_time_to_put.append(put_time)

    @thread_safe
    def update_interval_remote_time_to_get_sync(self, get_time_sync: float):
        self.interval_remote_time_to_get_sync.append(get_time_sync)

    def _clear(self):
        """
        Clear all the distribution stats
        """
        self.interval_retrieve_requests = 0
        self.interval_store_requests = 0

        self.interval_requested_tokens = 0
        self.interval_hit_tokens = 0

        self.interval_remote_read_requests = 0
        self.interval_remote_read_bytes = 0
        self.interval_remote_write_requests = 0
        self.interval_remote_write_bytes = 0

        self.interval_remote_time_to_get.clear()
        self.interval_remote_time_to_put.clear()
        self.interval_remote_time_to_get_sync.clear()

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
        cache_hit_rate = (
            0
            if self.interval_requested_tokens == 0
            else self.interval_hit_tokens / self.interval_requested_tokens
        )

        def filter_out_invalid(stats: List[float]):
            return [x for x in stats if x != 0]

        time_to_retrieve = filter_out_invalid(
            [stats.time_to_retrieve() for stats in self.retrieve_requests.values()]
        )

        time_to_store = filter_out_invalid(
            [stats.time_to_store() for stats in self.store_requests.values()]
        )

        retrieve_speed = filter_out_invalid(
            [stats.retrieve_speed() for stats in self.retrieve_requests.values()]
        )

        store_speed = filter_out_invalid(
            [stats.store_speed() for stats in self.store_requests.values()]
        )

        ret = LMCacheStats(
            interval_retrieve_requests=self.interval_retrieve_requests,
            interval_store_requests=self.interval_store_requests,
            interval_requested_tokens=self.interval_requested_tokens,
            interval_hit_tokens=self.interval_hit_tokens,
            interval_remote_read_requests=self.interval_remote_read_requests,
            interval_remote_read_bytes=self.interval_remote_read_bytes,
            interval_remote_write_requests=self.interval_remote_write_requests,
            interval_remote_write_bytes=self.interval_remote_write_bytes,
            interval_remote_time_to_get=self.interval_remote_time_to_get.copy(),
            interval_remote_time_to_put=self.interval_remote_time_to_put.copy(),
            interval_remote_time_to_get_sync=self.interval_remote_time_to_get_sync.copy(),
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
        # Ensure PROMETHEUS_MULTIPROC_DIR is set before any metric registration
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            default_dir = "/tmp/lmcache_prometheus"
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = default_dir
            if not os.path.exists(default_dir):
                os.makedirs(default_dir, exist_ok=True)

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

        self.counter_num_remote_read_requests = self._counter_cls(
            name="lmcache:num_remote_read_requests",
            documentation="Total number of requests read from "
            "remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_read_bytes = self._counter_cls(
            name="lmcache:num_remote_read_bytes",
            documentation="Total number of bytes read from remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_write_requests = self._counter_cls(
            name="lmcache:num_remote_write_requests",
            documentation="Total number of requests write to "
            "remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_write_bytes = self._counter_cls(
            name="lmcache:num_remote_write_bytes",
            documentation="Total number of bytes write to remote backends in lmcache",
            labelnames=labelnames,
        )

        self.gauge_cache_hit_rate = self._gauge_cls(
            name="lmcache:cache_hit_rate",
            documentation="Cache hit rate of lmcache since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        self.gauge_local_cache_usage = self._gauge_cls(
            name="lmcache:local_cache_usage",
            documentation="Local cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_remote_cache_usage = self._gauge_cls(
            name="lmcache:remote_cache_usage",
            documentation="Remote cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_local_storage_usage = self._gauge_cls(
            name="lmcache:local_storage_usage",
            documentation="Local storage usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        time_to_retrieve_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_retrieve = self._histogram_cls(
            name="lmcache:time_to_retrieve",
            documentation="Time to retrieve from lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_retrieve_buckets,
        )

        time_to_store_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_store = self._histogram_cls(
            name="lmcache:time_to_store",
            documentation="Time to store to lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_store_buckets,
        )

        retrieve_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_retrieve_speed = self._histogram_cls(
            name="lmcache:retrieve_speed",
            documentation="Retrieve speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=retrieve_speed_buckets,
        )

        store_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_store_speed = self._histogram_cls(
            name="lmcache:store_speed",
            documentation="Store speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=store_speed_buckets,
        )

        remote_time_to_get = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_get = self._histogram_cls(
            name="lmcache:remote_time_to_get",
            documentation="Time to get from remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get,
        )

        remote_time_to_put = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_put = self._histogram_cls(
            name="lmcache:remote_time_to_put",
            documentation="Time to put to remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_put,
        )

        remote_time_to_get_sync = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_get_sync = self._histogram_cls(
            name="lmcache:remote_time_to_get_sync",
            documentation="Time to get from remote backends synchronously(ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get_sync,
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

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging to histogram.
        for value in data:
            histogram.labels(**self.labels).observe(value)

    def log_prometheus(self, stats: LMCacheStats):
        self._log_counter(
            self.counter_num_retrieve_requests, stats.interval_retrieve_requests
        )
        self._log_counter(
            self.counter_num_store_requests, stats.interval_store_requests
        )

        self._log_counter(
            self.counter_num_requested_tokens, stats.interval_requested_tokens
        )
        self._log_counter(self.counter_num_hit_tokens, stats.interval_hit_tokens)

        self._log_counter(
            self.counter_num_remote_read_requests,
            stats.interval_remote_read_requests,
        )
        self._log_counter(
            self.counter_num_remote_read_bytes, stats.interval_remote_read_bytes
        )
        self._log_counter(
            self.counter_num_remote_write_requests,
            stats.interval_remote_write_requests,
        )
        self._log_counter(
            self.counter_num_remote_write_bytes,
            stats.interval_remote_write_bytes,
        )

        self._log_gauge(self.gauge_cache_hit_rate, stats.cache_hit_rate)

        self._log_gauge(self.gauge_local_cache_usage, stats.local_cache_usage_bytes)

        self._log_gauge(self.gauge_remote_cache_usage, stats.remote_cache_usage_bytes)

        self._log_gauge(self.gauge_local_storage_usage, stats.local_storage_usage_bytes)

        self._log_histogram(self.histogram_time_to_retrieve, stats.time_to_retrieve)

        self._log_histogram(self.histogram_time_to_store, stats.time_to_store)

        self._log_histogram(self.histogram_retrieve_speed, stats.retrieve_speed)

        self._log_histogram(self.histogram_store_speed, stats.store_speed)

        self._log_histogram(
            self.histogram_remote_time_to_get, stats.interval_remote_time_to_get
        )
        self._log_histogram(
            self.histogram_remote_time_to_put, stats.interval_remote_time_to_put
        )
        self._log_histogram(
            self.histogram_remote_time_to_get_sync,
            stats.interval_remote_time_to_get_sync,
        )

    @staticmethod
    def _metadata_to_labels(metadata: LMCacheEngineMetadata):
        return {
            "model_name": metadata.model_name,
            "worker_id": metadata.worker_id,
        }

    _instance = None

    @staticmethod
    def GetOrCreate(metadata: LMCacheEngineMetadata) -> "PrometheusLogger":
        if PrometheusLogger._instance is None:
            PrometheusLogger._instance = PrometheusLogger(metadata)
        # assert PrometheusLogger._instance.metadata == metadata, \
        #    "PrometheusLogger instance already created with different metadata"
        if PrometheusLogger._instance.metadata != metadata:
            logger.error(
                "PrometheusLogger instance already created with"
                "different metadata. This should not happen except "
                "in test"
            )
        return PrometheusLogger._instance

    @staticmethod
    def GetInstance() -> "PrometheusLogger":
        assert PrometheusLogger._instance is not None, (
            "PrometheusLogger instance not created yet"
        )
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
