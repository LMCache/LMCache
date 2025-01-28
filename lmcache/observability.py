from typing import List, Dict, Tuple
import time
from dataclasses import dataclass

from lmcache.config import LMCacheEngineMetadata

@dataclass
class LMCacheStats:
    # Counter
    num_retrieve_requests: int
    num_store_requests: int

    # Real time value measurements
    total_cache_hit_rate: float
    local_cache_hit_rate: float 
    remote_cache_hit_rate: float

    local_cache_usage_bytes: int    # Size of the used local cache in bytes
    remote_cache_usage_bytes: int   # Size of the used remote cache in bytes

    # Distribution measurements
    time_to_retrieve: List[float]
    time_to_store: List[float]
    retrieve_speed: List[float] # Tokens per second
    store_speed: List[float]    # Tokens per second


@dataclass
class RetrieveRequestStats:
    num_tokens: int
    local_hit_tokens: int
    remote_hit_tokens: int
    start_time: float
    end_time: float

    def time_to_retrieve(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def retrieve_speed(self):
        if self.time_to_retrieve() == 0:
            return 0
        return self.num_tokens / self.time_to_retrieve()

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
        self.num_retrieve_requests = 0
        self.num_store_requests = 0

        self.total_retrieve_tokens = 0
        self.total_local_hit_tokens = 0
        self.total_remote_hit_tokens = 0

        self.local_cache_usage_bytes = 0
        self.remote_cache_usage_bytes = 0
        
        self.retrieve_requests: Dict[int, RetrieveRequestStats] = {}
        self.store_requests: Dict[int, StoreRequestStats] = {}

        self.retrieve_request_id = 0
        self.store_request_id = 0

    def on_retrieve_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in on_retrieve_finished
        """
        curr_time = time.time()
        retrieve_stats = RetrieveRequestStats(
            num_tokens=num_tokens,
            local_hit_tokens=0,
            remote_hit_tokens=0,
            start_time=curr_time,
            end_time=0
        )
        self.total_retrieve_tokens += num_tokens
        self.num_retrieve_requests += 1
        self.retrieve_requests[self.retrieve_request_id] = retrieve_stats
        self.retrieve_request_id += 1
        return self.retrieve_request_id - 1

    def on_retrieve_finished(self, request_id: int, 
                             local_retrieved_tokens: int, 
                             remote_retrieved_tokens: int):
        curr_time = time.time()
        assert request_id in self.retrieve_requests
        retrieve_stats = self.retrieve_requests[request_id]
        retrieve_stats.local_hit_tokens = local_retrieved_tokens
        retrieve_stats.remote_hit_tokens = remote_retrieved_tokens
        self.total_local_hit_tokens = local_retrieved_tokens
        self.total_remote_hit_tokens = remote_retrieved_tokens
        retrieve_stats.end_time = curr_time

    def on_store_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in on_store_finished
        """
        curr_time = time.time()
        store_stats = StoreRequestStats(
            num_tokens=num_tokens,
            start_time=curr_time,
            end_time=0
        )
        self.num_store_requests += 1
        self.store_requests[self.store_request_id] = store_stats
        self.store_request_id += 1
        return self.store_request_id - 1

    def on_store_finished(self, request_id: int):
        curr_time = time.time()
        assert request_id in self.store_requests
        store_stats = self.store_requests[request_id]
        store_stats.end_time = curr_time

    def update_local_cache_usage(self, usage: int):
        self.local_cache_usage_bytes = usage

    def update_remote_cache_usage(self, usage: int):
        self.remote_cache_usage_bytes = usage

    def _clear(self):
        """
        Clear all the distribution stats 
        """
        self.total_retrieve_tokens = 0
        self.total_local_hit_tokens = 0
        self.total_remote_hit_tokens = 0

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

    def get_stats_and_clear(self) -> LMCacheStats:
        """
        This function should be called with by prometheus adapter with 
        a specific interval.
        The function will return the latest states between the current 
        call and the previous call.
        """
        local_cache_hit_rate = 0 if self.total_retrieve_tokens == 0 else \
                self.total_local_hit_tokens / self.total_retrieve_tokens
        remote_cache_hit_rate = 0 if self.total_retrieve_tokens == 0 else \
                self.total_remote_hit_tokens / self.total_retrieve_tokens
        total_cache_hit_rate = local_cache_hit_rate + remote_cache_hit_rate

        def filter_out_invalid(stats: List[float]):
            return [x for x in stats if x != 0]

        time_to_retrieve = filter_out_invalid(
                [stats.time_to_retrieve() 
                 for stats in self.retrieve_requests.values()])

        time_to_store = filter_out_invalid(
                [stats.time_to_store() 
                 for stats in self.store_requests.values()])

        retrieve_speed = filter_out_invalid(
                [stats.retrieve_speed() 
                 for stats in self.retrieve_requests.values()])

        store_speed = filter_out_invalid(
                [stats.store_speed() 
                 for stats in self.store_requests.values()])

        ret = LMCacheStats(
            num_retrieve_requests=self.num_retrieve_requests,
            num_store_requests=self.num_store_requests,
            total_cache_hit_rate=total_cache_hit_rate,
            local_cache_hit_rate=local_cache_hit_rate,
            remote_cache_hit_rate=remote_cache_hit_rate,
            local_cache_usage_bytes=self.local_cache_usage_bytes,
            remote_cache_usage_bytes=self.remote_cache_usage_bytes,
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
    def DestoryInstane():
        LMCStatsMonitor._instance = None
