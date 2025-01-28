import time
import pytest
from lmcache.observability import LMCStatsMonitor, LMCacheStats

@pytest.fixture(scope="function")
def stats_monitor():
    yield LMCStatsMonitor.GetOrCreate()
    LMCStatsMonitor.DestoryInstane()

def test_on_retrieve_request(stats_monitor):
    stats_monitor.on_retrieve_request(num_tokens=100)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.num_retrieve_requests == 1
    assert stats.total_cache_hit_rate == 0
    assert stats.local_cache_hit_rate == 0
    assert stats.remote_cache_hit_rate == 0
    assert stats.local_cache_usage_bytes == 0
    assert stats.remote_cache_usage_bytes == 0
    assert len(stats.time_to_retrieve) == 0

def test_on_retrieve_finished(stats_monitor):
    request_id = stats_monitor.on_retrieve_request(num_tokens=100)
    stats_monitor.on_retrieve_finished(
        request_id=request_id,
        local_retrieved_tokens=80,
        remote_retrieved_tokens=20
    )
    stats = stats_monitor.get_stats_and_clear()
    assert stats.num_retrieve_requests == 1
    assert stats.local_cache_hit_rate == 0.8
    assert stats.remote_cache_hit_rate == 0.2
    assert stats.total_cache_hit_rate == 1.0
    assert len(stats.time_to_retrieve) == 1

def test_on_store_request_and_finished(stats_monitor):
    request_id = stats_monitor.on_store_request(num_tokens=50)
    stats_monitor.on_store_finished(request_id=request_id)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.num_store_requests == 1
    assert len(stats.time_to_store) == 1

def test_update_local_cache_usage(stats_monitor):
    stats_monitor.update_local_cache_usage(usage=1024)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.local_cache_usage_bytes == 1024

def test_update_remote_cache_usage(stats_monitor):
    stats_monitor.update_remote_cache_usage(usage=2048)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.remote_cache_usage_bytes == 2048

def test_combined_operations(stats_monitor):
    retrieve_id = stats_monitor.on_retrieve_request(num_tokens=200)
    stats_monitor.on_retrieve_finished(
        request_id=retrieve_id,
        local_retrieved_tokens=150,
        remote_retrieved_tokens=50
    )
    store_id = stats_monitor.on_store_request(num_tokens=100)
    stats_monitor.on_store_finished(store_id)
    stats_monitor.update_local_cache_usage(usage=512)
    stats_monitor.update_remote_cache_usage(usage=1024)

    stats_monitor2 = LMCStatsMonitor.GetOrCreate()
    stats = stats_monitor2.get_stats_and_clear()

    assert stats.num_retrieve_requests == 1
    assert stats.num_store_requests == 1
    assert stats.local_cache_hit_rate == 0.75
    assert stats.remote_cache_hit_rate == 0.25
    assert stats.total_cache_hit_rate == 1.0
    assert stats.local_cache_usage_bytes == 512
    assert stats.remote_cache_usage_bytes == 1024
    assert len(stats.time_to_retrieve) == 1
    assert len(stats.time_to_store) == 1

