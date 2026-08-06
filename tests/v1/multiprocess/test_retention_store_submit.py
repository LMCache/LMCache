# SPDX-License-Identifier: Apache-2.0
"""Retention params must flow through the worker adapter's store submits."""

# Standard
from types import SimpleNamespace
from typing import cast

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    ParallelStrategy,
)


def _bare_worker_adapter() -> LMCacheMPWorkerAdapter:
    """Adapter with only the attrs _create_key needs; no ZMQ or heartbeat."""
    adapter = object.__new__(LMCacheMPWorkerAdapter)
    adapter.model_name = "test_model"
    adapter.parallel_strategy = cast(
        ParallelStrategy, SimpleNamespace(kv_world_size=1, kv_worker_id=0)
    )
    return adapter


def test_worker_create_key_carries_retention():
    key = _bare_worker_adapter()._create_key(
        [1, 2, 3],
        0,
        3,
        request_id="r",
        cache_salt="s",
        retention_ttl_sec=300,
    )
    assert key.retention_ttl_sec == 300
    assert key.cache_salt == "s"


def test_worker_create_key_defaults_to_no_retention():
    key = _bare_worker_adapter()._create_key([1, 2, 3], 0, 3, request_id="r")
    assert key.retention_ttl_sec == 0


def test_batched_store_fans_out_per_request_retention():
    adapter = _bare_worker_adapter()
    calls = []

    def fake_submit(request_id, op, event, cache_salt="", retention_ttl_sec=0):
        calls.append((request_id, cache_salt, retention_ttl_sec))

    adapter.submit_store_request = fake_submit

    adapter.batched_submit_store_requests(
        ["a", "b"],
        [object(), object()],
        object(),
        cache_salts=["", "s"],
        retention_ttl_secs=[0, 3600],
    )
    assert calls == [("a", "", 0), ("b", "s", 3600)]


def test_batched_store_defaults_mean_no_retention():
    """Callers that predate retention must behave exactly as before."""
    adapter = _bare_worker_adapter()
    calls = []

    def fake_submit(request_id, op, event, cache_salt="", retention_ttl_sec=0):
        calls.append((request_id, cache_salt, retention_ttl_sec))

    adapter.submit_store_request = fake_submit

    adapter.batched_submit_store_requests(
        ["a", "b"],
        [object(), object()],
        object(),
        cache_salts=["", "s"],
    )
    assert calls == [("a", "", 0), ("b", "s", 0)]
