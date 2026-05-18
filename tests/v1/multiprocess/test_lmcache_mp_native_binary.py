# SPDX-License-Identifier: Apache-2.0
"""Tests for the experimental native C++ MP server binary."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess import native_launcher as native_launcher_module
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CudaIPCWrapper,
    IPCCacheEngineKey,
    RawCudaIPCWrapper,
)
from lmcache.v1.multiprocess.mq import (
    MessageQueueClient,
    msgspec_decode,
    msgspec_encode,
)
from lmcache.v1.multiprocess.native_launcher import (
    ensure_native_binary,
    native_argv_from_args,
)
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str) -> None:
    deadline = time.time() + 10
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5).read()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.1)
    raise AssertionError(f"server did not become ready at {url}: {last_error}")


def _terminate_proc(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _fs_l2_filename(key: ObjectKey) -> str:
    safe_model = key.model_name.replace("/", "-SEP-")
    base = f"{safe_model}@{key.kv_rank:#010x}@{key.chunk_hash.hex()}"
    if key.cache_salt:
        return f"{base}@{key.cache_salt}.data"
    return f"{base}.data"


def _latency_histogram_count(status: dict[str, object]) -> int:
    metrics = status["metrics"]
    assert isinstance(metrics, dict)
    histogram = metrics["request_latency_histogram"]
    assert isinstance(histogram, dict)
    return sum(int(value) for value in histogram.values())


def _assert_native_http_not_implemented(url: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(url, timeout=5).read()
    assert exc_info.value.code == 501
    body = json.loads(exc_info.value.read())
    assert body["error"] == "native MP HTTP endpoint not implemented"
    assert body["endpoint"]
    assert body["detail"]


def _native_request_timeout(seconds: float) -> float:
    if os.environ.get("LMCACHE_RUN_CUDA_TSAN_STRESS") == "1":
        return seconds * 6
    return seconds


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    current = 1
    for size in reversed(shape):
        stride.append(current)
        current *= size
    return tuple(reversed(stride))


def _fake_cuda_ipc_wrapper(
    shape: tuple[int, ...] = (2, 16, 16, 1, 8),
) -> CudaIPCWrapper:
    wrapper = CudaIPCWrapper.__new__(CudaIPCWrapper)
    wrapper.handle = (0, b"cuda-handle", 4096, 0, b"ref", 0, b"event", False)
    wrapper.dtype = torch.float16
    wrapper.shape = shape
    wrapper.stride = _contiguous_stride(shape)
    wrapper.storage_offset = 0
    wrapper.device_uuid = "GPU-fake-uuid"
    return wrapper


def _cuda_python_runtime_available() -> bool:
    try:
        # Third Party
        from cuda.bindings import runtime as _runtime  # noqa: F401
    except ImportError:
        try:
            # Third Party
            from cuda import cudart as _cudart  # noqa: F401
        except ImportError:
            return False
    return True


def _cupy_available() -> bool:
    try:
        # Third Party
        import cupy as _cupy  # noqa: F401
    except ImportError:
        return False
    return True


def test_native_binary_speaks_controller_protocol_and_http(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    env = {
        **os.environ,
        "LMCACHE_NATIVE_COMMIT_ID": "native-test-commit",
        "LMCACHE_NATIVE_VERSION": "native-test-version",
    }
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--eviction-policy",
            "LRU",
            "--chunk-size",
            "128",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
            "--l2-adapter",
            json.dumps({"type": "fs", "base_path": str(tmp_path / "l2")}),
        ],
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")

        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert client.submit_request(RequestType.PING, []).result(timeout=5) is True
            assert (
                client.submit_request(RequestType.GET_CHUNK_SIZE, []).result(timeout=5)
                == 128
            )
            assert client.submit_request(RequestType.NOOP, []).result(timeout=5) == "OK"
            assert (
                client.submit_request(RequestType.CLEAR, []).result(timeout=5) is None
            )
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        999,
                        [],
                        "facebook/opt-125m",
                        2,
                        EngineType.VLLM,
                        {"kv_layout": "BAD"},
                    ],
                ).result(timeout=5)
                is None
            )
            status_after_bad_register = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_bad_register["registered_context_count"] == 0
            assert status_after_bad_register["metrics"]["active_client_count"] == 1
            assert status_after_bad_register["metrics"]["observed_client_count"] == 1

            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        123,
                        [],
                        "facebook/opt-125m",
                        2,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )
            status_after_register = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_register["registered_context_count"] == 1
            assert status_after_register["registered_contexts"] == [
                {
                    "engine_type": "vllm",
                    "first_kv_device_uuid": "",
                    "first_kv_dtype": "",
                    "first_kv_shape": [],
                    "first_kv_stride": [],
                    "first_kv_num_blocks": 0,
                    "first_kv_block_size": 0,
                    "first_kv_ipc_handle_bytes": 0,
                    "first_kv_storage_bytes": 0,
                    "first_kv_storage_offset_bytes": 0,
                    "first_kv_event_handle_bytes": 0,
                    "first_kv_event_sync_required": False,
                    "inference_engine_logical_block_size": 16,
                    "instance_id": 123,
                    "kv_cache_wrapper_count": 0,
                    "kv_layout": "NHD",
                    "model_name": "facebook/opt-125m",
                    "trt_llm_head_dim": 0,
                    "trt_llm_layout_hints": False,
                    "trt_llm_num_kv_heads": 0,
                    "trt_llm_tokens_per_block": 0,
                    "use_layerwise": False,
                    "world_size": 2,
                }
            ]
            with pytest.raises(urllib.error.HTTPError) as missing_kvcache_blocks:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/kvcache/check?instance_id=123",
                    timeout=5,
                )
            assert missing_kvcache_blocks.value.code == 400
            assert json.loads(missing_kvcache_blocks.value.read()) == {
                "error": "block_ids is required"
            }
            with pytest.raises(urllib.error.HTTPError) as invalid_kvcache_blocks:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/kvcache/check"
                    "?instance_id=123&block_ids=bad&chunk_size=1",
                    timeout=5,
                )
            assert invalid_kvcache_blocks.value.code == 400
            assert json.loads(invalid_kvcache_blocks.value.read()) == {
                "error": "Invalid block_ids format"
            }
            with pytest.raises(urllib.error.HTTPError) as invalid_kvcache_chunk:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/kvcache/check"
                    "?instance_id=123&block_ids=0,[2,3]&chunk_size=0",
                    timeout=5,
                )
            assert invalid_kvcache_chunk.value.code == 400
            assert json.loads(invalid_kvcache_chunk.value.read()) == {
                "error": "chunk_size must be positive"
            }
            with pytest.raises(urllib.error.HTTPError) as missing_kvcache_instance:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/kvcache/check"
                    "?instance_id=999&block_ids=0&chunk_size=1",
                    timeout=5,
                )
            assert missing_kvcache_instance.value.code == 404
            assert json.loads(missing_kvcache_instance.value.read()) == {
                "error": "instance_id 999 not registered"
            }
            with pytest.raises(urllib.error.HTTPError) as empty_kvcache:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/kvcache/check"
                    "?instance_id=123&block_ids=0,[2,3]&chunk_size=1",
                    timeout=5,
                )
            assert empty_kvcache.value.code == 404
            assert json.loads(empty_kvcache.value.read()) == {
                "error": "kv_caches empty"
            }

            lookup_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=2,
                worker_id=None,
                token_ids=list(range(256)),
                request_id="native-lookup-1",
                cache_salt="tenant-a",
            )
            assert (
                client.submit_request(RequestType.LOOKUP, [lookup_key, 2]).result(
                    timeout=5
                )
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_LOOKUP_HITS,
                    ["native-lookup-1"],
                ).result(timeout=5)
                == 0
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    ["native-lookup-1"],
                ).result(timeout=5)
                == 0
            )
            worker_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=2,
                worker_id=0,
                token_ids=list(range(256)),
                start=0,
                end=256,
                request_id="native-store-1",
                cache_salt="tenant-a",
            )
            assert client.submit_request(
                RequestType.STORE,
                [worker_key, 123, [0], b""],
            ).result(timeout=5) == (b"", False)
            assert client.submit_request(
                RequestType.RETRIEVE,
                [worker_key, 123, [0], b"", 0],
            ).result(timeout=5) == (b"", False)
            assert (
                client.submit_request(
                    RequestType.FREE_LOOKUP_LOCKS,
                    [worker_key, 2],
                ).result(timeout=5)
                is None
            )
            assert (
                client.submit_request(
                    RequestType.UNREGISTER_KV_CACHE,
                    [123],
                ).result(timeout=5)
                is None
            )
            assert (
                client.submit_request(
                    RequestType.REPORT_BLOCK_ALLOCATION,
                    [
                        123,
                        "facebook/opt-125m",
                        [
                            BlockAllocationRecord(
                                req_id="native-report-1",
                                new_block_ids=[1, 2],
                                new_token_ids=[11, 12, 13],
                            )
                        ],
                    ],
                ).result(timeout=5)
                is None
            )
        finally:
            client.close()

        health = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/healthcheck",
            timeout=5,
        ).read()
        assert json.loads(health) == {"status": "healthy"}
        root = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/",
            timeout=5,
        ).read()
        assert json.loads(root) == {"status": "ok", "service": "LMCache HTTP API"}
        conf = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/conf",
                timeout=5,
            ).read()
        )
        assert conf["native"] is True
        assert conf["version"] == {
            "commit_id": "native-test-commit",
            "lmcache_version": "native-test-version",
        }
        assert conf["mp"] == {
            "chunk_size": 128,
            "cuda_gpu_hot_cache_enabled": False,
            "eviction_policy": "LRU",
            "host": "127.0.0.1",
            "log_level": "",
            "max_queued_tasks": 1024,
            "max_workers": 1,
            "port": zmq_port,
        }
        assert conf["http"] == {
            "enabled": True,
            "http_host": "127.0.0.1",
            "http_port": http_port,
        }
        assert conf["storage_manager"]["disk_path"] == str(tmp_path / "disk")
        assert conf["storage_manager"]["dram_capacity_bytes"] > 0
        assert conf["storage_manager"]["l2_adapters"] == [
            {"type": "fs", "base_path": str(tmp_path / "l2")}
        ]
        assert conf["observability"] == {
            "metrics_endpoint": "/metrics",
            "metrics_reset_endpoint": "/metrics/reset",
            "status_endpoint": "/status",
        }
        assert (
            json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/lmc_version",
                    timeout=5,
                ).read()
            )
            == "native-test-version"
        )
        assert (
            json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/commit_id",
                    timeout=5,
                ).read()
            )
            == "native-test-commit"
        )
        assert (
            json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/version",
                    timeout=5,
                ).read()
            )
            == "native-test-version-native-test-commit"
        )
        env_response = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/env",
            timeout=5,
        )
        assert env_response.headers.get_content_type() == "text/plain"
        env_payload = json.loads(env_response.read())
        assert env_payload["LMCACHE_NATIVE_VERSION"] == "native-test-version"
        assert env_payload["LMCACHE_NATIVE_COMMIT_ID"] == "native-test-commit"
        loglevel_response = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/loglevel?logger_name=lmcache",
            timeout=5,
        )
        assert loglevel_response.headers.get_content_type() == "text/plain"
        assert loglevel_response.read().decode() == "lmcache: NOTSET"
        set_loglevel_response = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/loglevel?logger_name=lmcache&level=debug",
            timeout=5,
        )
        assert set_loglevel_response.read().decode() == (
            "Set lmcache level to DEBUG (including all handlers)"
        )
        assert (
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/loglevel?logger_name=lmcache",
                timeout=5,
            )
            .read()
            .decode()
            == "lmcache: DEBUG"
        )
        list_loglevel_response = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/loglevel",
            timeout=5,
        )
        assert "lmcache: DEBUG" in list_loglevel_response.read().decode()
        with pytest.raises(urllib.error.HTTPError) as invalid_loglevel:
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/loglevel"
                "?logger_name=lmcache&level=invalid",
                timeout=5,
            )
        assert invalid_loglevel.value.code == 400
        assert invalid_loglevel.value.read().decode() == ("Invalid log level: invalid")
        threads_response = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/threads?name=native-worker",
            timeout=5,
        )
        assert threads_response.headers.get_content_type() == "text/plain"
        threads_text = threads_response.read().decode()
        assert "Thread: native-worker-0" in threads_text
        assert "Total threads: 1" in threads_text
        periodic_threads_response = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/threads?name=periodic",
            timeout=5,
        )
        assert "Total threads: 0" in periodic_threads_response.read().decode()
        periodic_registry = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/periodic-threads",
                timeout=5,
            ).read()
        )
        assert periodic_registry == {
            "summary": {
                "total_count": 0,
                "running_count": 0,
                "active_count": 0,
                "by_level": {
                    "critical": {"total": 0, "running": 0, "active": 0},
                    "high": {"total": 0, "running": 0, "active": 0},
                    "medium": {"total": 0, "running": 0, "active": 0},
                    "low": {"total": 0, "running": 0, "active": 0},
                },
            },
            "threads": [],
        }
        periodic_health = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/periodic-threads-health",
                timeout=5,
            ).read()
        )
        assert periodic_health == {
            "healthy": True,
            "unhealthy_count": 0,
            "unhealthy_threads": [],
        }
        with pytest.raises(urllib.error.HTTPError) as missing_periodic_thread:
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/periodic-threads/storage-flush",
                timeout=5,
            )
        assert missing_periodic_thread.value.code == 404
        assert json.loads(missing_periodic_thread.value.read()) == {
            "error": "Thread not found: storage-flush"
        }
        with pytest.raises(urllib.error.HTTPError) as invalid_periodic_level:
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/periodic-threads?level=bad",
                timeout=5,
            )
        assert invalid_periodic_level.value.code == 400
        assert json.loads(invalid_periodic_level.value.read()) == {
            "error": "Invalid level: bad. Valid values: critical, high, medium, low"
        }
        assert json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/quota",
                timeout=5,
            ).read()
        ) == {"users": {}}
        assert json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/quota/_default",
                timeout=5,
            ).read()
        ) == {
            "cache_salt": "_default",
            "limit_gb": 0,
            "current_usage_gb": 0,
            "exists": False,
        }
        set_quota_request = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/quota/_default",
            data=b'{"limit_gb": 0.25}',
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        assert json.loads(
            urllib.request.urlopen(set_quota_request, timeout=5).read()
        ) == {
            "cache_salt": "_default",
            "limit_gb": 0.25,
            "status": "ok",
        }
        assert json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/quota",
                timeout=5,
            ).read()
        ) == {"users": {"_default": {"limit_gb": 0.25, "current_usage_gb": 0}}}
        tenant_payload = b"tenant-quota-usage"
        tenant_key = ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(12345),
            model_name="facebook/opt-125m",
            kv_rank=1,
            cache_salt="tenant-a",
        )
        tenant_l2_path = tmp_path / "l2" / _fs_l2_filename(tenant_key)
        tenant_l2_path.parent.mkdir(parents=True, exist_ok=True)
        tenant_l2_path.write_bytes(tenant_payload)
        set_tenant_quota_request = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/quota/tenant-a",
            data=b'{"limit_gb": 0.5}',
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        assert json.loads(
            urllib.request.urlopen(set_tenant_quota_request, timeout=5).read()
        ) == {
            "cache_salt": "tenant-a",
            "limit_gb": 0.5,
            "status": "ok",
        }
        tenant_quota = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/quota/tenant-a",
                timeout=5,
            ).read()
        )
        assert tenant_quota["cache_salt"] == "tenant-a"
        assert tenant_quota["limit_gb"] == 0.5
        assert tenant_quota["exists"] is True
        assert math.isclose(
            tenant_quota["current_usage_gb"],
            len(tenant_payload) / (1024**3),
        )
        with pytest.raises(urllib.error.HTTPError) as invalid_quota:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{http_port}/quota/_default",
                    data=b'{"limit_gb": -1}',
                    headers={"Content-Type": "application/json"},
                    method="PUT",
                ),
                timeout=5,
            )
        assert invalid_quota.value.code == 400
        assert json.loads(invalid_quota.value.read()) == {
            "error": "limit_gb must be non-negative"
        }
        delete_quota_request = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/quota/_default",
            method="DELETE",
        )
        assert json.loads(
            urllib.request.urlopen(delete_quota_request, timeout=5).read()
        ) == {
            "cache_salt": "_default",
            "status": "removed",
        }

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["native"] is True
        assert status["chunk_size"] == 128
        assert status["registered_context_count"] == 0
        assert status["num_l2_adapters"] == 1
        assert status["l2_adapters"][0]["type"] == "fs"
        assert status["l2_adapters"][0]["base_path"] == str(tmp_path / "l2")
        assert status["cache"]["dram_capacity_bytes"] > 0
        assert status["cache"]["disk_bytes"] == 0
        assert status["cache"]["lock_count"] == 0
        assert status["cache"]["locked_bytes"] == 0
        assert status["cache"]["eviction_count"] == 0
        assert status["metrics"]["request_count"] >= 13
        assert status["metrics"]["lookup_count"] >= 1
        assert status["metrics"]["lookup_result_fast_path_count"] >= 2
        assert status["metrics"]["store_count"] == 1
        assert status["metrics"]["retrieve_count"] == 1
        assert status["metrics"]["block_allocation_report_count"] == 1
        assert status["metrics"]["block_allocation_record_count"] == 1
        assert status["metrics"]["cache_misses"] == 4
        assert status["metrics"]["cache_hit_rate"] == 0.0
        assert status["metrics"]["worker_count"] == 1
        assert status["metrics"]["active_client_count"] == 1
        assert status["metrics"]["observed_client_count"] == 1
        assert status["metrics"]["active_worker_count"] == 0
        assert status["metrics"]["worker_queue_depth"] == 0
        assert status["metrics"]["max_worker_queue_depth"] == 1024
        assert status["metrics"]["response_queue_depth"] == 0
        assert (
            status["metrics"]["request_latency_count"]
            == status["metrics"]["request_count"]
        )
        assert status["metrics"]["request_latency_total_us"] >= 0
        assert status["metrics"]["request_latency_max_us"] >= 0
        assert (
            status["metrics"]["request_queue_wait_count"]
            == status["metrics"]["request_count"]
        )
        assert status["metrics"]["request_queue_wait_total_us"] >= 0
        assert status["metrics"]["request_queue_wait_max_us"] >= 0
        type_latency = status["metrics"]["request_type_latency"]
        type_queue_wait = status["metrics"]["request_type_queue_wait"]
        assert type_latency["lookup"]["count"] == 1
        assert type_latency["store"]["count"] == 1
        assert type_latency["retrieve"]["count"] == 1
        assert type_latency["free_lookup_locks"]["count"] == 1
        assert set(type_queue_wait) == set(type_latency)
        for name, request_type in type_latency.items():
            assert request_type["max_us"] <= request_type["total_us"]
            assert type_queue_wait[name]["count"] == request_type["count"]
            assert type_queue_wait[name]["max_us"] <= type_queue_wait[name][
                "total_us"
            ]
        assert (
            _latency_histogram_count(status)
            == status["metrics"]["request_latency_count"]
        )
        assert status["last_block_allocation"] == {
            "instance_id": 123,
            "last_new_block_count": 2,
            "last_new_token_count": 3,
            "last_request_id": "native-report-1",
            "model_name": "facebook/opt-125m",
            "record_count": 1,
        }

        clear_request = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/clear-cache",
            method="POST",
        )
        assert json.loads(urllib.request.urlopen(clear_request, timeout=5).read()) == {
            "status": "ok"
        }
        metrics_response = urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/metrics",
            timeout=5,
        )
        assert metrics_response.headers.get_content_type() == "text/plain"
        metrics_text = metrics_response.read().decode()
        assert "# TYPE lmcache_mp_native_requests_total counter" in metrics_text
        assert "# TYPE lmcache_mp_native_cache_hit_rate gauge" in metrics_text
        assert "lmcache_mp_native_cache_hit_rate 0" in metrics_text
        assert (
            "# TYPE lmcache_mp_native_request_queue_wait_count counter"
            in metrics_text
        )
        assert (
            f"lmcache_mp_native_observed_clients "
            f"{status['metrics']['observed_client_count']}"
        ) in metrics_text
        assert "lmcache_mp_native_cache_dram_capacity_bytes " in metrics_text
        assert "lmcache_mp_native_lookup_result_fast_path_total " in metrics_text

        reset_request = urllib.request.Request(
            f"http://127.0.0.1:{http_port}/metrics/reset",
            method="POST",
        )
        assert urllib.request.urlopen(reset_request, timeout=5).read() == b"ok"
        status_after_metrics_reset = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status_after_metrics_reset["metrics"]["request_count"] == 0
        assert status_after_metrics_reset["metrics"]["clear_count"] == 0
        assert (
            status_after_metrics_reset["metrics"]["lookup_result_fast_path_count"]
            == 0
        )
        assert status_after_metrics_reset["metrics"]["cache_hit_rate"] == 0.0
        assert status_after_metrics_reset["metrics"]["request_latency_count"] == 0
        assert (
            status_after_metrics_reset["metrics"]["request_queue_wait_count"] == 0
        )
        for request_type in status_after_metrics_reset["metrics"][
            "request_type_latency"
        ].values():
            assert request_type == {
                "count": 0,
                "max_us": 0,
                "total_us": 0,
            }
        for request_type in status_after_metrics_reset["metrics"][
            "request_type_queue_wait"
        ].values():
            assert request_type == {
                "count": 0,
                "max_us": 0,
                "total_us": 0,
            }
        assert status_after_metrics_reset["metrics"]["observed_client_count"] == 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def test_native_binary_kvcache_check_reports_missing_instance(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        with pytest.raises(urllib.error.HTTPError) as missing_instance:
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/kvcache/check",
                timeout=5,
            )
        assert missing_instance.value.code == 404
        assert json.loads(missing_instance.value.read()) == {
            "error": "instance_id 0 not registered"
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_native_binary_decodes_registered_kv_cache_metadata(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        777,
                        [_fake_cuda_ipc_wrapper()],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["registered_contexts"] == [
            {
                "engine_type": "vllm",
                "first_kv_device_uuid": "GPU-fake-uuid",
                "first_kv_dtype": "torch.float16",
                "first_kv_shape": [2, 16, 16, 1, 8],
                "first_kv_stride": [2048, 128, 8, 8, 1],
                "first_kv_num_blocks": 16,
                "first_kv_block_size": 16,
                "first_kv_ipc_handle_bytes": 11,
                "first_kv_storage_bytes": 4096,
                "first_kv_storage_offset_bytes": 0,
                "first_kv_event_handle_bytes": 5,
                "first_kv_event_sync_required": False,
                "inference_engine_logical_block_size": 16,
                "instance_id": 777,
                "kv_cache_wrapper_count": 1,
                "kv_layout": "NHD",
                "model_name": "facebook/opt-125m",
                "trt_llm_head_dim": 0,
                "trt_llm_layout_hints": False,
                "trt_llm_num_kv_heads": 0,
                "trt_llm_tokens_per_block": 0,
                "world_size": 1,
            }
        ]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_native_binary_rejects_unsupported_kv_layout_metadata(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        unsupported_cases: list[tuple[list[CudaIPCWrapper], dict[str, object]]] = [
            (
                [],
                {
                    "kv_layout": "NHD",
                    "inference_engine_logical_block_size": 16,
                    "compress_ratio": 2,
                },
            ),
            (
                [],
                {
                    "kv_layout": "NHD",
                    "inference_engine_logical_block_size": 16,
                    "kv_layer_groups": "mixed",
                },
            ),
            (
                [],
                {
                    "kv_layout": "NHD",
                    "inference_engine_logical_block_size": 16,
                    "num_kv_heads": 1,
                },
            ),
            (
                [],
                {
                    "kv_layout": "HND",
                    "inference_engine_logical_block_size": 16,
                    "num_kv_heads": 1,
                    "tokens_per_block": 16,
                    "head_dim": 8,
                },
            ),
            (
                [_fake_cuda_ipc_wrapper(shape=(2, 16, 8, 1, 8))],
                {
                    "kv_layout": "NHD",
                    "inference_engine_logical_block_size": 16,
                    "compress_ratio": 3,
                },
            ),
            (
                [_fake_cuda_ipc_wrapper(shape=(2, 16, 8, 1, 8))],
                {
                    "kv_layout": "NHD",
                    "inference_engine_logical_block_size": 16,
                    "group_compress_ratios": [3],
                    "group_physical_block_sizes": [8],
                },
            ),
            (
                [_fake_cuda_ipc_wrapper(shape=(2, 16, 10, 1, 8))],
                {
                    "kv_layout": "NHD",
                    "inference_engine_logical_block_size": 16,
                },
            ),
        ]
        try:
            for case_index, (kv_wrappers, layout_hints) in enumerate(unsupported_cases):
                assert (
                    client.submit_request(
                        RequestType.REGISTER_KV_CACHE,
                        [
                            900 + case_index,
                            kv_wrappers,
                            "facebook/opt-125m",
                            1,
                            EngineType.VLLM,
                            layout_hints,
                        ],
                    ).result(timeout=5)
                    is None
                )
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["registered_context_count"] == 0
        assert status["metrics"]["unsupported_count"] == len(unsupported_cases)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_native_binary_validates_kv_transfer_block_metadata(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "128",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        worker_key = IPCCacheEngineKey.from_token_ids(
            model_name="facebook/opt-125m",
            world_size=1,
            worker_id=0,
            token_ids=list(range(128)),
            start=0,
            end=128,
            request_id="native-kv-validate-1",
        )
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        888,
                        [_fake_cuda_ipc_wrapper()],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )
            assert client.submit_request(
                RequestType.STORE,
                [worker_key, 888, list(range(8)), b""],
            ).result(timeout=5) == (b"", False)
            assert client.submit_request(
                RequestType.RETRIEVE,
                [worker_key, 888, list(range(8)), b"", 16],
            ).result(timeout=5) == (b"", False)
            assert client.submit_request(
                RequestType.STORE,
                [worker_key, 888, [0], b""],
            ).result(timeout=5) == (b"", False)
            assert client.submit_request(
                RequestType.RETRIEVE,
                [worker_key, 888, list(range(7)) + [99], b"", 16],
            ).result(timeout=5) == (b"", False)
            assert client.submit_request(
                RequestType.RETRIEVE,
                [worker_key, 888, list(range(8)), b"", 1],
            ).result(timeout=5) == (b"", False)
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["store_count"] == 2
        assert status["metrics"]["retrieve_count"] == 3
        assert status["metrics"]["invalid_payload_count"] == 3
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_native_binary_rejects_malformed_kv_transfer_payloads(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "128",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        raw = context.socket(zmq.DEALER)
        raw.setsockopt(zmq.LINGER, 0)
        raw.connect(f"tcp://127.0.0.1:{zmq_port}")
        try:
            malformed_requests = [
                (RequestType.STORE, []),
                (
                    RequestType.STORE,
                    ["not-an-ipc-key", "not-an-instance-id", "not-blocks", b""],
                ),
                (RequestType.RETRIEVE, []),
                (
                    RequestType.RETRIEVE,
                    ["not-an-ipc-key", "not-an-instance-id", "not-blocks", b"", "0"],
                ),
            ]
            for uid, (request_type, payloads) in enumerate(malformed_requests):
                raw.send_multipart(
                    [
                        msgspec_encode(uid, cls=int),
                        msgspec_encode(request_type, cls=RequestType),
                        *[msgspec_encode(payload, cls=object) for payload in payloads],
                    ]
                )
                assert raw.poll(timeout=5000, flags=zmq.POLLIN)
                response = raw.recv_multipart()
                assert len(response) >= 3
        finally:
            raw.close()

        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert client.submit_request(RequestType.PING, []).result(timeout=5) is True
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["store_count"] == 2
        assert status["metrics"]["retrieve_count"] == 2
        assert status["metrics"]["invalid_payload_count"] == 4
        assert status["metrics"]["unsupported_count"] == 4
    finally:
        _terminate_proc(proc)


def test_native_binary_reports_invalid_core_metadata_payloads(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "128",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        raw = context.socket(zmq.DEALER)
        raw.setsockopt(zmq.LINGER, 0)
        raw.connect(f"tcp://127.0.0.1:{zmq_port}")
        malformed_requests = [
            (RequestType.REGISTER_KV_CACHE, []),
            (RequestType.UNREGISTER_KV_CACHE, ["not-an-instance-id"]),
            (RequestType.LOOKUP, ["not-an-ipc-key", 1]),
            (RequestType.FREE_LOOKUP_LOCKS, ["not-an-ipc-key", 1]),
            (RequestType.END_SESSION, [1]),
            (RequestType.REPORT_BLOCK_ALLOCATION, [1, "model", "not-records"]),
            (RequestType.QUERY_PREFETCH_STATUS, [1]),
            (RequestType.QUERY_PREFETCH_LOOKUP_HITS, []),
        ]
        try:
            for uid, (request_type, payloads) in enumerate(malformed_requests):
                raw.send_multipart(
                    [
                        msgspec_encode(uid, cls=int),
                        msgspec_encode(request_type, cls=RequestType),
                        *[msgspec_encode(payload, cls=object) for payload in payloads],
                    ]
                )
                assert raw.poll(timeout=5000, flags=zmq.POLLIN)
                assert len(raw.recv_multipart()) >= 2
        finally:
            raw.close()

        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert client.submit_request(RequestType.PING, []).result(timeout=5) is True
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["registered_context_count"] == 0
        assert status["metrics"]["invalid_payload_count"] == len(malformed_requests)
        assert status["metrics"]["unsupported_count"] == 0
    finally:
        _terminate_proc(proc)


def test_native_binary_rejects_oversized_zmq_payload_and_stays_healthy(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "128",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        raw = context.socket(zmq.DEALER)
        raw.setsockopt(zmq.LINGER, 0)
        raw.connect(f"tcp://127.0.0.1:{zmq_port}")
        try:
            raw.send_multipart(
                [
                    msgspec_encode(0, cls=int),
                    msgspec_encode(RequestType.LOOKUP, cls=RequestType),
                    b"x" * (16 * 1024 * 1024 + 1),
                ]
            )
            assert raw.poll(timeout=5000, flags=zmq.POLLIN)
            response = raw.recv_multipart()
            assert len(response) == 3
            assert msgspec_decode(response[-1], cls=object) is None
        finally:
            raw.close()

        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert client.submit_request(RequestType.PING, []).result(timeout=5) is True
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["invalid_payload_count"] == 1
        assert status["metrics"]["unsupported_count"] == 0
    finally:
        _terminate_proc(proc)


def test_native_binary_reports_worker_queue_backpressure(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "128",
            "--max-queued-tasks",
            "0",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        raw = context.socket(zmq.DEALER)
        raw.setsockopt(zmq.LINGER, 0)
        raw.connect(f"tcp://127.0.0.1:{zmq_port}")
        try:
            raw.send_multipart(
                [
                    msgspec_encode(123, cls=int),
                    msgspec_encode(RequestType.PING, cls=RequestType),
                ]
            )
            assert raw.poll(timeout=5000, flags=zmq.POLLIN)
            response = raw.recv_multipart()
            assert len(response) == 3
            assert msgspec_decode(response[0], cls=int) == 123
            assert msgspec_decode(response[1], cls=int) == RequestType.PING.value
            assert msgspec_decode(response[2], cls=object) is None
        finally:
            raw.close()

        assert urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/healthcheck",
            timeout=5,
        ).read()
        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["max_worker_queue_depth"] == 0
        assert status["metrics"]["worker_queue_depth"] == 0
        assert status["metrics"]["queue_full_count"] == 1
        assert status["metrics"]["unsupported_count"] == 0
        assert status["metrics"]["invalid_payload_count"] == 0
    finally:
        _terminate_proc(proc)


def test_native_binary_applies_startup_log_level(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--log-level",
            "warning",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        assert (
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/loglevel?logger_name=lmcache",
                timeout=5,
            )
            .read()
            .decode()
            == "lmcache: WARNING"
        )
        conf = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/conf",
                timeout=5,
            ).read()
        )
        assert conf["mp"]["log_level"] == "WARNING"
    finally:
        _terminate_proc(proc)


def test_native_binary_rejects_out_of_range_request_type(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        raw = context.socket(zmq.DEALER)
        raw.setsockopt(zmq.LINGER, 0)
        raw.connect(f"tcp://127.0.0.1:{zmq_port}")
        try:
            raw.send_multipart(
                [
                    msgspec_encode(123, cls=int),
                    msgspec_encode(267, cls=int),
                ]
            )
            assert raw.poll(timeout=5000, flags=zmq.POLLIN)
            response = raw.recv_multipart()
            assert len(response) == 3
            assert msgspec_decode(response[0], cls=int) == 123
            assert msgspec_decode(response[1], cls=int) == 267
            assert msgspec_decode(response[2], cls=object) is None
        finally:
            raw.close()

        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert client.submit_request(RequestType.PING, []).result(timeout=5) is True
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["request_count"] == 1
        assert status["metrics"]["invalid_payload_count"] == 1
        assert status["metrics"]["unsupported_count"] == 1
    finally:
        _terminate_proc(proc)


def test_native_binary_handles_sigterm_gracefully(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        proc.terminate()
        assert proc.wait(timeout=5) == 0
        assert proc.stderr is not None
        assert "LMCache native MP server listening" in proc.stderr.read()
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_gpu_hot_cache_defaults_off(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["cuda_transfer_enabled"] is True
        assert status["cuda_gpu_hot_cache_enabled"] is False
        hot_cache = status["cuda_gpu_hot_cache"]
        assert isinstance(hot_cache, dict)
        assert hot_cache["entries"] == 0
        assert hot_cache["bytes"] == 0
    finally:
        _terminate_proc(proc)


def _run_native_cuda_ipc_store_retrieve(
    tmp_path: Path,
    wrapper_cls: type[CudaIPCWrapper],
    request_id: str,
    kv_layout: str = "NHD",
    tensor_layout: str | None = None,
    retrieve_skip_first_n_tokens: int = 0,
    iterations: int = 1,
    extra_layout_hints: dict[str, object] | None = None,
    engine_type: EngineType = EngineType.VLLM,
    logical_block_size_hint: int | None = 16,
    token_count: int = 32,
    store_block_ids: list[int] | None = None,
    retrieve_block_ids: list[int] | None = None,
    free_lookup_before_retrieve: bool = True,
) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if iterations != 1 and retrieve_skip_first_n_tokens != 0:
        raise ValueError("repeated test helper does not support skip-token mode")
    if token_count % 32 != 0:
        raise ValueError("test helper token_count must be chunk aligned")
    store_block_ids = store_block_ids or [1, 2]
    retrieve_block_ids = retrieve_block_ids or [3, 4]
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    tensor_layout = tensor_layout or kv_layout
    shape: tuple[int, ...]
    block_slice: tuple[slice, ...]
    retrieve_slice: tuple[slice, ...]
    if tensor_layout == "NHD":
        shape = (2, 6, 16, 1, 8)
        block_slice = (slice(None), slice(1, 3))
        retrieve_slice = (slice(None), slice(3, 5))
    elif tensor_layout == "COMPACT_NHD":
        shape = (2, 6, 16, 8)
        block_slice = (slice(None), slice(1, 3))
        retrieve_slice = (slice(None), slice(3, 5))
    elif tensor_layout == "COMPRESSED_NHD":
        shape = (2, 6, 8, 1, 8)
        block_slice = (slice(None), slice(1, 3))
        retrieve_slice = (slice(None), slice(3, 5))
    elif tensor_layout == "LARGE_NHD":
        shape = (2, 8, 16, 4, 16)
        block_slice = (slice(None), slice(1, 3))
        retrieve_slice = (slice(None), slice(3, 5))
    elif tensor_layout == "MULTI_CHUNK_NHD":
        shape = (2, 10, 16, 1, 8)
        block_slice = (slice(None), slice(1, 5))
        retrieve_slice = (slice(None), slice(5, 9))
    elif tensor_layout == "HND":
        shape = (2, 6, 1, 16, 8)
        block_slice = (slice(None), slice(1, 3))
        retrieve_slice = (slice(None), slice(3, 5))
    elif tensor_layout == "CROSS_LAYER_NHD":
        shape = (6, 4, 2, 16, 1, 8)
        block_slice = (slice(1, 3),)
        retrieve_slice = (slice(3, 5),)
    elif tensor_layout == "CROSS_LAYER_HND":
        shape = (6, 4, 2, 1, 16, 8)
        block_slice = (slice(1, 3),)
        retrieve_slice = (slice(3, 5),)
    elif tensor_layout == "TRTLLM_4D":
        shape = (6, 4, 2, 1 * 16 * 8)
        block_slice = (slice(1, 3),)
        retrieve_slice = (slice(3, 5),)
    elif tensor_layout == "MLA":
        shape = (6, 16, 8)
        block_slice = (slice(1, 3),)
        retrieve_slice = (slice(3, 5),)
    else:
        raise ValueError(f"unsupported test tensor_layout: {tensor_layout!r}")
    numel = math.prod(shape)

    torch.cuda.set_device(0)
    if wrapper_cls is RawCudaIPCWrapper:
        # Third Party
        import cupy

        cupy_cache = cupy.arange(numel, dtype=cupy.float16).reshape(shape)
        kv_cache = torch.from_dlpack(cupy_cache)
    else:
        kv_cache = torch.arange(
            numel,
            device="cuda",
            dtype=torch.float16,
        ).reshape(shape)
    source = kv_cache[block_slice].clone()
    kv_cache[retrieve_slice].zero_()
    if retrieve_skip_first_n_tokens == 0:
        expected = source
    elif retrieve_skip_first_n_tokens == 16:
        expected = torch.zeros_like(source)
        if tensor_layout in {"NHD", "HND"}:
            expected[:, 1:2].copy_(source[:, 1:2])
        elif tensor_layout == "MLA":
            expected[1:2].copy_(source[1:2])
        else:
            raise ValueError(f"unsupported skip test tensor_layout: {tensor_layout!r}")
    else:
        raise ValueError(
            f"unsupported retrieve skip in test: {retrieve_skip_first_n_tokens}"
        )
    torch.cuda.synchronize()
    expected_chunks = token_count // 32

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cuda-gpu-hot-cache",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
            "--l2-adapter",
            json.dumps({"type": "fs", "base_path": str(tmp_path / "l2")}),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            layout_hints: dict[str, object] = {
                "kv_layout": kv_layout,
            }
            if logical_block_size_hint is not None:
                layout_hints["inference_engine_logical_block_size"] = (
                    logical_block_size_hint
                )
            if extra_layout_hints is not None:
                layout_hints.update(extra_layout_hints)
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [wrapper_cls(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        engine_type,
                        layout_hints,
                    ],
                ).result(timeout=5)
                is None
            )
            for iteration in range(iterations):
                iteration_request_id = (
                    request_id if iterations == 1 else f"{request_id}-{iteration}"
                )
                cache_salt = "" if iterations == 1 else iteration_request_id
                worker_key = IPCCacheEngineKey.from_token_ids(
                    model_name="facebook/opt-125m",
                    world_size=1,
                    worker_id=0,
                    token_ids=list(range(token_count)),
                    start=0,
                    end=token_count,
                    request_id=iteration_request_id,
                    cache_salt=cache_salt,
                )

                store_response = cast(
                    tuple[bytes, bool],
                    client.submit_request(
                        RequestType.STORE,
                        [worker_key, 4321, store_block_ids, b""],
                    ).result(timeout=10),
                )
                store_event, store_ok = store_response
                assert store_ok
                assert store_event

                lookup_key = IPCCacheEngineKey.from_token_ids(
                    model_name="facebook/opt-125m",
                    world_size=1,
                    worker_id=None,
                    token_ids=list(range(token_count)),
                    start=0,
                    end=token_count,
                    request_id=f"{iteration_request_id}-lookup",
                    cache_salt=cache_salt,
                )
                assert (
                    client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(
                        timeout=5
                    )
                    is None
                )
                assert (
                    client.submit_request(
                        RequestType.QUERY_PREFETCH_STATUS,
                        [f"{iteration_request_id}-lookup"],
                    ).result(timeout=5)
                    == expected_chunks
                )
                status_with_lookup_lock = json.loads(
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{http_port}/status",
                        timeout=5,
                    ).read()
                )
                assert (
                    status_with_lookup_lock["cache"]["locked_entries"]
                    == expected_chunks
                )
                assert status_with_lookup_lock["cache"]["lock_count"] == expected_chunks
                if free_lookup_before_retrieve:
                    assert (
                        client.submit_request(
                            RequestType.FREE_LOOKUP_LOCKS,
                            [lookup_key, 1],
                        ).result(timeout=5)
                        is None
                    )
                    status_after_free = json.loads(
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{http_port}/status",
                            timeout=5,
                        ).read()
                    )
                    assert status_after_free["cache"]["locked_entries"] == 0
                    assert status_after_free["cache"]["lock_count"] == 0

                kv_cache[retrieve_slice].zero_()
                torch.cuda.synchronize()
                retrieve_response = cast(
                    tuple[bytes, bool],
                    client.submit_request(
                        RequestType.RETRIEVE,
                        [
                            worker_key,
                            4321,
                            retrieve_block_ids,
                            b"",
                            retrieve_skip_first_n_tokens,
                        ],
                    ).result(timeout=10),
                )
                retrieve_event, retrieve_ok = retrieve_response
                assert retrieve_ok
                assert retrieve_event
                torch.cuda.synchronize()
                torch.testing.assert_close(kv_cache[retrieve_slice], expected)
                if not free_lookup_before_retrieve:
                    status_after_retrieve = json.loads(
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{http_port}/status",
                            timeout=5,
                        ).read()
                    )
                    assert status_after_retrieve["cache"]["locked_entries"] == (
                        expected_chunks
                    )
                    assert (
                        status_after_retrieve["cache"]["lock_count"] == expected_chunks
                    )
                    assert (
                        client.submit_request(
                            RequestType.FREE_LOOKUP_LOCKS,
                            [lookup_key, 1],
                        ).result(timeout=5)
                        is None
                    )
                    status_after_free = json.loads(
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{http_port}/status",
                            timeout=5,
                        ).read()
                    )
                    assert status_after_free["cache"]["locked_entries"] == 0
                    assert status_after_free["cache"]["lock_count"] == 0
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        return cast(dict[str, object], status)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_pytorch_cuda_ipc_store_retrieve(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-store-retrieve",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["cuda_gpu_hot_cache_enabled"] is True
    hot_cache = status["cuda_gpu_hot_cache"]
    assert isinstance(hot_cache, dict)
    assert hot_cache["entries"] == 1
    assert hot_cache["bytes"] > 0
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    type_latency = status["metrics"]["request_type_latency"]
    type_queue_wait = status["metrics"]["request_type_queue_wait"]
    assert set(type_latency) == {
        "lookup",
        "store",
        "retrieve",
        "free_lookup_locks",
    }
    assert set(type_queue_wait) == set(type_latency)
    for name, request_type in type_latency.items():
        assert isinstance(request_type["count"], int)
        assert isinstance(request_type["total_us"], int)
        assert isinstance(request_type["max_us"], int)
        assert request_type["max_us"] <= request_type["total_us"]
        assert isinstance(type_queue_wait[name]["count"], int)
        assert isinstance(type_queue_wait[name]["total_us"], int)
        assert isinstance(type_queue_wait[name]["max_us"], int)
        assert type_queue_wait[name]["count"] == request_type["count"]
        assert type_queue_wait[name]["max_us"] <= type_queue_wait[name][
            "total_us"
        ]
    assert type_latency["lookup"]["count"] == 1
    assert type_latency["store"]["count"] == 1
    assert type_latency["retrieve"]["count"] == 1
    assert type_latency["free_lookup_locks"]["count"] == 1
    assert isinstance(status["metrics"]["transfer_lock_wait_total_us"], int)
    assert isinstance(status["metrics"]["transfer_lock_wait_max_us"], int)
    assert isinstance(status["metrics"]["transfer_lock_hold_total_us"], int)
    assert isinstance(status["metrics"]["transfer_lock_hold_max_us"], int)
    assert (
        status["metrics"]["transfer_lock_wait_max_us"]
        <= status["metrics"]["transfer_lock_wait_total_us"]
    )
    assert (
        status["metrics"]["transfer_lock_hold_max_us"]
        <= status["metrics"]["transfer_lock_hold_total_us"]
    )
    assert status["metrics"]["cuda_transfer_memcpy_calls"] == 0
    assert status["metrics"]["cuda_transfer_kernel_calls"] == 3
    assert status["metrics"]["cuda_transfer_bytes"] == 3072
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_retrieve_keeps_lookup_lock_until_free(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-retrieve-with-lookup-lock",
        free_lookup_before_retrieve=False,
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["lookup_count"] == 1
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_rejects_store_over_lookup_locked_chunk(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    shape = (2, 6, 16, 1, 8)
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    expected_original = kv_cache[:, 1:3].clone()
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )

            store_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-locked-overwrite-store",
            )
            store_event, store_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [store_key, 4321, [1, 2], b""],
                ).result(timeout=10),
            )
            assert store_ok
            assert store_event

            lookup_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-locked-overwrite-lookup",
            )
            assert (
                client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(
                    timeout=5
                )
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    [lookup_key.request_id],
                ).result(timeout=5)
                == 1
            )

            kv_cache[:, 1:3].fill_(999)
            torch.cuda.synchronize()
            duplicate_store_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-locked-overwrite-store-duplicate",
            )
            failed_store_event, failed_store_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [duplicate_store_key, 4321, [1, 2], b""],
                ).result(timeout=10),
            )
            assert failed_store_event == b""
            assert failed_store_ok is False

            status_after_failed_store = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_failed_store["cache"]["total_entries"] == 1
            assert status_after_failed_store["cache"]["locked_entries"] == 1
            assert status_after_failed_store["cache"]["lock_count"] == 1
            assert status_after_failed_store["metrics"]["store_count"] == 2
            assert status_after_failed_store["metrics"]["unsupported_count"] == 1

            assert (
                client.submit_request(
                    RequestType.FREE_LOOKUP_LOCKS,
                    [lookup_key, 1],
                ).result(timeout=5)
                is None
            )

            kv_cache[:, 3:5].zero_()
            torch.cuda.synchronize()
            retrieve_event, retrieve_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.RETRIEVE,
                    [store_key, 4321, [3, 4], b"", 0],
                ).result(timeout=10),
            )
            assert retrieve_ok
            assert retrieve_event
            torch.cuda.synchronize()
            torch.testing.assert_close(kv_cache[:, 3:5], expected_original)
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["cuda_transfer_enabled"] is True
        assert status["metrics"]["store_count"] == 2
        assert status["metrics"]["lookup_count"] == 1
        assert status["metrics"]["retrieve_count"] == 1
        assert status["metrics"]["transfer_lock_count"] == 2
        assert status["metrics"]["transfer_lock_failure_count"] == 0
        assert status["metrics"]["unsupported_count"] == 1
        assert status["cache"]["total_entries"] == 1
        assert status["cache"]["locked_entries"] == 0
        assert status["cache"]["lock_count"] == 0
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_multi_chunk_pytorch_cuda_ipc(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-multi-chunk-store-retrieve",
        tensor_layout="MULTI_CHUNK_NHD",
        token_count=64,
        store_block_ids=[1, 2, 3, 4],
        retrieve_block_ids=[5, 6, 7, 8],
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 2
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 2
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 4
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_accepts_layerwise_hint_for_supported_cuda_ipc(
    tmp_path,
):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-layerwise-hint-store-retrieve",
        extra_layout_hints={"use_layerwise": True},
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["registered_contexts"][0]["use_layerwise"] is True
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("shape", "kv_layout", "block_axis"),
    [
        ((2, 6, 16, 1, 8), "NHD", 1),
        ((2, 6, 1, 16, 8), "HND", 1),
        ((6, 16, 8), "NHD", 0),
    ],
    ids=["nhd", "hnd", "mla"],
)
def test_native_cuda_binary_kvcache_check_returns_cuda_checksums(
    tmp_path, shape, kv_layout, block_axis
):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    instance_id = 4321
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    torch.cuda.synchronize()

    block_ids = [1, 3, 4]
    chunk_size = 2
    selected = kv_cache.index_select(
        block_axis,
        torch.tensor(block_ids, device=kv_cache.device),
    ).cpu()
    layer_checksums = []
    aggregate_checksums = []
    for start in range(0, len(block_ids), chunk_size):
        end = min(start + chunk_size, len(block_ids))
        if block_axis == 0:
            chunk = selected[start:end].contiguous()
        else:
            chunk = selected[:, start:end].contiguous()
        layer_digest = hashlib.md5(chunk.numpy().tobytes()).hexdigest()
        layer_checksums.append(layer_digest)
        aggregate_checksums.append(hashlib.md5(layer_digest.encode()).hexdigest())

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        instance_id,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": kv_layout,
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )

            query = urllib.parse.urlencode(
                {
                    "instance_id": instance_id,
                    "block_ids": "1,[3,4]",
                    "chunk_size": chunk_size,
                }
            )
            checksum_body = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/kvcache/check?{query}",
                    timeout=5,
                ).read()
            )
            assert checksum_body == {
                "status": "success",
                "chunk_size": chunk_size,
                "num_chunks": 2,
                "chunk_checksums": aggregate_checksums,
                "layerwise": False,
                "block_id_ranges": "1,[3,4]",
            }

            layerwise_query = urllib.parse.urlencode(
                {
                    "instance_id": instance_id,
                    "block_ids": "1,[3,4]",
                    "chunk_size": chunk_size,
                    "layerwise": "true",
                }
            )
            layerwise_body = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/kvcache/check?{layerwise_query}",
                    timeout=5,
                ).read()
            )
            assert layerwise_body == {
                "status": "success",
                "chunk_size": chunk_size,
                "num_chunks": 2,
                "chunk_checksums": {"layer_0": layer_checksums},
                "layerwise": True,
                "block_id_ranges": "1,[3,4]",
            }
        finally:
            client.close()
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_free_lookup_locks_releases_key_subset(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    shape = (2, 6, 16, 1, 8)
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )

            store_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(64)),
                start=0,
                end=64,
                request_id="native-free-lock-subset-store",
            )
            store_response = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [store_key, 4321, [1, 2, 3, 4], b""],
                ).result(timeout=10),
            )
            assert store_response[1]

            lookup_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(64)),
                start=0,
                end=64,
                request_id="native-free-lock-subset-lookup",
            )
            assert (
                client.submit_request(RequestType.LOOKUP, [lookup_key, 2]).result(
                    timeout=5
                )
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    ["native-free-lock-subset-lookup"],
                ).result(timeout=5)
                == 2
            )
            status_with_two_locks = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_with_two_locks["cache"]["locked_entries"] == 2
            assert status_with_two_locks["cache"]["lock_count"] == 2

            first_chunk_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(64)),
                start=0,
                end=32,
                request_id="native-free-lock-subset-lookup",
            )
            assert (
                client.submit_request(
                    RequestType.FREE_LOOKUP_LOCKS,
                    [first_chunk_key, 1],
                ).result(timeout=5)
                is None
            )
            status_after_partial_free = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_partial_free["cache"]["locked_entries"] == 1
            assert status_after_partial_free["cache"]["lock_count"] == 1

            assert (
                client.submit_request(RequestType.CLEAR, []).result(timeout=5) is None
            )
            status_after_clear = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_clear["cache"]["locked_entries"] == 0
            assert status_after_clear["cache"]["lock_count"] == 0
            assert status_after_clear["cache"]["total_entries"] == 0

            second_chunk_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(64)),
                start=32,
                end=64,
                request_id="native-free-lock-subset-lookup",
            )
            assert (
                client.submit_request(
                    RequestType.FREE_LOOKUP_LOCKS,
                    [second_chunk_key, 1],
                ).result(timeout=5)
                is None
            )
            status_after_full_free = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_full_free["cache"]["locked_entries"] == 0
            assert status_after_full_free["cache"]["lock_count"] == 0
            assert status_after_full_free["metrics"]["unsupported_count"] == 0
        finally:
            client.close()
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_end_session_releases_lookup_locks(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    shape = (2, 6, 16, 1, 8)
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )

            store_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-end-session-store",
            )
            store_response = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [store_key, 4321, [1, 2], b""],
                ).result(timeout=10),
            )
            assert store_response[1]

            lookup_request_id = "native-end-session-lookup"
            lookup_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id=lookup_request_id,
            )
            assert (
                client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(
                    timeout=5
                )
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    [lookup_request_id],
                ).result(timeout=5)
                == 1
            )

            status_locked = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_locked["cache"]["locked_entries"] == 1
            assert status_locked["cache"]["lock_count"] == 1
            assert status_locked["cache"]["total_entries"] == 1

            assert (
                client.submit_request(
                    RequestType.END_SESSION,
                    [lookup_request_id],
                ).result(timeout=5)
                is None
            )
            status_after_end = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_end["cache"]["locked_entries"] == 0
            assert status_after_end["cache"]["lock_count"] == 0
            assert status_after_end["cache"]["total_entries"] == 1

            assert (
                client.submit_request(RequestType.CLEAR, []).result(timeout=5) is None
            )
            status_after_clear = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_clear["cache"]["total_entries"] == 0
            assert status_after_clear["metrics"]["unsupported_count"] == 0
        finally:
            client.close()
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_tracks_same_chunk_lookup_lock_refcounts(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    shape = (2, 6, 16, 1, 8)
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )

            store_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-refcount-store",
            )
            store_response = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [store_key, 4321, [1, 2], b""],
                ).result(timeout=10),
            )
            assert store_response[1]

            lookup_key_a = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-refcount-lookup-a",
            )
            lookup_key_b = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-refcount-lookup-b",
            )
            for lookup_key in (lookup_key_a, lookup_key_b):
                assert (
                    client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(
                        timeout=5
                    )
                    is None
                )
                assert (
                    client.submit_request(
                        RequestType.QUERY_PREFETCH_STATUS,
                        [lookup_key.request_id],
                    ).result(timeout=5)
                    == 1
                )

            status_with_two_owners = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_with_two_owners["cache"]["total_entries"] == 1
            assert status_with_two_owners["cache"]["locked_entries"] == 1
            assert status_with_two_owners["cache"]["lock_count"] == 2

            assert (
                client.submit_request(
                    RequestType.FREE_LOOKUP_LOCKS,
                    [lookup_key_a, 1],
                ).result(timeout=5)
                is None
            )
            status_after_free_one = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_free_one["cache"]["total_entries"] == 1
            assert status_after_free_one["cache"]["locked_entries"] == 1
            assert status_after_free_one["cache"]["lock_count"] == 1

            assert (
                client.submit_request(
                    RequestType.END_SESSION,
                    [lookup_key_b.request_id],
                ).result(timeout=5)
                is None
            )
            status_after_end_second = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_end_second["cache"]["total_entries"] == 1
            assert status_after_end_second["cache"]["locked_entries"] == 0
            assert status_after_end_second["cache"]["lock_count"] == 0

            assert (
                client.submit_request(RequestType.CLEAR, []).result(timeout=5) is None
            )
            status_after_clear = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_clear["cache"]["total_entries"] == 0
            assert status_after_clear["metrics"]["unsupported_count"] == 0
        finally:
            client.close()
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_reused_lookup_request_releases_previous_locks(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    shape = (2, 6, 16, 1, 8)
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )

            store_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(64)),
                start=0,
                end=64,
                request_id="native-reused-lookup-store",
            )
            store_response = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [store_key, 4321, [1, 2, 3, 4], b""],
                ).result(timeout=10),
            )
            assert store_response[1]

            lookup_request_id = "native-reused-lookup"
            first_lookup_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(32)),
                request_id=lookup_request_id,
            )
            assert (
                client.submit_request(RequestType.LOOKUP, [first_lookup_key, 1]).result(
                    timeout=5
                )
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    [lookup_request_id],
                ).result(timeout=5)
                == 1
            )
            status_first_lookup = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_first_lookup["cache"]["locked_entries"] == 1
            assert status_first_lookup["cache"]["lock_count"] == 1
            assert status_first_lookup["cache"]["total_entries"] == 2

            second_lookup_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(64)),
                request_id=lookup_request_id,
            )
            assert (
                client.submit_request(
                    RequestType.LOOKUP, [second_lookup_key, 1]
                ).result(timeout=5)
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    [lookup_request_id],
                ).result(timeout=5)
                == 2
            )
            status_second_lookup = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_second_lookup["cache"]["locked_entries"] == 2
            assert status_second_lookup["cache"]["lock_count"] == 2
            assert status_second_lookup["cache"]["total_entries"] == 2

            assert (
                client.submit_request(
                    RequestType.END_SESSION,
                    [lookup_request_id],
                ).result(timeout=5)
                is None
            )
            status_after_end = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_end["cache"]["locked_entries"] == 0
            assert status_after_end["cache"]["lock_count"] == 0
            assert status_after_end["metrics"]["unsupported_count"] == 0
        finally:
            client.close()
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_repeats_pytorch_cuda_ipc_store_retrieve(tmp_path):
    iterations = 8
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-repeated-store-retrieve",
        iterations=iterations,
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == iterations
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == iterations
    assert status["metrics"]["retrieve_count"] == iterations
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == iterations
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == iterations * 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    os.getenv("LMCACHE_RUN_LONG_CUDA_STRESS") != "1",
    reason="set LMCACHE_RUN_LONG_CUDA_STRESS=1 to run the longer CUDA stress test",
)
def test_native_cuda_binary_long_pytorch_cuda_ipc_store_retrieve(tmp_path):
    iterations = 32
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-long-store-retrieve",
        iterations=iterations,
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == iterations
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == iterations
    assert status["metrics"]["retrieve_count"] == iterations
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == iterations
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == iterations * 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


def _run_native_cuda_concurrent_pytorch_cuda_ipc_round_trips(
    tmp_path,
    *,
    client_count: int,
    rounds_per_client: int,
    extra_layout_hints: dict[str, object] | None = None,
) -> dict[str, object]:
    if client_count < 1:
        raise ValueError("client_count must be at least 1")
    if rounds_per_client < 1:
        raise ValueError("rounds_per_client must be at least 1")

    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    shape = (2, 6, 16, 1, 8)
    numel = math.prod(shape)
    total_round_trips = client_count * rounds_per_client

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--max-workers",
            str(client_count),
            "--cxx-disk-path",
            str(tmp_path / "disk"),
            "--l2-adapter",
            json.dumps({"type": "fs", "base_path": str(tmp_path / "l2")}),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        short_timeout = _native_request_timeout(10)
        transfer_timeout = _native_request_timeout(20)

        def run_round_trip(client_index: int) -> None:
            torch.cuda.set_device(0)
            base = torch.arange(
                numel,
                device="cuda",
                dtype=torch.float16,
            ).reshape(shape)
            kv_cache = torch.empty_like(base)
            torch.cuda.synchronize()

            context = zmq.Context.instance()
            client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
            instance_id = 5000 + client_index
            try:
                layout_hints: dict[str, object] = {
                    "kv_layout": "NHD",
                    "inference_engine_logical_block_size": 16,
                }
                if extra_layout_hints is not None:
                    layout_hints.update(extra_layout_hints)
                assert (
                    client.submit_request(
                        RequestType.REGISTER_KV_CACHE,
                        [
                            instance_id,
                            [CudaIPCWrapper(kv_cache)],
                            "facebook/opt-125m",
                            1,
                            EngineType.VLLM,
                            layout_hints,
                        ],
                    ).result(timeout=short_timeout)
                    is None
                )
                for round_index in range(rounds_per_client):
                    round_name = f"native-concurrent-{client_index}-{round_index}"
                    kv_cache.copy_(base + client_index * 2048 + round_index * 256)
                    expected = kv_cache[:, 1:3].clone()
                    kv_cache[:, 3:5].zero_()
                    torch.cuda.synchronize()

                    worker_key = IPCCacheEngineKey.from_token_ids(
                        model_name="facebook/opt-125m",
                        world_size=1,
                        worker_id=0,
                        token_ids=list(range(32)),
                        start=0,
                        end=32,
                        request_id=f"{round_name}-store",
                        cache_salt=round_name,
                    )
                    store_event, store_ok = cast(
                        tuple[bytes, bool],
                        client.submit_request(
                            RequestType.STORE,
                            [worker_key, instance_id, [1, 2], b""],
                        ).result(timeout=transfer_timeout),
                    )
                    assert store_ok
                    assert store_event

                    lookup_key = IPCCacheEngineKey.from_token_ids(
                        model_name="facebook/opt-125m",
                        world_size=1,
                        worker_id=None,
                        token_ids=list(range(32)),
                        start=0,
                        end=32,
                        request_id=f"{round_name}-lookup",
                        cache_salt=round_name,
                    )
                    assert (
                        client.submit_request(
                            RequestType.LOOKUP,
                            [lookup_key, 1],
                        ).result(timeout=short_timeout)
                        is None
                    )
                    assert (
                        client.submit_request(
                            RequestType.QUERY_PREFETCH_STATUS,
                            [lookup_key.request_id],
                        ).result(timeout=short_timeout)
                        == 1
                    )

                    kv_cache[:, 3:5].zero_()
                    torch.cuda.synchronize()
                    retrieve_event, retrieve_ok = cast(
                        tuple[bytes, bool],
                        client.submit_request(
                            RequestType.RETRIEVE,
                            [worker_key, instance_id, [3, 4], b"", 0],
                        ).result(timeout=transfer_timeout),
                    )
                    assert retrieve_ok
                    assert retrieve_event
                    torch.cuda.synchronize()
                    torch.testing.assert_close(kv_cache[:, 3:5], expected)
                    assert (
                        client.submit_request(
                            RequestType.FREE_LOOKUP_LOCKS,
                            [lookup_key, 1],
                        ).result(timeout=short_timeout)
                        is None
                    )
            finally:
                client.close()

        with ThreadPoolExecutor(max_workers=client_count) as executor:
            list(executor.map(run_round_trip, range(client_count)))

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["cuda_transfer_enabled"] is True
        assert status["registered_context_count"] == client_count
        assert status["l2_adapters"][0]["stored_files"] == total_round_trips
        assert status["l2_adapters"][0]["stored_bytes"] > 0
        assert status["metrics"]["worker_count"] == client_count
        assert status["metrics"]["observed_client_count"] >= client_count
        assert status["metrics"]["store_count"] == total_round_trips
        assert status["metrics"]["retrieve_count"] == total_round_trips
        assert status["metrics"]["lookup_count"] == total_round_trips
        assert status["metrics"]["l2_store_count"] == total_round_trips
        assert status["metrics"]["l2_load_count"] == 0
        assert status["metrics"]["l2_error_count"] == 0
        assert status["metrics"]["transfer_lock_count"] == total_round_trips * 2
        assert status["metrics"]["transfer_lock_failure_count"] == 0
        assert status["cache"]["locked_entries"] == 0
        assert status["cache"]["lock_count"] == 0
        assert status["metrics"]["unsupported_count"] == 0
        return status
    finally:
        _terminate_proc(proc)
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_handles_concurrent_pytorch_cuda_ipc_round_trips(tmp_path):
    _run_native_cuda_concurrent_pytorch_cuda_ipc_round_trips(
        tmp_path,
        client_count=4,
        rounds_per_client=1,
    )


def _require_cuda_tsan_binary() -> None:
    binary = ensure_native_binary(enable_cuda=True)
    ldd = subprocess.run(
        ["ldd", str(binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if "libtsan" not in ldd.stdout:
        pytest.skip("native CUDA binary is not linked with ThreadSanitizer")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_CUDA_TSAN_STRESS") != "1",
    reason="set LMCACHE_RUN_CUDA_TSAN_STRESS=1 for CUDA TSAN data-path coverage",
)
def test_native_cuda_binary_tsan_concurrent_pytorch_cuda_ipc_round_trips(tmp_path):
    _require_cuda_tsan_binary()

    _run_native_cuda_concurrent_pytorch_cuda_ipc_round_trips(
        tmp_path,
        client_count=2,
        rounds_per_client=1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_CUDA_TSAN_STRESS") != "1",
    reason="set LMCACHE_RUN_CUDA_TSAN_STRESS=1 for CUDA TSAN data-path coverage",
)
def test_native_cuda_binary_tsan_pytorch_cuda_ipc_round_trip(tmp_path):
    _require_cuda_tsan_binary()

    _run_native_cuda_concurrent_pytorch_cuda_ipc_round_trips(
        tmp_path,
        client_count=1,
        rounds_per_client=1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_handles_layerwise_concurrent_pytorch_cuda_ipc_round_trips(
    tmp_path,
):
    status = _run_native_cuda_concurrent_pytorch_cuda_ipc_round_trips(
        tmp_path,
        client_count=4,
        rounds_per_client=1,
        extra_layout_hints={"use_layerwise": True},
    )
    assert all(
        context["use_layerwise"] is True for context in status["registered_contexts"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    os.getenv("LMCACHE_RUN_LONG_CUDA_CONCURRENCY_STRESS") != "1",
    reason=(
        "set LMCACHE_RUN_LONG_CUDA_CONCURRENCY_STRESS=1 to run the longer "
        "concurrent CUDA stress test"
    ),
)
def test_native_cuda_binary_long_concurrent_pytorch_cuda_ipc_round_trips(tmp_path):
    _run_native_cuda_concurrent_pytorch_cuda_ipc_round_trips(
        tmp_path,
        client_count=8,
        rounds_per_client=4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_pytorch_cuda_ipc_hnd_layout(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-hnd-store-retrieve",
        kv_layout="HND",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_compact_pytorch_cuda_ipc_layout(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-compact-nhd-store-retrieve",
        tensor_layout="COMPACT_NHD",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_layerwise_heterogeneous_cuda_ipc_layout(
    tmp_path,
):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()

    torch.cuda.set_device(0)
    layer0 = torch.arange(
        2 * 6 * 16 * 1 * 8,
        device="cuda",
        dtype=torch.float16,
    ).reshape(2, 6, 16, 1, 8)
    layer1 = (
        torch.arange(
            2 * 6 * 16 * 2 * 8,
            device="cuda",
            dtype=torch.float16,
        ).reshape(2, 6, 16, 2, 8)
        + 4096
    )
    expected0 = layer0[:, 1:3].clone()
    expected1 = layer1[:, 1:3].clone()
    layer0[:, 3:5].zero_()
    layer1[:, 3:5].zero_()
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
            "--l2-adapter",
            json.dumps({"type": "fs", "base_path": str(tmp_path / "l2")}),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(layer0), CudaIPCWrapper(layer1)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                            "use_layerwise": True,
                        },
                    ],
                ).result(timeout=10)
                is None
            )
            worker_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-pytorch-cuda-ipc-heterogeneous-store-retrieve",
            )
            store_event, store_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [worker_key, 4321, [1, 2], b""],
                ).result(timeout=20),
            )
            assert store_ok
            assert store_event

            lookup_key = worker_key.no_worker_id_version()
            assert (
                client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(
                    timeout=10
                )
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    [lookup_key.request_id],
                ).result(timeout=10)
                == 1
            )

            layer0[:, 3:5].zero_()
            layer1[:, 3:5].zero_()
            torch.cuda.synchronize()
            retrieve_event, retrieve_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.RETRIEVE,
                    [worker_key, 4321, [3, 4], b"", 0],
                ).result(timeout=20),
            )
            assert retrieve_ok
            assert retrieve_event
            torch.cuda.synchronize()
            torch.testing.assert_close(layer0[:, 3:5], expected0)
            torch.testing.assert_close(layer1[:, 3:5], expected1)
            assert (
                client.submit_request(
                    RequestType.FREE_LOOKUP_LOCKS,
                    [lookup_key, 1],
                ).result(timeout=10)
                is None
            )
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["cuda_transfer_enabled"] is True
        assert status["registered_context_count"] == 1
        assert status["registered_contexts"][0]["kv_cache_wrapper_count"] == 2
        assert status["registered_contexts"][0]["use_layerwise"] is True
        assert status["metrics"]["store_count"] == 1
        assert status["metrics"]["retrieve_count"] == 1
        assert status["metrics"]["l2_store_count"] == 1
        assert status["metrics"]["transfer_lock_failure_count"] == 0
        assert status["metrics"]["unsupported_count"] == 0
        assert status["cache"]["locked_entries"] == 0
        assert status["cache"]["lock_count"] == 0
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_mixed_compression_pytorch_cuda_ipc_layout(
    tmp_path,
):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()

    torch.cuda.set_device(0)
    uncompressed_layer = torch.arange(
        2 * 6 * 16 * 1 * 8,
        device="cuda",
        dtype=torch.float16,
    ).reshape(2, 6, 16, 1, 8)
    compressed_layer = (
        torch.arange(
            2 * 6 * 8 * 1 * 8,
            device="cuda",
            dtype=torch.float16,
        ).reshape(2, 6, 8, 1, 8)
        + 4096
    )
    expected_uncompressed = uncompressed_layer[:, 1:3].clone()
    expected_compressed = compressed_layer[:, 1:3].clone()
    uncompressed_layer[:, 3:5].zero_()
    compressed_layer[:, 3:5].zero_()
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
            "--l2-adapter",
            json.dumps({"type": "fs", "base_path": str(tmp_path / "l2")}),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [
                            CudaIPCWrapper(uncompressed_layer),
                            CudaIPCWrapper(compressed_layer),
                        ],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                            "group_compress_ratios": [1, 2],
                            "group_physical_block_sizes": [16, 8],
                        },
                    ],
                ).result(timeout=10)
                is None
            )
            worker_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id=("native-pytorch-cuda-ipc-mixed-compression-store-retrieve"),
            )
            store_event, store_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [worker_key, 4321, [1, 2], b""],
                ).result(timeout=20),
            )
            assert store_ok
            assert store_event

            lookup_key = worker_key.no_worker_id_version()
            assert (
                client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(
                    timeout=10
                )
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    [lookup_key.request_id],
                ).result(timeout=10)
                == 1
            )

            uncompressed_layer[:, 3:5].zero_()
            compressed_layer[:, 3:5].zero_()
            torch.cuda.synchronize()
            retrieve_event, retrieve_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.RETRIEVE,
                    [worker_key, 4321, [3, 4], b"", 0],
                ).result(timeout=20),
            )
            assert retrieve_ok
            assert retrieve_event
            torch.cuda.synchronize()
            torch.testing.assert_close(
                uncompressed_layer[:, 3:5], expected_uncompressed
            )
            torch.testing.assert_close(compressed_layer[:, 3:5], expected_compressed)
            assert (
                client.submit_request(
                    RequestType.FREE_LOOKUP_LOCKS,
                    [lookup_key, 1],
                ).result(timeout=10)
                is None
            )
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["cuda_transfer_enabled"] is True
        assert status["registered_context_count"] == 1
        assert status["registered_contexts"][0]["kv_cache_wrapper_count"] == 2
        assert status["metrics"]["store_count"] == 1
        assert status["metrics"]["retrieve_count"] == 1
        assert status["metrics"]["l2_store_count"] == 1
        assert status["metrics"]["transfer_lock_failure_count"] == 0
        assert status["metrics"]["unsupported_count"] == 0
        assert status["cache"]["locked_entries"] == 0
        assert status["cache"]["lock_count"] == 0
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_compressed_pytorch_cuda_ipc_layout(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-compressed-nhd-store-retrieve",
        tensor_layout="COMPRESSED_NHD",
        extra_layout_hints={"compress_ratio": 2},
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_larger_pytorch_cuda_ipc_layout(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-larger-store-retrieve",
        kv_layout="NHD",
        tensor_layout="LARGE_NHD",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 1024
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_pytorch_cuda_ipc_cross_layer_layout(
    tmp_path,
):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-cross-layer-store-retrieve",
        kv_layout="NHD",
        tensor_layout="CROSS_LAYER_NHD",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_pytorch_cuda_ipc_cross_layer_hnd_layout(
    tmp_path,
):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-cross-layer-hnd-store-retrieve",
        kv_layout="HND",
        tensor_layout="CROSS_LAYER_HND",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_trtllm_4d_pytorch_cuda_ipc_layout(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-trtllm-4d-store-retrieve",
        kv_layout="HND",
        tensor_layout="TRTLLM_4D",
        engine_type=EngineType.TRTLLM,
        logical_block_size_hint=None,
        extra_layout_hints={
            "num_kv_heads": 1,
            "tokens_per_block": 16,
            "head_dim": 8,
        },
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["registered_contexts"][0]["engine_type"] == "trtllm"
    assert status["registered_contexts"][0]["trt_llm_layout_hints"] is True
    assert status["registered_contexts"][0]["first_kv_block_size"] == 16
    assert status["registered_contexts"][0]["inference_engine_logical_block_size"] == 16
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_round_trips_pytorch_cuda_ipc_mla_layout(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-mla-store-retrieve",
        kv_layout="NHD",
        tensor_layout="MLA",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_retrieve_respects_skip_first_tokens(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        CudaIPCWrapper,
        "native-pytorch-cuda-ipc-retrieve-skip-first",
        retrieve_skip_first_n_tokens=16,
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_retrieve_missing_key_fails_cleanly(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    shape = (2, 6, 16, 1, 8)
    kv_cache = torch.zeros(shape, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )

            missing_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-missing-retrieve",
            )
            retrieve_event, retrieve_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.RETRIEVE,
                    [missing_key, 4321, [1, 2], b"", 0],
                ).result(timeout=10),
            )
            assert retrieve_event == b""
            assert retrieve_ok is False
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["retrieve_count"] == 1
        assert status["metrics"]["transfer_lock_count"] == 0
        assert status["metrics"]["transfer_lock_failure_count"] == 1
        assert status["cache"]["total_entries"] == 0
        assert status["cache"]["locked_entries"] == 0
        assert status["cache"]["lock_count"] == 0
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_retrieve_partial_missing_releases_locks(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    shape = (2, 6, 16, 1, 8)
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )
            stored_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-partial-missing-retrieve-store",
            )
            store_event, store_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [stored_key, 4321, [1, 2], b""],
                ).result(timeout=10),
            )
            assert store_ok
            assert store_event

            partial_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(64)),
                start=0,
                end=64,
                request_id="native-partial-missing-retrieve",
            )
            retrieve_event, retrieve_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.RETRIEVE,
                    [partial_key, 4321, [2, 3, 4, 5], b"", 0],
                ).result(timeout=10),
            )
            assert retrieve_event == b""
            assert retrieve_ok is False
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["store_count"] == 1
        assert status["metrics"]["retrieve_count"] == 1
        assert status["metrics"]["transfer_lock_count"] == 2
        assert status["metrics"]["transfer_lock_failure_count"] == 1
        assert status["cache"]["total_entries"] == 1
        assert status["cache"]["locked_entries"] == 0
        assert status["cache"]["lock_count"] == 0
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_cuda_binary_unregister_kv_cache_rejects_later_transfer(tmp_path):
    binary = ensure_native_binary(enable_cuda=True)
    zmq_port = _free_port()
    http_port = _free_port()
    instance_id = 4321
    shape = (2, 6, 16, 1, 8)
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    torch.cuda.synchronize()

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "32",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        instance_id,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )

            worker_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(32)),
                start=0,
                end=32,
                request_id="native-unregister-store",
            )
            store_event, store_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [worker_key, instance_id, [1, 2], b""],
                ).result(timeout=10),
            )
            assert store_ok
            assert store_event

            assert (
                client.submit_request(
                    RequestType.UNREGISTER_KV_CACHE,
                    [instance_id],
                ).result(timeout=5)
                is None
            )
            status_after_unregister = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            assert status_after_unregister["registered_context_count"] == 0
            assert status_after_unregister["cache"]["locked_entries"] == 0
            assert status_after_unregister["cache"]["lock_count"] == 0

            failed_store_event, failed_store_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [worker_key, instance_id, [3, 4], b""],
                ).result(timeout=10),
            )
            assert failed_store_event == b""
            assert failed_store_ok is False

            failed_retrieve_event, failed_retrieve_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.RETRIEVE,
                    [worker_key, instance_id, [3, 4], b"", 0],
                ).result(timeout=10),
            )
            assert failed_retrieve_event == b""
            assert failed_retrieve_ok is False
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["registered_context_count"] == 0
        assert status["metrics"]["store_count"] == 2
        assert status["metrics"]["retrieve_count"] == 1
        assert status["metrics"]["unsupported_count"] == 2
        assert status["metrics"]["invalid_payload_count"] == 2
        assert status["metrics"]["transfer_lock_count"] == 1
        assert status["metrics"]["transfer_lock_failure_count"] == 0
        assert status["cache"]["total_entries"] == 1
        assert status["cache"]["locked_entries"] == 0
        assert status["cache"]["lock_count"] == 0
    finally:
        _terminate_proc(proc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("first_server_stop", ["terminate", "kill"])
def test_native_cuda_binary_retrieves_filesystem_l2_after_restart(
    tmp_path,
    first_server_stop: str,
):
    binary = ensure_native_binary(enable_cuda=True)
    l2_dir = tmp_path / "l2"
    l2_dir.mkdir()

    shape = (2, 6, 16, 1, 8)
    torch.cuda.set_device(0)
    kv_cache = torch.arange(
        math.prod(shape),
        device="cuda",
        dtype=torch.float16,
    ).reshape(shape)
    expected = kv_cache[:, 1:3].clone()
    torch.cuda.synchronize()

    worker_key = IPCCacheEngineKey.from_token_ids(
        model_name="facebook/opt-125m",
        world_size=1,
        worker_id=0,
        token_ids=list(range(32)),
        start=0,
        end=32,
        request_id="native-pytorch-cuda-ipc-restart-l2",
    )

    def start_server(disk_name: str) -> tuple[subprocess.Popen[str], int, int]:
        zmq_port = _free_port()
        http_port = _free_port()
        proc = subprocess.Popen(
            [
                str(binary),
                "--host",
                "127.0.0.1",
                "--port",
                str(zmq_port),
                "--http-host",
                "127.0.0.1",
                "--http-port",
                str(http_port),
                "--l1-size-gb",
                "0.001",
                "--chunk-size",
                "32",
                "--cxx-disk-path",
                str(tmp_path / disk_name),
                "--l2-adapter",
                json.dumps({"type": "fs", "base_path": str(l2_dir)}),
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        return proc, zmq_port, http_port

    proc1, zmq_port1, http_port1 = start_server("disk-first")
    try:
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port1}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )
            store_event, store_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.STORE,
                    [worker_key, 4321, [1, 2], b""],
                ).result(timeout=10),
            )
            assert store_ok
            assert store_event
        finally:
            client.close()

        status_after_store = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port1}/status",
                timeout=5,
            ).read()
        )
        assert status_after_store["l2_adapters"][0]["stored_files"] == 1
        assert status_after_store["metrics"]["l2_store_count"] == 1
        assert status_after_store["metrics"]["l2_error_count"] == 0
    finally:
        if first_server_stop == "kill":
            proc1.kill()
            proc1.wait(timeout=5)
        else:
            _terminate_proc(proc1)

    kv_cache[:, 3:5].zero_()
    torch.cuda.synchronize()

    proc2, zmq_port2, http_port2 = start_server("disk-second")
    try:
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port2}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.REGISTER_KV_CACHE,
                    [
                        4321,
                        [CudaIPCWrapper(kv_cache)],
                        "facebook/opt-125m",
                        1,
                        EngineType.VLLM,
                        {
                            "kv_layout": "NHD",
                            "inference_engine_logical_block_size": 16,
                        },
                    ],
                ).result(timeout=5)
                is None
            )
            retrieve_event, retrieve_ok = cast(
                tuple[bytes, bool],
                client.submit_request(
                    RequestType.RETRIEVE,
                    [worker_key, 4321, [3, 4], b"", 0],
                ).result(timeout=10),
            )
            assert retrieve_ok
            assert retrieve_event
            torch.cuda.synchronize()
            torch.testing.assert_close(kv_cache[:, 3:5], expected)
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port2}/status",
                timeout=5,
            ).read()
        )
        assert status["cuda_transfer_enabled"] is True
        assert status["l2_adapters"][0]["stored_files"] == 1
        assert status["l2_adapters"][0]["stored_bytes"] > 0
        assert status["metrics"]["store_count"] == 0
        assert status["metrics"]["retrieve_count"] == 1
        assert status["metrics"]["l2_store_count"] == 0
        assert status["metrics"]["l2_load_count"] == 1
        assert status["metrics"]["l2_error_count"] == 0
        assert status["metrics"]["transfer_lock_count"] == 1
        assert status["metrics"]["transfer_lock_failure_count"] == 0
        assert status["metrics"]["unsupported_count"] == 0
        assert status["cache"]["disk_bytes"] == 0
    finally:
        _terminate_proc(proc2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    not _cuda_python_runtime_available(),
    reason="cuda-python runtime bindings are not available",
)
@pytest.mark.skipif(not _cupy_available(), reason="cupy is not available")
def test_native_cuda_binary_round_trips_raw_cuda_ipc_store_retrieve(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        RawCudaIPCWrapper,
        "native-raw-cuda-ipc-store-retrieve",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    not _cuda_python_runtime_available(),
    reason="cuda-python runtime bindings are not available",
)
@pytest.mark.skipif(not _cupy_available(), reason="cupy is not available")
def test_native_cuda_binary_round_trips_raw_cuda_ipc_hnd_layout(tmp_path):
    status = _run_native_cuda_ipc_store_retrieve(
        tmp_path,
        RawCudaIPCWrapper,
        "native-raw-cuda-ipc-hnd-store-retrieve",
        kv_layout="HND",
    )
    assert status["cuda_transfer_enabled"] is True
    assert status["l2_adapters"][0]["stored_files"] == 1
    assert status["l2_adapters"][0]["stored_bytes"] > 0
    assert status["metrics"]["store_count"] == 1
    assert status["metrics"]["retrieve_count"] == 1
    assert status["metrics"]["clear_count"] == 0
    assert status["metrics"]["l2_store_count"] == 1
    assert status["metrics"]["l2_load_count"] == 0
    assert status["metrics"]["l2_error_count"] == 0
    assert status["metrics"]["transfer_lock_count"] == 2
    assert status["metrics"]["transfer_lock_failure_count"] == 0
    assert status["cache"]["disk_bytes"] == 0
    assert status["cache"]["locked_entries"] == 0
    assert status["cache"]["lock_count"] == 0
    assert status["metrics"]["unsupported_count"] == 0


def test_native_binary_lookup_counts_filesystem_l2_hits(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    l2_dir = tmp_path / "l2"
    l2_dir.mkdir()

    lookup_key = IPCCacheEngineKey.from_token_ids(
        model_name="facebook/opt-125m",
        world_size=1,
        worker_id=None,
        token_ids=list(range(256)),
        request_id="native-l2-hit-1",
        cache_salt="tenant-a",
    )
    hasher = TokenHasher(chunk_size=128, hash_algorithm="blake3")
    object_keys = ipc_key_to_object_keys(
        lookup_key,
        hasher.compute_chunk_hashes(list(lookup_key.token_ids)),
    )
    assert len(object_keys) == 2
    (l2_dir / _fs_l2_filename(object_keys[0])).write_bytes(b"l2-payload")

    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--chunk-size",
            "128",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
            "--l2-adapter",
            json.dumps({"type": "fs", "base_path": str(l2_dir)}),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(
                    timeout=5
                )
                is None
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_LOOKUP_HITS,
                    ["native-l2-hit-1"],
                ).result(timeout=5)
                == 1
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_STATUS,
                    ["native-l2-hit-1"],
                ).result(timeout=5)
                == 1
            )
            lookup_with_result_key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=None,
                token_ids=list(range(256)),
                request_id="native-l2-hit-with-result",
                cache_salt="tenant-a",
            )
            assert (
                client.submit_request(
                    RequestType.LOOKUP_WITH_RESULT,
                    [lookup_with_result_key, 1],
                ).result(timeout=5)
                == 1
            )
            assert (
                client.submit_request(
                    RequestType.QUERY_PREFETCH_LOOKUP_HITS,
                    ["native-l2-hit-with-result"],
                ).result(timeout=5)
                == 0
            )
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["cache_hits"] == 2
        assert status["metrics"]["cache_misses"] == 2
        assert status["metrics"]["cache_hit_rate"] == 0.5
        assert status["metrics"]["partial_hit_count"] == 2
        assert status["metrics"]["l1_hit_count"] == 0
        assert status["metrics"]["l2_hit_count"] == 2
        assert status["metrics"]["l2_miss_count"] == 2
        assert status["metrics"]["request_type_latency"]["lookup"]["count"] == 2
        assert (
            status["metrics"]["request_type_queue_wait"]["lookup"]["count"] == 2
        )
        assert status["l2_adapters"][0]["stored_files"] == 1

        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(RequestType.CLEAR, []).result(timeout=5) is None
            )
        finally:
            client.close()

        status_after_clear = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status_after_clear["metrics"]["clear_count"] == 1
        assert status_after_clear["metrics"]["l2_error_count"] == 0
        assert status_after_clear["l2_adapters"][0]["stored_files"] == 0
        assert not (l2_dir / _fs_l2_filename(object_keys[0])).exists()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_native_binary_speaks_blend_protocol_safe_shapes(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            key = IPCCacheEngineKey.from_token_ids(
                model_name="facebook/opt-125m",
                world_size=1,
                worker_id=0,
                token_ids=list(range(128)),
                request_id="native-cb-1",
            )

            assert (
                client.submit_request(
                    RequestType.CB_REGISTER_KV_CACHE,
                    [321, [], "facebook/opt-125m", 1],
                ).result(timeout=5)
                is None
            )
            assert (
                client.submit_request(
                    RequestType.CB_LOOKUP_PRE_COMPUTED,
                    [key],
                ).result(timeout=5)
                == []
            )
            assert (
                client.submit_request(
                    RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
                    [key],
                ).result(timeout=5)
                == []
            )
            assert client.submit_request(
                RequestType.CB_STORE_PRE_COMPUTED,
                [key, 0, 321, b""],
            ).result(timeout=5) == (b"", False)
            assert client.submit_request(
                RequestType.CB_RETRIEVE_PRE_COMPUTED,
                [key, [(0, 16)], 0, 321, b""],
            ).result(timeout=5) == (b"", False)
            assert client.submit_request(
                RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
                [
                    key,
                    [
                        CBMatchResult(
                            old_st=0,
                            old_ed=16,
                            cur_st=0,
                            cur_ed=16,
                            hash=b"\x01" * 32,
                        )
                    ],
                    0,
                    321,
                    b"",
                ],
            ).result(timeout=5) == (b"", False)
            assert client.submit_request(
                RequestType.CB_STORE_FINAL,
                [key, 0, 321, b""],
            ).result(timeout=5) == (b"", False)
            assert (
                client.submit_request(
                    RequestType.CB_UNREGISTER_KV_CACHE,
                    [321],
                ).result(timeout=5)
                is None
            )
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["invalid_payload_count"] == 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_native_binary_reports_invalid_blend_payloads(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            assert (
                client.submit_request(
                    RequestType.CB_LOOKUP_PRE_COMPUTED,
                    ["not-an-ipc-key"],
                ).result(timeout=5)
                == []
            )
            assert client.submit_request(RequestType.PING, []).result(timeout=5) is True
        finally:
            client.close()

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["invalid_payload_count"] == 1
        assert status["metrics"]["unsupported_count"] >= 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_native_binary_survives_malformed_frame_and_ping_stress(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            "0.001",
            "--max-workers",
            "4",
            "--cxx-disk-path",
            str(tmp_path / "disk"),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        context = zmq.Context.instance()

        malformed = context.socket(zmq.DEALER)
        malformed.setsockopt(zmq.LINGER, 1000)
        malformed.connect(f"tcp://127.0.0.1:{zmq_port}")
        malformed.send_multipart([b"malformed-only-one-frame"])
        time.sleep(0.05)
        malformed.close()
        time.sleep(0.05)

        def worker(client_id: int) -> None:
            client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
            try:
                for i in range(20):
                    if (client_id + i) % 2 == 0:
                        assert (
                            client.submit_request(RequestType.PING, []).result(
                                timeout=5
                            )
                            is True
                        )
                    else:
                        assert (
                            client.submit_request(RequestType.NOOP, []).result(
                                timeout=5
                            )
                            == "OK"
                        )
            finally:
                client.close()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, client_id) for client_id in range(4)]
            for future in futures:
                future.result(timeout=20)

        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/status",
                timeout=5,
            ).read()
        )
        assert status["metrics"]["request_count"] >= 80
        assert status["metrics"]["worker_count"] == 4
        assert status["metrics"]["active_client_count"] == 4
        assert status["metrics"]["observed_client_count"] == 4
        assert status["metrics"]["active_worker_count"] == 0
        assert status["metrics"]["worker_queue_depth"] == 0
        assert status["metrics"]["max_worker_queue_depth"] == 1024
        assert (
            status["metrics"]["request_latency_count"]
            == status["metrics"]["request_count"]
        )
        assert (
            _latency_histogram_count(status)
            == status["metrics"]["request_latency_count"]
        )
        assert status["metrics"]["invalid_payload_count"] == 1
        assert status["native"] is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_native_binary_fails_loudly_for_unsupported_l2_adapter():
    binary = ensure_native_binary()
    proc = subprocess.run(
        [str(binary), "--l2-adapter", '{"type":"nixl","base_path":"/tmp/x"}'],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2
    assert "does not implement the NIXL L2 adapter yet" in proc.stderr


def test_native_binary_rejects_separate_worker_pools():
    binary = ensure_native_binary()
    proc = subprocess.run(
        [
            str(binary),
            "--l1-size-gb",
            "0.001",
            "--max-workers",
            "1",
            "--max-cpu-workers",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2
    assert "does not support separate --max-cpu-workers" in proc.stderr


def test_native_binary_rejects_unsupported_eviction_policy():
    binary = ensure_native_binary()
    proc = subprocess.run(
        [
            str(binary),
            "--l1-size-gb",
            "0.001",
            "--eviction-policy",
            "noop",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    assert "only supports --eviction-policy LRU" in proc.stderr


def test_native_binary_rejects_invalid_startup_log_level():
    binary = ensure_native_binary()
    proc = subprocess.run(
        [
            str(binary),
            "--l1-size-gb",
            "0.001",
            "--log-level",
            "verbose",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    assert "invalid native MP --log-level" in proc.stderr


def test_native_binary_uses_config_file_env_for_startup(tmp_path):
    binary = ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    file_l2_path = tmp_path / "file-l2"
    env_l2_path = tmp_path / "env-l2"
    config_file = tmp_path / "server.yaml"
    config_file.write_text(
        "\n".join(
            [
                "chunk_size: 128",
                "max_local_cpu_size: 0.001",
                "cache_policy: LRU",
                f"local_disk: {file_l2_path}",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["LMCACHE_CONFIG_FILE"] = str(config_file)
    env["LMCACHE_CHUNK_SIZE"] = "64"
    env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "0.002"
    env["LMCACHE_LOCAL_DISK"] = str(env_l2_path)
    env["LMCACHE_REMOTE_URL"] = "redis://localhost:6379"
    proc = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
        ],
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
        conf = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{http_port}/conf",
                timeout=5,
            ).read()
        )
        assert conf["mp"]["chunk_size"] == 128
        assert conf["storage_manager"]["dram_capacity_bytes"] == int(0.001 * (1 << 30))
        assert conf["mp"]["eviction_policy"] == "LRU"
        assert conf["storage_manager"]["l2_adapters"] == [
            {"type": "fs", "base_path": str(file_l2_path)}
        ]
    finally:
        _terminate_proc(proc)


def test_native_binary_rejects_unsupported_config_file_mode(tmp_path):
    binary = ensure_native_binary()
    config_file = tmp_path / "server.json"
    config_file.write_text(
        json.dumps(
            {
                "max_local_cpu_size": 0.001,
                "cache_policy": "LRU",
                "remote_url": "redis://localhost:6379",
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [str(binary), "--config-file", str(config_file)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 2
    assert "remote storage" in proc.stderr


def test_native_cli_argv_preserves_python_server_defaults(tmp_path, monkeypatch):
    fake_binary = tmp_path / "lmcache-mp-server-native-cuda"
    fake_binary.write_text("")
    monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA_BINARY", str(fake_binary))
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=1.5,
        eviction_policy="LRU",
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=False,
        native_disk_path=str(tmp_path / "native-disk"),
        l2_adapter=[],
    )

    argv = native_argv_from_args(args)

    assert argv[0] == str(fake_binary)
    assert argv[argv.index("--port") + 1] == "5555"
    assert argv[argv.index("--http-port") + 1] == "8080"
    assert argv[argv.index("--native-disk-path") + 1] == str(tmp_path / "native-disk")


def test_native_cli_argv_can_select_no_cuda_binary(tmp_path, monkeypatch):
    fake_binary = tmp_path / "lmcache-mp-server-native-no-cuda"
    fake_binary.write_text("")
    monkeypatch.setenv("LMCACHE_MP_NATIVE_BINARY", str(fake_binary))
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=1.5,
        eviction_policy="LRU",
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=True,
        native_disk_path=None,
        l2_adapter=[],
    )

    argv = native_argv_from_args(args)

    assert argv[0] == str(fake_binary)


def test_native_cli_argv_can_select_cuda_binary(tmp_path, monkeypatch):
    fake_binary = tmp_path / "lmcache-mp-server-native"
    fake_binary.write_text("")
    monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA", "1")
    monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA_BINARY", str(fake_binary))
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=1.5,
        eviction_policy="LRU",
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=False,
        native_disk_path=None,
        l2_adapter=[],
    )

    argv = native_argv_from_args(args)

    assert argv[0] == str(fake_binary)


def test_native_cli_argv_passes_log_level(tmp_path, monkeypatch):
    fake_binary = tmp_path / "lmcache-mp-server-native"
    fake_binary.write_text("")
    monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA_BINARY", str(fake_binary))
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=1.5,
        eviction_policy="LRU",
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=False,
        native_disk_path=None,
        l2_adapter=[],
        log_level="debug",
    )

    argv = native_argv_from_args(args)

    assert argv[argv.index("--log-level") + 1] == "debug"


def test_native_cli_argv_uses_supported_config_file_env(tmp_path, monkeypatch):
    fake_binary = tmp_path / "lmcache-mp-server-native"
    fake_binary.write_text("")
    disk_path = tmp_path / "l2"
    config_file = tmp_path / "server.yaml"
    config_file.write_text(
        "\n".join(
            [
                "chunk_size: 128",
                "max_local_cpu_size: 2.5",
                "cache_policy: LRU",
                f"local_disk: {disk_path}",
            ]
        )
    )
    monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA_BINARY", str(fake_binary))
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(config_file))
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=None,
        eviction_policy=None,
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=False,
        native_disk_path=None,
        l2_adapter=[],
        log_level=None,
    )

    argv = native_argv_from_args(args)

    assert argv[0] == str(fake_binary)
    assert argv[argv.index("--chunk-size") + 1] == "128"
    assert argv[argv.index("--l1-size-gb") + 1] == "2.5"
    assert argv[argv.index("--eviction-policy") + 1] == "LRU"
    assert json.loads(argv[argv.index("--l2-adapter") + 1]) == {
        "base_path": str(disk_path),
        "type": "fs",
    }


def test_native_cli_argv_config_file_precedes_engine_env(tmp_path, monkeypatch):
    fake_binary = tmp_path / "lmcache-mp-server-native"
    fake_binary.write_text("")
    file_disk_path = tmp_path / "file-l2"
    env_disk_path = tmp_path / "env-l2"
    config_file = tmp_path / "server.yaml"
    config_file.write_text(
        "\n".join(
            [
                "chunk_size: 128",
                "max_local_cpu_size: 2.5",
                "cache_policy: LRU",
                f"local_disk: {file_disk_path}",
            ]
        )
    )
    monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA_BINARY", str(fake_binary))
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "96")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "3.5")
    monkeypatch.setenv("LMCACHE_CACHE_POLICY", "noop")
    monkeypatch.setenv("LMCACHE_LOCAL_DISK", str(env_disk_path))
    monkeypatch.setenv("LMCACHE_REMOTE_URL", "redis://localhost:6379")
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=None,
        eviction_policy=None,
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=False,
        native_disk_path=None,
        l2_adapter=[],
        log_level=None,
    )

    argv = native_argv_from_args(args)

    assert argv[0] == str(fake_binary)
    assert argv[argv.index("--chunk-size") + 1] == "128"
    assert argv[argv.index("--l1-size-gb") + 1] == "2.5"
    assert argv[argv.index("--eviction-policy") + 1] == "LRU"
    assert json.loads(argv[argv.index("--l2-adapter") + 1]) == {
        "base_path": str(file_disk_path),
        "type": "fs",
    }


def test_native_cli_argv_rejects_unsupported_engine_env(tmp_path, monkeypatch):
    fake_binary = tmp_path / "lmcache-mp-server-native"
    fake_binary.write_text("")
    monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA_BINARY", str(fake_binary))
    monkeypatch.setenv("LMCACHE_REMOTE_URL", "redis://localhost:6379")
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=1.5,
        eviction_policy="LRU",
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=False,
        native_disk_path=None,
        l2_adapter=[],
        log_level=None,
    )

    with pytest.raises(ValueError, match="remote storage"):
        native_argv_from_args(args)


def test_run_native_server_strips_config_file_env_after_translation(
    tmp_path,
    monkeypatch,
):
    fake_binary = tmp_path / "lmcache-mp-server-native"
    fake_binary.write_text("")
    config_file = tmp_path / "server.yaml"
    config_file.write_text("max_local_cpu_size: 3\ncache_policy: LRU\n")
    monkeypatch.setenv("LMCACHE_MP_NATIVE_CUDA_BINARY", str(fake_binary))
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(config_file))
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=None,
        eviction_policy=None,
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=False,
        native_disk_path=None,
        l2_adapter=[],
        log_level=None,
    )
    captured: dict[str, object] = {}

    def fake_execve(
        path: str,
        argv: list[str],
        env: dict[str, str],
    ) -> None:
        captured["path"] = path
        captured["argv"] = argv
        captured["env"] = env
        raise RuntimeError("execve stopped")

    monkeypatch.setattr(os, "execve", fake_execve)

    with pytest.raises(RuntimeError, match="execve stopped"):
        native_launcher_module.run_native_server(args)

    assert captured["path"] == str(fake_binary)
    argv = cast(list[str], captured["argv"])
    env = cast(dict[str, str], captured["env"])
    assert argv[argv.index("--l1-size-gb") + 1] == "3.0"
    assert argv[argv.index("--eviction-policy") + 1] == "LRU"
    assert "LMCACHE_CONFIG_FILE" not in env


def test_ensure_native_binary_uses_packaged_binary(tmp_path, monkeypatch):
    packaged_binary = tmp_path / "lmcache-mp-server-native-cuda"
    packaged_binary.write_text("")
    monkeypatch.delenv("LMCACHE_MP_NATIVE_CUDA_BINARY", raising=False)
    monkeypatch.setattr(
        native_launcher_module,
        "packaged_native_binary_path",
        lambda *, enable_cuda=False: packaged_binary,
    )

    assert ensure_native_binary(enable_cuda=True) == packaged_binary


def test_native_cli_argv_rejects_unsupported_python_only_options():
    args = SimpleNamespace(
        host="localhost",
        port=5555,
        http_host="0.0.0.0",
        http_port=8080,
        chunk_size=256,
        l1_size_gb=1.5,
        eviction_policy="LRU",
        max_workers=2,
        max_cpu_workers=None,
        max_gpu_workers=None,
        native_cuda=False,
        native_no_cuda=False,
        native_disk_path=None,
        l2_adapter=[],
        hash_algorithm="sha256_cbor",
        engine_type="default",
        runtime_plugin_locations=[],
        runtime_plugin_config="{}",
        l1_use_lazy=True,
        l1_init_size_gb=20,
        l1_align_bytes=4096,
        l1_write_ttl_seconds=600,
        l1_read_ttl_seconds=300,
        eviction_trigger_watermark=0.8,
        eviction_ratio=0.2,
        l2_store_policy="default",
        l2_prefetch_policy="default",
        l2_prefetch_max_in_flight=8,
        disable_observability=False,
        disable_metrics=False,
        disable_logging=False,
        enable_tracing=False,
        otlp_endpoint=None,
        event_bus_queue_size=10_000,
        prometheus_port=9090,
        metrics_sample_rate=0.01,
        service_instance_id=None,
        lookup_hash_log_dir="",
        lookup_hash_log_rotation_interval=6 * 3600,
        lookup_hash_log_rotation_max_size=100 * 1024 * 1024,
        lookup_hash_log_max_files=100,
        trace_level=None,
        trace_output=None,
    )

    with pytest.raises(ValueError, match="hash-algorithm blake3"):
        native_argv_from_args(args)

    args.hash_algorithm = "blake3"
    args.engine_type = "blend"
    with pytest.raises(ValueError, match="engine-type default"):
        native_argv_from_args(args)

    args.engine_type = "default"
    args.eviction_policy = "noop"
    with pytest.raises(ValueError, match="eviction-policy LRU"):
        native_argv_from_args(args)

    args.eviction_policy = "LRU"
    args.runtime_plugin_locations = ["/tmp/plugin.py"]
    with pytest.raises(ValueError, match="runtime-plugin-locations"):
        native_argv_from_args(args)

    args.runtime_plugin_locations = []
    args.runtime_plugin_config = '{"plugin.frontend.heartbeat_url":"http://x"}'
    with pytest.raises(ValueError, match="runtime-plugin-config"):
        native_argv_from_args(args)

    args.runtime_plugin_config = "{}"
    args.max_cpu_workers = 3
    with pytest.raises(ValueError, match="separate --max-cpu-workers"):
        native_argv_from_args(args)

    args.max_cpu_workers = None
    args.max_gpu_workers = 3
    with pytest.raises(ValueError, match="separate --max-gpu-workers"):
        native_argv_from_args(args)

    args.max_gpu_workers = None
    args.native_cuda = True
    args.native_no_cuda = True
    with pytest.raises(ValueError, match="CUDA and no-CUDA"):
        native_argv_from_args(args)

    args.native_cuda = False
    args.native_no_cuda = False
    native_defaults = {
        "disable_logging": False,
        "disable_metrics": False,
        "disable_observability": False,
        "enable_tracing": False,
        "eviction_ratio": 0.2,
        "eviction_trigger_watermark": 0.8,
        "event_bus_queue_size": 10_000,
        "l1_align_bytes": 4096,
        "l1_init_size_gb": 20,
        "l1_read_ttl_seconds": 300,
        "l1_use_lazy": True,
        "l1_write_ttl_seconds": 600,
        "l2_prefetch_max_in_flight": 8,
        "l2_prefetch_policy": "default",
        "l2_store_policy": "default",
        "lookup_hash_log_dir": "",
        "lookup_hash_log_max_files": 100,
        "lookup_hash_log_rotation_interval": 6 * 3600,
        "lookup_hash_log_rotation_max_size": 100 * 1024 * 1024,
        "metrics_sample_rate": 0.01,
        "otlp_endpoint": None,
        "prometheus_port": 9090,
        "service_instance_id": None,
        "trace_level": None,
        "trace_output": None,
    }
    for attr, default_value in native_defaults.items():
        setattr(args, attr, default_value)

    unsupported_defaults = [
        ("l1_use_lazy", False, "--no-l1-use-lazy"),
        ("l1_init_size_gb", 10, "--l1-init-size-gb"),
        ("l1_align_bytes", 8192, "--l1-align-bytes"),
        ("l1_write_ttl_seconds", 120, "--l1-write-ttl-seconds"),
        ("l1_read_ttl_seconds", 60, "--l1-read-ttl-seconds"),
        ("eviction_trigger_watermark", 0.7, "--eviction-trigger-watermark"),
        ("eviction_ratio", 0.1, "--eviction-ratio"),
        ("l2_store_policy", "cleanup", "--l2-store-policy"),
        ("l2_prefetch_policy", "none", "--l2-prefetch-policy"),
        ("l2_prefetch_max_in_flight", 2, "--l2-prefetch-max-in-flight"),
        ("disable_observability", True, "--disable-observability"),
        ("disable_metrics", True, "--disable-metrics"),
        ("disable_logging", True, "--disable-logging"),
        ("enable_tracing", True, "--enable-tracing"),
        ("otlp_endpoint", "http://localhost:4317", "--otlp-endpoint"),
        ("event_bus_queue_size", 512, "--event-bus-queue-size"),
        ("prometheus_port", 9091, "--prometheus-port"),
        ("metrics_sample_rate", 0.5, "--metrics-sample-rate"),
        ("service_instance_id", "native-test", "--service-instance-id"),
        ("lookup_hash_log_dir", "/tmp/lookup-hashes", "--lookup-hash-log-dir"),
        (
            "lookup_hash_log_rotation_interval",
            60,
            "--lookup-hash-log-rotation-interval",
        ),
        (
            "lookup_hash_log_rotation_max_size",
            1024,
            "--lookup-hash-log-rotation-max-size",
        ),
        ("lookup_hash_log_max_files", 2, "--lookup-hash-log-max-files"),
        ("trace_level", "storage", "--trace-level"),
        ("trace_output", "/tmp/trace.bin", "--trace-output"),
    ]
    for attr, unsupported_value, flag in unsupported_defaults:
        setattr(args, attr, unsupported_value)
        with pytest.raises(ValueError, match=flag):
            native_argv_from_args(args)
        setattr(args, attr, native_defaults[attr])


def test_trace_replay_validates_real_vllm_checksum_match_rows(tmp_path):
    # Standard
    import sys

    response = {
        "chunk_size": 4,
        "num_chunks": 2,
        "block_id_ranges": [[1, 4], [5, 8]],
        "chunk_checksums": [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ],
    }
    row = {
        "kind": "vllm_kvcache_checksum_match",
        "model": "facebook/opt-125m",
        "raw_cuda_ipc": False,
        "use_layerwise": True,
        "requested_kv_cache_layout": "HND",
        "writer": {
            "request_type": "STORE",
            "instance_id": 123,
            "block_ids": [1, 2, 3, 4],
            "chunk_blocks": 4,
            "response": response,
        },
        "readers": [
            {
                "request_type": "RETRIEVE",
                "instance_id": 123,
                "block_ids": [1, 2, 3, 4],
                "chunk_blocks": 4,
                "response": response,
            }
        ],
    }
    trace_path = tmp_path / "vllm-checksum-trace.jsonl"
    trace_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/mp_trace_replay.py",
            "--server",
            "native",
            "--input",
            str(trace_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr


def test_trace_replay_rejects_real_vllm_checksum_mismatch_rows(tmp_path):
    # Standard
    import sys

    writer_response = {
        "chunk_size": 4,
        "num_chunks": 2,
        "block_id_ranges": [[1, 4], [5, 8]],
        "chunk_checksums": [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ],
    }
    reader_response = {
        **writer_response,
        "chunk_checksums": [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "cccccccccccccccccccccccccccccccc",
        ],
    }
    row = {
        "kind": "vllm_kvcache_checksum_match",
        "model": "facebook/opt-125m",
        "raw_cuda_ipc": False,
        "use_layerwise": True,
        "requested_kv_cache_layout": "HND",
        "writer": {
            "request_type": "STORE",
            "instance_id": 123,
            "block_ids": [1, 2, 3, 4],
            "chunk_blocks": 4,
            "response": writer_response,
        },
        "readers": [
            {
                "request_type": "RETRIEVE",
                "instance_id": 123,
                "block_ids": [1, 2, 3, 4],
                "chunk_blocks": 4,
                "response": reader_response,
            }
        ],
    }
    trace_path = tmp_path / "vllm-checksum-mismatch-trace.jsonl"
    trace_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/mp_trace_replay.py",
            "--server",
            "native",
            "--input",
            str(trace_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode != 0
    assert "checksum mismatch" in proc.stderr


def _load_server_command(monkeypatch: pytest.MonkeyPatch) -> type:
    # Standard
    import importlib.util
    import sys
    import types

    repo = Path(__file__).resolve().parents[3]
    commands_pkg = types.ModuleType("lmcache.cli.commands")
    commands_pkg.__path__ = [str(repo / "lmcache" / "cli" / "commands")]
    base_module = types.ModuleType("lmcache.cli.commands.base")

    class BaseCommand:
        pass

    base_module.__dict__["BaseCommand"] = BaseCommand
    monkeypatch.setitem(sys.modules, "lmcache.cli.commands", commands_pkg)
    monkeypatch.setitem(sys.modules, "lmcache.cli.commands.base", base_module)

    spec = importlib.util.spec_from_file_location(
        "lmcache.cli.commands.server",
        repo / "lmcache" / "cli" / "commands" / "server.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "lmcache.cli.commands.server", module)
    spec.loader.exec_module(module)
    return cast(type, module.ServerCommand)


@pytest.mark.parametrize(
    "env_name",
    [
        "LMCACHE_MP_NATIVE",
        "LMCACHE_MP_NATIVE_CUDA",
        "LMCACHE_MP_NATIVE_NO_CUDA",
    ],
)
@pytest.mark.parametrize("env_value", ["1", "true", "yes", "on"])
def test_server_command_native_env_launches_native(monkeypatch, env_name, env_value):
    ServerCommand = _load_server_command(monkeypatch)

    native_calls: list[argparse.Namespace] = []
    monkeypatch.setenv(env_name, env_value)
    monkeypatch.setattr(
        native_launcher_module,
        "run_native_server",
        lambda args: native_calls.append(args),
    )
    args = SimpleNamespace(
        native=False,
        native_cuda=False,
        native_no_cuda=False,
        python=False,
    )

    ServerCommand().execute(args)

    assert native_calls == [args]


@pytest.mark.parametrize(
    "env_name",
    [
        "LMCACHE_MP_NATIVE",
        "LMCACHE_MP_NATIVE_CUDA",
        "LMCACHE_MP_NATIVE_NO_CUDA",
    ],
)
@pytest.mark.parametrize("env_value", ["", "0", "false", "no", "off"])
def test_server_command_native_falsey_env_uses_python(
    monkeypatch,
    env_name,
    env_value,
):
    ServerCommand = _load_server_command(monkeypatch)

    # First Party
    from lmcache.v1.distributed import config as distributed_config
    from lmcache.v1.mp_observability import config as observability_config
    from lmcache.v1.multiprocess import config as multiprocess_config
    from lmcache.v1.multiprocess import http_server

    native_calls: list[argparse.Namespace] = []
    http_calls: list[dict[str, object]] = []
    monkeypatch.setenv(env_name, env_value)
    monkeypatch.setattr(
        native_launcher_module,
        "run_native_server",
        lambda args: native_calls.append(args),
    )
    monkeypatch.setattr(
        distributed_config,
        "parse_args_to_config",
        lambda args: "storage-config",
    )
    monkeypatch.setattr(
        observability_config,
        "parse_args_to_observability_config",
        lambda args: "observability-config",
    )
    monkeypatch.setattr(
        multiprocess_config,
        "parse_args_to_http_frontend_config",
        lambda args: "http-config",
    )
    monkeypatch.setattr(
        multiprocess_config,
        "parse_args_to_mp_server_config",
        lambda args: "mp-config",
    )
    monkeypatch.setattr(
        http_server,
        "run_http_server",
        lambda **kwargs: http_calls.append(kwargs),
    )
    args = SimpleNamespace(
        native=False,
        native_cuda=False,
        native_no_cuda=False,
        python=False,
    )

    ServerCommand().execute(args)

    assert native_calls == []
    assert http_calls == [
        {
            "http_config": "http-config",
            "mp_config": "mp-config",
            "storage_manager_config": "storage-config",
            "obs_config": "observability-config",
        }
    ]


def test_server_command_python_escape_overrides_native_env(monkeypatch):
    ServerCommand = _load_server_command(monkeypatch)

    # First Party
    from lmcache.v1.distributed import config as distributed_config
    from lmcache.v1.mp_observability import config as observability_config
    from lmcache.v1.multiprocess import config as multiprocess_config
    from lmcache.v1.multiprocess import http_server

    native_calls: list[argparse.Namespace] = []
    http_calls: list[dict[str, object]] = []
    monkeypatch.setenv("LMCACHE_MP_NATIVE", "1")
    monkeypatch.setattr(
        native_launcher_module,
        "run_native_server",
        lambda args: native_calls.append(args),
    )
    monkeypatch.setattr(
        distributed_config,
        "parse_args_to_config",
        lambda args: "storage-config",
    )
    monkeypatch.setattr(
        observability_config,
        "parse_args_to_observability_config",
        lambda args: "observability-config",
    )
    monkeypatch.setattr(
        multiprocess_config,
        "parse_args_to_http_frontend_config",
        lambda args: "http-config",
    )
    monkeypatch.setattr(
        multiprocess_config,
        "parse_args_to_mp_server_config",
        lambda args: "mp-config",
    )
    monkeypatch.setattr(
        http_server,
        "run_http_server",
        lambda **kwargs: http_calls.append(kwargs),
    )
    args = SimpleNamespace(
        native=False,
        native_cuda=False,
        native_no_cuda=False,
        python=True,
    )

    ServerCommand().execute(args)

    assert native_calls == []
    assert http_calls == [
        {
            "http_config": "http-config",
            "mp_config": "mp-config",
            "storage_manager_config": "storage-config",
            "obs_config": "observability-config",
        }
    ]


def test_vllm_smoke_round_summary_reports_percentiles():
    # Standard
    import importlib.util

    repo = Path(__file__).resolve().parents[3]
    script = repo / "benchmarks" / "mp_native_vs_python" / "vllm_native_smoke.py"
    spec = importlib.util.spec_from_file_location("vllm_native_smoke", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._summarize_rounds(
        [
            {
                "generate_elapsed_s": 1.0,
                "output_tokens_per_s": 10.0,
                "ttft_s_values": [0.1],
            },
            {
                "generate_elapsed_s": 2.0,
                "output_tokens_per_s": 20.0,
                "ttft_s_values": [0.2],
            },
            {
                "generate_elapsed_s": 3.0,
                "output_tokens_per_s": 30.0,
                "ttft_s_values": [0.3],
            },
        ]
    )

    assert summary["generate_elapsed_s_mean"] == 2.0
    assert summary["generate_elapsed_s_p50"] == 2.0
    assert summary["generate_elapsed_s_p95"] == 2.9
    assert summary["generate_elapsed_s"]["p99"] == 2.98
    assert summary["output_tokens_per_s_p50"] == 20.0
    assert summary["ttft_s_p95"] == pytest.approx(0.29)

    trace_summary = module._summarize_mp_trace(
        [
            {
                "phase": "submit",
                "pid": 10,
                "request_uid": 1,
                "request_type": "STORE",
                "time_s": 100.000,
                "worker_label": "writer",
            },
            {
                "phase": "response",
                "pid": 10,
                "request_uid": 1,
                "request_type": "STORE",
                "response": {"type": "tuple"},
                "time_s": 100.010,
                "worker_label": "writer",
            },
            {
                "phase": "submit",
                "pid": 11,
                "request_uid": 1,
                "request_type": "RETRIEVE",
                "time_s": 101.000,
                "worker_label": "reader",
            },
            {
                "phase": "response",
                "pid": 11,
                "request_uid": 1,
                "request_type": "RETRIEVE",
                "response": {"type": "tuple"},
                "time_s": 101.020,
                "worker_label": "reader",
            },
            {
                "phase": "submit",
                "pid": 11,
                "request_uid": 2,
                "request_type": "RETRIEVE",
                "time_s": 102.000,
                "worker_label": "reader",
            },
            {
                "phase": "response",
                "pid": 11,
                "request_uid": 2,
                "request_type": "RETRIEVE",
                "response": {"type": "tuple"},
                "time_s": 102.040,
                "worker_label": "reader",
            },
        ]
    )

    assert trace_summary["request_counts"] == {"STORE": 1, "RETRIEVE": 2}
    assert trace_summary["request_latency_ms"]["STORE"]["count"] == 1
    assert trace_summary["request_latency_ms"]["STORE"]["p50_ms"] == pytest.approx(10.0)
    assert trace_summary["request_latency_ms"]["RETRIEVE"]["count"] == 2
    assert trace_summary["request_latency_ms"]["RETRIEVE"]["p95_ms"] == pytest.approx(
        39.0
    )

    status_summary = module._summary(
        {
            "registered_context_count": 1,
            "metrics": {
                "store_count": 2,
                "retrieve_count": 3,
                "lookup_count": 4,
                "cache_hits": 6,
                "cache_misses": 2,
                "unsupported_count": 0,
            },
        }
    )

    assert status_summary["cache_hit_rate"] == 0.75


def test_controller_latency_summary_reports_percentiles_and_resources():
    # Standard
    import importlib.util

    repo = Path(__file__).resolve().parents[3]
    script = repo / "benchmarks" / "mp_native_vs_python" / "controller_latency.py"
    spec = importlib.util.spec_from_file_location("controller_latency", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._latency_summary([1.0, 2.0, 3.0])

    assert summary["count"] == 3
    assert summary["mean_ms"] == 2.0
    assert summary["p50_ms"] == 2.0
    assert summary["p95_ms"] == pytest.approx(2.9)
    assert summary["p99_ms"] == pytest.approx(2.98)
    assert summary["values_ms"] == [1.0, 2.0, 3.0]

    lookup_key = module._lookup_miss_key()
    assert lookup_key.model_name == "lmcache-controller-benchmark"
    assert lookup_key.world_size == 1
    assert lookup_key.worker_id is None
    assert lookup_key.request_id == "controller-latency-lookup-miss"
    assert len(lookup_key.token_ids) == 256

    custom_lookup_key = module._lookup_miss_key("controller-latency-custom")
    assert custom_lookup_key.request_id == "controller-latency-custom"

    fs_l2_key = module._lookup_fs_l2_partial_key()
    assert fs_l2_key.request_id == "controller-latency-l2-partial"
    assert len(fs_l2_key.token_ids) == module._CHUNK_SIZE * 2

    throughput = module._throughput_summary(
        request_count=12,
        elapsed_s=3.0,
        clients=4,
        iterations=3,
    )
    assert throughput == {
        "client_count": 4,
        "iterations_per_client": 3,
        "requests_per_s": 4.0,
        "total_elapsed_s": 3.0,
    }

    delta = module._process_resource_delta(
        {
            "rss_bytes": 100,
            "rss_peak_bytes": 100,
            "user_cpu_s": 1.0,
            "system_cpu_s": 2.0,
            "total_cpu_s": 3.0,
            "thread_count": 4,
        },
        {
            "rss_bytes": 160,
            "rss_peak_bytes": 180,
            "user_cpu_s": 1.5,
            "system_cpu_s": 2.25,
            "total_cpu_s": 3.75,
            "thread_count": 5,
        },
    )

    assert delta == {
        "rss_bytes_delta": 60,
        "rss_peak_bytes": 180,
        "user_cpu_s_delta": 0.5,
        "system_cpu_s_delta": 0.25,
        "total_cpu_s_delta": 0.75,
        "thread_count_end": 5,
    }
