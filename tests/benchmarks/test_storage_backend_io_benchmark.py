# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Optional

# Third Party
import pytest
import torch

# First Party
from benchmarks.storage_backend_io.storage_backend_io_benchmark import (
    StorageBackendBenchmark,
)
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


class _TestBenchmark(StorageBackendBenchmark):
    @property
    def extra_config_keys(self) -> dict:
        return {}

    def _create_backend(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: Any,
        local_cpu_backend: LocalCPUBackend,
    ) -> StorageBackendInterface:
        raise NotImplementedError

    def _close_backend(self) -> None:
        return None


class _LoadedObject:
    def __init__(self) -> None:
        self.released = False

    def ref_count_down(self) -> None:
        self.released = True


class _ResultBackend:
    def __init__(self, results: list[Optional[MemoryObj]]) -> None:
        self.results = results

    def batched_get_blocking(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        del keys
        return self.results


class _ErrorBackend:
    def batched_get_blocking(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        raise ValueError(f"worker-{keys[0].chunk_hash}")


def _make_benchmark(num_ops: int, concurrency: int) -> _TestBenchmark:
    benchmark = _TestBenchmark(
        "test",
        num_ops=num_ops,
        concurrency=concurrency,
        use_odirect=False,
        alignment=4096,
        write_bench=False,
    )
    benchmark._keys = [
        CacheEngineKey("model", 1, 0, index, torch.bfloat16) for index in range(num_ops)
    ]
    return benchmark


@pytest.mark.parametrize("returned_count", [0, 1])
def test_read_phase_rejects_empty_or_partial_results(returned_count: int) -> None:
    benchmark = _make_benchmark(num_ops=2, concurrency=1)
    returned = [_LoadedObject() for _ in range(returned_count)]
    benchmark._backend = _ResultBackend(returned)  # type: ignore[assignment]

    with pytest.raises(
        RuntimeError,
        match=f"missing_operations={2 - returned_count}",
    ):
        benchmark._execute_read_phase()

    assert all(obj.released for obj in returned)


def test_read_phase_collects_all_worker_exceptions() -> None:
    benchmark = _make_benchmark(num_ops=4, concurrency=2)
    benchmark._backend = _ErrorBackend()  # type: ignore[assignment]

    with pytest.raises(RuntimeError) as exc_info:
        benchmark._execute_read_phase()

    message = str(exc_info.value)
    assert "missing_operations=4" in message
    assert "worker-0" in message
    assert "worker-2" in message
