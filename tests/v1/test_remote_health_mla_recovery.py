# SPDX-License-Identifier: Apache-2.0
"""Real-FS regression coverage for MLA remote-health recovery."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Optional
import asyncio
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.checks.remote_backend_check import (
    RemoteBackendHealthCheck,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector import InstrumentedRemoteConnector
from lmcache.v1.storage_backend.connector.fs_connector import FSConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

pytestmark = pytest.mark.no_shared_allocator


@pytest.fixture
def running_loop():
    """Run the real FS connector loop with bounded teardown."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        assert not thread.is_alive()
        loop.close()


@contextmanager
def _open_remote_backend(
    tmp_path: Path,
    case: str,
    use_mla: bool,
    worker_id: int,
    worker_id_as0: Optional[bool],
    loop: asyncio.AbstractEventLoop,
    remote_root: Optional[Path] = None,
    logical_name: Optional[str] = None,
) -> Iterator[tuple[RemoteBackend, LocalCPUBackend, TensorMemoryAllocator]]:
    """Construct a tiny public RemoteBackend route backed by a fresh FS root."""
    logical_name = logical_name or case
    remote_root = remote_root or (tmp_path / case)
    extra_config = {
        "fs_connector_use_odirect": False,
        "get_blocking_failed_threshold": 1,
        "waiting_time_for_recovery": 0.0,
    }
    if worker_id_as0 is not None:
        extra_config["remote_enable_mla_worker_id_as0"] = worker_id_as0

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=4,
        local_cpu=True,
        remote_url=f"fs://host:0/{remote_root}",
        remote_serde="naive",
        lmcache_instance_id=f"health-recovery-{logical_name}",
        extra_config=extra_config,
    )
    metadata = LMCacheMetadata(
        model_name=f"health-model-{logical_name}",
        world_size=2,
        local_world_size=2,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.float32,
        kv_shape=(1, 1 if use_mla else 2, 4, 1, 1),
        use_mla=use_mla,
        chunk_size=4,
    )
    allocator = TensorMemoryAllocator(torch.zeros(128 * 1024, dtype=torch.uint8))
    local_cpu_backend = LocalCPUBackend(config, metadata, memory_allocator=allocator)
    with ExitStack() as resources:
        resources.callback(local_cpu_backend.close)
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )
        resources.callback(backend.close)
        assert isinstance(backend.connection, InstrumentedRemoteConnector)
        assert isinstance(backend.connection.getWrappedConnector(), FSConnector)
        yield backend, local_cpu_backend, allocator


def _missing_key(case: str, worker_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name=f"missing-health-key-{case}",
        world_size=2,
        worker_id=worker_id,
        chunk_hash=811,
        dtype=torch.float32,
    )


def _probe_paths(backend: RemoteBackend) -> list[Path]:
    """Observe only persisted probe files, without rebuilding a probe key."""
    assert isinstance(backend.connection, InstrumentedRemoteConnector)
    connector = backend.connection.getWrappedConnector()
    assert isinstance(connector, FSConnector)
    return sorted(
        (
            path
            for base_path in connector.base_paths
            for path in base_path.glob("*.data")
        ),
        key=str,
    )


@pytest.mark.parametrize(
    "case,use_mla,worker_id,worker_id_as0",
    [
        ("mla_nonzero_default", True, 1, None),
        ("mla_worker_zero", True, 0, None),
        ("non_mla_nonzero", False, 1, None),
        ("mla_nonzero_remap_disabled", True, 1, False),
    ],
)
def test_remote_health_recovers_after_real_missing_get(
    tmp_path,
    running_loop,
    case,
    use_mla,
    worker_id,
    worker_id_as0,
):
    """A reachable backend must recover after the measured failure threshold.

    The nonzero MLA default case is intentionally the baseline RED case.  The
    three controls prove that recovery works through the same actual FS route
    when the worker is canonical, not MLA, or has canonicalization disabled.
    """
    with _open_remote_backend(
        tmp_path, case, use_mla, worker_id, worker_id_as0, running_loop
    ) as (backend, local_cpu_backend, allocator):
        health_check = RemoteBackendHealthCheck(backend)
        assert health_check.check() is True
        assert backend.get_blocking(_missing_key(case, worker_id)) is None
        assert backend.get_blocking_failed_count == 1

        assert health_check.check() is False
        assert health_check.failure_time is not None

        # The configured recovery window is zero, but check() uses a strict
        # comparison.  Keep the wait finite and independent of a scheduler.
        time.sleep(0.02)
        assert health_check.check() is True
        assert health_check.failure_time is None
        initial_probe_paths = _probe_paths(backend)
        assert len(initial_probe_paths) == 1
        assert initial_probe_paths[0].stat().st_size > 0

        if case == "mla_nonzero_default":
            assert isinstance(backend.connection, InstrumentedRemoteConnector)
            connector = backend.connection.getWrappedConnector()
            assert isinstance(connector, FSConnector)
            memory_obj = local_cpu_backend.allocate(
                connector.meta_shapes, connector.meta_dtypes, connector.meta_fmt
            )
            assert memory_obj is not None
            try:
                memory_obj.raw_tensor.fill_(7)
                ordinary_key = CacheEngineKey(
                    model_name="ordinary-nonzero-mla-put",
                    world_size=2,
                    worker_id=1,
                    chunk_hash=812,
                    dtype=torch.float32,
                )
                future = backend.submit_put_task(ordinary_key, memory_obj)
                future.result(timeout=10)
                assert backend.contains(ordinary_key) is False
                assert _probe_paths(backend) == initial_probe_paths
            finally:
                memory_obj.ref_count_down()

        recreated_check = RemoteBackendHealthCheck(backend)
        assert backend.get_blocking(_missing_key(f"{case}-repeat", worker_id)) is None
        assert recreated_check.check() is False
        assert recreated_check.failure_time is not None
        time.sleep(0.02)
        assert recreated_check.check() is True
        assert recreated_check.failure_time is None
        assert _probe_paths(backend) == initial_probe_paths
    assert allocator.total_allocated_size == 0


def test_two_mla_ranks_recover_to_distinct_shared_fs_probes(
    tmp_path, running_loop, monkeypatch
):
    """Two logical MLA ranks recover through one FS root without a shared probe."""
    observed_put_keys: list[str] = []
    original_put = FSConnector.put

    async def record_real_put(self, key, memory_obj):
        observed_put_keys.append(key.to_string())
        await original_put(self, key, memory_obj)

    monkeypatch.setattr(FSConnector, "put", record_real_put)
    shared_root = tmp_path / "shared-mla-health"
    with ExitStack() as resources:
        backend_zero, _local_zero, allocator_zero = resources.enter_context(
            _open_remote_backend(
                tmp_path,
                "shared-rank-zero",
                True,
                0,
                None,
                running_loop,
                remote_root=shared_root,
                logical_name="shared-mla-health",
            )
        )
        backend_one, _local_one, allocator_one = resources.enter_context(
            _open_remote_backend(
                tmp_path,
                "shared-rank-one",
                True,
                1,
                None,
                running_loop,
                remote_root=shared_root,
                logical_name="shared-mla-health",
            )
        )
        allocator_zero.buffer.fill_(1)
        allocator_one.buffer.fill_(2)
        check_zero = RemoteBackendHealthCheck(backend_zero)
        check_one = RemoteBackendHealthCheck(backend_one)

        assert check_zero.check() is True
        assert check_one.check() is True
        assert backend_zero.get_blocking(_missing_key("shared-zero", 0)) is None
        assert backend_one.get_blocking(_missing_key("shared-one", 1)) is None
        assert check_zero.check() is False
        assert check_one.check() is False
        assert check_zero.failure_time is not None
        assert check_one.failure_time is not None

        time.sleep(0.02)
        start_barrier = threading.Barrier(2)

        def recover(check: RemoteBackendHealthCheck) -> bool:
            start_barrier.wait(timeout=10)
            return check.check()

        with ThreadPoolExecutor(max_workers=2) as executor:
            zero_future = executor.submit(recover, check_zero)
            one_future = executor.submit(recover, check_one)
            assert zero_future.result(timeout=10) is True
            assert one_future.result(timeout=10) is True

        probe_paths = _probe_paths(backend_zero)
        assert probe_paths == _probe_paths(backend_one)
        assert len(probe_paths) == 2
        assert all(path.stat().st_size > 0 for path in probe_paths)
        assert len(observed_put_keys) == 2
        assert len(set(observed_put_keys)) == 2
    assert allocator_zero.total_allocated_size == 0
    assert allocator_one.total_allocated_size == 0
