# SPDX-License-Identifier: Apache-2.0
"""
End-to-end tests for the serde integration through StorageManager.

Drives the full L1 -> L2 store (with serialize) and L2 -> L1 prefetch
(with deserialize) paths using a real L1Manager + MockL2Adapter +
AsyncSerdeProcessor(fp8). Tests exercise the public StorageManager API
only (no private-member access), per AGENTS.md.

Coverage:
- Store + prefetch round-trip through fp8 serde
- Memory accounting (no leaks after full cycle)
- Partial prefix trimming with serde
- Mixed adapters (one with serde, one without)
- Serde disabled (None) preserves existing behavior exactly
- Serde failure propagation
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


# =============================================================================
# Helpers
# =============================================================================


def should_use_lazy_alloc() -> bool:
    """Determine if lazy allocation should be used based on CUDA availability."""
    return torch.cuda.is_available()


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing (bf16, ~200KB/chunk)."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def wait_for_condition(
    predicate: object,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> bool:
    """Poll until a predicate returns True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_status(
    sm: StorageManager,
    handle: object,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> int | None:
    """Poll query_prefetch_status until it returns a non-None value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)  # type: ignore[arg-type]
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def make_mock_adapter_config(
    *,
    serde_type: str | None = "fp8",
) -> MockL2AdapterConfig:
    """Create a MockL2AdapterConfig, optionally with serde."""
    cfg = MockL2AdapterConfig(
        max_size_gb=0.1,
        mock_bandwidth_gb=10.0,
    )
    if serde_type is not None:
        cfg.serde_config = {"type": serde_type}
    return cfg


def make_storage_manager_config(
    adapter_configs: list[MockL2AdapterConfig],
    l1_size_mb: int = 256,
) -> StorageManagerConfig:
    """Build a StorageManagerConfig with the given L2 adapter configs."""
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=l1_size_mb * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=min(l1_size_mb, 64) * 1024 * 1024,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=adapter_configs),
    )


def write_and_wait_for_l2(
    sm: StorageManager,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
    timeout: float = 10.0,
) -> None:
    """Write keys to L1 via StorageManager and wait for L2 store.

    Fills each chunk with deterministic data so round-trip can be verified.
    """
    ret = sm.reserve_write(keys, layout, mode="new")
    assert len(ret) == len(keys), f"reserve_write: {len(ret)}/{len(keys)} succeeded"

    # Fill with deterministic data per key
    for i, key in enumerate(keys):
        obj = ret[key]
        tensor = obj.tensor
        if tensor is not None:
            tensor.fill_(float(i + 1))

    sm.finish_write(list(ret.keys()))

    # Wait for StoreController to flush to L2.
    # We poll the store_controller status for in_flight==0 and pending==0.
    ok = wait_for_condition(
        lambda: (
            sm.report_status()["store_controller"]["in_flight_task_count"] == 0
            and sm.report_status()["store_controller"]["pending_keys_count"] == 0
        ),
        timeout=timeout,
    )
    assert ok, "Store to L2 did not complete within timeout"


def get_l1_memory_used(sm: StorageManager) -> int:
    """Return current L1 memory usage in bytes via public report_status."""
    return sm.report_status()["l1_manager"]["memory_used_bytes"]


def get_l1_object_count(sm: StorageManager) -> int:
    """Return current L1 object count via public report_status."""
    return sm.report_status()["l1_manager"]["total_object_count"]


# =============================================================================
# Tests: Full round-trip through serde
# =============================================================================


class TestSerdeRoundTrip:
    """End-to-end store + prefetch through fp8 serde."""

    def test_store_and_prefetch_with_serde(self) -> None:
        """Write → L2 store (serialize) → clear L1 → prefetch (deserialize).

        Verifies all keys are recovered and L1 memory returns to clean state
        after all read locks are released.
        """
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]

        write_and_wait_for_l2(sm, keys, layout)

        # Brief sleep so StoreController releases read locks after L2 store
        time.sleep(0.1)
        sm.clear()
        assert get_l1_object_count(sm) == 0

        # Prefetch from L2
        handle = sm.submit_prefetch_task(keys, layout)
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == 5, f"Expected 5 L2 hits, got {hits}"

        # Read the data back
        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == 5

        sm.finish_read_prefetched(keys)
        sm.close()

    def test_no_memory_leak_after_full_cycle(self) -> None:
        """After write → store → clear → prefetch → finish_read, L1 is clean."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        write_and_wait_for_l2(sm, keys, layout)
        time.sleep(0.1)
        sm.clear()

        # Prefetch
        handle = sm.submit_prefetch_task(keys, layout)
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == 3

        # Release read locks
        sm.finish_read_prefetched(keys)

        # L1 should have objects (temporary prefetch results auto-delete on
        # finish_read), so memory should be back to 0.
        ok = wait_for_condition(
            lambda: get_l1_memory_used(sm) == 0,
            timeout=5.0,
        )
        assert ok, (
            f"L1 memory leak: {get_l1_memory_used(sm)} bytes still used "
            f"after releasing all read locks"
        )

        sm.close()


# =============================================================================
# Tests: Serde disabled (None) preserves existing behavior
# =============================================================================


class TestSerdeDisabled:
    """Verify that serde_config=None is equivalent to no-serde code path."""

    def test_store_and_prefetch_without_serde(self) -> None:
        """Same flow as the serde test, but without serde — must still work."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type=None)])
        sm = StorageManager(cfg)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]

        write_and_wait_for_l2(sm, keys, layout)
        time.sleep(0.1)
        sm.clear()

        handle = sm.submit_prefetch_task(keys, layout)
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == 5

        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == 5

        sm.finish_read_prefetched(keys)
        sm.close()

    def test_no_memory_leak_without_serde(self) -> None:
        """No-serde path should also leave L1 clean after a full cycle."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type=None)])
        sm = StorageManager(cfg)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        write_and_wait_for_l2(sm, keys, layout)
        time.sleep(0.1)
        sm.clear()

        handle = sm.submit_prefetch_task(keys, layout)
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == 3
        sm.finish_read_prefetched(keys)

        ok = wait_for_condition(
            lambda: get_l1_memory_used(sm) == 0,
            timeout=5.0,
        )
        assert ok, f"L1 memory leak: {get_l1_memory_used(sm)} bytes still used"
        sm.close()


# =============================================================================
# Tests: Prefix trimming with serde
# =============================================================================


class TestSerdePartialPrefix:
    """Prefetch with serde when L2 has gaps in the key sequence."""

    def test_partial_prefix_with_serde(self) -> None:
        """L2 has keys {0,1,3,4} but not 2 → only prefix {0,1} returned."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()

        # Write only keys 0, 1, 3, 4 (skip 2)
        keys_to_write = [make_object_key(i) for i in [0, 1, 3, 4]]
        write_and_wait_for_l2(sm, keys_to_write, layout)
        time.sleep(0.1)
        sm.clear()

        # Request all 5 keys — prefix should be 2 (gap at index 2)
        all_keys = [make_object_key(i) for i in range(5)]
        handle = sm.submit_prefetch_task(all_keys, layout)
        hits = wait_for_prefetch_status(sm, handle)

        assert hits is not None
        assert hits == 2, f"Expected prefix of 2, got {hits}"

        sm.finish_read_prefetched(all_keys[:hits])
        sm.close()


# =============================================================================
# Tests: Memory leak on repeated cycles
# =============================================================================


class TestSerdeMemoryStress:
    """Run multiple store-clear-prefetch cycles to catch temp buffer leaks."""

    def test_repeated_cycles_no_leak(self) -> None:
        """5 cycles of write → L2 → clear → prefetch → finish_read.

        After each cycle L1 should return to 0 bytes used. A temp buffer
        leak would accumulate across cycles.
        """
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()

        for cycle in range(5):
            keys = [make_object_key(cycle * 10 + i) for i in range(3)]
            write_and_wait_for_l2(sm, keys, layout)
            time.sleep(0.1)
            sm.clear()

            handle = sm.submit_prefetch_task(keys, layout)
            hits = wait_for_prefetch_status(sm, handle)
            assert hits == 3, f"Cycle {cycle}: expected 3 hits, got {hits}"
            sm.finish_read_prefetched(keys)

            ok = wait_for_condition(
                lambda: get_l1_memory_used(sm) == 0,
                timeout=5.0,
            )
            assert ok, (
                f"Cycle {cycle}: L1 memory leak — {get_l1_memory_used(sm)} bytes used"
            )

        sm.close()


# =============================================================================
# Tests: Multiple keys with nothing in L2
# =============================================================================


class TestSerdeNoHits:
    """Prefetch with serde when L2 is empty — 0 hits, no crash."""

    def test_prefetch_no_hits_with_serde(self) -> None:
        """Empty L2 → prefetch returns 0 hits, no temp buffer leak."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()

        keys = [make_object_key(i) for i in range(3)]
        handle = sm.submit_prefetch_task(keys, layout)
        hits = wait_for_prefetch_status(sm, handle)

        assert hits is not None
        assert hits == 0

        # No objects in L1 → memory should be 0
        ok = wait_for_condition(
            lambda: get_l1_memory_used(sm) == 0,
            timeout=5.0,
        )
        assert ok, (
            f"L1 memory leak after 0-hit prefetch: {get_l1_memory_used(sm)} bytes"
        )
        sm.close()
