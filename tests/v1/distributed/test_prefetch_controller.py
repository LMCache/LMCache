# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PrefetchController.

Tests verify the end-to-end prefetch flow: submit request → lookup in L2 →
compute prefix-trimmed load plan → reserve L1 buffers → load from L2 →
transition to read-locked → report prefix hits.

Uses a real L1Manager and MockL2Adapter (with debug methods) to exercise
the full integration without mocking internals.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2Adapter
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def should_use_lazy_alloc() -> bool:
    return torch.cuda.is_available()


def wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """Poll until a predicate returns True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_result(
    ctrl: PrefetchController,
    req_id: int,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> int | None:
    """Poll query_prefetch_result until it returns a non-None value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = ctrl.query_prefetch_result(req_id)
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def make_adapter() -> MockL2Adapter:
    """Create a MockL2Adapter with fast bandwidth."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return MockL2Adapter(config)


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


def store_keys_in_l2(
    adapter: MockL2Adapter,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> None:
    """Store test data directly in L2 adapter and wait for completion."""
    if not keys:
        return
    objs = []
    for _ in keys:
        tensor = torch.randn(layout.shapes[0], dtype=layout.dtypes[0])
        metadata = MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            address=0,
            phy_size=tensor.nelement() * tensor.element_size(),
            ref_count=0,
        )
        obj = TensorMemoryObj(raw_data=tensor, metadata=metadata, parent_allocator=None)
        objs.append(obj)
    adapter.submit_store_task(keys, objs)  # type: ignore
    ok = wait_for_condition(
        lambda: all(adapter.debug_has_key(k) for k in keys),
        timeout=5.0,
    )
    assert ok, "Failed to store test data in L2 adapter"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def l1_manager():
    """Create an L1Manager with a reasonable memory config."""
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestPrefetchControllerLifecycle:
    """Test PrefetchController start/stop behavior."""

    def test_start_stop(self, l1_manager):
        """Controller should start and stop cleanly."""
        adapter = make_adapter()
        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()
        ctrl.stop()
        adapter.close()

    def test_start_stop_no_adapters(self, l1_manager):
        """Controller should start and stop cleanly with no adapters."""
        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()
        ctrl.stop()


# =============================================================================
# Single Adapter Prefetch
# =============================================================================


class TestSingleAdapterPrefetch:
    """Test PrefetchController with one MockL2Adapter."""

    def test_full_prefix_hit(self, l1_manager):
        """All keys in L2 → all loaded, prefix hits = total keys."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]

        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 5, f"Expected 5 prefix hits, got {result}"

        # Verify prefix keys are read-locked in L1
        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Cleanup: release read locks
        l1_manager.finish_read(keys)

        ctrl.stop()
        adapter.close()

    def test_prefix_with_gap(self, l1_manager):
        """L2 has keys {0,1,3,4} but not 2 → only prefix {0,1} loaded."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(5)]
        # Store only keys 0, 1, 3, 4 (gap at index 2)
        stored_keys = [all_keys[i] for i in [0, 1, 3, 4]]
        store_keys_in_l2(adapter, stored_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 2, f"Expected 2 prefix hits (gap at index 2), got {result}"

        # Verify prefix keys {0,1} are read-locked in L1
        prefix_keys = all_keys[:2]
        read_results = l1_manager.unsafe_read(prefix_keys)
        for key in prefix_keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Verify keys beyond prefix {2,3,4} are NOT in L1
        non_prefix_keys = all_keys[2:]
        read_results = l1_manager.reserve_read(non_prefix_keys)
        for key in non_prefix_keys:
            assert read_results[key][0] == L1Error.KEY_NOT_EXIST

        l1_manager.finish_read(prefix_keys)
        ctrl.stop()
        adapter.close()

    def test_key0_missing(self, l1_manager):
        """L2 has keys {1,2,3} but not 0 → prefix = 0, nothing loaded."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(4)]
        # Store keys 1, 2, 3 but not 0
        stored_keys = all_keys[1:]
        store_keys_in_l2(adapter, stored_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 0, f"Expected 0 prefix hits (key 0 missing), got {result}"

        # Verify no keys are in L1
        read_results = l1_manager.reserve_read(all_keys)
        for key in all_keys:
            assert read_results[key][0] == L1Error.KEY_NOT_EXIST

        ctrl.stop()
        adapter.close()


# =============================================================================
# Multi Adapter Prefetch
# =============================================================================


class TestMultiAdapterPrefetch:
    """Test PrefetchController with multiple MockL2Adapters."""

    def test_disjoint_adapters(self, l1_manager):
        """Adapter 0 has {0,1}, adapter 1 has {2,3} → full prefix of 4."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]

        store_keys_in_l2(adapters[0], keys[:2], layout)
        store_keys_in_l2(adapters[1], keys[2:], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 4, f"Expected 4 prefix hits, got {result}"

        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        l1_manager.finish_read(keys)
        ctrl.stop()
        for a in adapters:
            a.close()

    def test_overlap_first_wins(self, l1_manager):
        """Both adapters have key 1. Adapter 0 (lower index) loads it."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        # Adapter 0 has keys {0, 1}, adapter 1 has keys {1, 2}
        store_keys_in_l2(adapters[0], keys[:2], layout)
        store_keys_in_l2(adapters[1], keys[1:], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 3, f"Expected 3 prefix hits, got {result}"

        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        l1_manager.finish_read(keys)
        ctrl.stop()
        for a in adapters:
            a.close()


# =============================================================================
# No Hits
# =============================================================================


class TestNoHits:
    """Test PrefetchController when no keys are found in L2."""

    def test_no_keys_in_l2(self, l1_manager):
        """Prefetch keys not in any L2 → 0 prefix hits."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        # Don't store anything in L2

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 0, f"Expected 0 prefix hits, got {result}"

        ctrl.stop()
        adapter.close()

    def test_no_adapters(self, l1_manager):
        """No adapters → 0 prefix hits immediately."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 0, f"Expected 0 prefix hits, got {result}"

        ctrl.stop()


# =============================================================================
# Query Result
# =============================================================================


class TestQueryResult:
    """Test query_prefetch_result semantics."""

    def test_query_returns_int_then_none(self, l1_manager):
        """Result is consumed on first query; second query returns None."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)

        assert result == 2
        # Second query should return None (already consumed)
        assert ctrl.query_prefetch_result(req_id) is None

        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_query_before_completion_returns_none(self, l1_manager):
        """Querying a nonexistent request ID returns None."""
        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        assert ctrl.query_prefetch_result(999) is None

        ctrl.stop()


# =============================================================================
# L2 Lock Release
# =============================================================================


class TestPrefetchL2LockRelease:
    """Test that L2 locks are properly released after prefetch."""

    def test_locks_released_after_full_hit(self, l1_manager):
        """All L2 locks should be released after a successful prefetch."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Wait for L2 unlock operations to be processed
        ok = wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == 0,
            timeout=5.0,
        )
        assert ok, "L2 locks should be released after prefetch completion"

        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_locks_released_after_prefix_trim(self, l1_manager):
        """L2 locks released for both prefix and non-prefix keys."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(5)]
        # Store keys 0, 1, 3, 4 (gap at index 2)
        stored_keys = [all_keys[i] for i in [0, 1, 3, 4]]
        store_keys_in_l2(adapter, stored_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 2

        # All L2 locks should be released (both prefix and trimmed keys)
        ok = wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == 0,
            timeout=5.0,
        )
        assert ok, "All L2 locks should be released after prefix-trimmed prefetch"

        l1_manager.finish_read(all_keys[:2])
        ctrl.stop()
        adapter.close()

    def test_locks_released_after_no_hits(self, l1_manager):
        """L2 locks released even when nothing is found."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        # Don't store anything

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 0

        ok = wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == 0,
            timeout=5.0,
        )
        assert ok, "L2 locks should be 0 when nothing was found"

        ctrl.stop()
        adapter.close()

    def test_multi_adapter_locks_released(self, l1_manager):
        """Both adapters' L2 locks released after overlapping prefetch."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        # Adapter 0 has {0, 1}, adapter 1 has {1, 2}
        store_keys_in_l2(adapters[0], keys[:2], layout)
        store_keys_in_l2(adapters[1], keys[1:], layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Both adapters should have all locks released
        for i, adapter in enumerate(adapters):
            ok = wait_for_condition(
                lambda a=adapter: a.debug_get_locked_key_count() == 0,
                timeout=5.0,
            )
            assert ok, f"Adapter {i} should have all L2 locks released"

        l1_manager.finish_read(keys)
        ctrl.stop()
        for a in adapters:
            a.close()


# =============================================================================
# Max In-Flight
# =============================================================================


class TestMaxInFlight:
    """Test PrefetchController max in-flight request limiting."""

    def test_queuing_beyond_max_in_flight(self, l1_manager):
        """Submit more requests than max_in_flight → all eventually complete."""
        adapter = make_adapter()
        layout = make_layout()

        # Store keys for 4 separate requests (2 keys each)
        all_keys = [make_object_key(i) for i in range(8)]
        store_keys_in_l2(adapter, all_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
            max_in_flight=2,
        )
        ctrl.start()

        # Submit 4 requests (max_in_flight=2, so 2 queued)
        req_ids = []
        for i in range(4):
            batch_keys = all_keys[i * 2 : (i + 1) * 2]
            req_id = ctrl.submit_prefetch_request(batch_keys, layout)
            req_ids.append(req_id)

        # All 4 requests should eventually complete
        results = []
        for req_id in req_ids:
            result = wait_for_prefetch_result(ctrl, req_id, timeout=10.0)
            assert result is not None, f"Request {req_id} should complete"
            results.append(result)

        assert results == [2, 2, 2, 2]

        # Release read locks for all keys
        l1_manager.finish_read(all_keys)

        ctrl.stop()
        adapter.close()


# =============================================================================
# Multiple Sequential Requests
# =============================================================================


class TestMultipleRequests:
    """Test multiple sequential prefetch requests."""

    def test_two_sequential_requests(self, l1_manager):
        """Two back-to-back requests should both complete correctly."""
        adapter = make_adapter()
        layout = make_layout()
        keys1 = [make_object_key(i) for i in range(3)]
        keys2 = [make_object_key(i) for i in range(10, 14)]

        store_keys_in_l2(adapter, keys1, layout)
        store_keys_in_l2(adapter, keys2, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req1 = ctrl.submit_prefetch_request(keys1, layout)
        result1 = wait_for_prefetch_result(ctrl, req1)
        assert result1 == 3

        req2 = ctrl.submit_prefetch_request(keys2, layout)
        result2 = wait_for_prefetch_result(ctrl, req2)
        assert result2 == 4

        l1_manager.finish_read(keys1)
        l1_manager.finish_read(keys2)
        ctrl.stop()
        adapter.close()


# =============================================================================
# extra_count Path
# =============================================================================


class TestExtraCountPrefetch:
    """Test that extra_count is correctly propagated through the prefetch path.

    When extra_count=N is passed to submit_prefetch_request, the controller
    must acquire 1 + N read locks per key (one for the prefetch controller
    itself, plus N for additional TP workers).  Each consumer must call
    finish_read (or finish_read_prefetched) once to release its lock.

    These tests verify:
    1. Keys remain accessible after the first finish_read when extra_count > 0.
    2. Keys are evictable only after ALL 1 + N locks are released.
    3. extra_count=0 (default) behaves identically to the original single-lock
       path.
    4. Prefix trimming still works correctly with extra_count > 0.
    5. Non-prefix loaded keys have all extra locks released by _finalize_load.
    """

    def test_extra_count_zero_default_behavior(self, l1_manager):
        """extra_count=0 (default): single read lock, key freed after one
        finish_read."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout, extra_count=0)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Keys should be readable immediately after prefetch
        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Release the single read lock — keys should become unlocked
        finish_results = l1_manager.finish_read(keys, extra_count=0)
        for key in keys:
            assert finish_results[key] == L1Error.SUCCESS

        ctrl.stop()
        adapter.close()

    def test_extra_count_one_requires_two_finish_reads(self, l1_manager):
        """extra_count=1: two read locks acquired; key stays locked after
        first finish_read and is released after second."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        # extra_count=1 → 2 read locks per key
        req_id = ctrl.submit_prefetch_request(keys, layout, extra_count=1)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 3

        # Keys must be readable right after prefetch
        read_results = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Release lock #1 (the "prefetch controller" lock, extra_count=0)
        finish_results = l1_manager.finish_read(keys, extra_count=0)
        for key in keys:
            assert finish_results[key] == L1Error.SUCCESS

        # Keys must STILL be readable — lock #2 (the TP worker lock) is held
        read_results2 = l1_manager.unsafe_read(keys)
        for key in keys:
            assert read_results2[key][0] == L1Error.SUCCESS, (
                f"Key {key} should still be read-locked after first finish_read"
            )

        # Release lock #2 (the TP worker lock, extra_count=0)
        finish_results2 = l1_manager.finish_read(keys, extra_count=0)
        for key in keys:
            assert finish_results2[key] == L1Error.SUCCESS

        ctrl.stop()
        adapter.close()

    def test_extra_count_three_requires_four_finish_reads(self, l1_manager):
        """extra_count=3 (TP=4): four read locks; key stays locked until all
        four are released."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        store_keys_in_l2(adapter, keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        # extra_count=3 → 4 read locks per key
        req_id = ctrl.submit_prefetch_request(keys, layout, extra_count=3)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 2

        # Release locks one by one; key must remain readable until the last
        for release_idx in range(3):
            finish_results = l1_manager.finish_read(keys, extra_count=0)
            for key in keys:
                assert finish_results[key] == L1Error.SUCCESS

            # Still readable — remaining locks are held
            read_results = l1_manager.unsafe_read(keys)
            for key in keys:
                assert read_results[key][0] == L1Error.SUCCESS, (
                    f"Key {key} should still be locked after {release_idx + 1} "
                    f"finish_read calls"
                )

        # Release the final lock
        finish_results = l1_manager.finish_read(keys, extra_count=0)
        for key in keys:
            assert finish_results[key] == L1Error.SUCCESS

        ctrl.stop()
        adapter.close()

    def test_extra_count_with_prefix_trim(self, l1_manager):
        """extra_count=1 with a gap in L2: only prefix keys get 2 locks;
        non-prefix keys are never loaded."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(5)]
        # Store keys 0, 1, 3, 4 — gap at index 2
        stored_keys = [all_keys[i] for i in [0, 1, 3, 4]]
        store_keys_in_l2(adapter, stored_keys, layout)

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(all_keys, layout, extra_count=1)
        result = wait_for_prefetch_result(ctrl, req_id)
        assert result == 2, f"Expected 2 prefix hits (gap at index 2), got {result}"

        prefix_keys = all_keys[:2]

        # Prefix keys must be readable
        read_results = l1_manager.unsafe_read(prefix_keys)
        for key in prefix_keys:
            assert read_results[key][0] == L1Error.SUCCESS

        # Release lock #1 — prefix keys still held by lock #2
        l1_manager.finish_read(prefix_keys, extra_count=0)

        read_results2 = l1_manager.unsafe_read(prefix_keys)
        for key in prefix_keys:
            assert read_results2[key][0] == L1Error.SUCCESS, (
                f"Prefix key {key} should still be locked after first finish_read"
            )

        # Release lock #2
        l1_manager.finish_read(prefix_keys, extra_count=0)

        # Non-prefix keys must NOT be in L1
        non_prefix_keys = all_keys[2:]
        reserve_results = l1_manager.reserve_read(non_prefix_keys)
        for key in non_prefix_keys:
            assert reserve_results[key][0] == L1Error.KEY_NOT_EXIST, (
                f"Non-prefix key {key} should not be in L1"
            )

        ctrl.stop()
        adapter.close()

    def test_extra_count_non_prefix_loaded_keys_fully_released(self, l1_manager):
        """Keys loaded beyond the prefix (due to partial load failure) must
        have ALL extra locks released by _finalize_load so they can be evicted.

        We simulate this by storing keys {0, 1, 2} in L2 but making key 1
        fail to reserve in L1 (by pre-occupying it), creating a gap so that
        key 2 is loaded but lies beyond the prefix.  _finalize_load must
        release 1 + extra_count locks for key 2.
        """
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)

        # Pre-occupy key[1] in L1 with a write lock so reserve_write fails for it.
        # This forces a gap: key 0 is prefix (hit), key 1 fails reservation
        # (gap), key 2 is loaded but beyond the prefix.
        pre_write_results = l1_manager.reserve_write(
            keys=[keys[1]],
            is_temporary=[False],
            layout_desc=layout,
            mode="new",
        )
        assert pre_write_results[keys[1]][0] == L1Error.SUCCESS

        ctrl = PrefetchController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultPrefetchPolicy(),
        )
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(keys, layout, extra_count=1)
        result = wait_for_prefetch_result(ctrl, req_id)
        # Only key 0 is in the prefix (key 1 reservation failed → gap)
        assert result == 1, f"Expected 1 prefix hit, got {result}"

        # key[0] should be in L1 with 2 read locks (1 + extra_count=1)
        read_results = l1_manager.unsafe_read([keys[0]])
        assert read_results[keys[0]][0] == L1Error.SUCCESS

        # key[2] was loaded but is beyond the prefix; _finalize_load must have
        # released all 1 + extra_count=2 locks, so it should be gone from L1
        # (it's a temporary object and its lock count should be 0).
        reserve_results = l1_manager.reserve_read([keys[2]])
        assert reserve_results[keys[2]][0] == L1Error.KEY_NOT_EXIST, (
            "Non-prefix loaded key[2] should have all locks released and be "
            "evicted from L1"
        )

        # Clean up: release key[0]'s 2 locks and key[1]'s write lock
        l1_manager.finish_read([keys[0]], extra_count=0)
        l1_manager.finish_read([keys[0]], extra_count=0)
        l1_manager.finish_write([keys[1]])
        l1_manager.delete([keys[1]])

        ctrl.stop()
        adapter.close()
