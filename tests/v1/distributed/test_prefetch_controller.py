# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the PrefetchController.

Tests verify the end-to-end prefetch flow through the public interface:
submit a grouped request, lock and look up L1 and L2, plan, reserve L1
buffers, load from L2, admit the loads, and publish one bitmap per key group.

Uses a real L1Manager and MockL2Adapter (with debug methods) to exercise
the full integration without mocking internals.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    FetchingPolicy,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchResult,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.fault_inject_l2_adapter import (
    FaultInjectL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
    PrefetchPolicy,
    RetainPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    Bitmap2D,
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from tests.v1.distributed.utils import should_use_lazy_alloc

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

FULL = FULL_ATTENTION_WINDOW_CHUNKS


# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int, gid: int = 0, kv_rank: int = 0) -> ObjectKey:
    """Create a test ObjectKey for one chunk of one object group and rank."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=kv_rank,
        object_group_id=gid,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def make_group(
    keys: list[ObjectKey], gid: int = 0, window: int = FULL
) -> GroupedObjectKeys:
    """Wrap chunk-ordered keys as one key group."""
    return GroupedObjectKeys(
        keys=keys,
        object_group_id=gid,
        layout_desc=make_layout(),
        sliding_window_size=window,
    )


def make_spec(
    rows: list[GroupedObjectKeys],
    num_kv_readers: int = 1,
    fetching_policy: FetchingPolicy = "prefix",
    lock_mode: PrefetchLockMode = PrefetchLockMode.LOCK,
) -> PrefetchTaskSpec:
    """Build a request over the given rows."""
    return PrefetchTaskSpec(
        key_groups=rows,
        num_kv_readers=num_kv_readers,
        fetching_policy=fetching_policy,
        lock_mode=lock_mode,
    )


def single_row_spec(keys: list[ObjectKey], **kwargs) -> PrefetchTaskSpec:
    """Build a single full-attention row request over ``keys``."""
    return make_spec([make_group(keys)], **kwargs)


def make_l1_config(
    size_in_bytes: int = 128 * 1024 * 1024, use_lazy: bool | None = None
) -> L1ManagerConfig:
    """Create an L1 manager config of the given capacity.

    A non-lazy pool enforces ``size_in_bytes`` exactly; the default follows
    the platform's preferred allocation mode.
    """
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=size_in_bytes,
            use_lazy=should_use_lazy_alloc() if use_lazy is None else use_lazy,
            init_size_in_bytes=min(size_in_bytes, 64 * 1024 * 1024),
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


def make_adapter(bandwidth_gb: float = 10.0) -> MockL2Adapter:
    """Create a MockL2Adapter; a low bandwidth keeps loads in flight longer."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=bandwidth_gb)
    return MockL2Adapter(config)


def make_descriptor(index: int) -> L2AdapterDescriptor:
    """Create an L2AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return L2AdapterDescriptor(index=index, config=config)


def make_controller(
    l1_manager: L1Manager,
    adapters: list,
    policy: PrefetchPolicy | None = None,
    max_in_flight: int = 8,
) -> PrefetchController:
    """Build a controller over one L1 manager and the given adapters."""
    return PrefetchController(
        l1_managers=[l1_manager],
        l1_manager_descriptors=[L1ManagerDescriptor(index=0, config=make_l1_config())],
        l2_adapters=adapters,
        adapter_descriptors=[make_descriptor(i) for i in range(len(adapters))],
        policy=policy or DefaultPrefetchPolicy(),
        max_in_flight=max_in_flight,
    )


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


def wait_for_result(
    ctrl: PrefetchController,
    req_id: int,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> PrefetchResult | None:
    """Poll query_prefetch_result until it returns the result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = ctrl.query_prefetch_result(req_id)
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def row_bits(result: PrefetchResult | None, row: int = 0) -> list[int]:
    """Return the set chunk indices of one row of a result, failing on None."""
    assert result is not None, "prefetch did not complete"
    return result.hit_cells[row].get_indices_list()


def hit_counts(result: PrefetchResult | None) -> tuple[int, int]:
    """Return the (L1, L2) hit cell counts of a result, failing on None.

    Also checks the documented invariant: the two tier grids are disjoint
    and together make up ``hit_cells``.
    """
    assert result is not None, "prefetch did not complete"
    l1, l2 = Bitmap2D(result.l1_hit_cells), Bitmap2D(result.l2_hit_cells)
    assert (l1 & l2).popcount() == 0
    assert [r.get_indices_list() for r in l1 + l2] == [
        r.get_indices_list() for r in result.hit_cells
    ]
    return l1.popcount(), l2.popcount()


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


def write_keys_to_l1(
    l1_manager: L1Manager, keys: list[ObjectKey], layout: MemoryLayoutDesc
) -> None:
    """Make ``keys`` resident and unlocked in L1 as permanent objects."""
    written = l1_manager.reserve_write(
        keys, is_temporary=[False] * len(keys), layout_desc=layout
    )
    for key in keys:
        assert written[key][0] == L1Error.SUCCESS
    l1_manager.finish_write(keys)


def assert_read_locked(l1_manager: L1Manager, keys: list[ObjectKey]) -> None:
    """Assert every key is resident and holds a read lock."""
    read_results = l1_manager.unsafe_read(keys)
    for key in keys:
        assert read_results[key][0] == L1Error.SUCCESS, f"{key} not read-locked"


def assert_absent(l1_manager: L1Manager, keys: list[ObjectKey]) -> None:
    """Assert no key is resident in L1."""
    read_results = l1_manager.reserve_read(keys)
    for key in keys:
        assert read_results[key][0] == L1Error.KEY_NOT_EXIST, f"{key} is resident"


def assert_l2_unlocked(adapter: MockL2Adapter) -> None:
    """Assert the adapter holds no read lock, waiting for async unlocks."""
    ok = wait_for_condition(lambda: adapter.debug_get_locked_key_count() == 0)
    assert ok, "L2 locks should be released"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def l1_manager():
    """Create an L1Manager with a reasonable memory config."""
    mgr = L1Manager(make_l1_config())
    yield mgr
    mgr.close()


# =============================================================================
# Lifecycle
# =============================================================================


class TestLifecycle:
    def test_start_stop(self, l1_manager):
        """Controller should start and stop cleanly."""
        adapter = make_adapter()
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()
        ctrl.stop()
        adapter.close()

    def test_start_stop_no_adapters(self, l1_manager):
        """Controller should start and stop cleanly with no adapters."""
        ctrl = make_controller(l1_manager, [])
        ctrl.start()
        ctrl.stop()

    def test_report_status_keys(self, l1_manager):
        """Status reports health, queue depths, and adapter counts."""
        adapter = make_adapter()
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        status = ctrl.report_status()

        assert status["is_healthy"] is True
        assert status["num_l2_adapters"] == 1
        assert status["num_active_adapters"] == 1
        assert status["num_draining_adapters"] == 0
        assert status["in_flight_request_count"] == 0
        assert status["pending_queue_size"] == 0
        ctrl.stop()
        adapter.close()


# =============================================================================
# Single adapter
# =============================================================================


class TestSingleAdapterPrefetch:
    def test_full_prefix_hit(self, l1_manager):
        """All keys in L2 -> all loaded and read-locked."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1, 2, 3, 4]
        assert hit_counts(result) == (0, 5)
        assert_read_locked(l1_manager, keys)
        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_prefix_with_gap(self, l1_manager):
        """L2 has {0,1,3,4} -> only the prefix {0,1} is loaded."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, [keys[i] for i in (0, 1, 3, 4)], layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1]
        assert_read_locked(l1_manager, keys[:2])
        assert_absent(l1_manager, keys[2:])
        l1_manager.finish_read(keys[:2])
        ctrl.stop()
        adapter.close()

    def test_full_fetching_keeps_gaps(self, l1_manager):
        """Under "full" every found key is loaded, gaps included."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        stored = [keys[i] for i in (0, 1, 3, 4)]
        store_keys_in_l2(adapter, stored, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            single_row_spec(keys, fetching_policy="full")
        )
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1, 3, 4]
        assert_read_locked(l1_manager, stored)
        assert_absent(l1_manager, [keys[2]])
        l1_manager.finish_read(stored)
        ctrl.stop()
        adapter.close()

    def test_key0_missing(self, l1_manager):
        """L2 has {1,2,3} but not 0 -> nothing is loaded."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]
        store_keys_in_l2(adapter, keys[1:], layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == []
        assert_absent(l1_manager, keys)
        assert_l2_unlocked(adapter)
        ctrl.stop()
        adapter.close()

    def test_load_failure_truncates_prefix(self, l1_manager):
        """A mid-prefix load failure trims the hit and frees the failed
        buffer; the keys past the failure are not retained."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        inner = make_adapter()
        store_keys_in_l2(inner, keys, layout)
        fault = FaultInjectL2Adapter(inner, rate=0.0, seed=0, gap_indices=(2,))
        ctrl = make_controller(l1_manager, [fault])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1]
        assert hit_counts(result) == (0, 2)
        assert_read_locked(l1_manager, keys[:2])
        assert_absent(l1_manager, [keys[2]])
        assert l1_manager.get_staging_memory_usage() == 0
        l1_manager.finish_read(keys[:2])
        ctrl.stop()
        fault.close()

    def test_load_failure_under_full_keeps_the_rest(self, l1_manager):
        """Under "full" a failed load drops only the failed key."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        inner = make_adapter()
        store_keys_in_l2(inner, keys, layout)
        fault = FaultInjectL2Adapter(inner, rate=0.0, seed=0, gap_indices=(2,))
        ctrl = make_controller(l1_manager, [fault])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            single_row_spec(keys, fetching_policy="full")
        )
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1, 3, 4]
        held = [keys[i] for i in (0, 1, 3, 4)]
        assert_read_locked(l1_manager, held)
        assert l1_manager.get_staging_memory_usage() == 0
        l1_manager.finish_read(held)
        ctrl.stop()
        fault.close()


# =============================================================================
# Multiple adapters
# =============================================================================


class TestMultiAdapterPrefetch:
    def test_disjoint_adapters(self, l1_manager):
        """Adapter 0 has {0,1}, adapter 1 has {2,3} -> full prefix of 4."""
        adapters = [make_adapter(), make_adapter()]
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]
        store_keys_in_l2(adapters[0], keys[:2], layout)
        store_keys_in_l2(adapters[1], keys[2:], layout)
        ctrl = make_controller(l1_manager, adapters)
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1, 2, 3]
        assert_read_locked(l1_manager, keys)
        for adapter in adapters:
            assert_l2_unlocked(adapter)
        l1_manager.finish_read(keys)
        ctrl.stop()
        for a in adapters:
            a.close()

    def test_overlap_both_release_locks(self, l1_manager):
        """Both adapters hold key 1; the full prefix is loaded once and
        every adapter's locks come back."""
        adapters = [make_adapter(), make_adapter()]
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapters[0], keys[:2], layout)
        store_keys_in_l2(adapters[1], keys[1:], layout)
        ctrl = make_controller(l1_manager, adapters)
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1, 2]
        assert_read_locked(l1_manager, keys)
        for adapter in adapters:
            assert_l2_unlocked(adapter)
        assert l1_manager.get_staging_memory_usage() == 0
        l1_manager.finish_read(keys)
        ctrl.stop()
        for a in adapters:
            a.close()


# =============================================================================
# No hits and L1-only paths
# =============================================================================


class TestNoHitsAndFastPath:
    def test_no_keys_in_l2(self, l1_manager):
        """Keys in no L2 -> empty result and no locks left anywhere."""
        adapter = make_adapter()
        keys = [make_object_key(i) for i in range(3)]
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == []
        assert_l2_unlocked(adapter)
        ctrl.stop()
        adapter.close()

    def test_no_adapters_serves_l1_synchronously(self, l1_manager):
        """Without adapters the request is answered on the calling thread,
        before the loop thread is even started."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]
        write_keys_to_l1(l1_manager, keys[:2], layout)
        ctrl = make_controller(l1_manager, [])

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = ctrl.query_prefetch_result(req_id)

        assert row_bits(result) == [0, 1]
        assert hit_counts(result) == (2, 0)
        assert_read_locked(l1_manager, keys[:2])
        l1_manager.finish_read(keys[:2])

    def test_skip_l2_ignores_l2_and_answers_synchronously(self, l1_manager):
        """skip_l2 serves the L1 prefix without touching L2."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]
        store_keys_in_l2(adapter, keys, layout)
        write_keys_to_l1(l1_manager, keys[:2], layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys), skip_l2=True)
        result = ctrl.query_prefetch_result(req_id)

        assert row_bits(result) == [0, 1]
        assert adapter.debug_get_locked_key_count() == 0
        assert_absent(l1_manager, keys[2:])
        l1_manager.finish_read(keys[:2])
        ctrl.stop()
        adapter.close()

    def test_skip_l2_releases_out_of_prefix_l1_hits(self, l1_manager):
        """L1 holds {0,2}: only {0} is a prefix hit; {2} is released."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        write_keys_to_l1(l1_manager, [keys[0], keys[2]], layout)
        ctrl = make_controller(l1_manager, [])

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys), skip_l2=True)
        result = ctrl.query_prefetch_result(req_id)

        assert row_bits(result) == [0]
        assert l1_manager.is_key_evictable(keys[2])
        l1_manager.finish_read([keys[0]])


# =============================================================================
# Result queries
# =============================================================================


class TestQueryResult:
    def test_result_consumed_once(self, l1_manager):
        """The first query returns the grid; the second returns None."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1]
        assert ctrl.query_prefetch_result(req_id) is None
        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_unknown_request_returns_none(self, l1_manager):
        """Querying an id that was never submitted returns None."""
        ctrl = make_controller(l1_manager, [])
        ctrl.start()

        assert ctrl.query_prefetch_result(999) is None
        ctrl.stop()

    def test_result_has_one_row_per_group(self, l1_manager):
        """A two-row request gets a two-row result of the group width."""
        adapter = make_adapter()
        layout = make_layout()
        rows = [
            make_group([make_object_key(i, gid=0) for i in range(3)], gid=0),
            make_group([make_object_key(i, gid=1) for i in range(3)], gid=1),
        ]
        store_keys_in_l2(adapter, rows[0].keys + rows[1].keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(make_spec(rows))
        result = wait_for_result(ctrl, req_id)

        assert result is not None
        assert len(result.hit_cells) == 2
        assert all(len(row) == 3 for row in result.hit_cells)
        assert row_bits(result, 0) == [0, 1, 2]
        assert row_bits(result, 1) == [0, 1, 2]
        l1_manager.finish_read(rows[0].keys + rows[1].keys)
        ctrl.stop()
        adapter.close()


class TestWaitPrefetchResult:
    def test_wait_blocks_until_ready_without_consuming(self, l1_manager):
        """wait returns True once published and leaves the result in place."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))

        assert ctrl.wait_prefetch_result(req_id, timeout=10.0) is True
        assert row_bits(ctrl.query_prefetch_result(req_id)) == [0, 1, 2, 3, 4]
        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_wait_times_out_for_unknown_request(self, l1_manager):
        """wait returns False after genuinely waiting for an unknown id."""
        ctrl = make_controller(l1_manager, [])
        ctrl.start()

        start = time.monotonic()
        assert ctrl.wait_prefetch_result(999999, timeout=0.2) is False
        assert time.monotonic() - start >= 0.2
        ctrl.stop()


# =============================================================================
# L2 lock release
# =============================================================================


class TestL2LockRelease:
    def test_locks_released_after_prefix_trim(self, l1_manager):
        """Locks on both prefix and trimmed keys come back."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, [keys[i] for i in (0, 1, 3, 4)], layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1]
        assert_l2_unlocked(adapter)
        l1_manager.finish_read(keys[:2])
        ctrl.stop()
        adapter.close()

    def test_locks_released_for_l1_served_keys(self, l1_manager):
        """Keys L2 also holds but L1 serves are unlocked on L2 at planning."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]
        store_keys_in_l2(adapter, keys, layout)
        write_keys_to_l1(l1_manager, keys[:2], layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1, 2, 3]
        assert_l2_unlocked(adapter)
        assert_read_locked(l1_manager, keys)
        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()


# =============================================================================
# Admission and ordering
# =============================================================================


class TestAdmission:
    def test_queuing_beyond_max_in_flight(self, l1_manager):
        """More requests than max_in_flight all complete eventually."""
        adapter = make_adapter()
        layout = make_layout()
        all_keys = [make_object_key(i) for i in range(8)]
        store_keys_in_l2(adapter, all_keys, layout)
        ctrl = make_controller(l1_manager, [adapter], max_in_flight=2)
        ctrl.start()

        req_ids = [
            ctrl.submit_prefetch_request(single_row_spec(all_keys[i * 2 : i * 2 + 2]))
            for i in range(4)
        ]
        results = [wait_for_result(ctrl, r, timeout=10.0) for r in req_ids]

        assert [row_bits(r) for r in results] == [[0, 1]] * 4
        l1_manager.finish_read(all_keys)
        ctrl.stop()
        adapter.close()

    def test_two_sequential_requests(self, l1_manager):
        """Back-to-back requests complete independently."""
        adapter = make_adapter()
        layout = make_layout()
        keys1 = [make_object_key(i) for i in range(3)]
        keys2 = [make_object_key(i) for i in range(10, 14)]
        store_keys_in_l2(adapter, keys1 + keys2, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req1 = ctrl.submit_prefetch_request(single_row_spec(keys1))
        assert row_bits(wait_for_result(ctrl, req1)) == [0, 1, 2]
        req2 = ctrl.submit_prefetch_request(single_row_spec(keys2))
        assert row_bits(wait_for_result(ctrl, req2)) == [0, 1, 2, 3]

        l1_manager.finish_read(keys1)
        l1_manager.finish_read(keys2)
        ctrl.stop()
        adapter.close()

    def test_two_requests_same_keys_both_hit(self, l1_manager):
        """Two in-flight requests over the same L2 keys both hit; one copy
        is resident and read-locked once per request."""
        adapter = make_adapter(bandwidth_gb=0.01)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_a = ctrl.submit_prefetch_request(single_row_spec(keys))
        req_b = ctrl.submit_prefetch_request(single_row_spec(keys))
        result_a = wait_for_result(ctrl, req_a, timeout=15.0)
        result_b = wait_for_result(ctrl, req_b, timeout=15.0)
        ctrl.stop()
        adapter.close()

        assert row_bits(result_a) == [0, 1, 2, 3]
        assert row_bits(result_b) == [0, 1, 2, 3]
        assert l1_manager.get_staging_memory_usage() == 0
        used, _ = l1_manager.get_memory_usage()
        one_copy = l1_manager.get_object_state(keys[0]).memory_obj.get_size()
        assert used == one_copy * len(keys)
        l1_manager.finish_read(keys)
        assert_read_locked(l1_manager, keys)
        l1_manager.finish_read(keys)
        for key in keys:
            assert l1_manager.get_object_state(key) is None


# =============================================================================
# Read locks per key
# =============================================================================


class TestNumKVReaders:
    @pytest.mark.parametrize("num_kv_readers", [1, 2, 4])
    def test_loaded_keys_hold_one_lock_per_reader(self, l1_manager, num_kv_readers):
        """Keys stay readable until every reader has released its lock."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            single_row_spec(keys, num_kv_readers=num_kv_readers)
        )
        assert row_bits(wait_for_result(ctrl, req_id)) == [0, 1]

        for _ in range(num_kv_readers - 1):
            l1_manager.finish_read(keys, read_locks=1)
            assert_read_locked(l1_manager, keys)
        l1_manager.finish_read(keys, read_locks=1)
        assert_absent(l1_manager, keys)
        ctrl.stop()
        adapter.close()

    def test_l1_hits_hold_one_lock_per_reader(self, l1_manager):
        """Keys served from L1 carry the same lock count as loaded keys."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(2)]
        write_keys_to_l1(l1_manager, keys, layout)
        ctrl = make_controller(l1_manager, [])

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys, num_kv_readers=2))
        assert row_bits(ctrl.query_prefetch_result(req_id)) == [0, 1]

        l1_manager.finish_read(keys, read_locks=1)
        assert_read_locked(l1_manager, keys)
        l1_manager.finish_read(keys, read_locks=1)
        for key in keys:
            assert l1_manager.is_key_evictable(key)


# =============================================================================
# Lock mode and retention
# =============================================================================


class TestLockModeAndRetention:
    def test_no_lock_loads_unlocked_and_permanent(self, l1_manager):
        """NO_LOCK leaves loaded keys resident, unlocked, and permanent."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            single_row_spec(keys, lock_mode=PrefetchLockMode.NO_LOCK)
        )
        assert row_bits(wait_for_result(ctrl, req_id)) == [0, 1, 2]

        unsafe = l1_manager.unsafe_read(keys)
        for key in keys:
            assert unsafe[key][0] == L1Error.KEY_NOT_READABLE
        probe = l1_manager.reserve_read(keys)
        for key in keys:
            assert probe[key][0] == L1Error.SUCCESS
        l1_manager.finish_read(keys)
        again = l1_manager.reserve_read(keys)
        for key in keys:
            assert again[key][0] == L1Error.SUCCESS
        l1_manager.finish_read(keys)
        l1_manager.delete(keys)
        ctrl.stop()
        adapter.close()

    def test_no_lock_releases_l1_hits(self, l1_manager):
        """NO_LOCK reports L1 hits but leaves no lock on them."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        write_keys_to_l1(l1_manager, keys, layout)
        ctrl = make_controller(l1_manager, [])

        req_id = ctrl.submit_prefetch_request(
            single_row_spec(keys, lock_mode=PrefetchLockMode.NO_LOCK)
        )

        assert row_bits(ctrl.query_prefetch_result(req_id)) == [0, 1, 2]
        for key in keys:
            assert l1_manager.is_key_evictable(key)

    def test_default_policy_deletes_keys_after_finish_read(self, l1_manager):
        """The default policy loads temporary objects."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert row_bits(wait_for_result(ctrl, req_id)) == [0, 1, 2]

        l1_manager.finish_read(keys)
        assert_absent(l1_manager, keys)
        ctrl.stop()
        adapter.close()

    def test_retain_policy_keeps_keys_after_finish_read(self, l1_manager):
        """The retain policy loads permanent objects."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter], policy=RetainPrefetchPolicy())
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        assert row_bits(wait_for_result(ctrl, req_id)) == [0, 1, 2]

        l1_manager.finish_read(keys)
        probe = l1_manager.reserve_read(keys)
        for key in keys:
            assert probe[key][0] == L1Error.SUCCESS
        l1_manager.finish_read(keys)
        l1_manager.delete(keys)
        ctrl.stop()
        adapter.close()


# =============================================================================
# L1 and L2 together
# =============================================================================


class TestL1AndL2:
    def test_l1_suffix_extends_l2_prefix(self, l1_manager):
        """L1 has 2-4, L2 has 0-1 -> the union prefix of 5 is retained."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, keys[:2], layout)
        write_keys_to_l1(l1_manager, keys[2:], layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)
        ctrl.stop()
        adapter.close()

        assert row_bits(result) == [0, 1, 2, 3, 4]
        assert hit_counts(result) == (3, 2)
        assert result.l1_hit_cells[0].get_indices_list() == [2, 3, 4]
        assert result.l2_hit_cells[0].get_indices_list() == [0, 1]
        assert_read_locked(l1_manager, keys)
        l1_manager.finish_read(keys)

    def test_l1_resident_key_is_not_reloaded(self, l1_manager):
        """A key in both tiers is served from L1; L2 loads only the rest."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, keys, layout)
        write_keys_to_l1(l1_manager, [keys[1]], layout)
        resident = l1_manager.get_object_state(keys[1]).memory_obj
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)
        ctrl.stop()
        adapter.close()

        assert row_bits(result) == [0, 1, 2, 3, 4]
        assert l1_manager.get_object_state(keys[1]).memory_obj is resident
        assert l1_manager.get_staging_memory_usage() == 0
        assert_read_locked(l1_manager, keys)
        l1_manager.finish_read(keys)

    def test_out_of_prefix_l1_hits_are_released(self, l1_manager):
        """L1 hits past a gap are unlocked at planning and stay evictable."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, keys[:1], layout)
        write_keys_to_l1(l1_manager, keys[3:], layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)
        ctrl.stop()
        adapter.close()

        assert row_bits(result) == [0]
        for key in keys[3:]:
            assert l1_manager.is_key_evictable(key)
        l1_manager.finish_read(keys[:1])


# =============================================================================
# Hybrid attention (sliding-window rows)
# =============================================================================


class TestSlidingWindowRows:
    def test_windowed_row_keeps_only_its_window(self, l1_manager):
        """Rows [full, window 2] over 4 chunks all in L1: the full row keeps
        every chunk, the windowed row keeps chunks 2-3 and releases 0-1."""
        layout = make_layout()
        full_keys = [make_object_key(i, gid=0) for i in range(4)]
        sw_keys = [make_object_key(i, gid=1) for i in range(4)]
        write_keys_to_l1(l1_manager, full_keys + sw_keys, layout)
        ctrl = make_controller(l1_manager, [])

        req_id = ctrl.submit_prefetch_request(
            make_spec(
                [make_group(full_keys, gid=0), make_group(sw_keys, gid=1, window=2)]
            )
        )
        result = ctrl.query_prefetch_result(req_id)

        assert row_bits(result, 0) == [0, 1, 2, 3]
        assert row_bits(result, 1) == [2, 3]
        assert_read_locked(l1_manager, full_keys + sw_keys[2:])
        for key in sw_keys[:2]:
            assert l1_manager.is_key_evictable(key)
        l1_manager.finish_read(full_keys + sw_keys[2:])

    def test_windowed_row_loads_only_its_window_from_l2(self, l1_manager):
        """L1 holds chunks 0-1 of both rows, L2 the rest: the final window
        moves to chunks 4-5, so the windowed row loads only those and its
        L1 chunks are released."""
        adapter = make_adapter()
        layout = make_layout()
        full_keys = [make_object_key(i, gid=0) for i in range(6)]
        sw_keys = [make_object_key(i, gid=1) for i in range(6)]
        write_keys_to_l1(l1_manager, full_keys[:2] + sw_keys[:2], layout)
        store_keys_in_l2(adapter, full_keys[2:] + sw_keys[2:], layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            make_spec(
                [make_group(full_keys, gid=0), make_group(sw_keys, gid=1, window=2)]
            )
        )
        result = wait_for_result(ctrl, req_id)
        ctrl.stop()
        adapter.close()

        assert row_bits(result, 0) == [0, 1, 2, 3, 4, 5]
        assert row_bits(result, 1) == [4, 5]
        assert_read_locked(l1_manager, full_keys + sw_keys[4:])
        assert_absent(l1_manager, sw_keys[2:4])
        for key in sw_keys[:2]:
            assert l1_manager.is_key_evictable(key)
        assert l1_manager.get_staging_memory_usage() == 0
        l1_manager.finish_read(full_keys + sw_keys[4:])

    def test_full_fetching_refuses_windowed_rows(self, l1_manager):
        """ "full" with a sliding-window row plans nothing and leaks nothing."""
        adapter = make_adapter()
        layout = make_layout()
        full_keys = [make_object_key(i, gid=0) for i in range(3)]
        sw_keys = [make_object_key(i, gid=1) for i in range(3)]
        store_keys_in_l2(adapter, full_keys + sw_keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            make_spec(
                [make_group(full_keys, gid=0), make_group(sw_keys, gid=1, window=1)],
                fetching_policy="full",
            )
        )
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result, 0) == []
        assert row_bits(result, 1) == []
        assert_l2_unlocked(adapter)
        assert_absent(l1_manager, full_keys + sw_keys)
        ctrl.stop()
        adapter.close()


# =============================================================================
# Reservation failures
# =============================================================================


class AdmissionRacingL1Manager:
    """L1Manager wrapper emulating a concurrent writer that admits a key
    right after the controller's L1 lock pass.

    ``staged_key`` must already be staged (``reserve_write`` without
    ``finish_write``) by the test. After the first delegated
    ``reserve_read`` returns, the wrapper admits it on the inner manager, so
    the controller's subsequent ``reserve_write`` finds the key resident.
    """

    def __init__(self, inner: L1Manager, staged_key: ObjectKey) -> None:
        self._inner = inner
        self._staged_key = staged_key
        self.admitted = False

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            result = attr(*args, **kwargs)
            if name == "reserve_read" and not self.admitted:
                self.admitted = True
                self._inner.finish_write([self._staged_key])
            return result

        return wrapped


class EvictionRacingL1Manager:
    """L1Manager wrapper emulating a concurrent evictor, deterministically.

    After every delegated call returns, it attempts to evict ``target_key``
    via the public ``delete`` API, which succeeds only while the key holds no
    read or write lock. The evictor stops after its first success.
    """

    def __init__(self, inner: L1Manager, target_key: ObjectKey) -> None:
        self._inner = inner
        self._target_key = target_key
        self._evicted = False
        self.eviction_attempts: list[tuple[str, L1Error]] = []

    def _run_evictor(self, after_call: str) -> None:
        if self._evicted:
            return
        result = self._inner.delete([self._target_key])[self._target_key]
        self.eviction_attempts.append((after_call, result))
        if result == L1Error.SUCCESS:
            self._evicted = True

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            result = attr(*args, **kwargs)
            self._run_evictor(name)
            return result

        return wrapped


class TestReservationFailures:
    @pytest.mark.parametrize(
        ("fetching_policy", "expected_rows"),
        [("full", [[0, 1, 2], []]), ("prefix", [[], []])],
    )
    def test_out_of_memory_row_is_dropped(self, fetching_policy, expected_rows):
        """An L1 with room for one row's buffers but not two loads what fits
        and leaks nothing. Under "prefix" the row that could not be reserved
        empties the servable prefix, so nothing is retained."""
        layout = make_layout()
        object_bytes = 100 * 2 * 512 * 2
        l1_manager = L1Manager(
            make_l1_config(size_in_bytes=object_bytes * 4 + 65536, use_lazy=False)
        )
        adapter = make_adapter()
        rows = [
            make_group([make_object_key(i, gid=0) for i in range(3)], gid=0),
            make_group([make_object_key(i, gid=1) for i in range(3)], gid=1),
        ]
        all_keys = rows[0].keys + rows[1].keys
        store_keys_in_l2(adapter, all_keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()
        try:
            req_id = ctrl.submit_prefetch_request(
                make_spec(rows, fetching_policy=fetching_policy)
            )
            result = wait_for_result(ctrl, req_id, timeout=10.0)

            assert [row_bits(result, 0), row_bits(result, 1)] == expected_rows
            held = [
                rows[r].keys[c] for r, cols in enumerate(expected_rows) for c in cols
            ]
            if held:
                assert_read_locked(l1_manager, held)
            assert_absent(l1_manager, [k for k in all_keys if k not in held])
            assert l1_manager.get_staging_memory_usage() == 0
            assert_l2_unlocked(adapter)
            if held:
                l1_manager.finish_read(held)
        finally:
            ctrl.stop()
            adapter.close()
            l1_manager.close()

    def test_contended_key_is_dropped_under_prefix(self, l1_manager):
        """A key admitted by another writer after the lock pass fails to
        reserve; under "prefix" the hit stops before it."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)
        staged = l1_manager.reserve_write(
            [keys[1]], is_temporary=[False], layout_desc=layout
        )
        assert staged[keys[1]][0] == L1Error.SUCCESS
        racing_l1 = AdmissionRacingL1Manager(l1_manager, staged_key=keys[1])
        ctrl = make_controller(racing_l1, [adapter])  # type: ignore[arg-type]
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)
        ctrl.stop()
        adapter.close()

        assert racing_l1.admitted, "the racing writer never interleaved"
        assert row_bits(result) == [0]
        assert_read_locked(l1_manager, keys[:1])
        assert_absent(l1_manager, keys[2:])
        assert l1_manager.get_staging_memory_usage() == 0
        l1_manager.finish_read(keys[:1])
        l1_manager.delete([keys[1]])

    def test_contended_key_is_dropped_under_full(self, l1_manager):
        """Under "full" only the contended key is dropped."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)
        staged = l1_manager.reserve_write(
            [keys[0]], is_temporary=[False], layout_desc=layout
        )
        assert staged[keys[0]][0] == L1Error.SUCCESS
        racing_l1 = AdmissionRacingL1Manager(l1_manager, staged_key=keys[0])
        ctrl = make_controller(racing_l1, [adapter])  # type: ignore[arg-type]
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(
            single_row_spec(keys, fetching_policy="full")
        )
        result = wait_for_result(ctrl, req_id)
        ctrl.stop()
        adapter.close()

        assert racing_l1.admitted
        assert row_bits(result) == [1, 2]
        assert_read_locked(l1_manager, keys[1:])
        assert l1_manager.get_staging_memory_usage() == 0
        l1_manager.finish_read(keys[1:])
        l1_manager.delete([keys[0]])

    def test_l1_hit_survives_racing_evictor(self, l1_manager):
        """A resident key is locked at lookup, so an evictor cannot remove
        it and the full prefix is retained."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        store_keys_in_l2(adapter, keys, layout)
        write_keys_to_l1(l1_manager, [keys[1]], layout)
        racing_l1 = EvictionRacingL1Manager(l1_manager, target_key=keys[1])
        ctrl = make_controller(racing_l1, [adapter])  # type: ignore[arg-type]
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)
        ctrl.stop()
        adapter.close()

        assert racing_l1.eviction_attempts, "evictor never interleaved"
        assert all(err != L1Error.SUCCESS for _, err in racing_l1.eviction_attempts)
        assert row_bits(result) == [0, 1, 2, 3, 4]
        assert_read_locked(l1_manager, keys)
        l1_manager.finish_read(keys)


# =============================================================================
# Runtime adapter add and remove
# =============================================================================


class TestRuntimeAdapters:
    def test_added_adapter_serves_new_requests(self, l1_manager):
        """A request after add_adapter loads from the new adapter."""
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        ctrl = make_controller(l1_manager, [])
        ctrl.start()
        adapter = make_adapter()
        store_keys_in_l2(adapter, keys, layout)

        ctrl.add_adapter(0, adapter, make_descriptor(0))
        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == [0, 1, 2]
        assert ctrl.report_status()["num_active_adapters"] == 1
        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_remove_waits_for_in_flight_load(self, l1_manager):
        """Draining completes only after the request using the adapter has
        returned its locks, and the request still hits."""
        adapter = make_adapter(bandwidth_gb=0.001)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        # The lookup has locked the keys on L2: the request now uses the
        # adapter, and the slow load keeps it in use for a while.
        assert wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == len(keys), timeout=5.0
        )
        done = ctrl.request_remove_adapter(0)
        assert not done.wait(timeout=0.2)
        result = wait_for_result(ctrl, req_id, timeout=30.0)
        assert done.wait(timeout=5.0)

        assert row_bits(result) == [0, 1, 2]
        assert ctrl.report_status()["num_l2_adapters"] == 0
        l1_manager.finish_read(keys)
        ctrl.stop()
        adapter.close()

    def test_draining_adapter_is_not_looked_up(self, l1_manager):
        """After removal a request never routes to the removed adapter."""
        adapter = make_adapter()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        assert ctrl.request_remove_adapter(0).wait(timeout=5.0)
        req_id = ctrl.submit_prefetch_request(single_row_spec(keys))
        result = wait_for_result(ctrl, req_id)

        assert row_bits(result) == []
        assert adapter.debug_get_locked_key_count() == 0
        ctrl.stop()
        adapter.close()

    def test_double_remove_is_safe(self, l1_manager):
        """Removing an already-removed adapter signals immediately."""
        adapter = make_adapter()
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        assert ctrl.request_remove_adapter(0).wait(timeout=5.0)
        assert ctrl.request_remove_adapter(0).wait(timeout=5.0)
        ctrl.stop()
        adapter.close()


# =============================================================================
# Shutdown
# =============================================================================


class TestShutdown:
    def test_stop_releases_in_flight_locks(self, l1_manager):
        """Stopping mid-load returns L2 locks and L1 buffers."""
        adapter = make_adapter(bandwidth_gb=0.001)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, keys, layout)
        ctrl = make_controller(l1_manager, [adapter])
        ctrl.start()

        ctrl.submit_prefetch_request(single_row_spec(keys))
        # The lookup locks the keys on L2; the slow load then keeps them held.
        assert wait_for_condition(
            lambda: adapter.debug_get_locked_key_count() == len(keys), timeout=5.0
        )
        ctrl.stop()

        assert_l2_unlocked(adapter)
        assert l1_manager.get_staging_memory_usage() == 0
        assert ctrl.report_status()["in_flight_request_count"] == 0
        adapter.close()
