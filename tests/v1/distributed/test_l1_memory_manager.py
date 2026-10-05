# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for L1MemoryManager.

These tests verify the behavior of L1MemoryManager as described in the
interface docstrings. The tests focus on:

1. allocate() - Thread-safe allocation returning (error_code, memory_objs)
   - Returns SUCCESS and non-empty list on successful allocation
   - Returns OUT_OF_MEMORY and empty list when allocation fails

2. free() - Thread-safe deallocation returning error_code
   - Returns SUCCESS when operation succeeds

3. get_l1_memory_desc() - Returns L1MemoryDesc with pointer, size, and
   alignment of the L1 buffer
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    _SPDK_DEFAULT_MEM_SIZE_MB,
    HUGEPAGE_SIZE_BYTES,
    L1MemoryManagerConfig,
    _check_hugepage_availability,
    _spdk_requires_hugepages,
)
from lmcache.v1.distributed.error import L1Error
from tests.v1.distributed.utils import should_use_lazy_alloc

try:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.distributed.memory_manager import (
        L1MemoryManager,
    )
except ImportError:
    # Skip the tests if the L1MemoryManager cannot be imported
    pytest.skip("L1MemoryManager could not be imported", allow_module_level=True)

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create a basic L1MemoryManagerConfig for testing."""
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,  # 128MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,  # 4KB
    )


@pytest.fixture
def small_config():
    """Create a small L1MemoryManagerConfig to test memory exhaustion.

    Note: Minimum size is 64MB due to LazyMemoryAllocator's PIN_CHUNK_SIZE.
    """
    return L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,  # 64MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64
        * 1024
        * 1024,  # 64MB (same as final to prevent expansion)
        align_bytes=0x1000,
    )


@pytest.fixture
def non_lazy_config():
    """Create a config that does not use lazy allocation."""
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,  # 128MB
        use_lazy=False,
        init_size_in_bytes=64 * 1024 * 1024,
        align_bytes=0x1000,
    )


@pytest.fixture
def basic_layout():
    """Create a basic MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def multi_tensor_layout():
    """Create a MemoryLayoutDesc with multiple tensor shapes."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512]), torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16, torch.bfloat16],
    )


@pytest.fixture
def large_layout():
    """Create a large MemoryLayoutDesc that will exhaust small memory.

    Each allocation is 8MB (2M elements * 4 bytes).
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size([2048, 1024])],  # 2M elements * 4 bytes = 8MB
        dtypes=[torch.float32],
    )


# =============================================================================
# Tests for L1MemoryManager.allocate()
# =============================================================================


class TestAllocate:
    """
    Tests for L1MemoryManager.allocate() method.

    Per the docstring:
    - This function should be thread-safe
    - Returns tuple[L1Error, list[MemoryObj]]
    - Error code is OUT_OF_MEMORY if allocation fails, otherwise SUCCESS
    - If allocation fails, the memory object list will be empty
    """

    def test_allocate_returns_success_and_memory_objs(self, basic_config, basic_layout):
        """Test that allocate returns SUCCESS and valid memory objects."""
        manager = L1MemoryManager(basic_config)

        error, mem_objs = manager.allocate(basic_layout, count=1)

        assert error == L1Error.SUCCESS
        assert isinstance(mem_objs, list)
        assert len(mem_objs) == 1
        for obj in mem_objs:
            assert obj is not None
            assert obj.is_valid()

        manager.close()

    def test_allocate_returns_correct_count(self, basic_config, basic_layout):
        """Test that allocate returns the requested number of memory objects."""
        manager = L1MemoryManager(basic_config)
        count = 5

        error, mem_objs = manager.allocate(basic_layout, count=count)

        assert error == L1Error.SUCCESS
        assert len(mem_objs) == count

        manager.close()

    def test_allocate_with_multi_tensor_layout(self, basic_config, multi_tensor_layout):
        """Test allocation with multiple tensor shapes in the layout."""
        manager = L1MemoryManager(basic_config)

        error, mem_objs = manager.allocate(multi_tensor_layout, count=2)

        assert error == L1Error.SUCCESS
        assert len(mem_objs) == 2

        manager.close()

    def test_allocate_returns_out_of_memory_when_exhausted(
        self, small_config, large_layout
    ):
        """
        Test that allocate returns OUT_OF_MEMORY and empty list when memory
        is exhausted.
        """
        manager = L1MemoryManager(small_config)

        # Request more memory than available (8MB * 10 = 80MB > 64MB)
        error, mem_objs = manager.allocate(large_layout, count=10)

        assert error == L1Error.OUT_OF_MEMORY
        assert isinstance(mem_objs, list)
        assert len(mem_objs) == 0

        manager.close()

    def test_allocate_returns_empty_list_on_failure(self, small_config, large_layout):
        """Test that the memory object list is empty when allocation fails."""
        manager = L1MemoryManager(small_config)

        # Request more memory than available (8MB * 10 = 80MB > 64MB)
        error, mem_objs = manager.allocate(large_layout, count=10)

        # Verify the docstring contract: "If the allocation fails, the memory
        # object list will be empty."
        assert error == L1Error.OUT_OF_MEMORY
        assert mem_objs == []

        manager.close()

    def test_allocate_is_thread_safe(self, basic_config, basic_layout):
        """Test that allocate is thread-safe with concurrent allocations."""
        manager = L1MemoryManager(basic_config)
        num_threads = 10
        allocations_per_thread = 5
        results = []
        errors = []

        def allocate_task():
            try:
                for _ in range(allocations_per_thread):
                    error, mem_objs = manager.allocate(basic_layout, count=1)
                    results.append((error, mem_objs))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=allocate_task) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should have been raised
        assert len(errors) == 0, f"Thread-safety errors: {errors}"

        # All results should have valid error codes
        for error, mem_objs in results:
            assert error in (
                L1Error.SUCCESS,
                L1Error.OUT_OF_MEMORY,
            )
            if error == L1Error.SUCCESS:
                assert len(mem_objs) == 1
            else:
                assert len(mem_objs) == 0

        manager.close()

    def test_allocate_with_non_lazy_config(self, non_lazy_config, basic_layout):
        """Test allocation with non-lazy (MixedMemoryAllocator) configuration."""
        manager = L1MemoryManager(non_lazy_config)

        error, mem_objs = manager.allocate(basic_layout, count=2)

        assert error == L1Error.SUCCESS
        assert len(mem_objs) == 2

        manager.close()


# =============================================================================
# Tests for L1MemoryManager.free()
# =============================================================================


class TestFree:
    """
    Tests for L1MemoryManager.free() method.

    Per the docstring:
    - This function should be thread-safe
    - Returns L1Error indicating the result
    - Returns SUCCESS if operation succeeds
    """

    def test_free_returns_success(self, basic_config, basic_layout):
        """Test that free returns SUCCESS for valid memory objects."""
        manager = L1MemoryManager(basic_config)

        error, mem_objs = manager.allocate(basic_layout, count=2)
        assert error == L1Error.SUCCESS

        free_error = manager.free(mem_objs)

        assert free_error == L1Error.SUCCESS

        manager.close()

    def test_free_invalidates_memory_objects(self, basic_config, basic_layout):
        """Test that free invalidates the memory objects."""
        manager = L1MemoryManager(basic_config)

        error, mem_objs = manager.allocate(basic_layout, count=1)
        assert error == L1Error.SUCCESS
        assert mem_objs[0].is_valid()

        free_error = manager.free(mem_objs)
        assert free_error == L1Error.SUCCESS

        # After freeing, the memory objects should be invalidated
        for obj in mem_objs:
            assert not obj.is_valid()

        manager.close()

    def test_free_empty_list_returns_success(self, basic_config):
        """Test that freeing an empty list returns SUCCESS."""
        manager = L1MemoryManager(basic_config)

        free_error = manager.free([])

        assert free_error == L1Error.SUCCESS

        manager.close()

    def test_free_allows_reallocation(self, basic_config, basic_layout):
        """Test that freed memory can be reallocated."""
        manager = L1MemoryManager(basic_config)

        # Allocate
        error1, mem_objs1 = manager.allocate(basic_layout, count=5)
        assert error1 == L1Error.SUCCESS

        # Free
        free_error = manager.free(mem_objs1)
        assert free_error == L1Error.SUCCESS

        # Reallocate
        error2, mem_objs2 = manager.allocate(basic_layout, count=5)
        assert error2 == L1Error.SUCCESS
        assert len(mem_objs2) == 5

        manager.close()

    def test_free_is_thread_safe(self, basic_config, basic_layout):
        """Test that free is thread-safe with concurrent operations."""
        manager = L1MemoryManager(basic_config)
        num_threads = 10
        errors_list = []
        lock = threading.Lock()

        def allocate_and_free_task():
            try:
                error, mem_objs = manager.allocate(basic_layout, count=1)
                if error == L1Error.SUCCESS:
                    free_error = manager.free(mem_objs)
                    with lock:
                        errors_list.append(free_error)
            except Exception as e:
                with lock:
                    errors_list.append(e)

        threads = [
            threading.Thread(target=allocate_and_free_task) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All free operations should have succeeded
        for err in errors_list:
            if isinstance(err, Exception):
                pytest.fail(f"Thread-safety error: {err}")
            assert err == L1Error.SUCCESS

        manager.close()


# =============================================================================
# Tests for L1MemoryManager.get_l1_memory_desc()
# =============================================================================


class TestGetL1MemoryDesc:
    """
    Tests for L1MemoryManager.get_l1_memory_desc() method.

    Per the docstring:
    - Returns an L1MemoryDesc with ptr, size, and align_bytes
    - Used by RDMA communication to register the underlying virtual memory space
    """

    def test_get_l1_memory_desc_returns_desc(self, basic_config):
        """Test that get_l1_memory_desc returns an L1MemoryDesc."""
        manager = L1MemoryManager(basic_config)

        desc = manager.get_l1_memory_desc()

        assert isinstance(desc, L1MemoryDesc)

        manager.close()

    def test_get_l1_memory_desc_returns_consistent_ptr(self, basic_config):
        """Test that get_l1_memory_desc returns the same pointer on multiple calls."""
        manager = L1MemoryManager(basic_config)

        desc1 = manager.get_l1_memory_desc()
        desc2 = manager.get_l1_memory_desc()

        assert desc1.ptr == desc2.ptr

        manager.close()

    def test_get_l1_memory_desc_size_matches_config(self, basic_config):
        """Test that the desc size matches the configured size."""
        manager = L1MemoryManager(basic_config)

        desc = manager.get_l1_memory_desc()

        assert desc.size == basic_config.size_in_bytes
        assert desc.align_bytes == basic_config.align_bytes
        # The reported alignment must actually hold for the reported pointer.
        # Asserting only the advertised value passes even when the underlying
        # buffer is misaligned, which is how a 4096-byte promise was delivered
        # as a 64-byte-aligned pointer and broke O_DIRECT consumers silently.
        assert desc.ptr % desc.align_bytes == 0, (
            f"L1MemoryDesc advertises align_bytes={desc.align_bytes} but ptr "
            f"{desc.ptr:#x} is {desc.ptr % desc.align_bytes} bytes past a boundary"
        )

        manager.close()

    def test_get_l1_memory_desc_with_non_lazy_config(self, non_lazy_config):
        """Test get_l1_memory_desc with non-lazy (MixedMemoryAllocator)
        configuration."""
        manager = L1MemoryManager(non_lazy_config)

        desc = manager.get_l1_memory_desc()

        assert isinstance(desc, L1MemoryDesc)
        assert desc.ptr != 0
        assert desc.size == non_lazy_config.size_in_bytes

        manager.close()


# =============================================================================
# Tests for L1MemoryManager integration
# =============================================================================


class TestL1MemoryManagerIntegration:
    """Integration tests for L1MemoryManager."""

    def test_allocate_free_cycle(self, basic_config, basic_layout):
        """Test a complete allocate-free cycle."""
        manager = L1MemoryManager(basic_config)

        # Allocate
        error, mem_objs = manager.allocate(basic_layout, count=3)
        assert error == L1Error.SUCCESS
        assert len(mem_objs) == 3

        # Verify objects are valid
        for obj in mem_objs:
            assert obj.is_valid()

        # Free
        free_error = manager.free(mem_objs)
        assert free_error == L1Error.SUCCESS

        # Verify objects are invalidated
        for obj in mem_objs:
            assert not obj.is_valid()

        manager.close()

    def test_multiple_allocate_free_cycles(self, basic_config, basic_layout):
        """Test multiple allocate-free cycles."""
        manager = L1MemoryManager(basic_config)

        for i in range(5):
            error, mem_objs = manager.allocate(basic_layout, count=2)
            assert error == L1Error.SUCCESS, f"Cycle {i} failed"
            assert len(mem_objs) == 2

            free_error = manager.free(mem_objs)
            assert free_error == L1Error.SUCCESS

        manager.close()

    def test_interleaved_allocate_and_free(self, basic_config, basic_layout):
        """Test interleaved allocation and free operations."""
        manager = L1MemoryManager(basic_config)

        # Allocate batch 1
        err1, objs1 = manager.allocate(basic_layout, count=2)
        assert err1 == L1Error.SUCCESS

        # Allocate batch 2
        err2, objs2 = manager.allocate(basic_layout, count=2)
        assert err2 == L1Error.SUCCESS

        # Free batch 1
        free_err1 = manager.free(objs1)
        assert free_err1 == L1Error.SUCCESS

        # Allocate batch 3 (should reuse freed memory)
        err3, objs3 = manager.allocate(basic_layout, count=2)
        assert err3 == L1Error.SUCCESS

        # Free remaining
        manager.free(objs2)
        manager.free(objs3)

        manager.close()

    def test_concurrent_allocate_and_free(self, basic_config, basic_layout):
        """
        Test concurrent allocate and free operations from multiple threads.
        This verifies the thread-safety guarantees of both methods.
        """
        manager = L1MemoryManager(basic_config)
        num_threads = 8
        iterations = 20
        exceptions = []

        def worker():
            try:
                for _ in range(iterations):
                    # Allocate
                    error, mem_objs = manager.allocate(basic_layout, count=1)
                    if error == L1Error.SUCCESS:
                        # Briefly hold the memory
                        assert len(mem_objs) == 1
                        # Free
                        free_error = manager.free(mem_objs)
                        assert free_error == L1Error.SUCCESS
            except Exception as e:
                exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raises exception if worker failed

        assert len(exceptions) == 0, f"Concurrency errors: {exceptions}"

        manager.close()


# =============================================================================
# Tests for error code semantics
# =============================================================================


class TestErrorCodeSemantics:
    """Tests verifying error code semantics as documented."""

    def test_success_error_code_value(self):
        """Test that SUCCESS is a valid enum member."""
        assert L1Error.SUCCESS is not None
        assert L1Error.SUCCESS.name == "SUCCESS"

    def test_out_of_memory_error_code_value(self):
        """Test that OUT_OF_MEMORY is a valid enum member."""
        assert L1Error.OUT_OF_MEMORY is not None
        assert L1Error.OUT_OF_MEMORY.name == "OUT_OF_MEMORY"

    def test_error_codes_are_distinct(self):
        """Test that error codes are distinct values."""
        assert L1Error.SUCCESS != L1Error.OUT_OF_MEMORY


# =============================================================================
# Tests for MP-mode L1 hugepage support
# =============================================================================


def test_use_hugepages_defaults_to_false():
    """use_hugepages defaults to False so existing behavior is unchanged."""
    config = L1MemoryManagerConfig(
        size_in_bytes=1 << 30,
        use_lazy=False,
        shm_name="",
    )
    assert config.use_hugepages is False


def test_use_hugepages_accepted_with_eager_non_shm():
    """use_hugepages is accepted for a non-shared, eagerly allocated pool."""
    config = L1MemoryManagerConfig(
        size_in_bytes=1 << 30,
        use_lazy=False,
        shm_name="",
        use_hugepages=True,
    )
    assert config.use_hugepages is True


def test_use_hugepages_rejects_shared_memory():
    """use_hugepages is incompatible with a non-empty shm_name."""
    with pytest.raises(ValueError, match="incompatible with shared memory"):
        L1MemoryManagerConfig(
            size_in_bytes=1 << 30,
            use_lazy=False,
            shm_name="lmcache_l1_pool_test",
            use_hugepages=True,
        )


def test_use_hugepages_rejects_lazy_allocation():
    """use_hugepages is incompatible with lazy allocation."""
    with pytest.raises(ValueError, match="incompatible with lazy allocation"):
        L1MemoryManagerConfig(
            size_in_bytes=1 << 30,
            use_lazy=True,
            shm_name="",
            use_hugepages=True,
        )


def _patch_open(monkeypatch, contents):
    """Patch builtins.open to return queued ``contents`` strings per call.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        contents: Iterable of strings (file contents) to return in order.
    """
    calls = list(contents)
    state = {"index": 0}

    def fake_open(file, mode="r"):
        assert file == "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"
        if state["index"] >= len(calls):
            raise FileNotFoundError(file)
        content = calls[state["index"]]
        state["index"] += 1
        mock = MagicMock()
        mock.__enter__.return_value.read.return_value = content
        mock.__exit__.return_value = False
        return mock

    monkeypatch.setattr("builtins.open", fake_open)


def test_check_hugepage_availability_passes_when_sufficient(monkeypatch):
    """No error when the pool has enough free 2 MiB pages for the buffer."""
    # 1 GiB buffer needs 512 pages; pool has 1000 free.
    _patch_open(monkeypatch, ["1000"])
    _check_hugepage_availability(1 << 30)


def test_check_hugepage_availability_raises_when_insufficient(monkeypatch):
    """RuntimeError when the pool has fewer free pages than the buffer needs."""
    # 1 GiB buffer needs 512 pages; pool has only 100 free.
    _patch_open(monkeypatch, ["100"])
    with pytest.raises(RuntimeError, match="Insufficient hugepages"):
        _check_hugepage_availability(1 << 30)


def test_check_hugepage_availability_rounds_up_pages(monkeypatch):
    """A buffer that is not an exact multiple of the page size rounds up."""
    # 2 MiB + 1 byte needs 2 pages; pool has exactly 2 free.
    _patch_open(monkeypatch, ["2"])
    _check_hugepage_availability(HUGEPAGE_SIZE_BYTES + 1)


def test_check_hugepage_availability_skips_when_unavailable(monkeypatch):
    """When the 2 MiB pool is unreadable the check is a no-op (best effort)."""
    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: (_ for _ in ()).throw(
            FileNotFoundError(
                "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"
            )
        ),
    )
    # Should not raise.
    _check_hugepage_availability(1 << 30)


def test_create_memory_allocator_passes_use_hugepages(monkeypatch):
    """create_memory_allocator forwards use_hugepages to MixedMemoryAllocator.

    The allocator is stubbed so no real (hugepage) allocation happens; we only
    verify the flag is forwarded, matching the docstring contract.
    """
    # First Party
    from lmcache.v1.distributed.memory_manager import l1_memory_manager

    captured = {}

    def fake_mixed_init(self, size, **kwargs):
        captured.update(kwargs)
        self.size = size

    monkeypatch.setattr(
        l1_memory_manager.MixedMemoryAllocator,
        "__init__",
        fake_mixed_init,
    )

    config = L1MemoryManagerConfig(
        size_in_bytes=1 << 30,
        use_lazy=False,
        shm_name="",
        use_hugepages=True,
    )
    allocator = l1_memory_manager.create_memory_allocator(config)

    assert allocator is not None
    assert captured.get("use_hugepages") is True
    assert captured.get("align_bytes") == config.align_bytes


# =============================================================================
# Tests for SPDK hugepage reservation accounting
# =============================================================================


def _make_l2_config(adapters):
    """Build an L2AdaptersConfig from a list of adapter-like objects."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig

    return L2AdaptersConfig(adapters=list(adapters))


def _spdk_adapter():
    """A minimal adapter that reports the SPDK I/O engine."""
    # Standard
    import types

    return types.SimpleNamespace(io_engine="spdk")


def _posix_adapter():
    """A minimal adapter that reports the posix I/O engine."""
    # Standard
    import types

    return types.SimpleNamespace(io_engine="posix")


def test_spdk_requires_hugepages_returns_default_per_adapter():
    """SPDK reservation is 4096 MiB per SPDK adapter and 0 without SPDK.

    A truthy return also signals that hugepage allocation is required, while
    ``0`` means the pool need not be grown for SPDK.
    """
    assert _spdk_requires_hugepages(_make_l2_config([_posix_adapter()])) == 0
    assert _spdk_requires_hugepages(_make_l2_config([_spdk_adapter()])) == (
        _SPDK_DEFAULT_MEM_SIZE_MB
    )
    # Two SPDK adapters double the reservation.
    assert (
        _spdk_requires_hugepages(_make_l2_config([_spdk_adapter(), _spdk_adapter()]))
        == 2 * _SPDK_DEFAULT_MEM_SIZE_MB
    )


def test_check_hugepage_availability_fails_when_spdk_exhausts_pool(monkeypatch):
    """RuntimeError when payload + SPDK reservation exceeds the pool.

    8192 free pages. Payload needs 6145 pages and SPDK reserves 2048 pages
    (4096 MiB); the combined 8193-page demand exceeds the pool by one page.
    """
    payload_bytes = 6145 * HUGEPAGE_SIZE_BYTES
    spdk_bytes = _SPDK_DEFAULT_MEM_SIZE_MB * 1024 * 1024
    _patch_open(monkeypatch, ["8192"])
    with pytest.raises(RuntimeError, match="Insufficient hugepages"):
        _check_hugepage_availability(payload_bytes + spdk_bytes)
