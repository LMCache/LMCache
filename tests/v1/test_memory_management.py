# SPDX-License-Identifier: Apache-2.0
# Standard
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    GPUMemoryAllocator,
    HostMemoryAllocator,
    MemoryFormat,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    PinMemoryAllocator,
    TensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.pin_monitor import PinMonitor


def check_allocator(allocator, max_size):
    # 512 * 512 * 4 = 1MB
    shape1 = torch.Size([512, 512])
    data1 = allocator.allocate(shape1, torch.float)
    assert data1 is not None
    assert data1.tensor.dtype == torch.float
    assert data1.tensor.shape == shape1

    # 1024 * 1024 * 2 = 2MB
    shape2 = torch.Size([1024, 1024])
    data2 = allocator.allocate(shape2, torch.bfloat16)
    assert data2 is not None
    assert data2.tensor.dtype == torch.bfloat16
    assert data2.tensor.shape == shape2

    # 2048 * 2048 * 1 = 4MB
    shape3 = torch.Size([2048, 2048])
    data3 = allocator.allocate(shape3, torch.int8)
    assert data3 is not None
    assert data3.tensor.dtype == torch.int8
    assert data3.tensor.shape == shape3

    allocator.free(data2)
    assert data2.tensor is None
    assert allocator.memcheck()

    allocator.free(data1)
    assert data1.tensor is None
    assert allocator.memcheck()

    allocator.free(data2)  # This should not crash

    shape4 = torch.Size([3, 5, 7])
    data4 = allocator.allocate(shape4, torch.half)
    assert data4 is not None
    assert data4.tensor.dtype == torch.half
    assert data4.tensor.shape == shape4

    data_fail = allocator.allocate(
        torch.Size([max_size]), torch.float
    )  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.free(data1)
    allocator.free(data2)
    allocator.free(data3)
    allocator.free(data4)

    assert allocator.memcheck()

    allocator.close()


def check_paged_allocator(allocator, shape, dtype, fmt, max_num_pages):
    # Allocate one page
    data1 = allocator.allocate(shape, dtype, fmt)
    assert data1 is not None
    assert data1.tensor.dtype == dtype
    assert data1.tensor.shape == shape

    # Allocate another 2 pages
    data2 = allocator.batched_allocate(shape, dtype, 2, fmt)

    for data in data2:
        assert data is not None
        assert data.tensor.dtype == dtype
        assert data.tensor.shape == shape

    # Allocate a smaller page
    smaller_shape = torch.Size([2, 32, 8, 1024])
    data3 = allocator.allocate(smaller_shape, dtype, fmt)
    assert data3 is not None
    assert data3.tensor.dtype == dtype
    assert data3.tensor.shape == smaller_shape

    allocator.free(data3)
    assert allocator.memcheck()

    allocator.batched_free(data2)
    assert allocator.memcheck()

    allocator.free(data1)
    assert allocator.memcheck()

    data_fail = allocator.batched_allocate(
        shape, dtype, max_num_pages + 1, fmt
    )  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.close()


@pytest.mark.parametrize(
    "use_paging",
    [True, False],
)
def test_tensor_allocator(use_paging):
    total_size = 1024 * 1024 * 128  # 128MB
    tensor_buffer = torch.zeros(total_size, dtype=torch.uint8, device="cpu")
    if use_paging:
        shape = torch.Size([2, 32, 16, 1024])  # 64 pages
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_2LTD
        num_pages = 64
        allocator = PagedTensorMemoryAllocator(tensor_buffer, [shape], [dtype], fmt)
        check_paged_allocator(allocator, shape, dtype, fmt, num_pages)
    else:
        allocator = TensorMemoryAllocator(tensor_buffer)
        check_allocator(allocator, total_size)

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
@pytest.mark.parametrize(
    "use_paging",
    [
        False,
        True,
    ],
)
def test_device_allocators(alloc_cls, use_paging):
    total_size = 1024 * 1024 * 128  # 128MB

    shape = torch.Size([2, 32, 16, 1024])  # 64 pages
    dtype = torch.bfloat16
    fmt = MemoryFormat.KV_2LTD

    allocator = alloc_cls(
        total_size, use_paging=use_paging, shapes=[shape], dtypes=[dtype], fmt=fmt
    )

    if use_paging:
        num_pages = 64
        check_paged_allocator(allocator, shape, dtype, fmt, num_pages)
    else:
        check_allocator(allocator, total_size)

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_inplace_modification(alloc_cls):
    total_size = 1024 * 1024
    allocator = alloc_cls(total_size)

    shape = torch.Size([4096])
    data = allocator.allocate(shape, torch.float)
    assert data is not None
    assert data.tensor.dtype == torch.float
    assert data.tensor.shape == shape

    data.tensor.fill_(1.0)
    assert torch.all(data.tensor == 1.0)

    data.tensor[1] = 2.0
    assert data.tensor[1] == 2.0

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_boundary_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)

    shape = torch.Size([512, 10])
    data1 = allocator.allocate(shape, torch.float)
    allocator.allocate(shape, torch.float)
    allocator.free(data1)

    # `FreeBlock` with size 0 shouldn't exist in the allocator
    allocator.allocate(shape, torch.float)

    assert allocator.memcheck()
    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_batched_alloc(alloc_cls):
    total_size = 32 * 100 * 2 * 1024 * 2
    batch_size = 32
    allocator = alloc_cls(total_size)
    shape = torch.Size([100, 2, 1024])
    objs = allocator.batched_allocate(
        shape, torch.bfloat16, batch_size, MemoryFormat.KV_T2D
    )

    assert len(objs) == batch_size
    for obj in objs:
        assert obj is not None
        assert obj.tensor is not None
        assert obj.tensor.dtype == torch.bfloat16
        assert obj.tensor.shape == shape
    allocator.batched_free(objs)

    assert allocator.memcheck()
    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        MixedMemoryAllocator,
    ],
)
def test_mixed_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    shape = torch.Size([512, 10])
    data1 = allocator.allocate(shape, [], MemoryFormat.BINARY_BUFFER)
    allocator.allocate(shape, torch.float)
    allocator.free(data1)

    assert isinstance(data1, BytesBufferMemoryObj)

    assert len(data1.byte_array) == 512

    allocator.memcheck()
    allocator.close()


def test_memory_obj_metadata_to_and_from_dict():
    shape1 = torch.Size([128, 10])
    dtype1 = torch.float
    shape2 = torch.Size([256, 10])
    dtype2 = torch.uint8
    shapes = [shape1, shape2]
    dtypes = [dtype1, dtype2]
    metadata1 = MemoryObjMetadata(
        shape=shape1,
        dtype=dtype1,
        address=0,
        phy_size=0,
        ref_count=0,
        pin_count=0,
        fmt=MemoryFormat.KV_T2D,
    )
    dict1 = metadata1.to_dict()
    metadata_from_dict_1 = MemoryObjMetadata.from_dict(dict1)
    assert metadata_from_dict_1.shape == shape1
    assert metadata_from_dict_1.dtype == dtype1
    assert metadata_from_dict_1.shapes is None
    assert metadata_from_dict_1.dtypes is None

    metadata2 = MemoryObjMetadata(
        shape=shape1,
        dtype=dtype1,
        address=0,
        phy_size=0,
        ref_count=0,
        pin_count=0,
        fmt=MemoryFormat.KV_T2D,
        shapes=shapes,
        dtypes=dtypes,
    )
    dict2 = metadata2.to_dict()
    metadata_from_dict_2 = MemoryObjMetadata.from_dict(dict2)
    assert metadata_from_dict_2.shape == shape1
    assert metadata_from_dict_2.dtype == dtype1
    assert metadata_from_dict_2.shapes == shapes
    assert metadata_from_dict_2.dtypes == dtypes


@pytest.mark.parametrize(
    "alloc_cls,custom_timeout,elapsed_time",
    [
        (HostMemoryAllocator, None, 360),
        (PinMemoryAllocator, None, 360),
        (GPUMemoryAllocator, None, 360),
        (MixedMemoryAllocator, None, 360),
        (HostMemoryAllocator, 60, 90),
    ],
)
def test_pin_timeout(alloc_cls, custom_timeout, elapsed_time):
    # Reset the singleton to ensure clean state
    LMCStatsMonitor.DestroyInstance()
    # Also reset the class variable to use the new singleton
    TensorMemoryObj.monitor = LMCStatsMonitor.GetOrCreate()

    # Reset and initialize PinMonitor
    PinMonitor._instance = None
    config = LMCacheEngineConfig.from_defaults()
    PinMonitor.GetOrCreate(config)

    try:
        total_size = 1024 * 1024
        allocator = alloc_cls(total_size)

        # Create a memory object
        data = allocator.allocate(torch.Size([4096]), torch.float)
        assert data is not None

        # Pin the object
        data.pin()
        assert data.metadata.pin_count == 1

        # Get initial forced unpin count
        monitor = LMCStatsMonitor.GetOrCreate()
        initial_forced_unpin_count = monitor.interval_forced_unpin_count

        # Get the PinMonitor instance that was used by pin()
        pin_monitor = PinMonitor.GetOrCreate()

        # Override timeout if custom timeout is specified
        if custom_timeout is not None:
            pin_monitor._pin_timeout_sec = custom_timeout

        # Simulate timeout by manually setting register time in PinMonitor
        obj_id = id(data)
        with pin_monitor._objects_lock:
            if obj_id in pin_monitor._pinned_objects:
                memory_obj, _ = pin_monitor._pinned_objects[obj_id]
                pin_monitor._pinned_objects[obj_id] = (
                    memory_obj,
                    time.time() - elapsed_time,
                )

        # Force a timeout check
        pin_monitor._check_timeouts()

        # Verify that pin_count is now 0
        assert data.metadata.pin_count == 0

        # Verify that forced unpin count increased
        assert monitor.interval_forced_unpin_count == initial_forced_unpin_count + 1

        allocator.close()
    finally:
        pass


def test_pin_monitor_timeout():
    """Test that PinMonitor correctly detects and handles pin timeouts."""

    # Create a mock memory object for testing
    class MockMemoryObjMetadata:
        def __init__(self):
            self.address = 12345
            self.pin_count = 0
            self.ref_count = 1

    class MockMemoryObj:
        def __init__(self):
            self.meta = MockMemoryObjMetadata()
            self.lock = threading.Lock()
            self.parent_allocator = None

        def unpin(self):
            self.meta.pin_count -= 1
            if self.meta.pin_count == 0:
                PinMonitor.GetOrCreate().on_unpin(self)
            if self.meta.pin_count < 0:
                self.meta.pin_count = 0

    # Reset PinMonitor singleton for testing
    PinMonitor._instance = None

    # Create PinMonitor with short timeout for testing
    config = LMCacheEngineConfig.from_defaults(
        pin_timeout_sec=1, pin_check_interval_sec=1
    )
    pin_monitor = PinMonitor.GetOrCreate(config)

    # Create a mock memory object
    mock_obj = MockMemoryObj()

    # Test registration
    pin_monitor.on_pin(mock_obj)
    assert pin_monitor.get_monitored_count() == 1

    # Test unregistration
    pin_monitor.on_unpin(mock_obj)
    assert pin_monitor.get_monitored_count() == 0

    # Test timeout detection
    try:
        # Register object first
        mock_obj.meta.pin_count = 1
        pin_monitor.on_pin(mock_obj)

        # Manually set old register time to simulate timeout
        # Set to 2 seconds ago to exceed the 1 second timeout
        obj_id = id(mock_obj)
        with pin_monitor._objects_lock:
            if obj_id in pin_monitor._pinned_objects:
                memory_obj, _ = pin_monitor._pinned_objects[obj_id]
                pin_monitor._pinned_objects[obj_id] = (
                    memory_obj,
                    time.time() - 2.0,
                )

        # Force a timeout check
        pin_monitor._check_timeouts()

        # Verify object was unpinned
        assert mock_obj.meta.pin_count == 0
        assert pin_monitor.get_monitored_count() == 0

    finally:
        pass


def test_pin_monitor_background_thread():
    """Test that PinMonitor background thread starts correctly."""
    # Reset singleton and create with config
    PinMonitor._instance = None
    config = LMCacheEngineConfig.from_defaults()
    pin_monitor = PinMonitor.GetOrCreate(config)

    # PinMonitor auto-starts in __init__, so it should already be running
    # PinMonitor now inherits from PeriodicThread, use is_running property
    assert pin_monitor.is_running
    assert pin_monitor._thread is not None
    assert pin_monitor._thread.is_alive()

    # Give thread a moment to start
    time.sleep(0.1)

    # Test basic functionality without stopping the thread
    # (thread stopping is handled by daemon thread behavior)


def test_tensor_memory_obj_pin_monitor_integration():
    """Test integration between TensorMemoryObj and PinMonitor."""

    # Create a simple allocator for testing
    class MockAllocator:
        def free(self, obj):
            pass

    # Create a real TensorMemoryObj
    raw_data = torch.empty(100, dtype=torch.float32)
    metadata = MemoryObjMetadata(
        shape=torch.Size([100]),
        dtype=torch.float32,
        address=12345,
        phy_size=400,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )

    allocator = MockAllocator()
    memory_obj = TensorMemoryObj(raw_data, metadata, allocator)

    # Get PinMonitor instance
    pin_monitor = PinMonitor.GetOrCreate()
    initial_count = pin_monitor.get_monitored_count()

    # Test pinning registers with PinMonitor
    memory_obj.pin()
    assert pin_monitor.get_monitored_count() == initial_count + 1

    # Test unpinning unregisters from PinMonitor
    memory_obj.unpin()
    assert pin_monitor.get_monitored_count() == initial_count

    # Test multiple pins/unpins
    memory_obj.pin()
    memory_obj.pin()  # Pin twice
    assert pin_monitor.get_monitored_count() == initial_count + 1

    memory_obj.unpin()
    assert pin_monitor.get_monitored_count() == initial_count + 1  # Still monitored

    memory_obj.unpin()
    assert pin_monitor.get_monitored_count() == initial_count  # Fully unregistered


# =============================================================================
# LazyMemoryAllocator Tests
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LazyMemoryAllocator requires CUDA for memory pinning",
)
class TestLazyMemoryAllocator:
    """
    Test suite for LazyMemoryAllocator.

    These tests focus on the public interface defined by MemoryAllocatorInterface:
    - allocate(shapes, dtypes, fmt, allocator_type) -> Optional[MemoryObj]
    - batched_allocate(shapes, dtypes, batch_size, fmt, allocator_type)
        -> Optional[List[MemoryObj]]
    - free(memory_obj, allocator_type)
    - batched_free(memory_objs, allocator_type, update_stats)
    - close()
    - memcheck() -> bool
    """

    # Use sizes that are multiples of PIN_CHUNK_SIZE (16 MB)
    INIT_SIZE = 1 << 25  # 32 MB
    FINAL_SIZE = 1 << 27  # 128 MB

    @pytest.fixture
    def lazy_allocator_cls(self):
        """Lazily import LazyMemoryAllocator to avoid import errors
        on CPU-only builds.
        """
        # First Party
        from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator

        return LazyMemoryAllocator

    def test_allocate_basic(self, lazy_allocator_cls):
        """Test basic allocation returns a valid MemoryObj."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([512, 512])
        dtype = torch.float32
        memory_obj = allocator.allocate(shape, dtype)

        assert memory_obj is not None
        assert memory_obj.is_valid()
        assert memory_obj.tensor is not None
        assert memory_obj.tensor.shape == shape
        assert memory_obj.tensor.dtype == dtype

        allocator.close()

    def test_allocate_with_format(self, lazy_allocator_cls):
        """Test allocation with explicit memory format."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([100, 2, 1024])
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_T2D

        memory_obj = allocator.allocate(shape, dtype, fmt)

        assert memory_obj is not None
        assert memory_obj.is_valid()
        assert memory_obj.get_memory_format() == fmt

        allocator.close()

    def test_allocate_multiple_shapes_and_dtypes(self, lazy_allocator_cls):
        """Test allocation with multiple shapes and dtypes."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shapes = [torch.Size([100, 2, 512]), torch.Size([100, 2, 512])]
        dtypes = [torch.bfloat16, torch.bfloat16]

        memory_obj = allocator.allocate(shapes, dtypes)

        assert memory_obj is not None
        assert memory_obj.is_valid()

        allocator.close()

    def test_allocate_returns_none_when_out_of_memory(self, lazy_allocator_cls):
        """Test that allocation returns None when memory is exhausted."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.INIT_SIZE,  # Same as init to prevent expansion
        )

        # Try to allocate more than available
        huge_shape = torch.Size([self.INIT_SIZE])
        memory_obj = allocator.allocate(huge_shape, torch.float32)

        assert memory_obj is None

        allocator.close()

    def test_free_basic(self, lazy_allocator_cls):
        """Test that free invalidates the MemoryObj."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([512, 512])
        memory_obj = allocator.allocate(shape, torch.float32)
        assert memory_obj is not None
        assert memory_obj.is_valid()

        allocator.free(memory_obj)
        assert not memory_obj.is_valid()
        assert memory_obj.tensor is None

        allocator.close()

    def test_free_idempotent(self, lazy_allocator_cls):
        """Test that freeing an already freed object does not crash."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([256, 256])
        memory_obj = allocator.allocate(shape, torch.float32)
        assert memory_obj is not None

        allocator.free(memory_obj)
        # This should not crash
        allocator.free(memory_obj)

        assert allocator.memcheck()
        allocator.close()

    def test_batched_allocate_basic(self, lazy_allocator_cls):
        """Test batched allocation returns correct number of MemoryObjs."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([100, 2, 512])
        dtype = torch.bfloat16
        batch_size = 8

        memory_objs = allocator.batched_allocate(shape, dtype, batch_size)

        assert memory_objs is not None
        assert len(memory_objs) == batch_size
        for obj in memory_objs:
            assert obj is not None
            assert obj.is_valid()
            assert obj.tensor is not None
            assert obj.tensor.shape == shape
            assert obj.tensor.dtype == dtype

        allocator.close()

    def test_batched_allocate_with_format(self, lazy_allocator_cls):
        """Test batched allocation with explicit memory format."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([100, 2, 512])
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_T2D
        batch_size = 4

        memory_objs = allocator.batched_allocate(shape, dtype, batch_size, fmt)

        assert memory_objs is not None
        for obj in memory_objs:
            assert obj.get_memory_format() == fmt

        allocator.close()

    def test_batched_allocate_returns_none_when_out_of_memory(self, lazy_allocator_cls):
        """Test that batched allocation returns None when memory is exhausted."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.INIT_SIZE,
        )

        shape = torch.Size([1024 * 1024])  # 1M elements
        dtype = torch.float32  # 4 bytes each = 4MB per allocation
        batch_size = 100  # Would need 400MB, more than available

        memory_objs = allocator.batched_allocate(shape, dtype, batch_size)

        assert memory_objs is None

        allocator.close()

    def test_batched_free_basic(self, lazy_allocator_cls):
        """Test batched free invalidates all MemoryObjs."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([100, 2, 512])
        dtype = torch.bfloat16
        batch_size = 4

        memory_objs = allocator.batched_allocate(shape, dtype, batch_size)
        assert memory_objs is not None

        allocator.batched_free(memory_objs)

        for obj in memory_objs:
            assert not obj.is_valid()

        assert allocator.memcheck()
        allocator.close()

    def test_memcheck_returns_true_after_operations(self, lazy_allocator_cls):
        """Test that memcheck returns True after valid operations."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        # Initial state
        assert allocator.memcheck()

        # After allocation
        shape = torch.Size([512, 512])
        memory_obj = allocator.allocate(shape, torch.float32)
        assert allocator.memcheck()

        # After free
        allocator.free(memory_obj)
        assert allocator.memcheck()

        # After batched operations
        objs = allocator.batched_allocate(shape, torch.float32, 4)
        assert allocator.memcheck()

        allocator.batched_free(objs)
        assert allocator.memcheck()

        allocator.close()

    def test_inplace_tensor_modification(self, lazy_allocator_cls):
        """Test that allocated tensor data can be modified in place."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([1024])
        memory_obj = allocator.allocate(shape, torch.float32)
        assert memory_obj is not None

        # Modify the tensor in place
        memory_obj.tensor.fill_(42.0)
        assert torch.all(memory_obj.tensor == 42.0)

        memory_obj.tensor[0] = 123.0
        assert memory_obj.tensor[0] == 123.0

        allocator.close()

    def test_lazy_expansion_allows_larger_allocations(self, lazy_allocator_cls):
        """
        Test that lazy expansion allows allocations beyond init_size.

        The background thread should expand the available memory over time,
        allowing allocations that exceed the initial size.
        """
        # Start with small init_size, larger final_size
        init_size = 1 << 25  # 32 MB
        final_size = 1 << 27  # 128 MB

        allocator = lazy_allocator_cls(
            init_size=init_size,
            final_size=final_size,
        )

        # Wait for background expansion to complete
        # This gives the lazy allocator time to expand memory
        time.sleep(0.5)

        # Try to allocate more than init_size (but less than final_size)
        # 64 MB > 32 MB init_size
        large_shape = torch.Size([16 * 1024 * 1024])  # 16M elements * 4 bytes = 64MB
        memory_obj = allocator.allocate(large_shape, torch.float32)

        assert memory_obj is not None
        assert memory_obj.is_valid()

        allocator.close()

    def test_allocate_various_dtypes(self, lazy_allocator_cls):
        """Test allocation with various data types."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        test_cases = [
            (torch.Size([512, 512]), torch.float32),
            (torch.Size([1024, 1024]), torch.bfloat16),
            (torch.Size([2048, 2048]), torch.int8),
            (torch.Size([256, 256]), torch.half),
        ]

        memory_objs = []
        for shape, dtype in test_cases:
            obj = allocator.allocate(shape, dtype)
            assert obj is not None, f"Failed to allocate {shape} with {dtype}"
            assert obj.tensor.dtype == dtype
            assert obj.tensor.shape == shape
            memory_objs.append(obj)

        # Free all
        for obj in memory_objs:
            allocator.free(obj)

        assert allocator.memcheck()
        allocator.close()

    def test_allocation_and_free_interleaved(self, lazy_allocator_cls):
        """Test interleaved allocation and free operations."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([256, 256])
        dtype = torch.float32

        obj1 = allocator.allocate(shape, dtype)
        obj2 = allocator.allocate(shape, dtype)

        allocator.free(obj1)

        obj3 = allocator.allocate(shape, dtype)

        allocator.free(obj2)
        allocator.free(obj3)

        assert allocator.memcheck()
        allocator.close()
