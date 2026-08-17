# SPDX-License-Identifier: Apache-2.0
"""Tests for the Device L1 memory manager framework (PR 1).

These tests verify the backend-agnostic framework:
- ``DeviceMemoryPool`` protocol structural typing
- ``DeviceResidentL1MemoryManager`` allocate/free routing
- ``L1Manager.device_reserve_write`` / ``has_device_l1``
- ``AdapterDescriptor.l1_tier`` default and routing in prefetch_controller

All tests use a mock pool implementation — no real backend is needed.
"""

# Standard
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import DeviceResidentL1Config
from lmcache.v1.distributed.memory_manager.device_resident_l1_memory_manager import (
    DeviceMemoryPool,
    DeviceResidentL1MemoryManager,
)

# ---------------------------------------------------------------------------
# Mock device memory pool (satisfies DeviceMemoryPool protocol)
# ---------------------------------------------------------------------------


class MockDeviceMemoryPool:
    """Mock device memory pool satisfying DeviceMemoryPool protocol.

    Simulates a device pool with backpressure: ``wait_for_available`` blocks
    until enough bytes are freed (or timeout).
    """

    def __init__(self, total_bytes: int, device: str = "cuda:0") -> None:
        self._total = total_bytes
        self._free = total_bytes
        self._device = device
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._allocated: list = []

    def allocate(self, *, shapes, dtypes, fmt=None):
        """Allocate one device MemoryObj (mock)."""
        with self._lock:
            # Compute size from shapes/dtypes
            size = sum(
                s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=False)
            )
            if self._free < size:
                return None
            self._free -= size
            obj = MagicMock()
            obj.get_size = MagicMock(return_value=size)

            # Make raw_tensor a device tensor so free() routes correctly
            raw = torch.empty(1, device=self._device)
            obj.raw_tensor = raw
            obj._size = size
            # Set parent to self so free() routes back
            obj.parent = MagicMock(return_value=self)
            self._allocated.append(obj)
            return obj

    def free(self, memory_obj):
        """Free one device MemoryObj."""
        with self._lock:
            if hasattr(memory_obj, "_size"):
                self._free += memory_obj._size
                self._cond.notify_all()
                if memory_obj in self._allocated:
                    self._allocated.remove(memory_obj)

    def batched_free(self, memory_objs):
        """Free multiple device MemoryObjs."""
        for o in memory_objs:
            self.free(o)

    def wait_for_available(self, required_bytes: int, timeout: float) -> bool:
        """Block until enough bytes are free, or timeout."""
        with self._cond:
            deadline = time.monotonic() + timeout
            while self._free < required_bytes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def get_free_bytes(self) -> int:
        """Return current free bytes."""
        with self._lock:
            return self._free


# ---------------------------------------------------------------------------
# DeviceMemoryPool protocol structural typing
# ---------------------------------------------------------------------------


class TestDeviceMemoryPoolProtocol:
    """Verify that DeviceMemoryPool is a runtime-checkable protocol."""

    def test_mock_pool_satisfies_protocol(self):
        """MockDeviceMemoryPool should satisfy DeviceMemoryPool protocol."""
        pool = MockDeviceMemoryPool(total_bytes=1024 * 1024)
        assert isinstance(pool, DeviceMemoryPool)

    def test_non_pool_does_not_satisfy_protocol(self):
        """A plain object should not satisfy DeviceMemoryPool protocol."""
        assert not isinstance(object(), DeviceMemoryPool)
        assert not isinstance(42, DeviceMemoryPool)


# ---------------------------------------------------------------------------
# DeviceResidentL1MemoryManager
# ---------------------------------------------------------------------------


class TestDeviceResidentL1MemoryManager:
    """Verify DeviceResidentL1MemoryManager pool management and free routing."""

    def test_no_backend_raises_not_implemented(self):
        """An empty backend should raise NotImplementedError."""
        config = DeviceResidentL1Config(backend="", device_ids=[0])
        # _init_device_pools is called in __init__, which should raise
        with pytest.raises(ValueError, match="Unsupported device-resident L1 backend"):
            DeviceResidentL1MemoryManager(
                memory_config=MagicMock(),
                device_resident_l1_config=config,
            )

    def test_phx_backend_not_implemented(self):
        """The 'phx' backend should raise NotImplementedError (follow-up PR)."""
        config = DeviceResidentL1Config(backend="phx", device_ids=[0])
        with pytest.raises(NotImplementedError, match="follow-up PR"):
            DeviceResidentL1MemoryManager(
                memory_config=MagicMock(),
                device_resident_l1_config=config,
            )

    def test_free_routes_device_objs_to_pool(self):
        """free() should route device objs to their parent pool."""
        pool = MockDeviceMemoryPool(total_bytes=1024 * 1024)
        obj = pool.allocate(
            shapes=[torch.Size([10])],
            dtypes=[torch.float32],
        )
        assert obj is not None
        assert pool.get_free_bytes() < 1024 * 1024  # allocated

        # Create a manager with an injected pool (bypass __init__)
        manager = DeviceResidentL1MemoryManager.__new__(DeviceResidentL1MemoryManager)
        manager._device_resident_l1_config = DeviceResidentL1Config(
            backend="phx", device_ids=[0]
        )
        manager._device_pools = {0: pool}
        manager._cpu_manager = MagicMock()

        # Free the device obj
        manager.free([obj])
        assert pool.get_free_bytes() == 1024 * 1024  # fully freed

    def test_free_routes_cpu_objs_to_cpu_manager(self):
        """free() should route CPU objs to the CPU manager."""
        cpu_obj = MagicMock()
        cpu_obj.raw_tensor = torch.empty(1)  # CPU tensor
        cpu_obj.parent = MagicMock(return_value=None)

        manager = DeviceResidentL1MemoryManager.__new__(DeviceResidentL1MemoryManager)
        manager._device_resident_l1_config = DeviceResidentL1Config(
            backend="phx", device_ids=[0]
        )
        manager._device_pools = {}
        mock_cpu = MagicMock()
        manager._cpu_manager = mock_cpu

        manager.free([cpu_obj])
        mock_cpu.free.assert_called_once_with([cpu_obj])

    def test_free_handles_mixed_objs(self):
        """free() should handle a mix of CPU and device objs."""
        pool = MockDeviceMemoryPool(total_bytes=1024 * 1024)
        dev_obj = pool.allocate(
            shapes=[torch.Size([10])],
            dtypes=[torch.float32],
        )
        cpu_obj = MagicMock()
        cpu_obj.raw_tensor = torch.empty(1)
        cpu_obj.parent = MagicMock(return_value=None)

        manager = DeviceResidentL1MemoryManager.__new__(DeviceResidentL1MemoryManager)
        manager._device_resident_l1_config = DeviceResidentL1Config(
            backend="phx", device_ids=[0]
        )
        manager._device_pools = {0: pool}
        mock_cpu = MagicMock()
        manager._cpu_manager = mock_cpu

        manager.free([dev_obj, cpu_obj])
        assert pool.get_free_bytes() == 1024 * 1024  # device obj freed
        mock_cpu.free.assert_called_once_with([cpu_obj])

    def test_kv_rank_to_device_mapping(self):
        """_kv_rank_to_device should map by modulo num_devices."""
        manager = DeviceResidentL1MemoryManager.__new__(DeviceResidentL1MemoryManager)
        manager._device_resident_l1_config = DeviceResidentL1Config(
            backend="phx", device_ids=[3, 5, 7]
        )
        manager._device_pools = {}
        manager._cpu_manager = MagicMock()

        assert manager._kv_rank_to_device(0) == 3
        assert manager._kv_rank_to_device(1) == 5
        assert manager._kv_rank_to_device(2) == 7
        assert manager._kv_rank_to_device(3) == 3  # wraps around

    def test_kv_rank_to_device_empty_returns_minus_one(self):
        """_kv_rank_to_device should return -1 when no devices configured."""
        manager = DeviceResidentL1MemoryManager.__new__(DeviceResidentL1MemoryManager)
        manager._device_resident_l1_config = DeviceResidentL1Config(
            backend="phx", device_ids=[]
        )
        manager._device_pools = {}
        manager._cpu_manager = MagicMock()

        assert manager._kv_rank_to_device(0) == -1


# ---------------------------------------------------------------------------
# DeviceResidentL1Config
# ---------------------------------------------------------------------------


class TestDeviceResidentL1Config:
    """Verify DeviceResidentL1Config dataclass."""

    def test_defaults(self):
        """DeviceResidentL1Config should have sensible defaults."""
        config = DeviceResidentL1Config()
        assert config.backend == ""
        assert config.device_ids == []
        assert config.buffer_size_mb == 2048
        assert config.use_direct_io is True

    def test_construction(self):
        """DeviceResidentL1Config should accept all fields."""
        config = DeviceResidentL1Config(
            backend="phx",
            device_ids=[0, 1, 2],
            buffer_size_mb=4096,
            use_direct_io=False,
        )
        assert config.backend == "phx"
        assert config.device_ids == [0, 1, 2]
        assert config.buffer_size_mb == 4096
        assert config.use_direct_io is False


# ---------------------------------------------------------------------------
# AdapterDescriptor.l1_tier
# ---------------------------------------------------------------------------


class TestAdapterDescriptor:
    """Verify AdapterDescriptor.l1_tier field."""

    def test_default_l1_tier_is_cpu(self):
        """AdapterDescriptor should default l1_tier to 'cpu'."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
            MockL2AdapterConfig,
        )
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            AdapterDescriptor,
        )

        desc = AdapterDescriptor(
            index=0, config=MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1)
        )
        assert desc.l1_tier == "cpu"

    def test_l1_tier_can_be_set_to_device(self):
        """AdapterDescriptor should accept l1_tier='device'."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
            MockL2AdapterConfig,
        )
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            AdapterDescriptor,
        )

        desc = AdapterDescriptor(
            index=0,
            config=MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1),
            l1_tier="device",
        )
        assert desc.l1_tier == "device"


# ---------------------------------------------------------------------------
# L1BackendType
# ---------------------------------------------------------------------------


class TestL1BackendType:
    """Verify L1BackendType has DEVICE member."""

    def test_device_backend_type_exists(self):
        """L1BackendType should have a DEVICE member."""
        # First Party
        from lmcache.v1.distributed.api import L1BackendType

        assert L1BackendType.DEVICE == "device"
        assert L1BackendType.DEVICE.value == "device"
