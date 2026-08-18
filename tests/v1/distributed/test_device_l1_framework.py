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

            # Mock the raw tensor's device type — free() routes by
            # raw_tensor.device.type, which is all these tests exercise.
            # No real GPU/CUDA device is required (keeps CI green).
            obj.raw_tensor.device.type = self._device.split(":")[0]  # "cuda"
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

    def get_total_bytes(self) -> int:
        """Return total capacity bytes."""
        return self._total


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


# ---------------------------------------------------------------------------
# get_memory_usage / get_device_memory_usage accounting
# ---------------------------------------------------------------------------


class TestMemoryUsageAccounting:
    """get_memory_usage() must report the CPU tier (eviction semantics);
    device pool usage is exposed separately via get_device_memory_usage().
    """

    @staticmethod
    def _make_manager(pools, cpu_usage=(100, 200)):
        """Build a manager bypassing __init__ (no backend in this PR)."""
        # First Party
        from lmcache.v1.distributed.memory_manager import (
            device_resident_l1_memory_manager as drl1,
        )

        mgr = object.__new__(drl1.DeviceResidentL1MemoryManager)
        mgr._device_pools = {i: p for i, p in enumerate(pools)}
        mgr._cpu_manager = MagicMock()
        mgr._cpu_manager.get_memory_usage.return_value = cpu_usage
        return mgr

    def test_get_memory_usage_reports_cpu_tier_only(self):
        """get_memory_usage() must delegate to the CPU manager, even when
        device pools hold allocations (device footprint is transient and
        must not skew the eviction watermark)."""
        pool = MockDeviceMemoryPool(total_bytes=1024)
        pool.allocate(shapes=[torch.Size([256])], dtypes=[torch.float32])
        mgr = self._make_manager(pools=[pool], cpu_usage=(100, 200))

        assert mgr.get_memory_usage() == (100, 200)

    def test_get_device_memory_usage_reports_per_device(self):
        """get_device_memory_usage() must report per-device
        ``(used, total)`` — aggregation could hide a single exhausted
        pool behind healthy ones."""
        pool0 = MockDeviceMemoryPool(total_bytes=1024)
        pool1 = MockDeviceMemoryPool(total_bytes=2048)
        # Allocate 128 bytes from pool0 (float32 x 32 elements)
        pool0.allocate(shapes=[torch.Size([32])], dtypes=[torch.float32])
        mgr = self._make_manager(pools=[pool0, pool1])

        assert mgr.get_device_memory_usage() == {
            0: (128, 1024),
            1: (0, 2048),
        }

    def test_get_device_memory_usage_empty_pools(self):
        """get_device_memory_usage() with no pools returns {}."""
        mgr = self._make_manager(pools=[])
        assert mgr.get_device_memory_usage() == {}


# ---------------------------------------------------------------------------
# L1Manager.get_device_memory_usage forwarding
# ---------------------------------------------------------------------------


class TestL1ManagerDeviceUsageForwarding:
    """L1Manager.get_device_memory_usage() forwards when the device-resident
    tier is present and returns an empty dict otherwise.
    """

    def test_forwards_to_device_resident_manager(self):
        """With a device-resident tier, L1Manager must forward the call."""
        # First Party
        from lmcache.v1.distributed.l1_manager import L1Manager

        wrapper = object.__new__(L1Manager)
        mm = MagicMock()
        mm.get_device_memory_usage.return_value = {0: (42, 84)}
        wrapper._memory_manager = mm

        assert L1Manager.get_device_memory_usage(wrapper) == {0: (42, 84)}
        mm.get_device_memory_usage.assert_called_once()

    def test_plain_manager_reports_empty(self):
        """Managers without the device tier must report {} via the
        L1Manager forwarding wrapper (getattr-based probe)."""
        # First Party
        from lmcache.v1.distributed.l1_manager import L1Manager

        wrapper = object.__new__(L1Manager)
        wrapper._memory_manager = MagicMock(spec=[])  # no device API
        assert L1Manager.get_device_memory_usage(wrapper) == {}


# ---------------------------------------------------------------------------
# PrefetchController._reserve_load_buffers: kv_rank split (regression)
# ---------------------------------------------------------------------------


class TestReserveLoadBuffersKvRankSplit:
    """Device keys must be split by kv_rank before device_reserve_write.

    Regression test: the whole device batch used to inherit
    ``device_keys[0].kv_rank``, so keys later in the batch could get
    buffers from the wrong device pool (a single prefetch request can
    span multiple kv_ranks).
    """

    @staticmethod
    def _make_keys(n: int, num_ranks: int, gid: int = 7):
        # First Party
        from lmcache.v1.distributed.api import ObjectKey

        return [
            ObjectKey(
                chunk_hash=bytes([i]),
                model_name="test-model",
                kv_rank=i % num_ranks,
                object_group_id=gid,
            )
            for i in range(n)
        ]

    def test_device_keys_split_by_kv_rank(self):
        """Each device_reserve_write batch must carry a single kv_rank."""
        # First Party
        from lmcache.v1.distributed.api import PrefetchMode
        from lmcache.v1.distributed.error import L1Error
        from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
            PrefetchController,
        )

        num_ranks = 3
        keys = self._make_keys(n=6, num_ranks=num_ranks)

        l1_manager = MagicMock()
        l1_manager.has_device_l1.return_value = True

        def fake_device_reserve_write(*, keys, is_temporary, layout_desc, kv_rank):
            return {k: (L1Error.SUCCESS, MagicMock()) for k in keys}

        l1_manager.device_reserve_write.side_effect = fake_device_reserve_write

        controller = MagicMock()
        controller._l1_manager = l1_manager
        controller._adapter_descriptors = {0: MagicMock(l1_tier="device")}
        controller._event_bus = MagicMock()

        request = MagicMock()
        request.mode = PrefetchMode.WARM  # skip retention policy
        request.keys = keys
        request.group_layout_descs = {7: MagicMock()}
        request.write_reserved_keys = []
        request.write_reserved_objs = {}

        plan_bitmap = MagicMock()
        plan_bitmap.gather.return_value = keys  # all keys on adapter 0

        reserved = PrefetchController._reserve_load_buffers(
            controller, request, keys, {0: plan_bitmap}
        )

        calls = l1_manager.device_reserve_write.call_args_list
        assert len(calls) == num_ranks
        seen_ranks: set[int] = set()
        for call in calls:
            batch_keys = call.kwargs["keys"]
            batch_ranks = {k.kv_rank for k in batch_keys}
            assert len(batch_ranks) == 1, (
                f"device_reserve_write batch mixes kv_ranks: {batch_ranks}"
            )
            assert call.kwargs["kv_rank"] in batch_ranks
            seen_ranks |= batch_ranks
        assert seen_ranks == {0, 1, 2}

        # All keys went through the device path; no CPU fallback reserve.
        l1_manager.reserve_write.assert_not_called()
        assert len(reserved) == len(keys)
        assert set(request.write_reserved_keys) == set(keys)


# ---------------------------------------------------------------------------
# DeviceResidentL1MemoryManager.close(): cascade to the CPU manager
# ---------------------------------------------------------------------------


class TestCloseCascadesToCpuManager:
    """close() must release the internal CPU manager too.

    ``L1Manager.close()`` only calls ``close()`` once on its single
    ``_memory_manager`` (this class), so the CPU slab / SHM path is only
    released if close() cascades into ``_cpu_manager``.
    """

    def test_close_releases_cpu_manager(self):
        """close() should close both device pools and the CPU manager."""
        # First Party
        from lmcache.v1.distributed.memory_manager import (
            device_resident_l1_memory_manager as drl1,
        )

        pool = MagicMock()

        # Bypass __init__ (no backend is implemented in this PR): patch the
        # attributes close() consumes directly.
        mgr = object.__new__(drl1.DeviceResidentL1MemoryManager)
        mgr._device_pools = {0: pool}
        mgr._cpu_manager = MagicMock()

        mgr.close()

        pool.close.assert_called_once()
        mgr._cpu_manager.close.assert_called_once()
