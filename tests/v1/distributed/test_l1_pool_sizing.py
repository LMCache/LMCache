# SPDX-License-Identifier: Apache-2.0
"""
Sizing rules for the MP L1 pool.

Pure arithmetic against a stubbed memory reading and dummy allocators, so
these run in the CPU lane -- unlike test_l1_memory_manager.py, which skips
the whole module without a device runtime.
"""

# Third Party

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager import (
    L1MemoryManager,
)
from lmcache.v1.distributed.memory_manager import (
    l1_memory_manager as l1_memory_manager_module,
)

GIB = 1024**3


class TestL1PoolAvailableMemoryGuard:
    """MP allocates the L1 pool once per cache server, so the configured size
    is its memory footprint directly. A pool larger than the available memory
    is OOM-killed at startup."""

    @staticmethod
    def _patch_available(monkeypatch, gb: float):
        reading = int(gb * GIB) if gb > 0 else None
        monkeypatch.setattr(
            l1_memory_manager_module,
            "_available_memory_bytes",
            lambda: reading,
        )

    def test_size_that_fits_is_untouched(self, monkeypatch):
        self._patch_available(monkeypatch, 2048.0)
        assert (
            l1_memory_manager_module._clamp_to_available_memory(64 * 1024**3)
            == 64 * 1024**3
        )

    def test_oversized_pool_is_clamped(self, monkeypatch):
        """512 GB asked for on a host with 400 GB free."""
        self._patch_available(monkeypatch, 400.0)
        clamped = l1_memory_manager_module._clamp_to_available_memory(512 * 1024**3)
        assert clamped == int(400.0 * 1024**3)
        assert clamped < 512 * 1024**3

    def test_unknown_available_memory_leaves_it_alone(self, monkeypatch):
        """Better to allocate unchecked than to refuse on a bad reading."""
        self._patch_available(monkeypatch, 0.0)
        assert (
            l1_memory_manager_module._clamp_to_available_memory(512 * 1024**3)
            == 512 * 1024**3
        )

    def test_clamp_reaches_the_reported_size(self, monkeypatch):
        """The size L1MemoryDesc reports is what consumers use to address the
        shared pool. If the clamp did not reach it, they would be handed a
        length running past the end of the mapping."""
        self._patch_available(monkeypatch, 8.0)
        created = {}

        class DummyAllocator:
            def __init__(self, size, **kwargs):
                created["size"] = size

        monkeypatch.setattr(
            l1_memory_manager_module, "MixedMemoryAllocator", DummyAllocator
        )
        config = L1MemoryManagerConfig(
            size_in_bytes=64 * 1024**3, use_lazy=False, shm_name=""
        )
        manager = L1MemoryManager(config)

        assert created["size"] == int(8.0 * 1024**3)
        assert manager._size_in_bytes == created["size"], (
            "reported size must match the buffer actually allocated"
        )

    def test_lazy_init_size_never_exceeds_the_clamped_total(self, monkeypatch):
        """LazyMemoryAllocator reserves init_size up front and grows to the
        total, so the cap has to apply to both ends."""
        self._patch_available(monkeypatch, 8.0)
        created = {}

        class DummyLazy:
            def __init__(self, init_size, size, align_bytes, **kwargs):
                created["init"] = init_size
                created["size"] = size

        monkeypatch.setattr(l1_memory_manager_module, "LazyMemoryAllocator", DummyLazy)
        config = L1MemoryManagerConfig(
            size_in_bytes=64 * 1024**3,
            use_lazy=True,
            init_size_in_bytes=20 * 1024**3,
            shm_name="",
        )
        # __post_init__ turns lazy off when the backend cannot pin memory,
        # which is every CPU-only runner. Force it back on: the rule under
        # test is the sizing arithmetic, not backend support.
        config.use_lazy = True
        l1_memory_manager_module.create_memory_allocator(config)

        assert created["size"] == int(8.0 * 1024**3)
        assert created["init"] <= created["size"]


class TestAvailableMemoryReading:
    """The reading behind the clamp: the smaller of host MemAvailable and the
    cgroup headroom, because a container is killed at its cgroup limit even
    when the node has free memory."""

    @staticmethod
    def _point_at(monkeypatch, tmp_path, meminfo_gb=None):
        meminfo = tmp_path / "meminfo"
        if meminfo_gb is not None:
            meminfo.write_text(
                "MemTotal:       2147483648 kB\n"
                f"MemAvailable:   {int(meminfo_gb * 1024**2)} kB\n"
            )
        cgroup = tmp_path / "cgroup"
        cgroup.mkdir()
        monkeypatch.setattr(l1_memory_manager_module, "_PROC_MEMINFO", str(meminfo))
        monkeypatch.setattr(l1_memory_manager_module, "_CGROUP_ROOT", str(cgroup))
        return cgroup

    def test_cgroup_v2_limit_below_host_wins(self, monkeypatch, tmp_path):
        """A 500 GiB pod on a node with 1800 GiB available."""
        cgroup = self._point_at(monkeypatch, tmp_path, meminfo_gb=1800)
        (cgroup / "memory.max").write_text(str(500 * GIB))
        (cgroup / "memory.current").write_text(str(60 * GIB))
        (cgroup / "memory.stat").write_text(
            f"anon {40 * GIB}\ninactive_file {20 * GIB}\n"
        )
        # 500 - (60 - 20) = 460 GiB of headroom.
        assert l1_memory_manager_module._available_memory_bytes() == 460 * GIB

    def test_cgroup_v2_without_limit_uses_host(self, monkeypatch, tmp_path):
        cgroup = self._point_at(monkeypatch, tmp_path, meminfo_gb=300)
        (cgroup / "memory.max").write_text("max")
        assert l1_memory_manager_module._available_memory_bytes() == 300 * GIB

    def test_cgroup_v1_limit_below_host_wins(self, monkeypatch, tmp_path):
        cgroup = self._point_at(monkeypatch, tmp_path, meminfo_gb=1800)
        v1 = cgroup / "memory"
        v1.mkdir()
        (v1 / "memory.limit_in_bytes").write_text(str(256 * GIB))
        (v1 / "memory.usage_in_bytes").write_text(str(16 * GIB))
        (v1 / "memory.stat").write_text(f"total_inactive_file {6 * GIB}\n")
        assert l1_memory_manager_module._available_memory_bytes() == 246 * GIB

    def test_cgroup_v1_unlimited_uses_host(self, monkeypatch, tmp_path):
        cgroup = self._point_at(monkeypatch, tmp_path, meminfo_gb=300)
        v1 = cgroup / "memory"
        v1.mkdir()
        (v1 / "memory.limit_in_bytes").write_text("9223372036854771712")
        (v1 / "memory.usage_in_bytes").write_text(str(16 * GIB))
        assert l1_memory_manager_module._available_memory_bytes() == 300 * GIB

    def test_nothing_readable_is_unknown(self, monkeypatch, tmp_path):
        self._point_at(monkeypatch, tmp_path)
        assert l1_memory_manager_module._available_memory_bytes() is None

    def test_oversized_pool_in_a_pod_is_clamped_to_the_pod(self, monkeypatch, tmp_path):
        """1 TiB of L1 configured in a 500 GiB pod on a large node."""
        cgroup = self._point_at(monkeypatch, tmp_path, meminfo_gb=1800)
        (cgroup / "memory.max").write_text(str(500 * GIB))
        (cgroup / "memory.current").write_text(str(10 * GIB))
        (cgroup / "memory.stat").write_text("inactive_file 0\n")
        assert (
            l1_memory_manager_module._clamp_to_available_memory(1024 * GIB) == 490 * GIB
        )
