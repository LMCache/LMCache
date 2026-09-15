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
    l1_memory_manager as l1_memory_manager_module,
)

GIB = 1024**3


class TestL1PoolAvailableMemoryGuard:
    """MP allocates the L1 pool once per cache server, so the configured size
    is the host footprint directly — but nothing used to check it against the
    host. An over-large pool was killed by the OOM killer at startup with no
    log line saying how big the request had been."""

    @staticmethod
    def _patch_available(monkeypatch, gb: float):
        monkeypatch.setattr(
            l1_memory_manager_module.SystemMemoryDetector,
            "get_available_memory_gb",
            staticmethod(lambda: gb),
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
            def __init__(self, init_size, size, align_bytes):
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
