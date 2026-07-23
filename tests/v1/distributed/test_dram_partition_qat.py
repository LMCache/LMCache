# SPDX-License-Identifier: Apache-2.0
"""Integration test: DRAM partition + pressure eviction with QAT hardware.

Run on SPR QAT machine:
    cd /home/xuezhan/LMCache_QAT/LMCache
    KVCLIP_QZIP_LIB_PATH=/home/xuezhan/LMCache_QAT/KVCacheClip/kvclip/lib/libkvclip_qzip.so \
    LD_LIBRARY_PATH=/home/xuezhan/LMCache_QAT/KVCacheClip/kvclip/infra/qat:$LD_LIBRARY_PATH \
    python -m pytest tests/v1/distributed/test_dram_partition_qat.py -v
"""

import os
import time

import pytest

# Skip if QAT library not available
def _has_native_ops() -> bool:
    try:
        import lmcache.native_storage_ops  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("KVCLIP_QZIP_LIB_PATH"),
        reason="KVCLIP_QZIP_LIB_PATH not set (no QAT hardware)",
    ),
    pytest.mark.skipif(
        not _has_native_ops(),
        reason="native_storage_ops not built (L1Manager requires TTLLock)",
    ),
]


from lmcache.v1.distributed.config import parse_args
from lmcache.v1.distributed.dram_partition import (
    DramPartitionConfig,
    DramPartitionCoordinator,
    StagingParams,
)


class TestBudgetWithQatCli:
    """Validate CLI budget args produce correct config for QAT deployment."""

    def test_budget_splits_l1_and_dram_l2(self):
        """--total-memory-budget-gb splits correctly with QAT serde."""
        config = parse_args([
            "--l1-size-gb", "1",  # will be overridden
            "--no-l1-use-lazy",
            "--eviction-policy", "LRU",
            "--l2-adapter", '{"type":"dram","max_size_gb":1.0,"serde":{"type":"accel_kv_compress","kwargs":{"element_size":2,"truncate_bits":2}}}',
            "--l2-store-policy", "skip_l1",
            "--total-memory-budget-gb", "4.0",
            "--l1-fraction", "0.3",
            "--enable-pressure-eviction",
        ])
        # L1 = 4.0 * 0.3 = 1.2 GiB
        assert config.l1_manager_config.memory_config.size_in_bytes == int(4.0 * 0.3 * (1 << 30))
        assert config.enable_pressure_eviction is True
        assert config.dram_partition_config.enabled

    def test_staging_validation_with_real_chunk_size(self):
        """Validate staging formula with a realistic KV chunk size.

        Llama-3-8B: 2 * 32 layers * 8 pages * 8 heads * 128 dim * 2 bytes = 8 MB
        """
        chunk_size = 2 * 32 * 8 * 8 * 128 * 2  # 8 MiB
        cfg = DramPartitionConfig(
            total_memory_budget_gb=4.0,
            l1_fraction=0.3,
        )
        coord = DramPartitionCoordinator(cfg)
        params = StagingParams(
            chunk_size_bytes=chunk_size,
            max_prefetch_in_flight=8,
            max_write_in_flight=1,
        )
        alloc = coord.allocate(staging_params=params)
        # staging_min = 8 * 2.5 * 8MB + 1 * 8MB = 168 MB
        # L1 = 4.0 * 0.3 = 1.2 GiB >> 168 MB → should pass
        assert alloc.l1_size_bytes > alloc.l1_staging_min_bytes
        print(f"  L1: {alloc.l1_size_bytes / (1<<20):.0f} MiB")
        print(f"  L2: {alloc.l2_max_bytes / (1<<20):.0f} MiB")
        print(f"  Staging min: {alloc.l1_staging_min_bytes / (1<<20):.0f} MiB")


class TestPressureEvictionWithQat:
    """Validate pressure eviction triggers with a tiny L1 slab."""

    def test_pressure_eviction_fires_on_l1_oom(self):
        """Create a small L1, fill it, verify pressure eviction frees space."""
        from lmcache.v1.distributed.config import (
            EvictionConfig,
            L1ManagerConfig,
            L1MemoryManagerConfig,
            StorageManagerConfig,
        )
        from lmcache.v1.distributed.dram_partition import DramPartitionConfig
        from lmcache.v1.distributed.l1_manager import L1Manager
        from lmcache.v1.distributed.storage_controllers.eviction_controller import (
            L1EvictionController,
        )
        from lmcache.v1.distributed.api import ObjectKey, MemoryLayoutDesc
        import torch

        # Tiny L1: 4 MB (fits ~4 x 1MB objects)
        l1_size = 4 * 1024 * 1024
        memory_config = L1MemoryManagerConfig(
            size_in_bytes=l1_size,
            use_lazy=False,
            init_size_in_bytes=l1_size,
            shm_name="",
        )
        l1_config = L1ManagerConfig(memory_config=memory_config)
        l1_manager = L1Manager(l1_config)

        eviction_config = EvictionConfig(
            eviction_policy="LRU",
            trigger_watermark=0.8,
            eviction_ratio=0.5,
        )
        controller = L1EvictionController(
            l1_manager=l1_manager,
            eviction_config=eviction_config,
        )

        # Layout for ~1.5 MB objects (2 fit in 4 MB slab, 3rd triggers OOM)
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([1536 * 1024])],
            dtypes=[torch.uint8],
        )

        # Fill L1 with 2 objects (3 MB / 4 MB = 75% usage)
        keys = []
        for i in range(2):
            key = ObjectKey(
                chunk_hash=f"pressure_test_{i}".encode().ljust(32, b'\0'),
                model_name="test",
                kv_rank=0,
            )
            results = l1_manager.reserve_write(
                keys=[key], is_temporary=[False], layout_desc=layout, mode="new"
            )
            err, obj = results[key]
            assert err.name == "SUCCESS", f"Write {i} failed: {err}"
            l1_manager.finish_write_and_reserve_read([key])
            l1_manager.finish_read([key])
            keys.append(key)

        # Verify usage is near capacity
        used, total = l1_manager.get_memory_usage()
        print(f"  L1 usage before pressure: {used}/{total} ({used/total*100:.0f}%)")

        # Try to write a 4th object — should fail (OOM)
        key4 = ObjectKey(
            chunk_hash=b"pressure_test_overflow".ljust(32, b'\0'),
            model_name="test",
            kv_rank=0,
        )
        results = l1_manager.reserve_write(
            keys=[key4], is_temporary=[False], layout_desc=layout, mode="new"
        )
        err4, _ = results[key4]
        assert err4.name == "OUT_OF_MEMORY", f"Expected OOM, got {err4}"

        # Trigger pressure eviction
        evicted = controller.trigger_pressure_eviction()
        print(f"  Pressure eviction freed {evicted} keys")
        assert evicted > 0

        # Now the 4th write should succeed
        results = l1_manager.reserve_write(
            keys=[key4], is_temporary=[False], layout_desc=layout, mode="new"
        )
        err4, obj4 = results[key4]
        assert err4.name == "SUCCESS", f"After eviction still failed: {err4}"
        print("  Post-eviction write succeeded!")

        # Cleanup
        l1_manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
