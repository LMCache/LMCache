# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Phase 2 — Memory Budget Coordination."""

import pytest

from lmcache.v1.distributed.dram_partition import (
    DramAllocation,
    DramPartitionConfig,
    DramPartitionCoordinator,
    StagingParams,
)


class TestDramPartitionConfig:
    """Tests for DramPartitionConfig validation."""

    def test_defaults(self):
        cfg = DramPartitionConfig()
        assert not cfg.enabled
        assert cfg.total_memory_budget_gb == 0.0

    def test_enabled(self):
        cfg = DramPartitionConfig(total_memory_budget_gb=16.0)
        assert cfg.enabled

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="total_memory_budget_gb"):
            DramPartitionConfig(total_memory_budget_gb=-1.0)

    def test_invalid_l1_fraction_raises(self):
        with pytest.raises(ValueError, match="l1_fraction"):
            DramPartitionConfig(total_memory_budget_gb=16.0, l1_fraction=0.0)
        with pytest.raises(ValueError, match="l1_fraction"):
            DramPartitionConfig(total_memory_budget_gb=16.0, l1_fraction=1.0)

    def test_invalid_watermarks_raise(self):
        with pytest.raises(ValueError, match="l2_high_watermark"):
            DramPartitionConfig(
                total_memory_budget_gb=16.0, l2_high_watermark=0.0
            )
        with pytest.raises(ValueError, match="l1_high_watermark"):
            DramPartitionConfig(
                total_memory_budget_gb=16.0, l1_high_watermark=1.1
            )


class TestDramPartitionCoordinator:
    """Tests for DramPartitionCoordinator allocation logic."""

    def test_basic_allocation(self):
        cfg = DramPartitionConfig(total_memory_budget_gb=16.0, l1_fraction=0.25)
        coord = DramPartitionCoordinator(cfg)
        alloc = coord.allocate()

        total = int(16.0 * (1 << 30))
        assert alloc.l1_size_bytes == int(total * 0.25)
        assert alloc.l2_max_bytes == total - alloc.l1_size_bytes
        assert alloc.l1_high_watermark == 0.8
        assert alloc.l2_high_watermark == 0.8

    def test_staging_validation_passes(self):
        # 16 GiB total, 0.3 fraction => L1 = 4.8 GiB
        # staging_min for 8 prefetches, 5 MB chunk:
        # 8 * 2.5 * 5MB + 1 * 5MB = 105 MB — well under 4.8 GiB
        cfg = DramPartitionConfig(total_memory_budget_gb=16.0, l1_fraction=0.3)
        coord = DramPartitionCoordinator(cfg)
        params = StagingParams(
            chunk_size_bytes=5 * 1024 * 1024,
            max_prefetch_in_flight=8,
            max_write_in_flight=1,
        )
        alloc = coord.allocate(staging_params=params)
        assert alloc.l1_staging_min_bytes == int(8 * 2.5 * 5 * 1024 * 1024) + 5 * 1024 * 1024

    def test_staging_validation_fails(self):
        # Tiny budget with large chunk → should fail
        cfg = DramPartitionConfig(total_memory_budget_gb=0.1, l1_fraction=0.1)
        coord = DramPartitionCoordinator(cfg)
        params = StagingParams(
            chunk_size_bytes=50 * 1024 * 1024,  # 50 MB chunk
            max_prefetch_in_flight=8,
        )
        with pytest.raises(ValueError, match="staging minimum"):
            coord.allocate(staging_params=params)

    def test_disabled_raises(self):
        cfg = DramPartitionConfig()  # disabled
        with pytest.raises(ValueError, match="total_memory_budget_gb > 0"):
            DramPartitionCoordinator(cfg)

    def test_custom_watermarks(self):
        cfg = DramPartitionConfig(
            total_memory_budget_gb=8.0,
            l1_high_watermark=0.9,
            l2_high_watermark=0.7,
        )
        coord = DramPartitionCoordinator(cfg)
        alloc = coord.allocate()
        assert alloc.l1_high_watermark == 0.9
        assert alloc.l2_high_watermark == 0.7

    def test_no_staging_params(self):
        cfg = DramPartitionConfig(total_memory_budget_gb=16.0)
        coord = DramPartitionCoordinator(cfg)
        alloc = coord.allocate(staging_params=None)
        assert alloc.l1_staging_min_bytes == 0

    def test_zero_chunk_size_skips_validation(self):
        cfg = DramPartitionConfig(total_memory_budget_gb=0.01, l1_fraction=0.1)
        coord = DramPartitionCoordinator(cfg)
        params = StagingParams(chunk_size_bytes=0)
        alloc = coord.allocate(staging_params=params)
        assert alloc.l1_staging_min_bytes == 0


class TestCliParsing:
    """Test that budget CLI args flow into StorageManagerConfig."""

    def test_budget_disabled_by_default(self):
        from lmcache.v1.distributed.config import parse_args

        config = parse_args([
            "--l1-size-gb", "4",
            "--l1-use-lazy",
            "--eviction-policy", "LRU",
            "--l2-adapter", '{"type":"dram","max_size_gb":8.0}',
        ])
        assert not config.dram_partition_config.enabled
        # L1 size should be unmodified
        assert config.l1_manager_config.memory_config.size_in_bytes == int(4 * (1 << 30))

    def test_budget_overrides_sizes(self):
        from lmcache.v1.distributed.config import parse_args

        config = parse_args([
            "--l1-size-gb", "4",  # will be overridden
            "--l1-use-lazy",
            "--eviction-policy", "LRU",
            "--l2-adapter", '{"type":"dram","max_size_gb":1.0}',
            "--total-memory-budget-gb", "16.0",
            "--l1-fraction", "0.25",
            "--l1-high-watermark", "0.9",
            "--l2-high-watermark", "0.7",
        ])
        assert config.dram_partition_config.enabled
        total = int(16.0 * (1 << 30))
        expected_l1 = int(total * 0.25)
        assert config.l1_manager_config.memory_config.size_in_bytes == expected_l1
        assert config.eviction_config.trigger_watermark == 0.9

    def test_pressure_eviction_flag(self):
        from lmcache.v1.distributed.config import parse_args

        config = parse_args([
            "--l1-size-gb", "4",
            "--l1-use-lazy",
            "--eviction-policy", "LRU",
            "--enable-pressure-eviction",
        ])
        assert config.enable_pressure_eviction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
