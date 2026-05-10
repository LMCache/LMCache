# SPDX-License-Identifier: Apache-2.0

"""Tests for io_uring command (passthrough) support in Rust raw block backend."""

# Standard
from unittest.mock import MagicMock
import asyncio
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)

logger = init_logger(__name__)


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


class MockConfig:
    """Mock configuration for testing."""

    def __init__(
        self,
        device_path: str,
        use_uring_cmd: bool = False,
        meta_total_bytes=4 * 1024 * 1024,
    ):
        self.extra_config = {
            "rust_raw_block.device_path": device_path,
            "rust_raw_block.use_odirect": False,
            "rust_raw_block.use_uring": True,
            "rust_raw_block.use_uring_cmd": use_uring_cmd,
            "rust_raw_block.capacity_bytes": 1024 * 1024 * 1024,  # 1GB
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": meta_total_bytes,
        }


class MockMetadata:
    """Mock metadata for testing."""

    def __init__(self, worker_id: int = 0, world_size: int = 1):
        self.worker_id = worker_id
        self.world_size = world_size


class MockLocalCPUBackend:
    """Mock local CPU backend for testing."""

    def __init__(self):
        pass

    def get_memory_allocator(self):
        return None

    def get_full_chunk_size_bytes(self) -> int:
        """return a default chunk size only for testing."""
        return 256 * 1024


def _build_rust_raw_block_metadata(
    worker_id: int = 0,
    world_size: int = 1,
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _build_rust_raw_block_local_cpu_backend() -> MagicMock:
    local_cpu_backend = MagicMock()
    local_cpu_backend.get_full_chunk_size_bytes.return_value = 4096
    return local_cpu_backend


def test_uring_cmd_requires_character_device(loop_in_thread):
    """Test that io_uring_cmd requires a character device, not a block device."""
    # This test requires a block device device /dev/nvme0n1
    # Skip if this doesn't exist
    device_path = os.environ.get("LMCACHE_TEST_BLOCK_DEVICE", "/dev/nvme0n1")

    if not os.path.exists(device_path):
        pytest.skip(f"Test device {device_path} not found.")

    config = MockConfig(device_path=device_path, use_uring_cmd=True)
    metadata = MockMetadata(worker_id=0, world_size=1)
    local_cpu_backend = MockLocalCPUBackend()

    # This should raise an error because the device is not a character device
    with pytest.raises(
        ValueError, match="io_uring_cmd requires a NVMe namespace character device"
    ):
        RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop_in_thread,
        )


def test_uring_cmd_get_nvme_info(loop_in_thread):
    """Test getting NVMe namespace ID and LBA size from character device."""
    # This test requires a block device device /dev/nvme0n1
    # Skip if this doesn't exist
    device_path = os.environ.get("LMCACHE_TEST_BLOCK_DEVICE", "/dev/ng0n1")

    if not os.path.exists(device_path):
        pytest.skip(f"Test device {device_path} not found.")

    config = MockConfig(device_path=device_path, use_uring_cmd=True)
    metadata = MockMetadata(worker_id=0, world_size=1)
    local_cpu_backend = MockLocalCPUBackend()

    try:
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop_in_thread,
        )

        # Get the raw device
        raw_device = backend._core.raw_device()

        # Test getting namespace ID
        nsid = raw_device.nvme_nsid()
        assert nsid > 0, f"Expected positive nsid, got {nsid}"
        logger.info(f"NVMe namespace ID: {nsid}")

        # Test getting LBA size
        lba_size = raw_device.nvme_lba_size()
        assert lba_size > 0, f"Expected positive lba_size, got {lba_size}"
        logger.info(f"NVMe LBA size: {lba_size} bytes")

    except Exception as e:
        pytest.fail(f"Failed to get NVMe info: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
