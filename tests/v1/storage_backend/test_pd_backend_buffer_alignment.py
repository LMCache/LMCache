# SPDX-License-Identifier: Apache-2.0
"""
Test for buffer size alignment in PD backend.

This test verifies that the PDBackend correctly aligns the buffer size
to be a multiple of align_bytes (chunk size).
"""

# Standard

try:
    # Third Party
    import pytest
except ImportError:
    pytest = None

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.pd_backend import PDBackend


def create_test_metadata(kv_shape=(4, 2, 256, 8, 128)) -> LMCacheMetadata:
    """Create test metadata with configurable KV shape."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )


def test_buffer_size_alignment_cpu():
    """
    Test that buffer size is correctly aligned for CPU device.

    This test verifies the fix for the issue:
    "Buffer size 4317511681 must be a multiple of align bytes 8994816"
    """
    # Create a metadata with KV shape that results in a specific chunk size
    metadata = create_test_metadata(kv_shape=(28, 2, 256, 8, 128))

    # Calculate expected chunk size:
    # 28 * 2 * 256 * 8 * 128 * 2 (bfloat16) = 29360128 bytes

    # Create a config with a buffer size that is NOT a multiple of chunk size
    # This should trigger the alignment logic
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        pd_buffer_size=4317511681,  # NOT a multiple of 29360128
        pd_buffer_device="cpu",
        pd_role="receiver",
        pd_peer_host="localhost",
        pd_peer_init_port=[12345],
        pd_peer_alloc_port=[12346],
        transfer_channel="mock_memory",
    )

    # This should NOT raise an assertion error anymore
    # The buffer size should be automatically aligned
    try:
        backend = PDBackend(config, metadata)

        # Verify that the allocator was initialized successfully
        assert backend.memory_allocator is not None

        # Get the actual buffer size used
        if config.pd_buffer_device == "cpu":
            actual_buffer_size = backend.memory_allocator.cpu_allocator.buffer_size
            align_bytes = backend.memory_allocator.cpu_allocator.align_bytes
        else:
            actual_buffer_size = backend.memory_allocator.gpu_allocator.buffer_size
            align_bytes = backend.memory_allocator.gpu_allocator.align_bytes

        # Verify that the actual buffer size is aligned
        assert actual_buffer_size % align_bytes == 0, (
            f"Buffer size {actual_buffer_size} is not a multiple of "
            f"align bytes {align_bytes}"
        )

        # Verify that the aligned size is <= original size
        assert actual_buffer_size <= config.pd_buffer_size, (
            f"Aligned buffer size {actual_buffer_size} is greater than "
            f"original size {config.pd_buffer_size}"
        )

        # Signal shutdown without joining blocked threads
        backend.running = False

    except AssertionError as e:
        if "must be a multiple of align bytes" in str(e):
            raise AssertionError(f"Buffer size alignment failed: {e}") from e
        else:
            raise


def test_buffer_size_already_aligned():
    """
    Test that when buffer size is already aligned, no adjustment is made.
    """
    metadata = create_test_metadata(kv_shape=(28, 2, 256, 8, 128))

    # Calculate a buffer size that is already a multiple of chunk size
    # chunk_size = 28 * 2 * 256 * 8 * 128 * 2 = 29360128
    # Use exactly 147 chunks
    aligned_buffer_size = 29360128 * 147  # = 4315938816

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        pd_buffer_size=aligned_buffer_size,
        pd_buffer_device="cpu",
        pd_role="receiver",
        pd_peer_host="localhost",
        pd_peer_init_port=[12347],
        pd_peer_alloc_port=[12348],
        transfer_channel="mock_memory",
    )

    # This should work without any alignment
    backend = PDBackend(config, metadata)

    # Verify initialization
    assert backend.memory_allocator is not None

    # Get the actual buffer size
    if config.pd_buffer_device == "cpu":
        actual_buffer_size = backend.memory_allocator.cpu_allocator.buffer_size
    else:
        actual_buffer_size = backend.memory_allocator.gpu_allocator.buffer_size

    # Verify that the size was not changed
    assert actual_buffer_size == aligned_buffer_size

    # Signal shutdown without joining blocked threads
    backend.running = False


if __name__ == "__main__":
    # Run tests
    test_buffer_size_alignment_cpu()
    test_buffer_size_already_aligned()
    print("All tests passed!")
