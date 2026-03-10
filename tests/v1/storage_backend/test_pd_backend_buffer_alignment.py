# SPDX-License-Identifier: Apache-2.0
"""
Test for buffer size alignment in PD backend.

This test verifies that the PDBackend correctly aligns the buffer size
to be a multiple of align_bytes (chunk size).
"""

# Standard
from unittest.mock import MagicMock, patch

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


def _get_allocator(backend, config):
    """Get the appropriate memory allocator based on the buffer device config."""
    if config.pd_buffer_device == "cpu":
        return backend.memory_allocator.cpu_allocator
    return backend.memory_allocator.gpu_allocator


@patch.object(PDBackend, "_init_receiver")
@patch(
    "lmcache.v1.storage_backend.pd_backend.CreateTransferChannel",
    return_value=MagicMock(),
)
@patch(
    "lmcache.v1.storage_backend.pd_backend.get_zmq_context", return_value=MagicMock()
)
def test_buffer_size_exact_alignment(_mock_ctx, _mock_channel, _mock_receiver):
    """
    Test that buffer size is correctly aligned to the exact expected size.
    """

    # Create a metadata with KV shape that results in a specific chunk size
    metadata = create_test_metadata(kv_shape=(28, 2, 256, 8, 128))

    # Calculate expected chunk size:
    # 28 * 2 * 256 * 8 * 128 * 2 (bfloat16) = 29360128 bytes
    expected_chunk_size = 29360128

    # Create a config with a buffer size that is NOT a multiple of chunk size
    # Original size: 4317511681 bytes
    # Chunks count: 4317511681 // 29360128 = 147 chunks
    # Expected aligned size: 147 * 29360128 = 4315938816 bytes
    expected_aligned_size = 4315938816

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

    backend = PDBackend(config, metadata)
    try:
        # Get the actual buffer size used
        allocator = _get_allocator(backend, config)
        actual_buffer_size = allocator.buffer_size
        align_bytes = allocator.align_bytes

        assert align_bytes == expected_chunk_size, (
            f"Expected chunk size {expected_chunk_size}, got {align_bytes}"
        )

        assert actual_buffer_size == expected_aligned_size, (
            f"Expected exact aligned size {expected_aligned_size}, "
            f"but got {actual_buffer_size}"
        )
    finally:
        backend.close()
