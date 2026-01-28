# SPDX-License-Identifier: Apache-2.0
"""CUDA kernel Block-first vs K/V-first layout support test.

This test validates that the CUDA kernel correctly handles both:
1. K/V-first layout: [2, num_blocks, block_size, num_heads, head_dim] (BF16)
2. Block-first layout: [num_blocks, 2, block_size, num_heads, head_dim] (FP8)
"""

import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.config import LMCacheEngineMetadata



# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available on this system",
)
def test_kv_first_layout_detection():
    """Test K/V-first layout detection (BF16 style)."""
    # Create metadata for K/V-first layout
    # Shape: [2, num_blocks, block_size, num_heads, head_dim]
    num_layers = 32
    num_blocks = 2561
    block_size = 8
    num_kv_head = 16
    head_size = 128
    chunk_size = 256

    metadata = LMCacheEngineMetadata(
        model_name="test-kvfirst",
        world_size=1,
        worker_id=0,
        fmt="normal",
        kv_shape=(num_layers, 2, chunk_size, num_kv_head, head_size),
        kv_dtype=torch.bfloat16,
        chunk_size=chunk_size,
        use_mla=False,
    )

    # Create connector
    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata)

    # Create mock KV caches with K/V-first layout
    # shape[0] == 2 indicates K/V-first
    kv_caches = [
        torch.empty((2, num_blocks, block_size, num_kv_head, head_size), dtype=torch.bfloat16, device='cuda')
        for _ in range(num_layers)
    ]
    

    # Verify default values
    assert connector.block_size == 8, "Default block_size should be 8"
    assert connector.two_major == True, "Default two_major should be True (K/V-first)"

    # Initialize pointers should detect K/V-first layout
    _ = connector._initialize_pointers(kv_caches)

    # Check detection results
    assert connector.two_major == True, "Should detect K/V-first layout when shape[0] == 2"
    assert connector.block_size == block_size, f"Block size should be {block_size}"
    assert connector.page_buffer_size == num_blocks * block_size, (
        "page_buffer_size should be num_blocks * block_size for K/V-first"
    )


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available on this system",
)
def test_block_first_layout_detection():
    """Test Block-first layout detection (FP8 style)."""
    # Create metadata for Block-first layout
    # Shape: [num_blocks, 2, block_size, num_heads, head_dim]
    num_layers = 32
    num_blocks = 2701
    block_size = 8
    num_kv_head = 16
    head_size = 128
    chunk_size = 256

    metadata = LMCacheEngineMetadata(
        model_name="test-blockfirst",
        world_size=1,
        worker_id=0,
        fmt="normal",
        kv_shape=(num_layers, 2, chunk_size, num_kv_head, head_size),
        kv_dtype=torch.uint8,  # FP8 stored as uint8
        chunk_size=chunk_size,
        use_mla=False,
    )

    # Create connector
    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata)

    # Create mock KV caches with Block-first layout
    # shape[0] != 2 indicates Block-first
    kv_caches = [
        torch.empty((num_blocks, 2, block_size, num_kv_head, head_size), dtype=torch.uint8, device='cuda')
        for _ in range(num_layers)
    ]

    # Initialize pointers should detect Block-first layout
    _ = connector._initialize_pointers(kv_caches)

    # Check detection results
    assert connector.two_major == False, "Should detect Block-first layout when shape[0] != 2"
    assert connector.block_size == block_size, f"Block size should be {block_size}"
    assert connector.page_buffer_size == num_blocks, (
        "page_buffer_size should be num_blocks for Block-first"
    )


def test_page_buffer_size_kvf16():
    """Test page_buffer_size calculation for K/V-first (BF16) layout."""
    num_blocks = 2561
    block_size = 8

    metadata = LMCacheEngineMetadata(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        fmt="normal",
        kv_shape=(32, 2, 256, 16, 128),
        kv_dtype=torch.bfloat16,
        chunk_size=256,
        use_mla=False,
    )

    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata)

    kv_caches = [torch.empty((2, num_blocks, block_size, 16, 128), dtype=torch.bfloat16, device='cuda')]

    _ = connector._initialize_pointers(kv_caches)

    # K/V-first: page_buffer_size = num_blocks * block_size
    expected = num_blocks * block_size
    assert connector.page_buffer_size == expected, (
        f"page_buffer_size should be {expected} for K/V-first, got {connector.page_buffer_size}"
    )


def test_page_buffer_size_block_first_fp8():
    """Test page_buffer_size calculation for Block-first (FP8) layout."""
    num_blocks = 2701
    block_size = 8

    metadata = LMCacheEngineMetadata(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        fmt="normal",
        kv_shape=(32, 2, 256, 16, 128),
        kv_dtype=torch.uint8,
        chunk_size=256,
        use_mla=False,
    )

    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata)

    kv_caches = [torch.empty((num_blocks, 2, block_size, 16, 128), dtype=torch.uint8, device='cuda')]

    _ = connector._initialize_pointers(kv_caches)

    # Block-first: page_buffer_size = num_blocks
    expected = num_blocks
    assert connector.page_buffer_size == expected, (
        f"page_buffer_size should be {expected} for Block-first, got {connector.page_buffer_size}"
    )


def test_block_size_extraction():
    """Test block_size extraction from KV cache shape."""
    block_size_8 = 8
    block_size_16 = 16

    metadata = LMCacheEngineMetadata(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        fmt="normal",
        kv_shape=(32, 2, 256, 16, 128),
        kv_dtype=torch.bfloat16,
        chunk_size=256,
        use_mla=False,
    )

    # Test with block_size = 8
    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata)
    kv_caches = [
        torch.empty((2561, 2, block_size_8, 16, 128), dtype=torch.bfloat16, device='cuda')
    ]
    _ = connector._initialize_pointers(kv_caches)
    assert connector.block_size == block_size_8, f"Block size should be {block_size_8}"

    # Test with block_size = 16
    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata)
    kv_caches = [torch.empty((2701, 2, block_size_16, 16, 128), dtype=torch.uint8, device='cuda')]
    _ = connector._initialize_pointers(kv_caches)
    assert connector.block_size == block_size_16, f"Block size should be {block_size_16}"


def test_detection_idempotency():
    """Test that layout detection is idempotent (same result on repeated calls)."""
    num_blocks = 2701
    block_size = 8

    metadata = LMCacheEngineMetadata(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        fmt="normal",
        kv_shape=(32, 2, 256, 16, 128),
        kv_dtype=torch.uint8,
        chunk_size=256,
        use_mla=False,
    )

    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata)

    kv_caches = [torch.empty((num_blocks, 2, block_size, 16, 128), dtype=torch.uint8, device='cuda')]

    # First detection
    _ = connector._initialize_pointers(kv_caches)
    first_block_size = connector.block_size
    first_two_major = connector.two_major
    first_page_buffer_size = connector.page_buffer_size

    # Second detection (should return cached result)
    _ = connector._initialize_pointers(kv_caches)
    second_block_size = connector.block_size
    second_two_major = connector.two_major
    second_page_buffer_size = connector.page_buffer_size

    # Results should be identical
    assert first_block_size == second_block_size, "block_size should be consistent"
    assert first_two_major == second_two_major, "two_major should be consistent"
    assert first_page_buffer_size == second_page_buffer_size, (
        "page_buffer_size should be consistent"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])