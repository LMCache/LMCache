# SPDX-License-Identifier: Apache-2.0
"""Tests for MLA multi-group KV cache support.

MLA (Multi-Latent Attention) models like GLM-5 and DeepSeek V3 store KV cache
as two separate groups with different dtypes:
  - Group 0: K_rope (e.g., 78 layers, uint8/FP8, head_dim=132)
  - Group 1: Latent KV (e.g., 78 layers, bfloat16, head_dim=576)

These tests verify that LMCache correctly handles multi-group allocation,
tensor access, and pointer initialization without crashing.

Related issues: #2774, #2881
"""

# Standard
from typing import List

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    TensorMemoryObj,
    get_size_bytes,
)
from lmcache.v1.metadata import LMCacheMetadata

# Local
from .utils import dumb_metadata


# -- Helpers ------------------------------------------------------------------


def _mla_shapes_and_dtypes(
    num_layers: int = 78,
    chunk_size: int = 256,
    k_rope_dim: int = 132,
    latent_dim: int = 576,
) -> tuple[list[torch.Size], list[torch.dtype]]:
    """Return (shapes, dtypes) mimicking a two-group MLA KV cache.

    Group 0: K_rope  — [1, num_layers, chunk_size, k_rope_dim], uint8
    Group 1: Latent  — [1, num_layers, chunk_size, latent_dim], bfloat16
    """
    shapes = [
        torch.Size([1, num_layers, chunk_size, k_rope_dim]),
        torch.Size([1, num_layers, chunk_size, latent_dim]),
    ]
    dtypes = [torch.uint8, torch.bfloat16]
    return shapes, dtypes


def _allocate_mla_memory_obj(
    num_layers: int = 78,
    chunk_size: int = 256,
    k_rope_dim: int = 132,
    latent_dim: int = 576,
    device: str = "cpu",
) -> TensorMemoryObj:
    """Allocate a TensorMemoryObj with two-group MLA layout."""
    shapes, dtypes = _mla_shapes_and_dtypes(
        num_layers, chunk_size, k_rope_dim, latent_dim
    )
    allocator = AdHocMemoryAllocator(device=device)
    memory_obj = allocator.allocate(shapes, dtypes, fmt=MemoryFormat.KV_MLA_FMT)
    assert memory_obj is not None
    return memory_obj


# -- Tests: memory_management.py ----------------------------------------------


class TestMLAMultiGroupTensorAccess:
    """Verify TensorMemoryObj handles multi-group shapes correctly."""

    def test_tensor_property_does_not_crash(self) -> None:
        """The .tensor property must not raise when multiple shapes exist.

        Previously this crashed with:
            RuntimeError: shape '[1, 78, 256, 132]' is invalid for input
            of size 25638912
        """
        obj = _allocate_mla_memory_obj()
        tensor = obj.tensor
        assert tensor is not None

    def test_tensor_returns_flat_buffer_for_multi_group(self) -> None:
        """When multiple groups exist, .tensor returns a flat uint8 buffer."""
        obj = _allocate_mla_memory_obj()
        tensor = obj.tensor
        assert tensor is not None
        assert tensor.dtype == torch.uint8
        assert tensor.ndim == 1

    def test_tensor_returns_shaped_for_single_group(self) -> None:
        """Single-group (non-MLA) still returns a shaped tensor."""
        shapes = [torch.Size([2, 32, 256, 128])]
        dtypes = [torch.bfloat16]
        allocator = AdHocMemoryAllocator(device="cpu")
        obj = allocator.allocate(shapes, dtypes, fmt=MemoryFormat.KV_2LTD)
        assert obj is not None
        tensor = obj.tensor
        assert tensor is not None
        assert tensor.shape == torch.Size([2, 32, 256, 128])
        assert tensor.dtype == torch.bfloat16

    def test_get_tensor_per_group(self) -> None:
        """get_tensor(index) returns correctly shaped per-group views."""
        num_layers = 78
        chunk_size = 256
        k_rope_dim = 132
        latent_dim = 576

        obj = _allocate_mla_memory_obj(
            num_layers=num_layers,
            chunk_size=chunk_size,
            k_rope_dim=k_rope_dim,
            latent_dim=latent_dim,
        )

        group0 = obj.get_tensor(0)
        assert group0 is not None
        assert group0.shape == torch.Size([1, num_layers, chunk_size, k_rope_dim])
        assert group0.dtype == torch.uint8

        group1 = obj.get_tensor(1)
        assert group1 is not None
        assert group1.shape == torch.Size([1, num_layers, chunk_size, latent_dim])
        assert group1.dtype == torch.bfloat16

    def test_raw_buffer_size_equals_sum_of_groups(self) -> None:
        """The raw buffer should be exactly large enough for both groups."""
        num_layers = 78
        chunk_size = 256
        k_rope_dim = 132
        latent_dim = 576

        shapes, dtypes = _mla_shapes_and_dtypes(
            num_layers, chunk_size, k_rope_dim, latent_dim
        )
        expected_size = get_size_bytes(shapes, dtypes)

        obj = _allocate_mla_memory_obj(
            num_layers=num_layers,
            chunk_size=chunk_size,
            k_rope_dim=k_rope_dim,
            latent_dim=latent_dim,
        )
        assert obj.get_size() == expected_size

    def test_group_views_do_not_overlap(self) -> None:
        """Per-group tensor views should reference non-overlapping memory."""
        obj = _allocate_mla_memory_obj()

        group0 = obj.get_tensor(0)
        group1 = obj.get_tensor(1)
        assert group0 is not None
        assert group1 is not None

        # Write distinct values into each group
        group0.fill_(42)
        group1.fill_(0)

        # Verify group0 data wasn't overwritten
        assert group0.flatten()[0].item() == 42


# -- Tests: gpu_connectors.py -------------------------------------------------


class TestMLAMultiGroupPointerInit:
    """Verify VLLMPagedMemGPUConnectorV2 handles multi-group kv_caches."""

    def test_initialize_pointers_expands_for_multi_group(self) -> None:
        """_initialize_pointers should handle 156 entries when num_layers=78.

        Previously this crashed with:
            ValueError: could not broadcast input array from shape (156,)
            into shape (78,)
        """
        # Skip if CUDA is not available (connector needs GPU)
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import (
            VLLMPagedMemGPUConnectorV2,
        )

        num_layers = 78
        connector = VLLMPagedMemGPUConnectorV2(
            hidden_dim_size=576,
            num_layers=num_layers,
            use_mla=True,
        )

        # Simulate MLA: 156 kv_cache tensors (2 groups × 78 layers)
        # Group 0: K_rope — small tensors, uint8
        # Group 1: Latent — larger tensors, bfloat16
        mock_kv_caches: List[torch.Tensor] = []
        for _ in range(num_layers):
            mock_kv_caches.append(
                torch.empty(100, 64, 132, dtype=torch.uint8, device="cuda:0")
            )
        for _ in range(num_layers):
            mock_kv_caches.append(
                torch.empty(100, 64, 576, dtype=torch.bfloat16, device="cuda:0")
            )

        # This should not crash
        pointers = connector._initialize_pointers(mock_kv_caches)
        assert pointers.shape[0] == 156
        assert connector.num_layers == 156
