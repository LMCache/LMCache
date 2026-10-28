# SPDX-License-Identifier: Apache-2.0
"""Translate vLLM's producer descriptors into engine-independent wire hints."""

# Standard
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Protocol

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.declared import validate_tensor_axes
from lmcache.v1.gpu_connector.kv_format.types import TensorLayoutHints


class TensorLayout(Protocol):
    """Structural subset of vLLM's KVCacheTensorLayout; usable with older vLLM."""

    @property
    def axes(self) -> Sequence[Enum]:
        """Semantic axes in tensor indexing order."""
        ...


def encode_tensor_layouts(
    kv_caches: Mapping[str, torch.Tensor], layouts: Mapping[str, TensorLayout]
) -> TensorLayoutHints:
    """Snapshot producer descriptors in registration order for the MP wire.

    Args:
        kv_caches: Original registered layer tensors, before connector edits.
        layouts: vLLM descriptors; absent layers remain unspecified.

    Returns:
        Version 1 data-only layout hints using enum names, not ordinal values.

    Raises:
        ValueError: Unknown layer, non-tensor entry or invalid axis declaration.
    """
    if layouts.keys() - kv_caches.keys():
        raise ValueError("Layout supplied for an unregistered KV cache layer")
    axes_per_layer: list[list[str] | None] = []
    for name, tensor in kv_caches.items():
        layout = layouts.get(name)
        if layout is None:
            axes_per_layer.append(None)
            continue
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Producer layout requires a single tensor per layer")
        axes = [axis.name for axis in layout.axes]
        validate_tensor_axes(tensor, axes)
        axes_per_layer.append(axes)
    return {"version": 1, "axes": axes_per_layer}
