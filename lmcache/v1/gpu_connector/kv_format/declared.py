# SPDX-License-Identifier: Apache-2.0
"""Consume producer axis semantics without guessing axes from dimension sizes."""

# Standard
from collections.abc import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
    normalize_declared_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
    TensorLayoutHints,
)
import lmcache.lmcache_native as lmcache_native

_ROLES = {"BLOCK", "KV", "HEAD", "STATE", "HEAD_DIM", "CONTENT", "SINGLETON"}
_FORMATS = {
    (
        "KV",
        "BLOCK",
        "STATE",
        "HEAD",
        "HEAD_DIM",
    ): lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    (
        "KV",
        "BLOCK",
        "HEAD",
        "STATE",
        "HEAD_DIM",
    ): lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
    (
        "BLOCK",
        "KV",
        "STATE",
        "HEAD",
        "HEAD_DIM",
    ): lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
    (
        "BLOCK",
        "KV",
        "HEAD",
        "STATE",
        "HEAD_DIM",
    ): lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ("BLOCK", "STATE", "HEAD_DIM"): lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
    ("BLOCK", "STATE", "CONTENT"): lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
    (
        "BLOCK",
        "HEAD",
        "STATE",
        "CONTENT",
    ): lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS,
    (
        "BLOCK",
        "STATE",
        "HEAD",
        "CONTENT",
    ): lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS,
}


def tensor_axes(hints: LayoutHints, num_layers: int) -> list[list[str] | None]:
    """Validate the registration envelope and return its per-layer declarations.

    Args:
        hints: Engine registration hints, possibly without declarations.
        num_layers: Number of entries in the registered cache list.

    Returns:
        Axes per layer, with None for unspecified layers.

    Raises:
        ValueError: Unsupported schema version or mismatched layer count.
    """
    layouts = hints.get("tensor_layouts")
    if layouts is None:
        return [None] * num_layers
    if layouts["version"] != 1:
        raise ValueError("Unsupported tensor_layouts version")
    if len(layouts["axes"]) != num_layers:
        raise ValueError("tensor_layouts must match the registered layer count")
    return layouts["axes"]


def validate_tensor_axes(tensor: torch.Tensor, axes: Sequence[str]) -> None:
    """Validate semantic roles against a registered tensor, without reading data.

    Args:
        tensor: The tensor view described by axes.
        axes: Producer role names in tensor indexing order.

    Raises:
        ValueError: Unknown/duplicate roles, rank mismatch or invalid fixed size.
    """
    if len(axes) != tensor.ndim:
        raise ValueError("Tensor rank does not match declared KV cache axes")
    named = [axis for axis in axes if axis != "SINGLETON"]
    if set(axes) - _ROLES or len(named) != len(set(named)):
        raise ValueError(f"Invalid declared KV cache axes: {axes}")
    for axis, size in zip(axes, tensor.shape, strict=True):
        if axis == "KV" and size != 2:
            raise ValueError("Declared KV axis must have size 2")
        if axis == "SINGLETON" and size != 1:
            raise ValueError("Declared SINGLETON axis must have size 1")


def normalize_declared_caches(
    caches: list[DiscoverableKVCache], layouts: TensorLayoutHints
) -> tuple[list[DiscoverableKVCache], TensorLayoutHints]:
    """Normalize registered views and return declarations for the new views.

    Args:
        caches: Original per-layer views for this request.
        layouts: Immutable registration snapshot describing those original views.

    Returns:
        Zero-copy views and transformed declarations in the same layer order.

    Raises:
        ValueError: Invalid declaration or unsupported tensor strides/structure.
    """
    roles = tensor_axes({"tensor_layouts": layouts}, len(caches))
    normalized: list[DiscoverableKVCache] = []
    output_axes: list[list[str] | None] = []
    for cache, axes in zip(caches, roles, strict=True):
        if axes is None:
            normalized.append(attempt_permute_to_contiguous_view(cache))
            output_axes.append(None)
        else:
            if not isinstance(cache, torch.Tensor):
                raise ValueError("Declared layout requires a single tensor per layer")
            validate_tensor_axes(cache, axes)
            view, ordered_axes = normalize_declared_tensor(cache, axes)
            normalized.append(view)
            output_axes.append(ordered_axes)
    return normalized, {"version": 1, "axes": output_axes}


def select_layout_hints(
    hints: LayoutHints | None, indices: Sequence[int]
) -> LayoutHints | None:
    """Select declarations in the same order as a subset of registered layers.

    Args:
        hints: Full registration hints, or None.
        indices: Selected layer indices, in the new tensor order.

    Returns:
        Independent hints for that subset; legacy hints pass through unchanged.
    """
    if hints is None or "tensor_layouts" not in hints:
        return hints
    selected = hints.copy()
    layouts = hints["tensor_layouts"]
    selected["tensor_layouts"] = {
        "version": layouts["version"],
        "axes": [layouts["axes"][i] for i in indices],
    }
    return selected


def resolve_declared_tensor(
    tensor: torch.Tensor, axes: Sequence[str]
) -> tuple[lmcache_native.EngineKVFormat, torch.Tensor]:
    """Normalize a declared view and map its roles to an existing kernel format.

    Args:
        tensor: Registered per-layer tensor, before normalization.
        axes: Semantic role names describing that exact tensor.

    Returns:
        Existing EngineKVFormat and its zero-copy tensor view.

    Raises:
        ValueError: Invalid axes, unsupported role order or unsupported strides.

    CONTENT stays opaque. Its encoding is not specified by these axis labels;
    the existing uint8 DSA row convention still selects the blocked-page format.
    """
    validate_tensor_axes(tensor, axes)
    normalized, ordered_axes = normalize_declared_tensor(tensor, axes)
    fmt = _FORMATS.get(tuple(ordered_axes))
    if fmt is None:
        raise ValueError(f"Unsupported declared KV cache axes: {ordered_axes}")
    if "KV" not in ordered_axes and normalized.dtype == torch.uint8:
        if normalized.shape[-1] == 132:
            if "HEAD" in ordered_axes:
                head_axis = ordered_axes.index("HEAD")
                if normalized.shape[head_axis] != 1:
                    raise ValueError("Blocked indexer cache requires one head")
                normalized = normalized.squeeze(head_axis)
            return lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS, normalized
    return fmt, normalized
