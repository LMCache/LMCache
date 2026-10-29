# SPDX-License-Identifier: Apache-2.0
"""Reconstruct producer region views over imported native IPC storage."""

# Standard
from collections.abc import Mapping, Sequence
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
    TransferRegionView,
    TransferRegionViews,
)


def _span(shape: Sequence[int], stride: Sequence[int]) -> int:
    """Exclusive extent in elements, relative to a view's first element."""
    if not all(shape):
        return 0
    return 1 + sum((n - 1) * s for n, s in zip(shape, stride, strict=True))


def _validate_view(view: TransferRegionView, native_span: int) -> None:
    """Validate a dense/permuted/padded plane before constructing a view."""
    shape, stride, offset = view["shape"], view["stride"], view["offset"]
    if (
        len(shape) != 4
        or len(stride) != 4
        or any(type(n) is not int or n <= 0 for n in shape)
        or any(type(s) is not int or s < 0 for s in stride)
        or type(offset) is not int
        or offset < 0
    ):
        raise ValueError("Invalid transfer-region shape, stride or relative offset")
    extent = 1
    for step, size in sorted(
        (s, n) for n, s in zip(shape, stride, strict=True) if n > 1
    ):
        if step < extent:
            raise ValueError(
                "Transfer regions require non-overlapping separable strides"
            )
        extent += (size - 1) * step
    if offset + _span(shape, stride) > native_span:
        raise ValueError("Transfer region exceeds its registered native view")


def describe_transfer_regions(
    native_caches: Mapping[str, torch.Tensor],
    kv_caches: Mapping[str, DiscoverableKVCache],
) -> TransferRegionViews:
    """Describe canonical planes relative to native views before IPC wrapping.

    Args:
        native_caches: Engine-owned native tensors, in registration order.
        kv_caches: Matching native tensors or exported NHD K/V plane lists.

    Returns:
        Version-one per-layer shape/stride/relative-offset recipes.

    Raises:
        ValueError: Layer order, geometry, ownership or view bounds are invalid.
    """
    if tuple(native_caches) != tuple(kv_caches):
        raise ValueError("Native and region registration layer order must match")
    layers: list[list[TransferRegionView] | None] = []
    for name, native in native_caches.items():
        cache = kv_caches[name]
        if isinstance(cache, torch.Tensor):
            if cache is not native:
                raise ValueError("Unexported layers must retain their native tensor")
            layers.append(None)
            continue
        if len(cache) != 2 or not all(isinstance(p, torch.Tensor) for p in cache):
            raise ValueError("Transfer regions require exactly two tensor planes")
        planes = cast(list[torch.Tensor], cache)
        recipes: list[TransferRegionView] = []
        for plane in planes:
            if (
                plane.dtype != native.dtype
                or plane.device != native.device
                or plane.untyped_storage().data_ptr()
                != native.untyped_storage().data_ptr()
            ):
                raise ValueError("Transfer regions must alias native cache storage")
            view: TransferRegionView = {
                "shape": cast(tuple[int, int, int, int], tuple(plane.shape)),
                "stride": cast(tuple[int, int, int, int], tuple(plane.stride())),
                "offset": plane.storage_offset() - native.storage_offset(),
            }
            _validate_view(view, _span(native.shape, native.stride()))
            recipes.append(view)
        if recipes[0]["shape"] != recipes[1]["shape"]:
            raise ValueError("Transfer-region K/V geometry must match")
        layers.append(recipes)
    return {"version": 1, "layers": layers}


def restore_transfer_regions(
    native_tensors: Sequence[torch.Tensor], layout_hints: LayoutHints | None
) -> list[DiscoverableKVCache]:
    """Materialize canonical views after opening native IPC handles.

    Offsets are relative to each imported native view, not its allocation base;
    this also covers Raw/VMM CUDA wrappers whose storage origin is rebased.

    Args:
        native_tensors: Imported native tensors in registration order.
        layout_hints: Optional version-one region recipes from the worker.

    Returns:
        Original native tensors or pairs of zero-copy NHD region views.

    Raises:
        ValueError: Declared recipes are missing, malformed or out of bounds.
    """
    recipes = layout_hints.get("transfer_region_views") if layout_hints else None
    if recipes is None:
        if layout_hints and layout_hints.get("spec_transfer_regions"):
            raise ValueError("Spec transfer regions require IPC view recipes")
        return list(native_tensors)
    if recipes["version"] != 1 or len(recipes["layers"]) != len(native_tensors):
        raise ValueError("Invalid transfer-region version or layer count")
    result: list[DiscoverableKVCache] = []
    for native, layer in zip(native_tensors, recipes["layers"], strict=True):
        if layer is None:
            result.append(native)
            continue
        if len(layer) != 2 or layer[0]["shape"] != layer[1]["shape"]:
            raise ValueError("Transfer regions require matching K/V planes")
        planes: list[DiscoverableKVCache] = []
        for view in layer:
            _validate_view(view, _span(native.shape, native.stride()))
            planes.append(
                torch.as_strided(
                    native,
                    view["shape"],
                    view["stride"],
                    native.storage_offset() + view["offset"],
                )
            )
        result.append(planes)
    return result
