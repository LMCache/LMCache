# SPDX-License-Identifier: Apache-2.0
"""Conversions between L1 types and memory-orchestrator wire types."""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.memory_orchestrator.api import WireLayout


def _dtype_from_name(name: str) -> torch.dtype:
    """Resolve a wire dtype name written by another server."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype {name!r} in a shared-L1 layout")
    return dtype


def layout_to_wire(layout: MemoryLayoutDesc) -> WireLayout:
    """Return ``layout`` as plain integer dims and dtype names."""
    return WireLayout(
        shapes=tuple(tuple(int(dim) for dim in shape) for shape in layout.shapes),
        dtypes=tuple(str(dtype).removeprefix("torch.") for dtype in layout.dtypes),
    )


def layout_from_wire(layout: WireLayout) -> MemoryLayoutDesc:
    """Return the memory layout of a wire layout.

    Raises:
        ValueError: A dtype name is unknown to this torch build.
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size(shape) for shape in layout.shapes],
        dtypes=[_dtype_from_name(name) for name in layout.dtypes],
    )
