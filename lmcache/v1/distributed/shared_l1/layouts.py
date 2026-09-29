# SPDX-License-Identifier: Apache-2.0
"""Conversions between ``MemoryLayoutDesc`` and the coordinator wire layout."""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.memory_coordinator.api import WireLayout, wire_dtype_itemsize


def dtype_to_wire(dtype: torch.dtype) -> str:
    """Return dtype's canonical wire name; reject unsupported dtypes with ValueError."""
    name = str(dtype).removeprefix("torch.")
    wire_dtype_itemsize(name)
    return name


def wire_to_dtype(name: str) -> torch.dtype:
    """Resolve a canonical name to torch.dtype; raise ValueError if unsupported."""
    wire_dtype_itemsize(name)
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"wire dtype {name!r} does not resolve to a torch dtype")
    return dtype


def layout_to_wire(layout: MemoryLayoutDesc) -> WireLayout:
    """Encode layout as WireLayout; raise ValueError for unsupported dtypes."""
    return WireLayout(
        shapes=[list(shape) for shape in layout.shapes],
        dtypes=[dtype_to_wire(dtype) for dtype in layout.dtypes],
    )


def wire_to_layout(wire: WireLayout) -> MemoryLayoutDesc:
    """Decode wire layout; raise ValueError for invalid dtypes or shapes."""
    return MemoryLayoutDesc(
        shapes=[torch.Size(shape) for shape in wire.shapes],
        dtypes=[wire_to_dtype(name) for name in wire.dtypes],
    )
