# SPDX-License-Identifier: Apache-2.0
"""Conversions between ``MemoryLayoutDesc`` and the coordinator wire layout."""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.memory_coordinator.api import WireLayout, wire_dtype_itemsize


def dtype_to_wire(dtype: torch.dtype) -> str:
    """Return the canonical wire name of a torch dtype.

    Args:
        dtype: The torch dtype to encode.

    Returns:
        The canonical name, e.g. ``"float16"``.

    Raises:
        ValueError: The dtype has no canonical wire name.
    """
    name = str(dtype).removeprefix("torch.")
    # Fails closed for dtypes outside the fixed wire table.
    wire_dtype_itemsize(name)
    return name


def wire_to_dtype(name: str) -> torch.dtype:
    """Return the torch dtype for a canonical wire name.

    Args:
        name: A canonical dtype name from the wire table.

    Returns:
        The corresponding torch dtype.

    Raises:
        ValueError: The name is not in the wire table or does not resolve
            to a torch dtype.
    """
    wire_dtype_itemsize(name)
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"wire dtype {name!r} does not resolve to a torch dtype")
    return dtype


def layout_to_wire(layout: MemoryLayoutDesc) -> WireLayout:
    """Encode a ``MemoryLayoutDesc`` as the explicit wire representation.

    Args:
        layout: The layout to encode.

    Returns:
        The JSON-safe :class:`WireLayout`.

    Raises:
        ValueError: A dtype in the layout has no canonical wire name.
    """
    return WireLayout(
        shapes=[list(shape) for shape in layout.shapes],
        dtypes=[dtype_to_wire(dtype) for dtype in layout.dtypes],
    )


def wire_to_layout(wire: WireLayout) -> MemoryLayoutDesc:
    """Decode the wire representation back into a ``MemoryLayoutDesc``.

    Args:
        wire: The wire layout to decode.

    Returns:
        The equivalent :class:`MemoryLayoutDesc`.

    Raises:
        ValueError: A dtype name is unknown or shapes and dtypes disagree.
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size(shape) for shape in wire.shapes],
        dtypes=[wire_to_dtype(name) for name in wire.dtypes],
    )
