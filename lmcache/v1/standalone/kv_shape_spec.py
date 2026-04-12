# SPDX-License-Identifier: Apache-2.0
"""Shared KV cache shape specification utilities.

Provides :class:`LayerGroupSpec` and :func:`parse_kvcache_shape_spec`
for parsing multi-group KV shape strings such as
``(2,2,256,4,16):float16:2;(3,2,256,4,4):bfloat16:2``.
"""

# Future
from __future__ import annotations

# Third Party
import torch

# ------------------------------------------------------------------ #
#  dtype mapping                                                       #
# ------------------------------------------------------------------ #

DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


# ------------------------------------------------------------------ #
#  LayerGroupSpec                                                      #
# ------------------------------------------------------------------ #


class LayerGroupSpec:
    """Specification for a layer group with KV shape and dtype.

    Attributes:
        layer_count: Number of layers in this group.
        shape: Shape as tuple of integers, e.g.
            ``(num_layers, kv_dim, num_blocks, num_heads, head_size)``.
        dtype: Data type for this layer group.
    """

    def __init__(
        self,
        layer_count: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.layer_count = layer_count
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        return "LayerGroupSpec(%s):%s:%d" % (self.shape, self.dtype, self.layer_count)


# ------------------------------------------------------------------ #
#  Parser                                                              #
# ------------------------------------------------------------------ #


def parse_kvcache_shape_spec(
    spec_str: str,
) -> list[LayerGroupSpec]:
    """Parse KV shape specification with multiple layer groups.

    Format examples:
    - ``(2,2,256,4,16):float16:2`` (single group)
    - ``(2,2,256,4,16):float16:2;(3,2,256,4,4):bfloat16:2``

    Returns a list of :class:`LayerGroupSpec` objects.
    """
    if not spec_str:
        raise ValueError("KV shape specification cannot be empty")

    groups: list[LayerGroupSpec] = []

    for group_spec in spec_str.split(";"):
        group_spec = group_spec.strip()
        if not group_spec:
            continue

        if not (group_spec.startswith("(") and "):" in group_spec):
            raise ValueError("Invalid group specification format: %s" % group_spec)

        shape_end = group_spec.find(")")
        shape_str = group_spec[1:shape_end]

        remaining = group_spec[shape_end + 2 :]  # Skip "):"
        parts = remaining.split(":")

        if len(parts) != 2:
            raise ValueError("Invalid group specification format: %s" % group_spec)

        dtype_str = parts[0].strip()
        layer_count_str = parts[1].strip()

        try:
            shape_parts = shape_str.split(",")
            shape = tuple(int(part.strip()) for part in shape_parts)
            layer_count = int(layer_count_str)
            dtype = DTYPE_MAP.get(dtype_str.strip().lower(), torch.float16)
            groups.append(LayerGroupSpec(layer_count, shape, dtype))
        except ValueError as e:
            raise ValueError(
                "Invalid number format in group specification: %s" % group_spec
            ) from e

    if not groups:
        raise ValueError("No valid layer groups found in specification")

    return groups
