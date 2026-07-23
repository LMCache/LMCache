# SPDX-License-Identifier: Apache-2.0
"""Tests for public distributed API layout descriptors."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectGroupLayoutDesc,
)


def _layout(num_tokens: int) -> MemoryLayoutDesc:
    """Create a distinct layout for object-group tests."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([num_tokens, 4])],
        dtypes=[torch.float16],
    )


def test_object_group_layout_requires_at_least_one_layout() -> None:
    """An empty object-group descriptor is invalid."""
    with pytest.raises(ValueError, match="at least one layout"):
        ObjectGroupLayoutDesc(layouts=())


@pytest.mark.parametrize("object_group_id", [-1, 2])
def test_object_group_layout_rejects_invalid_group_id(
    object_group_id: int,
) -> None:
    """Object-group lookup rejects negative and out-of-range IDs."""
    layouts = ObjectGroupLayoutDesc(layouts=(_layout(1), _layout(2)))

    with pytest.raises(ValueError, match="outside"):
        layouts.get_layout(object_group_id)


def test_object_group_layout_selects_distinct_group_layouts() -> None:
    """Each object-group ID resolves to its corresponding layout."""
    first = _layout(1)
    second = _layout(2)
    layouts = ObjectGroupLayoutDesc(layouts=(first, second))

    assert layouts.num_object_groups == 2
    assert layouts.get_layout(0) is first
    assert layouts.get_layout(1) is second


def test_uniform_layout_expands_to_requested_group_count() -> None:
    """Uniform expansion applies the same layout to every group."""
    layout = _layout(1)

    layouts = ObjectGroupLayoutDesc.from_uniform_layout(
        layout_desc=layout,
        num_object_groups=3,
    )

    assert layouts.num_object_groups == 3
    assert all(group_layout is layout for group_layout in layouts.layouts)


@pytest.mark.parametrize("num_object_groups", [0, -1])
def test_uniform_layout_requires_positive_group_count(
    num_object_groups: int,
) -> None:
    """Uniform expansion rejects non-positive group counts."""
    with pytest.raises(ValueError, match="must be positive"):
        ObjectGroupLayoutDesc.from_uniform_layout(
            layout_desc=_layout(1),
            num_object_groups=num_object_groups,
        )
