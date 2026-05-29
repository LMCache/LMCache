# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.layout_desc_registry import LayoutDescRegistry

pytestmark = pytest.mark.no_shared_allocator


def _layout(width: int) -> MemoryLayoutDesc:
    """Create a distinct memory layout descriptor for registry tests."""
    return MemoryLayoutDesc(shapes=[torch.Size([width])], dtypes=[torch.float32])


def test_unregister_removes_only_matching_owner() -> None:
    """Unregistering one worker must preserve sibling layouts for the same key."""
    registry = LayoutDescRegistry()
    first_layout = _layout(1)
    second_layout = _layout(2)

    registry.register("model", 1, first_layout, instance_id=101)
    registry.register("model", 1, second_layout, instance_id=102)

    registry.unregister("model", 1, instance_id=101)

    assert registry.find("model", 1) is second_layout

    registry.unregister("model", 1, instance_id=102)

    assert registry.find("model", 1) is None


def test_find_returns_first_available_layout() -> None:
    """Lookup should return an available descriptor while owners remain live."""
    registry = LayoutDescRegistry()
    first_layout = _layout(1)
    second_layout = _layout(2)

    registry.register("model", 1, first_layout, instance_id=201)
    registry.register("model", 1, second_layout, instance_id=301)

    assert registry.find("model", 1) is first_layout

    registry.unregister("model", 1, instance_id=201)

    assert registry.find("model", 1) is second_layout
