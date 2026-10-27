# SPDX-License-Identifier: Apache-2.0
"""DevDaxL1MemoryManager in shared mode: one mapping, bounds-checked views."""

# Standard
from contextlib import closing
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.memory_orchestrator.api import RegionContract
from tests.v1.distributed.shared_l1.utils import CAPACITY, region_file

pytestmark = pytest.mark.no_shared_allocator
ALIGN = OFFSET = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.uint8])


def test_views_alias_the_region_and_reject_invalid_extents(tmp_path: Path) -> None:
    region = region_file(tmp_path / "region", OFFSET + CAPACITY)
    contract = RegionContract(
        region_id="region",
        region_epoch=1,
        capacity_bytes=CAPACITY,
        alignment_bytes=ALIGN,
        layout_fingerprint=b"",
        visibility_mode="coherent",
        max_batch_entries=0,
        reset_required=False,
    )
    config = L1MemoryManagerConfig(
        CAPACITY,
        use_lazy=False,
        shm_name="",
        devdax_path=str(region),
        devdax_offset_in_bytes=OFFSET,
    )
    with closing(DevDaxL1MemoryManager(config, shared_region=contract)) as mapping:
        first = mapping.view(ALIGN, ALIGN, LAYOUT)
        second = mapping.view(ALIGN, ALIGN, LAYOUT)
        assert first.tensor is not None and second.tensor is not None
        first.tensor.fill_(5)
        assert torch.all(second.tensor == 5)
        assert first.get_l1_manager() is None and first.parent() is None
        size = first.get_size()
        assert region.read_bytes()[OFFSET + ALIGN :][:size] == bytes([5]) * size
        assert mapping.owns_range(str(region), OFFSET, 1)
        assert not mapping.owns_range(str(region), 0, OFFSET)
        # Negative, misaligned, outside and short extents; LAYOUT needs 1024 bytes.
        for start, length in [(-ALIGN, ALIGN), (1, ALIGN), (CAPACITY, ALIGN), (0, 512)]:
            with pytest.raises(ValueError, match="does not fit"):
                mapping.view(start, length, LAYOUT)
