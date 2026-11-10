# SPDX-License-Identifier: Apache-2.0
"""Shared Device-DAX mapping without an allocator interface."""

# Standard
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock
import mmap
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager.shared_devdax_region import (
    SharedDevDaxRegion,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import DevDaxMemoryAllocator
from lmcache.v1.memory_orchestrator.api import RegionContract
from lmcache.v1.platform import current_device_spec
from tests.v1.distributed.shared_l1.utils import CAPACITY, region_file

pytestmark = pytest.mark.no_shared_allocator
ALIGN = OFFSET = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.uint8])


@pytest.fixture
def region_config(tmp_path: Path) -> tuple[L1MemoryManagerConfig, RegionContract]:
    """A file-backed region with a nonzero device offset."""
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
    return config, contract


def test_views_alias_the_region_and_reject_invalid_extents(
    region_config: tuple[L1MemoryManagerConfig, RegionContract],
) -> None:
    config, contract = region_config
    assert config.devdax_path is not None
    region = Path(config.devdax_path)
    with closing(SharedDevDaxRegion(config, contract)) as mapping:
        for name in ("allocate", "free", "get_memory_usage", "memcheck"):
            assert not hasattr(mapping, name)
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


@pytest.mark.parametrize("registration_succeeds", [False, True])
def test_registration_failure_and_close_release_resources(
    region_config: tuple[L1MemoryManagerConfig, RegionContract],
    registration_succeeds: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed registration and repeated close both release the opened handles."""
    config, contract = region_config
    opened: list[tuple[int, mmap.mmap]] = []
    open_mapping = DevDaxMemoryAllocator.open_devdax_mapping

    def record_mapping(
        path: str, size: int, offset: int
    ) -> tuple[int, mmap.mmap, object, torch.Tensor]:
        result = open_mapping(path, size, offset)
        opened.append((result[0], result[1]))
        return result

    monkeypatch.setattr(
        DevDaxMemoryAllocator, "open_devdax_mapping", staticmethod(record_mapping)
    )
    monkeypatch.setattr(
        current_device_spec, "pin_memory", Mock(return_value=registration_succeeds)
    )
    unpin = Mock()
    monkeypatch.setattr(current_device_spec, "unpin_memory", unpin)
    if registration_succeeds:
        region = SharedDevDaxRegion(config, contract)
        region.close()
        region.close()
        unpin.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="must be GPU-registered"):
            SharedDevDaxRegion(config, contract)
        unpin.assert_not_called()
    assert len(opened) == 1
    fd, mapped = opened[0]
    assert mapped.closed
    with pytest.raises(OSError):
        os.fstat(fd)
