# SPDX-License-Identifier: Apache-2.0
"""Mapping aliasing, bounds and failure cleanup."""

# Standard
from collections.abc import Iterator
from pathlib import Path
import mmap
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.shared_l1 import media
from lmcache.v1.distributed.shared_l1.media import RegionMedium

pytestmark = pytest.mark.no_shared_allocator
ALIGN, CAPACITY = 4096, 65536
LAYOUT = MemoryLayoutDesc([torch.Size([2, 1000])], [torch.float16])


@pytest.fixture
def medium(tmp_path: Path) -> Iterator[RegionMedium]:
    path = tmp_path / "backing"
    path.write_bytes(b"\0" * ALIGN)
    mapped = RegionMedium(str(path), CAPACITY, ALIGN, L1BackendType.DRAM)
    try:
        yield mapped
    finally:
        mapped.close()


def test_views_share_bytes_and_backing(medium: RegionMedium, tmp_path: Path) -> None:
    assert Path(medium.path).stat().st_size == CAPACITY
    first, second = medium.view(0, ALIGN, LAYOUT), medium.view(3 * ALIGN, ALIGN, LAYOUT)
    assert first.tensor is not None and second.tensor is not None
    assert second.data_ptr == first.data_ptr + 3 * ALIGN
    second.tensor.fill_(1.5)
    again = medium.view(3 * ALIGN, ALIGN, LAYOUT)
    assert torch.all(again.tensor == 1.5) and torch.all(first.tensor == 0)
    assert Path(medium.path).read_bytes()[3 * ALIGN : 4 * ALIGN] != b"\0" * ALIGN
    alias = tmp_path / "alias"
    alias.symlink_to(medium.path)
    assert medium.owns_device(str(alias))
    assert not medium.owns_device(str(tmp_path / "absent"))


@pytest.mark.parametrize(
    "offset,length",
    [(1, ALIGN), (-ALIGN, ALIGN), (0, 2048), (CAPACITY - ALIGN, 2 * ALIGN)],
)
def test_invalid_extents(medium: RegionMedium, offset: int, length: int) -> None:
    with pytest.raises(ValueError, match="does not fit"):
        medium.view(offset, length, LAYOUT)


def test_backing_validation(
    medium: RegionMedium, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="regular file"):
        RegionMedium("/dev/null", CAPACITY, ALIGN, L1BackendType.DRAM)
    with pytest.raises(ValueError, match="character device"):
        RegionMedium(medium.path, CAPACITY, ALIGN, L1BackendType.DEVDAX)
    with pytest.raises(ValueError, match="DRAM or DEVDAX"):
        RegionMedium(medium.path, CAPACITY, ALIGN, L1BackendType.GDS)
    with pytest.raises(OSError):
        RegionMedium("/dev/null", CAPACITY, ALIGN, L1BackendType.DEVDAX)
    monkeypatch.setattr(Path, "read_text", lambda path: str(CAPACITY // 2))
    with pytest.raises(ValueError, match="less than the region capacity"):
        RegionMedium("/dev/null", CAPACITY, ALIGN, L1BackendType.DEVDAX)


def test_failed_registration_closes_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings: list[tuple[int, mmap.mmap]] = []
    open_mapping = media.open_devdax_mapping

    def track(path: str, size: int) -> tuple[int, mmap.mmap, object, torch.Tensor]:
        result = open_mapping(path, size)
        mappings.append(result[:2])
        return result

    monkeypatch.setattr(media, "open_devdax_mapping", track)
    monkeypatch.setattr(media.current_device_spec, "pin_memory", lambda *args: False)
    with pytest.raises(RuntimeError, match="requires GPU-registered memory"):
        RegionMedium(str(tmp_path / "backing"), CAPACITY, ALIGN, L1BackendType.DRAM)
    fd, mapping = mappings.pop()
    assert mapping.closed
    with pytest.raises(OSError):
        os.fstat(fd)
