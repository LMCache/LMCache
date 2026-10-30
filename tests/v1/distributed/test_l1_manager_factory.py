# SPDX-License-Identifier: Apache-2.0
"""Selecting the L1 manager medium from its configuration."""

# Standard
from collections.abc import Iterator
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.config import (
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l1_manager import (
    DevDaxL1Manager,
    DramL1Manager,
    create_l1_manager,
    create_l1_managers,
)
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

LOCAL = MemoryResidency.LOCAL


@pytest.fixture
def fake_pinned_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)


@pytest.fixture
def managers() -> Iterator[list[L1ManagerInterface]]:
    created: list[L1ManagerInterface] = []
    yield created
    for manager in reversed(created):
        manager.close()


def _dram_config(tag: str = "_default") -> L1ManagerConfig:
    return L1ManagerConfig(
        L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False, shm_name=""),
        tag=tag,
    )


def _devdax_file(tmp_path: Path, name: str, size: int = 8192) -> str:
    path = tmp_path / name
    path.write_bytes(b"\0" * size)
    return str(path)


def _devdax_config(path: str, tag: str, offset: int = 0) -> L1ManagerConfig:
    return L1ManagerConfig(
        L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
            devdax_path=path,
            devdax_offset_in_bytes=offset,
        ),
        tag=tag,
    )


def test_dram_config_builds_a_dram_manager(
    fake_pinned_memory: None, managers: list[L1ManagerInterface]
) -> None:
    manager = create_l1_manager(_dram_config())
    managers.append(manager)
    assert isinstance(manager, DramL1Manager)
    assert manager.get_memory_tags() == MemoryTags(MemorySegment.DRAM, LOCAL)
    assert manager.backend_type is L1BackendType.DRAM
    assert manager.get_l1_memory_desc() is not None


def test_devdax_path_builds_a_devdax_manager(
    tmp_path: Path, managers: list[L1ManagerInterface]
) -> None:
    manager = create_l1_manager(_devdax_config(_devdax_file(tmp_path, "a.bin"), "dax"))
    managers.append(manager)
    assert isinstance(manager, DevDaxL1Manager)
    assert manager.get_memory_tags() == MemoryTags(MemorySegment.CXL, LOCAL)
    assert manager.backend_type is L1BackendType.DEVDAX


def test_gds_and_devdax_together_are_rejected(tmp_path: Path) -> None:
    config = _devdax_config(_devdax_file(tmp_path, "a.bin"), "dax")
    config.gds_l1_config = GdsL1Config(file_location=str(tmp_path), size_in_bytes=4096)
    with pytest.raises(ValueError, match="separate L1 managers"):
        create_l1_manager(config)


def test_managers_are_created_in_placement_order(
    fake_pinned_memory: None, tmp_path: Path, managers: list[L1ManagerInterface]
) -> None:
    created = create_l1_managers(
        [_dram_config("a"), _devdax_config(_devdax_file(tmp_path, "a.bin"), "b")]
    )
    managers.extend(created)
    assert [type(m) for m in created] == [DramL1Manager, DevDaxL1Manager]
    assert [m.config.tag for m in created] == ["a", "b"]
    assert len({m.l1_manager_id for m in created}) == 2


def test_overlapping_devdax_ranges_are_rejected_and_earlier_managers_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _devdax_file(tmp_path, "shared.bin")
    closed: list[str] = []
    original_close = DevDaxL1Manager.close

    def record_close(manager: DevDaxL1Manager) -> None:
        closed.append(manager.config.tag)
        original_close(manager)

    monkeypatch.setattr(DevDaxL1Manager, "close", record_close)
    with pytest.raises(ValueError, match="already owned by an L1"):
        create_l1_managers(
            [_devdax_config(path, "first"), _devdax_config(path, "second")]
        )
    assert closed == ["first"]
