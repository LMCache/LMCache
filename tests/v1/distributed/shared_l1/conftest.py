# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for coordinator-owned shared-L1 tests."""

# Standard
from pathlib import Path
from typing import Any
import mmap

# Third Party
import pytest

# First Party
from lmcache.v1.memory_coordinator.pool import MemoryPool

_CAPACITY = 4096
_ALIGNMENT = 64


class InProcessCoordinatorClient:
    """Delegate metadata operations to a pool shared by several test clients."""

    def __init__(self, pool: MemoryPool) -> None:
        self._pool = pool
        self.closed = False

    def get_memory_usage(self) -> tuple[int, int]:
        status = self._pool.status()
        return status.used_bytes, status.region.capacity_bytes

    def close(self) -> None:
        self.closed = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


class RecordingVisibility:
    """Visibility fake that records every exact range it is asked to touch."""

    def __init__(self, fail_operation: int | None = None) -> None:
        self.calls: list[tuple[int, int, int, int]] = []
        self.fail_operation = fail_operation

    @property
    def granularity(self) -> int:
        return _ALIGNMENT

    def apply(
        self,
        operation: int,
        _device_fd: int,
        _mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        self.calls.append((operation, device_offset, length, generation))
        if operation == self.fail_operation:
            raise RuntimeError("injected visibility failure")


@pytest.fixture
def region_pool() -> MemoryPool:
    return MemoryPool("region", _CAPACITY, _ALIGNMENT, "layout")


@pytest.fixture
def region_file(tmp_path: Path) -> Path:
    path = tmp_path / "region.bin"
    with path.open("wb") as region:
        region.truncate(mmap.PAGESIZE + _CAPACITY)
    return path
