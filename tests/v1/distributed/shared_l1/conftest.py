# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for coordinator-owned shared-L1 tests."""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.memory_coordinator.api import (
    EncodedObjectKey,
    LookupHit,
    RegionContract,
    ReservationRef,
    StatusResponse,
    WriteGrant,
    WriteReserveItem,
)
from lmcache.v1.memory_coordinator.pool import MemoryPool

_CAPACITY = 4096
_ALIGNMENT = 64


class InProcessCoordinatorClient:
    """MemoryCoordinatorClientProtocol over an in-process MemoryPool.

    Lets backend tests exercise the full reservation lifecycle without an
    HTTP server; several clients may share one pool to model several MP
    servers sharing one coordinator.
    """

    def __init__(self, pool: MemoryPool) -> None:
        self._pool = pool
        self.closed = False

    def region_contract(self) -> RegionContract:
        return self._pool.region_contract()

    def reserve_writes(
        self,
        items: list[WriteReserveItem],
    ) -> list[WriteGrant | None]:
        return self._pool.reserve_writes(items)

    def finish_writes(self, reservations: list[ReservationRef]) -> None:
        self._pool.finish_writes(reservations)

    def abort_writes(self, reservations: list[ReservationRef]) -> None:
        self._pool.abort_writes(reservations)

    def lookup(
        self,
        keys: list[EncodedObjectKey],
    ) -> list[LookupHit | None]:
        return self._pool.lookup(keys)

    def status(self) -> StatusResponse:
        return self._pool.status()

    def get_memory_usage(self) -> tuple[int, int]:
        status = self._pool.status()
        return status.used_bytes, status.region.capacity_bytes

    def close(self) -> None:
        self.closed = True


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
def region_capacity() -> int:
    return _CAPACITY


@pytest.fixture
def region_alignment() -> int:
    return _ALIGNMENT


@pytest.fixture
def region_pool(region_capacity: int, region_alignment: int) -> MemoryPool:
    return MemoryPool("region", region_capacity, region_alignment, "layout")


@pytest.fixture
def region_file(tmp_path: Path, region_capacity: int) -> Path:
    # Standard
    import mmap

    path = tmp_path / "region.bin"
    with path.open("wb") as region:
        region.truncate(mmap.PAGESIZE + region_capacity)
    return path
