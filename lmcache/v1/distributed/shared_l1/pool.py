# SPDX-License-Identifier: Apache-2.0
"""Coordinator-owned metadata for the non-reclaiming shared-L1 M0."""

# Standard
from dataclasses import dataclass, field
import threading
import uuid

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc

_MAX_GENERATION = (1 << 64) - 1


def _layout_size(layout: MemoryLayoutDesc) -> int:
    return sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
    )


class SharedL1Error(RuntimeError):
    """Base error for shared-L1 metadata operations."""


class OutOfSpaceError(SharedL1Error):
    """Raised when a complete write batch cannot fit."""


class InvalidReservationError(SharedL1Error):
    """Raised when a reservation does not own an operation."""


class StaleHandleError(SharedL1Error):
    """Raised when a reservation no longer names the current object."""


@dataclass(frozen=True)
class SharedRegionContract:
    """Immutable identity and geometry shared by every pool mapping."""

    region_id: str
    capacity: int
    alignment: int
    layout_id: str
    generation_epoch: int

    def __post_init__(self) -> None:
        if not self.region_id or not self.layout_id:
            raise ValueError("region_id and layout_id must not be empty")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if self.generation_epoch <= 0:
            raise ValueError("generation_epoch must be positive")


@dataclass(frozen=True)
class SharedObjectHandle:
    """Stable location independent of each process's virtual address."""

    region_id: str
    offset: int
    length: int
    generation: int

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must not be empty")
        if self.offset < 0 or self.length <= 0:
            raise ValueError("offset must be non-negative and length positive")
        if not 0 < self.generation <= _MAX_GENERATION:
            raise ValueError("generation must fit in an unsigned 64-bit integer")


@dataclass(frozen=True)
class WriteReservation:
    """Exclusive authority to initialize one shared object."""

    object_key: str
    handle: SharedObjectHandle
    token: str
    layout: MemoryLayoutDesc


@dataclass(frozen=True)
class ReadReservation:
    """Authority for one active read of a VALID shared object."""

    object_key: str
    handle: SharedObjectHandle
    token: str
    layout: MemoryLayoutDesc


@dataclass
class _ObjectRecord:
    handle: SharedObjectHandle
    layout: MemoryLayoutDesc
    state: str
    write_token: str | None
    read_tokens: set[str] = field(default_factory=set)


class SharedL1Pool:
    """The child process's single strong allocation and lifetime index.

    M0 deliberately uses a monotonic allocator: extents are never reclaimed,
    so aborting a write leaks its bytes until the coordinator is restarted.
    This is safe for a bounded functional run and avoids pretending that
    eviction, TTL recovery, or restart fencing are already implemented.
    """

    def __init__(
        self,
        region_id: str,
        capacity: int,
        alignment: int,
        layout_id: str,
    ) -> None:
        self._contract = SharedRegionContract(
            region_id,
            capacity,
            alignment,
            layout_id,
            uuid.uuid4().int,
        )
        self._next_offset = 0
        self._next_generation = 1
        self._objects: dict[str, _ObjectRecord] = {}
        self._lock = threading.RLock()

    def region_contract(self) -> SharedRegionContract:
        """Return the immutable region identity and restart epoch."""
        return self._contract

    def reserve_writes(
        self,
        requests: list[tuple[str, MemoryLayoutDesc]],
    ) -> list[WriteReservation | None]:
        """Reserve absent keys in one atomic coordinator operation.

        ``None`` means that the key is already WRITING or VALID. Capacity and
        request validation happen before any allocator state changes.
        """
        keys = [key for key, _ in requests]
        if len(keys) != len(set(keys)):
            raise ValueError("a write batch must not contain duplicate keys")
        lengths = [_layout_size(layout) for _, layout in requests]
        if any(not key for key in keys) or any(length <= 0 for length in lengths):
            raise ValueError("object keys and layouts must not be empty")

        with self._lock:
            cursor = self._next_offset
            generation = self._next_generation
            planned: dict[int, WriteReservation] = {}
            for index, ((key, layout), length) in enumerate(
                zip(requests, lengths, strict=True)
            ):
                if key in self._objects:
                    continue
                if generation > _MAX_GENERATION:
                    raise OutOfSpaceError("generation space is exhausted")
                offset = self._align_up(cursor, self._contract.alignment)
                if length > self._contract.capacity - offset:
                    raise OutOfSpaceError("write batch does not fit in shared L1")
                handle = SharedObjectHandle(
                    self._contract.region_id,
                    offset,
                    length,
                    generation,
                )
                planned[index] = WriteReservation(
                    key,
                    handle,
                    uuid.uuid4().hex,
                    layout,
                )
                cursor = offset + length
                generation += 1

            result: list[WriteReservation | None] = []
            for index, (key, _) in enumerate(requests):
                reservation = planned.get(index)
                result.append(reservation)
                if reservation is not None:
                    self._objects[key] = _ObjectRecord(
                        reservation.handle,
                        reservation.layout,
                        "WRITING",
                        reservation.token,
                    )
            self._next_offset = cursor
            self._next_generation = generation
            return result

    def finish_writes(
        self,
        reservations: list[WriteReservation],
    ) -> list[SharedObjectHandle]:
        """Atomically publish a batch after every payload range is visible."""
        with self._lock:
            records = self._validate_writes(reservations)
            for record in records:
                record.state = "VALID"
                record.write_token = None
            return [reservation.handle for reservation in reservations]

    def abort_writes(self, reservations: list[WriteReservation]) -> None:
        """Drop failed WRITING metadata without reusing its extents."""
        with self._lock:
            self._validate_writes(reservations)
            for reservation in reservations:
                del self._objects[reservation.object_key]

    def reserve_reads(
        self,
        object_keys: list[str],
    ) -> list[ReadReservation | None]:
        """Pin every VALID hit in one partial batch operation."""
        if len(object_keys) != len(set(object_keys)):
            raise ValueError("a read batch must not contain duplicate keys")
        with self._lock:
            result: list[ReadReservation | None] = []
            for key in object_keys:
                record = self._objects.get(key)
                if record is None or record.state != "VALID":
                    result.append(None)
                    continue
                token = uuid.uuid4().hex
                record.read_tokens.add(token)
                result.append(ReadReservation(key, record.handle, token, record.layout))
            return result

    def finish_reads(self, reservations: list[ReadReservation]) -> None:
        """Release completed reads after their H2D operations finish."""
        self._release_reads(reservations)

    def abort_reads(self, reservations: list[ReadReservation]) -> None:
        """Release reads whose local transfer was cancelled."""
        self._release_reads(reservations)

    def snapshot(self) -> dict[str, object]:
        """Return token-free metadata for diagnostics and usage reporting."""
        with self._lock:
            return {
                "region_id": self._contract.region_id,
                "generation_epoch": self._contract.generation_epoch,
                "capacity": self._contract.capacity,
                "next_offset": self._next_offset,
                "objects": {
                    key: {
                        "handle": record.handle,
                        "state": record.state,
                        "active_readers": len(record.read_tokens),
                    }
                    for key, record in self._objects.items()
                },
            }

    def _validate_writes(
        self,
        reservations: list[WriteReservation],
    ) -> list[_ObjectRecord]:
        keys = [reservation.object_key for reservation in reservations]
        if len(keys) != len(set(keys)):
            raise InvalidReservationError("duplicate write reservation")
        records = []
        for reservation in reservations:
            record = self._objects.get(reservation.object_key)
            if (
                record is None
                or record.state != "WRITING"
                or record.write_token != reservation.token
            ):
                raise InvalidReservationError("reservation does not own the write")
            if record.handle != reservation.handle:
                raise StaleHandleError("write reservation has a stale handle")
            records.append(record)
        return records

    def _release_reads(self, reservations: list[ReadReservation]) -> None:
        tokens = [reservation.token for reservation in reservations]
        if len(tokens) != len(set(tokens)):
            raise InvalidReservationError("duplicate read reservation")
        with self._lock:
            records = []
            for reservation in reservations:
                record = self._objects.get(reservation.object_key)
                if record is None or reservation.token not in record.read_tokens:
                    raise InvalidReservationError(
                        "reservation does not own an active read"
                    )
                if record.handle != reservation.handle:
                    raise StaleHandleError("read reservation has a stale handle")
                records.append(record)
            for reservation, record in zip(reservations, records, strict=True):
                record.read_tokens.remove(reservation.token)

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment
