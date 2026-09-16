# SPDX-License-Identifier: Apache-2.0
"""Atomic metadata for one immutable region; aborted extents are never reused."""

# Standard
from time import monotonic
import threading
import uuid

# First Party
from lmcache.utils import round_up

# Local
from .api import (
    EncodedObjectKey,
    InvalidReservationError,
    LookupHit,
    OutOfSpaceError,
    RegionContract,
    ReservationRef,
    SharedObjectHandle,
    StaleEpochError,
    StatusResponse,
    WriteGrant,
    WriteReserveItem,
    canonical_key,
)

_MAX_GENERATION = (1 << 64) - 1
# Experimental metadata policy, not a GPU-quiescence deadline.
_RESERVATION_TTL_SECONDS = 60.0


class _ObjectRecord:
    """Mutable per-object state private to the pool."""

    __slots__ = ("handle", "layout", "write_token", "expires_at")

    def __init__(self, grant: WriteGrant, expires_at: float) -> None:
        self.handle = grant.handle
        self.layout = grant.layout
        self.write_token: str | None = grant.token
        self.expires_at = expires_at


class MemoryPool:
    """Own one region's allocation index, serialized under one lock.

    ``region_id`` names the shared region; ``layout_id`` identifies its immutable
    layout. ``capacity_bytes`` is its size and ``alignment_bytes`` its
    power-of-two allocation alignment. Invalid settings raise ValueError.
    Batches validate before mutation; extent reuse requires all mapped workers
    to stop.
    """

    def __init__(
        self,
        region_id: str,
        capacity_bytes: int,
        alignment_bytes: int,
        layout_id: str,
    ) -> None:
        if not region_id.strip():
            raise ValueError("region_id must not be empty")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
            raise ValueError("alignment_bytes must be a positive power of two")
        if not layout_id.strip():
            raise ValueError("layout_id must not be empty")
        self._contract = RegionContract(
            region_id=region_id,
            capacity_bytes=capacity_bytes,
            alignment_bytes=alignment_bytes,
            layout_id=layout_id,
            region_epoch=uuid.uuid4().hex,
        )
        self._next_offset = 0
        self._next_generation = 1
        self._objects: dict[EncodedObjectKey, _ObjectRecord] = {}
        self._lock = threading.RLock()

    def region_contract(self) -> RegionContract:
        """Return the immutable region contract, restart epoch included."""
        return self._contract

    def check_epoch(self, region_epoch: str) -> None:
        """Raise StaleEpochError unless ``region_epoch`` matches this pool."""
        if region_epoch != self._contract.region_epoch:
            raise StaleEpochError(
                "request epoch does not match the coordinator region epoch"
            )

    def reserve_writes(
        self,
        items: list[WriteReserveItem],
    ) -> list[WriteGrant | None]:
        """Reserve key/layout ``items`` at fresh offsets, replacing expired writes.

        Committed and unexpired pending keys return None. Expired records are
        replaced only after the whole batch fits; old extents remain consumed.
        Duplicate keys or invalid layouts raise ValueError. Insufficient capacity
        raises OutOfSpaceError. Every failure leaves the whole batch unchanged.
        """
        canonicals = [canonical_key(item.key) for item in items]
        if len(canonicals) != len(set(canonicals)):
            raise ValueError("a write batch must not contain duplicate keys")
        lengths = [item.layout.size_bytes() for item in items]
        if any(length <= 0 for length in lengths):
            raise ValueError("write layouts must describe a positive size")

        with self._lock:
            now = monotonic()
            cursor = self._next_offset
            generation = self._next_generation
            result: list[WriteGrant | None] = []
            for item, canonical, length in zip(items, canonicals, lengths, strict=True):
                record = self._objects.get(canonical)
                if record is not None and (
                    record.write_token is None or now < record.expires_at
                ):
                    result.append(None)
                    continue
                if generation > _MAX_GENERATION:
                    raise OutOfSpaceError("generation space is exhausted")
                offset = round_up(cursor, self._contract.alignment_bytes)
                if length > self._contract.capacity_bytes - offset:
                    raise OutOfSpaceError(
                        "write batch does not fit in the shared region"
                    )
                grant = WriteGrant(
                    key=item.key,
                    handle=SharedObjectHandle(
                        region_id=self._contract.region_id,
                        offset=offset,
                        length=length,
                        generation=generation,
                    ),
                    token=uuid.uuid4().hex,
                    layout=item.layout,
                )
                result.append(grant)
                cursor = offset + length
                generation += 1

            for reserved in result:
                if reserved is not None:
                    self._objects[reserved.key] = _ObjectRecord(
                        reserved, now + _RESERVATION_TTL_SECONDS
                    )
            self._next_offset = cursor
            self._next_generation = generation
            return result

    def finish_writes(self, reservations: list[ReservationRef]) -> None:
        """Atomically publish token-bearing ``reservations`` as VALID.

        Invalid, duplicate or expired tokens raise InvalidReservationError.
        The entire batch remains unchanged on failure.
        """
        with self._lock:
            for record in self._validate_writes(reservations):
                record.write_token = None

    def abort_writes(self, reservations: list[ReservationRef]) -> None:
        """Abort token-bearing ``reservations`` without reusing their extents.

        Invalid, duplicate or expired tokens raise InvalidReservationError.
        The entire batch remains unchanged on failure.
        """
        with self._lock:
            self._validate_writes(reservations)
            for reservation in reservations:
                del self._objects[canonical_key(reservation.key)]

    def lookup(self, keys: list[EncodedObjectKey]) -> list[LookupHit | None]:
        """Return VALID hits for ``keys``, None for missing or WRITING entries.

        Duplicate or non-canonical keys raise ValueError.
        """
        canonicals = [canonical_key(key) for key in keys]
        if len(canonicals) != len(set(canonicals)):
            raise ValueError("a lookup batch must not contain duplicate keys")
        with self._lock:
            result: list[LookupHit | None] = []
            for key in canonicals:
                record = self._objects.get(key)
                if record is None or record.write_token is not None:
                    result.append(None)
                    continue
                result.append(
                    LookupHit(
                        key=key,
                        handle=record.handle,
                        layout=record.layout,
                    )
                )
            return result

    def status(self) -> StatusResponse:
        """Return constant-size region usage and object-count diagnostics."""
        with self._lock:
            return StatusResponse(
                region=self._contract,
                used_bytes=self._next_offset,
                object_count=len(self._objects),
            )

    def _validate_writes(
        self,
        reservations: list[ReservationRef],
    ) -> list[_ObjectRecord]:
        now = monotonic()
        tokens = [reservation.token for reservation in reservations]
        if len(tokens) != len(set(tokens)):
            raise InvalidReservationError("duplicate write reservation")
        records = []
        for reservation in reservations:
            record = self._objects.get(canonical_key(reservation.key))
            if (
                record is None
                or record.write_token != reservation.token
                or now >= record.expires_at
            ):
                raise InvalidReservationError("reservation does not own the write")
            records.append(record)
        return records
