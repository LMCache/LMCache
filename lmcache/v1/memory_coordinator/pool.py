# SPDX-License-Identifier: Apache-2.0
"""The Memory Coordinator's single strong allocation and lifetime index.

Extents are never reclaimed, including after an aborted write. Reuse requires
an operator-coordinated reset after all mapped workers stop; the persistent
startup latch refuses ordinary restarts. Eviction and recovery are not supported.

The pool holds exactly one immutable :class:`RegionContract`. There is no
API that adds a region or mutates capacity, alignment, or layout after
construction; a restart constructs a new pool and therefore a new
``region_epoch``.
"""

# Standard
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


class _ObjectRecord:
    """Mutable per-object state private to the pool."""

    __slots__ = ("handle", "layout", "write_token")

    def __init__(self, grant: WriteGrant) -> None:
        self.handle = grant.handle
        self.layout = grant.layout
        self.write_token: str | None = grant.token


class MemoryPool:
    """Strong allocation, object-lifetime, and reservation state for M0.

    All operations are serialized under one lock. Batches validate completely
    before mutating state. Lookups are partial by key.

    Args:
        region_id: Operator-provisioned identity of the shared region.
        capacity_bytes: Logical capacity of the shared pool.
        alignment_bytes: Allocation alignment; positive power of two.
        layout_id: Operator-supplied immutable layout-profile fingerprint.

    Raises:
        ValueError: A contract parameter is empty, non-positive, or the
            alignment is not a power of two.
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
        """Fail closed when a request names a different epoch.

        Args:
            region_epoch: The epoch the requesting client latched.

        Raises:
            StaleEpochError: The epoch does not match this pool's epoch.
        """
        if region_epoch != self._contract.region_epoch:
            raise StaleEpochError(
                "request epoch does not match the coordinator region epoch"
            )

    def reserve_writes(
        self,
        items: list[WriteReserveItem],
    ) -> list[WriteGrant | None]:
        """Reserve every absent key in one capacity-atomic batch.

        Args:
            items: Keys and layouts to reserve. Keys must be unique within
                the batch and layouts must describe a positive size.

        Returns:
            One entry per item: a :class:`WriteGrant` for each newly
            ``WRITING`` key, ``None`` for keys that already exist (one
            writer wins a duplicate-key race).

        Raises:
            ValueError: Duplicate keys in the batch, or an invalid layout.
            OutOfSpaceError: The complete batch of absent keys does not fit;
                no state changes in that case (capacity-atomic).
        """
        canonicals = [canonical_key(item.key) for item in items]
        if len(canonicals) != len(set(canonicals)):
            raise ValueError("a write batch must not contain duplicate keys")
        lengths = [item.layout.size_bytes() for item in items]
        if any(length <= 0 for length in lengths):
            raise ValueError("write layouts must describe a positive size")

        with self._lock:
            cursor = self._next_offset
            generation = self._next_generation
            planned: dict[int, WriteGrant] = {}
            for index, (item, canonical, length) in enumerate(
                zip(items, canonicals, lengths, strict=True)
            ):
                if canonical in self._objects:
                    continue
                if generation > _MAX_GENERATION:
                    raise OutOfSpaceError("generation space is exhausted")
                offset = round_up(cursor, self._contract.alignment_bytes)
                if length > self._contract.capacity_bytes - offset:
                    raise OutOfSpaceError(
                        "write batch does not fit in the shared region"
                    )
                planned[index] = WriteGrant(
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
                cursor = offset + length
                generation += 1

            result: list[WriteGrant | None] = []
            for index, canonical in enumerate(canonicals):
                grant = planned.get(index)
                result.append(grant)
                if grant is not None:
                    self._objects[canonical] = _ObjectRecord(grant)
            self._next_offset = cursor
            self._next_generation = generation
            return result

    def finish_writes(self, reservations: list[ReservationRef]) -> None:
        """Atomically publish a batch: every reservation becomes ``VALID``.

        Args:
            reservations: The grants being committed, tokens included.

        Raises:
            InvalidReservationError: Duplicate reservations, or a token does
                not own a ``WRITING`` object. No state changes on error.
        """
        with self._lock:
            records = self._validate_writes(reservations)
            for record in records:
                record.write_token = None

    def abort_writes(self, reservations: list[ReservationRef]) -> None:
        """Drop ``WRITING`` metadata without reusing its extents.

        Args:
            reservations: The grants being aborted, tokens included.

        Raises:
            InvalidReservationError: Duplicate reservations, or a token does
                not own a ``WRITING`` object. No state changes on error.
        """
        with self._lock:
            self._validate_writes(reservations)
            for reservation in reservations:
                del self._objects[canonical_key(reservation.key)]

    def lookup(self, keys: list[EncodedObjectKey]) -> list[LookupHit | None]:
        """Return every ``VALID`` hit in one partial batch operation.

        Args:
            keys: Keys to read. Must be unique within the batch.

        Returns:
            One entry per key: a :class:`LookupHit` for ``VALID`` objects,
            ``None`` for absent or still-``WRITING`` keys.

        Raises:
            ValueError: Duplicate keys in the batch.
        """
        canonicals = [canonical_key(key) for key in keys]
        if len(canonicals) != len(set(canonicals)):
            raise ValueError("a lookup batch must not contain duplicate keys")
        with self._lock:
            result: list[LookupHit | None] = []
            for key, canonical in zip(keys, canonicals, strict=True):
                record = self._objects.get(canonical)
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
        tokens = [reservation.token for reservation in reservations]
        if len(tokens) != len(set(tokens)):
            raise InvalidReservationError("duplicate write reservation")
        records = []
        for reservation in reservations:
            record = self._objects.get(canonical_key(reservation.key))
            if record is None or record.write_token != reservation.token:
                raise InvalidReservationError("reservation does not own the write")
            records.append(record)
        return records
