# SPDX-License-Identifier: Apache-2.0
"""JSON metadata contracts for the shared Device-DAX coordinator."""

# Standard
import math

# Third Party
from pydantic import BaseModel, ConfigDict, Field, StrictInt

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey

_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


def wire_dtype_itemsize(name: str) -> int:
    """Return bytes per element for ``name``; raise ValueError if unsupported."""
    try:
        return _DTYPE_BYTES[name]
    except KeyError:
        raise ValueError(f"unknown wire dtype {name!r}") from None


def canonical_key(key: EncodedObjectKey) -> EncodedObjectKey:
    """Return immutable ``key``; raise ValueError for non-canonical encodings."""
    if type(key.kv_rank) is not int or type(key.object_group_id) is not int:
        raise ValueError("kv_rank and object_group_id must be integers")
    canonical = key.to_object_key().to_encoded_object_key()
    if canonical != key:
        raise ValueError("encoded object key must use its canonical representation")
    return canonical


class MemoryCoordinatorError(RuntimeError):
    """Base error for rejected coordinator operations."""


class OutOfSpaceError(MemoryCoordinatorError):
    """The complete write batch cannot fit."""


class InvalidReservationError(MemoryCoordinatorError):
    """A token does not own the requested operation."""


class StaleEpochError(MemoryCoordinatorError):
    """The caller names another coordinator incarnation."""


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WireLayout(_WireModel):
    """Tensor shapes and canonical dtype names; payload bytes are excluded."""

    shapes: tuple[tuple[StrictInt, ...], ...]
    dtypes: tuple[str, ...]

    def size_bytes(self) -> int:
        """Return total tensor bytes; raise ValueError for invalid shapes/dtypes."""
        if len(self.shapes) != len(self.dtypes):
            raise ValueError("shapes and dtypes must align")
        total = 0
        for shape, dtype in zip(self.shapes, self.dtypes, strict=True):
            if any(dim < 0 for dim in shape):
                raise ValueError("dimensions must be non-negative")
            total += math.prod(shape) * wire_dtype_itemsize(dtype)
        return total


class RegionContract(_WireModel):
    """Immutable identity and geometry for one shared physical region."""

    region_id: str
    capacity_bytes: int
    alignment_bytes: int
    layout_id: str
    region_epoch: str


class SharedObjectHandle(_WireModel):
    """Region-relative location and generation of one shared object."""

    region_id: str
    offset: int = Field(ge=0)
    length: int = Field(gt=0)
    generation: int = Field(gt=0, le=(1 << 64) - 1)


class WriteReserveItem(_WireModel):
    """Object key and layout requested for a new write."""

    key: EncodedObjectKey
    layout: WireLayout


class ReservationRef(_WireModel):
    """Exact key and opaque token needed to finish or abort a write."""

    key: EncodedObjectKey
    token: str


class WriteGrant(ReservationRef):
    """Exclusive authority to initialize a WRITING object."""

    handle: SharedObjectHandle
    layout: WireLayout


class LookupHit(_WireModel):
    """Location and layout of one immutable VALID object."""

    key: EncodedObjectKey
    handle: SharedObjectHandle
    layout: WireLayout


class WriteReserveRequest(_WireModel):
    """Batched write reservation request."""

    region_epoch: str
    items: list[WriteReserveItem]


class WriteReserveResponse(_WireModel):
    """One grant or duplicate-key miss per requested write."""

    region_epoch: str
    grants: list[WriteGrant | None]


class LookupRequest(_WireModel):
    """Batched lookup request."""

    region_epoch: str
    keys: list[EncodedObjectKey]


class LookupResponse(_WireModel):
    """One immutable hit or cache miss per requested key."""

    region_epoch: str
    hits: list[LookupHit | None]


class ReservationBatchRequest(_WireModel):
    """Batched finish or abort request."""

    region_epoch: str
    reservations: list[ReservationRef]


class EpochResponse(_WireModel):
    """Acknowledgement carrying the current region epoch."""

    region_epoch: str


class StatusResponse(_WireModel):
    """Constant-size pool status; used bytes include abandoned extents."""

    region: RegionContract
    used_bytes: int
    object_count: int
