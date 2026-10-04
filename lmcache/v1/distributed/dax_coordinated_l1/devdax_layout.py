# SPDX-License-Identifier: Apache-2.0
"""Payload placement, slot layout and participant ownership for DAX-Coordinated L1."""

# Standard
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, NamedTuple
import json

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import DaxCoordinatedL1Config


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, str):
        try:
            value = int(value, 0)
        except ValueError as error:
            raise ValueError(f"Device-DAX {name} must be an integer") from error
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Device-DAX {name} must be an integer >= {minimum}")
    return value


def normalize_devdax_offset(value: object, name: str) -> int:
    """Parse an integer/base-prefixed offset and require uint64 page alignment."""
    offset = _integer(value, name)
    if offset >= 1 << 64 or offset % 4096:
        raise ValueError(f"Device-DAX {name} must be a page-aligned 64-bit offset")
    return offset


@dataclass(frozen=True)
class DevDaxTPRegion:
    """One device/bank's payload and metadata arena, in local DAX offsets.

    The first region requires the common metadata offset and reservation size;
    subsequent regions must leave both unset. Each assigned rank gets an equal
    whole-GiB payload slice. Paths may differ between hosts; physical bank
    correspondence must agree.
    """

    devdax_path: str
    metadata_offset_bytes: int | str | None = field(default=None, kw_only=True)
    metadata_reservation_bytes: int | str | None = field(default=None, kw_only=True)
    payload_offset_bytes: int | str
    payload_size_GiB: int

    def __post_init__(self) -> None:
        if not isinstance(self.devdax_path, str) or not self.devdax_path.strip():
            raise ValueError("rank_placement region requires devdax_path")
        object.__setattr__(self, "devdax_path", self.devdax_path.strip())
        for name in ("metadata_offset_bytes", "payload_offset_bytes"):
            if name == "metadata_offset_bytes" and self.metadata_offset_bytes is None:
                continue
            object.__setattr__(
                self, name, normalize_devdax_offset(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "payload_size_GiB",
            _integer(self.payload_size_GiB, "payload_size_GiB", 1),
        )
        if self.metadata_reservation_bytes is not None:
            size = normalize_devdax_offset(
                self.metadata_reservation_bytes, "metadata_reservation_bytes"
            )
            if size == 0:
                raise ValueError("metadata_reservation_bytes must be positive")
            object.__setattr__(self, "metadata_reservation_bytes", size)


@dataclass(frozen=True)
class DevDaxTPRankPlacement:
    """One rank's contiguous DMA payload slice in the shared logical arena."""

    rank: int
    region_index: int
    devdax_path: str
    payload_offset_bytes: int
    payload_size_GiB: int


@dataclass(frozen=True)
class DaxCoordinatedL1RankPlacementConfig:
    """Place TP=1..8, PP=1, with all ranks local to one LMCache server.

    Configure physical DAX payload placement, not engine tensor parallelism.
    ``tp_size`` is the expected rank count, checked against runtime registration.
    The enclosing backend config uses the JSON field name ``rank_placement``.

    ``rank % len(regions)`` selects a region. Ranks sharing a region receive
    disjoint slices; a remainder smaller than one GiB per rank is unused.
    The first region supplies both metadata offset and reservation size for the
    common index; the reservation is not multiplied by TP size.
    Invalid topology, alignment or overlapping ranges raises ``ValueError``.
    """

    tp_size: int
    regions: tuple[DevDaxTPRegion, ...]

    def __post_init__(self) -> None:
        tp_size = _integer(self.tp_size, "tp_size", 1)
        if tp_size > 8:
            raise ValueError("rank_placement supports local TP=1..8 with PP=1")
        if not isinstance(self.regions, (list, tuple)):
            raise ValueError("rank_placement regions must be a list")
        try:
            regions = tuple(
                DevDaxTPRegion(**item) if isinstance(item, dict) else item
                for item in self.regions
            )
        except TypeError as error:
            raise ValueError(f"invalid rank_placement region: {error}") from error
        if not 1 <= len(regions) <= tp_size or any(
            not isinstance(item, DevDaxTPRegion) for item in regions
        ):
            raise ValueError("rank_placement requires 1..tp_size valid regions")
        if regions[0].metadata_offset_bytes is None:
            raise ValueError("rank_placement regions[0] requires metadata_offset_bytes")
        if regions[0].metadata_reservation_bytes is None:
            raise ValueError(
                "rank_placement regions[0] requires metadata_reservation_bytes"
            )
        for index, region in enumerate(regions[1:], start=1):
            for name in ("metadata_offset_bytes", "metadata_reservation_bytes"):
                if getattr(region, name) is not None:
                    raise ValueError(
                        f"rank_placement regions[{index}] must omit {name}; "
                        "only regions[0] supplies the common metadata arena"
                    )
        reservation_bytes = int(regions[0].metadata_reservation_bytes)
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "regions", regions)
        # Validate even before device access. Runtime repeats this comparison
        # using st_rdev so symlinks/alternate names cannot hide overlap.
        ranges: dict[str, list[tuple[int, int]]] = {}
        for index, region in enumerate(regions):
            ranks = len(range(index, tp_size, len(regions)))
            if region.payload_size_GiB < ranks:
                raise ValueError(
                    "rank_placement needs at least one GiB per assigned rank"
                )
            reserved = [
                (int(region.payload_offset_bytes), region.payload_size_GiB << 30)
            ]
            if index == 0:
                assert region.metadata_offset_bytes is not None
                reserved.append((int(region.metadata_offset_bytes), reservation_bytes))
            for begin, size in reserved:
                end = begin + size
                previous = ranges.setdefault(region.devdax_path, [])
                if end > 1 << 64 or any(begin < b and a < end for a, b in previous):
                    raise ValueError(
                        "rank_placement metadata/payload ranges overlap or overflow"
                    )
                previous.append((begin, end))

    def placements(self) -> tuple[DevDaxTPRankPlacement, ...]:
        """Return deterministic rank slices without accessing devices or CUDA."""
        result = []
        for rank in range(self.tp_size):
            region_index = rank % len(self.regions)
            region = self.regions[region_index]
            ranks = len(range(region_index, self.tp_size, len(self.regions)))
            ordinal = rank // len(self.regions)
            size_gib = region.payload_size_GiB // ranks
            result.append(
                DevDaxTPRankPlacement(
                    rank=rank,
                    region_index=region_index,
                    devdax_path=region.devdax_path,
                    payload_offset_bytes=int(region.payload_offset_bytes)
                    + ordinal * (size_gib << 30),
                    payload_size_GiB=size_gib,
                )
            )
        return tuple(result)

    def layout_digest(self) -> bytes:
        """Fence topology and offsets across hosts, excluding local path names."""
        value = {
            # Preserve the v1 schema identifier across the rank_placement rename.
            "schema": "devdax_tp_payload_v1",
            "tp_size": self.tp_size,
            # Keep the v1 digest field name so this config rename does not
            # invalidate existing shared arenas. It encodes reserved bytes.
            "metadata_stride_bytes": self.regions[0].metadata_reservation_bytes,
            "metadata_offset_bytes": self.regions[0].metadata_offset_bytes,
            "regions": [
                [r.payload_offset_bytes, r.payload_size_GiB] for r in self.regions
            ],
        }
        return sha256(json.dumps(value, sort_keys=True).encode()).digest()

    def rank_from_kv_rank(self, kv_rank: int) -> int:
        """Decode ObjectKey.ComputeKVRank; reject other worlds or nonlocal ranks.

        Plain rank integers are not encoded ObjectKey ranks and are rejected.
        """
        if isinstance(kv_rank, bool) or not isinstance(kv_rank, int):
            raise ValueError("rank_placement requires an encoded local TP kv_rank")
        rank = (kv_rank >> 16) & 0xFF
        expected = (self.tp_size << 24) | (rank << 16) | (self.tp_size << 8) | rank
        if kv_rank != expected or rank >= self.tp_size:
            raise ValueError(
                "rank_placement requires an encoded local TP kv_rank matching tp_size"
            )
        return rank


@dataclass(frozen=True)
class DevDaxOwnerRankSlots:
    """One owner's slots for a rank, with an offset within that rank's mapping."""

    owner: int
    rank: int
    first_slot: int
    slot_count: int
    mapping_offset_bytes: int


class DevDaxPayloadMapping(NamedTuple):
    """A device extent; offsets and sizes are in bytes."""

    devdax_path: str
    offset_bytes: int
    size_bytes: int


@dataclass(frozen=True)
class DevDaxPayloadGeometry:
    """Model-bound fixed-slot geometry carved from a payload device range."""

    model_payload_bytes: int
    payload_slot_bytes: int
    payload_slot_count: int
    owner_slot_begin: int
    owner_slot_count: int
    payload_bytes_used: int
    payload_bytes_unused: int
    rank_slot_counts: tuple[int, ...]
    participant_0_slot_count: int
    owner_rank_slot_counts: tuple[int, ...]
    slot_ranges: tuple[DevDaxOwnerRankSlots, ...]


def resolve_payload_mappings(
    config: "DaxCoordinatedL1Config",
) -> list[DevDaxPayloadMapping]:
    """Resolve payload ranges without accessing devices or CUDA.

    Args:
        config: Validated DAX-Coordinated L1 configuration.

    Returns:
        Device path, payload offset and capacity in bytes for each TP rank,
        or one range when TP placement is not configured.
    """
    return [
        DevDaxPayloadMapping(
            p.devdax_path, p.payload_offset_bytes, p.payload_size_GiB << 30
        )
        for p in config.rank_placement.placements()
    ]


def resolve_payload_geometry(
    config: "DaxCoordinatedL1Config", model_payload_bytes: int
) -> DevDaxPayloadGeometry:
    """Carve the selected ownership policy from the configured payload.

    ``model_payload_bytes`` is the complete object size of the one
    homogeneous PR1 model layout. Slots are cache-line aligned and the
    unusable byte tail is deliberately left outside the allocator. Equal
    ownership leaves slots not divisible by the participant count unused;
    participant-0 ownership can use every complete slot.

    Args:
        config: Validated DAX-Coordinated L1 configuration.
        model_payload_bytes: Complete model object size in bytes.

    Returns:
        Slot geometry and this participant's ownership range.

    Raises:
        ValueError: If the payload range cannot hold the required slots.
    """
    if model_payload_bytes <= 0:
        raise ValueError("model payload bytes must be positive")
    slot_bytes = _align_up(model_payload_bytes, config.DEVDAX_CACHE_LINE_BYTES)
    capacities = [mapping.size_bytes for mapping in resolve_payload_mappings(config)]
    owners = config.participant_count if config.ownership_mode == "equal" else 1
    rank_slots = tuple(size // slot_bytes // owners * owners for size in capacities)
    if any(count < owners for count in rank_slots):
        raise ValueError("payload range cannot hold one model slot per participant")
    slot_count = sum(rank_slots)
    used = slot_count * slot_bytes
    if used > config.payload_size_bytes:
        raise ValueError("configured payload slots exceed payload_size_GiB capacity")
    first_count = slot_count // owners
    owner_begin = min(config.participant_id, owners) * first_count
    owner_count = first_count if config.participant_id < owners else 0
    per_owner_rank_slots = tuple(count // owners for count in rank_slots)
    slot_ranges = []
    begin = 0
    for owner in range(owners):
        for rank, count in enumerate(per_owner_rank_slots):
            slot_ranges.append(
                DevDaxOwnerRankSlots(
                    owner=owner,
                    rank=rank,
                    first_slot=begin,
                    slot_count=count,
                    mapping_offset_bytes=owner * count * slot_bytes,
                )
            )
            begin += count
    return DevDaxPayloadGeometry(
        model_payload_bytes=model_payload_bytes,
        payload_slot_bytes=slot_bytes,
        payload_slot_count=slot_count,
        owner_slot_begin=owner_begin,
        owner_slot_count=owner_count,
        payload_bytes_used=used,
        payload_bytes_unused=config.payload_size_bytes - used,
        rank_slot_counts=rank_slots,
        participant_0_slot_count=first_count,
        owner_rank_slot_counts=(
            per_owner_rank_slots
            if config.participant_id < owners
            else (0,) * len(rank_slots)
        ),
        slot_ranges=tuple(slot_ranges),
    )
