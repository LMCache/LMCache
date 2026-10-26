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
    """Physical regions only; KV rank count is supplied by engine registration."""

    regions: tuple[DevDaxTPRegion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.regions, (list, tuple)) or not self.regions:
            raise ValueError("rank_placement regions must be a nonempty list")
        # Validate physical bounds independently of the engine's rank count.
        validated = DevDaxRankPlacement(len(self.regions), self.regions)
        object.__setattr__(self, "regions", validated.regions)

    def resolve(self, kv_world_size: int) -> "DevDaxRankPlacement":
        """Validate engine KV world size and derive its physical rank slices."""
        return DevDaxRankPlacement(kv_world_size, self.regions)


@dataclass(frozen=True)
class DevDaxRankPlacement:
    """Engine-resolved KV rank count and validated physical placement.

    ``rank % len(regions)`` selects a region. Ranks sharing a region receive
    disjoint whole-GiB slices. This runtime object is not the JSON config.
    """

    kv_world_size: int
    regions: tuple[DevDaxTPRegion, ...]

    @property
    def payload_size_bytes(self) -> int:
        """Return total mapped capacity after whole-GiB rank partitioning."""
        return sum(p.payload_size_GiB << 30 for p in self.placements())

    def __post_init__(self) -> None:
        kv_world_size = _integer(self.kv_world_size, "kv_world_size", 1)
        if kv_world_size > 8:
            raise ValueError("rank_placement supports kv_world_size=1..8")
        if not isinstance(self.regions, (list, tuple)):
            raise ValueError("rank_placement regions must be a list")
        try:
            regions = tuple(
                DevDaxTPRegion(**item) if isinstance(item, dict) else item
                for item in self.regions
            )
        except TypeError as error:
            raise ValueError(f"invalid rank_placement region: {error}") from error
        if not 1 <= len(regions) <= kv_world_size or any(
            not isinstance(item, DevDaxTPRegion) for item in regions
        ):
            raise ValueError("rank_placement requires 1..kv_world_size valid regions")
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
        object.__setattr__(self, "kv_world_size", kv_world_size)
        object.__setattr__(self, "regions", regions)
        # Validate even before device access. Runtime repeats this comparison
        # using st_rdev so symlinks/alternate names cannot hide overlap.
        ranges: dict[str, list[tuple[int, int]]] = {}
        for index, region in enumerate(regions):
            ranks = len(range(index, kv_world_size, len(regions)))
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
        for rank in range(self.kv_world_size):
            region_index = rank % len(self.regions)
            region = self.regions[region_index]
            ranks = len(range(region_index, self.kv_world_size, len(self.regions)))
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
            # Preserve persisted v1 schema and field names across config renames.
            "schema": "devdax_tp_payload_v1",
            "tp_size": self.kv_world_size,
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
            raise ValueError("rank_placement requires an encoded local kv_rank")
        rank = (kv_rank >> 16) & 0xFF
        expected = (
            (self.kv_world_size << 24) | (rank << 16) | (self.kv_world_size << 8) | rank
        )
        if kv_rank != expected or rank >= self.kv_world_size:
            raise ValueError(
                "rank_placement requires an encoded local kv_rank "
                "matching kv_world_size"
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
    group: int
    stride_bytes: int
    logical_offset_bytes: int


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
    group_slot_bytes: tuple[int, ...]


def resolve_payload_mappings(
    placement: DevDaxRankPlacement,
) -> list[DevDaxPayloadMapping]:
    """Return device ranges for an engine-resolved KV rank placement."""
    return [
        DevDaxPayloadMapping(
            p.devdax_path, p.payload_offset_bytes, p.payload_size_GiB << 30
        )
        for p in placement.placements()
    ]


def resolve_payload_geometry(
    config: "DaxCoordinatedL1Config",
    model_payload_bytes: int | tuple[int, ...],
    placement: DevDaxRankPlacement,
) -> DevDaxPayloadGeometry:
    """Carve the selected ownership policy from the configured payload.

    ``model_payload_bytes`` contains sizes in object-group order, or one size.
    Each group gets an equal count of slots with its own cache-line stride. The
    unusable byte tail is deliberately left outside the allocator. Equal
    ownership leaves slots not divisible by the participant count unused;
    participant-0 ownership can use every complete slot.

    Args:
        config: Validated DAX-Coordinated L1 configuration.
        model_payload_bytes: Registered object-group sizes in bytes.
        placement: Engine-resolved KV rank placement.

    Returns:
        Slot geometry and this participant's ownership range.

    Raises:
        ValueError: If the payload range cannot hold the required slots.
    """
    sizes = (
        (model_payload_bytes,)
        if isinstance(model_payload_bytes, int)
        else tuple(model_payload_bytes)
    )
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("model payload bytes must be positive")
    strides = tuple(_align_up(size, config.DEVDAX_CACHE_LINE_BYTES) for size in sizes)
    capacities = [mapping.size_bytes for mapping in resolve_payload_mappings(placement)]
    owners = config.participant_count if config.ownership_mode == "equal" else 1
    group_counts = tuple(size // sum(strides) // owners * owners for size in capacities)
    rank_slots = tuple(count * len(strides) for count in group_counts)
    if any(count < owners for count in group_counts):
        raise ValueError("payload range cannot hold one model slot per participant")
    slot_count = sum(rank_slots)
    used = sum(group_counts) * sum(strides)
    if used > sum(capacities):
        raise ValueError("configured payload slots exceed payload_size_GiB capacity")
    first_count = slot_count // owners
    owner_begin = min(config.participant_id, owners) * first_count
    owner_count = first_count if config.participant_id < owners else 0
    per_owner_rank_slots = tuple(count // owners for count in rank_slots)
    slot_ranges = []
    begin = 0
    for owner in range(owners):
        logical_base = 0
        for rank, total in enumerate(group_counts):
            count = total // owners
            group_base = 0
            for group, stride in enumerate(strides):
                offset = group_base + owner * count * stride
                slot_ranges.append(
                    DevDaxOwnerRankSlots(
                        owner=owner,
                        rank=rank,
                        first_slot=begin,
                        slot_count=count,
                        mapping_offset_bytes=offset,
                        group=group,
                        stride_bytes=stride,
                        logical_offset_bytes=logical_base + offset,
                    )
                )
                begin += count
                group_base += total * stride
            logical_base += capacities[rank]
    return DevDaxPayloadGeometry(
        model_payload_bytes=max(sizes),
        payload_slot_bytes=max(strides),
        payload_slot_count=slot_count,
        owner_slot_begin=owner_begin,
        owner_slot_count=owner_count,
        payload_bytes_used=used,
        payload_bytes_unused=sum(capacities) - used,
        rank_slot_counts=rank_slots,
        participant_0_slot_count=first_count,
        owner_rank_slot_counts=(
            per_owner_rank_slots
            if config.participant_id < owners
            else (0,) * len(rank_slots)
        ),
        slot_ranges=tuple(slot_ranges),
        group_slot_bytes=strides,
    )
