# SPDX-License-Identifier: Apache-2.0
"""Device-DAX mapping and offset-backed memory views for DAX-Coordinated L1."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
import ctypes
import mmap
import os
import stat

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    DaxCoordinatedL1Config,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_layout import (
    DevDaxPayloadGeometry,
    DevDaxPayloadMapping,
    resolve_payload_mappings,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import current_device_spec

# CUDA 13 on the qualified system accepts a 128 GiB Device-DAX registration,
# but rejects one monolithic 512 GiB cudaHostRegister call.  The transfer
# kernel already splits DMA at the Device-DAX alignment, so adjacent registered
# ranges remain transparent to payload views and asynchronous copies.
_CUDA_REGISTRATION_SEGMENT_BYTES = 128 << 30


def _parse_sysfs_integer(path: Path) -> int:
    value = path.read_text(encoding="ascii").strip()
    try:
        return int(value, 0)
    except ValueError:
        return int(value, 16)


def _read_devdax_attribute(devdax_path: str, attribute: str, sysfs_root: Path) -> int:
    """Read size/align through either sysfs device path, preserving access errors."""
    device_stat = os.stat(devdax_path)
    device_number = f"{os.major(device_stat.st_rdev)}:{os.minor(device_stat.st_rdev)}"
    for device_root in (
        sysfs_root / "dev" / "char" / device_number,
        sysfs_root / "bus" / "dax" / "devices" / Path(devdax_path).name,
    ):
        try:
            return _parse_sysfs_integer(device_root / attribute)
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise PermissionError(
                f"Device-DAX requires permission to read {attribute} "
                f"under {device_root}"
            ) from error
        except ValueError as error:
            raise ValueError(
                f"invalid Device-DAX {attribute} under {device_root}"
            ) from error
    if attribute == "size" and device_stat.st_size > 0:
        return device_stat.st_size
    raise FileNotFoundError(
        f"could not determine Device-DAX {attribute} for {devdax_path}"
    )


def _read_devdax_size(devdax_path: str, sysfs_root: Path = Path("/sys")) -> int:
    """Return positive capacity without reading the privileged HPA resource."""
    size = _read_devdax_attribute(devdax_path, "size", sysfs_root)
    if size <= 0:
        raise ValueError(f"invalid Device-DAX size for {devdax_path}")
    return size


def _read_devdax_alignment(devdax_path: str, sysfs_root: Path = Path("/sys")) -> int:
    """Return the device's power-of-two, page-multiple mmap alignment."""
    alignment = _read_devdax_attribute(devdax_path, "align", sysfs_root)
    if alignment < mmap.PAGESIZE or alignment & (alignment - 1):
        raise ValueError(f"invalid Device-DAX alignment for {devdax_path}")
    return alignment


def _align_up(value: int, alignment: int) -> int:
    """Round a positive byte count up to a power-of-two alignment."""
    return (value + alignment - 1) & ~(alignment - 1)


def _validate_mapping_ranges(
    mappings: list[DevDaxPayloadMapping], payload_alignment: int, portable: bool
) -> None:
    """Reject invalid device ranges and aliases before acquiring mappings."""
    ranges: dict[int, list[tuple[int, int]]] = {}
    for path, offset, size in mappings:
        device = os.stat(path)
        if not stat.S_ISCHR(device.st_mode):
            raise ValueError(
                "DAX-Coordinated L1 production mapping must be a character device"
            )
        device_alignment = _read_devdax_alignment(path)
        alignment = max(payload_alignment, device_alignment)
        if offset % alignment or size % (alignment if portable else device_alignment):
            raise ValueError(
                "DAX-Coordinated L1 range violates Device-DAX mmap alignment"
            )
        if offset + size > _read_devdax_size(path):
            raise ValueError(
                "DAX-Coordinated L1 metadata/payload offset range exceeds "
                "Device-DAX capacity"
            )
        previous = ranges.setdefault(device.st_rdev, [])
        if any(offset < end and begin < offset + size for begin, end in previous):
            raise ValueError(
                "DAX-Coordinated L1 metadata and payload offset ranges overlap"
            )
        previous.append((offset, offset + size))


@dataclass(frozen=True)
class _PayloadViewRange:
    """Translate a logical byte range into a rank mapping's byte range."""

    logical_offset_bytes: int
    size_bytes: int
    rank: int
    mapping_offset_bytes: int


class DaxCoordinatedL1Region:
    """Own one shared index mapping and the configured rank payload mappings."""

    def __init__(
        self,
        dax_coordinated_l1_config: DaxCoordinatedL1Config,
        memory_config: L1MemoryManagerConfig,
    ) -> None:
        """Validate configured ranges, map payloads and register them for GPU DMA.

        Metadata is mapped later by map_metadata when its model-dependent size
        is known. TP uses the first configured region's metadata arena for the common
        index. Other regions must leave the metadata offset unset. Raises ValueError
        for invalid geometry and OSError/RuntimeError for mapping/pinning failure.
        """
        if not memory_config.devdax_path:
            raise ValueError("DAX-Coordinated L1 requires --l1-devdax-path")
        if dax_coordinated_l1_config.devdax_path != memory_config.devdax_path:
            raise ValueError(
                "DAX-Coordinated L1 devdax_path must match memory_config.devdax_path"
            )
        self._alignment = memory_config.align_bytes
        if self._alignment < 64 or self._alignment & (self._alignment - 1):
            raise ValueError(
                "DAX-Coordinated L1 alignment must be a power of two >= 64"
            )
        rank_placement = dax_coordinated_l1_config.rank_placement
        self._portable = rank_placement.tp_size > 1
        metadata_region = rank_placement.regions[0]
        metadata_path = metadata_region.devdax_path
        assert metadata_region.metadata_offset_bytes is not None
        assert metadata_region.metadata_reservation_bytes is not None
        self._mapping_offset = int(metadata_region.metadata_offset_bytes)
        payloads = resolve_payload_mappings(dax_coordinated_l1_config)
        self._metadata_path = metadata_path
        self._metadata_alignment = _read_devdax_alignment(metadata_path)
        self._metadata_reservation = int(metadata_region.metadata_reservation_bytes)
        self._payload_mappings = payloads
        self._size = 0
        self._metadata_mapping: tuple[mmap.mmap, torch.Tensor | None] | None = None
        # Validate the full reservation before mapping payloads at any TP size.
        _validate_mapping_ranges(
            [
                DevDaxPayloadMapping(
                    metadata_path,
                    self._mapping_offset,
                    self._metadata_reservation,
                ),
                *payloads,
            ],
            self._alignment,
            self._portable,
        )
        self._payload_size = sum(payload.size_bytes for payload in payloads)
        self._payload_mapping_offset = payloads[0].offset_bytes
        self._mappings: list[tuple[mmap.mmap, torch.Tensor | None]] = []
        self._pinned_ranges: list[tuple[int, int]] = []
        # Logical byte start, byte count, payload rank, offset within its mapping.
        self._payload_ranges: list[_PayloadViewRange] = (
            [_PayloadViewRange(0, self._payload_size, 0, 0)]
            if len(payloads) == 1
            else []
        )
        try:
            for path, offset, size in payloads:
                fd = os.open(path, os.O_RDWR)
                try:
                    mapping = mmap.mmap(
                        fd,
                        size,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                        offset=offset,
                    )
                finally:
                    os.close(fd)
                self._mappings.append((mapping, None))
                # Tensor owns the ctypes export, which prevents unmapping live views.
                self._mappings[-1] = (
                    mapping,
                    torch.frombuffer(
                        (ctypes.c_uint8 * size).from_buffer(mapping), dtype=torch.uint8
                    ),
                )
            self._register_cuda_mapping()
        except Exception:
            self.close()
            raise

    @property
    def base_address(self) -> int:
        """Return the local address of the common metadata mapping."""
        if self._metadata_mapping is None or self._metadata_mapping[1] is None:
            raise RuntimeError("DAX-Coordinated L1 metadata is not mapped")
        return self._metadata_mapping[1].data_ptr()

    @property
    def size(self) -> int:
        """Return the common metadata mapping size."""
        return self._size

    @property
    def mapping_offset(self) -> int:
        """Return the common metadata device offset."""
        return self._mapping_offset

    @property
    def payload_mapping_offset(self) -> int:
        """Return the first payload mapping's device offset."""
        return self._payload_mapping_offset

    @property
    def cuda_registered(self) -> bool:
        """Whether every payload byte is registered for GPU transfer."""
        return sum(size for _, size in self._pinned_ranges) == self._payload_size

    def map_metadata(self, metadata_size_bytes: int) -> None:
        """Map the model-dependent index without changing payload registrations.

        Args:
            metadata_size_bytes: Positive native metadata extent in bytes.

        Repeated calls with the same aligned extent reuse the mapping. Raises
        ValueError for invalid ranges, reservation overflow or a changed extent.
        """
        if not self._mappings:
            raise RuntimeError("DAX-Coordinated L1 region is closed")
        if metadata_size_bytes <= 0:
            raise ValueError("DAX-Coordinated L1 requires a positive metadata size")
        size = _align_up(metadata_size_bytes, self._metadata_alignment)
        if self._metadata_mapping is not None:
            if size != self._size:
                raise ValueError("DAX-Coordinated L1 metadata size is already bound")
            return
        if size > self._metadata_reservation:
            raise ValueError("native metadata exceeds metadata_reservation_bytes")
        _validate_mapping_ranges(
            [
                DevDaxPayloadMapping(
                    self._metadata_path,
                    self._mapping_offset,
                    self._metadata_reservation,
                ),
                *self._payload_mappings,
            ],
            self._alignment,
            self._portable,
        )
        fd = os.open(self._metadata_path, os.O_RDWR)
        try:
            mapping = mmap.mmap(
                fd,
                size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=self._mapping_offset,
            )
        finally:
            os.close(fd)
        try:
            tensor = torch.frombuffer(
                (ctypes.c_uint8 * size).from_buffer(mapping), dtype=torch.uint8
            )
        except Exception:
            mapping.close()
            raise
        self._metadata_mapping = (mapping, tensor)
        self._size = size

    def payload_ranges(
        self,
        geometry: DevDaxPayloadGeometry,
    ) -> list[tuple[int, int, int, int]]:
        """Bind owner-contiguous logical slots to rank-specific physical ranges.

        Return native tuples (first slot, slot count, local address, rank).
        Each rank retains the original equal or participant-0-only ownership.
        Process addresses are local; the placement digest fences the shared plan.

        Args:
            geometry: Resolved owner/rank slot ranges, in native slot order.

        Returns:
            Native slot tuples with process-local addresses for this mapping.
        """
        self._payload_ranges = []
        native = []
        for slots in geometry.slot_ranges:
            native.append(
                (
                    slots.first_slot,
                    slots.slot_count,
                    self._tensor(slots.rank).data_ptr() + slots.mapping_offset_bytes,
                    slots.rank,
                )
            )
            self._payload_ranges.append(
                _PayloadViewRange(
                    logical_offset_bytes=slots.first_slot * geometry.payload_slot_bytes,
                    size_bytes=slots.slot_count * geometry.payload_slot_bytes,
                    rank=slots.rank,
                    mapping_offset_bytes=slots.mapping_offset_bytes,
                )
            )
        return native

    def make_memory_obj(
        self,
        payload_offset: int,
        payload_length: int,
        layout_desc: MemoryLayoutDesc | None = None,
    ) -> MemoryObj:
        """Create a non-owning view from the native logical payload offset.

        Raises ValueError if the view crosses a placement boundary. A remote
        layout unknown locally is represented as bytes, as in the TP=1 path.
        """
        for view_range in self._payload_ranges:
            relative = payload_offset - view_range.logical_offset_bytes
            if (
                0 <= relative
                and payload_length > 0
                and relative + payload_length <= view_range.size_bytes
            ):
                begin = view_range.mapping_offset_bytes + relative
                raw_data = self._tensor(view_range.rank)[begin : begin + payload_length]
                break
        else:
            raise ValueError("DAX-Coordinated L1 payload view is outside the mapping")
        shapes = (
            list(layout_desc.shapes) if layout_desc else [torch.Size([payload_length])]
        )
        dtypes = list(layout_desc.dtypes) if layout_desc else [torch.uint8]
        metadata = MemoryObjMetadata(
            shape=shapes[0],
            dtype=dtypes[0],
            address=payload_offset,
            phy_size=payload_length,
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
            shapes=shapes,
            dtypes=dtypes,
        )
        return TensorMemoryObj(raw_data, metadata, parent_allocator=None)

    def get_memory_desc(self) -> tuple[int, int, int]:
        """Return a contiguous descriptor, or an empty one for multiple mappings."""
        if len(self._mappings) != 1:
            return 0, 0, self._alignment
        return self._tensor(0).data_ptr(), self._payload_size, self._alignment

    def synchronize(self) -> None:
        """Wait for DMA before releasing any shared reservations or registrations."""
        if self._pinned_ranges and torch_dev.is_available():
            if self._portable:
                for device in range(torch_dev.device_count()):
                    torch_dev.synchronize(device)
            else:
                torch_dev.synchronize()

    def close(self) -> None:
        """Drain DMA, unregister payloads, and unmap."""
        self.synchronize()
        while self._pinned_ranges:
            pointer, _ = self._pinned_ranges[-1]
            current_device_spec.unpin_memory(pointer)
            self._pinned_ranges.pop()
        if self._metadata_mapping is not None:
            mapping = self._metadata_mapping[0]
            self._metadata_mapping = (mapping, None)
            mapping.close()
            self._metadata_mapping = None
            self._size = 0
        while self._mappings:
            mapping = self._mappings[-1][0]
            # Drop exported ctypes/Tensor references before closing the mmap.
            self._mappings[-1] = (mapping, None)
            mapping.close()
            self._mappings.pop()

    def _tensor(self, index: int) -> torch.Tensor:
        tensor = self._mappings[index][1]
        assert tensor is not None
        return tensor

    def _register_cuda_mapping(self) -> None:
        """Register every rank mapping in bounded contiguous CUDA ranges."""
        if not current_device_spec.is_pin_supported:
            raise RuntimeError(
                "DAX-Coordinated L1 requires host-memory registration support"
            )
        for index in range(len(self._mappings)):
            length = len(self._mappings[index][0])
            for offset in range(0, length, _CUDA_REGISTRATION_SEGMENT_BYTES):
                size = min(_CUDA_REGISTRATION_SEGMENT_BYTES, length - offset)
                pointer = self._tensor(index).data_ptr() + offset
                registered = (
                    current_device_spec.pin_memory(pointer, size, flags=0x01)
                    if self._portable
                    else current_device_spec.pin_memory(pointer, size)
                )
                if not registered:
                    raise RuntimeError(
                        "cudaHostRegister failed for DAX-Coordinated L1 "
                        f"segment offset={offset} size={size}; "
                        "refusing pageable or CPU-staged payload transfers"
                    )
                self._pinned_ranges.append((pointer, size))
