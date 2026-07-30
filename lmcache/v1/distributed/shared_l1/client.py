# SPDX-License-Identifier: Apache-2.0
"""MP-side Device-DAX mapping for coordinator-owned shared objects."""

# Standard
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol
import ctypes
import json
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1MemoryManagerConfig, SharedL1Config
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.shared_l1.pool import (
    ReadReservation,
    SharedObjectHandle,
    SharedRegionContract,
    WriteReservation,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)

_PUBLISH = 1
_ACQUIRE = 2
_VISIBILITY_SYMBOL = "lmcache_shared_l1_visibility_v1"


class _Visibility(Protocol):
    """Platform operation that publishes or acquires one exact DAX range."""

    @property
    def granularity(self) -> int: ...

    def apply(
        self,
        operation: int,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None: ...


class NativeDeviceDaxVisibility:
    """Call a platform-qualified visibility implementation supplied by the operator."""

    def __init__(self, library_path: str) -> None:
        path = Path(library_path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "shared-L1 visibility library must be an absolute regular file"
            )
        try:
            library = ctypes.CDLL(str(path), use_errno=True)
            function = getattr(library, _VISIBILITY_SYMBOL)
        except (OSError, AttributeError) as exc:
            raise RuntimeError(f"{path} does not provide {_VISIBILITY_SYMBOL}") from exc
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        function.restype = ctypes.c_int
        self._library = library
        self._function = function

    @property
    def granularity(self) -> int:
        """The current ABI isolates x86 cache-line-sized ranges."""
        return 64

    def apply(
        self,
        operation: int,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        """Apply publish/acquire without embedding host-specific code in LMCache."""
        result = self._function(
            b"software_fenced",
            operation,
            device_fd,
            ctypes.c_void_p(mapped_address),
            device_offset,
            length,
            generation,
        )
        if result:
            error_number = -int(result)
            detail = (
                os.strerror(error_number)
                if 0 < error_number < 4096
                else f"native status {result}"
            )
            raise RuntimeError(
                f"shared-L1 visibility operation {operation} failed: {detail}"
            )


def _stable_object_key(key: ObjectKey) -> str:
    """Return the same metadata key in every MP process."""
    return json.dumps(
        asdict(key.to_encoded_object_key()),
        sort_keys=True,
        separators=(",", ":"),
    )


class SharedL1Client:
    """Map one shared region locally while the coordinator owns its offsets."""

    def __init__(
        self,
        config: SharedL1Config,
        memory_config: L1MemoryManagerConfig,
        *,
        pool: Any | None = None,
        visibility: _Visibility | None = None,
        register_cuda: bool = True,
    ) -> None:
        if not memory_config.devdax_path:
            raise ValueError("shared L1 requires a Device-DAX path")
        self._manager: object | None = None
        if pool is None:
            # Import lazily because the coordinator imports the pool state.
            # First Party
            from lmcache.v1.mp_coordinator.shared_l1_service import (
                connect_shared_l1_manager,
                read_shared_l1_authkey,
            )

            manager = connect_shared_l1_manager(
                config.coordinator_host,
                config.coordinator_port,
                read_shared_l1_authkey(config.authkey_file),
            )
            pool = manager.get_pool()
            self._manager = manager
        self._pool = pool

        contract = pool.region_contract()
        expected = (
            config.region_id,
            memory_config.size_in_bytes,
            memory_config.align_bytes,
            config.layout_id,
        )
        actual = (
            contract.region_id,
            contract.capacity,
            contract.alignment,
            contract.layout_id,
        )
        if not isinstance(contract, SharedRegionContract) or actual != expected:
            raise ValueError(
                f"shared-L1 coordinator/local contract mismatch: {actual!r} "
                f"!= {expected!r}"
            )

        self._visibility = visibility or NativeDeviceDaxVisibility(
            config.visibility_library_path
        )
        granularity = self._visibility.granularity
        if (
            granularity <= 0
            or granularity & (granularity - 1)
            or contract.alignment % granularity
            or config.mapping_offset % max(mmap.PAGESIZE, granularity)
        ):
            raise ValueError("shared-L1 mapping violates visibility alignment")

        file_descriptor = os.open(memory_config.devdax_path, os.O_RDWR)
        mapping: mmap.mmap | None = None
        marker: ctypes.c_ubyte | None = None
        buffer = torch.empty(0, dtype=torch.uint8)
        registered_ptr: int | None = None
        try:
            mapping = mmap.mmap(
                file_descriptor,
                contract.capacity,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=config.mapping_offset,
            )
            marker = ctypes.c_ubyte.from_buffer(mapping)
            mapped_address = ctypes.addressof(marker)
            if mapped_address % granularity:
                raise RuntimeError("shared-L1 mapping base is not visibility-aligned")
            buffer = torch.frombuffer(mapping, dtype=torch.uint8)
            if register_cuda:
                pointer = buffer.data_ptr()
                if not current_device_spec.pin_memory(
                    pointer,
                    contract.capacity,
                ):
                    raise RuntimeError(
                        "CUDA host registration failed for shared Device-DAX; "
                        "pageable staging is not accepted"
                    )
                registered_ptr = pointer
        except BaseException:
            if registered_ptr is not None:
                current_device_spec.unpin_memory(registered_ptr)
            buffer = torch.empty(0, dtype=torch.uint8)
            if marker is not None:
                del marker
            if mapping is not None:
                mapping.close()
            os.close(file_descriptor)
            raise

        self._contract = contract
        self._mapping = mapping
        self._mapping_marker: ctypes.c_ubyte | None = marker
        self._mapped_address = mapped_address
        self._file_descriptor = file_descriptor
        self._buffer = buffer
        self._registered_ptr = registered_ptr
        self._mapping_offset = config.mapping_offset
        self._write_reservations: dict[ObjectKey, WriteReservation] = {}
        self._read_reservations: dict[ObjectKey, deque[ReadReservation]] = defaultdict(
            deque
        )
        # One immutable physical handle owns one reusable tensor view.
        self._memory_objects: dict[SharedObjectHandle, TensorMemoryObj] = {}
        self._lock = threading.RLock()
        self._closed = False

    def reserve_writes(
        self,
        keys: list[ObjectKey],
        layout: MemoryLayoutDesc,
    ) -> list[TensorMemoryObj | None]:
        """Reserve a whole L1Manager key list with one coordinator call."""
        with self._lock:
            self._ensure_open()
            if any(key in self._write_reservations for key in keys):
                raise RuntimeError("a key already has a local write reservation")
            reservations = list(
                self._pool.reserve_writes(
                    [(_stable_object_key(key), layout) for key in keys]
                )
            )
            granted = [item for item in reservations if item is not None]
            try:
                result = [
                    None
                    if reservation is None
                    else self._memory_object(reservation.handle, reservation.layout)
                    for reservation in reservations
                ]
            except BaseException:
                self._pool.abort_writes(granted)
                raise
            for key, reservation in zip(keys, reservations, strict=True):
                if reservation is not None:
                    self._write_reservations[key] = reservation
            return result

    def finish_writes(self, keys: list[ObjectKey]) -> None:
        """Publish every D2H-complete range, then atomically mark the batch VALID."""
        with self._lock:
            reservations = [self._write_reservations[key] for key in keys]
            try:
                for reservation in reservations:
                    self._apply_visibility(_PUBLISH, reservation.handle)
                self._pool.finish_writes(reservations)
            except BaseException:
                try:
                    self._pool.abort_writes(reservations)
                except BaseException:
                    logger.exception("failed to abort shared-L1 write batch")
                else:
                    for key, reservation in zip(keys, reservations, strict=True):
                        del self._write_reservations[key]
                        self._forget_memory_object(reservation.handle)
                raise
            for key in keys:
                del self._write_reservations[key]

    def abort_writes(self, keys: list[ObjectKey]) -> None:
        """Abort a failed batch while keeping a failed RPC retryable."""
        with self._lock:
            pairs = [
                (key, self._write_reservations[key])
                for key in keys
                if key in self._write_reservations
            ]
            if not pairs:
                return
            self._pool.abort_writes([reservation for _, reservation in pairs])
            for key, reservation in pairs:
                del self._write_reservations[key]
                self._forget_memory_object(reservation.handle)

    def reserve_reads(self, keys: list[ObjectKey]) -> list[TensorMemoryObj | None]:
        """Acquire all VALID hits from one coordinator batch."""
        with self._lock:
            self._ensure_open()
            reservations = list(
                self._pool.reserve_reads([_stable_object_key(key) for key in keys])
            )
            granted = [item for item in reservations if item is not None]
            try:
                result: list[TensorMemoryObj | None] = []
                for reservation in reservations:
                    if reservation is None:
                        result.append(None)
                        continue
                    self._apply_visibility(_ACQUIRE, reservation.handle)
                    result.append(
                        self._memory_object(reservation.handle, reservation.layout)
                    )
            except BaseException:
                self._pool.abort_reads(granted)
                raise
            for key, reservation in zip(keys, reservations, strict=True):
                if reservation is not None:
                    self._read_reservations[key].append(reservation)
            return result

    def finish_reads(self, keys: list[ObjectKey]) -> None:
        """Release a read batch after its H2D transfers complete."""
        with self._lock:
            reservations = [self._read_reservations[key][0] for key in keys]
            self._pool.finish_reads(reservations)
            for key in keys:
                self._read_reservations[key].popleft()
                if not self._read_reservations[key]:
                    del self._read_reservations[key]

    def abort_reads(self, keys: list[ObjectKey]) -> None:
        """Release the newest reservations for a cancelled batch."""
        with self._lock:
            pairs = [
                (key, self._read_reservations[key][-1])
                for key in keys
                if self._read_reservations.get(key)
            ]
            if not pairs:
                return
            self._pool.abort_reads([reservation for _, reservation in pairs])
            for key, _ in pairs:
                self._read_reservations[key].pop()
                if not self._read_reservations[key]:
                    del self._read_reservations[key]

    def get_memory_usage(self) -> tuple[int, int]:
        """Return monotonic allocated bytes and region capacity."""
        return int(self._pool.snapshot()["next_offset"]), self._contract.capacity

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Describe the CUDA-registered shared mapping."""
        return L1MemoryDesc(
            ptr=self._buffer.data_ptr(),
            size=self._contract.capacity,
            align_bytes=self._contract.alignment,
        )

    def memcheck(self) -> bool:
        """Check the mapping and the coordinator restart epoch."""
        return (
            not self._closed
            and self._buffer.numel() == self._contract.capacity
            and self._pool.region_contract() == self._contract
        )

    def close(self) -> None:
        """Quiesce transfers, release reservations, unpin, and unmap."""
        with self._lock:
            if self._closed:
                return
            self.abort_writes(list(self._write_reservations))
            all_reads = [
                reservation
                for reservations in self._read_reservations.values()
                for reservation in reservations
            ]
            if all_reads:
                self._pool.abort_reads(all_reads)
                self._read_reservations.clear()
            if self._registered_ptr is not None:
                if not current_device_spec.unpin_memory(self._registered_ptr):
                    raise RuntimeError(
                        "CUDA host unregistration failed for shared Device-DAX"
                    )
                self._registered_ptr = None
            for memory_obj in self._memory_objects.values():
                memory_obj.invalidate()
                memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)
            self._memory_objects.clear()
            self._buffer = torch.empty(0, dtype=torch.uint8)
            marker = self._mapping_marker
            self._mapping_marker = None
            del marker
            self._mapping.close()
            os.close(self._file_descriptor)
            self._closed = True

    def _apply_visibility(
        self,
        operation: int,
        handle: SharedObjectHandle,
    ) -> None:
        if handle.region_id != self._contract.region_id:
            raise ValueError("shared-L1 handle belongs to another region")
        end = handle.offset + handle.length
        if end > self._contract.capacity or handle.offset % self._contract.alignment:
            raise ValueError("shared-L1 handle lies outside the aligned region")
        self._visibility.apply(
            operation,
            self._file_descriptor,
            self._mapped_address + handle.offset,
            self._mapping_offset + handle.offset,
            handle.length,
            handle.generation,
        )

    def _memory_object(
        self,
        handle: SharedObjectHandle,
        layout: MemoryLayoutDesc,
    ) -> TensorMemoryObj:
        expected_length = get_size_bytes(layout.shapes, layout.dtypes)
        if handle.length != expected_length:
            raise ValueError(
                "shared-L1 object length does not match its layout: "
                f"{handle.length} != {expected_length}"
            )
        memory_obj = self._memory_objects.get(handle)
        if memory_obj is not None:
            if (
                memory_obj.get_shapes() != layout.shapes
                or memory_obj.get_dtypes() != layout.dtypes
            ):
                raise ValueError("shared-L1 handle has a different layout")
            return memory_obj
        end = handle.offset + handle.length
        memory_obj = TensorMemoryObj(
            raw_data=self._buffer[handle.offset : end],
            metadata=MemoryObjMetadata(
                shape=layout.shapes[0],
                dtype=layout.dtypes[0],
                address=handle.offset,
                phy_size=(
                    (handle.length + self._contract.alignment - 1)
                    // self._contract.alignment
                    * self._contract.alignment
                ),
                ref_count=1,
                fmt=MemoryFormat.KV_2LTD,
                shapes=layout.shapes,
                dtypes=layout.dtypes,
            ),
            parent_allocator=None,
        )
        self._memory_objects[handle] = memory_obj
        return memory_obj

    def _forget_memory_object(self, handle: SharedObjectHandle) -> None:
        memory_obj = self._memory_objects.pop(handle, None)
        if memory_obj is not None:
            memory_obj.invalidate()
            memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("shared-L1 client is closed")
