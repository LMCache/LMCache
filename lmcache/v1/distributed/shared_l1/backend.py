# SPDX-License-Identifier: Apache-2.0
"""CUDA-registered views of coordinator-owned Device-DAX objects."""

# Standard
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import round_up
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.shared_l1.layouts import layout_to_wire, wire_to_layout
from lmcache.v1.distributed.shared_l1.visibility import (
    ACQUIRE,
    PUBLISH,
    NativeDeviceDaxVisibility,
)
from lmcache.v1.memory_coordinator.api import (
    LookupHit,
    ReservationRef,
    SharedObjectHandle,
    WireLayout,
    WriteGrant,
    WriteReserveItem,
)
from lmcache.v1.memory_coordinator.client import MemoryCoordinatorHttpClient
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)


def _ref(grant: WriteGrant) -> ReservationRef:
    return ReservationRef(key=grant.key, token=grant.token)


class SharedDevDaxL1Backend:
    """Map one fixed shared pool; payload bytes never reach the coordinator.

    Args:
        devdax_path: Host-local Device-DAX device.
        capacity_bytes: Expected pool capacity.
        alignment_bytes: Expected allocation alignment.
        region_id: Expected physical-region identity.
        layout_id: Expected layout fingerprint.
        mapping_offset_bytes: Host-local start of the pool in the device.
        coordinator_endpoint: HTTP service URL.
        coordinator_token_file: Absolute token path.
        visibility_library_path: Qualified library path.

    Raises:
        ValueError: The contract or mapping alignment is incompatible.
        RuntimeError: CUDA registration fails; no pageable fallback is used.
        OSError: Opening, mapping, or loading the visibility library fails.
    """

    def __init__(
        self,
        *,
        devdax_path: str,
        capacity_bytes: int,
        alignment_bytes: int,
        region_id: str,
        layout_id: str,
        mapping_offset_bytes: int = 0,
        coordinator_endpoint: str,
        coordinator_token_file: str,
        visibility_library_path: str,
    ) -> None:
        if not devdax_path:
            raise ValueError("shared L1 requires a Device-DAX path")
        client = MemoryCoordinatorHttpClient(
            coordinator_endpoint, coordinator_token_file
        )
        self._client = client

        contract = client.region_contract()
        expected = (region_id, capacity_bytes, alignment_bytes, layout_id)
        actual = (
            contract.region_id,
            contract.capacity_bytes,
            contract.alignment_bytes,
            contract.layout_id,
        )
        if actual != expected:
            client.close()
            raise ValueError(
                f"shared-L1 coordinator/local contract mismatch: {actual!r} "
                f"!= {expected!r}"
            )

        file_descriptor: int | None = None
        mapping: mmap.mmap | None = None
        buffer = torch.empty(0, dtype=torch.uint8)
        registered_ptr: int | None = None
        try:
            visibility = NativeDeviceDaxVisibility(visibility_library_path)
            self._visibility = visibility
            granularity = visibility.granularity
            if (
                granularity <= 0
                or granularity & (granularity - 1)
                or contract.alignment_bytes % granularity
                or mapping_offset_bytes % max(mmap.PAGESIZE, granularity)
            ):
                raise ValueError("shared-L1 mapping violates visibility alignment")

            file_descriptor = os.open(devdax_path, os.O_RDWR)
            mapping = mmap.mmap(
                file_descriptor,
                contract.capacity_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=mapping_offset_bytes,
            )
            buffer = torch.frombuffer(mapping, dtype=torch.uint8)
            mapped_address = buffer.data_ptr()
            if mapped_address % granularity:
                raise RuntimeError("shared-L1 mapping base is not visibility-aligned")
            if not current_device_spec.pin_memory(
                mapped_address, contract.capacity_bytes
            ):
                raise RuntimeError(
                    "CUDA host registration failed for shared Device-DAX; "
                    "pageable staging is not accepted"
                )
            registered_ptr = mapped_address
        except BaseException:
            if registered_ptr is not None:
                current_device_spec.unpin_memory(registered_ptr)
            buffer = torch.empty(0, dtype=torch.uint8)
            if mapping is not None:
                mapping.close()
            if file_descriptor is not None:
                os.close(file_descriptor)
            client.close()
            raise

        self._contract = contract
        self._mapping = mapping
        self._mapped_address = mapped_address
        self._file_descriptor = file_descriptor
        self._buffer = buffer
        self._registered_ptr = registered_ptr
        self._mapping_offset = mapping_offset_bytes
        self._write_grants: dict[ObjectKey, WriteGrant] = {}
        # One immutable physical handle owns one reusable tensor view.
        self._memory_objects: dict[SharedObjectHandle, TensorMemoryObj] = {}
        self._lock = threading.RLock()
        self._closed = False

    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> list[TensorMemoryObj | None]:
        """Reserve unique keys with layout_desc; return views or None for duplicates.

        Raise RuntimeError for local reservations, OutOfSpaceError if the batch
        cannot fit, or StaleEpochError if the client is fenced. Invalid grants
        raise ValueError before exposing views.
        """
        with self._lock:
            self._ensure_open()
            if any(key in self._write_grants for key in keys):
                raise RuntimeError("a key already has a local write reservation")
            wire_layout = layout_to_wire(layout_desc)
            grants = self._client.reserve_writes(
                [
                    WriteReserveItem(
                        key=key.to_encoded_object_key(),
                        layout=wire_layout,
                    )
                    for key in keys
                ]
            )
            try:
                layouts = self._validate_results(keys, grants, wire_layout)
            except BaseException:
                self._client.abort_writes(
                    [_ref(grant) for grant in grants if grant is not None]
                )
                raise
            pending = {
                key: grant
                for key, grant in zip(keys, grants, strict=True)
                if grant is not None
            }
            self._write_grants.update(pending)
            try:
                result = [
                    None
                    if grant is None or layout is None
                    else self._memory_object(grant.handle, layout)
                    for grant, layout in zip(grants, layouts, strict=True)
                ]
            except BaseException:
                self.abort_write(list(pending))
                raise
            return result

    def finish_write(self, keys: list[ObjectKey]) -> None:
        """Publish D2H-complete ranges for keys, then atomically commit the batch.

        Propagate visibility/RPC errors after attempting abort. Failed aborts
        retain local grants so cleanup can be retried.
        """
        with self._lock:
            grants = [self._write_grants[key] for key in keys]
            refs = [_ref(grant) for grant in grants]
            try:
                for grant in grants:
                    self._apply_visibility(PUBLISH, grant.handle)
                self._client.finish_writes(refs)
            except BaseException:
                try:
                    self.abort_write(keys)
                except BaseException:
                    logger.exception("failed to abort shared-L1 write batch")
                raise
            for key in keys:
                del self._write_grants[key]

    def abort_write(self, keys: list[ObjectKey]) -> None:
        """Abort a failed batch while keeping a failed RPC retryable."""
        with self._lock:
            pairs = [
                (key, self._write_grants[key])
                for key in keys
                if key in self._write_grants
            ]
            if not pairs:
                return
            self._client.abort_writes([_ref(grant) for _, grant in pairs])
            for key, grant in pairs:
                del self._write_grants[key]
                self._forget_memory_object(grant.handle)

    def reserve_read(self, keys: list[ObjectKey]) -> list[TensorMemoryObj | None]:
        """Look up keys and acquire readable views; return None for each miss.

        Propagate stale-epoch, descriptor-validation and visibility errors.
        """
        with self._lock:
            self._ensure_open()
            hits = self._client.lookup([key.to_encoded_object_key() for key in keys])
            layouts = self._validate_results(keys, hits)
            result: list[TensorMemoryObj | None] = []
            for hit, layout in zip(hits, layouts, strict=True):
                if hit is None or layout is None:
                    result.append(None)
                    continue
                self._apply_visibility(ACQUIRE, hit.handle)
                result.append(self._memory_object(hit.handle, layout))
            return result

    def get_memory_usage(self) -> tuple[int, int]:
        """Return monotonic allocated bytes and region capacity."""
        return self._client.get_memory_usage()

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Describe the CUDA-registered shared mapping."""
        return L1MemoryDesc(
            ptr=self._buffer.data_ptr(),
            size=self._contract.capacity_bytes,
            align_bytes=self._contract.alignment_bytes,
        )

    def memcheck(self) -> bool:
        """Check the mapping and the coordinator restart epoch."""
        if self._closed or self._buffer.numel() != self._contract.capacity_bytes:
            return False
        try:
            status = self._client.status()
        except Exception:
            return False
        return status.region == self._contract

    def close(self) -> None:
        """Abort pending writes, unregister CUDA, invalidate views, then unmap.

        Callers must first drain GPU streams and release exported references.
        Raise RuntimeError, retaining the mapping for retry, if exports remain
        or CUDA unregistration fails. An export raises the reference count
        above the backend cache's own reference.
        """
        with self._lock:
            if self._closed:
                return
            if any(
                obj.is_valid() and obj.get_ref_count() > 1
                for obj in self._memory_objects.values()
            ):
                raise RuntimeError("shared-L1 close refused: exported views are live")
            self.abort_write(list(self._write_grants))
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
            self._mapping.close()
            os.close(self._file_descriptor)
            self._client.close()
            self._closed = True

    def _apply_visibility(self, operation: int, handle: SharedObjectHandle) -> None:
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
        memory_obj = self._memory_objects.get(handle)
        if memory_obj is not None:
            if (
                memory_obj.get_shapes() != layout.shapes
                or memory_obj.get_dtypes() != layout.dtypes
            ):
                raise ValueError("shared-L1 handle has a different layout")
            return memory_obj
        memory_obj = TensorMemoryObj(
            raw_data=self._buffer[handle.offset : handle.offset + handle.length],
            metadata=MemoryObjMetadata(
                shape=layout.shapes[0],
                dtype=layout.dtypes[0],
                address=handle.offset,
                phy_size=round_up(handle.length, self._contract.alignment_bytes),
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

    def _validate_results(
        self,
        keys: list[ObjectKey],
        results: list[WriteGrant | LookupHit | None],
        expected_layout: WireLayout | None = None,
    ) -> list[MemoryLayoutDesc | None]:
        """Validate the entire batch before exposing views or calling native code."""
        if len(results) != len(keys):
            raise ValueError("coordinator returned the wrong number of results")
        layouts: list[MemoryLayoutDesc | None] = []
        for key, result in zip(keys, results, strict=True):
            if result is None:
                layouts.append(None)
                continue
            if result.key != key.to_encoded_object_key() or (
                isinstance(result, WriteGrant) and not result.token
            ):
                raise ValueError("coordinator returned a result for another key")
            if expected_layout is not None and result.layout != expected_layout:
                raise ValueError("coordinator changed the requested write layout")
            layout = wire_to_layout(result.layout)
            self._validate_handle(
                result.handle,
                get_size_bytes(layout.shapes, layout.dtypes),
            )
            layouts.append(layout)
        return layouts

    def _validate_handle(
        self,
        handle: SharedObjectHandle,
        expected_length: int,
    ) -> None:
        if handle.region_id != self._contract.region_id:
            raise ValueError("shared-L1 handle belongs to another region")
        capacity = self._contract.capacity_bytes
        if (
            handle.length != expected_length
            or handle.offset % self._contract.alignment_bytes
            or handle.offset > capacity
            or handle.length > capacity - handle.offset
        ):
            raise ValueError("shared-L1 handle lies outside the aligned region")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("shared-L1 backend is closed")
