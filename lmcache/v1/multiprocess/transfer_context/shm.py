# SPDX-License-Identifier: Apache-2.0
"""Shared-memory EngineDrivenContext implementation for multiprocess mode."""

# Standard
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
from typing import Any
import ctypes

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    EngineDrivenPayload,
    EngineDrivenStorePreparation,
)
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)


@dataclass(frozen=True)
class ShmSlotDescriptor:
    """Describe one tensor slot in the shared-memory pool.

    Args:
        offset: Byte offset into the shared-memory pool.
        length: Byte length of the slot.
        shape: Logical tensor shape to view at the slot.
        dtype: Torch dtype attribute name, such as ``"bfloat16"``.
    """

    offset: int
    length: int
    shape: list[int]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the slot descriptor into the MQ context schema.

        Returns:
            Dict payload shared between the server and worker for one SHM slot.
        """
        return {
            "offset": self.offset,
            "length": self.length,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShmSlotDescriptor":
        """Parse a slot descriptor from the MQ context schema.

        Args:
            d: Mapping containing ``offset``, ``length``, ``shape``, and
                ``dtype`` fields.

        Returns:
            Parsed immutable slot descriptor.

        Raises:
            KeyError: If any required field is missing.
            TypeError: If ``shape`` cannot be converted with ``list(...)``.
            ValueError: If numeric fields cannot be coerced to integers.
        """
        return cls(
            offset=int(d["offset"]),
            length=int(d["length"]),
            shape=list(d["shape"]),
            dtype=str(d["dtype"]),
        )


class EngineDrivenContextShm(EngineDrivenContext):
    """Shared-memory implementation of :class:`EngineDrivenContext`."""

    def __init__(
        self,
        metadata: EngineDrivenContextMetadata,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        shm_name: str,
        pool_size: int,
    ) -> None:
        super().__init__(metadata, mq_client, mq_timeout)
        if not shm_name or pool_size <= 0:
            raise ValueError("shm_name must be non-empty and pool_size must be > 0")

        self._shm_name = shm_name
        self._pool_size = pool_size
        self._shm: shared_memory.SharedMemory | None = None
        self._shm_buffer: memoryview | None = None
        self._pinned = False
        self._pinned_ptr = 0
        self._pinned_size = 0
        try:
            self._shm = shared_memory.SharedMemory(
                name=shm_name.lstrip("/"), create=False
            )
            # The SHM segment is owned by the server process. Unregister it
            # from this worker's resource tracker so that Python does not
            # unlink the segment when this worker exits.
            unregister(f"/{self._shm.name}", "shared_memory")
            self._shm_buffer = self._shm.buf
            # pin memory is per process
            # the shm might be pinned on lmcache server side already
            # pin memory here is for worker side for fast DMA copy
            self._pin_shm_buffer()
            logger.info("SHM pinned=%s for shm_name=%s", self._pinned, self._shm_name)
        except Exception:
            self._shm = None
            self._shm_buffer = None
            raise

    def _make_tensor_view(
        self,
        offset: int,
        length: int,
        shape: list[int],
        dtype_str: str,
    ) -> torch.Tensor:
        """Create a tensor view over a SHM slot via ``torch.frombuffer``."""
        dtype = getattr(torch, dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid torch dtype string: {dtype_str}")
        if offset < 0 or length < 0:
            raise ValueError("SHM slot offset and length must be non-negative")
        if any(dimension < 0 for dimension in shape):
            raise ValueError("SHM slot shape dimensions must be non-negative")
        itemsize = torch.empty((), dtype=dtype).element_size()
        if itemsize <= 0:
            raise ValueError(f"Invalid dtype size for {dtype_str}")
        if length % itemsize != 0:
            raise ValueError(
                f"SHM slot length {length} is not aligned to dtype {dtype_str}"
            )
        expected_length = torch.Size(shape).numel() * itemsize
        if length != expected_length:
            raise ValueError(
                f"SHM slot length {length} does not match shape {shape} and "
                f"dtype {dtype_str} ({expected_length} bytes)"
            )
        count = length // itemsize
        if self._shm_buffer is None:
            raise RuntimeError(
                f"Shared memory buffer not initialized for shm_name={self._shm_name}"
            )
        if offset + length > min(self._pool_size, len(self._shm_buffer)):
            raise ValueError(
                f"SHM slot [{offset}, {offset + length}) exceeds pool size "
                f"{min(self._pool_size, len(self._shm_buffer))}"
            )
        tensor_1d = torch.frombuffer(
            self._shm_buffer, dtype=dtype, count=count, offset=offset
        )
        return tensor_1d.view(torch.Size(shape))

    def _build_slot_tensors(self, slots: list[dict[str, Any]]) -> list[torch.Tensor]:
        descriptors = [ShmSlotDescriptor.from_dict(slot) for slot in slots]
        return [
            self._make_tensor_view(
                offset=descriptor.offset,
                length=descriptor.length,
                shape=descriptor.shape,
                dtype_str=descriptor.dtype,
            )
            for descriptor in descriptors
        ]

    def _build_multi_group_slot_tensors(
        self, object_groups: list[dict[str, Any]]
    ) -> tuple[list[list[list[torch.Tensor]]], list[list[int]]]:
        """Build and validate multi-object-group tensor views from an MQ context.

        The wire order is fixed: object group, chunk within that object group,
        then kernel-group tensor within that chunk.  Every registered object
        group is represented, including groups with no writable cache misses.

        Args:
            object_groups: ``context["object_groups"]`` sent by the server.

        Returns:
            Nested SHM views and their sparse per-group chunk indices.

        Raises:
            ValueError: If the context does not exactly match the registered
                object-group layouts or any slot descriptor is invalid.
        """
        layouts = self.metadata.object_group_layout_descs
        if len(object_groups) != len(layouts):
            raise ValueError(
                "SHM object-group response count does not match registration"
            )

        grouped_tensors: list[list[list[torch.Tensor]]] = []
        grouped_chunk_indices: list[list[int]] = []
        for object_group_id, (group_context, layout) in enumerate(
            zip(object_groups, layouts, strict=True)
        ):
            if group_context.get("object_group_id") != object_group_id:
                raise ValueError(
                    "SHM object groups are not in deterministic registration order"
                )
            chunk_indices = group_context.get("chunk_indices")
            slot_chunks = group_context.get("slots")
            if not isinstance(chunk_indices, list) or not isinstance(slot_chunks, list):
                raise ValueError(
                    f"SHM object group {object_group_id} has invalid slots or "
                    "chunk_indices"
                )
            if len(chunk_indices) != len(slot_chunks):
                raise ValueError(
                    f"SHM object group {object_group_id} has {len(slot_chunks)} "
                    f"slot chunks but {len(chunk_indices)} chunk indices"
                )
            if any(
                not isinstance(chunk_index, int)
                or isinstance(chunk_index, bool)
                or chunk_index < 0
                for chunk_index in chunk_indices
            ) or chunk_indices != sorted(set(chunk_indices)):
                raise ValueError(
                    f"SHM object group {object_group_id} has invalid chunk ordering"
                )

            group_tensors: list[list[torch.Tensor]] = []
            for chunk_id, slot_descriptors in enumerate(slot_chunks):
                if not isinstance(slot_descriptors, list) or len(
                    slot_descriptors
                ) != len(layout.shapes):
                    raise ValueError(
                        f"SHM object group {object_group_id}, chunk {chunk_id} "
                        "does not match the registered kernel-group layout"
                    )
                chunk_tensors = self._build_slot_tensors(slot_descriptors)
                for tensor_id, (tensor, shape, dtype) in enumerate(
                    zip(
                        chunk_tensors,
                        layout.shapes,
                        layout.dtypes,
                        strict=True,
                    )
                ):
                    if tensor.shape != shape or tensor.dtype != dtype:
                        raise ValueError(
                            f"SHM object group {object_group_id}, chunk {chunk_id}, "
                            f"tensor {tensor_id} does not match the registered layout"
                        )
                group_tensors.append(chunk_tensors)
            grouped_tensors.append(group_tensors)
            grouped_chunk_indices.append(chunk_indices)
        return grouped_tensors, grouped_chunk_indices

    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> EngineDrivenStorePreparation | None:
        """Request writable SHM slots for one store operation."""
        future = self.mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_STORE),
        )
        # wait() first so a timeout raises exactly one LMCacheTimeoutError
        # (one event); result() then returns without its own timeout.
        if not future.wait(timeout=self.mq_timeout):
            raise LMCacheTimeoutError(
                f"PREPARE_STORE timed out for instance_id={instance_id} "
                f"after {self.mq_timeout}s",
                session_id=key.request_id,
            )
        response = future.result()
        context = response.context if isinstance(response.context, dict) else {}
        if self.metadata.object_group_layout_descs:
            object_groups = context.get("object_groups")
            if not isinstance(object_groups, list):
                raise ValueError("Multi-group SHM store response has no object_groups")
            return self._build_multi_group_slot_tensors(object_groups)

        slots = context.get("slots")
        if not isinstance(slots, list):
            return None
        if not slots:
            # Server explicitly signals all chunks are already cached.
            return [], []
        chunk_indices: list[int] = context["chunk_indices"]
        return self._build_slot_tensors(slots), chunk_indices

    def commit_store(
        self, key: IPCCacheServerKey, instance_id: int, _chunks: EngineDrivenPayload
    ) -> bool:
        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, instance_id, b""],
            get_response_class(RequestType.COMMIT_STORE),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def abort_store(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Release SHM write reservations after a failed worker-side store."""
        future = self.mq_client.submit_request(
            RequestType.ABORT_STORE,
            [key, instance_id],
            get_response_class(RequestType.ABORT_STORE),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> EngineDrivenPayload | None:
        """Request readable SHM slots for one retrieve operation."""
        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success:
            return None
        context = response.context if isinstance(response.context, dict) else {}
        if self.metadata.object_group_layout_descs:
            object_groups = context.get("object_groups")
            if not isinstance(object_groups, list):
                raise ValueError(
                    "Multi-group SHM retrieve response has no object_groups"
                )
            grouped_tensors, _ = self._build_multi_group_slot_tensors(object_groups)
            return grouped_tensors

        slots = context.get("slots", [])
        return self._build_slot_tensors(slots) if slots else None

    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        future = self.mq_client.submit_request(
            RequestType.COMMIT_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.COMMIT_RETRIEVE),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def close(self) -> None:
        if self._shm is None:
            return
        self._unpin_shm_buffer()
        try:
            self._shm.close()
        finally:
            self._shm = None
            self._shm_buffer = None

    def _pin_shm_buffer(self) -> None:
        """Pin the SHM buffer as page-locked host memory via cudaHostRegister.

        Enables faster async D2H CUDA copies to the SHM region. If pinning is
        not available or fails, logs a warning and continues without pinning.
        """
        if self._shm_buffer is None or not torch_dev.is_available():
            return
        try:
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm_buffer))
        except Exception as exc:
            logger.warning(
                "Failed to get pointer for shm_name=%s: %r; "
                "D2H copies will be synchronous",
                self._shm_name,
                exc,
            )
            return
        if current_device_spec.pin_memory(ptr, self._pool_size):
            self._pinned = True
            self._pinned_ptr = ptr
            self._pinned_size = self._pool_size
        else:
            logger.warning(
                "pin_memory failed for shm_name=%s ptr=%#x size=%d; "
                "D2H copies will be synchronous",
                self._shm_name,
                ptr,
                self._pool_size,
            )

    def _unpin_shm_buffer(self) -> None:
        """Unpin the SHM buffer if it was previously pinned via cudaHostRegister."""
        if not self._pinned or self._pinned_ptr == 0:
            return
        current_device_spec.unpin_memory(self._pinned_ptr)
        self._pinned = False
        self._pinned_ptr = 0
        self._pinned_size = 0
