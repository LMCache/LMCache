# SPDX-License-Identifier: Apache-2.0
"""Shared-memory NonGpuContext implementation for multiprocess mode."""

# Standard
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transfer_context.base import (
    NonGpuContext,
    NonGpuContextMetadata,
)


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


class NonGpuContextShm(NonGpuContext):
    """Shared-memory implementation of :class:`NonGpuContext`."""

    def __init__(
        self,
        metadata: NonGpuContextMetadata,
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
        try:
            self._shm = shared_memory.SharedMemory(
                name=shm_name.lstrip("/"), create=False
            )
            # The SHM segment is owned by the server process. Unregister it
            # from this worker's resource tracker so that Python does not
            # unlink the segment when this worker exits.
            unregister(f"/{self._shm.name}", "shared_memory")
            self._shm_buffer = self._shm.buf
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
        itemsize = torch.empty((), dtype=dtype).element_size()
        if itemsize <= 0:
            raise ValueError(f"Invalid dtype size for {dtype_str}")
        count = length // itemsize
        if self._shm_buffer is None:
            raise RuntimeError(
                f"Shared memory buffer not initialized for shm_name={self._shm_name}"
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

    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        future = self.mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_STORE),
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError as err:
            raise TimeoutError(
                f"PREPARE_STORE timed out for instance_id={instance_id} "
                f"after {self.mq_timeout}s"
            ) from err
        context = response.context if isinstance(response.context, dict) else {}
        slots = context.get("slots")
        if not isinstance(slots, list):
            return None
        if not slots:
            # Server explicitly signals all chunks are already cached.
            return [], []
        chunk_indices: list[int] = context["chunk_indices"]
        return self._build_slot_tensors(slots), chunk_indices

    def commit_store(
        self, key: IPCCacheServerKey, instance_id: int, _chunks: list[torch.Tensor]
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

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[torch.Tensor] | None:
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
        slots = response.context.get("slots", [])
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
        try:
            self._shm.close()
        finally:
            self._shm = None
            self._shm_buffer = None
