# SPDX-License-Identifier: Apache-2.0
"""Inline EngineDrivenContext implementation for multiprocess mode."""

# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
)
from lmcache.v1.multiprocess.transfer_context.tensor_wire import (
    decode_grouped_tensors,
    decode_tensors,
    encode_grouped_tensors,
    encode_tensors,
)
from lmcache.v1.multiprocess.transport.base import RequestClient


class EngineDrivenContextInline(EngineDrivenContext):
    """Safe inline implementation of :class:`EngineDrivenContext`.

    Transport mechanism:
    - **Store**: ``prepare_store`` sends ``PREPARE_STORE`` (returns empty slots
      for inline mode); ``commit_store`` serialises chunks and sends
      ``COMMIT_STORE``.
    - **Retrieve**: ``prepare_retrieve`` sends ``PREPARE_RETRIEVE`` and
      deserialises the returned bytes; ``commit_retrieve`` sends
      ``COMMIT_RETRIEVE`` (no-op for inline transfer).
    """

    def __init__(
        self,
        metadata: EngineDrivenContextMetadata,
        req_client: RequestClient,
        mq_timeout: float,
    ) -> None:
        super().__init__(metadata, req_client, mq_timeout)

    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Send PREPARE_STORE RPC and return no pre-allocated buffers."""
        future = self.req_client.prepare_store(key, instance_id)
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return None

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        chunks: "list[torch.Tensor] | list[tuple[list[int], list[torch.Tensor]]]",
    ) -> bool:
        """Serialize chunks and send via COMMIT_STORE.

        Single-group callers pass a flat chunk list, sent as a tensor
        envelope. Multi-group callers pass, per group, ``(chunk_indices,
        tensors)`` -- the chunks that group stores, which omit all-null
        chunks -- sent as a grouped envelope that the server validates
        against each group's registered layout.

        Returns:
            ``True`` on success, ``False`` on failure or timeout.
        """
        if chunks and isinstance(chunks[0], tuple):
            serialised = encode_grouped_tensors(
                cast("list[tuple[list[int], list[torch.Tensor]]]", chunks)
            )
        else:
            serialised = encode_tensors(cast("list[torch.Tensor]", chunks))
        future = self.req_client.commit_store(key, instance_id, serialised)
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int, expected_count: int
    ) -> list[torch.Tensor] | None:
        """Send PREPARE_RETRIEVE and deserialize the response data.

        Args:
            key: Cache key to retrieve.
            instance_id: Worker instance identifier.
            expected_count: Number of LMCache chunks derived from the worker's
                block IDs and blocks-per-chunk geometry.

        Returns:
            Chunks on hit, or None on miss/timeout.
        """
        future = self.req_client.prepare_retrieve(key, instance_id)
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success or not response.data:
            return None
        return decode_tensors(
            response.data,
            expected_shape=self.metadata.layout_desc.shapes[0],
            expected_dtype=self.metadata.layout_desc.dtypes[0],
            expected_count=expected_count,
        )

    def prepare_retrieve_multigroup(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        expected: list[tuple[torch.Size, torch.dtype, int]],
    ) -> list[tuple[list[int], list[torch.Tensor]]] | None:
        """Send PREPARE_RETRIEVE and decode a group-major payload.

        The server responds with each group's in-window chunks and their
        chunk positions (all-or-nothing), or a miss. Named apart from the SHM
        ``prepare_retrieve_grouped`` because the result is the chunk data
        itself, not slot views.

        Args:
            key: Cache key to retrieve.
            instance_id: Worker instance identifier.
            expected: One ``(shape, dtype, num_chunks)`` per group, from the
                worker's registered group layouts and block geometry.

        Returns:
            Per group ``(chunk_indices, tensors)`` on hit, or None on
            miss/timeout.
        """
        future = self.req_client.prepare_retrieve(key, instance_id)
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success or not response.data:
            return None
        return decode_grouped_tensors(response.data, expected=expected)

    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Send COMMIT_RETRIEVE (no-op for inline path)."""
        future = self.req_client.commit_retrieve(key, instance_id)
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return True

    def close(self) -> None:
        """No-op: the inline path holds no persistent resources."""
