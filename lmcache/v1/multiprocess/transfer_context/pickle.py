# SPDX-License-Identifier: Apache-2.0
"""Pickle-based EngineDrivenContext implementation for multiprocess mode."""

# Standard
import pickle

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
)
from lmcache.v1.multiprocess.transport.base import RequestClient


class EngineDrivenContextPickle(EngineDrivenContext):
    """Pickle-based implementation of :class:`EngineDrivenContext`.

    Transport mechanism:
    - **Store**: ``prepare_store`` sends ``PREPARE_STORE`` (returns empty slots
      for pickle mode); ``commit_store`` serialises chunks and sends
      ``COMMIT_STORE``.
    - **Retrieve**: ``prepare_retrieve`` sends ``PREPARE_RETRIEVE`` and
      deserialises the returned bytes; ``commit_retrieve`` sends
      ``COMMIT_RETRIEVE`` (no-op for pickle).
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
        """Send PREPARE_STORE RPC. For pickle, returns no pre-allocated buffers."""
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
        chunks: "list[torch.Tensor] | list[list[torch.Tensor]]",
    ) -> bool:
        """Serialize chunks and send via COMMIT_STORE.

        Single-group callers pass a flat chunk list; multi-group callers pass
        a group-major ``chunks[group][chunk]`` list (the server side picks the
        shape from its registered group count).

        Returns:
            ``True`` on success, ``False`` on failure or timeout.
        """
        serialised = pickle.dumps(chunks)
        future = self.req_client.commit_store(key, instance_id, serialised)
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[torch.Tensor] | None:
        """Send PREPARE_RETRIEVE and deserialize the response data.

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
        chunks: list[torch.Tensor] = pickle.loads(response.data)
        return chunks

    def prepare_retrieve_multigroup(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[list[torch.Tensor]] | None:
        """Send PREPARE_RETRIEVE and deserialize a group-major payload.

        The server responds with ``chunks[group][chunk]`` covering every chunk
        of every group (all-or-nothing), or a miss. Named apart from the SHM
        ``prepare_retrieve_grouped`` because the result is the chunk data
        itself, not slot views.

        Returns:
            Group-major chunk lists on hit, or None on miss/timeout.
        """
        future = self.req_client.prepare_retrieve(key, instance_id)
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success or not response.data:
            return None
        group_chunks: list[list[torch.Tensor]] = pickle.loads(response.data)
        return group_chunks

    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Send COMMIT_RETRIEVE (no-op for pickle path)."""
        future = self.req_client.commit_retrieve(key, instance_id)
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return True

    def close(self) -> None:
        """No-op: the pickle path holds no persistent resources."""
