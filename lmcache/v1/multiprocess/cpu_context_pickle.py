# SPDX-License-Identifier: Apache-2.0
"""Pickle-based CPUContext implementation for multiprocess mode."""

# Standard
from typing import Any
import pickle

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.cpu_context import CPUContext, CPUContextMetadata
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class


class CPUContextPickle(CPUContext):
    """Pickle-based implementation of :class:`CPUContext`.

    Transport mechanism:
    - **Store**: ``prepare_store`` serialises chunks with ``pickle.dumps``; \
``commit_store`` sends a ``STORE_CPU_CHUNKS`` message and waits for the \
server acknowledgment.
    - **Retrieve**: ``prepare_retrieve`` sends a ``RETRIEVE_CPU_CHUNKS`` \
message, waits for the response, and deserialises the returned bytes with \
``pickle.loads``; ``commit_retrieve`` is a no-op (no locks to release).

    Args:
        metadata: Layout metadata for the CPU context.
        mq_client: Message-queue client for server communication.
        mq_timeout: Timeout in seconds for blocking MQ requests.
    """

    def __init__(
        self,
        metadata: CPUContextMetadata,
        mq_client: Any,
        mq_timeout: float,
    ) -> None:
        super().__init__(metadata, mq_client, mq_timeout)

    def prepare_store(
        self, key: Any, instance_id: int, chunks: list[torch.Tensor]
    ) -> Any:
        """Serialise *chunks* with ``pickle.dumps``.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            chunks: CPU chunk tensors to serialise.

        Returns:
            Opaque handle ``(key, instance_id, serialised_bytes)`` to be
            passed to :meth:`commit_store`.
        """
        serialised = pickle.dumps(chunks)
        return (key, instance_id, serialised)

    def commit_store(self, handle: Any) -> bool:
        """Send pickled chunks to the server via ``STORE_CPU_CHUNKS``.

        Blocks until the server acknowledges the write.

        Args:
            handle: The ``(key, instance_id, bytes)`` tuple returned by
                :meth:`prepare_store`.

        Returns:
            ``True`` on success, ``False`` on failure or timeout.
        """
        key, instance_id, serialised = handle
        future = self.mq_client.submit_request(
            RequestType.STORE_CPU_CHUNKS,
            [key, instance_id, serialised],
            get_response_class(RequestType.STORE_CPU_CHUNKS),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve(
        self, key: Any, instance_id: int
    ) -> tuple[Any, list[torch.Tensor] | None]:
        """Fetch serialised chunks from the server via ``RETRIEVE_CPU_CHUNKS``.

        Blocks until the server responds with the cached data (or reports a
        miss).

        Args:
            key: Cache key for the token range to retrieve.
            instance_id: Worker instance identifier.

        Returns:
            ``(None, chunks)`` on cache hit where *chunks* is the
            deserialised list of CPU tensors, or ``(None, None)`` on cache
            miss or timeout.  The handle is ``None`` because the pickle path
            has no resources to release in :meth:`commit_retrieve`.
        """
        future = self.mq_client.submit_request(
            RequestType.RETRIEVE_CPU_CHUNKS,
            [key, instance_id],
            get_response_class(RequestType.RETRIEVE_CPU_CHUNKS),
        )
        try:
            success, cpu_data_bytes = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return (None, None)
        if not success or not cpu_data_bytes:
            return (None, None)
        chunks: list[torch.Tensor] = pickle.loads(cpu_data_bytes)
        return (None, chunks)

    def commit_retrieve(self, handle: Any) -> None:
        """No-op: the pickle path holds no server-side locks.

        Args:
            handle: Ignored.
        """

    def close(self) -> None:
        """No-op: the pickle path holds no persistent resources."""
