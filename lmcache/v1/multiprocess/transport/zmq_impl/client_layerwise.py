# SPDX-License-Identifier: Apache-2.0
"""ZMQ client that also serves the layer-wise retrieve path."""

# Standard
from typing import Any

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.base_layerwise import (
    LayerwiseRequestClient,
)
from lmcache.v1.multiprocess.transport.zmq_impl.client import (
    ZmqMultiprocessClient,
)
from lmcache.v1.multiprocess.transport.zmq_impl.mq_streaming import (
    StreamingMessageQueueClient,
)


class ZmqLayerwiseClient(ZmqMultiprocessClient, LayerwiseRequestClient):
    """:class:`ZmqMultiprocessClient` plus the two layer-wise operations.

    Every shared operation is inherited unchanged. The layer-wise ones are
    not in the shared RPC contract, so they are written out here and submitted
    through :class:`StreamingMessageQueueClient`, which knows their specs.

    Args:
        message_queue_client: Streaming ZMQ message queue client to wrap.
    """

    def __init__(self, message_queue_client: StreamingMessageQueueClient) -> None:
        super().__init__(message_queue_client)
        self._streaming_client = message_queue_client

    def register_layerwise_ipc_event_pool(
        self, instance_id: int
    ) -> MessagingFuture[Any]:
        """Import the server's per-layer IPC event pool."""
        return self._streaming_client.submit_layerwise_request(
            "register_layerwise_ipc_event_pool", [instance_id]
        )

    def retrieve_layerwise(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: Any,
        skip_first_n_tokens: int,
        future: Any,
    ) -> MessagingFuture[Any]:
        """Retrieve one chunk layer by layer.

        The server answers with one frame per layer batch. Unlike the other
        methods the future is supplied by the caller, because it carries the
        per-layer state and has to be bound to the pending-request table
        before the first frame can arrive.
        """
        return self._streaming_client.submit_streaming_request(
            "retrieve_layerwise",
            [key, instance_id, block_ids, event_ipc_handle, skip_first_n_tokens],
            future,
        )
