# SPDX-License-Identifier: Apache-2.0
"""Worker-side transfer context for the layer-wise KV retrieve path.

Subclasses :class:`LMCacheDrivenTransferContext` so the default per-chunk
transport stays untouched. Two things differ:

* after ``REGISTER_KV_CACHE``, a follow-up ``REGISTER_LAYERWISE_IPC_EVENT_POOL``
  request imports the server's per-layer IPC event pool;
* ``submit_retrieve`` issues ``RETRIEVE_LAYERWISE`` and returns a future
  that can be awaited one layer at a time.
"""

# Standard
from collections.abc import Sequence
from typing import Any

# Third Party
import torch

# First Party
from lmcache.utils import EngineType, init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.futures_layerwise import LayerwiseDeviceMessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.mq_streaming import submit_streaming_request
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    IPCEvent,
    LMCacheDrivenTransferContext,
    SendRequest,
)
from lmcache.v1.platform.base.event_pool import EventPool

logger = init_logger(__name__)


class LMCacheLayerwiseTransferContext(LMCacheDrivenTransferContext):
    """LMCache-driven transfer context that retrieves layer by layer."""

    def __init__(self) -> None:
        super().__init__()
        self._event_pool: EventPool | None = None
        self._layerwise_batch: int = 0

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register as usual, then import the server's per-layer event pool.

        See :meth:`LMCacheDrivenTransferContext.register` for the argument
        semantics. The extra round trip keeps ``REGISTER_KV_CACHE`` and its
        ``None`` response identical to the non-layer-wise deployment.
        """
        super().register(
            instance_id,
            kv_caches,
            model_name,
            world_size,
            _blocks_in_chunk,
            mq_client,
            mq_timeout,
            send_request,
            layout_hints=layout_hints,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
        )
        future = send_request(
            mq_client,
            RequestType.REGISTER_LAYERWISE_IPC_EVENT_POOL,
            [instance_id],
        )
        try:
            layerwise_batch, pool_handles = future.result(timeout=mq_timeout)
        except TimeoutError:
            # A server started without ``--layerwise-batch`` never registers
            # a REGISTER_LAYERWISE_IPC_EVENT_POOL handler, and an unhandled
            # request type is only logged server-side -- no reply is ever
            # sent, so the wait above burns the full timeout. Name that cause
            # here, otherwise the caller reports the generic "is the server
            # running?" error and sends the operator looking at a server that
            # is perfectly healthy.
            raise RuntimeError(
                "The LMCache MP server did not answer "
                "REGISTER_LAYERWISE_IPC_EVENT_POOL "
                f"within {mq_timeout}s. The most likely cause is a mode "
                "mismatch: a server node serves either per-chunk or layer-wise "
                "requests, and a per-chunk server silently drops this request. "
                "Start the server with --layerwise-batch > 0, or load the "
                "per-chunk connector (LMCacheMPConnector)."
            ) from None
        if (
            layerwise_batch <= 0
            or not pool_handles
            or self._event_backend is None
            or self._device is None
        ):
            # Backstop for a layer-wise server that answered without a usable
            # pool. Unlike the timeout above this should not be reachable, so
            # report the negotiated values to make the bug diagnosable.
            raise RuntimeError(
                "Layerwise connector is loaded but the MP server did not "
                f"negotiate layer-wise mode (batch={layerwise_batch}, "
                f"handles={len(pool_handles or ())}, "
                f"backend={self._event_backend is not None}, "
                f"device={self._device is not None})."
            )
        self._layerwise_batch = layerwise_batch
        self._event_pool = EventPool.import_pool(
            self._event_backend, self._device, pool_handles
        )
        logger.info(
            "Layerwise transfer context registered (batch=%d, pool_size=%d)",
            self._layerwise_batch,
            self._event_pool.size,
        )

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a layer-wise retrieve ordered by ``event``.

        ``register`` fails unless the server negotiated layer-wise mode, so
        this always issues ``RETRIEVE_LAYERWISE``. See
        :meth:`LMCacheDrivenTransferContext.submit_retrieve` for the argument
        semantics.

        Returns:
            A future whose per-layer completions can be awaited individually.

        Raises:
            RuntimeError: If the context is not registered, or ``event`` is
                ``None``; the layer-wise path always needs one to order the
                engine-side writes.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "Layerwise transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        if event is None:
            raise RuntimeError("Layer-wise transfer requires an IPC event.")
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        # The server answers a layer-wise retrieve with one message per
        # layer batch, so the future has to exist, and know how to re-arm
        # itself, before the request reaches the polling loop.  That is what
        # ``submit_streaming_request`` arranges; the per-chunk
        # ``submit_request`` is left untouched by this path.
        layerwise_future: LayerwiseDeviceMessagingFuture = (
            LayerwiseDeviceMessagingFuture(
                device=self._device,
                event_pool=self._event_pool,
            )
        )
        submit_streaming_request(
            self._mq_client,
            RequestType.RETRIEVE_LAYERWISE,
            [key, instance_id, block_ids, event_ipc_handle, skip_first_n_tokens],
            layerwise_future.raw_future_,
        )
        return layerwise_future

    def close(self) -> None:
        """Drop the imported event pool, then run the base teardown."""
        self._event_pool = None
        super().close()
