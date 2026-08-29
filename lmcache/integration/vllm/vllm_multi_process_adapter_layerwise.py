# SPDX-License-Identifier: Apache-2.0
"""Worker adapter for the layer-wise KV retrieve path.

Subclasses :class:`LMCacheMPWorkerAdapter` and changes exactly two things:
it builds a layer-wise transfer context, and it exposes
:meth:`wait_for_layer_load` so vLLM can block per layer instead of waiting
for the whole retrieve to land.
"""

# Standard
import re

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import LMCacheMPWorkerAdapter
from lmcache.utils import _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.layerwise_futures import LayerwiseDeviceMessagingFuture
from lmcache.v1.multiprocess.transfer_context.worker_layerwise_transfer import (
    LMCacheLayerwiseTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import TransferContext

logger = init_logger(__name__)

# Extracts the integer layer index from vLLM layer names such as
# "model.layers.5.self_attn".
_LAYER_RE = re.compile(r"model\.layers\.(\d+)")


class LMCacheLayerwiseMPWorkerAdapter(LMCacheMPWorkerAdapter):
    """Worker adapter that loads KV cache one layer batch at a time."""

    def _create_transfer_context(self, kv_caches: dict) -> TransferContext:
        """Build a layer-wise transfer context.

        Args:
            kv_caches: The KV cache dict about to be registered.

        Returns:
            An unregistered :class:`LMCacheLayerwiseTransferContext`.
        """
        return LMCacheLayerwiseTransferContext()

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report finished stores only, never finished retrieves.

        The layer-wise connector reports ``load_kv_async=False``, so vLLM
        keeps the request RUNNING and never parks it in
        ``WAITING_FOR_REMOTE_KVS``. Reporting the retrieve in
        ``finished_recving`` then trips the scheduler's status assertion,
        because the request is neither waiting for remote KV nor finished.
        The per-layer waits already guarantee the KV has landed before it is
        read, so the completion report carries no information vLLM needs.

        Args:
            finished_req_ids_from_engine: Request ids vLLM reports finished.

        Returns:
            The finished store ids, and ``None`` for the retrieve set. The
            base call still drains and reaps ``retrieve_futures``, so this
            only suppresses the report, not the bookkeeping.
        """
        finished_stores, _ = super().get_finished(finished_req_ids_from_engine)
        return finished_stores, None

    def get_finished_with_lazy_offload(
        self,
    ) -> tuple[set[str] | None, set[str] | None]:
        """Lazy-offload variant of :meth:`get_finished`.

        Returns:
            The finished store ids, and ``None`` for the retrieve set, for
            the same reason as :meth:`get_finished`.
        """
        finished_stores, _ = super().get_finished_with_lazy_offload()
        return finished_stores, None

    def wait_for_layer_load(
        self, layer_name: str, request_ids: list[str] | None = None
    ) -> None:
        """Block until KV for one layer has landed for the active retrieves.

        Args:
            layer_name: vLLM layer name, e.g. ``"model.layers.5.self_attn"``.
            request_ids: If given, only wait for these request ids;
                otherwise wait for every pending layer-wise retrieve.

        Raises:
            RuntimeError: If this adapter is bound to a per-chunk transfer
                context, which cannot order KV loads per layer.
        """
        transfer_ctx = self.transfer_ctx
        if transfer_ctx is None:
            # Called before register_kv_caches or after shutdown, so there is
            # nothing pending to wait for.
            return
        if not isinstance(transfer_ctx, LMCacheLayerwiseTransferContext):
            # register_kv_caches() raises unless the server negotiated
            # layer-wise mode, so reaching here means the adapter was paired
            # with the per-chunk transfer context.
            raise RuntimeError(
                "LMCacheLayerwiseMPWorkerAdapter is bound to a "
                f"{type(transfer_ctx).__name__}, which cannot order KV loads "
                "per layer. A server node serves one mode only: use "
                "LMCacheMPConnector against a per-chunk server."
            )

        match = _LAYER_RE.search(layer_name)
        if match is None:
            return
        layer_idx = int(match.group(1))

        ids = (
            request_ids
            if request_ids is not None
            else list(self.retrieve_futures.keys())
        )
        for req_id in ids:
            entry = self.retrieve_futures.get(req_id)
            if entry is None:
                continue
            future, _ = entry
            if isinstance(future, LayerwiseDeviceMessagingFuture):
                future.wait_for_layer(layer_idx)
