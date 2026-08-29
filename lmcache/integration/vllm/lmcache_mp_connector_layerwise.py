# SPDX-License-Identifier: Apache-2.0
"""vLLM connector for the layer-wise KV retrieve path.

Select it at runtime with::

    --kv-transfer-config '{
      "kv_connector": "LMCacheLayerwiseMPConnector",
      "kv_connector_module_path":
        "lmcache.integration.vllm.lmcache_mp_connector_layerwise",
      "kv_role": "kv_both"
    }'

Everything else -- server handshake, store path, lookup -- is inherited
unchanged from :class:`LMCacheMPConnector`.
"""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.vllm_multi_process_adapter_layerwise import (
    LMCacheLayerwiseMPWorkerAdapter,
)
from lmcache.utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request


class LMCacheLayerwiseMPConnector(LMCacheMPConnector):
    """LMCache MP connector that loads KV cache layer by layer."""

    _worker_adapter_cls = LMCacheLayerwiseMPWorkerAdapter

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Report the external hit as a synchronous load.

        vLLM parks a request in ``WAITING_FOR_REMOTE_KVS`` whenever a
        connector reports an asynchronous load, and only runs the forward
        pass after the whole transfer has been reported finished. That gate
        makes per-layer waiting unreachable: ``retrieve_futures`` is already
        drained by the time attention runs, so every
        :meth:`wait_for_layer_load` call finds nothing pending. Reporting a
        synchronous load keeps the request in the same step, so
        ``start_load_kv`` issues the transfer and the per-layer waits gate
        each attention layer against it.

        Args:
            request: The request vLLM is scheduling.
            num_computed_tokens: Tokens already computed for this request.

        Returns:
            The number of tokens to load and ``False`` for the async flag.
            A ``None`` token count is passed through untouched: it means the
            lookup has not resolved yet, which is unrelated to how the KV is
            loaded once it does.
        """
        num_tokens, load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if num_tokens is None:
            return None, load_async
        return num_tokens, False

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until KV for ``layer_name`` has landed on the device.

        Args:
            layer_name: The name of the layer vLLM is about to run.
        """
        self.worker_adapter.wait_for_layer_load(layer_name)
