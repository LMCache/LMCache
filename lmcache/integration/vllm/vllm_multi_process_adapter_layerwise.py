# SPDX-License-Identifier: Apache-2.0
"""Worker adapter for the layer-wise KV retrieve path.

Subclasses :class:`LMCacheMPWorkerAdapter` and changes exactly two things:
it builds a layer-wise transfer context, and it exposes
:meth:`wait_for_layer_load` so vLLM can block per layer instead of waiting
for the whole retrieve to land.
"""

# Standard
from typing import Any
import os
import re

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    _IpcEvent,
)
from lmcache.utils import _lmcache_nvtx_annotate, init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.futures_layerwise import LayerwiseDeviceMessagingFuture
from lmcache.v1.multiprocess.transfer_context.worker_transfer import TransferContext
from lmcache.v1.multiprocess.transfer_context.worker_transfer_layerwise import (
    LMCacheLayerwiseTransferContext,
)

logger = init_logger(__name__)

# Extracts the integer layer index from vLLM layer names such as
# "model.layers.5.self_attn".
_LAYER_RE = re.compile(r"model\.layers\.(\d+)")

_LAYERWISE_DEBUG = os.getenv("LMCACHE_LAYERWISE_DEBUG", "0").lower() not in (
    "0",
    "",
    "false",
    "no",
)
_dbg_seen_names: set[str] = set()


def _build_layer_wait_map(
    kv_caches: dict,
) -> tuple[dict[int, int], list[str]]:
    """Map each vLLM layer index to the registration index to wait on.

    :meth:`LMCacheLayerwiseMPWorkerAdapter.wait_for_layer_load` is handed a
    vLLM layer name, but the producer keys its per-layer events by
    *registration* index -- the position of the cache in ``kv_caches``. The
    two coincide only for models that register exactly one KV cache per
    transformer layer. Models with separate sliding-window, indexer, or
    compressor-state caches register several, so a layer must wait on all of
    them.

    Waiting on the highest registration index a layer owns is sufficient:
    the transfer enqueues its copies in registration order on one stream and
    announces one event per index, so index ``n`` landing implies every index
    below ``n`` has landed too.

    Args:
        kv_caches: The KV cache dict about to be registered, keyed by vLLM
            layer name, in the order that determines registration indices.

    Returns:
        A mapping from vLLM layer index to the registration index that layer
        must wait on, and the cache names that could not be attributed to any
        layer.
    """
    wait_map: dict[int, int] = {}
    unmatched: list[str] = []
    for reg_idx, name in enumerate(kv_caches):
        match = _LAYER_RE.search(name)
        if match is None:
            unmatched.append(name)
            continue
        layer_idx = int(match.group(1))
        if reg_idx > wait_map.get(layer_idx, -1):
            wait_map[layer_idx] = reg_idx
    return wait_map, unmatched


def _log_kv_cache_pools(kv_caches: dict) -> None:
    """Log which registered caches are views into the same allocation.

    Diagnostic only, and deliberately kept here rather than in the shared
    layout banner, because only the layer-major audit needs it: the banner is
    also emitted on the per-chunk path, which does not care.  Delete this once
    the layouts in play are understood.

    Two caches whose storages report the same ``data_ptr`` are views into one
    pool, which is what makes their byte ranges comparable.  Equal
    ``storage_offset()`` does not show that on its own, since it counts
    ELEMENTS and caches of different dtypes are on different scales.
    """
    pools: dict[int, tuple[int, list[str]]] = {}
    for name, cache in kv_caches.items():
        try:
            storage = cache.untyped_storage()
            data_ptr, nbytes = storage.data_ptr(), storage.nbytes()
        except Exception:
            # Best-effort: a cache may not be a plain tensor on every backend.
            data_ptr, nbytes = -1, -1
        pools.setdefault(data_ptr, (nbytes, []))[1].append(name)
    logger.info(
        "kv_caches occupy %d distinct storage pool(s) across %d cache(s)",
        len(pools),
        len(kv_caches),
    )
    for data_ptr, (nbytes, names) in pools.items():
        logger.info(
            "  storage 0x%x (%d bytes): %d cache(s), e.g. %s",
            data_ptr,
            nbytes,
            len(names),
            names[:4],
        )


class LMCacheLayerwiseMPWorkerAdapter(LMCacheMPWorkerAdapter):
    """Worker adapter that loads KV cache one layer batch at a time."""

    # Set once vLLM is observed calling wait_for_layer_load() with a layer
    # name this adapter can resolve. Until then every retrieve is drained
    # before it is handed back, so KV can never be read while still in
    # flight. The flag only ever goes False -> True.
    _gate_verified: bool = False
    _drain_warned: bool = False

    # vLLM layer index -> registration index to wait on, built at
    # registration time by _build_layer_wait_map(). None until
    # _create_transfer_context() runs, in which case wait_for_layer_load()
    # falls back to treating the two index spaces as identical.
    _layer_wait_idx: dict[int, int] | None = None

    def _create_transfer_context(self, kv_caches: dict) -> TransferContext:
        """Build a layer-wise transfer context.

        Args:
            kv_caches: The KV cache dict about to be registered.

        Returns:
            An unregistered :class:`LMCacheLayerwiseTransferContext`.
        """
        if _LAYERWISE_DEBUG:
            # Same dict, and so the same order, that
            # create_engine_group_infos_from_vllm enumerates to assign the
            # registration indices the producer keys its events by.
            logger.info(
                "kv_caches registration order (%d entries): %s",
                len(kv_caches),
                list(kv_caches.keys()),
            )
            _log_kv_cache_pools(kv_caches)

        wait_map, unmatched = _build_layer_wait_map(kv_caches)
        if unmatched:
            raise RuntimeError(
                "Layer-wise KV loading is not supported for this model: "
                f"{len(unmatched)} of {len(kv_caches)} registered KV cache "
                f"name(s) do not match {_LAYER_RE.pattern!r}, e.g. "
                f"{unmatched[:3]}. wait_for_layer_load() is handed a vLLM "
                "layer name, so a cache that cannot be attributed to a layer "
                "can never be ordered against the compute that reads it. "
                'Re-run on the per-chunk path: "kv_connector": '
                '"LMCacheMPConnector", "kv_connector_module_path": '
                '"lmcache.integration.vllm.lmcache_mp_connector", and start '
                "the MP server with --layerwise-batch 0."
            )
        self._layer_wait_idx = wait_map
        if wait_map and any(reg != layer for layer, reg in wait_map.items()):
            # Report how many layers are held back rather than how early the
            # earliest one starts. The cheapest layer is not representative:
            # under type-major registration one layer can own a cache at a low
            # index and still be the only one that starts early, while every
            # other layer waits near the end.
            n_caches = len(kv_caches)
            threshold = 0.75
            blocked = sum(
                1 for reg in wait_map.values() if (reg + 1) / n_caches >= threshold
            )
            logger.warning(
                "This model registers %d KV cache(s) for %d layer(s), so "
                "layers do not map one-to-one onto registration indices. "
                "Loads are still ordered correctly, but overlap is limited: "
                "%d of %d layer(s) cannot start until %.0f%% of the transfer "
                "has landed. Expect little or no speed-up over "
                "--layerwise-batch 0 until the transfer emits layers in model "
                "order.",
                n_caches,
                len(wait_map),
                blocked,
                len(wait_map),
                100.0 * threshold,
            )
        return LMCacheLayerwiseTransferContext(
            instance_id=self.instance_id,
            req_client=self.req_client,
        )

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

    def submit_retrieve_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: _IpcEvent | None,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        """Submit a retrieve, draining it until the per-layer gate is proven.

        A layer-wise retrieve only records one device event per layer; it
        never blocks the compute stream by itself. Ordering is therefore
        supplied entirely by vLLM calling :meth:`wait_for_layer_load` before
        each layer is read. A model whose attention does not dispatch through
        vLLM's instrumented attention op never issues that call, and would
        silently read KV that is still in flight.

        Registration-time checks cannot see this: the call comes from the
        model's forward pass, so the only sound test is to watch for it. Until
        it is observed, each retrieve is drained before this returns, which
        restores the per-chunk ordering guarantee at the cost of the overlap.
        The first genuine gate call sets ``_gate_verified`` for good.

        Args:
            request_id: The ID of the request.
            op: The LoadStoreOp describing the retrieve operation.
            event: Device event recorded after the current inference step, or
                ``None`` for synchronous engine-driven transfer.
            cache_salt: Per-user isolation salt.
            request_configs: Optional LMCache request configs for the IPC key.
        """
        super().submit_retrieve_request(
            request_id, op, event, cache_salt, request_configs
        )
        if self._gate_verified:
            return
        entry = self.retrieve_futures.get(request_id)
        if entry is None:
            # Unhealthy server: the base call dropped the retrieve and already
            # flagged the blocks for recompute, so there is nothing to drain.
            return
        future, _ = entry
        if not isinstance(future, LayerwiseDeviceMessagingFuture):
            return
        if not self._drain_warned:
            self._drain_warned = True
            logger.warning(
                "Draining layer-wise KV retrieves synchronously until vLLM "
                "calls wait_for_layer_load(). If this model's attention does "
                "not dispatch through vLLM's instrumented attention op the "
                "call never arrives, and layer-wise loading will stay at "
                "per-chunk speed instead of corrupting KV. Pass "
                "--layerwise-batch 0 to turn the feature off explicitly."
            )
        future.wait()

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
        if _LAYERWISE_DEBUG and layer_name not in _dbg_seen_names:
            if len(_dbg_seen_names) < 8:
                m = _LAYER_RE.search(layer_name)
                logger.info(
                    "layerwise-adapter: wait_for_layer_load(%r) -> regex %s, "
                    "ctx=%s, %d pending retrieve future(s)",
                    layer_name,
                    f"idx={m.group(1)}" if m else "NO MATCH",
                    type(self.transfer_ctx).__name__,
                    len(self.retrieve_futures),
                )
            _dbg_seen_names.add(layer_name)
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
        wait_map = self._layer_wait_idx
        if wait_map is None:
            wait_idx = layer_idx
        else:
            resolved = wait_map.get(layer_idx)
            if resolved is None:
                # This layer registered no KV cache -- an MTP or draft layer,
                # for example -- so there is nothing to order against it, and
                # the call proves nothing about the gate.
                return
            wait_idx = resolved
        if not self._gate_verified:
            self._gate_verified = True
            logger.info(
                "Layer-wise gate confirmed at %r; KV loads now overlap with "
                "compute (the start-up drain is disabled from here on).",
                layer_name,
            )

        ids = (
            request_ids
            if request_ids is not None
            else list(self.retrieve_futures.keys())
        )
        for req_id in ids:
            entry = self.retrieve_futures.get(req_id)
            if entry is None:
                continue
            future, block_ids = entry
            if not isinstance(future, LayerwiseDeviceMessagingFuture):
                continue
            try:
                future.wait_for_layer(wait_idx)
            except LMCacheTimeoutError:
                # The transfer stalled without an outcome, so the raw future
                # never resolves, get_finished() skips it on query(), and the
                # blocks would never be reported.  Retire the retrieve here so
                # the step can finish and vLLM recomputes these blocks, rather
                # than letting the timeout escape into attention and take the
                # whole forward pass down.  Catching per request also keeps one
                # stalled retrieve from abandoning the wait for the others.
                # ids is materialised above, so mutating the dict is safe.
                logger.error(
                    "Layer-wise retrieve for request_id=%s timed out waiting "
                    "for layer %d; marking %d block(s) for recomputation",
                    req_id,
                    wait_idx,
                    len(block_ids),
                )
                self.error_block_ids.update(block_ids)
                self.retrieve_futures.pop(req_id, None)
                self.retrieve_events.pop(req_id, None)
