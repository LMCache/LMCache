# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM KV Cache Connector adapter for LMCache.

Implements the two classes required by TRT-LLM's ``kv_connector_config``::

    kv_connector_config:
      connector_module: lmcache.integration.tensorrt_llm.tensorrt_adapter
      connector_scheduler_class: LMCacheKvConnectorScheduler
      connector_worker_class: LMCacheKvConnectorWorker

``LMCacheKvConnectorScheduler`` runs on the scheduler (leader) process and
calls ``engine.lookup()`` to find cached prefixes. ``LMCacheKvConnectorWorker``
runs on each worker and calls ``engine.retrieve()`` / ``engine.store()`` to
move KV data between TRT-LLM's GPU block pool and LMCache's CPU backend.

All imports of ``tensorrt_llm`` types are confined to this module so that the
rest of LMCache does not acquire TRT-LLM as a hard dependency.
"""

# Standard
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Third Party
import torch
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Engine singleton (module-level, like SGLang's init_lmcache_engine pattern)
# ---------------------------------------------------------------------------

_ENGINE_SINGLETON: Optional[LMCacheEngine] = None
_GPU_CONNECTOR_SINGLETON: Optional[TRTLLMGPUConnector] = None

_LMCACHE_INSTANCE_ID = "trtllm_lmcache_connector"
_DEFAULT_LMCACHE_CPU_SIZE_GB = 20.0


def _make_lmcache_config(block_size: int) -> LMCacheEngineConfig:
    """Build an LMCacheEngineConfig for TRT-LLM.

    Sets ``chunk_size = block_size`` so each LMCache chunk maps exactly to one
    TRT-LLM KV block. CPU memory is used as the cache backend.  The pool size
    is read from the ``LMCACHE_CPU_SIZE_GB`` environment variable.

    Args:
        block_size: TRT-LLM ``tokens_per_block`` (becomes LMCache chunk size).

    Returns:
        Configured :class:`~lmcache.v1.config.LMCacheEngineConfig`.
    """
    cpu_size_gb = float(
        os.environ.get("LMCACHE_CPU_SIZE_GB", str(_DEFAULT_LMCACHE_CPU_SIZE_GB))
    )
    return LMCacheEngineConfig(
        chunk_size=block_size,
        local_cpu=True,
        max_local_cpu_size=cpu_size_gb,
        local_disk=None,
        enable_controller=False,
        pre_caching_hash_algorithm="builtin",
    )


def _get_or_create_engine(
    kv_cache_tensor: torch.Tensor,
    block_size: int,
    model_name: str,
) -> Tuple[LMCacheEngine, TRTLLMGPUConnector]:
    """Return the module-level LMCache engine singleton, creating it if needed.

    Follows the same pattern as SGLang's ``init_lmcache_engine``: check for an
    existing instance first; create and fully initialise on first call. On
    subsequent calls (e.g. after TRT-LLM re-allocates its KV pool) only the
    connector's tensor reference is refreshed so cached data is preserved.

    Args:
        kv_cache_tensor: TRT-LLM KV pool tensor, shape ``[num_blocks, ...]``.
        block_size: Tokens per KV block (TRT-LLM's ``tokens_per_block``).
        model_name: Model identifier used in LMCache content-addressed keys.

    Returns:
        ``(engine, gpu_connector)`` tuple.
    """
    global _ENGINE_SINGLETON, _GPU_CONNECTOR_SINGLETON

    if _ENGINE_SINGLETON is not None:
        assert _GPU_CONNECTOR_SINGLETON is not None
        _GPU_CONNECTOR_SINGLETON.initialize(kv_cache_tensor, block_size)
        logger.info("LMCache TRT-LLM: reusing existing engine singleton")
        return _ENGINE_SINGLETON, _GPU_CONNECTOR_SINGLETON

    block_numel = int(kv_cache_tensor[0].numel())
    hidden_dim = block_numel // (2 * block_size)

    gpu_connector = TRTLLMGPUConnector()
    gpu_connector.initialize(kv_cache_tensor, block_size)

    # kv_shape: (num_layers=1, 2, chunk_size=block_size, num_kv_heads=1, head_size)
    # get_shapes(block_size) → [torch.Size([2, 1, block_size, hidden_dim])]
    # total elements = 2 * block_size * hidden_dim == block_numel  ✓
    metadata = LMCacheMetadata(
        model_name=model_name,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_cache_tensor.dtype,
        kv_shape=(1, 2, block_size, 1, hidden_dim),
        use_mla=False,
        chunk_size=block_size,
    )

    config = _make_lmcache_config(block_size)
    engine = LMCacheEngineBuilder.get_or_create(
        _LMCACHE_INSTANCE_ID,
        config,
        metadata,
        gpu_connector,
        broadcast_fn=lambda t, _rank: t,
        broadcast_object_fn=lambda obj, _rank: obj,
    )
    engine.post_init()

    _ENGINE_SINGLETON = engine
    _GPU_CONNECTOR_SINGLETON = gpu_connector
    logger.info(
        "LMCache TRT-LLM: created engine (chunk_size=%d, hidden_dim=%d, "
        "dtype=%s, cpu_pool=%.1f GiB)",
        block_size,
        hidden_dim,
        kv_cache_tensor.dtype,
        config.max_local_cpu_size,
    )
    return engine, gpu_connector


def destroy_engine() -> None:
    """Destroy the engine singleton and release all cached data.

    Safe to call even if the engine has not been created.
    """
    global _ENGINE_SINGLETON, _GPU_CONNECTOR_SINGLETON
    if _ENGINE_SINGLETON is not None:
        LMCacheEngineBuilder.destroy(_LMCACHE_INSTANCE_ID)
        _ENGINE_SINGLETON = None
        _GPU_CONNECTOR_SINGLETON = None
        logger.info("LMCache TRT-LLM: engine destroyed")


# ---------------------------------------------------------------------------
# Per-request metadata passed from Scheduler → Worker each iteration
# ---------------------------------------------------------------------------


@dataclass
class _BlockSpec:
    """Describes the LMCache load or save work for a single request.

    Attributes:
        tokens: Full token sequence for this request.
        block_ids: All physical GPU block IDs allocated for this request.
    """

    tokens: List[int]
    block_ids: List[int]


@dataclass
class LMCacheConnectorMetadata:
    """Connector metadata exchanged between scheduler and worker each step.

    Attributes:
        loads: Mapping from request ID to :class:`_BlockSpec` for blocks that
            should be loaded from LMCache into the GPU KV pool.
        saves: Mapping from request ID to :class:`_BlockSpec` for blocks that
            should be saved from the GPU KV pool into LMCache.
    """

    loads: dict = field(default_factory=dict)
    saves: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scheduler (leader process)
# ---------------------------------------------------------------------------


class LMCacheKvConnectorScheduler(KvCacheConnectorScheduler):
    """Scheduler-side connector that consults LMCache for KV prefix lookups.

    On each new request:

    1. :meth:`get_num_new_matched_tokens` calls ``engine.lookup(tokens)`` to
       find how many prefix tokens LMCache has cached beyond what TRT-LLM's own
       GPU radix cache already matched.
    2. :meth:`build_connector_meta` builds per-request load and save specs that
       are passed to the worker each iteration.
    """

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size: int = self._llm_args.kv_cache_config.tokens_per_block
        # Maps request_id → (all_tokens, num_new_matched) between
        # get_num_new_matched_tokens and build_connector_meta calls.
        self._pending: dict = {}

    def get_num_new_matched_tokens(
        self,
        request: LlmRequest,
        num_computed_tokens: int,
    ) -> Tuple[int, bool]:
        """Return how many additional prefix tokens LMCache has cached.

        Skips the ``engine.lookup()`` call (short-circuit) when TRT-LLM's GPU
        prefix cache already matched every full block — LMCache cannot
        contribute anything beyond that. This mirrors the vLLM connector's
        fast-path and avoids unnecessary Python-side hash lookups.

        Args:
            request: The incoming TRT-LLM request.
            num_computed_tokens: Tokens already matched by TRT-LLM's own cache.

        Returns:
            ``(new_matched_tokens, False)``.  The boolean indicates whether
            the result is speculative; we always return ``False``.
        """
        t0 = time.perf_counter()

        if _ENGINE_SINGLETON is None:
            self._pending[request.request_id] = ([], 0)
            return 0, False

        if num_computed_tokens % self._block_size != 0:
            self._pending[request.request_id] = ([], 0)
            return 0, False

        all_tokens = list(request.get_tokens(0))

        # Short-circuit: if TRT-LLM already matched every full block-aligned
        # token, there is nothing more LMCache can contribute.
        max_block_aligned = (len(all_tokens) // self._block_size) * self._block_size
        if num_computed_tokens >= max_block_aligned:
            self._pending[request.request_id] = (all_tokens, 0)
            logger.debug(
                "LMCache TRT-LLM scheduler: req %d short-circuit "
                "(TRT matched %d of %d block-aligned tokens) %.3fms",
                request.request_id,
                num_computed_tokens,
                max_block_aligned,
                (time.perf_counter() - t0) * 1000,
            )
            return 0, False

        t1 = time.perf_counter()
        cached_tokens = _ENGINE_SINGLETON.lookup(tokens=all_tokens)
        t2 = time.perf_counter()

        new_matched = max(0, cached_tokens - num_computed_tokens)
        new_matched = (new_matched // self._block_size) * self._block_size

        self._pending[request.request_id] = (all_tokens, new_matched)

        logger.debug(
            "LMCache TRT-LLM scheduler: req %d lookup=%.3fms total=%.3fms "
            "trt_matched=%d lmcache_cached=%d new_matched=%d",
            request.request_id,
            (t2 - t1) * 1000,
            (time.perf_counter() - t0) * 1000,
            num_computed_tokens,
            cached_tokens,
            new_matched,
        )
        return new_matched, False

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> LMCacheConnectorMetadata:
        """Build per-request load/save specs from the pending state.

        Called once per scheduling step after :meth:`get_num_new_matched_tokens`
        has been called for all new requests.

        Args:
            scheduler_output: TRT-LLM scheduler output for this step.

        Returns:
            :class:`LMCacheConnectorMetadata` with load and save dicts.
        """
        meta = LMCacheConnectorMetadata()

        for req in scheduler_output.new_requests:
            if req.request_id not in self._pending:
                continue

            all_tokens, num_matched = self._pending[req.request_id]
            block_ids: List[int] = list(req.new_block_ids)
            num_computed_blocks = req.computed_position // self._block_size

            if num_matched > 0:
                meta.loads[req.request_id] = _BlockSpec(
                    tokens=all_tokens,
                    block_ids=block_ids,
                )

            # Only schedule saves for full newly-computed blocks that are not
            # already covered by TRT-LLM's GPU cache or by LMCache's retrieve.
            # LMCache requires chunk-aligned token sequences for correct hashing.
            save_start = max(num_computed_blocks, num_matched // self._block_size)
            num_full_new_blocks = len(req.new_tokens) // self._block_size
            if (
                save_start < len(block_ids)
                and num_full_new_blocks > 0
                and save_start < num_computed_blocks + num_full_new_blocks
            ):
                meta.saves[req.request_id] = _BlockSpec(
                    tokens=all_tokens,
                    block_ids=block_ids,
                )

        self._pending.clear()
        return meta

    def request_finished(
        self,
        request: LlmRequest,
        cache_block_ids: List[int],
    ) -> bool:
        """Called when a request completes; no-op for LMCache.

        Args:
            request: The completed request.
            cache_block_ids: GPU block IDs that were used.

        Returns:
            ``False`` — LMCache manages its own eviction policy.
        """
        return False

    def update_state_after_alloc(
        self,
        request: LlmRequest,
        block_ids: List[int],
    ) -> None:
        """Called after block allocation; no-op for LMCache.

        Args:
            request: The request that was allocated.
            block_ids: Newly allocated GPU block IDs.
        """


# ---------------------------------------------------------------------------
# Worker (GPU process)
# ---------------------------------------------------------------------------


class LMCacheKvConnectorWorker(KvCacheConnectorWorker):
    """Worker-side connector that drives actual KV data movement via LMCache.

    Lifecycle::

        __init__            → allocate placeholder
        register_kv_caches  → initialize LMCache engine with KV pool shape
        start_load_kv       → engine.retrieve() for scheduled loads
        wait_for_save       → engine.store() for scheduled saves

    Stream ordering guarantees:

    * **Load**: after ``engine.retrieve()`` the caller's ``stream`` is made to
      wait on ``load_stream`` so subsequent compute sees the transferred data.
    * **Save**: ``store_stream`` is made to wait on the caller's ``stream``
      before ``engine.store()`` so the KV data is fully written before the
      host-side copy starts; ``store_stream`` is then synchronised to ensure
      the CPU buffer is populated before the backend indexes it.
    """

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size: int = self._llm_args.kv_cache_config.tokens_per_block

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor) -> None:
        """Initialise the LMCache engine with the TRT-LLM KV pool tensor.

        Called once after model loading. If the engine singleton already exists
        (e.g. a second LLM instantiation) the tensor reference is refreshed.

        Args:
            kv_cache_tensor: TRT-LLM KV pool, shape ``[num_blocks, ...]``.
        """
        model_name = str(getattr(self._llm_args, "model", "unknown_model"))
        _get_or_create_engine(
            kv_cache_tensor=kv_cache_tensor,
            block_size=self._block_size,
            model_name=model_name,
        )

    def start_load_kv(self, stream: torch.cuda.Stream) -> None:
        """Load LMCache-cached KV blocks into the GPU pool (host→device).

        Calls ``engine.retrieve()`` for each scheduled load, then makes
        ``stream`` wait on ``load_stream`` so subsequent GPU compute sees
        the transferred blocks.

        Args:
            stream: The caller's CUDA stream; will wait on ``load_stream``
                after all retrieve calls complete.
        """
        meta: Optional[LMCacheConnectorMetadata] = self._metadata  # type: ignore[assignment]
        if meta is None or not meta.loads:
            return
        if _ENGINE_SINGLETON is None:
            return

        t0 = time.perf_counter()
        for _req_id, spec in meta.loads.items():
            if not spec.tokens or not spec.block_ids:
                continue
            _ENGINE_SINGLETON.retrieve(tokens=spec.tokens, block_ids=spec.block_ids)
        t1 = time.perf_counter()

        if _GPU_CONNECTOR_SINGLETON is not None:
            stream.wait_stream(_GPU_CONNECTOR_SINGLETON.load_stream)
        t2 = time.perf_counter()

        logger.debug(
            "LMCache TRT-LLM worker: start_load_kv retrieve=%.3fms "
            "stream_wait=%.3fms num_loads=%d",
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            len(meta.loads),
        )

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        """Per-layer load barrier; no-op for the non-layerwise LMCache path.

        Args:
            layer_idx: Index of the layer being waited on.
            stream: The caller's CUDA stream.
        """

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        """Per-layer save trigger; no-op for the non-layerwise LMCache path.

        Args:
            layer_idx: Index of the layer to save.
            stream: The caller's CUDA stream.
        """

    def wait_for_save(self, stream: torch.cuda.Stream) -> None:
        """Store newly computed KV blocks into LMCache's CPU backend.

        Stream ordering: ``store_stream`` waits on ``stream`` (so all GPU KV
        writes finish before the device→host copy begins), then
        ``engine.store()`` issues the async copies, and finally
        ``store_stream.synchronize()`` ensures the CPU buffer is fully
        populated before the LMCache backend indexes the data.

        Args:
            stream: The caller's CUDA stream; ``store_stream`` will wait on it.
        """
        meta: Optional[LMCacheConnectorMetadata] = self._metadata  # type: ignore[assignment]
        if meta is None or not meta.saves:
            return
        if _ENGINE_SINGLETON is None:
            return

        t0 = time.perf_counter()
        if _GPU_CONNECTOR_SINGLETON is not None:
            _GPU_CONNECTOR_SINGLETON.store_stream.wait_stream(stream)
        t1 = time.perf_counter()

        for _req_id, spec in meta.saves.items():
            if not spec.tokens or not spec.block_ids:
                continue
            _ENGINE_SINGLETON.store(tokens=spec.tokens, block_ids=spec.block_ids)
        t2 = time.perf_counter()

        if _GPU_CONNECTOR_SINGLETON is not None:
            _GPU_CONNECTOR_SINGLETON.store_stream.synchronize()
        t3 = time.perf_counter()

        logger.debug(
            "LMCache TRT-LLM worker: wait_for_save stream_wait=%.3fms "
            "store=%.3fms sync=%.3fms num_saves=%d",
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            len(meta.saves),
        )

    def get_finished(
        self,
        finished_gen_req_ids: List[int],
        started_loading_req_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Report finished request IDs; LMCache manages eviction internally.

        Args:
            finished_gen_req_ids: Request IDs that finished generation.
            started_loading_req_ids: Request IDs that started KV loading.

        Returns:
            ``([], [])`` — no connector-side bookkeeping needed.
        """
        return [], []
