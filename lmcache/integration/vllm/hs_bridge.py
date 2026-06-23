# SPDX-License-Identifier: Apache-2.0
"""LMCacheHSBridge — thin put/get wrappers over HiddenStateStore.

Wraps :class:`~lmcache.v1.cache_engine.LMCacheEngine` and exposes two
methods that mirror the signature described in the plan:

- :meth:`put_rows` — store one layer's rows per token chunk, keyed the same
  way as KV so that HS and KV are co-evicted.
- :meth:`get_rows` — retrieve cached prefix rows for one layer with a
  per-token boolean hit mask (``prefix_strict`` policy).

Layer index convention
~~~~~~~~~~~~~~~~~~~~~~
- ``layer_idx = 0`` — main hidden states tensor.
- ``layer_idx = 1 + N`` — multimodal output tensors (assigned on first write).

The layer mapping is deterministic within a process lifetime.  It is stored
on the bridge instance so that both the write path (``put_rows``) and the
read path (``get_rows``) use identical indices.

Config: ``LMCacheEngineConfig.hidden_state_layers`` filters **these** storage
slots (not transformer layer numbers).  Leave it unset (``None``) unless the
allowlist matches every ``layer_idx`` this bridge emits—otherwise multimodal
slots (1+) may be dropped by HiddenStateStore.
"""

# Standard
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)

# layer_idx reserved for main hidden states.
_HS_LAYER = 0


class LMCacheHSBridge:
    """Thin adapter between vLLM-Omni's prefix cache and LMCache's HS store.

    A single instance is constructed once per worker and held by
    :class:`~vllm_omni.core.lmcache_prefix_cache.LMCacheOmniTensorPrefixCache`.

    Args:
        engine: :class:`~lmcache.v1.cache_engine.LMCacheEngine` with
            ``enable_hidden_state_cache=True``.  When HS caching is disabled
            ``put_rows`` is a no-op and ``get_rows`` always returns a miss.
    """

    def __init__(self, engine: "LMCacheEngine") -> None:
        self._engine = engine
        # Deterministic mm_key → layer_idx mapping; built at first put_rows call
        # that uses a non-zero layer_idx.
        self._mm_key_to_layer: Dict[str, int] = {}
        self._next_mm_layer: int = 1  # 0 reserved for main HS

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def put_rows(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        rows: torch.Tensor,
        *,
        layer_idx: int = _HS_LAYER,
    ) -> None:
        """Store ``rows`` for one layer under the chunk keys matching ``token_ids``.

        Uses the same ``token_database.process_tokens`` chunking as KV so that
        HS and KV chunks share identical :class:`~lmcache.v1.kv_cache.CacheEngineKey`
        values and are co-evicted by :meth:`_register_hidden_eviction_callback`.

        Args:
            token_ids: 1-D integer tensor or list of the request's full token
                IDs (prompt + any previously decoded tokens).  Must match the
                token sequence used for KV storage so chunk boundaries align.
            rows: CPU tensor of shape ``[num_tokens, feat_dim]``.
            layer_idx: Storage layer index.  Defaults to ``0`` (main HS).
        """
        n = self._engine.store_hidden_state_chunks(
            hidden_states={layer_idx: rows},
            tokens=token_ids,
        )
        logger.debug(
            "LMCacheHSBridge.put_rows: stored %d chunks (layer=%d, tokens=%d)",
            n,
            layer_idx,
            len(token_ids) if isinstance(token_ids, list) else token_ids.shape[0],
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_rows(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        *,
        layer_idx: int = _HS_LAYER,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Retrieve cached rows for one layer with a per-token hit mask.

        Uses ``prefix_strict`` policy: stops at the first missing chunk and
        returns only the contiguous leading prefix that is cached.

        Args:
            token_ids: 1-D integer tensor or list of the request's full token
                IDs (same sequence as passed to :meth:`put_rows`).
            layer_idx: Storage layer index to retrieve.  Defaults to ``0``.

        Returns:
            ``(rows, mask)`` where:

            - ``rows`` is a CPU float tensor of shape
              ``[num_cached_prefix_tokens, feat_dim]``, or ``None`` when no
              prefix is cached.
            - ``mask`` is a bool tensor of shape ``[len(token_ids)]`` with
              ``True`` for the first ``num_cached_prefix_tokens`` positions.
        """
        n_toks = (
            len(token_ids) if isinstance(token_ids, list) else int(token_ids.shape[0])
        )
        miss_mask = torch.zeros(n_toks, dtype=torch.bool)

        result = self._engine.retrieve_hidden_states(token_ids)
        if result is None:
            return None, miss_mask

        rows = result.get(layer_idx)
        if rows is None:
            return None, miss_mask

        n_cached = int(rows.shape[0])
        mask = torch.zeros(n_toks, dtype=torch.bool)
        mask[:n_cached] = True
        logger.debug(
            "LMCacheHSBridge.get_rows: %d/%d tokens cached (layer=%d)",
            n_cached,
            n_toks,
            layer_idx,
        )
        if n_cached > 0:
            logger.info(
                "LMCacheHSBridge: hidden_state_prefix_hit "
                "cached_rows=%d seq_tokens=%d layer_idx=%d",
                n_cached,
                n_toks,
                layer_idx,
            )
        return rows, mask

    # ------------------------------------------------------------------
    # MM-key layer assignment (helper used by the subclass)
    # ------------------------------------------------------------------

    def mm_layer_idx(self, mm_key: str) -> int:
        """Return (and register) a stable ``layer_idx`` for ``mm_key``.

        Indices start at ``1``; ``0`` is reserved for main HS.  The mapping is
        deterministic within a process lifetime (first-seen order).
        """
        idx = self._mm_key_to_layer.get(mm_key)
        if idx is None:
            idx = self._next_mm_layer
            self._mm_key_to_layer[mm_key] = idx
            self._next_mm_layer += 1
        return idx
