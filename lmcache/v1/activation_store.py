# SPDX-License-Identifier: Apache-2.0
"""ActivationStore: per-chunk cache for arbitrary per-token activations.

This is a direct generalization of :class:`HiddenStateStore` (PR #3221). The
storage/eviction machinery is unchanged; the only difference is the key axis.
Where ``HiddenStateStore`` keys each chunk by ``layer_idx`` alone, this store
keys by ``(ActivationKind, layer_idx)`` so a single pinned pool can hold hidden
states, query (Q) projections, K/V, and MLP intermediates side by side.

Constructed by :class:`~lmcache.v1.cache_engine.LMCacheEngine` when
``config.enable_activation_cache`` is True and exposed as
``engine.activation_store``. See ``docs/design/v1/activation_store.md``.
"""
from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
)
from lmcache.v1.token_database import TokenDatabase

if TYPE_CHECKING:
    from lmcache.v1.storage_backend.storage_manager import StorageManager

logger = init_logger(__name__)

_GIB = 1024**3

# Maximum LRU-evict-and-retry attempts in _alloc_chunk. Each retry evicts one
# entry from this store's own pool (never from KV) before re-attempting the
# allocation, so the effective "wait" is exhausting our own LRU queue.
_ACT_ALLOC_MAX_RETRIES = 8


class ActivationKind(Enum):
    """Kind of per-token activation cached alongside KV.

    Every kind is a dense ``[num_tokens, feature_dim]`` tensor that maps 1:1 to
    tokens, exactly like KV. ``feature_dim`` is kind-dependent (e.g. hidden_size
    for HIDDEN, ``num_q_heads * head_dim`` for QUERY) and is captured implicitly
    in the stored tensor shape; the store does not need to know it ahead of time.
    """

    HIDDEN = "hidden"
    QUERY = "query"
    KEY = "key"
    VALUE = "value"
    MLP_INTERMEDIATE = "mlp_intermediate"


# Composite per-chunk slot key: which activation, at which layer.
_SlotKey = Tuple[ActivationKind, int]


class ActivationStore:
    """Stand-alone, chunk-aligned cache for per-token activations.

    Keeps one MemoryObj per ``(chunk-key, kind, layer_idx)`` tuple on its own
    pinned-CPU pool. Chunk keys come from the engine's TokenDatabase, so every
    activation chunk shares the exact :class:`CacheEngineKey` of the
    corresponding KV chunk.

    Eviction is "lazy coupled": when a retrieve walks chunks in order, the store
    asks the bound StorageManager whether KV is still present for each key. If KV
    is gone, the orphan activation entry is dropped and the prefix ends there (KV
    evict -> activation evict). When the store's own pool is full, the store
    evicts its own LRU entry; it never evicts KV.

    Args:
        config: Engine config. Reads ``enable_activation_cache``,
            ``max_activation_cpu_size`` (GiB), and ``activation_layers``.
            Retrieval always uses prefix-strict assembly (stop at the first chunk
            missing KV or the requested activation slot).
        token_database: The same TokenDatabase used by the engine, so chunk
            boundaries and keys match KV exactly.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        token_database: TokenDatabase,
    ) -> None:
        self._config = config
        self._token_database = token_database
        self._storage_manager: Optional["StorageManager"] = None

        size_bytes = int(config.max_activation_cpu_size * _GIB)
        if size_bytes <= 0:
            raise ValueError(
                "max_activation_cpu_size must be > 0 when "
                "enable_activation_cache=True"
            )

        self._allocator = MixedMemoryAllocator(size_bytes, config=config)

        # CacheEngineKey -> {(kind, layer_idx): MemoryObj}. LRU ordering is held
        # separately in _lru so we can refresh it cheaply on access.
        self._chunks: Dict[CacheEngineKey, Dict[_SlotKey, MemoryObj]] = {}
        self._lru: "OrderedDict[CacheEngineKey, None]" = OrderedDict()

        allowlist = config.activation_layers
        self._layer_allowlist: Optional[set] = (
            set(allowlist) if allowlist is not None else None
        )

        logger.info(
            "ActivationStore initialized: pool=%.2f GB, layer_allowlist=%s",
            config.max_activation_cpu_size,
            self._layer_allowlist,
        )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def bind_storage_manager(self, storage_manager: "StorageManager") -> None:
        """Attach the engine's :class:`StorageManager`.

        Required for coupled eviction. Without it, retrieve falls back to
        activation-only presence checks.
        """
        self._storage_manager = storage_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_activation(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        activation: torch.Tensor,
        *,
        kind: ActivationKind,
        layer_idx: int = 0,
        token_offset: int = 0,
    ) -> int:
        """Store ``activation`` chunked under the same keys as KV.

        Args:
            token_ids: 1-D int tensor or list of the **full** token-ID prefix
                (same sequence used for KV storage so chunk boundaries and chunk
                keys align with KV exactly).
            activation: 2-D tensor of shape
                ``[len(token_ids) - token_offset, feature_dim]`` (CPU or GPU).
                Moved to CPU internally; its dtype is preserved (unlike the
                hidden-state store, which forced float32 -- activations are
                typically bf16/fp16 and forcing float32 doubles the footprint).
                Corresponds to ``token_ids[token_offset:]``.
            kind: Which activation this is (hidden, query, ...).
            layer_idx: Transformer layer the activation came from. Defaults to 0.
            token_offset: Number of leading tokens in ``token_ids`` that are
                **not** present in ``activation`` because they were cached by a
                prior incremental call. Defaults to 0 (full sequence provided).
                Only chunks whose token range starts at or after ``token_offset``
                are written; partially-covered chunks
                (``start < token_offset < end``) are skipped to keep each chunk
                atomic.

        Returns:
            Number of chunks stored (0 when filtered by allowlist or on
            allocation failure).

        Raises:
            ValueError: If ``token_offset`` is out of range, or if ``activation``
                has an unexpected shape.
            RuntimeError: If an allocated MemoryObj has no backing tensor
                (indicates an allocator bug).
        """
        if (
            self._layer_allowlist is not None
            and layer_idx not in self._layer_allowlist
        ):
            logger.debug(
                "ActivationStore: dropping layer_idx=%d (not in allowlist=%s)",
                layer_idx,
                self._layer_allowlist,
            )
            return 0

        if activation.dim() != 2:
            raise ValueError(
                f"activation must be 2-D [num_tokens, feature_dim], "
                f"got shape {tuple(activation.shape)}"
            )

        n_toks = (
            len(token_ids)
            if isinstance(token_ids, list)
            else int(token_ids.shape[0])
        )
        if not (0 <= token_offset <= n_toks):
            raise ValueError(
                f"token_offset ({token_offset}) must be in [0, len(token_ids) "
                f"({n_toks})]"
            )
        expected_rows = n_toks - token_offset
        if activation.shape[0] != expected_rows:
            raise ValueError(
                f"activation first dim ({activation.shape[0]}) must equal "
                f"len(token_ids) - token_offset ({expected_rows})"
            )

        act_cpu = activation.detach().to("cpu").contiguous()
        feature_dim = act_cpu.shape[1]
        slot: _SlotKey = (kind, layer_idx)

        chunks = self._chunk(token_ids)
        stored = 0
        for start, end, key in chunks:
            # Skip chunks entirely before the provided activation, or partially
            # covered (keep chunks atomic with KV boundaries).
            if start < token_offset:
                continue

            existing = self._chunks.get(key)
            if existing is not None and slot in existing:
                # Already cached for this slot; bump LRU.
                self._lru.pop(key, None)
                self._lru[key] = None
                continue

            n = end - start
            obj = self._alloc_chunk(n, feature_dim, act_cpu.dtype)
            if obj is None:
                logger.warning(
                    "ActivationStore: out of pool memory after eviction; "
                    "stopping store at chunk start=%d",
                    start,
                )
                break
            tensor = obj.tensor
            if tensor is None:
                obj.ref_count_down()
                raise RuntimeError(
                    "ActivationStore: allocator returned MemoryObj with no "
                    "backing tensor"
                )
            tensor.copy_(act_cpu[start - token_offset : end - token_offset])

            slot_map = self._chunks.setdefault(key, {})
            slot_map[slot] = obj
            self._lru.pop(key, None)
            self._lru[key] = None
            stored += 1

        return stored

    def retrieve_activation(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        *,
        kind: ActivationKind,
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """Retrieve cached rows for ``(kind, layer_idx)`` as a contiguous prefix.

        Walks chunks of ``token_ids`` in order. Stops at the first chunk where
        either KV is no longer present (lazy coupled-eviction cleanup) or the
        requested activation slot is missing (prefix-strict).

        Args:
            token_ids: The token-ID prefix to look up (same keying as KV).
            kind: Which activation to retrieve.
            layer_idx: Transformer layer to retrieve. Defaults to 0.

        Returns:
            CPU tensor of shape ``[num_cached_prefix_tokens, feature_dim]`` in
            the dtype it was stored with, or None if no chunk is cached.

        Raises:
            RuntimeError: If a cached MemoryObj has no backing tensor (indicates
                an allocator bug).
        """
        slot: _SlotKey = (kind, layer_idx)
        chunks = self._chunk(token_ids)
        out_rows: List[torch.Tensor] = []
        for _, _, key in chunks:
            if not self._kv_present(key):
                # KV evicted -> drop activations for this key and stop.
                if key in self._chunks:
                    self._free_key(key)
                break

            slot_map = self._chunks.get(key)
            if slot_map is None or slot not in slot_map:
                # Activation missing for this chunk: prefix-strict stop.
                break

            obj = slot_map[slot]
            tensor = obj.tensor
            if tensor is None:
                raise RuntimeError(
                    "ActivationStore: cached MemoryObj has no backing tensor"
                )
            out_rows.append(tensor)
            # Update LRU on hit.
            self._lru.pop(key, None)
            self._lru[key] = None

        if not out_rows:
            return None
        return torch.cat(out_rows, dim=0)

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    def num_cached_chunks(self) -> int:
        """Return the number of distinct chunk keys currently cached."""
        return len(self._chunks)

    def has_chunk(
        self,
        key: CacheEngineKey,
        kind: ActivationKind,
        layer_idx: int = 0,
    ) -> bool:
        """Return True if a chunk is cached for ``(key, kind, layer_idx)``."""
        slot_map = self._chunks.get(key)
        return slot_map is not None and (kind, layer_idx) in slot_map

    def drop_key(self, key: CacheEngineKey) -> bool:
        """Manually drop all activation slots for ``key``. Test/admin use."""
        if key not in self._chunks:
            return False
        self._free_key(key)
        return True

    def close(self) -> None:
        """Free every cached chunk and the underlying pinned pool."""
        for key in list(self._chunks.keys()):
            self._free_key(key)
        self._allocator.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk(
        self, token_ids: Union[torch.Tensor, List[int]]
    ) -> List["tuple[int, int, CacheEngineKey]"]:
        """Return ``[(start, end, key)]`` for ``token_ids`` using the engine TDB.

        Materialized so we can iterate twice (key check and per-chunk copy)
        without re-running the hashing path.
        """
        return list(self._token_database.process_tokens(tokens=token_ids))

    def _kv_present(self, key: CacheEngineKey) -> bool:
        """True if KV exists in any active backend, or unknown when SM unbound."""
        if self._storage_manager is None:
            # Without an SM we cannot distinguish "KV evicted" from "no KV";
            # treat as present so the prefix walk doesn't truncate spuriously.
            return True
        return self._storage_manager.contains(key) is not None

    def _alloc_chunk(
        self, n_tokens: int, feature_dim: int, dtype: torch.dtype
    ) -> Optional[MemoryObj]:
        shape = torch.Size([n_tokens, feature_dim])
        # TODO(activation-store): add a dedicated MemoryFormat.ACT_TD instead of
        # reusing EC_TD (same [T, D] layout; distinct name aids introspection).
        for _ in range(_ACT_ALLOC_MAX_RETRIES):
            obj = self._allocator.allocate(shape, dtype, MemoryFormat.EC_TD)
            if obj is not None:
                return obj
            # Pressure: drop our LRU entry and retry. Never touches KV.
            if not self._evict_one_lru():
                break
        return None

    def _evict_one_lru(self) -> bool:
        if not self._lru:
            return False
        key, _ = self._lru.popitem(last=False)
        self._free_key(key)
        logger.debug("ActivationStore: LRU evicted key=%s", key)
        return True

    def _free_key(self, key: CacheEngineKey) -> None:
        slot_map = self._chunks.pop(key, None)
        self._lru.pop(key, None)
        if not slot_map:
            return
        # ref_count_down is the public release path: it free()s the underlying
        # buffer when the count reaches zero (mirrors LocalCPUBackend.remove).
        for obj in slot_map.values():
            obj.ref_count_down()
