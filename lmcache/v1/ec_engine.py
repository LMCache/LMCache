# SPDX-License-Identifier: Apache-2.0

"""Encoder Cache (EC) engine.

This is a minimal engine that mirrors the KV cache engine's layering, but for
vLLM encoder outputs:

- Key granularity: 1 per multimodal input (mm_hash)
- Value: a single tensor [num_tokens, hidden_size]

v1 scope: uses any configured LMCache storage backend.

Unlike KV caching, EC does not require token chunking, layerwise operations, or
paged gather/scatter.
"""

# Future
from __future__ import annotations

# Standard
import hashlib

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


def _stable_u64_from_str(s: str) -> int:
    digest = hashlib.sha256(str(s).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class ECCacheEngine:
    """LMCache-backed engine for vLLM encoder cache (EC) tensors.

    Mirrors the KV cache engine's layering, but each EC entry is keyed by a
    single multimodal hash and stores one tensor of shape
    ``[num_tokens, hidden_size]``. EC does not require token chunking,
    layerwise operations, or paged gather/scatter.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        encoder_dtype: torch.dtype,
    ) -> None:
        """Initialize the EC cache engine.

        Args:
            config: LMCache engine configuration; supplies storage backends.
            metadata: LMCache metadata describing model identity, parallelism,
                and worker placement.
            encoder_dtype: dtype of the encoder output tensors. Used as the
                dtype field of the on-disk cache key, so it must be stable
                across processes that share the same EC cache. Decoupled
                from ``metadata.kv_dtype`` so that changing KV quantization
                does not invalidate EC entries.

        Raises:
            ValueError: if no non-allocator storage backend is configured.
        """
        self.config = config
        self.metadata = metadata
        self._model_name = metadata.model_name
        self._world_size = metadata.world_size
        self._worker_id = metadata.worker_id
        self._dtype = encoder_dtype

        # Mirror KV engine layering: StorageManager owns backends + allocator.
        # First Party
        from lmcache.v1.event_manager import EventManager
        from lmcache.v1.storage_backend.storage_manager import StorageManager

        self._event_manager = EventManager()
        self._storage_manager = StorageManager(
            config=config,
            metadata=metadata,
            event_manager=self._event_manager,
            lmcache_worker=None,
            async_lookup_server=None,
        )

        available_backends = self._storage_manager.get_non_allocator_backends()
        if len(available_backends) == 0:
            raise ValueError(
                "EC cache engine found no storage backends. Configure at least one "
                "backend (e.g. local_disk, remote_url, gds_path, nixl storage plugin)."
            )

        logger.info(
            "Initialized EC cache engine with storage backends=%s",
            available_backends,
        )

    def close(self) -> None:
        """Close EC storage resources and background workers."""
        if hasattr(self, "_storage_manager") and self._storage_manager is not None:
            self._storage_manager.close()

    def _make_cache_key(self, mm_hash: str) -> CacheEngineKey:
        return CacheEngineKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._worker_id,
            chunk_hash=_stable_u64_from_str(mm_hash),
            dtype=self._dtype,
            request_configs={},
        )

    def contains(self, mm_hash: str) -> bool:
        """Return whether encoder cache exists for the given multimodal hash."""
        key = self._make_cache_key(mm_hash)
        return self._storage_manager.contains(key) is not None

    def put(self, encoder_cache: dict[str, torch.Tensor], mm_hash: str) -> bool:
        """Store one encoder cache tensor from encoder_cache into LMCache.

        Returns:
            True if a store task is submitted, False otherwise.
        """
        if mm_hash not in encoder_cache:
            return False

        key = self._make_cache_key(mm_hash)
        tensor = encoder_cache[mm_hash]

        # Allocate via LMCache allocator (LocalCPUBackend) through StorageManager.
        # Preserve the source tensor dtype to avoid precision loss.
        mem_obj = self._storage_manager.allocate(
            shapes=tensor.shape,
            dtypes=tensor.dtype,
            fmt=MemoryFormat.EC_TD,
            eviction=True,
            busy_loop=False,
        )
        if mem_obj is None or mem_obj.tensor is None:
            logger.warning("EC allocate failed; skipping put for key %s", key)
            return False

        # Single copy: GPU -> pinned CPU buffer, handles device transfer + dtype cast.
        mem_obj.tensor.copy_(tensor)

        self._storage_manager.batched_put(
            [key],
            [mem_obj],
        )
        nbytes = tensor.element_size() * tensor.numel()
        logger.info("EC put: stored %d bytes for mm_hash=%s", nbytes, mm_hash)
        return True

    def get(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        device: str,
    ) -> bool:
        """Load one encoder cache tensor into encoder_cache if present.

        Returns:
            True if encoder_cache is populated by this call, False otherwise.
        """
        if mm_hash in encoder_cache:
            return False

        key = self._make_cache_key(mm_hash)

        mem_objs = self._storage_manager.batched_get(
            [key],
        )
        mem_obj = mem_objs[0]
        if mem_obj is None:
            return False
        if mem_obj.tensor is None:
            mem_obj.ref_count_down()
            return False

        try:
            out = mem_obj.tensor.to(device=device)
            if out.data_ptr() == mem_obj.tensor.data_ptr():
                out = out.clone()
            encoder_cache[mm_hash] = out
            return True
        finally:
            mem_obj.ref_count_down()
