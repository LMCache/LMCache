# SPDX-License-Identifier: Apache-2.0

"""Encoder Cache (EC) engine.

This is a minimal engine that mirrors the KV cache engine's layering, but for
vLLM encoder outputs:

- Key granularity: 1 per multimodal input (mm_hash)
- Value: a single tensor [num_tokens, hidden_size]

v1 scope: uses any configured LMCache storage backend.

Unlike KV caching, EC does not require token chunking, layerwise operations, or
paged KV GPU gather/scatter.
"""

from __future__ import annotations

from typing import Optional

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class ECCacheEngine:
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        *,
        storage_location: Optional[str] = None,
    ):
        if not config.enable_pd and (
            not config.local_cpu or config.max_local_cpu_size <= 0
        ):
            raise ValueError(
                "EC cache engine requires an allocator backend. Enable local_cpu with "
                "max_local_cpu_size > 0, or enable PD."
            )

        self.config = config
        self.metadata = metadata
        self._storage_location: str = ""

        # Mirror KV engine layering: StorageManager owns backends + allocator.
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

        available_backends = self._storage_manager.non_allocator_backends
        if len(available_backends) == 0:
            raise ValueError(
                "EC cache engine found no storage backends. Configure at least one "
                "backend (e.g. local_disk, remote_url, gds_path, nixl storage plugin)."
            )

        if storage_location is not None:
            if storage_location not in available_backends:
                raise ValueError(
                    f"Requested EC storage backend '{storage_location}' is not available. "
                    f"Available backends: {available_backends}"
                )
            resolved_location = storage_location
        else:
            preferred_order = [
                "LocalDiskBackend",
                "RemoteBackend",
                "GdsBackend",
                "NixlStorageBackend",
                "P2PBackend",
                "PDBackend",
                "LocalCPUBackend",
            ]
            resolved_location = available_backends[0]
            for candidate in preferred_order:
                if candidate in available_backends:
                    resolved_location = candidate
                    break

        self._storage_location = resolved_location
        logger.info(
            "Initialized EC cache engine with storage backend '%s' (available=%s)",
            self._storage_location,
            available_backends,
        )

        # EC transfer is simple contiguous tensor copy.
        # v1: we normalize storage dtype to fp16 for key stability.
        self._storage_dtype = torch.float16

    def close(self) -> None:
        if hasattr(self, "_storage_manager") and self._storage_manager is not None:
            self._storage_manager.close()

    def contains(self, key: CacheEngineKey) -> bool:
        return (
            self._storage_manager.contains(
                key,
                search_range=[self._storage_location],
            )
            is not None
        )

    def put(self, key: CacheEngineKey, tensor: torch.Tensor) -> None:
        # Allocate via LMCache allocator (LocalCPUBackend) through StorageManager.
        # Use the original tensor's shape but normalize to storage dtype (fp16).
        mem_obj = self._storage_manager.allocate(
            shapes=tensor.shape,
            dtypes=self._storage_dtype,
            fmt=MemoryFormat.EC_T2D,
            eviction=True,
            busy_loop=True,
        )
        if mem_obj is None or mem_obj.tensor is None:
            logger.warning("EC allocate failed; skipping put for key %s", key)
            return

        # Single copy: GPU -> pinned CPU buffer, handles device transfer + dtype cast.
        mem_obj.tensor.copy_(tensor.detach())

        self._storage_manager.batched_put(
            [key],
            [mem_obj],
            location=self._storage_location,
        )

    def get(
        self,
        key: CacheEngineKey,
        device: str,
    ) -> Optional[torch.Tensor]:
        logger.debug("Getting encoder cache for key %s", key)

        mem_objs = self._storage_manager.batched_get(
            [key],
            location=self._storage_location,
        )
        mem_obj = mem_objs[0]
        if mem_obj is None or mem_obj.tensor is None:
            return None

        try:
            return mem_obj.tensor.to(device=device)
        finally:
            mem_obj.ref_count_down()
