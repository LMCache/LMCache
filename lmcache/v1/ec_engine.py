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
    ):
        # EC always stages through allocator-backed memory. If PD is not used,
        # enforce a minimal LocalCPU allocator here so connector-side code does
        # not need to duplicate this bootstrap logic.
        if not config.enable_pd:
            if not config.local_cpu:
                logger.info("EC enabling local_cpu allocator backend")
                config.local_cpu = True
            if config.max_local_cpu_size <= 0:
                logger.info("EC setting max_local_cpu_size to 1 GB")
                config.max_local_cpu_size = 1

        # Keep LocalDiskBackend backwards compatible when only the path is set.
        if config.local_disk and config.max_local_disk_size <= 0:
            logger.info("EC setting max_local_disk_size to 64 GB")
            config.max_local_disk_size = 64

        self.config = config
        self.metadata = metadata

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

        # EC transfer is simple contiguous tensor copy.
        # v1: we normalize storage dtype to fp16 for key stability.
        self._storage_dtype = torch.float16

    def close(self) -> None:
        if hasattr(self, "_storage_manager") and self._storage_manager is not None:
            self._storage_manager.close()

    def contains(self, key: CacheEngineKey) -> bool:
        return self._storage_manager.contains(key) is not None

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
        mem_obj.tensor.copy_(tensor)

        self._storage_manager.batched_put(
            [key],
            [mem_obj],
        )

    def get(
        self,
        key: CacheEngineKey,
        device: str,
    ) -> Optional[torch.Tensor]:
        logger.debug("Getting encoder cache for key %s", key)

        mem_objs = self._storage_manager.batched_get(
            [key],
        )
        mem_obj = mem_objs[0]
        if mem_obj is None or mem_obj.tensor is None:
            return None

        try:
            return mem_obj.tensor.to(device=device)
        finally:
            mem_obj.ref_count_down()
