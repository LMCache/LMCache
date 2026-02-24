# SPDX-License-Identifier: Apache-2.0

"""Encoder Cache (EC) engine.

This is a minimal engine that mirrors the KV cache engine's layering, but for
vLLM encoder outputs:

- Key granularity: 1 per multimodal input (mm_hash)
- Value: a single tensor [num_tokens, hidden_size]

v1 scope: LocalDiskBackend only.

We intentionally reuse LMCache's LocalDiskBackend implementation to get:
- consistent disk I/O behavior
- eviction/policy hooks (even if we keep it simple initially)

Unlike KV caching, EC does not require token chunking, layerwise operations, or
paged KV GPU gather/scatter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


def _stable_u64_from_str(s: str) -> int:
    # CacheEngineKey expects an int chunk_hash; for EC we derive one from mm_hash.
    digest = hashlib.sha256(str(s).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass(slots=True)
class ECKey:
    """EC logical key.

    We keep model + mm_hash, but convert to CacheEngineKey for reuse of backends.
    """

    model_name: str
    mm_hash: str
    dtype: torch.dtype

    def to_cache_engine_key(self, world_size: int, worker_id: int) -> CacheEngineKey:
        # Store EC independent of rank by default (worker_id=0) unless you decide
        # otherwise later.
        return CacheEngineKey(
            model_name=self.model_name,
            world_size=world_size,
            worker_id=worker_id,
            chunk_hash=_stable_u64_from_str(self.mm_hash),
            dtype=self.dtype,
            request_configs={},
        )


class ECLocalDiskEngine:
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheMetadata):
        if not config.local_disk or config.max_local_disk_size <= 0:
            raise ValueError(
                "EC LocalDiskEngine requires config.local_disk and max_local_disk_size > 0"
            )
        if not config.local_cpu or config.max_local_cpu_size <= 0:
            raise ValueError(
                "EC LocalDiskEngine currently requires local_cpu enabled with max_local_cpu_size > 0 "
                "(LocalDiskBackend uses LocalCPUBackend for allocations)."
            )

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

        # EC transfer is simple contiguous tensor copy.
        # v1: we normalize storage dtype to fp16 for key stability.
        self._storage_dtype = torch.float16

    def close(self) -> None:
        # StorageManager owns its own loop/thread; best-effort shutdown.
        if hasattr(self, "_storage_manager") and self._storage_manager is not None:
            # Close backends if they expose close().
            for _name, backend in self._storage_manager.storage_backends.items():
                close_fn = getattr(backend, "close", None)
                if callable(close_fn):
                    close_fn()

    def contains(self, key: ECKey) -> bool:
        cek = ECKey(key.model_name, key.mm_hash, self._storage_dtype).to_cache_engine_key(
            world_size=1, worker_id=0
        )
        # Directly query disk tier.
        return self._storage_manager.storage_backends["LocalDiskBackend"].contains(cek)

    def put(self, key: ECKey, tensor: torch.Tensor) -> None:
        # v1: normalize storage dtype for stable keying.
        t = tensor.detach().to(device="cpu", dtype=self._storage_dtype)

        # Allocate via LMCache allocator (LocalCPUBackend) through StorageManager.
        mem_obj = self._storage_manager.allocate(
            shapes=t.shape,
            dtypes=t.dtype,
            fmt=MemoryFormat.UNDEFINED,
            eviction=True,
            busy_loop=True,
        )
        if mem_obj is None or mem_obj.tensor is None:
            logger.warning("EC allocate failed; skipping put for %s", key.mm_hash)
            return

        # Copy data into allocator-managed buffer.
        mem_obj.tensor.copy_(t)

        cek = ECKey(key.model_name, key.mm_hash, self._storage_dtype).to_cache_engine_key(
            world_size=1, worker_id=0
        )
        self._storage_manager.batched_put([cek], [mem_obj], location="LocalDiskBackend")

    def get(self, key: ECKey, device: Optional[str] = None) -> Optional[torch.Tensor]:
        logger.debug("Getting encoder cache for key %s", key)

        cek = ECKey(key.model_name, key.mm_hash, self._storage_dtype).to_cache_engine_key(
            world_size=1, worker_id=0
        )
        mem_objs = self._storage_manager.batched_get([cek], location="LocalDiskBackend")
        mem_obj = mem_objs[0]
        if mem_obj is None or mem_obj.tensor is None:
            return None

        # Always clone before releasing allocator-owned buffer.
        cpu_t = mem_obj.tensor.detach().clone()
        mem_obj.ref_count_down()

        if device is None:
            return cpu_t
        return cpu_t.to(device=device)
