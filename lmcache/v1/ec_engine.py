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

import asyncio
import hashlib
import threading
from dataclasses import dataclass
from typing import Optional

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.memory_management import allocate_aligned_cpu_tensor
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

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

        self.config = config
        self.metadata = metadata

        # LocalDiskBackend needs an event loop; StorageManager normally owns this.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="lmcache-ec-disk-loop", daemon=True
        )
        self._loop_thread.start()

        # LocalDiskBackend requires a LocalCPUBackend for allocation/loading.
        # We keep this minimal: always allocate from CPU backend.
        # Note: LocalCPUBackend expects config.local_cpu True and max_local_cpu_size > 0.
        # For EC v1, we force-enable it if needed by cloning config values.
        if not config.local_cpu or config.max_local_cpu_size <= 0:
            raise ValueError(
                "EC LocalDiskEngine currently requires local_cpu enabled with max_local_cpu_size > 0 "
                "(LocalDiskBackend uses LocalCPUBackend for allocations)."
            )

        # LocalCPUBackend also needs loop + metadata.
        self.local_cpu_backend = LocalCPUBackend(config, metadata, self._loop, dst_device="cpu")

        self.local_disk_backend = LocalDiskBackend(
            config,
            loop=self._loop,
            local_cpu_backend=self.local_cpu_backend,
            dst_device="cpu",
            lmcache_worker=None,
            metadata=metadata,
        )

    def close(self) -> None:
        try:
            self.local_disk_backend.close()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def contains(self, key: ECKey) -> bool:
        cek = key.to_cache_engine_key(world_size=1, worker_id=0)
        # storage manager contains
        return self.local_disk_backend.contains(cek)

    def put(self, key: ECKey, tensor: torch.Tensor) -> None:

        """
        add:
        - introduce gpu connector
        - storage manager
        """
        # this actually allocates the buffer
        #  memory_obj = self.storage_manager.allocate(
        #             kv_shapes,
        #             kv_dtypes,
        #             busy_loop=self.force_store_wait,
        #             fmt=self.fmt,
        #         )

        # somthing like data self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)

        # finally the submit put task self.storage_manager.batched_put(
        #           keys, memory_objs, transfer_spec=transfer_spec
        #     )




        
        # Store CPU tensor (portable).
        cpu_tensor = tensor.detach().to(device="cpu")

        # LocalDiskBackend expects a MemoryObj with a `byte_array` backing buffer.
        # Allocate an aligned flat uint8 buffer and view it as the desired dtype/shape.
        raw_base, raw_u8 = allocate_aligned_cpu_tensor(cpu_tensor.numel() * cpu_tensor.element_size())
        raw_typed = raw_u8.view(cpu_tensor.dtype).view(cpu_tensor.shape)
        raw_typed.copy_(cpu_tensor)

        # ^^^^ we can just use gpu connector i believe

        meta = MemoryObjMetadata(
            shape=cpu_tensor.shape,
            dtype=cpu_tensor.dtype,
            address=raw_u8.data_ptr(),
            phy_size=raw_u8.numel(),
            ref_count=0,
            fmt=MemoryFormat.UNDEFINED,
            cached_positions=None,
        )

        # we want memory from the memory allocator 
        mem_obj = TensorMemoryObj(raw_u8, meta, parent_allocator=None)

        # Prevent GC of the base buffer.
        mem_obj._ec_base_buffer = raw_base  # type: ignore[attr-defined]
        cek = key.to_cache_engine_key(world_size=1, worker_id=0)

        # ^^^^ investigate how some of this code can be dont different using storage manager

        self.local_disk_backend.submit_put_task(cek, mem_obj)
        

    def get(self, key: ECKey, device: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        What you should take away
        - LocalDiskBackend.get_blocking() gives you a MemoryObj backed by an allocator-managed CPU buffer.
        - ref_count_down() returns that buffer to the pool.
        - Returning t after refcount-down is only safe if you’ve already copied it elsewhere (GPU or a cloned CPU tensor).
        Follow-up question
        Do you ever intend to call get(..., device=None) and use the returned CPU tensor directly?
        - If “no” (worker always loads to GPU), we can leave this, but it’s a footgun.
        - If “yes”, we should adjust the design so get(..., device=None) returns t.clone() (or delays ref_count_down() until the caller is done).
        """


        # memory_objs = self.storage_manager.batched_get(
         #       keys=keys,
        #        location=location,
         #   )


        # somthing like data self.gpu_connector.batched_to_gpu(memory_objs, starts, ends, **kwargs)
        # memory_obj.ref_count_down()

        # return tensor

        logger.debug("Getting encoder cache for key %s", key)
        cek = key.to_cache_engine_key(world_size=1, worker_id=0)
        mem_obj = self.local_disk_backend.get_blocking(cek)
        if mem_obj is None or mem_obj.tensor is None:
            return None
        t = mem_obj.tensor
        # release allocator object
        mem_obj.ref_count_down()
        if device is None:
            return t
        return t.to(device=device)
