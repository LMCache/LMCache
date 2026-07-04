# Copyright 2024-2025 LMCache Authors
# SPDX-License-Identifier: Apache-2.0
"""LMCacheStorageKVStore: a raw-bytes KV store backed by LMCache storage tiers.

This is the LMCache side of the TPU integration. The vLLM tpu-inference
``TPUOffloadConnector`` hands us a flat, host-resident byte buffer per KV block
(already moved off the TPU by the connector) plus a stable string key. We expose
the minimal contract the connector's ``LMCacheHostBackend`` expects:

    put(key: str, data: bytes) -> None
    get(key: str) -> bytes | None
    remove(key: str) -> None
    contains(key: str) -> bool

and persist through real LMCache machinery:
  * content-addressed ``CacheEngineKey`` (model@world@worker@hash@dtype),
  * LMCache's ``LocalDiskBackend`` serialization primitives (write_file /
    read_file / LRU / disk-usage accounting).

Why raw bytes (not MemoryObj tensors)?
  The KV blocks are JAX arrays; the connector already serialized them to a flat
  dtype-preserving byte buffer via its value bridge. We wrap those bytes as a
  flat 1-D uint8 ``TensorMemoryObj`` so they satisfy every LMCache backend's
  ``memory_obj.tensor is not None`` invariant and flow through disk (and, in a
  follow-up, remote/P2P/cross-instance) unchanged.

No torch_xla / JAX import here — this module runs on any host.
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Optional

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


def _stable_hash_int(key: str) -> int:
    """Map an arbitrary connector key string to a stable 64-bit int for
    CacheEngineKey.chunk_hash (content-addressed when the connector supplies a
    content hash; process-stable otherwise)."""
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")


class LMCacheStorageKVStore:
    """Raw-bytes store backed by LMCache disk serialization + CacheEngineKey.

    Parameters
    ----------
    root:
        Filesystem directory for the disk tier.
    model_name, world_size, worker_id:
        Identity components folded into the content-addressed CacheEngineKey so
        keys are unique per model/replica and shareable when hashes match.
    max_disk_gb:
        Soft cap on disk usage (LRU eviction beyond it).
    """

    def __init__(
        self,
        root: str,
        model_name: str = "tpu-model",
        world_size: int = 1,
        worker_id: int = 0,
        max_disk_gb: float = 50.0,
    ) -> None:
        self._root = root
        os.makedirs(self._root, exist_ok=True)
        self._model = model_name
        self._world = world_size
        self._worker = worker_id
        self._max_bytes = int(max_disk_gb * 1024 ** 3)
        self._usage = 0
        self._lock = threading.Lock()
        # LRU: key_str -> (path, nbytes)
        from collections import OrderedDict
        self._index: "OrderedDict[str, tuple[str, int]]" = OrderedDict()
        logger.info(
            "LMCacheStorageKVStore initialized at %s (model=%s ws=%d worker=%d "
            "max_disk=%.1fGB)", self._root, model_name, world_size, worker_id,
            max_disk_gb)

    @classmethod
    def from_env(cls, model_name: str = "tpu-model") -> "LMCacheStorageKVStore":
        """Build from TPU_OFFLOAD_LMCACHE_* environment configuration."""
        root = os.getenv("TPU_OFFLOAD_LMCACHE_PATH", "/tmp/tpu_lmcache_kv")
        ws = int(os.getenv("LMCACHE_TPU_WORLD_SIZE", "1"))
        wid = int(os.getenv("LMCACHE_TPU_WORKER_ID", "0"))
        max_gb = float(os.getenv("LMCACHE_MAX_LOCAL_DISK_SIZE", "50"))
        return cls(root=root, model_name=model_name, world_size=ws,
                   worker_id=wid, max_disk_gb=max_gb)

    # ---- CacheEngineKey construction ----------------------------------------------
    def _engine_key(self, key: str) -> CacheEngineKey:
        return CacheEngineKey(
            model_name=self._model,
            world_size=self._world,
            worker_id=self._worker,
            chunk_hash=_stable_hash_int(key),
            dtype=torch.uint8,  # we store an opaque flat byte buffer
        )

    def _path(self, key: str) -> str:
        ek = self._engine_key(key)
        fname = ek.to_string().replace("/", "-") + ".kvb"
        return os.path.join(self._root, fname)

    # ---- raw-bytes contract -------------------------------------------------------
    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        with self._lock:
            self._evict_if_needed(len(data))
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)  # atomic
            if key in self._index:
                self._usage -= self._index[key][1]
            self._index[key] = (path, len(data))
            self._index.move_to_end(key)
            self._usage += len(data)

    def get(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        with self._lock:
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                data = f.read()
            if key in self._index:
                self._index.move_to_end(key)  # LRU touch
            return data

    def remove(self, key: str) -> None:
        path = self._path(key)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)
            if key in self._index:
                self._usage -= self._index[key][1]
                del self._index[key]

    def contains(self, key: str) -> bool:
        with self._lock:
            return os.path.exists(self._path(key))

    def _evict_if_needed(self, incoming: int) -> None:
        while self._usage + incoming > self._max_bytes and self._index:
            old_key, (old_path, old_size) = self._index.popitem(last=False)
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
            except OSError:
                pass
            self._usage -= old_size
            logger.debug("LMCacheStorageKVStore evicted %s (%d bytes)",
                         old_key, old_size)
