# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from lmcache.integration.vllm.utils import (
    create_lmcache_metadata,
    lmcache_get_or_create_config,
)
from lmcache.utils import CacheEngineKey
from lmcache.v1.ec_engine import ECCacheEngine

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _stable_u64_from_str(s: str) -> int:
    digest = hashlib.sha256(str(s).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass
class MMMeta:
    mm_hash: str

    @staticmethod
    def make_meta(mm_hash: str) -> "MMMeta":
        return MMMeta(mm_hash=mm_hash)


@dataclass
class LMCacheECConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta) -> None:
        self.mm_datas.append(mm_data)


class LMCacheECConnectorImpl:
    def __init__(self, vllm_config: "VllmConfig", role: "ECConnectorRole", parent):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role

        # Scheduler-side state: set of multimodal hashes to load.
        self._mm_hashes_need_loads: set[str] = set()

        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

        # Mirror KV connector style: use LMCache config system.
        config = lmcache_get_or_create_config()

        ec_storage_backend = transfer_config.get_from_extra_config(
            "storage_backend", None
        )
        if ec_storage_backend is None:
            ec_storage_backend = transfer_config.get_from_extra_config(
                "storage_location", None
            )
        if isinstance(ec_storage_backend, str):
            ec_storage_backend = ec_storage_backend.strip() or None

        # Backward-compatible default: use LocalDiskBackend with shared_storage_path.
        if ec_storage_backend is None or ec_storage_backend == "LocalDiskBackend":
            ec_storage_backend = "LocalDiskBackend"
            config.local_disk = transfer_config.get_from_extra_config(
                "shared_storage_path", "/tmp"
            )
        logger.info(
            "LMCache EC connector using storage backend: %s", ec_storage_backend
        )

        # LocalDiskBackend currently requires LocalCPUBackend for allocations.
        # For EC v1, if user didn't configure local_cpu, we default a small CPU pool.
        if not config.local_cpu:
            config.local_cpu = True
        if config.max_local_cpu_size <= 0:
            config.max_local_cpu_size = 1  # GB

        # Ensure disk budget is set.
        if config.max_local_disk_size <= 0:
            config.max_local_disk_size = 64  # GB default for EC v1

        # Build metadata (model_name/world_size/worker_id mainly). We'll treat EC as rank-agnostic.
        lmcache_metadata, _ = create_lmcache_metadata(vllm_config, role="worker")
        self._model_name = lmcache_metadata.model_name
        self._cache_world_size = 1
        self._cache_worker_id = 0
        self._cache_dtype = torch.float16

        self._ec_engine = ECCacheEngine(
            config=config,
            metadata=lmcache_metadata,
            storage_location=ec_storage_backend,
        )

    def _make_cache_key(self, mm_hash: str) -> CacheEngineKey:
        return CacheEngineKey(
            model_name=self._model_name,
            world_size=self._cache_world_size,
            worker_id=self._cache_worker_id,
            chunk_hash=_stable_u64_from_str(mm_hash),
            dtype=self._cache_dtype,
            request_configs={},
        )

    # ------------------------------
    # Worker-side methods
    # ------------------------------

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        from vllm.platforms import current_platform

        metadata = self._parent._get_connector_metadata()
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata is None"
            )
            return
        if not isinstance(metadata, LMCacheECConnectorMetadata):
            raise TypeError(f"Unexpected metadata type: {type(metadata)}")

        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            # Use LMCache storage via ECCacheEngine
            key = self._make_cache_key(mm_data.mm_hash)
            t = self._ec_engine.get(key, device=current_platform.device_type)
            if t is None:
                continue
            encoder_cache[mm_data.mm_hash] = t
            logger.debug("Loaded encoder cache for hash %s", mm_data.mm_hash)

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:

        if not getattr(self._parent, "is_producer", False):
            return

        if mm_hash not in encoder_cache:
            return

        t = encoder_cache[mm_hash]
        key = self._make_cache_key(mm_hash)
        self._ec_engine.put(key, t)
        logger.debug("Saved encoder cache for mm_hash %s", mm_hash)

    # ------------------------------
    # Scheduler-side methods
    # ------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        key = self._make_cache_key(identifier)
        return self._ec_engine.contains(key)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        self._mm_hashes_need_loads.add(mm_hash)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        _ = scheduler_output
        meta = LMCacheECConnectorMetadata()
        for mm_hash in sorted(self._mm_hashes_need_loads):
            meta.add_mm_data(MMMeta.make_meta(mm_hash))
        self._mm_hashes_need_loads.clear()
        return meta

    # ------------------------------
    # Helpers
    # ------------------------------

    def close(self) -> None:
        if hasattr(self, "_ec_engine") and self._ec_engine is not None:
            self._ec_engine.close()
