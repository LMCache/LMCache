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
        shared_storage_path = transfer_config.get_from_extra_config(
            "shared_storage_path", None
        )
        if shared_storage_path:
            config.local_disk = shared_storage_path

        # Build metadata from vLLM configuration.
        lmcache_metadata, _ = create_lmcache_metadata(vllm_config, role="worker")
        self._model_name = lmcache_metadata.model_name
        self._world_size = lmcache_metadata.world_size
        self._worker_id = lmcache_metadata.worker_id
        self._dtype = lmcache_metadata.kv_dtype

        self._ec_engine = ECCacheEngine(
            config=config,
            metadata=lmcache_metadata,
        )

    def _make_cache_key(self, mm_hash: str) -> CacheEngineKey:
        return CacheEngineKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._worker_id,
            chunk_hash=_stable_u64_from_str(mm_hash),
            dtype=self._dtype,
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
            mm_tensor = self._ec_engine.get(key, device=current_platform.device_type)
            if mm_tensor is None:
                continue
            encoder_cache[mm_data.mm_hash] = mm_tensor
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

        mm_tensor = encoder_cache[mm_hash]
        key = self._make_cache_key(mm_hash)
        self._ec_engine.put(key, mm_tensor)
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
