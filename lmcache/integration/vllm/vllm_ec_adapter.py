# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from lmcache.integration.vllm.utils import (
    create_lmcache_metadata,
    lmcache_get_or_create_config,
)
from lmcache.v1.ec_engine import ECKey, ECLocalDiskEngine

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _get_num_encoder_tokens(request: "Request", index: int) -> int:
    # vLLM has renamed this across versions.
    if hasattr(request, "get_num_encoder_embeds"):
        return int(request.get_num_encoder_embeds(index))
    if hasattr(request, "get_num_encoder_tokens"):
        return int(request.get_num_encoder_tokens(index))
    mm_feature = request.mm_features[index]
    if hasattr(mm_feature, "mm_position") and hasattr(mm_feature.mm_position, "length"):
        return int(mm_feature.mm_position.length)
    raise AttributeError(
        "Cannot determine num encoder tokens; missing get_num_encoder_embeds/get_num_encoder_tokens/mm_position.length"
    )


@dataclass
class MMMeta:
    mm_hash: str
    num_token: int  # SAM: dont need dont need

    @staticmethod
    def make_meta(mm_hash: str, num_token: int) -> "MMMeta":
        return MMMeta(mm_hash=mm_hash, num_token=num_token)


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

        # Scheduler-side state: mm_hash -> num_token
        self._mm_datas_need_loads: dict[str, int] = {}

        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

        # Mirror KV connector style: use LMCache config system.
        config = lmcache_get_or_create_config()
        # v1: force local_disk location from vLLM ec_transfer_config.
        config.local_disk = transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
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

        self._ec_engine = ECLocalDiskEngine(config=config, metadata=lmcache_metadata)

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
            # Use LMCache LocalDiskBackend via ECLocalDiskEngine
            key = ECKey(
                model_name=self._ec_engine.metadata.model_name,
                mm_hash=mm_data.mm_hash,
                dtype=torch.float16,
            )
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
        key = ECKey(
            model_name=self._ec_engine.metadata.model_name,
            mm_hash=mm_hash,
            dtype=t.dtype,
        )
        self._ec_engine.put(key, t)
        logger.debug("Saved encoder cache for mm_hash %s", mm_hash)

    # ------------------------------
    # Scheduler-side methods
    # ------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        key = ECKey(
            model_name=self._ec_engine.metadata.model_name,
            mm_hash=identifier,
            dtype=torch.float16,  # v1: assume fp16
        )
        return self._ec_engine.contains(key)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        # SAM: Maybe just a set of MMhashes.
        mm_hash = request.mm_features[index].identifier
        num_encoder_token = _get_num_encoder_tokens(request, index)
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        # SAM: look thorugh the set above, and create the meta data based on the set
        _ = scheduler_output
        meta = LMCacheECConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    # ------------------------------
    # Helpers
    # ------------------------------

    def close(self) -> None:
        if hasattr(self, "_ec_engine") and self._ec_engine is not None:
            self._ec_engine.close()
