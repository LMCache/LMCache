# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from lmcache.integration.vllm.utils import (
    create_lmcache_metadata,
    get_vllm_device_type,
    lmcache_create_ec_config,
)
from lmcache.v1.ec_engine import ECCacheEngine

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MMMeta:
    mm_hash: str

    @staticmethod
    def make_meta(mm_hash: str) -> "MMMeta":
        """Create metadata for a single multimodal hash."""
        return MMMeta(mm_hash=mm_hash)


@dataclass
class LMCacheECConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta] = field(default_factory=list)

    def add_mm_data(self, mm_data: MMMeta) -> None:
        """Append one multimodal metadata entry."""
        self.mm_datas.append(mm_data)


class LMCacheECConnectorImpl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "ECConnectorRole",
        parent: "ECConnectorBase",
    ) -> None:
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role

        # Scheduler-side state: set of multimodal hashes to load.
        self._mm_hashes_need_loads: set[str] = set()

        if vllm_config.ec_transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

        # Build EC config from standard LMCache config + EC-prefixed overrides.
        config = lmcache_create_ec_config()

        # Build metadata from vLLM configuration.
        lmcache_metadata, _ = create_lmcache_metadata(vllm_config, role="worker")

        self._ec_engine = ECCacheEngine(
            config=config,
            metadata=lmcache_metadata,
        )

    # ------------------------------
    # Worker-side methods
    # ------------------------------

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Load needed encoder caches from LMCache into vLLM encoder_cache."""
        metadata = self._parent._get_connector_metadata()
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata is None"
            )
            return
        if not isinstance(metadata, LMCacheECConnectorMetadata):
            raise TypeError(f"Unexpected metadata type: {type(metadata)}")

        for mm_data in metadata.mm_datas:
            # vllm cache hit, lmcache skip
            did_retrieve = self._ec_engine.get(
                encoder_cache=encoder_cache,
                mm_hash=mm_data.mm_hash,
                device=get_vllm_device_type(),
            )
            if not did_retrieve:
                continue
            logger.debug("Loaded encoder cache for hash %s", mm_data.mm_hash)

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:
        """Save one encoder cache entry from vLLM into LMCache."""

        if not getattr(self._parent, "is_producer", False):
            return

        if mm_hash not in encoder_cache:
            return

        did_store = self._ec_engine.put(encoder_cache=encoder_cache, mm_hash=mm_hash)
        if not did_store:
            return
        logger.debug("Saved encoder cache for mm_hash %s", mm_hash)

    # ------------------------------
    # Scheduler-side methods
    # ------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        """Return whether LMCache already contains the encoder cache for hash."""
        return self._ec_engine.contains(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Track which multimodal item (request.mm_features[index]) needs loading."""
        mm_hash = request.mm_features[index].identifier
        self._mm_hashes_need_loads.add(mm_hash)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Build worker-load metadata for hashes queued this scheduler step."""
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
        """Release EC engine resources."""
        if hasattr(self, "_ec_engine") and self._ec_engine is not None:
            self._ec_engine.close()
