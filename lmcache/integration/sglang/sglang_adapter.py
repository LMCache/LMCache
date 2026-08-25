# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Optional
import uuid

# Third Party
import torch
import torch.distributed as dist

# First Party
from lmcache import torch_device_type
from lmcache.integration.sglang.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheStoreEvent,
    EngineType,
    mock_up_broadcast_fn,
    mock_up_broadcast_object_fn,
)
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import CreateGPUConnector
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # Third Party
    from sglang.srt.configs.model_config import ModelConfig

logger = init_logger(__name__)


@dataclass
class StoreMetadata:
    last_node: object
    token_ids: List[int]
    kv_indices: torch.Tensor
    offset: int
    request_id: str = ""


@dataclass
class LoadMetadata:
    token_ids: List[int]
    slot_mapping: torch.Tensor
    offset: int
    prefix_pad: int = 0
    request_id: str = ""


def init_lmcache_engine(
    model_config: ModelConfig,
    tp_size: int,
    local_rank: int,
    global_rank: int,
    kv_dtype: torch.dtype,
    config_file: str,
    use_mla: bool = False,
    kv_cache_dim: Optional[int] = None,
    page_size: int = 1,
) -> LMCacheEngine:
    """
    Initialize LMCache engine for SGLang integration.

    Args:
        model_config: SGLang model configuration
        tp_size: Tensor parallel size
        local_rank: Local GPU device index (for device selection)
        global_rank: Global tensor parallel rank (for metadata)
        kv_dtype: Data type for KV cache tensors
        config_file: Path to the LMCache YAML configuration file
        use_mla: Whether the model uses MLA (fused KV, single buffer).
            When True, ``kv_cache_dim`` must be provided.
        kv_cache_dim: The fused KV cache dimension for MLA/DSA models
            (``kv_lora_rank + qk_rope_head_dim``). Required when
            ``use_mla`` is True; ignored otherwise.
        page_size: SGLang page size (tokens per block). Passed as
            ``tokens_per_block`` in ``layout_hints`` to the GPU connector
            so that ``KVLayerGroupsManager`` can resolve ``block_size``
            for fused formats (``NL_X_NBBS_ONE_HS``) whose
            ``block_size()`` is undefined.

    Returns:
        The initialized or existing SGLang LMCache engine.

    Raises:
        ValueError: If ``use_mla=True`` but ``kv_cache_dim`` is not
            provided.
    """
    if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
        return curr_engine

    config = lmcache_get_config(config_file)
    assert isinstance(config, LMCacheEngineConfig), (
        "LMCache v1 configuration is should be passed."
    )

    # construct kv shape (for mem pool)
    num_layer = model_config.num_hidden_layers
    chunk_size = config.chunk_size

    if use_mla:
        if kv_cache_dim is None:
            raise ValueError(
                "kv_cache_dim must be provided when use_mla=True"
            )
        # MLA/DSA: fused KV, single head, kv_size=1
        kv_shape = (num_layer, 1, chunk_size, 1, kv_cache_dim)
    else:
        num_kv_head = model_config.get_num_kv_heads(tp_size)
        head_dim = model_config.head_dim
        kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_dim)

    # Change current device using local GPU index
    # Use global rank for metadata (tensor parallel rank)
    metadata = LMCacheMetadata(
        model_name=model_config.model_path,
        world_size=tp_size,
        local_world_size=tp_size,
        worker_id=global_rank,
        local_worker_id=local_rank,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
    )

    gpu_connector = CreateGPUConnector(
        config, metadata, EngineType.SGLANG,
        layout_hints={"tokens_per_block": page_size},
    )
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        config,
        metadata,
        gpu_connector,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )

    return engine


class LMCacheConnector:
    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        config_file: str,
        k_pool: Optional[List[torch.Tensor]] = None,
        v_pool: Optional[List[torch.Tensor]] = None,
        kv_caches: Optional[List[torch.Tensor]] = None,
        use_mla: bool = False,
        kv_cache_dim: Optional[int] = None,
        page_size: int = 1,
    ):
        """Initialize the LMCache connector for SGLang.

        Args:
            sgl_config: SGLang model configuration.
            tp_size: Tensor parallel size.
            rank: Global tensor parallel rank.
            config_file: Path to the LMCache YAML configuration file.
            k_pool: List of key cache tensors (MHA models). Mutually
                exclusive with ``kv_caches``.
            v_pool: List of value cache tensors (MHA models). Mutually
                exclusive with ``kv_caches``.
            kv_caches: Flat list of KV cache tensors for MLA/DSA models
                (fused buffer). When provided, ``k_pool``/``v_pool`` are
                ignored. For DSA this list includes both the MLA latent
                layers and the indexer layers appended in order.
            use_mla: Whether the model uses MLA (fused KV).
            kv_cache_dim: Fused KV cache dimension for MLA/DSA models.
            page_size: SGLang page size (tokens per block). Forwarded
                to ``init_lmcache_engine`` as ``layout_hints`` so the
                GPU connector can resolve ``block_size`` for fused
                formats.
        """
        if kv_caches is not None:
            self.kvcaches = kv_caches
        else:
            if not k_pool:
                raise ValueError(
                    "Either kv_caches or k_pool must be provided."
                )
            self.kvcaches = k_pool + v_pool
        self.use_mla = use_mla

        kv_dtype = self.kvcaches[0].dtype
        if (
            self.kvcaches[0].device.type == torch_device_type
            and self.kvcaches[0].device.index is not None
        ):
            local_rank = self.kvcaches[0].device.index
        else:
            # Fallback for CPU / odd cases
            local_rank = rank

        # rank is the global tensor parallel rank (tp_rank) from SGLang
        # local_rank is the local GPU device index
        self.lmcache_engine = init_lmcache_engine(
            sgl_config,
            tp_size,
            local_rank,
            rank,  # global_rank (tp_rank) for metadata
            kv_dtype,
            config_file,
            use_mla=use_mla,
            kv_cache_dim=kv_cache_dim,
            page_size=page_size,
        )
        self.sgl_config = sgl_config
        self.tp_size = tp_size
        self.rank = local_rank  # Use local_rank for torch.device() calls
        self.num_layer = sgl_config.num_hidden_layers

        self.lmcache_engine.post_init(kvcaches=self.kvcaches)

    ####################
    # Worker side APIs
    ####################

    def load_kv(self, load_metadata: LoadMetadata) -> int:
        token_ids = torch.tensor(load_metadata.token_ids, dtype=torch.int64).to(
            torch_device_type
        )
        slot_mapping = load_metadata.slot_mapping.to(torch_device_type)
        offset = load_metadata.offset
        if (len(token_ids) - offset) != len(slot_mapping):
            raise ValueError(
                "Length of token_ids (minus offset) must match slot_mapping length"
            )
        load_mask = torch.ones_like(token_ids, dtype=torch.bool)
        load_mask[:offset] = False
        ret_token_mask = self.lmcache_engine.retrieve(
            token_ids,
            mask=load_mask,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            offset=offset,
        )

        num_retrieved_tokens = ret_token_mask.sum().item()

        return num_retrieved_tokens

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        token_ids = torch.tensor(store_metadata.token_ids, dtype=torch.int64).to(
            torch_device_type
        )
        slot_mapping = store_metadata.kv_indices.to(torch.int64).to(torch_device_type)
        offset = store_metadata.offset
        if len(token_ids) != len(slot_mapping):
            raise ValueError("Length of token_ids must match slot_mapping length")
        store_mask = torch.ones_like(token_ids, dtype=torch.bool)

        self.lmcache_engine.store(
            token_ids,
            mask=store_mask,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            offset=offset,
        )

    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []

    def chunk_size(self):
        return self.lmcache_engine.config.chunk_size

    def reset(self):
        self.lmcache_engine.clear()

    def close(self):
        self.lmcache_engine.close()


class LMCacheLayerwiseConnector(LMCacheConnector):
    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        config_file: str,
        k_pool: Optional[List[torch.Tensor]] = None,
        v_pool: Optional[List[torch.Tensor]] = None,
        kv_caches: Optional[List[torch.Tensor]] = None,
        use_mla: bool = False,
        kv_cache_dim: Optional[int] = None,
        page_size: int = 1,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            sgl_config,
            tp_size,
            rank,
            config_file,
            k_pool=k_pool,
            v_pool=v_pool,
            kv_caches=kv_caches,
            use_mla=use_mla,
            kv_cache_dim=kv_cache_dim,
            page_size=page_size,
        )
        self._lmcache_chunk_size = self.lmcache_engine.config.chunk_size
        self.layerwise_retrievers: List[Any] = []
        self.layer_load_layer: List[int] = []
        if kv_caches is not None:
            self.kvcaches = [kv_caches]
        else:
            self.kvcaches = [k_pool, v_pool]
        self.tp_group = tp_group
        self.lookup_id_list: List[str] = []

    @torch.no_grad()
    def global_min_tokens(
        self, local_tokens: int, tp_group: dist.ProcessGroup, device: torch.device
    ):
        # If tensor parallel size is 1, no need for all_reduce
        if self.tp_size == 1:
            return local_tokens

        t = torch.tensor([local_tokens], dtype=torch.int32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=tp_group)
        return int(t.item())

    def load_kv_layerwise(self, layer_id: int) -> None:
        if len(self.layerwise_retrievers) == 0:
            return

        indices_to_remove = []
        for i in range(len(self.layerwise_retrievers)):
            if self.layer_load_layer[i] == layer_id + 1:
                next(self.layerwise_retrievers[i])
                self.layer_load_layer[i] += 1
                if self.layer_load_layer[i] == self.sgl_config.num_hidden_layers:
                    indices_to_remove.append(i)

        for i in sorted(indices_to_remove, reverse=True):
            del self.layerwise_retrievers[i]
            del self.layer_load_layer[i]
            self.lmcache_engine.lookup_unpin(self.lookup_id_list[i])
            del self.lookup_id_list[i]

        return

    def start_load_kv(self, load_metadata: LoadMetadata) -> int:
        token_ids = torch.tensor(load_metadata.token_ids, dtype=torch.int64).to(
            torch_device_type
        )
        slot_mapping = load_metadata.slot_mapping.to(torch_device_type)
        offset = load_metadata.offset

        assert self.lmcache_engine is not None

        load_mask = torch.ones_like(token_ids, dtype=torch.bool)
        load_mask[:offset] = False

        lookup_id = str(uuid.uuid4())
        retrieve_token_num = self.lmcache_engine.lookup(
            token_ids,
            lookup_id=lookup_id,
            pin=True,
        )

        retrieve_token_num = self.global_min_tokens(
            retrieve_token_num,
            self.tp_group,
            torch.device(f"{torch_device_type}:{self.rank}"),
        )

        # No new tokens to retrieve from LMCache
        if retrieve_token_num <= offset:
            self.lmcache_engine.lookup_unpin(lookup_id)
            logger.info(
                "LMCache retrieve skipped: lookup=%d, "
                "offset=%d, no new tokens to retrieve",
                retrieve_token_num,
                offset,
            )
            return 0

        layerwise_retriever = self.lmcache_engine.retrieve_layer(
            token_ids[:retrieve_token_num],
            mask=load_mask[:retrieve_token_num],
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping[:retrieve_token_num],
            offset=offset,
            sync=False,
        )

        next(layerwise_retriever)
        # Load First Layer
        next(layerwise_retriever)

        self.layerwise_retrievers.append(layerwise_retriever)
        self.layer_load_layer.append(1)

        self.lookup_id_list.append(lookup_id)

        num_new_tokens = retrieve_token_num - offset
        logger.info(
            "LMCache retrieve started: lookup=%d, offset=%d, retrieve %d new tokens",
            retrieve_token_num,
            offset,
            num_new_tokens,
        )

        return num_new_tokens

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        slot_mapping = store_metadata.kv_indices.to(torch.int64).to(torch_device_type)
        token_ids = torch.tensor(store_metadata.token_ids, dtype=torch.int64).to(
            torch_device_type
        )
        store_mask = torch.ones_like(token_ids, dtype=torch.bool)

        logger.info(
            "LMCache store_kv started: tokens=%d, num_layers=%d, offset=%d",
            len(token_ids),
            self.sgl_config.num_hidden_layers,
            store_metadata.offset,
        )

        lookup_id = str(uuid.uuid4())
        try:
            self.lmcache_engine.lookup(token_ids, lookup_id=lookup_id, pin=True)

            layerwise_storer = self.lmcache_engine.store_layer(
                token_ids,
                mask=store_mask,
                kvcaches=self.kvcaches,
                slot_mapping=slot_mapping,
                offset=store_metadata.offset,
                sync=False,
            )

            # Initial next() to start the generator
            try:
                next(layerwise_storer)
            except StopIteration:
                logger.error(
                    "store_layer generator stopped prematurely before layer loop"
                )
                return

            # Iterate through each layer
            for layer_idx in range(self.sgl_config.num_hidden_layers):
                try:
                    next(layerwise_storer)
                except StopIteration:
                    logger.error(
                        "store_layer generator stopped at layer %d/%d",
                        layer_idx,
                        self.sgl_config.num_hidden_layers,
                    )
                    break

            self.lmcache_engine.lookup_unpin(lookup_id)
            logger.info("LMCache store_kv completed: stored %d tokens", len(token_ids))
        except Exception as e:
            logger.error(
                "LMCache store_kv failed: %s: %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            try:
                self.lmcache_engine.lookup_unpin(lookup_id)
            except Exception as unpin_err:
                logger.error("Failed to unpin lookup: %s", unpin_err, exc_info=True)
