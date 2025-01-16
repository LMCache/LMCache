import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.sequence import SequenceGroupMetadata
from vllm.utils import get_kv_cache_torch_dtype

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.cache_engine import (LMCacheEngine,
                                               LMCacheEngineBuilder)
from lmcache.experimental.gpu_connector import VLLMPagedMemGPUConnector
from lmcache.integration.vllm.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

# FIXME(Jiayi): temporarily comment this out
#from lmcache_vllm.blend_adapter import remove_request_id_indices

logger = init_logger(__name__)

LMCACHE_CUDA_STREAM = torch.cuda.Stream()


class StoreStatus(Enum):
    PREFILL = 1
    CHUNK_PREFILL = 2
    DECODE = 3
    SUFFIX_PREFILL = 4
    NONE = 5


class RetrieveStatus(Enum):
    PREFILL = 1
    CHUNK_PREFILL = 2  # not last chunk
    CHUNK_PREFILL_LAST = 3
    NONE = 4


SUPPORTED_MODELS = SimpleNamespace(
    llama_family=["meta-llama/Llama-3.1-8B-Instruct"],
    longchat_family=["lmsys/longchat-7b-16k"],
    mistral_family=["mistralai/Mistral-7B-Instruct-v0.2"],
    glm_family=["THUDM/glm-4-9b-chat"],
    qwen_family=["Qwen/Qwen-7B"],
)


@dataclass
class ModelInputSubset:
    model_layers: List[nn.Module]
    attn_layers: List[nn.Module]
    start_layer: int
    end_layer: int


def create_model_input_subset(
    model_name: str,
    model_executable: torch.nn.Module,
) -> ModelInputSubset:
    if model_name in SUPPORTED_MODELS.llama_family or \
        model_name in SUPPORTED_MODELS.mistral_family:
        model = model_executable.model
        model_layers = model.layers
        attn_layers = [layer.self_attn for layer in model_layers]
    elif model_name in SUPPORTED_MODELS.glm_family:
        model = model_executable.transformer
        model_layers = model.encoder.layers
        attn_layers = [layer.self_attention for layer in model_layers]
    else:
        # FIXME(Jiayi): `else` is the default setting, which could be wrong
        model = model_executable.model
        model_layers = model.layers
        attn_layers = [layer.self_attn for layer in model_layers]

    # FIXME(Jiayi): ChatGLM does not have `model` or `start_layer`
    # How does PP work in this case?
    if hasattr(model, "start_layer"):
        start_layer = model.start_layer
    else:
        start_layer = 0

    if hasattr(model, "end_layer"):
        end_layer = model.end_layer
    else:
        end_layer = len(model_layers)

    model_input_subset = ModelInputSubset(model_layers=model_layers,
                                          attn_layers=attn_layers,
                                          start_layer=start_layer,
                                          end_layer=end_layer)

    return model_input_subset


# FIXME(Jiayi): temporarily comment this out
#def lmcache_remove_request_id_indices(request_id):
#    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
#    if engine is None:
#        return
#    if not engine.config.enable_blending:
#        return
#    remove_request_id_indices(request_id)


def init_lmcache_engine(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
) -> Optional[LMCacheEngine]:
    """Initialize the LMCache engine by the given model config and parallel 
    config. This function will check the environment variable 
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param model_config: The model configuration in vLLM.
    :type model_config: ModelConfig
    :param parallel_config: The parallel configuration in vLLM.
    :type parallel_config: ParallelConfig
    :param cache_config: The KV cache configuration in vLLM.
    :type cache_config: CacheConfig

    :return: The initialized LMCache engine or None (if the environment variable
        `LMCACHE_CONFIG_FILE` is not set).
    :rtype: Optional[LMCacheEngine]
    """
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return None

    config = lmcache_get_config()

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype,
                                        model_config.dtype)

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    chunk_size = config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)

    # Change current device.
    torch.cuda.device(parallel_config.rank)
    metadata = LMCacheEngineMetadata(model_config.model,
                                     parallel_config.world_size,
                                     parallel_config.rank, "vllm", kv_dtype,
                                     kv_shape)
    hidden_dim_size = num_kv_head * head_size
    vllm_gpu_connector = VLLMPagedMemGPUConnector(hidden_dim_size, num_layer)
    engine = LMCacheEngineBuilder.get_or_create(ENGINE_NAME, config, metadata,
                                                vllm_gpu_connector)

    return engine


# TODO(Jiayi): This function is not used for now
def broadcast_seq_group_metadata(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    is_driver_worker: bool,
) -> "ModelInputForGPUWithSamplingMetadata":
    """Broadcast the `model_input` from driver worker to non-driver workers.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param is_driver_worker: Whether the code is executed in driver worker. 
    :type is_driver_worker: bool

    : return: Original `model_input` if driver_worker.
              Broadcasted `model_input` otherwise.
    """

    # broadcast len of `seq_group_metadata_list`
    if is_driver_worker:
        assert model_input.seq_group_metadata_list is not None
        seq_group_len_list = [len(model_input.seq_group_metadata_list)]
    else:
        seq_group_len_list = [0]
    dist.broadcast_object_list(seq_group_len_list, src=0)
    seq_group_len = seq_group_len_list[0]

    # broadcast `seq_group_metadata_list`
    seq_group_metadata_list: Sequence[Optional[SequenceGroupMetadata]]
    if is_driver_worker:
        assert model_input.seq_group_metadata_list is not None
        seq_group_metadata_list = model_input.seq_group_metadata_list
    else:
        seq_group_metadata_list = [None] * seq_group_len
    dist.broadcast_object_list(seq_group_metadata_list, src=0)

    if is_driver_worker:
        return model_input
    else:
        return dataclasses.replace(
            model_input,
            seq_group_metadata_list=\
                seq_group_metadata_list # type: ignore[arg-type]

        )


def close_lmcache_engine() -> None:
    """Close the LMCache engine if it is initialized.
    """
    logger.debug("Closing LMCache Engine")
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


# FIXME(Jiayi): Need to modify this for lmcache_connector
# This function is not used for now
def lmcache_should_retrieve(
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]) -> RetrieveStatus:
    """Check should we retrieve KV from LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory
    :type kv_caches: List[torch.Tensor]

    :return: RetrieveStatus.
    """

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    # model_input doesn't have seq_lens in tp
    # but attn_metadata does
    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    has_engine = LMCacheEngineBuilder.get(ENGINE_NAME) is not None
    if not has_engine or kv_caches is None:
        return RetrieveStatus.NONE

    attn_meta = model_input.attn_metadata
    prefill_meta = attn_meta.prefill_metadata

    # check if the current run is profiling
    is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
    if is_profile_run:
        return RetrieveStatus.NONE

    # check if the current run is prefill
    # TODO (Jiayi): chunked prefill + prefix caching in a single batch
    # is not and should not be supported here
    # what about multiple chunk prefills in a single batch??

    # Assume all chunks are prefills
    is_all_prefill_run = ((attn_meta.num_prefills == len(seq_lens))\
        and prefill_meta is not None)
    if is_all_prefill_run:
        assert model_input.sampling_metadata is not None
        selected_token_indices = \
            model_input.sampling_metadata.selected_token_indices
        if len(selected_token_indices) == 0:
            # There should only be 1 chunk in chunked prefill
            assert len(seq_lens) == 1
            return RetrieveStatus.CHUNK_PREFILL

        # `<` means chunked prefill is batched with decode
        if len(selected_token_indices) == len(seq_lens):
            return RetrieveStatus.PREFILL

    return RetrieveStatus.NONE


# FIXME(Jiayi): Need to modify this for lmcache_connector
# This function is not used for now
def lmcache_should_store(model_input: "ModelInputForGPUWithSamplingMetadata",
                         kv_caches: List[torch.Tensor]) -> List[StoreStatus]:
    """Check should we store KV into LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory
    :type kv_caches: List[torch.Tensor]

    :return: A list of StoreStatus.
             StoreStatus.PREFILL/DECODE/CHUNK_PREFILL if 
             we should store KV after PREFILL/DECODE.
             StoreStatus.NONE if no storing is required.
    """

    def is_blend_effective(attn_metadata):
        """Check if the blend is effective for the current request
        """
        blend_metadata = getattr(attn_metadata, "blend_metadata", None)
        if blend_metadata is None:
            return False

        return blend_metadata.processed_layer_count > 0

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None
    store_status = [StoreStatus.NONE] * len(seq_lens)
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    has_engine = engine is not None
    if not has_engine:
        return store_status
    assert engine is not None

    attn_meta = model_input.attn_metadata
    prefill_meta = attn_meta.prefill_metadata

    # Don't store if this request is processed by cacheblend
    if is_blend_effective(attn_meta):
        return store_status

    # TODO (yihua): Current implementation is in GPUModelRunner, so we do
    #               not need to check the type of model_runner
    #from vllm.worker.model_runner import GPUModelRunnerBase
    #if not isinstance(model_runner, GPUModelRunnerBase):
    #    return False

    # check if the current run is profiling
    is_profile_run = (kv_caches is None) or (kv_caches[0] is None)

    if is_profile_run:
        return store_status

    # FIXME(Jiayi): need to support chunked prefill (batch prefill and decode)
    # check if the current run is prefill
    is_all_prefill_run = ((attn_meta.num_prefills == len(seq_lens))\
        and (prefill_meta is not None))

    if is_all_prefill_run:
        assert model_input.seq_group_metadata_list is not None
        assert model_input.sampling_metadata is not None

        selected_token_indices = \
            model_input.sampling_metadata.selected_token_indices
        seq_group_metadata_list = model_input.seq_group_metadata_list
        seq_data_idx = 0
        selected_token_indices_idx = 0
        for seq_group_idx, seq_group_metadata in enumerate(
                seq_group_metadata_list):

            # TODO(Jiayi): Figure out scenarios (other than chunk prefill)
            # where `do_sample`` is False
            if not seq_group_metadata.do_sample:
                store_status[seq_data_idx] = StoreStatus.CHUNK_PREFILL
                seq_data_idx += 1
                continue

            for seqid, seq_data in seq_group_metadata.seq_data.items():
                if seq_data.get_len(
                ) - 1 != selected_token_indices[selected_token_indices_idx]:
                    # last chunk in chunk prefill
                    # or prefix already hit in retrieve
                    store_status[seq_data_idx] = StoreStatus.SUFFIX_PREFILL
                else:
                    store_status[seq_data_idx] = StoreStatus.PREFILL
                seq_data_idx += 1
                selected_token_indices_idx += 1
        return store_status

    # Determine whether to save decoded KV cache
    if engine.config.save_decode_cache:
        for idx, seq_len in enumerate(seq_lens):
            if seq_len % engine.config.chunk_size == 0:
                store_status[idx] = StoreStatus.DECODE
    return store_status


@_lmcache_nvtx_annotate
def lmcache_store_kv(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
    store_status: List[StoreStatus],
) -> None:
    """Store the KV caches into LMCache for the current model_input.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to get KV from
    :type kv_caches: List[torch.Tensor]
    
    :param store_status: Indicate whether and how KV cache of each req is stored
    :type store_status: List[StoreStatus]
    """
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
    assert slot_mapping is not None

    query_start_loc = model_input.attn_metadata.query_start_loc
    assert query_start_loc is not None

    # TODO (Jiayi): commenting the following out for now
    # as Turing architecture is not supported yet
    # For Turing GPU
    # num_heads = model_config.get_num_kv_heads(parallel_config)
    # head_size = model_config.get_head_size()
    # gpu_capability = torch.cuda.get_device_capability()

    seq_data_idx = 0
    assert model_input.sampling_metadata is not None
    seq_group_list = model_input.sampling_metadata.seq_groups
    assert seq_group_list is not None

    next_start_pos = 0

    for seq_group in seq_group_list:
        for seqid, seq_data in seq_group.seq_data.items():
            status = store_status[seq_data_idx]
            # TODO (Jiayi): can chunk prefill and vllm prefix
            # caching use the same logic?
            if status in [StoreStatus.NONE]:
                continue
            elif status in [
                    StoreStatus.SUFFIX_PREFILL, StoreStatus.CHUNK_PREFILL
            ]:
                seq_len = seq_lens[seq_data_idx]
            else:
                seq_len = seq_data.get_len()
                if status == StoreStatus.DECODE:
                    if seq_len % engine.config.chunk_size != 0:
                        continue
            current_tokens = torch.tensor(seq_data.get_token_ids()[:seq_len],
                                          device="cpu")

            skip_leading_tokens = engine.lookup(current_tokens)
            assert skip_leading_tokens <= seq_len

            vllm_num_required_tokens = (query_start_loc[seq_data_idx + 1] -
                                        query_start_loc[seq_data_idx]).item()
            assert isinstance(vllm_num_required_tokens, int)

            start_pos = next_start_pos
            end_pos = start_pos + vllm_num_required_tokens
            next_start_pos = end_pos

            slot_mapping_seq = slot_mapping[start_pos:end_pos]
            if skip_leading_tokens < seq_len:
                assert skip_leading_tokens % engine.config.chunk_size == 0

                # TODO(Jiayi): Turing is not supported yet
                # need to write mem kernels for turing architecture

                # TODO(Jiayi): prefix caching and chunk prefill
                # might error here. `slot_mapping_seq` could be wrong
                if len(slot_mapping) > 0:

                    stored_token_num = seq_len - skip_leading_tokens
                    skipped_token_num = skip_leading_tokens
                    kv_tensors_mask = torch.ones_like(current_tokens,
                                                      dtype=torch.bool)
                    kv_tensors_mask[:skipped_token_num] = False

                    engine.store(current_tokens.cpu(),
                                 kv_tensors_mask,
                                 kvcaches=kv_caches,
                                 slot_mapping=slot_mapping_seq,
                                 offset=skipped_token_num)
            else:
                stored_token_num = 0
                skipped_token_num = seq_len
            logger.debug(f"Store skips {skipped_token_num} tokens "\
                    f"and then stores {stored_token_num} tokens")
            seq_data_idx += 1


@_lmcache_nvtx_annotate
def lmcache_retrieve_kv(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    cache_config: CacheConfig,
    kv_caches: List[torch.Tensor],
    retrieve_status: RetrieveStatus,
) -> Tuple["ModelInputForGPUWithSamplingMetadata", bool]:
    """Retrieve the KV caches from LMCache for the current model_input. And 
    rebuild the model_input to reflect the changes in KV if necessary.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to put KV to
    :type kv_caches: List[torch.Tensor]

    :param retrieve_status: Indicate whether and how 
                            KV cache of each req is retrieved
    :type retrieve_status: List[RetrieveStatus]
    
    :return: The rebuilt model_input to reflect the changes in KV.
    :return: The boolean value to indicate whether the 
             entire execute_model should be skipped
    """
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    if engine.config.enable_blending:
        return model_input, False

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    query_start_loc = model_input.attn_metadata.query_start_loc
    assert query_start_loc is not None
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
    assert slot_mapping is not None
    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    # The following metadata are needed to rebuilt the model input
    full_tokens_list = []
    num_computed_tokens_list = []
    lmc_num_computed_tokens_list = []

    start_pos_list = []
    is_prefill_list = []

    next_start_pos = 0
    num_request_not_found = 0

    # idx is on a sequence, not a sequence group.
    idx = 0
    assert model_input.sampling_metadata is not None
    seq_group_list = model_input.sampling_metadata.seq_groups
    assert seq_group_list is not None

    for seq_group in seq_group_list:
        seq_ids = seq_group.seq_ids
        for seq_id in seq_ids:
            seq_data = seq_group.seq_data[seq_id]
            is_prefill_list.append(seq_group.is_prompt)
            if retrieve_status == RetrieveStatus.CHUNK_PREFILL:
                total_seq_len = seq_lens[idx]
            else:
                total_seq_len = seq_data.get_len()

            full_token_tensor = torch.tensor(
                seq_data.get_token_ids()[:total_seq_len], device="cpu")
            full_tokens_list.append(full_token_tensor)

            vllm_num_required_tokens = (query_start_loc[idx + 1] -
                                        query_start_loc[idx]).item()
            assert isinstance(vllm_num_required_tokens, int)

            start_pos = next_start_pos
            end_pos = start_pos + vllm_num_required_tokens
            next_start_pos = end_pos
            start_pos_list.append(start_pos)

            # number of tokens already computed by vllm
            # (e.g., chunk prefill, prefix caching)
            vllm_num_computed_tokens = total_seq_len - vllm_num_required_tokens

            # construct token mesk to indicate what tokens should be retrieved
            # from lmc. Tokens computed in vllm already should be skipped
            token_mask = torch.ones_like(full_token_tensor, dtype=torch.bool)
            lmc_chunk_size = engine.config.chunk_size
            vllm_num_computed_tokens_align = vllm_num_computed_tokens\
                // lmc_chunk_size * lmc_chunk_size
            token_mask[:vllm_num_computed_tokens_align] = False

            # TODO(Jiayi): Please get rid of this in the future
            # Please only pass the required slot_mapping to the engine
            if vllm_num_computed_tokens > 0:
                slot_mapping_req_full = torch.full((total_seq_len, ),
                                                   -1,
                                                   device=slot_mapping.device,
                                                   dtype=slot_mapping.dtype)
                slot_mapping_req_full[vllm_num_computed_tokens:] = \
                    slot_mapping[start_pos:end_pos]
            else:
                slot_mapping_req_full = slot_mapping[start_pos:end_pos]

            # call lmcache retrieve
            ret_token_mask = engine.retrieve(
                full_token_tensor,
                token_mask,
                kvcaches=kv_caches,
                slot_mapping=slot_mapping_req_full)
            lmc_num_computed_tokens = torch.sum(ret_token_mask).item() - \
                (vllm_num_computed_tokens - vllm_num_computed_tokens_align)

            assert isinstance(lmc_num_computed_tokens, int)

            # total number of computed tokens (vllm + lmc)
            num_computed_tokens = vllm_num_computed_tokens + \
                lmc_num_computed_tokens

            # TODO(Jiayi): currently we do not skip anything if chunked prefill
            # is batched with any decode or other chunked prefills.
            if retrieve_status == RetrieveStatus.CHUNK_PREFILL:
                if num_computed_tokens != total_seq_len:
                    return model_input, False
            else:
                # Avoid error when prefix is exactly the same as the retrieved
                # However, the entire prefill should be skipped in chunk prefill
                if num_computed_tokens == total_seq_len:
                    lmc_num_computed_tokens -= 1
                    num_computed_tokens -= 1

            num_computed_tokens_list.append(num_computed_tokens)
            lmc_num_computed_tokens_list.append(lmc_num_computed_tokens)

            # No cache found, move on
            if lmc_num_computed_tokens == 0:
                num_request_not_found += 1

            # Inject the lmc retrieved kv cache
            logger.debug(f"Injected token number: {lmc_num_computed_tokens}")

            idx += 1

    seq_cnt = len(query_start_loc) - 1
    assert idx == seq_cnt
    assert len(lmc_num_computed_tokens_list) == seq_cnt
    assert len(num_computed_tokens_list) == seq_cnt

    if retrieve_status == RetrieveStatus.CHUNK_PREFILL and \
        num_request_not_found == 0:
        return model_input, True

    # Some of the request can be skipped for a bit
    # TODO(Jiayi): need e2e test full prefill and partial prefill
    # in a single batch
    if num_request_not_found < seq_cnt:
        rebuilt_model_input = build_partial_prefill_input(
            model_input,
            full_tokens_list,
            num_computed_tokens_list,
            start_pos_list,
            slot_mapping,
            lmc_num_computed_tokens_list,
            is_prefill_list,
            kv_caches[0][0].device,
            cache_config,
        )
        logger.debug("Rebuilt the input!")
        return rebuilt_model_input, False

    logger.debug("Returning the original input!")
    return model_input, False


def build_partial_prefill_input(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    full_tokens_list: List[torch.Tensor],
    num_computed_tokens_list: List[int],
    start_pos_list: List[int],
    slot_mapping_flat: torch.Tensor,
    lmc_num_computed_tokens_list: List[int],
    is_prefill_list: List[bool],
    device: torch.device,
    cache_config: CacheConfig,
) -> "ModelInputForGPUWithSamplingMetadata":
    """Helper function to rebuild the model input for the current request.
    """
    assert model_input.attn_metadata is not None
    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."
    assert model_input.attn_metadata.context_lens_tensor is not None
    assert model_input.attn_metadata.block_tables is not None
    assert model_input.attn_metadata.query_start_loc is not None
    assert model_input.input_positions is not None

    rebuilt_input_tokens = []
    rebuilt_input_positions = []
    rebuilt_query_lens = []
    rebuilt_num_prefills = 0
    rebuilt_num_prefill_tokens = 0
    rebuilt_slot_mapping = []
    rebuilt_max_query_len = 0

    rebuilt_block_tables = []

    rebuilt_query_start_loc = [0]
    rebuilt_context_lens_tensor = []
    rebuilt_selected_token_indices = []

    last_query_start_loc = 0

    # recounting query and context lengths
    for idx in range(len(full_tokens_list)):
        token_tensor = full_tokens_list[idx]
        num_token = len(token_tensor)
        num_computed_token = num_computed_tokens_list[idx]
        start_pos = start_pos_list[idx]
        is_prefill = is_prefill_list[idx]
        lmc_num_computed_tokens = lmc_num_computed_tokens_list[idx]
        rebuilt_input_tokens.append(token_tensor[num_computed_token:])
        q_len = num_token - num_computed_token
        assert q_len > 0
        rebuilt_query_lens.append(q_len)
        start_input_pos_idx = start_pos + lmc_num_computed_tokens
        end_input_pos_idx = start_input_pos_idx + q_len
        rebuilt_input_positions.append(
            model_input.input_positions[start_input_pos_idx:end_input_pos_idx])
        # Attn metadata-related
        if is_prefill:
            rebuilt_num_prefills += 1
            rebuilt_num_prefill_tokens += q_len
        else:
            assert q_len == 1

        start_slot_idx = start_pos + lmc_num_computed_tokens
        end_slot_idx = start_slot_idx + q_len
        new_slot_mapping = slot_mapping_flat[start_slot_idx:end_slot_idx]
        rebuilt_slot_mapping.append(new_slot_mapping)
        rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)

        last_query_start_loc += q_len
        rebuilt_query_start_loc.append(last_query_start_loc)  # start with 0
        rebuilt_context_lens_tensor.append(num_computed_token)

        # recover `block_table`
        if len(model_input.attn_metadata.block_tables[idx]) > 0:
            rebuilt_block_tables.append(
                model_input.attn_metadata.block_tables[idx])
        else:
            slot_mapping_req = slot_mapping_flat[start_pos:end_slot_idx]
            vllm_block_size = cache_config.block_size
            rebuilt_block_table = slot_mapping_req[::16].to(torch.int32) \
                // vllm_block_size
            rebuilt_block_tables.append(rebuilt_block_table)

        # Sampling metadata related
        # seq_groups (use rebuilt query lens)
        rebuilt_selected_token_indices.append(last_query_start_loc - 1)

    # rebuilt attn_metadata
    rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
    rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
    rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
    rebuilt_attn_metadata.slot_mapping = torch.cat(rebuilt_slot_mapping).to(
        device)
    rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

    rebuilt_attn_metadata.block_tables = pad_sequence(
        rebuilt_block_tables, batch_first=True).to(device)

    rebuilt_attn_metadata.query_start_loc = torch.tensor(
        rebuilt_query_start_loc,
        dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
    rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
        rebuilt_context_lens_tensor,
        dtype=model_input.attn_metadata.context_lens_tensor.dtype,
    ).to(device)

    rebuilt_attn_metadata._cached_prefill_metadata = None
    rebuilt_sampling_metadata = None
    # rebuilt sampling_metadata
    if model_input.sampling_metadata is not None:
        rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
        for idx, q_len in enumerate(rebuilt_query_lens):
            if rebuilt_sampling_metadata.seq_groups is not None:
                rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

        rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
            rebuilt_selected_token_indices,
            dtype=model_input.sampling_metadata.selected_token_indices.dtype,
        ).to(device)

    # import here to avoid circular import.
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
        input_tokens=torch.cat(rebuilt_input_tokens).to(device),
        input_positions=torch.cat(rebuilt_input_positions).to(device),
        seq_lens=model_input.seq_lens,
        query_lens=rebuilt_query_lens,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=rebuilt_attn_metadata,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        sampling_metadata=rebuilt_sampling_metadata,
        is_prompt=model_input.is_prompt,
        async_callback=model_input.async_callback,
    )

    return rebuilt_model_input
