# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import contextlib
import os
import time

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.sampling_params import SamplingParams

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache.utils import cdiv

# Try to import from old location before merged https://github.com/vllm-project/vllm/pull/26908
try:
    # Third Party
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
except ImportError:
    # Third Party
    from vllm.utils import get_kv_cache_torch_dtype

# Third Party
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
from lmcache import utils
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_image_grid_thw,
    extract_mm_features,
    lmcache_get_or_create_config,
    mla_enabled,
)
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.compute.ar_mux import ar_mux_env_requested, get_ar_mux
from lmcache.v1.compute.blend.blender import DeferredBatchedBlendDriver
from lmcache.v1.config import LMCacheEngineConfig, _validate_and_set_config_value
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupServer,
)
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.plugin_launcher import PluginLauncher
from lmcache.v1.compute.models.utils import VLLMModelTracker

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Blend-parts probe (LMCACHE_BLEND_PARTS=1, default OFF). After the B2 fetch
# work, `[blend-timing]` reads embed=14.8 fetch=32.8 select=0.8 ms -- so `embed`
# (_reconstruct_inputs_embeds) is now ~31% of the blend and has never been
# decomposed. Read ONCE at import; `_EMBED_PARTS` is filled by
# `_reconstruct_inputs_embeds` and `_scatter_vision_embeds` (single-threaded:
# the blend runs inline in start_load_kv on the model-runner thread) and
# drained by the `[embed-parts]` log at the end of the same call.
_BLEND_PARTS = os.environ.get("LMCACHE_BLEND_PARTS", "0") == "1"
_EMBED_PARTS: dict = {}


def _log_embed_parts(num_tokens: int, n_mm: int, path: str) -> None:
    """Emit one `[embed-parts]` line per `_reconstruct_inputs_embeds` call.

    `accounted` vs the `embed=` figure in `[blend-timing]` is the check that
    the split is complete; a large gap means a phase was missed, not that the
    remainder is free. `debug_total` = the part that exists only to feed
    disabled `logger.debug` calls (see the NOTE comments at each site).
    """
    p = _EMBED_PARTS
    acc = sum(p.values()) - p["scat_sync"] - p["scat_nan"]  # both are ⊂ scat
    dbg_total = p["dbg"] + p["scat_sync"] + p["scat_nan"]
    logger.info(
        "[embed-parts] active=1 tokens=%d mm_items=%d path=%s | tok=%.2fms "
        "enc=%.2fms emb=%.2fms devsync=%.2fms dbg=%.2fms norm=%.2fms "
        "scat=%.2fms (of which sync=%.2fms nan=%.2fms) | accounted=%.2fms "
        "debug_only=%.2fms",
        num_tokens, n_mm, path,
        p["tok"] * 1e3, p["enc"] * 1e3, p["emb"] * 1e3, p["devsync"] * 1e3,
        p["dbg"] * 1e3, p["norm"] * 1e3, p["scat"] * 1e3,
        p["scat_sync"] * 1e3, p["scat_nan"] * 1e3,
        acc * 1e3, dbg_total * 1e3,
    )


def _patch_vllm_model_registration():
    """
    Some vLLM builds miss the LMCache registration hook. Patch GPUModelRunner
    so the underlying model is registered once it is loaded, allowing the
    blender to fetch it safely later.
    """
    try:
        # Third Party
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Skip GPUModelRunner patch: %s", exc)
        return

    if getattr(GPUModelRunner.load_model, "_lmcache_patched", False):
        return

    orig_load_model = GPUModelRunner.load_model

    def _load_model_with_register(self, *args, **kwargs):
        orig_load_model(self, *args, **kwargs)
        try:
            VLLMModelTracker.register_model(ENGINE_NAME, self.model)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to register vLLM model for LMCache: %s", exc)
        try:
            VLLMModelTracker.register_encoder_cache(
                ENGINE_NAME, self.encoder_cache
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not register encoder_cache: %s", exc)

    _load_model_with_register._lmcache_patched = True  # type: ignore[attr-defined]
    GPUModelRunner.load_model = _load_model_with_register


_patch_vllm_model_registration()


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # Per-image grid dimensions [t, h, w] for M-RoPE position computation
    image_grid_thw: Optional[list] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # list[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)
        image_grid_thw = extract_image_grid_thw(new_request)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            image_grid_thw=image_grid_thw or None,
            skip_save=skip_save,
            request_configs=request_configs,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """

        self.token_ids.extend(new_token_ids)

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.allocated_block_ids.extend(new_block_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
    # Number of tokens produced by one frame.
    tokens_per_frame: Optional[int] = None
    # Multimodal placeholder positions for precise frame alignment.
    mm_positions: Optional[list["PlaceholderRange"]] = None
    # Multimodal content hashes for encoder_cache lookup.
    mm_hashes: Optional[list[str]] = None
    # Per-image grid dimensions [t, h, w] for M-RoPE position computation
    image_grid_thw: Optional[list] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 1024,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)
        
        is_last_prefill = False
        if input_token_len == tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len
        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!"
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.lmcache_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        tokens_per_frame: Optional[int] = None
        if tracker.mm_positions and len(tracker.mm_positions) > 0:
            tokens_per_frame = int(getattr(tracker.mm_positions[0], "length", 0))
            if tokens_per_frame <= 0:
                tokens_per_frame = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            tokens_per_frame=tokens_per_frame,
            mm_positions=tracker.mm_positions,
            mm_hashes=tracker.mm_hashes,
            image_grid_thw=tracker.image_grid_thw,
        )


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    if lmcache_config.enable_pd:
        return False
    else:
        return True


def _calculate_draft_layers(vllm_config, model_config):
    num_draft_layers = 0
    if vllm_config is not None and vllm_config.speculative_config is not None:
        logger.info(f"vllm_config.speculative_config: {vllm_config.speculative_config}")
        # TODO(baoloongmao): Support other MTP/draft methods
        if vllm_config.speculative_config.method == "deepseek_mtp":
            num_draft_layers = getattr(
                model_config.hf_config, "num_nextn_predict_layers", 0
            )
        elif vllm_config.speculative_config.use_eagle():
            try:
                draft_model_config = vllm_config.speculative_config.draft_model_config
                num_draft_layers = draft_model_config.get_num_layers(
                    vllm_config.parallel_config
                )
                logger.info(f"EAGLE detected {num_draft_layers} extra layer(s)")
            except Exception:
                logger.info(
                    "EAGLE detected, but failed to get the number of extra layers"
                    "falling back to 1"
                )
                num_draft_layers = 1
    return num_draft_layers


def _init_lmcache_engine(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
    role: str,
) -> LMCacheEngine:
    """Initialize the LMCache engine by the given model config and parallel
    config. This function will check the environment variable
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param lmcache_config: The LMCache configuration.
    :type lmcache_config: LMCacheEngineConfig
    :param vllm_config: The vLLM configuration.
    :type vllm_config: VllmConfig

    :return: The initialized LMCache engine
    :rtype: LMCacheEngine
    """
    if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
        return curr_engine

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    assert isinstance(lmcache_config, LMCacheEngineConfig), (
        "LMCache v1 configuration is should be passed."
    )

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

    use_mla = mla_enabled(model_config)
    if use_mla and (
        lmcache_config.remote_serde != "naive"
        and lmcache_config.remote_serde is not None
    ):
        raise ValueError("MLA only works with naive serde mode..")

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    num_draft_layers = _calculate_draft_layers(vllm_config, model_config)
    num_layer += num_draft_layers
    chunk_size = lmcache_config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
    logger.info(
        f"use mla: {use_mla}, kv shape: {kv_shape}, num_draft_layers:{num_draft_layers}"
    )

    # Change current device.
    num_gpus = torch.cuda.device_count()
    local_rank = parallel_config.rank % num_gpus
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    metadata = LMCacheEngineMetadata(
        model_config.model,
        parallel_config.world_size,
        parallel_config.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
        role,
    )

    use_gpu = need_gpu_interm_buffer(lmcache_config)
    vllm_gpu_connector: Optional[GPUConnectorInterface]

    if use_mla and lmcache_config.use_layerwise:
        raise ValueError("layerwise MLA connector is not supported yet")

    # When use_mla is True, num_kv_head is 1
    hidden_dim_size = num_kv_head * head_size
    if role == "scheduler":
        vllm_gpu_connector = None
        # Create a dummy tpg object with broadcast and broadcast_object methods
        tpg = SimpleNamespace()
        tpg.broadcast = lambda tensor, src: tensor
        tpg.broadcast_object = lambda obj, src: obj
    elif lmcache_config.use_layerwise:
        if lmcache_config.enable_blending:
            # Use layerwise connector for blending
            vllm_gpu_connector = VLLMBufferLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        else:
            vllm_gpu_connector = VLLMPagedMemLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        tpg = get_tp_group()
    else:
        vllm_gpu_connector = VLLMPagedMemGPUConnectorV2(
            hidden_dim_size,
            num_layer,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=kv_dtype,
            device=device,
            use_mla=use_mla,
        )
        tpg = get_tp_group()
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        lmcache_config,
        metadata,
        vllm_gpu_connector,
        tpg.broadcast,
        tpg.broadcast_object,
    )
    if role == "scheduler" and lmcache_config.enable_scheduler_bypass_lookup:
        assert engine.save_only_first_rank or lmcache_config.get_extra_config_value(
            "remote_enable_mla_worker_id_as0", metadata.use_mla
        ), (
            "enable_scheduler_bypass_lookup is only supported with "
            "save_only_first_rank or remote_enable_mla_worker_id_as0"
        )
    return engine


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        # Put the leading with "lmcache." and matched configs from
        # vllm extra_config to the config
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if _validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            f"Updated config {config_key} from vLLM "
                            f"extra config: {value}"
                        )

        self.config = config

        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        # Deferred codecsight blenders (pipelining A): stepped per-layer from
        # wait_for_layer_load so the per-layer KV load overlaps prefill.
        self.layerwise_blenders: list[
            Generator[None, None, None]
        ] = []
        self._codecsight_pipeline = (
            os.environ.get("VLLM_CODECSIGHT_PIPELINE", "0") == "1"
        )
        # Async overlap prototype: run the deferred blender on a side CUDA
        # stream one layer ahead so blend(L+1)'s recompute overlaps prefill(L).
        # Implies the deferred-blender path. Requires the gpu_connector's
        # per-layer global sync to be scoped (gated on the same env var).
        self._async_overlap = (
            os.environ.get("VLLM_CODECSIGHT_ASYNC_OVERLAP", "0") == "1"
        )
        if self._async_overlap:
            self._codecsight_pipeline = True
        # Tier-2: batched selective recompute -- gather all blend requests in a
        # step and recompute their anchor tokens in ONE packed forward instead
        # of N serial forwards. Default off => existing serial path unchanged.
        # See codecsight-bench/TIER2_BATCHED_BLEND.md.
        self._batched_blend = (
            os.environ.get("LMCACHE_BATCHED_BLEND", "0") == "1"
        )
        # Scheduler surgery (LMCACHE_MERGED_BLEND=1): after fetch+select, skip the
        # separate compute_layer_batched recompute and instead stash anchor
        # segments for the vLLM runner to append into the same prefill forward
        # (one QKV / attn / AR per layer covering suffix + anchors). Requires
        # batched blend. Forces OVERLAP recompute off (fetch stays eager-drained
        # here; runner applies the plan before model()). Default OFF.
        self._merged_blend = (
            os.environ.get("LMCACHE_MERGED_BLEND", "0") == "1"
            and self._batched_blend
        )
        self._merged_blend_plan = None  # set by _batched_blend_load_kv
        # Tier-2 Level-2 (LMCACHE_BATCHED_BLEND_OVERLAP=1): overlap the BATCHED
        # blend with prefill. Defers the batched fetch+recompute into a per-layer
        # generator stepped by wait_for_layer_load on the side stream -- the same
        # pipelining the serial path uses -- WITHOUT routing into the per-request
        # serial path. Requires _batched_blend. Default off => batched path stays
        # eager/blocking (unchanged). See codecsight-bench/TIER2_BATCHED_BLEND.md.
        # Merged blend is incompatible with deferred recompute (recompute IS the
        # prefill loop); keep fetch-eager when MERGED is on.
        self._batched_overlap = (
            os.environ.get("LMCACHE_BATCHED_BLEND_OVERLAP", "0") == "1"
            and self._batched_blend
            and not self._merged_blend
        )
        if self._merged_blend and os.environ.get(
                "LMCACHE_BATCHED_BLEND_OVERLAP", "0") == "1":
            logger.warning(
                "[merged-blend] LMCACHE_BATCHED_BLEND_OVERLAP=1 ignored: "
                "merged path drains fetch eagerly and folds recompute into "
                "prefill (no DeferredBatchedBlendDriver)."
            )
        if self._batched_overlap:
            # Turn on side-stream stepping in wait_for_layer_load, but do NOT set
            # _codecsight_pipeline: that flag routes to the serial path and would
            # disable the batched path (see the gate in start_load_kv). Placed
            # AFTER the _async_overlap->_codecsight_pipeline coupling above so it
            # enables overlap without the coupling.
            self._async_overlap = True
        # AR multiplexer (LMCACHE_AR_MUX=1): serialize all TP all-reduces onto one
        # CUDA stream so blend RECOMPUTE local GEMMs may run on `_blend_stream`
        # concurrent with prefill. Requires batched overlap. Without the mux,
        # recompute must stay on the prefill stream (Fix C hang under TP>1).
        # Install only on the worker (TP ranks); scheduler has no collectives.
        self._ar_mux = (
            ar_mux_env_requested()
            and self._batched_overlap
            and role != KVConnectorRole.SCHEDULER
        )
        if self._ar_mux:
            get_ar_mux().ensure_installed()
            get_ar_mux().enable()
            logger.info(
                "[ar-mux] LMCACHE_AR_MUX=1 with OVERLAP: recompute-on-blend-stream enabled"
            )
        elif (
            ar_mux_env_requested()
            and role != KVConnectorRole.SCHEDULER
            and not self._batched_overlap
        ):
            logger.warning(
                "[ar-mux] LMCACHE_AR_MUX=1 ignored: requires "
                "LMCACHE_BATCHED_BLEND=1 and LMCACHE_BATCHED_BLEND_OVERLAP=1"
            )
        # Fix C-fix (2026-07-27): latch set when a deferred blend driver could not
        # be finished cleanly. Once set, _batched_blend_load_kv falls back to the
        # eager (blocking, always-correct) path for the rest of the process, so a
        # single stalled generator degrades performance instead of deadlocking the
        # engine. Cleared only by a restart.
        self._overlap_degraded = False
        # Per-driver count of next() calls made by wait_for_layer_load, keyed by
        # id(driver). The blend_batched contract (blender.py:869-872) is
        # num_layers + 2 yields = 2 primes at creation + num_layers drives from
        # the forward. So a HEALTHY driver has had exactly num_layers drives and
        # is still un-exhausted -- next() on it would do work it should not do.
        self._blender_steps: dict[int, int] = {}

        # Hang diagnostics (LMCACHE_HANG_DUMP_S=<seconds>, default off).
        # The deferred/overlap path wedges BOTH TP workers with no traceback: the
        # last log line is blend_batched's "packing ..." and then silence until
        # vLLM's 5-minute RPC timeout kills the engine (jobs 15000264, 15001730).
        # faulthandler on a repeating timer dumps every thread's Python stack to
        # stderr -> the server log, so the wedged frame is identifiable instead of
        # guessed at. Fires unconditionally on the timer (it is a timer, not a hang
        # detector), so the useful dump is the one AFTER the last blend line.
        _hang_dump = os.environ.get("LMCACHE_HANG_DUMP_S", "")
        if _hang_dump:
            import faulthandler
            faulthandler.enable()
            faulthandler.dump_traceback_later(
                float(_hang_dump), repeat=True, exit=False
            )
            logger.warning(
                "LMCACHE_HANG_DUMP_S=%s: dumping all thread stacks every %ss",
                _hang_dump, _hang_dump,
            )
        # Per-phase blend timing (LMCACHE_BLEND_TIMING=1). Times each phase of the
        # batched blend path (embed reconstruct / KV fetch / selection / recompute)
        # with CUDA events and logs one summary line per step, so the fetch-vs-
        # recompute split is measurable. Events don't barrier the device (only a
        # single sync at the end to read them back), so this is non-perturbing --
        # but still off by default; enable only for measurement runs.
        self._blend_timing = (
            os.environ.get("LMCACHE_BLEND_TIMING", "0") == "1"
        )
        # Serial-path counterpart of the batched [blend-timing] state,
        # accumulated per engine step in start_load_kv and flushed at the end
        # of it. Defaulted here so no call site can hit a missing attribute.
        self._serial_blend_n = 0
        self._serial_blend_embed_evts = []
        self._serial_blend_layer_evts = []
        # Per-step aggregated fetch/recompute CUDA events + select wall ms
        # (from blender.take_serial_phase_timing), so serial logs match batched
        # embed/fetch/select/recompute breakdown.
        self._serial_blend_fetch_evts = []
        self._serial_blend_recompute_evts = []
        self._serial_blend_select_ms = 0.0
        # Completed-but-unread event batches, drained LAZILY one engine step
        # later. This exists to avoid adding a torch.cuda.synchronize() to the
        # serial path: the batched path can afford one because it already had
        # one, but the serial path had NO timing code at all before this, and
        # a new sync in start_load_kv would make the unbatched arm slower than
        # it is in every previously recorded job -- instrumentation perturbing
        # the very A/B it exists to measure. Events are polled with .query()
        # instead, so the device is never blocked; a batch whose events are
        # not yet complete simply waits for the next step. See the measurement
        # protocol's "instrumentation must be symmetric across arms" rule.
        self._serial_blend_pending = []
        # Packing census (LMCACHE_BLEND_PACK_DEBUG=1, implied by BLEND_TIMING).
        # Logs how many requests the scheduler put in each step vs how many
        # actually reached the packer, and names the gate whenever the batched
        # path aborts to serial. Those aborts are otherwise SILENT -- the step
        # emits no [blend-timing] line at all -- which makes "N=1" and "batched
        # path never ran" indistinguishable in a log. Pure logging, no device
        # work, so it is safe to leave on with timing.
        self._blend_pack_debug = (
            os.environ.get("LMCACHE_BLEND_PACK_DEBUG",
                           os.environ.get("LMCACHE_BLEND_TIMING", "0")) == "1"
        )
        self._blend_stream = None  # lazily created (needs CUDA device)
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        if role == KVConnectorRole.SCHEDULER:
            self.lmcache_engine: Optional[LMCacheEngine] = None
            # Check if bypass lookup is enabled for scheduler
            if config.enable_scheduler_bypass_lookup:
                # Create LMCacheEngine for scheduler when bypass is enabled
                self.lmcache_engine = _init_lmcache_engine(
                    config,
                    vllm_config,
                    role="scheduler",
                )
            # Create lookup client using factory
            self.lookup_client = LookupClientFactory.create_lookup_client(
                vllm_config, config, self.lmcache_engine
            )
            self._unfinished_requests: dict[str, Request] = {}
            self.lmcache_engine = None
        else:
            self.lmcache_engine = _init_lmcache_engine(
                config,
                vllm_config,
                role="worker",
            )

            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            # Blender is built lazily after model registration.
            self.blender = None

            # Create lookup server using factory
            assert self.lmcache_engine is not None
            self.lookup_server = LookupClientFactory.create_lookup_server(
                self.lmcache_engine, vllm_config
            )

            self.offload_server = ZMQOffloadServer(
                self.lmcache_engine,
                vllm_config,
                get_tensor_model_parallel_rank(),
            )

            # In case of MLA, the lookup server is only created on worker 0
            if self.async_loading and self.lookup_server is not None:
                assert isinstance(self.lookup_server, LMCacheAsyncLookupServer)
                self.lmcache_engine.post_init(async_lookup_server=self.lookup_server)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        # request_id -> (vllm cached tokens, lmcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}

        self.kv_cache_manager: Optional[KVCacheManager] = None

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}

        # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size

        self._save_decode_cache = config.save_decode_cache

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))

        self._requests_priority: dict[str, int] = {}

        # Track block IDs associated with failed load attempts.
        self._invalid_block_ids: set[int] = set()

        # TODO(baoloongmao): Internal api server & plugin framework support dp > 1
        if vllm_config.parallel_config.data_parallel_rank_local == 0:
            # Start internal API server if enabled
            # The enabled check is in the InternalAPIServer constructor
            self.api_server = InternalAPIServer(self)
            self.api_server.start()
            # Launch plugins
            self.plugin_launcher = PluginLauncher(
                self.config,
                role,
                self.worker_count,
                -1
                if self.lmcache_engine is None  # scheduler side
                else self.lmcache_engine.metadata.worker_id,
            )
            self.plugin_launcher.launch_plugins()
        else:
            self.api_server = None  # type: ignore[assignment]
            self.plugin_launcher = None  # type: ignore[assignment]
        logger.info(
            f"LMCache initialized for role {role} with version {utils.get_version()}, "
            f"vllm version {VLLM_VERSION}, "
            "lmcache cache_engine metadata: "
            f"{getattr(self.lmcache_engine, 'metadata', None)}"
        )

    def _ensure_blender_initialized(self):
        """
        Lazily build the blender once the vLLM model has been registered.
        If the model is unavailable, skip blending for this round instead of
        failing startup.
        """
        if not self.enable_blending or self.blender is not None:
            return

        try:
            _ = VLLMModelTracker.get_model(ENGINE_NAME)
        except Exception as exc:
            logger.warning(
                "Blending requested but vLLM model not registered yet: %s", exc
            )
            return

        assert self.lmcache_engine.gpu_connector is not None, (
            "GPU connector must be available for blending"
        )
        self.blender = LMCBlenderBuilder.get_or_create(
            ENGINE_NAME,
            self.lmcache_engine,
            self.lmcache_engine.gpu_connector,
            self.config,
        )

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    ####################
    # Worker side APIs
    ####################

    @staticmethod
    def _scatter_vision_embeds(
        text_embeds: torch.Tensor,
        vision_embeds: list[Optional[torch.Tensor]],
        mm_positions: list,
        num_tokens: int,
    ) -> torch.Tensor:
        """Overlay vision embeddings onto text embeddings at mm_positions.

        Handles NaN rows from ``scatter_mm_placeholders`` (structural tokens
        like ``<img>``/``</img>``) by preserving the text embedding there.
        ``None`` entries in *vision_embeds* (encoder_cache misses) are
        skipped, leaving the text embedding in place at that position.
        """
        inputs_embeds = text_embeds.clone()
        ve_idx = 0
        merged = 0
        for placeholder in mm_positions:
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= num_tokens:
                continue
            end = min(start + length, num_tokens)
            if ve_idx >= len(vision_embeds):
                break
            ve = vision_embeds[ve_idx]
            ve_idx += 1
            if ve is None:
                continue
            actual_len = end - start
            ve_slice = ve[:actual_len].to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            valid_mask = ~torch.isnan(ve_slice).any(dim=-1)
            # NOTE (probe finding candidate): `.item()` here and `bool()` on
            # `valid_mask.all()` below are BOTH device->host syncs, once per mm
            # item -- ~80 of them per request on this workload. The `.item()`
            # one exists only for a disabled logger.debug.
            if merged == 0:
                if _BLEND_PARTS:
                    _t0 = time.perf_counter()
                    nan_count = int((~valid_mask).sum().item())
                    _EMBED_PARTS["scat_nan"] += time.perf_counter() - _t0
                else:
                    nan_count = int((~valid_mask).sum().item())
                logger.debug(
                    "vision_embed[0]: shape=%s, nan_rows=%d/%d, dtype=%s",
                    ve.shape, nan_count, actual_len, ve.dtype)
            if ve_slice.shape[0] >= actual_len:
                if _BLEND_PARTS:
                    _t0 = time.perf_counter()
                    _all_valid = bool(valid_mask.all())
                    _EMBED_PARTS["scat_sync"] += time.perf_counter() - _t0
                else:
                    _all_valid = valid_mask.all()
                if _all_valid:
                    inputs_embeds[start:end] = ve_slice
                else:
                    inputs_embeds[start:end][valid_mask] = \
                        ve_slice[valid_mask]
            else:
                sub_len = ve_slice.shape[0]
                sub_mask = valid_mask[:sub_len]
                if sub_mask.any():
                    inputs_embeds[start:start + sub_len][sub_mask] = \
                        ve_slice[sub_mask]
            merged += 1
        # NOTE (probe finding candidate): a full [num_tokens, hidden] NaN scan
        # plus a device sync, on every blend, consumed only by the disabled
        # logger.debug below. Counted into `scat_nan`.
        if _BLEND_PARTS:
            _t0 = time.perf_counter()
            final_has_nan = bool(torch.isnan(inputs_embeds).any())
            _EMBED_PARTS["scat_nan"] += time.perf_counter() - _t0
        else:
            final_has_nan = bool(torch.isnan(inputs_embeds).any())
        logger.debug(
            "Reconstructed inputs_embeds: shape=%s, "
            "vision_embeds_merged=%d/%d, num_tokens=%d, has_nan=%s",
            inputs_embeds.shape, merged, len(vision_embeds), num_tokens,
            final_has_nan,
        )
        return inputs_embeds

    @staticmethod
    def _normalize_cached_mm_embed(
        embed: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Normalize cached multimodal embed to [tokens, dim] then crop tokens."""
        if embed.ndim == 1:
            embed = embed.unsqueeze(0)
        elif embed.ndim > 2:
            embed = embed.reshape(-1, embed.shape[-1])
        return embed[:length]

    def _reconstruct_inputs_embeds(
        self,
        token_ids: list[int],
        mm_hashes: Optional[list[str]],
        mm_positions: Optional[list["PlaceholderRange"]],
        num_tokens: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Reconstruct ``inputs_embeds`` (and optionally Deepstack embeds)
        for the cached prefix of a request by looking up the ViT encoder
        outputs stored in vLLM's ``encoder_cache``.

        Returns ``(inputs_embeds, deepstack_input_embeds)`` or
        ``(None, None)`` when the cache is unavailable.
        """
        if not mm_hashes or not mm_positions:
            return None, None

        try:
            vllm_model = VLLMModelTracker.get_model(ENGINE_NAME)
        except (ValueError, KeyError):
            return None, None

        encoder_cache = VLLMModelTracker.get_encoder_cache(ENGINE_NAME)
        if encoder_cache is None:
            logger.warning(
                "encoder_cache not registered; vision token recompute disabled"
            )
            return None, None

        # --- blend-parts probe (LMCACHE_BLEND_PARTS=1, default OFF) ----------
        # Host-side perf_counter buckets, one pair per PHASE (not per mm item),
        # so the probe cannot perturb its own subject (rule 12). See the
        # `[embed-parts]` log at the bottom of this method for how to read the
        # attribution -- CUDA launches are async, so a bucket that ends in a
        # device sync absorbs the queued device time of everything before it.
        _bp = _BLEND_PARTS
        if _bp:
            _EMBED_PARTS.clear()
            _EMBED_PARTS.update(
                {k: 0.0 for k in (
                    "tok", "enc", "emb", "devsync", "dbg", "norm",
                    "scat", "scat_sync", "scat_nan",
                )}
            )
            _bp_t = time.perf_counter()

        token_ids_t = torch.tensor(
            token_ids[:num_tokens], dtype=torch.long, device="cuda"
        )
        if _bp:
            _EMBED_PARTS["tok"] = time.perf_counter() - _bp_t
        # Incomplete cached prefix: under high-concurrency KV pressure vLLM can
        # preempt/evict part (or all) of a request's cached tokens, so the
        # available token_ids are fewer than the expected num_tokens. Downstream
        # scatter/blend still clips with num_tokens, overrunning the shorter
        # [have, hidden] tensor -> empty: token_ids_t.min() raised (N>=16);
        # partial: "mask [num_tokens] vs tensor [have]" IndexError (N=12). Both
        # killed the whole EngineCore. A partial prefix can't be blended
        # faithfully anyway, so abort and fall back to layerwise retrieval --
        # the same safe path as the encoder-miss guards below.
        if token_ids_t.shape[0] < num_tokens:
            logger.warning(
                "Cached prefix incomplete (%d of %d expected tokens present; "
                "preempted/evicted under load); aborting blend, falling back "
                "to layerwise retrieval.", int(token_ids_t.shape[0]), num_tokens,
            )
            return None, None

        # Collect vision embeddings that fall within the cached prefix.
        # Use None as sentinel for encoder_cache misses so that the list
        # stays aligned 1:1 with the mm_positions that pass the filter.
        vision_embeds: list[Optional[torch.Tensor]] = []
        num_encoder_misses = 0
        if _bp:
            _bp_t = time.perf_counter()
        for mm_hash, placeholder in zip(mm_hashes, mm_positions):
            start = int(getattr(placeholder, "offset", 0))
            length = int(getattr(placeholder, "length", 0))
            if length <= 0 or start >= num_tokens:
                continue
            end = min(start + length, num_tokens)
            enc_out = encoder_cache.get(mm_hash)
            if enc_out is None:
                logger.debug("encoder_cache miss: hash=%s", mm_hash)
                vision_embeds.append(None)
                num_encoder_misses += 1
                continue

            if isinstance(enc_out, torch.Tensor):
                enc_slice = self._normalize_cached_mm_embed(
                    enc_out, end - start)
            else:
                enc_slice = self._normalize_cached_mm_embed(
                    torch.as_tensor(enc_out), end - start)
            vision_embeds.append(enc_slice)
        if _bp:
            _EMBED_PARTS["enc"] = time.perf_counter() - _bp_t
        if num_encoder_misses > 0:
            logger.warning(
                "encoder_cache missed %d/%d items in cached prefix "
                "(evicted before blending); aborting blend, falling "
                "back to layerwise retrieval for correctness",
                num_encoder_misses, len(vision_embeds),
            )
            return None, None

        if not any(ve is not None for ve in vision_embeds):
            logger.warning(
                "No vision embeds found from encoder_cache "
                "(mm_hashes=%d, mm_positions=%d, num_tokens=%d)",
                len(mm_hashes), len(mm_positions), num_tokens,
            )
            return None, None

        # Build text embeddings then overlay vision embeddings directly
        # using mm_positions (not placeholder token ID matching, because
        # token_ids may have been rewritten with content hashes).
        lang_model = getattr(vllm_model, "language_model", vllm_model)
        embed_fn = getattr(lang_model, "get_input_embeddings", None)
        if embed_fn is None:
            embed_fn = getattr(lang_model, "embed_tokens", None)
        if embed_fn is None:
            return None, None

        if _bp:
            _bp_t = time.perf_counter()
        text_embeds = embed_fn(token_ids_t)
        if _bp:
            # `emb` is LAUNCH cost only -- the embedding gather is async. The
            # probe-only barrier below then drains everything queued so far
            # (token H2D + encoder-cache slicing + this gather), so `devsync`
            # is the real device time of the phases above and `dbg` is measured
            # against an already-idle device instead of silently absorbing it.
            # Probe-only: the flag-off path keeps the original single implicit
            # sync inside the `bool(...)` below. The ctrl arm checks the total
            # did not move.
            _EMBED_PARTS["emb"] = time.perf_counter() - _bp_t
            _bp_t = time.perf_counter()
            torch.cuda.synchronize()
            _EMBED_PARTS["devsync"] = time.perf_counter() - _bp_t
            _bp_t = time.perf_counter()
        # NOTE (probe finding candidate): every term below is computed
        # unconditionally and consumed ONLY by a logger.debug that is disabled
        # in production -- `bool(...)` on a device tensor is a full-tensor scan
        # plus a device->host sync, and `.norm().item()` / `.min().item()` /
        # `.max().item()` are evaluated as CALL ARGUMENTS, so lazy %-formatting
        # does not save them.
        text_has_nan = bool(torch.isnan(text_embeds).any())
        logger.debug(
            "text_embeds: shape=%s, has_nan=%s, norm=%.4f, "
            "token_ids min=%d max=%d",
            text_embeds.shape, text_has_nan,
            text_embeds.norm().item() if not text_has_nan else float('nan'),
            token_ids_t.min().item(), token_ids_t.max().item(),
        )
        if _bp:
            _EMBED_PARTS["dbg"] = time.perf_counter() - _bp_t

        # --- Deepstack (Qwen3-VL): encoder_cache stores concatenated
        # [main | multiscale] embeddings whose dim > text hidden size.
        # Split via _compute_deepstack_embeds before scattering. ----------
        deepstack_input_embeds: Optional[torch.Tensor] = None
        compute_deepstack = getattr(vllm_model, "_compute_deepstack_embeds", None)
        use_deepstack = getattr(vllm_model, "use_deepstack", False)
        visual_dim = int(getattr(vllm_model, "visual_dim", text_embeds.shape[-1]))
        multiscale_dim = int(getattr(vllm_model, "multiscale_dim", 0))
        expected_mm_dim = visual_dim + multiscale_dim

        # Qwen codec path may append 4 mRoPE channels to cached vision embeds.
        # Strip these channels before deepstack split / scatter reconstruction.
        # None entries (encoder_cache misses) are preserved for alignment.
        vision_embeds_norm: list[Optional[torch.Tensor]] = []
        if _bp:
            _bp_t = time.perf_counter()
        for ve in vision_embeds:
            if ve is None:
                vision_embeds_norm.append(None)
                continue
            if ve.ndim == 1:
                ve = ve.unsqueeze(0)
            if ve.ndim > 2:
                ve = ve.reshape(-1, ve.shape[-1])

            last_dim = ve.shape[-1]
            if expected_mm_dim > 0 and last_dim == expected_mm_dim + 4:
                ve = ve[:, :-4]
                last_dim = ve.shape[-1]
            vision_embeds_norm.append(ve)

        if _bp:
            _EMBED_PARTS["norm"] = time.perf_counter() - _bp_t
        vision_embeds = vision_embeds_norm
        if use_deepstack and compute_deepstack is not None:
            try:
                # If cache provides full [main|multiscale], use model split path.
                ds_embeds, vision_embeds_main = compute_deepstack(
                    token_ids_t, text_embeds, vision_embeds,
                )
                if _bp:
                    _bp_t = time.perf_counter()
                inputs_embeds = self._scatter_vision_embeds(
                    text_embeds, vision_embeds_main, mm_positions, num_tokens,
                )
                if _bp:
                    _EMBED_PARTS["scat"] = time.perf_counter() - _bp_t
                    _log_embed_parts(num_tokens, len(mm_positions), "deepstack")
                deepstack_input_embeds = ds_embeds
                return inputs_embeds, deepstack_input_embeds
            except Exception as exc:
                logger.warning("Deepstack reconstruction failed: %s", exc)

        # Fallback: ensure scatter dims match language hidden size.
        # If cache carries concatenated [main|multiscale], keep main slice only.
        # None entries (encoder_cache misses) are preserved for alignment.
        hidden = text_embeds.shape[-1]
        vision_embeds_scatter: list[Optional[torch.Tensor]] = []
        for idx, ve in enumerate(vision_embeds):
            if ve is None:
                vision_embeds_scatter.append(None)
                continue
            if ve.shape[-1] == hidden:
                vision_embeds_scatter.append(ve)
                continue
            if expected_mm_dim > 0 and ve.shape[-1] == expected_mm_dim:
                vision_embeds_scatter.append(ve[:, :hidden])
                continue
            if ve.shape[-1] > hidden:
                logger.warning(
                    "Fallback scatter: truncating cached vision dim %d -> %d "
                    "(item %d)",
                    ve.shape[-1], hidden, idx,
                )
                vision_embeds_scatter.append(ve[:, :hidden])
            else:
                logger.warning(
                    "Fallback scatter: cached vision dim %d < hidden %d "
                    "(item %d); skipping this embed",
                    ve.shape[-1], hidden, idx,
                )

        # --- Standard path (InternVL, etc.): encoder_cache dim matches
        # text hidden size. Scatter directly. --------------------------
        if _bp:
            _bp_t = time.perf_counter()
        inputs_embeds = self._scatter_vision_embeds(
            text_embeds, vision_embeds_scatter, mm_positions, num_tokens,
        )
        if _bp:
            _EMBED_PARTS["scat"] = time.perf_counter() - _bp_t
            _log_embed_parts(num_tokens, len(mm_positions), "standard")
        return inputs_embeds, deepstack_input_embeds

    def _batched_blend_load_kv(self, metadata, kvcaches, attn_metadata) -> bool:
        """Tier-2 batched selective recompute (LMCACHE_BATCHED_BLEND=1).

        Two phases (see codecsight-bench/TIER2_BATCHED_BLEND.md):
          Phase 1 (per request, memory-only): a plain ``retrieve_layer`` loads +
            RoPE-corrects each request's full cached KV into the paged cache,
            and (via the gpu_connector) records its gap positions; we then run
            the codec I-frame selection and gather anchor embeds/positions/slots.
          Phase 2 (one packed forward): ``blender.blend_batched`` recomputes all
            requests' anchor tokens together and scatters refreshed K/V back.

        Returns True if the batched path handled the step; False to fall back to
        the serial loop (any precondition miss => safe fallback, no degradation).
        """
        # First Party
        from lmcache.v1.compute.blend.metadata import LMCBlendMetadata

        blender = self.blender
        if blender is None:
            return False

        # Merged-blend hygiene: a plan stashed last step that the runner never
        # consumed (e.g. an exception before the merge hook) is stale -- its
        # slot mappings/embeds belong to a previous batch. Never apply it late;
        # drop it. The KV for those requests was fetched but not refreshed,
        # which is the same state as a soft-skipped (retrieve-only) request:
        # correct output, no anchor refresh.
        if self._merged_blend_plan is not None:
            logger.warning(
                "[merged-blend] dropping stale unconsumed plan (N=%d) from a "
                "previous step",
                len(self._merged_blend_plan.get("requests") or []),
            )
            self._merged_blend_plan = None

        # Per-phase GPU timing via CUDA events (see self._blend_timing).
        # event.record() only enqueues a timestamp marker on the current stream;
        # unlike a host-side torch.cuda.synchronize() at each boundary it does
        # NOT barrier the device, so the phases run back-to-back exactly as they
        # do with timing off -- no wall-time perturbation. We pay a SINGLE sync
        # at the very end to make the recorded events readable. Events measure
        # GPU-timeline time (what fetch/recompute are dominated by); a mostly-CPU
        # phase (select) is under-counted but negligible here. Each phase collects
        # (start, end) event pairs across the per-request loop; elapsed_time()
        # returns milliseconds.
        timing = self._blend_timing
        spans = {"embed": [], "fetch": [], "select": []}

        def _evt():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        # Measurement: snapshot the connector's global-barrier counter so we can
        # report how many device syncs the fetch phase costs (serial fetch =>
        # N*num_layers; a coalesced fetch would collapse this to num_layers).
        gconn = getattr(self.lmcache_engine, "gpu_connector", None)
        sync0 = getattr(gconn, "global_sync_count", 0) if timing else 0

        # --- packing census (LMCACHE_BLEND_PACK_DEBUG) ------------------------
        # The batch size N the packer finally sees is decided HERE, not in
        # blend_batched: N = however many of this step's scheduled requests
        # survive the gates below. Measurements to date show N=1 even under
        # confirmed request-level concurrency, and the gates are silent, so
        # there is no way to tell "the scheduler only gave us one request" from
        # "several arrived but one tripped a gate and aborted the whole batch".
        # Note the asymmetry (fixed by LMCACHE_PARTIAL_BAIL=1, default ON): a
        # missing load_spec SKIPS that request. Soft gates (encoder miss, short
        # cache, 0 anchors) used to `return False` and discard the whole pack;
        # with partial bail they soft-skip (retrieve-only) and keep packing.
        # Hard gate (len mismatch) still aborts the step.
        pack_dbg = self._blend_pack_debug
        n_sched = len(metadata.requests)
        n_no_loadspec = 0
        # Partial bail (LMCACHE_PARTIAL_BAIL=1, default ON): soft gates drop ONLY
        # the offending request (retrieve-only / already-fetched) and keep packing
        # the rest. Old behaviour aborted the WHOLE step to serial on any gate --
        # one encoder-cache miss cost batching for every co-scheduled request.
        # Hard gate (len mismatch) still aborts: that is data corruption.
        _partial_bail = os.environ.get("LMCACHE_PARTIAL_BAIL", "1") == "1"
        # Test hook: soft-skip the first eligible request of a multi-request step
        # so equality/timing can exercise the path without waiting for a real
        # encoder eviction. No-op when PARTIAL_BAIL=0 (falls through to _bail).
        _force_skip = os.environ.get("LMCACHE_PARTIAL_BAIL_FORCE_SKIP", "0") == "1"
        _force_skip_done = False
        n_soft_skip = 0

        def _bail(reason: str, req_idx: int) -> bool:
            """Uniform abort: log why the whole step left the batched path."""
            if pack_dbg:
                logger.info(
                    "[blend-pack] ABORT to serial: scheduled=%d no_load_spec=%d "
                    "collected=%d soft_skip=%d, request #%d hit gate '%s'",
                    n_sched, n_no_loadspec, len(requests_info), n_soft_skip,
                    req_idx, reason,
                )
            return False

        requests_info = []
        chunk = self._lmcache_chunk_size

        def _retrieve_only(tokens, c, token_mask, slot_mapping) -> None:
            """Drain a plain layerwise retrieve for a soft-skipped request.

            Used when embeds are unavailable (cannot CodecSight-recompute) or the
            window is too short to blend. KV still lands in the paged cache so
            the request is correct; it just does not join blend_batched. Fully
            drained here (sync) so we do not mix deferred retrievers with the
            packed path that returns handled=True and skips the serial loop.
            """
            retr = self.lmcache_engine.retrieve_layer(
                tokens[:c], token_mask[:c], kvcaches=kvcaches,
                slot_mapping=slot_mapping[:c], sync=True,
            )
            for _ in retr:
                pass

        def _soft_skip(reason: str, req_idx: int, *,
                       tokens=None, c=None, token_mask=None, slot_mapping=None,
                       need_retrieve: bool = False) -> bool:
            """Drop this request from the pack. Returns True if soft-skip engaged.

            When PARTIAL_BAIL is off, returns False so the caller can _bail.
            """
            nonlocal n_soft_skip
            if not _partial_bail:
                return False
            n_soft_skip += 1
            if pack_dbg:
                logger.info(
                    "[blend-pack] SOFT-SKIP req #%d '%s' (need_retrieve=%s); "
                    "continuing pack (collected=%d soft_skip=%d scheduled=%d)",
                    req_idx, reason, need_retrieve,
                    len(requests_info), n_soft_skip, n_sched,
                )
            if need_retrieve:
                assert tokens is not None and c is not None
                assert token_mask is not None and slot_mapping is not None
                _retrieve_only(tokens, c, token_mask, slot_mapping)
            return True

        def _fallback_fetch_and_select(request, tokens, c, token_mask, slot_mapping, embeds) -> bool:
            """Per-request eager fetch+select, used ONLY as the coalesced-fetch
            overflow fallback (see the token-budget pre-check in Pass 1.5) --
            always eager (coalesced_fetch requires overlap=False), so unlike
            the Pass-1 inline branch this never needs the overlap/fetch_gens
            case. Returns False (no append) when selection finds 0 anchors,
            mirroring the `_bail` trigger condition used elsewhere.
            """
            _s = _evt() if timing else None
            retr = self.lmcache_engine.retrieve_layer(
                tokens[:c], token_mask[:c], kvcaches=kvcaches,
                slot_mapping=slot_mapping[:c], sync=True,
            )
            for _ in retr:
                pass
            if timing:
                spans["fetch"].append((_s, _evt()))

            _s = _evt() if timing else None
            md = LMCBlendMetadata(imp_indices=None, attn_mask=None, positions=None)
            md.tokens_per_frame = int(request.tokens_per_frame or 0)
            md.mm_positions = request.mm_positions
            md.image_grid_thw = request.image_grid_thw
            md.input_ids = list(tokens[:c])
            blender._active_metadata = md

            dev = slot_mapping.device
            hit = blender._compute_hit_indices(c, dev)
            anchor_local = blender._codecsight_select(hit, c, dev)
            if anchor_local.numel() == 0:
                return False

            if blender.is_mrope and blender._mrope_model_config is not None:
                positions_full = blender._compute_mrope_positions(c, dev)
                positions = positions_full[:, anchor_local]
            else:
                positions = torch.arange(c, device=dev, dtype=torch.int64)[anchor_local]

            requests_info.append({
                "req_id": request.req_id,
                "prefix_len": int(c),
                "anchor_embeds": embeds[anchor_local],
                "positions": positions,
                "slot_full": slot_mapping[:c],
                "anchor_local": anchor_local,
            })
            if timing:
                spans["select"].append((_s, _evt()))
            return True
        # Level-2 overlap: collect per-request fetch generators (warmed up, not
        # drained) so the deferred driver can step them per-layer. The side
        # stream must exist before we prime fetches on it (mirrors serial path).
        # Fix C-fix: once a deferred driver has failed to finish cleanly we stop
        # using the overlap path entirely. Eager is slower but always correct and
        # cannot wedge the engine.
        overlap = self._batched_overlap and not self._overlap_degraded

        # Fix C-fix (2): BUFFER-POOL ADMISSION CONTROL -- the actual deadlock cause.
        # gpu_connector.batched_to_gpu allocates TWO staging buffers per generator
        # and only releases them at the very END of the generator
        # (load/compute_gpu_buffer_obj.ref_count_down()). Eager drains the generator
        # inside one call, so exactly one is ever live. Overlap keeps generators
        # alive ACROSS engine steps, so buffers accumulate at 2 per in-flight blend.
        # The pool is one layer of the whole KV cache (76,992 tokens here), so with
        # p95 windows of ~14,160 tokens only floor(76992 / (2*14160)) = 2 fit and the
        # THIRD blend exhausts it -- which is exactly where jobs 15000264 and
        # 15001730 both wedged (3rd deferred blend, both TP ranks, no traceback).
        # Cap the number of concurrent deferred drivers and send the overflow down
        # the eager path: correctness is identical either way, and the eager path
        # frees its buffers immediately.
        if overlap:
            cap = int(os.environ.get("LMCACHE_MAX_DEFERRED_BLENDS", "2"))
            if cap >= 0 and len(self.layerwise_blenders) >= cap:
                logger.info(
                    "[blend-pack] %d deferred blend(s) already in flight (cap=%d); "
                    "using the eager path for this step to stay inside the GPU "
                    "staging-buffer pool.",
                    len(self.layerwise_blenders), cap,
                )
                overlap = False
        fetch_gens: list = []
        if overlap and self._blend_stream is None:
            self._blend_stream = torch.cuda.Stream()

        # B1 (4/4): coalesced multi-request fetch. Only wired for the EAGER
        # path -- overlap/deferred (C) is a separate, currently-deadlocking
        # optimization and is not built on top of here. IMPORTANT: when this
        # is off (default), the loop below is BYTE-IDENTICAL in structure to
        # before -- fetch and select stay inline, back-to-back, per request --
        # because `_compute_hit_indices` (blender.py) reads gap positions off
        # a single shared mutable `self.gpu_connector.current_gap_positions`
        # that each request's fetch overwrites; splitting fetch and select
        # into two full passes over ALL requests would make every selection
        # read the LAST request's gap positions instead of its own. Only the
        # coalesced branch defers selection to Pass 2, and only it needs to
        # slice the coalesced gap positions back into each request's own
        # local coordinates first (done below via
        # `current_gap_positions_per_request`).
        coalesced_fetch = (
            not overlap
            and os.environ.get("LMCACHE_COALESCED_FETCH", "0") == "1"
        )

        pending: list = []
        for req_idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                n_no_loadspec += 1
                continue
            tokens = request.token_ids
            slot_mapping = request.slot_mapping.cuda()
            if len(tokens) != len(slot_mapping):
                # Hard gate: structural inconsistency -- abort the step.
                return _bail("len(tokens) != len(slot_mapping)", req_idx)
            c = min(request.load_spec.lmcache_cached_tokens, len(tokens))
            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked = request.load_spec.vllm_cached_tokens // chunk * chunk
            token_mask[:masked] = False

            if c < 128:  # _MIN_BLEND_TOKENS: too short to blend
                if _soft_skip(
                    f"cached_tokens {c} < 128", req_idx,
                    tokens=tokens, c=c, token_mask=token_mask,
                    slot_mapping=slot_mapping, need_retrieve=True,
                ):
                    continue
                return _bail(f"cached_tokens {c} < 128", req_idx)

            # Test hook: drop the first eligible request of a multi-request step
            # so partial-bail can be exercised without a real encoder eviction.
            if (
                _force_skip and not _force_skip_done and n_sched >= 2
                and (n_sched - n_no_loadspec) >= 2
            ):
                _force_skip_done = True
                if _soft_skip(
                    "force_skip_test", req_idx,
                    tokens=tokens, c=c, token_mask=token_mask,
                    slot_mapping=slot_mapping, need_retrieve=True,
                ):
                    continue
                return _bail("force_skip_test", req_idx)

            _s = _evt() if timing else None
            embeds, _deepstack = self._reconstruct_inputs_embeds(
                tokens, request.mm_hashes, request.mm_positions, c,
            )
            if timing:
                spans["embed"].append((_s, _evt()))
            if embeds is None:
                # Encoder cache miss: cannot CodecSight-recompute this req, but
                # other co-scheduled reqs can still form a pack.
                if _soft_skip(
                    "encoder cache miss (embeds is None)", req_idx,
                    tokens=tokens, c=c, token_mask=token_mask,
                    slot_mapping=slot_mapping, need_retrieve=True,
                ):
                    continue
                return _bail("encoder cache miss (embeds is None)", req_idx)

            if coalesced_fetch:
                # Fetch AND select are both deferred -- see Pass 1.5 / Pass 2
                # below. Nothing else about this request's gating changes.
                pending.append({
                    "request": request, "tokens": tokens, "c": c,
                    "token_mask": token_mask, "slot_mapping": slot_mapping,
                    "embeds": embeds, "req_idx": req_idx,
                })
                continue

            # Phase 1: plain retrieve -> paged cache (load + RoPE-correct).
            # gap_positions are set on the connector at the FIRST next (warmup),
            # before any layer is sent, so selection below is correct after a
            # single prime. Eager mode drains fully here (blocking). Overlap mode
            # primes ONCE (warmup) on the side stream and KEEPS the generator so
            # wait_for_layer_load can drain it per-layer, overlapped with prefill.
            _s = _evt() if timing else None
            retr = self.lmcache_engine.retrieve_layer(
                tokens[:c], token_mask[:c], kvcaches=kvcaches,
                slot_mapping=slot_mapping[:c], sync=not overlap,
            )
            if overlap:
                with torch.cuda.stream(self._blend_stream):
                    next(retr)  # warmup: sets gap_positions, loads layer 0
                fetch_gens.append(retr)
            else:
                for _ in retr:
                    pass
            if timing:
                spans["fetch"].append((_s, _evt()))

            # Selection: install this request's metadata, run codec I-frame pick.
            _s = _evt() if timing else None
            md = LMCBlendMetadata(imp_indices=None, attn_mask=None, positions=None)
            md.tokens_per_frame = int(request.tokens_per_frame or 0)
            md.mm_positions = request.mm_positions
            md.image_grid_thw = request.image_grid_thw
            md.input_ids = list(tokens[:c])
            blender._active_metadata = md

            dev = slot_mapping.device
            hit = blender._compute_hit_indices(c, dev)
            anchor_local = blender._codecsight_select(hit, c, dev)
            if anchor_local.numel() == 0:
                # Eager: fetch already drained into paged KV -- omit from pack.
                # Overlap: fetch_gens must stay 1:1 with requests_info (we already
                # primed this req onto the side stream), so abort rather than
                # soft-skip (orphan generators deadlock the deferred driver).
                if overlap:
                    return _bail("codecsight selected 0 anchors", req_idx)
                if _soft_skip("codecsight selected 0 anchors", req_idx):
                    continue
                return _bail("codecsight selected 0 anchors", req_idx)

            if blender.is_mrope and blender._mrope_model_config is not None:
                positions_full = blender._compute_mrope_positions(c, dev)
                positions = positions_full[:, anchor_local]
            else:
                positions = torch.arange(c, device=dev, dtype=torch.int64)[anchor_local]

            requests_info.append({
                "req_id": request.req_id,
                "prefix_len": int(c),
                "anchor_embeds": embeds[anchor_local],
                "positions": positions,
                "slot_full": slot_mapping[:c],
                "anchor_local": anchor_local,
            })
            if timing:
                spans["select"].append((_s, _evt()))

        # Pass 1.5 (coalesced_fetch only): the ONE fetch across every request
        # collected above. `pending` is empty here whenever coalesced_fetch is
        # off, since every request took the inline branch above instead.
        per_request_gaps = None
        if coalesced_fetch and pending:
            # DEFENSE-IN-DEPTH (added after job 15005984): batched_to_gpu_multi
            # needs TWO staging buffers alive at once, each sized to the SUM of
            # every pending request's tokens -- unlike the per-request path,
            # which only ever holds one small, single-request-sized buffer
            # pair. A burst of large windows packed together can need more
            # than the pool holds; that job hit exactly this
            # (AssertionError: Failed to allocate GPU buffer -> EngineDeadError
            # -> every later request failed). LMCACHE_COALESCED_BUFFER_MULT
            # (gpu_connector.py) sizes the pool for a KNOWN worst case, but
            # that's an assumption about window sizes, not a guarantee -- so
            # check the ACTUAL pending group against the pool's own advertised
            # budget and fall back to the old per-request path (still correct,
            # just not coalesced) instead of trusting the assumption blindly.
            # `gconn` is the same gpu_connector fetched at the top of this
            # function (for the fetch_syncs telemetry).
            if getattr(gconn, "gpu_buffer_allocator", None) is None:
                # First coalesced call in this server's lifetime -- the pool
                # doesn't exist yet (it's created lazily on first use), so
                # max_coalesced_tokens isn't set either. Initialize it now
                # (idempotent, cheap) so the check below has a real budget to
                # compare against instead of skipping unchecked.
                gconn._lazy_initialize_buffer(kvcaches)
            max_safe = getattr(gconn, "max_coalesced_tokens", None)

            # Gate 2 (added after job 15081983): every request in a coalesced
            # group must contribute >= 1 retrievable chunk. retrieve_layer_multi
            # asserts this and has no per-request fallback once it starts, and
            # that assert runs inside the worker -- so one unretrievable request
            # does not degrade the step, it kills the EngineCore
            # (EngineDeadError at N=8, which then took out the N=10 and N=12
            # phases as downstream fallout). Common on category-diverse
            # workloads, where a window's KV may simply not be cached yet;
            # invisible on the single-video set every earlier run used.
            #
            # FILTER, don't bail: drop only the zero-chunk requests to the
            # per-request path and still coalesce the rest, so one cache miss
            # costs that request's coalescing rather than the whole step's.
            coalescable, misses = [], []
            for p in pending:
                (coalescable if self.lmcache_engine.has_retrievable_chunk(
                    p["tokens"][: p["c"]], p["token_mask"][: p["c"]])
                 else misses).append(p)

            if misses:
                if pack_dbg:
                    logger.info(
                        "[blend-pack] %d of %d pending request(s) contribute no "
                        "retrievable chunk (req_idx=%s); handling those per-request "
                        "and coalescing the remaining %d.",
                        len(misses), len(pending), [p["req_idx"] for p in misses],
                        len(coalescable),
                    )
                for p in misses:
                    if not _fallback_fetch_and_select(
                        p["request"], p["tokens"], p["c"], p["token_mask"],
                        p["slot_mapping"], p["embeds"],
                    ):
                        # Fetch ran; 0 anchors -- soft-skip keeps the pack.
                        if _soft_skip(
                            "coalesced-fallback (no chunk): 0 anchors",
                            p["req_idx"],
                        ):
                            continue
                        return _bail(
                            "coalesced-fallback (no retrievable chunk): "
                            "codecsight selected 0 anchors",
                            p["req_idx"],
                        )
                # _fallback_fetch_and_select already did BOTH fetch and select
                # for these and appended their requests_info entry, so they must
                # not be revisited by Pass 2. Narrowing `pending` to the
                # coalesced group also keeps it index-aligned with
                # `per_request_gaps`, which Pass 2 slices positionally.
                pending = coalescable

            est_tokens = sum(int(p["token_mask"][: p["c"]].sum()) for p in pending)

            if not pending:
                # Every request was a miss; they were all handled inline above.
                # Calling retrieve_layer_multi with an empty list would trip its
                # own n_reqs > 0 assert.
                pass
            elif max_safe is not None and est_tokens > max_safe:
                if pack_dbg:
                    logger.info(
                        "[blend-pack] coalesced fetch needs ~%d tokens > pool "
                        "budget %d; falling back to per-request fetch for "
                        "this step's %d pending request(s).",
                        est_tokens, max_safe, len(pending),
                    )
                for p in pending:
                    if not _fallback_fetch_and_select(
                        p["request"], p["tokens"], p["c"], p["token_mask"],
                        p["slot_mapping"], p["embeds"],
                    ):
                        if _soft_skip(
                            "coalesced-fallback: 0 anchors", p["req_idx"],
                        ):
                            continue
                        return _bail(
                            "coalesced-fallback: codecsight selected 0 anchors",
                            p["req_idx"],
                        )
                pending = []  # handled inline above; Pass 2 below is then a no-op
            else:
                _s = _evt() if timing else None
                retr_multi = self.lmcache_engine.retrieve_layer_multi(
                    [p["tokens"][: p["c"]] for p in pending],
                    [p["token_mask"][: p["c"]] for p in pending],
                    [p["slot_mapping"][: p["c"]] for p in pending],
                    kvcaches=kvcaches,
                )
                for _ in retr_multi:
                    pass
                if timing:
                    spans["fetch"].append((_s, _evt()))
                per_request_gaps = getattr(
                    gconn, "current_gap_positions_per_request", None,
                )
                assert per_request_gaps is not None and len(per_request_gaps) == len(pending), (
                    "retrieve_layer_multi did not populate "
                    "current_gap_positions_per_request for every pending request."
                )

        # Pass 2 (coalesced_fetch only): selection, per request, deferred from
        # above because it needed the coalesced fetch to finish first. Same
        # selection logic as the inline branch; the only difference is that
        # `current_gap_positions` is swapped in per request from the sliced
        # coalesced result instead of having just been set by that request's
        # own retrieve_layer call.
        for i, p in enumerate(pending):
            request, tokens, c, slot_mapping, embeds, req_idx = (
                p["request"], p["tokens"], p["c"], p["slot_mapping"],
                p["embeds"], p["req_idx"],
            )

            self.lmcache_engine.gpu_connector.current_gap_positions = (
                per_request_gaps[i]
            )

            _s = _evt() if timing else None
            md = LMCBlendMetadata(imp_indices=None, attn_mask=None, positions=None)
            md.tokens_per_frame = int(request.tokens_per_frame or 0)
            md.mm_positions = request.mm_positions
            md.image_grid_thw = request.image_grid_thw
            md.input_ids = list(tokens[:c])
            blender._active_metadata = md

            dev = slot_mapping.device
            hit = blender._compute_hit_indices(c, dev)
            anchor_local = blender._codecsight_select(hit, c, dev)
            if anchor_local.numel() == 0:
                if _soft_skip("codecsight selected 0 anchors", req_idx):
                    continue
                return _bail("codecsight selected 0 anchors", req_idx)

            if blender.is_mrope and blender._mrope_model_config is not None:
                positions_full = blender._compute_mrope_positions(c, dev)
                positions = positions_full[:, anchor_local]
            else:
                positions = torch.arange(c, device=dev, dtype=torch.int64)[anchor_local]

            requests_info.append({
                "req_id": request.req_id,
                "prefix_len": int(c),
                "anchor_embeds": embeds[anchor_local],
                "positions": positions,
                "slot_full": slot_mapping[:c],
                "anchor_local": anchor_local,
            })
            if timing:
                spans["select"].append((_s, _evt()))

        if not requests_info:
            if n_soft_skip > 0:
                # Every eligible request was soft-skipped and already retrieve-
                # only (or fetch-only) handled above. Returning False would
                # fall into the serial loop and load/blend them a second time.
                if pack_dbg:
                    logger.info(
                        "[blend-pack] all %d soft-skipped (scheduled=%d "
                        "no_load_spec=%d); nothing to pack, already handled",
                        n_soft_skip, n_sched, n_no_loadspec,
                    )
                return True
            if pack_dbg:
                logger.info(
                    "[blend-pack] no blendable requests: scheduled=%d "
                    "no_load_spec=%d soft_skip=%d -> serial path",
                    n_sched, n_no_loadspec, n_soft_skip,
                )
            return False

        # The answer to "why is N always 1": if scheduled==1 the scheduler never
        # co-located two blendable requests in a step (a stagger/admission
        # problem, upstream of this code); if scheduled>1 but packed==1 the loss
        # is here in the gates above.
        if pack_dbg:
            logger.info(
                "[blend-pack] scheduled=%d no_load_spec=%d soft_skip=%d -> "
                "packed N=%d (anchors=%d) mode=%s",
                n_sched, n_no_loadspec, n_soft_skip, len(requests_info),
                sum(int(r["anchor_local"].numel()) for r in requests_info),
                "deferred/overlap" if overlap else "eager",
            )

        if overlap:
            # Level-2: deferred batched blend.
            # Default (no AR mux): FETCH on `_blend_stream`, RECOMPUTE on current
            # (TP all-reduces must not share a communicator across streams —
            # Fix C hang, jobs 15002378 / 15099974).
            # With LMCACHE_AR_MUX=1: ARs are remuxed onto a dedicated stream, so
            # wait_for_layer_load may also run RECOMPUTE on `_blend_stream` and
            # overlap local blend GEMMs with prefill. Prime still finishes
            # recompute(0) on the current stream before the forward starts.
            # No [blend-timing] here: work finishes during the forward via
            # wait_for_layer_load.
            driver = blender.blend_batched(
                requests_info, kvcaches,
                fetch_gens=fetch_gens, defer=True,
            )
            assert isinstance(driver, DeferredBatchedBlendDriver)

            # ISOLATION EXPERIMENT (LMCACHE_DEFER_DRAIN_EAGER=1): deferred CODE
            # with EAGER timing — drain fully here, no wait_for_layer_load.
            if os.environ.get("LMCACHE_DEFER_DRAIN_EAGER", "0") == "1":
                logger.info(
                    "[blend-pack] LMCACHE_DEFER_DRAIN_EAGER=1: draining the deferred "
                    "driver in place (deferred code path, eager timing)."
                )
                for _ in range(self.num_layers):
                    next(driver)
                driver.close()
                if self._blend_stream is not None:
                    torch.cuda.current_stream().wait_stream(self._blend_stream)
                return True

            # Prime: fetch(0) on side stream, recompute(0) on current (layer-0 KV
            # must be ready before forward), fetch(1) on side for wait_for_layer_load.
            with torch.cuda.stream(self._blend_stream):
                driver.step_fetch()
            torch.cuda.current_stream().wait_stream(self._blend_stream)
            driver.step_recompute()
            if not driver.done:
                with torch.cuda.stream(self._blend_stream):
                    driver.step_fetch()
            self.layerwise_blenders.append(driver)
            return True

        # Phase 2: either stash for runner-merged prefill (LMCACHE_MERGED_BLEND)
        # or run the classic packed recompute forward.
        anchors = sum(int(r["anchor_local"].numel()) for r in requests_info)
        n = len(requests_info)
        # skip_ffn zeroes the FFN on recompute rows only; the merged forward
        # runs the real mlp on anchor rows and cannot reproduce that. Refuse
        # to merge rather than silently change the skip_ffn semantics.
        merged_ok = self._merged_blend and not bool(
            getattr(blender, "skip_ffn", False))
        if self._merged_blend and not merged_ok:
            logger.warning(
                "[merged-blend] skip_ffn=True is incompatible with the merged "
                "path; using classic blend_batched for this step."
            )
        if merged_ok:
            # Fetch+select done; recompute will be folded into the upcoming
            # vLLM prefill by gpu_model_runner._apply_merged_blend_plan.
            # Keep kvcaches so a runner-side fallback can still blend_batched.
            self._merged_blend_plan = {
                "requests": requests_info,
                "kvcaches": kvcaches,
            }
            if timing:
                torch.cuda.synchronize()

                def _sum_m(key):
                    return sum(a.elapsed_time(b) for a, b in spans[key])

                t_embed = _sum_m("embed")
                t_fetch = _sum_m("fetch")
                t_select = _sum_m("select")
                total = t_embed + t_fetch + t_select
                n_sync = getattr(gconn, "global_sync_count", 0) - sync0
                logger.info(
                    "[blend-timing] merged-stash N=%d anchors=%d | embed=%.2fms "
                    "fetch=%.2fms select=%.2fms recompute=0.00ms (deferred to "
                    "prefill) | total=%.2fms fetch_syncs=%d",
                    n, anchors, t_embed, t_fetch, t_select, total, n_sync,
                )
            else:
                logger.info(
                    "[merged-blend] stashed N=%d anchors=%d for runner merge",
                    n, anchors,
                )
            return True

        # Phase 2 (eager): one packed forward for all requests (KV REFRESH).
        _s = _evt() if timing else None
        blender.blend_batched(requests_info, kvcaches)
        if timing:
            _e = _evt()
            # The ONLY device sync: makes every recorded event complete so
            # elapsed_time() can read it. Runs AFTER all measured work, so it
            # does not serialize the phases being timed.
            torch.cuda.synchronize()

            def _sum(key):  # ms; elapsed_time already returns milliseconds
                return sum(a.elapsed_time(b) for a, b in spans[key])

            t_embed = _sum("embed")
            t_fetch = _sum("fetch")
            t_select = _sum("select")
            t_recompute = _s.elapsed_time(_e)
            total = t_embed + t_fetch + t_select + t_recompute
            n_sync = getattr(gconn, "global_sync_count", 0) - sync0
            logger.info(
                "[blend-timing] batched N=%d anchors=%d | embed=%.2fms "
                "fetch=%.2fms select=%.2fms recompute=%.2fms | total=%.2fms "
                "(fetch=%.0f%% recompute=%.0f%%) fetch_syncs=%d",
                n, anchors, t_embed, t_fetch, t_select,
                t_recompute, total,
                100.0 * t_fetch / total if total else 0.0,
                100.0 * t_recompute / total if total else 0.0,
                n_sync,
            )
        return True

    def take_merged_blend_plan(self):
        """Return the stashed merged-blend plan (or None). Does not clear.

        Cleared by ``clear_merged_blend_plan`` after the runner applies it or
        after ``run_merged_blend_recompute_fallback``.
        """
        return self._merged_blend_plan

    def clear_merged_blend_plan(self) -> None:
        self._merged_blend_plan = None

    def run_merged_blend_recompute_fallback(self) -> bool:
        """Run classic ``blend_batched`` on the stashed plan (runner could not
        append pseudo-segments). Returns True if a recompute ran."""
        plan = self._merged_blend_plan
        self._merged_blend_plan = None
        if not plan or not plan.get("requests"):
            return False
        blender = self.blender
        if blender is None:
            return False
        logger.warning(
            "[merged-blend] runner could not apply plan; falling back to "
            "eager blend_batched (N=%d)",
            len(plan["requests"]),
        )
        blender.blend_batched(plan["requests"], plan["kvcaches"])
        return True

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None

        self.lmcache_engine.post_init(kvcaches=kvcaches)

        self.layerwise_retrievers = []
        # Fix C-fix: NEVER drop deferred blend drivers on the floor. See
        # _drain_layerwise_blenders for why the old bare reassignment deadlocked
        # the engine at N>=4 (job 15000264).
        self._drain_layerwise_blenders("new engine step")

        # Serial per-request blend timing (LMCACHE_BLEND_TIMING=1), the
        # non-batched analogue of _batched_blend_load_kv's [blend-timing]
        # line. self.blender.blend(defer=self._codecsight_pipeline) only
        # returns a deferred per-layer generator when _codecsight_pipeline is
        # set (VLLM_CODECSIGHT_PIPELINE=1); that env var is NOT set in any of
        # the benchmark qsub scripts, so defer=False and blend() instead
        # drains the whole per-layer generator SYNCHRONOUSLY right there
        # (blender.py:1066-1067, `for _ in range(num_layers+2): next(...)`),
        # exactly like the batched path's eager blend_batched() call -- so it
        # can be bracketed the same way, around the call site below. If
        # _codecsight_pipeline IS set, only the two priming next() calls run
        # eagerly and the rest happens later in wait_for_layer_load, which
        # this bracket does NOT cover -- see the warning at the flush site.
        if self._blend_timing:
            self._serial_blend_n = 0
            self._serial_blend_embed_evts = []
            self._serial_blend_layer_evts = []
            self._serial_blend_fetch_evts = []
            self._serial_blend_recompute_evts = []
            self._serial_blend_select_ms = 0.0

        # Tier-2: batched selective recompute. Eligible only for the eager
        # codecsight blend path (the validated batch-safe mode). Falls through
        # to the serial loop on any miss so behavior is never silently degraded.
        if (
            self._batched_blend
            and self.use_layerwise
            and self.enable_blending
            and not self._codecsight_pipeline
            and getattr(self.blender, "blend_mode", "") == "codecsight"
        ):
            handled = self._batched_blend_load_kv(
                metadata, kvcaches, attn_metadata,
            )
            if handled:
                return

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                logger.debug("skip request due to load spec is None")
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.cuda()
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            # Clamp to the request's ACTUAL available tokens. Under high-N KV
            # pressure load_spec.lmcache_cached_tokens can exceed len(tokens)
            # (part of the cached prefix was preempted/evicted), and every
            # downstream slice [:lmcache_cached_tokens] then overran the shorter
            # tensors -> EngineCore crash (empty min() at N>=16; partial-prefix
            # IndexError at N=12). Clamping keeps the blend CONSISTENT on the
            # surviving prefix (reuse what's cached, let vLLM recompute the
            # evicted tail) instead of aborting -- preserves throughput at high N.
            # No-op in the normal case (cached prefix <= full request length).
            lmcache_cached_tokens = min(
                request.load_spec.lmcache_cached_tokens, len(tokens)
            )
            logger.debug(f"enter self.enable_blending {self.enable_blending}, self.use_layerwise {self.use_layerwise}")
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                logger.debug(f"self.enable_blending {self.enable_blending}")
                if self.enable_blending:
                    self._ensure_blender_initialized()
                    if self.blender is None:
                        logger.warning(
                            "Blender unavailable; falling back to layerwise retrieve."
                        )
                        self.enable_blending = False
                        layerwise_retriever = self.lmcache_engine.retrieve_layer(
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            kvcaches=kvcaches,
                            slot_mapping=slot_mapping[:lmcache_cached_tokens],
                            sync=sync,
                        )
                        next(layerwise_retriever)
                        next(layerwise_retriever)
                        self.layerwise_retrievers.append(layerwise_retriever)
                        continue

                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    page_stream = self.lmcache_engine.gpu_connector.get_page_stream()

                    skip_embeds = (
                        getattr(self.blender, "blend_mode", "") == "direct_reuse"
                        and getattr(
                            self.blender, "direct_reuse_retrieve_only", False)
                    )
                    if skip_embeds:
                        inputs_embeds, deepstack_input_embeds = None, None
                    else:
                        _blend_t_evt = None
                        if self._blend_timing:
                            _blend_t_evt = torch.cuda.Event(enable_timing=True)
                            _blend_t_evt.record()
                        inputs_embeds, deepstack_input_embeds = (
                            self._reconstruct_inputs_embeds(
                                tokens, request.mm_hashes,
                                request.mm_positions, lmcache_cached_tokens,
                            )
                        )
                        if self._blend_timing:
                            _blend_t_end = torch.cuda.Event(enable_timing=True)
                            _blend_t_end.record()
                            self._serial_blend_embed_evts.append(
                                (_blend_t_evt, _blend_t_end)
                            )

                    if inputs_embeds is None and not skip_embeds:
                        logger.warning(
                            "inputs_embeds unavailable (encoder_cache "
                            "eviction); falling back to layerwise "
                            "retrieval for this request"
                        )
                        # This request never reaches blend(), so drop the embed
                        # span just recorded for it -- keeping it would charge
                        # embed time to a step whose reported N excludes this
                        # request.
                        if self._blend_timing and self._serial_blend_embed_evts:
                            self._serial_blend_embed_evts.pop()
                        layerwise_retriever = \
                            self.lmcache_engine.retrieve_layer(
                                tokens[:lmcache_cached_tokens],
                                token_mask[:lmcache_cached_tokens],
                                kvcaches=kvcaches,
                                slot_mapping=slot_mapping[
                                    :lmcache_cached_tokens],
                                sync=sync,
                            )
                        next(layerwise_retriever)
                        next(layerwise_retriever)
                        self.layerwise_retrievers.append(
                            layerwise_retriever)
                        continue

                    logger.debug(
                        "start_load_kv: inputs_embeds=%s, "
                        "deepstack=%s, mm_hashes=%d, mm_positions=%d, "
                        "cached_tokens=%d",
                        inputs_embeds.shape if inputs_embeds is not None else None,
                        deepstack_input_embeds.shape if deepstack_input_embeds is not None else None,
                        len(request.mm_hashes) if request.mm_hashes else 0,
                        len(request.mm_positions) if request.mm_positions else 0,
                        lmcache_cached_tokens,
                    )

                    if self._async_overlap and self._blend_stream is None:
                        self._blend_stream = torch.cuda.Stream()
                    # Async overlap: prime the deferred blender ON the side
                    # stream so layer-0's blend runs there (one layer ahead).
                    _blend_ctx = (
                        torch.cuda.stream(self._blend_stream)
                        if self._async_overlap
                        else contextlib.nullcontext()
                    )
                    _blend_l_evt = None
                    if self._blend_timing:
                        _blend_l_evt = torch.cuda.Event(enable_timing=True)
                        _blend_l_evt.record()
                    with _blend_ctx:
                        deferred_blender = self.blender.blend(
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            defer=self._codecsight_pipeline,
                            kvcaches=kvcaches,
                            slot_mapping=slot_mapping[:lmcache_cached_tokens],
                            tokens_per_frame=request.tokens_per_frame,
                            mm_positions=request.mm_positions,
                            image_grid_thw=request.image_grid_thw,
                            page_stream=page_stream,
                            sync=sync,
                            inputs_embeds=inputs_embeds,
                            deepstack_input_embeds=deepstack_input_embeds,
                        )
                    if self._blend_timing:
                        # Only a complete measurement when defer=False (the
                        # eager path -- see the reset-block comment above):
                        # blend() has then already drained every layer
                        # synchronously, so this span covers the whole
                        # request. When _codecsight_pipeline forces defer=
                        # True, this span covers only the 2-step primer and
                        # UNDER-reports -- flagged at the flush site instead
                        # of silently mixing complete and partial spans.
                        _blend_l_end = torch.cuda.Event(enable_timing=True)
                        _blend_l_end.record()
                        self._serial_blend_layer_evts.append(
                            (_blend_l_evt, _blend_l_end, deferred_blender is not None)
                        )
                        self._serial_blend_n += 1
                        # Pull per-phase fetch/recompute/select out of the
                        # blender (eager path only -- deferred has not run
                        # the full layer loop yet).
                        if deferred_blender is None and self.blender is not None:
                            _tf, _tr, _ts = self.blender.take_serial_phase_timing()
                            self._serial_blend_fetch_evts.extend(_tf)
                            self._serial_blend_recompute_evts.extend(_tr)
                            self._serial_blend_select_ms += float(_ts)
                    if deferred_blender is not None:
                        # Pipelining A: step per-layer in wait_for_layer_load
                        # so the KV load overlaps prefill compute.
                        self.layerwise_blenders.append(deferred_blender)
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                    skip_contains_check=True,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!"
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

            self._stats_monitor.update_interval_vllm_hit_tokens(
                request.load_spec.vllm_cached_tokens
            )
            self._stats_monitor.update_interval_prompt_tokens(len(tokens))

        # Queue this step's serial blend timing (LMCACHE_BLEND_TIMING=1) --
        # the [blend-timing] serial counterpart of _batched_blend_load_kv's
        # line, aggregated over however many requests this step's per-request
        # loop actually blended (self._serial_blend_n). NOT read here: the
        # events are read a step later by _drain_serial_blend_timing, which
        # polls rather than syncing (see _serial_blend_pending in __init__ for
        # why no sync is added to this path).
        if self._blend_timing and self._serial_blend_n:
            self._serial_blend_pending.append(
                (self._serial_blend_n, self._serial_blend_embed_evts,
                 self._serial_blend_layer_evts,
                 self._serial_blend_fetch_evts,
                 self._serial_blend_recompute_evts,
                 self._serial_blend_select_ms)
            )
            self._serial_blend_n = 0
            self._serial_blend_embed_evts = []
            self._serial_blend_layer_evts = []
            self._serial_blend_fetch_evts = []
            self._serial_blend_recompute_evts = []
            self._serial_blend_select_ms = 0.0
        if self._blend_timing:
            self._drain_serial_blend_timing()

    def _drain_serial_blend_timing(self) -> None:
        """Log any queued serial [blend-timing] batches whose CUDA events have
        completed, WITHOUT synchronizing the device.

        Called at the end of every start_load_kv, so a batch queued by step k
        is normally read at step k+1, by which time its events are long done.
        Completion is polled with Event.query() -- a batch that is somehow not
        ready yet stays queued for the following step instead of blocking.
        Batches are read in order and reading stops at the first incomplete
        one, so the log stays chronological.

        The final step's batch is never read (nothing follows it to trigger a
        drain). That costs one blend step per phase out of the tens-to-hundreds
        each phase produces, and the analysis averages over the phase, so it
        is not worth a sync to recover.
        """
        while self._serial_blend_pending:
            item = self._serial_blend_pending[0]
            # Backward-compatible: old 3-tuples vs new 6-tuples with phases.
            if len(item) == 3:
                n, embed_evts, layer_evts = item
                fetch_evts, recompute_evts, select_ms = [], [], 0.0
            else:
                (n, embed_evts, layer_evts, fetch_evts,
                 recompute_evts, select_ms) = item
            last = (layer_evts or embed_evts or fetch_evts or recompute_evts)[-1][1]
            if not last.query():
                break                      # not finished; try again next step
            self._serial_blend_pending.pop(0)
            t_embed = sum(a.elapsed_time(b) for a, b in embed_evts)
            t_blend = sum(a.elapsed_time(b) for a, b, _ in layer_evts)
            partial = any(deferred for _, _, deferred in layer_evts)
            t_fetch = sum(a.elapsed_time(b) for a, b in fetch_evts) if fetch_evts else None
            t_recompute = (
                sum(a.elapsed_time(b) for a, b in recompute_evts)
                if recompute_evts else None
            )
            total = t_embed + t_blend
            if t_fetch is not None and t_recompute is not None and not partial:
                # Match batched line shape. select is CPU wall time nested in
                # the first recompute layer — reported but not added again.
                logger.info(
                    "[blend-timing] serial N=%d | embed=%.2fms fetch=%.2fms "
                    "select=%.2fms recompute=%.2fms | total=%.2fms "
                    "(fetch=%.0f%% recompute=%.0f%%; select⊂recompute, "
                    "not double-counted)",
                    n, t_embed, t_fetch, select_ms, t_recompute, total,
                    100.0 * t_fetch / total if total else 0.0,
                    100.0 * t_recompute / total if total else 0.0,
                )
            else:
                logger.info(
                    "[blend-timing] serial N=%d | embed=%.2fms blend=%.2fms | "
                    "total=%.2fms%s",
                    n, t_embed, t_blend, total,
                    " (PARTIAL -- _codecsight_pipeline deferred >=1 request; "
                    "blend span covers only the 2-step primer, not the full "
                    "per-layer cost, which ran later in wait_for_layer_load "
                    "and is NOT included here)" if partial else "",
                )

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def _step_blenders(self, blenders: list, fetch_stream=None) -> list:
        """Advance each deferred blend driver one layer; return those still alive.

        For ``DeferredBatchedBlendDriver`` (batched Fix C path):
          * Default (no AR mux): ``step_recompute()`` on the *current* stream
            (TP all-reduces); ``step_fetch()`` for the next layer on
            ``fetch_stream`` if given (memcpy/RoPE only).
          * With ``LMCACHE_AR_MUX=1`` and ``fetch_stream``: both recompute and
            next fetch run on ``fetch_stream`` without waiting afterward, so
            local blend GEMMs overlap the remainder of the current prefill
            layer. ARs are serialized on the mux stream. The next
            ``wait_for_layer_load`` does ``cur.wait_stream(fetch_stream)``
            before attention needs the blended KV.

        Legacy generator drivers still use a single ``next()`` (optionally under
        ``fetch_stream`` when the caller wrapped the call).
        """
        alive = []
        ar_mux = bool(getattr(self, "_ar_mux", False))
        for blender in blenders:
            try:
                if isinstance(blender, DeferredBatchedBlendDriver):
                    if blender.done:
                        blender.close()
                        self._blender_steps.pop(id(blender), None)
                        continue
                    if ar_mux and fetch_stream is not None:
                        # Full ahead step on blend stream; do not wait here.
                        with torch.cuda.stream(fetch_stream):
                            blender.step_recompute()
                            self._blender_steps[id(blender)] = (
                                self._blender_steps.get(id(blender), 0) + 1
                            )
                            if blender.done:
                                blender.close()
                                self._blender_steps.pop(id(blender), None)
                                continue
                            blender.step_fetch()
                        alive.append(blender)
                        continue
                    # RECOMPUTE on current stream — never on _blend_stream under TP
                    # unless AR mux is active (branch above).
                    blender.step_recompute()
                    self._blender_steps[id(blender)] = (
                        self._blender_steps.get(id(blender), 0) + 1
                    )
                    if blender.done:
                        blender.close()
                        self._blender_steps.pop(id(blender), None)
                        continue
                    # Ahead FETCH on side stream (overlaps subsequent prefill).
                    if fetch_stream is not None:
                        with torch.cuda.stream(fetch_stream):
                            blender.step_fetch()
                    else:
                        blender.step_fetch()
                    alive.append(blender)
                else:
                    if fetch_stream is not None:
                        with torch.cuda.stream(fetch_stream):
                            next(blender)
                    else:
                        next(blender)
                    self._blender_steps[id(blender)] = (
                        self._blender_steps.get(id(blender), 0) + 1
                    )
                    alive.append(blender)
            except StopIteration:
                self._blender_steps.pop(id(blender), None)
                try:
                    blender.close()
                except Exception:
                    logger.exception("Error closing finished blend driver")
            except Exception:
                logger.exception(
                    "Deferred blend driver raised mid-layer; disabling overlap and "
                    "continuing eagerly rather than killing the engine."
                )
                self._overlap_degraded = True
                self._blender_steps.pop(id(blender), None)
                try:
                    blender.close()
                except Exception:
                    pass
        return alive

    def _drain_layerwise_blenders(self, reason: str) -> None:
        """Finish (or safely discard) deferred blend drivers left from a prior step.

        THE BUG THIS FIXES (job 15000264, engine dead at N>=4):
        a driver is built in ``start_load_kv`` and expects to be advanced once per
        layer by ``wait_for_layer_load``. If a step does not drive it to completion
        -- chunked prefill splitting the request across steps, preemption, or simply
        the next ``start_load_kv`` arriving first -- the old code dropped it by
        reassigning ``self.layerwise_blenders = []``. The abandoned generator keeps
        half-finished work queued on ``self._blend_stream``, and the NEXT step's
        ``cur.wait_stream(self._blend_stream)`` then waits on work whose remainder
        nobody will ever enqueue: ``execute_model`` hangs, the RPC times out, and the
        engine dies. Observed exactly that way -- two deferred blends 0.64 s apart,
        then Running:0/Waiting:0 forever.

        Policy, in order of preference:
        1. Drive each leftover to completion on the blend stream (correct: the blend
           it was created for actually happens), then synchronize so nothing dangles.
        2. On ANY failure, close the generator, synchronize the stream anyway, and
           latch ``_overlap_degraded`` so every later step takes the eager path.
        Either way the stream is left clean, which is what stops the deadlock.
        """
        leftover = self.layerwise_blenders
        self.layerwise_blenders = []
        if not leftover:
            return

        # NOT a warning: the common case is a driver that completed its whole
        # num_layers contract and is merely un-exhausted. Only the genuinely-short
        # case below warrants a warning.
        #
        # INFO, not debug (promoted 2026-07-27). This used to be logger.debug, which
        # is not emitted at INFO, so NO run on record could show whether the drain
        # engaged at all -- and that is precisely the question the v1->v2->v3 drain
        # rewrites turn on. The per-driver driven counts are included because
        # "healthy (driven == num_layers) vs genuinely short" is the distinction that
        # decides which drain branch runs. Volume is ~1 line per deferred blend,
        # comparable to the existing [blend-pack] line.
        driven = []
        for d in leftover:
            if isinstance(d, DeferredBatchedBlendDriver):
                driven.append(d.recompute_steps)
            else:
                driven.append(self._blender_steps.get(id(d), 0))
        logger.info(
            "[blend-drain] retiring %d deferred blend driver(s) (%s); "
            "driven=%s of num_layers=%d%s",
            len(leftover),
            reason,
            driven,
            self.num_layers,
            "" if all(n >= self.num_layers for n in driven) else " SHORT",
        )
        stream_ctx = (
            torch.cuda.stream(self._blend_stream)
            if self._blend_stream is not None
            else contextlib.nullcontext()
        )
        for driver in leftover:
            # New Fix-C driver: finish any remaining layers on the CURRENT stream
            # (TP-safe), then close. No generator epilogue next().
            if isinstance(driver, DeferredBatchedBlendDriver):
                self._blender_steps.pop(id(driver), None)
                try:
                    if self._blend_stream is not None:
                        torch.cuda.current_stream().wait_stream(self._blend_stream)
                    while not driver.done:
                        driver.step_fetch()
                        driver.step_recompute()
                    driver.close()
                except Exception:
                    logger.exception(
                        "DeferredBatchedBlendDriver drain failed; disabling overlap."
                    )
                    self._overlap_degraded = True
                    try:
                        driver.close()
                    except Exception:
                        pass
                continue

            done = self._blender_steps.pop(id(driver), 0)
            remaining = self.num_layers - done
            if remaining <= 0:
                # Healthy: the forward already drove it num_layers times, which is
                # its whole contract, leaving it at the last yield but NOT exhausted.
                # v1 drove it up to num_layers+2 MORE times -> ran work it must not
                # run -> corrupted 7/18 outputs (15001729).
                # v2 called close() instead -> GeneratorExit at the last yield skips
                # the epilogue, and the epilogue is where batched_to_gpu does
                # load/compute_gpu_buffer_obj.ref_count_down() -> the two GPU staging
                # buffers LEAK, the pool drains, and the run dies even sooner
                # (15002006 died after only 1 deferred blend vs 3 in 15001730 --
                # both TP ranks log [blend-pack], so halve the grep count).
                # CAVEAT (2026-07-27): a 1-blend death is NOT quantitatively
                # explained by that leak (2 x 14,160 << the 76,992-token pool) and
                # 15002006 CRASHED rather than hung. See BLEND_OPT_IMPLEMENTATION.md
                # "C's LIVENESS IS STILL UNTESTED" -- re-run the N=4 repro on v3.
                # v3: advance EXACTLY ONCE. That runs the epilogue (freeing the
                # buffers) and raises StopIteration. It performs no extra blend work
                # -- all num_layers+2 yields are already consumed.
                try:
                    next(driver)
                except StopIteration:
                    continue          # correct, buffers released
                except Exception:
                    logger.exception("Blend driver epilogue failed")
                    self._overlap_degraded = True
                    continue
                logger.error(
                    "Blend driver yielded again after its full cadence; closing and "
                    "disabling overlap."
                )
                self._overlap_degraded = True
                try:
                    driver.close()
                except Exception:
                    pass
                continue
            logger.warning(
                "Deferred blend driver is genuinely short by %d layer(s) "
                "(%d/%d driven); completing it before reset.",
                remaining, done, self.num_layers,
            )
            finished = False
            try:
                with stream_ctx:
                    for _ in range(remaining):
                        next(driver)
                finished = True
            except StopIteration:
                finished = True
            except Exception:
                logger.exception(
                    "Deferred blend driver failed while draining; falling back to "
                    "the eager blend path for the rest of this process."
                )
                self._overlap_degraded = True
                finished = True
            if not finished:
                # Ran the full bound without StopIteration -> the generator is not
                # following the expected cadence. Do not trust the overlap path.
                logger.error(
                    "Deferred blend driver did not finish within num_layers+2 "
                    "steps; discarding it and disabling overlap."
                )
                self._overlap_degraded = True
            try:
                driver.close()
            except Exception:
                logger.exception("Error closing deferred blend driver")

        # Leave no work in flight on the side stream: the next step's
        # cur.wait_stream(self._blend_stream) must not inherit a partial blend.
        if self._blend_stream is not None:
            self._blend_stream.synchronize()
        if getattr(self, "_ar_mux", False):
            get_ar_mux().wait_for_ar()
            mux = get_ar_mux()
            if mux.ar_stream is not None:
                mux.ar_stream.synchronize()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.debug(f"Retrieved {num_retrieved_tokens} tokens")

        # Pipelining A / async overlap: step deferred codecsight blenders.
        # Batched deferred drivers (DeferredBatchedBlendDriver):
        #   * Default: wait for ahead FETCH on `_blend_stream`, RECOMPUTE on
        #     current (TP-safe), next FETCH on side stream.
        #   * AR mux: wait for prior blend-stream work (so layer-L KV is ready
        #     before attention), then launch RECOMPUTE(L+1)+FETCH on the side
        #     stream without waiting — overlaps the rest of prefill layer L;
        #     collectives go through the AR multiplexer.
        # Legacy generator drivers keep the old side-stream step path.
        if self.layerwise_blenders:
            if self._async_overlap and self._blend_stream is not None:
                cur = torch.cuda.current_stream()
                cur.wait_stream(self._blend_stream)
                if getattr(self, "_ar_mux", False):
                    get_ar_mux().wait_for_ar(cur)
                self.layerwise_blenders = self._step_blenders(
                    self.layerwise_blenders, fetch_stream=self._blend_stream
                )
            else:
                self.layerwise_blenders = self._step_blenders(
                    self.layerwise_blenders, fetch_stream=None
                )

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        if self.current_layer == 0:
            self.layerwise_storers = []

            is_first = True

            for idx, request in enumerate(connector_metadata.requests):
                save_spec = request.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue

                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.cuda()

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                logger.debug(f"kv_role: {self.kv_role}, layer_name: {layer_name}, skip_leading_tokens: {skip_leading_tokens}")
                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.debug(
                    "save_kv_layer->Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                )
                self.layerwise_storers.append(layerwise_storer)
                if is_first:
                    is_first = False

        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)

        self.current_layer += 1

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        # logger.info("Waiting for saving KV caches to LMCache, kv_role=%s, and "
                    # "use_layerwise=%s", self.kv_role, self.use_layerwise)
        if self.use_layerwise:
            for layerwise_storer in self.layerwise_storers:
                next(layerwise_storer)

            # unpin the kv caches according to req_id
            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        for request in connector_metadata.requests:
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.cuda()

            skip_leading_tokens = save_spec.skip_leading_tokens
            if self.kv_role == "kv_producer":
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            logger.debug(f"kv_role: {self.kv_role}, before-skip_leading_tokens: {skip_leading_tokens}")
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            logger.debug(f"kv_role: {self.kv_role}, after-skip_leading_tokens: {skip_leading_tokens}")

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.debug(
                "wait_for_save->Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
            )

            # NOTE(Jiayi): We assume all tokens are saved
            save_spec.skip_leading_tokens = len(token_ids)
            if request.disagg_spec:
                request.disagg_spec.num_transferred_tokens = len(token_ids)

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        self._requests_priority[request.request_id] = getattr(request, "priority", 0)

        token_ids = request.prompt_token_ids

        # If the request has multimodal hashes, apply them to the token ids
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            # logger.info("Applying multimodal hashes to token ids for request %s, mm_hashes: %s, mm_positions: %s",
                        #  request.request_id, mm_hashes, mm_positions)
            # TODO(Jiayi): Optimize this
            token_ids = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
            token_ids = token_ids.tolist()

        request_configs = extract_request_configs(request.sampling_params)

        if self.skip_last_n_tokens > 0:
            token_ids = token_ids[: -self.skip_last_n_tokens]

        lookup_id = request.request_id
        logger.debug("request %s: Looking up KV cache for %d tokens with configs %s",
                     request.request_id, len(token_ids), request_configs)
        
        num_external_hit_tokens = self.lookup_client.lookup(
            token_ids,
            lookup_id=lookup_id,
            request_configs=request_configs,
        )
        logger.debug("Lookup result for request %s: %s", request.request_id, num_external_hit_tokens)

        if num_external_hit_tokens is None:
            logger.info(
                "Reqid: %s, Total tokens %d, LMCache hit tokens: None.",
                request.request_id,
                request.num_tokens,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            logger.info("Full prompt hit for request %s, need to recompute the last token", request.request_id)
            need_to_allocate -= 1

        logger.info(
            "Reqid: %s, Total tokens %d, LMCache hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if need_to_allocate <= 0:
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        self.lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return
        logger.debug(f"num_external_tokens is {num_external_tokens}")
        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        # Only check for non-prompt-hit case
        if (
            self.load_specs[request.request_id].lmcache_cached_tokens
            != request.num_tokens
        ):
            assert (
                num_external_tokens > 0
                and num_external_tokens
                == self.load_specs[request.request_id].lmcache_cached_tokens
                - self.load_specs[request.request_id].vllm_cached_tokens
            ), (
                f"Mismatch in number of tokens: {num_external_tokens} vs "
                f"{self.load_specs[request.request_id].lmcache_cached_tokens} - "
                f"{self.load_specs[request.request_id].vllm_cached_tokens}"
                f" for request {request.request_id}"
            )

        self.load_specs[request.request_id].can_load = True

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                request_tracker = self._request_trackers[req.req_id]
                request_tracker.update(req.new_token_ids, req.new_block_ids)

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=None,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = len(request_tracker.token_ids)
                new_token_ids = request.all_token_ids[
                    num_current_tokens : num_current_tokens + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            request_tracker.update(new_token_ids, new_block_ids)

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=None,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        return False, return_params
