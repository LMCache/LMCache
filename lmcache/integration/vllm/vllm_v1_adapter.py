# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import contextlib
import os

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
        # Deferred costream blenders (pipelining A): stepped per-layer from
        # wait_for_layer_load so the per-layer KV load overlaps prefill.
        self.layerwise_blenders: list[
            Generator[None, None, None]
        ] = []
        self._costream_pipeline = (
            os.environ.get("VLLM_COSTREAM_PIPELINE", "0") == "1"
        )
        # Async overlap prototype: run the deferred blender on a side CUDA
        # stream one layer ahead so blend(L+1)'s recompute overlaps prefill(L).
        # Implies the deferred-blender path. Requires the gpu_connector's
        # per-layer global sync to be scoped (gated on the same env var).
        self._async_overlap = (
            os.environ.get("VLLM_COSTREAM_ASYNC_OVERLAP", "0") == "1"
        )
        if self._async_overlap:
            self._costream_pipeline = True
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
            if merged == 0:
                nan_count = int((~valid_mask).sum().item())
                logger.debug(
                    "vision_embed[0]: shape=%s, nan_rows=%d/%d, dtype=%s",
                    ve.shape, nan_count, actual_len, ve.dtype)
            if ve_slice.shape[0] >= actual_len:
                if valid_mask.all():
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

        token_ids_t = torch.tensor(
            token_ids[:num_tokens], dtype=torch.long, device="cuda"
        )

        # Collect vision embeddings that fall within the cached prefix.
        # Use None as sentinel for encoder_cache misses so that the list
        # stays aligned 1:1 with the mm_positions that pass the filter.
        vision_embeds: list[Optional[torch.Tensor]] = []
        num_encoder_misses = 0
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

        text_embeds = embed_fn(token_ids_t)
        text_has_nan = bool(torch.isnan(text_embeds).any())
        logger.debug(
            "text_embeds: shape=%s, has_nan=%s, norm=%.4f, "
            "token_ids min=%d max=%d",
            text_embeds.shape, text_has_nan,
            text_embeds.norm().item() if not text_has_nan else float('nan'),
            token_ids_t.min().item(), token_ids_t.max().item(),
        )

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

        vision_embeds = vision_embeds_norm
        if use_deepstack and compute_deepstack is not None:
            try:
                # If cache provides full [main|multiscale], use model split path.
                ds_embeds, vision_embeds_main = compute_deepstack(
                    token_ids_t, text_embeds, vision_embeds,
                )
                inputs_embeds = self._scatter_vision_embeds(
                    text_embeds, vision_embeds_main, mm_positions, num_tokens,
                )
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
        inputs_embeds = self._scatter_vision_embeds(
            text_embeds, vision_embeds_scatter, mm_positions, num_tokens,
        )
        return inputs_embeds, deepstack_input_embeds

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
        self.layerwise_blenders = []

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

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
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
                        inputs_embeds, deepstack_input_embeds = (
                            self._reconstruct_inputs_embeds(
                                tokens, request.mm_hashes,
                                request.mm_positions, lmcache_cached_tokens,
                            )
                        )

                    if inputs_embeds is None and not skip_embeds:
                        logger.warning(
                            "inputs_embeds unavailable (encoder_cache "
                            "eviction); falling back to layerwise "
                            "retrieval for this request"
                        )
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
                    with _blend_ctx:
                        deferred_blender = self.blender.blend(
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            defer=self._costream_pipeline,
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

        # Pipelining A / async overlap: step deferred costream blenders one
        # layer at a time. blend_layer yields None each step (no token mask).
        if self._async_overlap and self.layerwise_blenders:
            # The blender is one layer ahead (2x prime). At this hook for
            # prefill layer L, blend(L) was already issued on the side stream
            # last call; make the current (prefill) stream wait for it, then
            # issue blend(L+1) on the side stream so it overlaps prefill(L).
            cur = torch.cuda.current_stream()
            cur.wait_stream(self._blend_stream)
            with torch.cuda.stream(self._blend_stream):
                for layerwise_blender in self.layerwise_blenders:
                    next(layerwise_blender)
        else:
            for layerwise_blender in self.layerwise_blenders:
                next(layerwise_blender)

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
