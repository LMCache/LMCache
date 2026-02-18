# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Sequence, Union, Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        # layerwise_model 内部已兼容 Qwen2.5-VL 的层解析
        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # 使用 layerwise_model 暴露的层与切片信息，避免硬编码 vllm_model.model.layers
        self.layers = self.layerwise_model.layers
        self.start_layer = getattr(self.layerwise_model, "start_layer", 0)
        self.end_layer = getattr(self.layerwise_model, "end_layer", len(self.layers))
        self.num_layers = self.end_layer - self.start_layer

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
            is_costream=config.is_costream,
            GOP=config.GOP,
        )
        self.is_costream = bool(config.is_costream)
        self.gop = max(int(config.GOP), 1)
        self.skip_ffn = False
        self.skip_ffn_only_costream = True
        self._single_zero_idx: dict[torch.device, torch.Tensor] = {}
        if config.extra_config is not None:
            self.skip_ffn = bool(config.extra_config.get("skip_ffn", False))
            self.skip_ffn_only_costream = bool(
                config.extra_config.get("skip_ffn_only_costream", True)
            )
        if self.skip_ffn:
            logger.warning(
                "FFN skip is enabled (only_costream=%s). This may reduce output quality.",
                self.skip_ffn_only_costream,
            )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )
        self._rotary_by_layer = [
            self._get_rotary_emb(layer.self_attn) for layer in self.layers
        ]

    def _get_rotary_emb(self, attn_layer):
        # 兼容不同实现的 rotary 暴露方式
        if hasattr(attn_layer, "rotary_emb"):
            return attn_layer.rotary_emb
        if hasattr(attn_layer, "rotary_emb_func"):
            return attn_layer.rotary_emb_func  # 少数实现
        raise AttributeError("Attention layer does not expose rotary embedding module.")

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        """
        layer_id 为全局层号（与 layerwise_model 内部一致），支持 Qwen2.5-VL。
        """
        logger.debug("Blender is processing KV for layer %d", layer_id)
        try:
            old_k, old_v = self.gpu_connector.get_kv(layer_id)
        except ValueError:
            # When the layerwise retriever finds no cached KV for this layer,
            # the GPU buffer mapping will be empty. Instead of crashing the
            # engine, fall back to the original QKV so the request can
            # continue without blending.
            logger.warning(
                "KV cache for layer %s is not loaded into GPU buffer, skip blending.",
                layer_id,
            )
            return q, k, v, residual, attn_output, attn_metadata

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # Initialize (or inherit) positions: length is consistent with the dimension of the current batch of tokens
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )

        # Rotary (common in the Qwen family)
        rotary = self._rotary_by_layer[layer_id]

        # CoStream: recompute only key-frame tokens
        if self.is_costream:
            # Build recompute indices once, then keep using the same reduced path
            # for following layers in this request.
            if self.metadata.imp_indices is None:
                num_tokens = q.shape[0]
                cache_len = old_k.shape[0]
                effective_len = min(num_tokens, cache_len)
                logger.debug("num_tokens=%d, cache_len=%d", num_tokens, cache_len)
                # Gap positions are cache-miss token positions; the remaining
                # tokens are cache-hit positions.
                gap_positions = getattr(
                    self.gpu_connector, "current_gap_positions", None
                )
                if gap_positions is None or gap_positions.numel() == 0:
                    hit_indices = torch.arange(
                        effective_len, device=q.device, dtype=torch.long
                    )
                else:
                    hit_mask = torch.ones(
                        effective_len, device=q.device, dtype=torch.bool
                    )
                    if gap_positions.device != q.device or gap_positions.dtype != torch.long:
                        gap_positions = gap_positions.to(
                            device=q.device, dtype=torch.long
                        )
                    valid_gap = gap_positions[
                        (gap_positions >= 0) & (gap_positions < effective_len)
                    ]
                    hit_mask[valid_gap] = False
                    hit_indices = torch.where(hit_mask)[0]

                gop = self.gop
                tokens_per_frame = int(self.metadata.tokens_per_frame or 0)
                mm_positions: Optional[Sequence[Any]] = self.metadata.mm_positions
                # logger.info(f'tokens_per_frame is {tokens_per_frame}, gop is {gop}')

                if mm_positions:
                    # Precise alignment using mm_positions (offset/length per frame).
                    selected_chunks = []
                    first_key_start: Optional[int] = None
                    for frame_id, placeholder in enumerate(mm_positions):
                        start = int(getattr(placeholder, "offset", 0))
                        length = int(getattr(placeholder, "length", 0))
                        if length <= 0 or start >= effective_len:
                            continue
                        end = min(start + length, effective_len)
                        if gop > 1 and ((frame_id+1) % gop) != 0:
                            continue
                        if first_key_start is None:
                            first_key_start = start
                        hit_in_range = hit_indices[
                            (hit_indices >= start) & (hit_indices < end)
                        ]
                        if hit_in_range.numel() > 0:
                            selected_chunks.append(hit_in_range)

                    if selected_chunks:
                        selected_indices = torch.cat(selected_chunks, dim=0)
                    else:
                        selected_indices = hit_indices.new_empty((0,))

                    # If no hit tokens in key frames, keep one token from the first
                    # key frame range to avoid empty selection.
                    if selected_indices.numel() == 0 and first_key_start is not None:
                        if first_key_start < effective_len:
                            selected_indices = torch.tensor(
                                [first_key_start], device=q.device, dtype=torch.long
                            )
                    logger.debug("selected_indices length=%d", len(selected_indices))
                # Recompute key frames by GOP at frame granularity based on tokens_per_frame.
                elif tokens_per_frame > 0:
                    if gop > 1:
                        frame_ids = hit_indices // tokens_per_frame
                        selected_indices = hit_indices[(frame_ids % gop) == 0]
                    else:
                        selected_indices = hit_indices
                    if selected_indices.numel() == 0 and hit_indices.numel() > 0:
                        selected_indices = hit_indices[:1]
                    elif selected_indices.numel() == 0:
                        selected_indices = hit_indices
                    logger.debug("selected_indices length=%d", len(selected_indices))
                elif gop > 1:
                    # Fallback: no frame info, degrade to token-based selection.
                    selected_indices = hit_indices[(hit_indices % gop) == 0]
                    if selected_indices.numel() == 0 and hit_indices.numel() > 0:
                        selected_indices = hit_indices[:1]
                else:
                    selected_indices = hit_indices

                self.metadata.imp_indices = selected_indices
                self.metadata.positions = self.metadata.positions[selected_indices]
                self.metadata.selection_effective_len = effective_len
                self.metadata.is_full_selection = selected_indices.numel() == effective_len
                logger.debug(
                    "CoStream mode selected %d/%d hit tokens for recompute (GOP=%d).",
                    int(selected_indices.numel()),
                    int(num_tokens),
                    gop,
                )

            imp_indices = self.metadata.imp_indices
            assert imp_indices is not None
            effective_len = min(q.shape[0], old_k.shape[0])
            selection_effective_len = int(self.metadata.selection_effective_len or 0)
            if selection_effective_len > 0 and effective_len >= selection_effective_len:
                # Fast path: indices are already valid for this layer.
                layer_imp = imp_indices
                layer_positions = self.metadata.positions
            else:
                valid_mask = imp_indices < effective_len
                layer_imp = imp_indices[valid_mask]
                layer_positions = self.metadata.positions[valid_mask]
            if layer_imp.numel() == 0 and effective_len > 0:
                if q.device not in self._single_zero_idx:
                    self._single_zero_idx[q.device] = torch.tensor(
                        [0], device=q.device, dtype=torch.long
                    )
                layer_imp = self._single_zero_idx[q.device]
                layer_positions = self._single_zero_idx[q.device]

            # Each layer gets a fresh attention metadata object; keep it aligned
            # with the reduced token set.
            full_range_selected = (
                self.metadata.is_full_selection
                and effective_len == q.shape[0]
                and selection_effective_len == q.shape[0]
            )
            if not full_range_selected:
                attn_metadata.update_from_top_indices(layer_imp)
                k = k.index_select(0, layer_imp)
                v = v.index_select(0, layer_imp)
                q = q.index_select(0, layer_imp)
                residual = residual.index_select(0, layer_imp)
                attn_output = attn_output[: len(layer_imp)]
            # Apply rotary after selecting tokens to keep positions aligned.
            q, k = rotary(layer_positions, q, k)
            write_indices = layer_imp
        else:
            q, k = rotary(self.metadata.positions, q, k)
            write_indices = self.metadata.imp_indices

        # Recomputation/selection logic for important layers
        if layer_id in self.common_metadata.check_layers and not self.is_costream:
            logger.info(f'layer_id is {layer_id}, len(layers) is {len(self.layers)}')   
            # Select the KV rows that need to be recalculated based on L2 differences
            diff_k = torch.sum(
                (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
            )
            total_len = diff_k.shape[0]

            assert self.common_metadata.recomp_ratios is not None
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])

            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.update_from_top_indices(top_indices)

        # Write the selected key-value pairs back/return
        if self.metadata.imp_indices is not None:
            if write_indices is None:
                write_indices = self.metadata.imp_indices
            old_k[write_indices] = k
            old_v[write_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to orchestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwise retrieve + blending.
        """
        # TODO(Jiayi): store is currently not included in this function
        check_layers = self.common_metadata.check_layers
        layerwise_model_executor = self.layerwise_model.compute_layer(check_layers, tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        # warmup retriever
        warmup_retrieved = next(layerwise_retriever)
        has_retrieved_tokens = False
        if warmup_retrieved is not None:
            if torch.is_tensor(warmup_retrieved):
                has_retrieved_tokens = int(warmup_retrieved.item()) > 0
            else:
                has_retrieved_tokens = int(warmup_retrieved) > 0
        yield

        if not has_retrieved_tokens:
            logger.debug("No retrievable tokens in layerwise retrieve; skip blending compute.")
            for _ in range(self.num_layers):
                next(layerwise_retriever)
                yield
            next(layerwise_retriever)
            self.metadata.clean()
            yield
            return

        for _ in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)

        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()
        logger.info("enter blend")
        tokens_per_frame = kwargs.get("tokens_per_frame")
        if tokens_per_frame is not None:
            self.metadata.tokens_per_frame = int(tokens_per_frame)
        mm_positions = kwargs.get("mm_positions")
        if mm_positions is not None:
            self.metadata.mm_positions = mm_positions
        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        # +2 is for the handshake/closing process with the retriever at both the beginning and end.
        for _ in range(self.num_layers + 2):
            next(layerwise_blender)
