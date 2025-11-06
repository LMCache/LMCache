# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

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
        )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )

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
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # 初始化（或继承）positions：长度与当前批次 token 维度一致
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )

        # 通过已解析的 layers 取当前层，避免 vllm_model.model.layers 的硬编码
        layer = self.layers[layer_id]
        attn_layer = layer.self_attn

        # Rotary 编码（Qwen 家族常见）
        rotary = self._get_rotary_emb(attn_layer)
        q, k = rotary(self.metadata.positions, q, k)

        # 重要层的重计算/挑选逻辑
        if layer_id in self.common_metadata.check_layers:
            # 以 L2 差异挑选需重算的 KV 行
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

        # 将挑选后的 KV 写回/返回
        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
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

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        # 预热 retriever
        next(layerwise_retriever)
        yield

        # 分层执行（使用 num_layers，已按 start/end 计算）
        for _ in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        # 结束收尾
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

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        # +2 是为了首尾两次与 retriever 的握手/收尾
        for _ in range(self.num_layers + 2):
            next(layerwise_blender)
