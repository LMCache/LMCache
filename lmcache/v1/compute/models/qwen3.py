# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# Local
from .base import LMCBaseModel

# TODO(Jiayi): A few things need to be tested/supported:
# TP, PP, Multimodal


class LMCQwen3Model(LMCBaseModel):
    """LMC Qwen3 model implementation, inheriting from base model class"""

    def __init__(
        self,
        vllm_model,
        blender,
        enable_sparse: bool = False,
    ):
        # Call parent initialization with all common logic
        super().__init__(vllm_model, blender, enable_sparse)

    def _init_rotary_embedding(self):
        """Qwen3-specific rotary embedding initialization"""
        rotary_emb = self.vllm_model.model.layers[0].self_attn.rotary_emb
        head_dim = rotary_emb.head_size
        max_position_embeddings = rotary_emb.max_position_embeddings
        rope_scaling = getattr(
            rotary_emb, "rope_scaling", None
        )  # Qwen3 may have rope_scaling
        base = rotary_emb.base
        is_neox_style = rotary_emb.is_neox_style
        dtype = rotary_emb.dtype

        # First Party
        from lmcache.v1.compute.positional_encoding import get_fused_rope

        self.fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    def preprocess_attention_qk(self, q: torch.Tensor, k: torch.Tensor, attn_layer):
        """
        Qwen3-specific Q/K preprocessing - apply q_norm and k_norm

        Args:
            q: Query tensor
            k: Key tensor
            attn_layer: Attention layer

        Returns:
            tuple: (processed_q, processed_k)
        """
        # Qwen3-specific Q/K normalization processing
        if hasattr(attn_layer, "q_norm") and hasattr(attn_layer, "k_norm"):
            head_dim = attn_layer.head_dim

            def apply_norm(tensor, norm_layer):
                """Apply norm to tensor, reshaping by head dimension"""
                shape = tensor.shape
                by_head = tensor.view(*shape[:-1], shape[-1] // head_dim, head_dim)
                normalized = norm_layer(by_head)
                return normalized.view(shape)

            q = apply_norm(q, attn_layer.q_norm)
            k = apply_norm(k, attn_layer.k_norm)

        return q, k
