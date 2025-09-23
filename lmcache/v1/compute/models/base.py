# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC

# Third Party
from torch import nn
import torch

# First Party
from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from lmcache.v1.compute.positional_encoding import get_fused_rope

# TODO(Jiayi): A few things need to be tested/supported:
# TP, PP, Multimodal


class LMCBaseModel(nn.Module, ABC):
    """Base LMC model class containing common logic for all models"""

    def __init__(
        self,
        vllm_model,
        blender,
        enable_sparse: bool = False,
    ):
        super().__init__()
        self.vllm_model = vllm_model
        self.blender = blender
        self.enable_sparse = enable_sparse

        self.num_layers = len(vllm_model.model.layers)

        # Initialize attention layers
        self.vllm_attn_layers = []
        self.lmc_attn_layers = []
        for i in range(self.num_layers):
            vllm_attn = vllm_model.model.layers[i].self_attn.attn
            self.vllm_attn_layers.append(vllm_attn)
            self.lmc_attn_layers.append(
                infer_attn_backend_from_vllm(vllm_attn, enable_sparse)
            )

        # Initialize rotary embedding (subclasses can override this method)
        self._init_rotary_embedding()

    def _init_rotary_embedding(self):
        """Initialize rotary embedding, subclasses can override this method
        for custom setup"""
        rotary_emb = self.vllm_model.model.layers[0].self_attn.rotary_emb
        head_dim = rotary_emb.head_size
        max_position_embeddings = rotary_emb.max_position_embeddings
        rope_scaling = getattr(rotary_emb, "rope_scaling", None)
        base = rotary_emb.base
        is_neox_style = rotary_emb.is_neox_style
        dtype = rotary_emb.dtype

        self.fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    def preprocess_attention_qk(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, attn_layer
    ):
        """
        Model-specific Q/K preprocessing, subclasses can override this method

        Args:
            q: Query tensor
            k: Key tensor
            positions: token positions for positional encoding
            attn_layer: Attention layer

        Returns:
            tuple: (processed_q, processed_k)
        """
        # Default implementation: no processing
        return q, k

    @torch.compile
    def compute_layer(
        self,
        input_ids: torch.Tensor,
    ):
        """Common layer computation logic"""
        input_ids = input_ids.cuda()
        hidden_states = self.vllm_model.get_input_embeddings(input_ids)
        residual = None
        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )

        for idx, layer in enumerate(
            self.vllm_model.model.layers[
                self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
            ]
        ):
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)

            # Get Q, K, V
            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )

            # Note: Q/K preprocessing (norm + RoPE) is handled in blender with positions

            # Process QKV through blender
            q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
                q, k, v, residual, idx, attn_output, attn_metadata
            )

            # Reshape tensors for attention computation
            num_heads = self.vllm_attn_layers[idx].num_heads
            num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
            head_size = self.vllm_attn_layers[idx].head_size

            q = q.view(-1, num_heads, head_size)
            k = k.view(-1, num_kv_heads, head_size)
            v = v.view(-1, num_kv_heads, head_size)
            attn_output = attn_output.view(-1, num_heads, head_size)

            # Compute attention
            attn_output = self.lmc_attn_layers[idx].forward_contiguous(
                q, k, v, attn_output, attn_metadata
            )

            # Reshape back to original shape
            attn_output = attn_output.view(-1, num_heads * head_size)
            k = k.view(-1, num_kv_heads * head_size)
            v = v.view(-1, num_kv_heads * head_size)

            # Output projection
            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

            yield
