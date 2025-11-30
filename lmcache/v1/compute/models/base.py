# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# Third Party
from torch import nn
import torch

# First Party
from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from lmcache.v1.compute.positional_encoding import get_fused_rope

# TODO(Jiayi): A few things need to be tested/supported:
# TP, PP, Multimodal


def _get_attr_rec(obj, dotted: str) -> Optional[object]:
    cur = obj
    for name in dotted.split("."):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def _resolve_decoder_layers(vllm_model: nn.Module) -> Tuple[nn.Module, nn.ModuleList]:
    """
    Return (decoder_root, layers) where `layers` is the ModuleList of transformer blocks.
    Compatible with LLaMA/Qwen3 和 Qwen2.5-VL 等结构。
    """
    # 常见路径优先
    candidates = [
        "model.layers",
        "language_model.model.layers",   # Qwen2.5-VL
        "transformer.layers",
        "backbone.model.layers",
        "base_model.model.layers",
        "model.decoder.layers",
    ]
    for path in candidates:
        layers = _get_attr_rec(vllm_model, path)
        if isinstance(layers, nn.ModuleList):
            # decoder_root = path 的上一级
            parent_path = ".".join(path.split(".")[:-1]) or ""
            decoder_root = _get_attr_rec(vllm_model, parent_path) if parent_path else vllm_model
            return decoder_root, layers

    # 兜底：广搜 named_children，找到带 layers 的模块
    for name, mod in vllm_model.named_modules():
        if hasattr(mod, "layers") and isinstance(mod.layers, nn.ModuleList):
            return mod, mod.layers

    raise AttributeError(
        "Cannot locate decoder layers. Tried common paths for LLaMA/Qwen families."
    )


def _pick(attr_owner, *names, default=None):
    for n in names:
        if hasattr(attr_owner, n):
            return getattr(attr_owner, n)
    return default


class LMCBaseModel(nn.Module, ABC):
    def __init__(
        self,
        vllm_model,
        blender,
        enable_sparse: bool = False,
    ):
        super().__init__()
        self.vllm_model = vllm_model

        # --- 解码器与层解析（兼容 Qwen2.5-VL） ---
        self.decoder_root, self.layers = _resolve_decoder_layers(vllm_model)
        self.num_layers = len(self.layers)

        # vLLM 里有些 wrapper 会在 decoder_root 挂 start_layer/end_layer
        self.start_layer = getattr(self.decoder_root, "start_layer", 0)
        self.end_layer = getattr(self.decoder_root, "end_layer", self.num_layers)

        # --- 注意力后端解析 ---
        self.vllm_attn_layers = []
        self.lmc_attn_layers = []
        for i in range(self.num_layers):
            block = self.layers[i]
            self_attn = block.self_attn
            # vLLM 常见为 self_attn.attn；若无则用 self_attn 自身
            vllm_attn_mod = getattr(self_attn, "attn", self_attn)
            self.vllm_attn_layers.append(vllm_attn_mod)
            self.lmc_attn_layers.append(
                infer_attn_backend_from_vllm(vllm_attn_mod, enable_sparse)
            )

        # --- Rotary Embedding（字段名兼容） ---
        # 取第0层 rotary_emb
        rotary = self.layers[0].self_attn.rotary_emb
        head_dim = _pick(rotary, "head_size", "head_dim")
        max_position_embeddings = _pick(rotary, "max_position_embeddings", "max_seq_len_cached", default=8192)
        # Qwen 系常见 base 字段也可能叫 rope_theta
        base = _pick(rotary, "base", "rope_theta", default=10000.0)
        is_neox_style = getattr(rotary, "is_neox_style", True)
        dtype = getattr(rotary, "dtype", torch.get_default_dtype())
        rope_scaling = getattr(rotary, "rope_scaling", None)

        self.fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

        # NOTE(Jiayi): better not to pass the blender in init
        # if we want to make this LMCModel more general.
        self.blender = blender

    @abstractmethod
    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors. Model-specific implementation."""
        pass

    def _project_qkv(self, layer, hidden_states: torch.Tensor):
        """
        统一执行 qkv 投影，兼容不同返回签名与 size 字段。
        返回 (q, k, v) in last-dim concat space。
        """
        out = layer.self_attn.qkv_proj(hidden_states)
        # 兼容 (tensor, None) 或 仅 tensor
        if isinstance(out, tuple):
            qkv = out[0]
        else:
            qkv = out

        # 优先用显式 size
        q_size = getattr(layer.self_attn, "q_size", None)
        kv_size = getattr(layer.self_attn, "kv_size", None)

        num_heads = _pick(self.vllm_attn_layers[layer.layer_id if hasattr(layer, "layer_id") else 0],
                          "num_heads", default=None)
        num_kv_heads = _pick(self.vllm_attn_layers[layer.layer_id if hasattr(layer, "layer_id") else 0],
                             "num_kv_heads", "num_key_value_heads", default=None)
        head_size = _pick(self.vllm_attn_layers[layer.layer_id if hasattr(layer, "layer_id") else 0],
                          "head_size", "head_dim", default=None)

        if q_size is None or kv_size is None:
            # 退化：从 heads 推导
            assert num_heads is not None and num_kv_heads is not None and head_size is not None, \
                "Cannot infer Q/K/V split sizes; missing heads/head_size info."
            q_size = num_heads * head_size
            kv_size = num_kv_heads * head_size

        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        return q, k, v

    def rope_cache_to_device(self, device: torch.device):
        self.fused_rotary_emb.rope_cache_to_device(device)

    @torch.compile
    def compute_layer(
        self,
        input_ids: torch.Tensor,
        page_stream=None,
        **kwargs,
    ):
        input_ids = input_ids.cuda()

        # 某些集成里 get_input_embeddings 可直接接收 ids；保持现有调用方式
        hidden_states = self.vllm_model.get_input_embeddings(input_ids)

        residual = None
        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )

        stream = page_stream if page_stream is not None else torch.cuda.current_stream()

        for layer_idx, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            # Self Attention 前的 LN/残差
            with torch.cuda.stream(stream):
                if residual is None:
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = layer.input_layernorm(hidden_states, residual)

                # QKV 投影（兼容多实现）
                q, k, v = self._project_qkv(layer, hidden_states)

                # Model-specific QKV processing（如 Qwen3 的 q_norm/k_norm）
                q, k, v = self._process_qkv(q, k, v, layer)

                # 交给 blender
                q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
                    q, k, v, residual, self.start_layer + layer_idx, attn_output, attn_metadata
                )

                # 取本层注意力维度（从 vllm_attn_layers 提取，已在 __init__ 存好）
                attn_core = self.vllm_attn_layers[self.start_layer + layer_idx]
                num_heads = _pick(attn_core, "num_heads")
                num_kv_heads = _pick(attn_core, "num_kv_heads", "num_key_value_heads")
                head_size = _pick(attn_core, "head_size", "head_dim")

                # 变形并前向
                q = q.view(-1, num_heads, head_size)
                k = k.view(-1, num_kv_heads, head_size)
                v = v.view(-1, num_kv_heads, head_size)
                attn_output = (attn_output if attn_output is not None else torch.zeros_like(q)).view(
                    -1, num_heads, head_size
                )

                attn_output = self.lmc_attn_layers[self.start_layer + layer_idx].forward_contiguous(
                    q, k, v, attn_output, attn_metadata
                )

                attn_output = attn_output.view(-1, num_heads * head_size)
                # K/V reshape 回平面以便下游（即便未直接使用，也保持与原实现一致）
                _ = k.view(-1, num_kv_heads * head_size)
                _ = v.view(-1, num_kv_heads * head_size)

                # 输出投影
                hidden_states, _ = layer.self_attn.o_proj(attn_output)

                # FFN
                hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
                hidden_states = layer.mlp(hidden_states)

            yield
