# SPDX-License-Identifier: Apache-2.0
"""Per-layer, NHD, blocks-first with packed fused K/V: ``NL x [NB, BS, NHP, P, HS]``.

A ``list[NL]`` of the 5-D tensor vllm-torchtpu allocates per layer:
``[NB, BS, align(2 * NH, P) / P, P, align(head_dim, 128)]``, where ``P`` is the
dtype packing (2 for bf16, 4 for fp8). Each head is stored fused as
``[k_h | v_h]``, so the bytes are those of ``NL_X_NB_BS_NH_CS``; the 5-D shape
is kept because reshaping it on TPU forces a relayout.

Geometry treats each pair of packed rows as one fused head, like the CS
formats: ``head_size`` is the packed ``2 * HS``, and ``num_heads`` includes the
heads that alignment padding adds, so ``num_heads * head_size == hidden_dim``.
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import (
    KVFormatSpec,
    concrete_shape,
)
import lmcache.lmcache_native as lmcache_native


class NL_X_NB_BS_NHP_P_HS_Spec(KVFormatSpec):
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_NB_BS_NHP_P_HS
    attention_backends = ("vLLM TorchTPU attention, packed fused K/V",)
    is_layer_list = True
    is_fused_packed = True

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[1]

    def page_buffer_size(self) -> int:
        return self.num_blocks() * self.block_size()

    def kv_size(self) -> int:
        # K/V stay packed in adjacent rows; a single fused plane.
        return 1

    def num_heads(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[2] * t.shape[3] // 2

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[2] * t.shape[3] * t.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        # The packed per-head content size (2 * HS, K/V adjacent).
        return 2 * self.kv_caches[layer_idx].shape[4]

    def tokens_per_layer(self) -> int:
        return self.num_blocks() * self.block_size()

    def elements_per_layer(self) -> int:
        return self.kv_caches[0].numel()

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def concrete_shape_str(self) -> str:
        # The HS axis is half of head_size(), so render the raw axes instead.
        nb, bs, nhp, p, hs = self.kv_caches[0].shape
        sizes = {
            "NL": self.num_layers(),
            "NB": nb,
            "BS": bs,
            "NHP": nhp,
            "P": p,
            "HS": hs,
        }
        return concrete_shape(self.engine_kv_format, sizes.__getitem__)
