# SPDX-License-Identifier: Apache-2.0
"""Per-layer, HND, blocks-first with fused K/V: ``NL x [NB, NH, BS, 2, HS]``.

A ``list[NL]`` of a 5-D tensor whose K/V (size-2) axis is second-to-last.
The engine registers it raw as 4-D ``[NB, NH, BS, 2*HS]`` (K/V fused into the
trailing dim); detection splits that into this canonical 5-D shape. Produced
by vLLM's non-MLA blocks-first CPU attention backend.
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
import lmcache.c_ops as lmc_ops


class NL_X_NB_NH_BS_TWO_HS_Spec(KVFormatSpec):
    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS
    attention_backends = ("vLLM non-MLA blocks-first, fused K/V",)

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[2]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[1]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[1] * t.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[4]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[2]

    def elements_per_layer(self) -> int:
        t = self.kv_caches[0]
        return t.shape[0] * t.shape[1] * t.shape[2] * t.shape[4] * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]
