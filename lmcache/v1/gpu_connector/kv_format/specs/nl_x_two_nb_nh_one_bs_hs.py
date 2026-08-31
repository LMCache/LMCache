# SPDX-License-Identifier: Apache-2.0
"""Per-layer HND with a singleton between heads and tokens:
``NL x [2, NB, NH, 1, BS, HS]`` (vLLM-RBLN).

A ``list[NL]`` of a 6-D tensor with the K/V pair leading. Axis 3 is always 1 --
the RBLN attention backend requires it -- so the bytes and strides are those of
``NL_X_TWO_NB_NH_BS_HS``, and only the rank differs. The geometry accessors here
therefore read the same axes shifted by one past the singleton.
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
import lmcache.lmcache_native as lmcache_native


class NL_X_TWO_NB_NH_ONE_BS_HS_Spec(KVFormatSpec):
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS
    attention_backends = ("vLLM-RBLN attention (HND layout)",)
    is_layer_list = True
    is_hnd = True
    is_two_major = True

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[1]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[4]

    def page_buffer_size(self) -> int:
        return self.num_blocks() * self.block_size()

    def kv_size(self) -> int:
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.num_heads(layer_idx) * self.head_size(layer_idx)

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[5]

    def tokens_per_layer(self) -> int:
        return self.num_blocks() * self.block_size()

    def elements_per_layer(self) -> int:
        return self.kv_caches[0].numel()

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]
