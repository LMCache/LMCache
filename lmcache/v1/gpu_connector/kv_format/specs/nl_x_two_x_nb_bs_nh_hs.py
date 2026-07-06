# SPDX-License-Identifier: Apache-2.0
"""Per-layer ``(K, V)`` tuple format: ``NL x [(K, V) x [NB, BS, NH, HS]]``.

Each layer's KV cache is stored as a ``(K, V)`` tuple of 4-D paged tensors
``[num_blocks, block_size, num_heads, head_size]``. Layers are outermost; each
layer carries its own ``(K, V)`` pair rather than a single stacked tensor. This
is a general per-layer tuple layout (currently produced by vLLM-Ascend), distinct
from ``TWO_X_NL_X_NB_BS_NH_HS`` (SGLang MP), which splits K/V at the *outermost*
level. The transfer kernels receive this native structure plus the engine's own
``KVCacheFormat``.
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list/tuple) per its format,
# so the ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
import lmcache.lmcache_native as lmcache_native


class NL_X_TWO_X_NB_BS_NH_HS_Spec(KVFormatSpec):
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS
    attention_backends = ("vLLM-Ascend: per-layer (K, V) tuples",)
    is_layer_list = True
    is_kv_second_tuple = True

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0][0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx][0].shape[1]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def kv_size(self) -> int:
        # Each layer carries its own (K, V) pair -- two planes per layer.
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx][0].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        inner = self.kv_caches[layer_idx][0]
        return inner.shape[2] * inner.shape[3]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx][0].shape[-1]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx][0].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        # Interleaved [K_i, V_i, ...] per layer -- the order the
        # multi_layer_kv_transfer kernel iterates planes.
        layers = cast(list, self.kv_caches)
        ptrs: list[int] = []
        for i in layer_indices:
            ptrs.append(layers[i][0].data_ptr())  # K
            ptrs.append(layers[i][1].data_ptr())  # V
        return ptrs
