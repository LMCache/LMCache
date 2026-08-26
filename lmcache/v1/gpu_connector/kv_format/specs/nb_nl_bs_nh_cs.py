# SPDX-License-Identifier: Apache-2.0
"""Cross-layer, HND, fused K/V: a single tensor ``[NB, NL, BS, NH, CS]``.

vLLM's standardized ``BLNHC`` layout: blocks first, tokens before heads; every layer's
``[BS, NH, CS]`` run packed inside each block. Reconstructed by detection
from the per-layer views vLLM registers into one shared buffer.
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


class NB_NL_BS_NH_CS_Spec(KVFormatSpec):
    engine_kv_format = lmcache_native.EngineKVFormat.NB_NL_BS_NH_CS
    attention_backends = ("vLLM BLNHC (blocks-first, unified KV cache)",)
    is_cross_layer = True
    is_fused_packed = True

    def num_layers(self) -> int:
        return self.kv_caches.shape[1]

    def num_blocks(self) -> int:
        return self.kv_caches.shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[2]

    def page_buffer_size(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[2]

    def kv_size(self) -> int:
        return 1

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[3]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[3] * self.kv_caches.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[4]

    def tokens_per_layer(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[2]

    def elements_per_layer(self) -> int:
        t = self.kv_caches
        return t.shape[0] * t.shape[2] * t.shape[3] * t.shape[4]

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches.dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        # Per-layer base pointers: layer l's data starts one per-(layer,
        # block) chunk after layer l-1's within every block, so the bases
        # step by chunk bytes and the kernels stride whole blocks.
        tensor = cast(torch.Tensor, self.kv_caches)
        chunk_bytes = tensor.stride(1) * tensor.element_size()
        base = tensor.data_ptr()
        return [base + layer * chunk_bytes for layer in layer_indices]
