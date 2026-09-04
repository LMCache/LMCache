# SPDX-License-Identifier: Apache-2.0
"""Per-layer MLA/DSA tuple format: ``NL x [planes x [NB, BS, 1, HS]]``.

Each layer's KV cache is stored as a tuple of 2 (MLA: ``latent, rope``) or
3 (DSA: ``latent, rope, dsa``) paged tensors
``[num_blocks, block_size, 1, width]``, produced by vLLM-Ascend for
DeepSeek-V2/V3 MLA and V3.2 DSA models. All planes share a single latent
KV head and their widths are mutually unequal, which is what distinguishes
this format from the per-layer ``(K, V)`` tuple format
(:class:`NL_X_TWO_X_NB_BS_NH_HS_Spec`) shape-wise.

Like ``NL_X_NB_BS_HS`` the transferred object is one flat plane of the
**summed** width (``kv_size == 1``, ``is_mla``); only the paged source
arrives as per-plane tuples, so device transfer kernels carve the planes
by offset instead of by a leading K/V axis.
"""

# Each spec indexes ``kv_caches`` (nested list/tuple) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
import lmcache.lmcache_native as lmcache_native


class NL_X_TWO_X_NB_BS_HS_Spec(KVFormatSpec):
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    attention_backends = (
        "vLLM-Ascend MLA (latent, rope) tuples",
        "vLLM-Ascend DSA (latent, rope, dsa) tuples",
    )
    is_layer_list = True
    is_mla = True
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
        # One flat plane of the summed width; the per-plane carve lives in
        # the transfer kernels, not in the object layout.
        return 1

    def num_heads(self, layer_idx: int = 0) -> int:
        return 1

    def hidden_dim(self, layer_idx: int = 0) -> int:
        widths = (int(t.shape[-1]) for t in self.kv_caches[layer_idx])
        return sum(widths, 0)

    def head_size(self, layer_idx: int = 0) -> int:
        return self.hidden_dim(layer_idx)

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def elements_per_layer(self) -> int:
        return sum(t.numel() for t in self.kv_caches[0])

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx][0].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        # Interleaved [latent_i, rope_i(, dsa_i), ...] per layer -- the order
        # the multi_layer_kv_transfer kernel iterates planes.
        layers = cast(
            "list[tuple[torch.Tensor, ...]]",
            self.kv_caches,
        )
        ptrs: list[int] = []
        for i in layer_indices:
            for plane in layers[i]:
                ptrs.append(plane.data_ptr())
        return ptrs
