# SPDX-License-Identifier: Apache-2.0
"""Per-layer MLA/DSA tuple format: ``NL x NP x [NB, BS, 1, HS]`` (NP planes).

Each layer's KV cache is a tuple of NP >= 1 paged tensors ``[num_blocks,
block_size, 1, width]`` -- 2 (MLA: latent, rope), 3 (DSA: latent, rope,
dsa) or 1 (latent-only) -- from vLLM-Ascend (DeepSeek-V2/V3 MLA, V3.2
DSA). One latent KV head; widths may differ between planes, which is what
distinguishes this format from ``(K, V)`` tuples
(:class:`NL_X_TWO_X_NB_BS_NH_HS_Spec`) shape-wise. The transferred object
is one flat plane of the summed width (``kv_size == 1``, ``is_mla``).
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


class NL_X_NP_X_NB_BS_ONE_HS_Spec(KVFormatSpec):
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS
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
        # One flat plane of the summed width; planes are carved by the
        # transfer kernels, not the object layout.
        return 1

    def num_heads(self, layer_idx: int = 0) -> int:
        return 1

    def hidden_dim(self, layer_idx: int = 0) -> int:
        # Columns counted in dtype elements so mixed-item-size planes
        # (int8 latent + fp16 scale) pack losslessly.
        planes = cast(
            "tuple[torch.Tensor, ...]",
            self.kv_caches[layer_idx],
        )
        total_bytes = sum(int(t.shape[-1]) * int(t.element_size()) for t in planes)
        itemsize = int(self.dtype(layer_idx).itemsize)
        if itemsize <= 0 or total_bytes % itemsize != 0:
            raise ValueError(
                "NL_X_NP_X_NB_BS_ONE_HS hidden_dim: plane byte total "
                f"{total_bytes} is not a multiple of dtype itemsize {itemsize}"
            )
        return total_bytes // itemsize

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
