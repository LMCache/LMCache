# SPDX-License-Identifier: Apache-2.0
"""SGLang MHA KV layout (outer K/V split list)."""

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.families import SGLangFusedPBSSpec
import lmcache.c_ops as lmc_ops


class SGLangMHASpec(SGLangFusedPBSSpec):
    """SGLang MHA: ``2 x NL x [PBS, NH, HS]``."""

    abstract: ClassVar[bool] = False
    gpu_kv_format: ClassVar = lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS
    shape_desc: ClassVar = "2 x NL x [PBS, NH, HS]"
    backend_label: ClassVar = "SGLang MHA (flash attention and flash infer)"

    def num_layers(self) -> int:
        return len(self._as_kv_layer_list()[0])

    def num_heads(self, layer_idx: int = 0) -> int:
        return self._as_kv_layer_list()[0][layer_idx].shape[1]

    def head_size(self, layer_idx: int = 0) -> int:
        return self._as_kv_layer_list()[0][layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self._as_kv_layer_list()[0][layer_idx]
        return t.shape[1] * t.shape[2]

    def page_buffer_size(self) -> int:
        return self._as_kv_layer_list()[0][0].shape[0]

    def elements_per_layer(self) -> int:
        # Separate K and V tensors per layer.
        return self._as_kv_layer_list()[0][0].numel() * 2

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        kv = self._as_kv_layer_list()
        k_list, v_list = kv[0], kv[1]
        return [k_list[i].data_ptr() for i in layer_indices] + [
            v_list[i].data_ptr() for i in layer_indices
        ]

    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        return self._as_kv_layer_list()[0][layer_idx]

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self._as_kv_layer_list()[0][layer_idx].dtype
