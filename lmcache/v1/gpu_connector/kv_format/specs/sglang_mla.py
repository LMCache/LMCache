# SPDX-License-Identifier: Apache-2.0
"""SGLang MLA KV layout (fused PBS, NB and BS collapsed into one axis)."""

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.families import SGLangFusedPBSSpec
import lmcache.c_ops as lmc_ops


class SGLangMLASpec(SGLangFusedPBSSpec):
    """SGLang MLA: ``NL x [PBS, 1, HS]`` (PBS = NB * BS fused)."""

    abstract: ClassVar[bool] = False
    gpu_kv_format: ClassVar = lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS
    shape_desc: ClassVar = "NL x [PBS, 1, HS]"
    backend_label: ClassVar = "SGLang MLA"
    is_mla: ClassVar[bool] = True

    def num_layers(self) -> int:
        return len(self._as_layer_list())

    def num_heads(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[1]

    def head_size(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[2]

    def page_buffer_size(self) -> int:
        return self._as_layer_list()[0].shape[0]

    def elements_per_layer(self) -> int:
        return self._as_layer_list()[0].numel()

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = self._as_layer_list()
        return [layers[i].data_ptr() for i in layer_indices]

    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        return self._as_layer_list()[layer_idx]
