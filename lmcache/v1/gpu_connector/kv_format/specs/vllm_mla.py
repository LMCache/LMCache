# SPDX-License-Identifier: Apache-2.0
"""vLLM MLA KV layout."""

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.base import KVFormatSpec
import lmcache.c_ops as lmc_ops


class VLLMMLASpec(KVFormatSpec):
    """vLLM MLA: ``NL x [NB, BS, HS]``."""

    abstract: ClassVar[bool] = False
    engine: ClassVar[str] = "vllm"
    gpu_kv_format: ClassVar = lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    shape_desc: ClassVar = "NL x [NB, BS, HS]"
    backend_label: ClassVar = "vLLM MLA"
    is_mla: ClassVar[bool] = True
    # MLA is the only format today that exercises dim-0 padding via
    # mixed-compression KV pools.
    is_block_axis_dim0: ClassVar[bool] = True

    def num_layers(self) -> int:
        return len(self._as_layer_list())

    def num_blocks(self) -> int:
        return self._as_layer_list()[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[1]

    def num_heads(self, layer_idx: int = 0) -> int:
        # Heads are absorbed into hidden dim for MLA.
        return 1

    def head_size(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[2]

    def page_buffer_size(self) -> int:
        t = self._as_layer_list()[0]
        return t.shape[0] * t.shape[1]

    def elements_per_layer(self) -> int:
        return self._as_layer_list()[0].numel()

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = self._as_layer_list()
        return [layers[i].data_ptr() for i in layer_indices]

    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        return self._as_layer_list()[layer_idx]
