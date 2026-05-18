# SPDX-License-Identifier: Apache-2.0
"""SGLang MHA via MP daemon: 4-D inner per-layer tensor."""

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.base import KVFormatSpec
import lmcache.c_ops as lmc_ops


class SGLangMHAMPSpec(KVFormatSpec):
    """SGLang MHA reached via MP daemon: ``2 x NL x [NB, BS, NH, HS]``.

    The in-process SGLang MHA layout collapses ``num_blocks`` and
    ``block_size`` into a single fused ``page_buffer_size`` axis (3-D
    inner). The MP path un-flattens that axis using the
    ``tokens_per_block`` layout hint so ``num_blocks`` and
    ``block_size`` are individually addressable here.
    """

    abstract: ClassVar[bool] = False
    engine: ClassVar[str] = "sglang"
    gpu_kv_format: ClassVar = lmc_ops.GPUKVFormat.TWO_X_NL_X_NB_BS_NH_HS
    shape_desc: ClassVar = "2 x NL x [NB, BS, NH, HS]"
    backend_label: ClassVar = "SGLang MHA via MP daemon (4-D inner)"

    def num_layers(self) -> int:
        """Return the number of transformer layers in the KV cache."""
        return len(self._as_kv_layer_list()[0])

    def num_blocks(self) -> int:
        """Return the number of pre-allocated KV cache blocks."""
        return self._as_kv_layer_list()[0][0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        """Return the per-block token capacity for ``layer_idx``."""
        return self._as_kv_layer_list()[0][0].shape[1]

    def num_heads(self, layer_idx: int = 0) -> int:
        """Return the number of KV heads for ``layer_idx``."""
        return self._as_kv_layer_list()[0][layer_idx].shape[2]

    def head_size(self, layer_idx: int = 0) -> int:
        """Return the per-head hidden size for ``layer_idx``."""
        return self._as_kv_layer_list()[0][layer_idx].shape[-1]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        """Return ``num_heads * head_size`` for ``layer_idx``."""
        t = self._as_kv_layer_list()[0][layer_idx]
        return t.shape[2] * t.shape[3]

    def page_buffer_size(self) -> int:
        """Return ``num_blocks * block_size``."""
        t = self._as_kv_layer_list()[0][0]
        return t.shape[0] * t.shape[1]

    def elements_per_layer(self) -> int:
        """Return the per-layer element count (K + V combined)."""
        return self._as_kv_layer_list()[0][0].numel() * 2

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        """Return K data pointers followed by V data pointers."""
        kv = self._as_kv_layer_list()
        k_list, v_list = kv[0], kv[1]
        return [k_list[i].data_ptr() for i in layer_indices] + [
            v_list[i].data_ptr() for i in layer_indices
        ]

    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        """Return the per-layer K tensor used for layout introspection."""
        return self._as_kv_layer_list()[0][layer_idx]

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        """Return the dtype of the per-layer tensor for ``layer_idx``."""
        return self._as_kv_layer_list()[0][layer_idx].dtype

    def concrete_shape_str(self) -> str:
        """Return ``shape_desc`` with concrete numeric values filled in."""
        nl = self.num_layers()
        nb = self.num_blocks()
        bs = self.block_size()
        nh = self.num_heads()
        hs = self.head_size()
        return f"2 x {nl} x [{nb}, {bs}, {nh}, {hs}]"
