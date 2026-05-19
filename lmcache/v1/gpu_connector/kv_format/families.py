# SPDX-License-Identifier: Apache-2.0
"""Mid-level :class:`KVFormatSpec` family bases.

These classes capture the *common machinery* shared by formats that
differ only in axis ordering. Concrete leaf specs declare the position
of each axis as small ``ClassVar[int]`` constants and inherit
everything else.

All classes here set ``abstract = True`` so they are skipped by the
auto-registration in :meth:`KVFormatSpec.__init_subclass__`.
"""

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.base import KVFormatSpec


class PerLayer5DSpec(KVFormatSpec):
    """vLLM per-layer non-MLA formats: ``NL x [<5 axes>]``.

    The 5 axes are: the K/V dim of size 2, NB, NH, BS, HS. Subclasses
    declare each axis index in the inner tensor by overriding the
    ``_AX_*`` ClassVars.
    """

    abstract: ClassVar[bool] = True

    # Axis index inside the per-layer tensor. Override in leaves.
    _AX_TWO: ClassVar[int]
    _AX_NB: ClassVar[int]
    _AX_NH: ClassVar[int]
    _AX_BS: ClassVar[int]
    _AX_HS: ClassVar[int]

    def num_layers(self) -> int:
        """Return the number of transformer layers in the KV cache."""
        return len(self._as_layer_list())

    def num_blocks(self) -> int:
        """Return the number of pre-allocated KV cache blocks."""
        return self._as_layer_list()[0].shape[self._AX_NB]

    def block_size(self, layer_idx: int = 0) -> int:
        """Return the per-block token capacity for ``layer_idx``."""
        return self._as_layer_list()[layer_idx].shape[self._AX_BS]

    def num_heads(self, layer_idx: int = 0) -> int:
        """Return the number of KV heads for ``layer_idx``."""
        return self._as_layer_list()[layer_idx].shape[self._AX_NH]

    def head_size(self, layer_idx: int = 0) -> int:
        """Return the per-head hidden size for ``layer_idx``."""
        return self._as_layer_list()[layer_idx].shape[self._AX_HS]

    def page_buffer_size(self) -> int:
        """Return ``num_blocks * block_size`` for the cache."""
        t = self._as_layer_list()[0]
        return t.shape[self._AX_NB] * t.shape[self._AX_BS]

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        """Return one device pointer per requested layer."""
        layers = self._as_layer_list()
        return [layers[i].data_ptr() for i in layer_indices]

    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        """Return the per-layer tensor used for layout introspection."""
        return self._as_layer_list()[layer_idx]

    def concrete_shape_str(self) -> str:
        """Return ``shape_desc`` with concrete numeric values filled in."""
        # Reconstruct the shape skeleton from the declared axis order.
        names = {
            self._AX_TWO: "2",
            self._AX_NB: str(self.num_blocks()),
            self._AX_NH: str(self.num_heads()),
            self._AX_BS: str(self.block_size()),
            self._AX_HS: str(self.head_size()),
        }
        ordered = [names[i] for i in range(5)]
        return f"{self.num_layers()} x [{', '.join(ordered)}]"


class CrossLayer6DSpec(KVFormatSpec):
    """Cross-layer formats: a single tensor ``[<6 axes>]`` packing all
    layers. Axes: NB, NL, the K/V dim of size 2, NH, BS, HS.
    """

    abstract: ClassVar[bool] = True
    is_cross_layer: ClassVar[bool] = True

    _AX_NB: ClassVar[int]
    _AX_NL: ClassVar[int]
    _AX_TWO: ClassVar[int]
    _AX_NH: ClassVar[int]
    _AX_BS: ClassVar[int]
    _AX_HS: ClassVar[int]

    def num_layers(self) -> int:
        """Return the number of transformer layers in the KV cache."""
        return self._as_tensor().shape[self._AX_NL]

    def num_blocks(self) -> int:
        """Return the number of pre-allocated KV cache blocks."""
        return self._as_tensor().shape[self._AX_NB]

    def block_size(self, layer_idx: int = 0) -> int:
        """Return the per-block token capacity (shared across layers)."""
        return self._as_tensor().shape[self._AX_BS]

    def num_heads(self, layer_idx: int = 0) -> int:
        """Return the number of KV heads (shared across layers)."""
        return self._as_tensor().shape[self._AX_NH]

    def head_size(self, layer_idx: int = 0) -> int:
        """Return the per-head hidden size (shared across layers)."""
        return self._as_tensor().shape[self._AX_HS]

    def page_buffer_size(self) -> int:
        """Return ``num_blocks * block_size`` for the cache."""
        t = self._as_tensor()
        return t.shape[self._AX_NB] * t.shape[self._AX_BS]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        """Return ``num_heads * head_size`` for the cache."""
        t = self._as_tensor()
        return t.shape[self._AX_NH] * t.shape[self._AX_HS]

    def elements_per_layer(self) -> int:
        """Return the per-layer element count (K + V combined)."""
        t = self._as_tensor()
        return (
            t.shape[self._AX_NB]
            * 2
            * t.shape[self._AX_BS]
            * t.shape[self._AX_NH]
            * t.shape[self._AX_HS]
        )

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        """Return the single device pointer for the cross-layer tensor."""
        return [self._as_tensor().data_ptr()]

    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        """Return the cross-layer tensor used for layout introspection."""
        return self._as_tensor()

    def concrete_shape_str(self) -> str:
        """Return ``shape_desc`` with concrete numeric values filled in."""
        t = self._as_tensor()
        names = {
            self._AX_NB: str(t.shape[self._AX_NB]),
            self._AX_NL: str(t.shape[self._AX_NL]),
            self._AX_TWO: "2",
            self._AX_NH: str(t.shape[self._AX_NH]),
            self._AX_BS: str(t.shape[self._AX_BS]),
            self._AX_HS: str(t.shape[self._AX_HS]),
        }
        ordered = [names[i] for i in range(6)]
        return f"[{', '.join(ordered)}]"


class SGLangFusedPBSSpec(KVFormatSpec):
    """SGLang-style fused-PBS formats: NB and BS are collapsed into a
    single ``page_buffer_size`` axis.
    """

    abstract: ClassVar[bool] = True
    engine: ClassVar[str] = "sglang"
    has_separate_block_dims: ClassVar[bool] = False
