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
        return len(self._as_layer_list())

    def num_blocks(self) -> int:
        return self._as_layer_list()[0].shape[self._AX_NB]

    def block_size(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[self._AX_BS]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[self._AX_NH]

    def head_size(self, layer_idx: int = 0) -> int:
        return self._as_layer_list()[layer_idx].shape[self._AX_HS]

    def page_buffer_size(self) -> int:
        t = self._as_layer_list()[0]
        return t.shape[self._AX_NB] * t.shape[self._AX_BS]

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = self._as_layer_list()
        return [layers[i].data_ptr() for i in layer_indices]

    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        return self._as_layer_list()[layer_idx]

    def concrete_shape_str(self) -> str:
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
        return self._as_tensor().shape[self._AX_NL]

    def num_blocks(self) -> int:
        return self._as_tensor().shape[self._AX_NB]

    def block_size(self, layer_idx: int = 0) -> int:
        return self._as_tensor().shape[self._AX_BS]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self._as_tensor().shape[self._AX_NH]

    def head_size(self, layer_idx: int = 0) -> int:
        return self._as_tensor().shape[self._AX_HS]

    def page_buffer_size(self) -> int:
        t = self._as_tensor()
        return t.shape[self._AX_NB] * t.shape[self._AX_BS]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self._as_tensor()
        return t.shape[self._AX_NH] * t.shape[self._AX_HS]

    def elements_per_layer(self) -> int:
        t = self._as_tensor()
        return (
            t.shape[self._AX_NB]
            * 2
            * t.shape[self._AX_BS]
            * t.shape[self._AX_NH]
            * t.shape[self._AX_HS]
        )

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        return [self._as_tensor().data_ptr()]

    def layout_probe_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        return self._as_tensor()

    def concrete_shape_str(self) -> str:
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
