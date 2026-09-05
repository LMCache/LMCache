# SPDX-License-Identifier: Apache-2.0
"""Object model for Engine KV layouts.

LMCache's native kernels still dispatch on small integer layout codes, but the
Python/public surface does not need enum semantics. This module defines one
singleton :class:`KVLayout` object per supported layout and installs those
objects onto ``lmcache.lmcache_native`` so call sites can depend on layout
facts instead of enum categories.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
    from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache


class KVLayout(int):
    """One KV layout singleton plus its static facts.

    ``KVLayout`` is an ``int`` subclass so native bindings that accept integer
    layout codes continue to work unchanged. Each singleton also carries the
    Python-side metadata that call sites actually care about.
    """

    name: str
    code: int
    is_cross_layer: bool
    is_kv_list: bool
    is_layer_list: bool
    is_mla: bool
    is_hnd: bool
    is_fused_packed: bool
    is_two_major: bool
    is_pbs_fused: bool
    is_kv_second_tuple: bool
    NB_NL_TWO_BS_NH_HS: ClassVar["KVLayout"]
    NL_X_TWO_NB_BS_NH_HS: ClassVar["KVLayout"]
    NL_X_NB_TWO_BS_NH_HS: ClassVar["KVLayout"]
    NL_X_NB_BS_HS: ClassVar["KVLayout"]
    TWO_X_NL_X_NBBS_NH_HS: ClassVar["KVLayout"]
    NL_X_NBBS_ONE_HS: ClassVar["KVLayout"]
    NL_X_TWO_NB_NH_BS_HS: ClassVar["KVLayout"]
    NL_X_NB_TWO_NH_BS_HS: ClassVar["KVLayout"]
    NB_NL_TWO_NH_BS_HS: ClassVar["KVLayout"]
    TWO_X_NL_X_NB_BS_NH_HS: ClassVar["KVLayout"]
    NL_X_NB_NH_BS_TWO_HS: ClassVar["KVLayout"]
    NL_X_NB_BS_NH_TWO_HS: ClassVar["KVLayout"]
    NL_X_NB_NH_BS_CS: ClassVar["KVLayout"]
    NL_X_NB_BS_NH_CS: ClassVar["KVLayout"]
    NL_X_NB_BSV_BSS: ClassVar["KVLayout"]
    NL_X_TWO_NB_NH_ONE_BS_HS: ClassVar["KVLayout"]
    NL_X_TWO_X_NB_BS_NH_HS: ClassVar["KVLayout"]

    def __new__(
        cls,
        code: int,
        *,
        name: str,
        is_cross_layer: bool = False,
        is_kv_list: bool = False,
        is_layer_list: bool = False,
        is_mla: bool = False,
        is_hnd: bool = False,
        is_fused_packed: bool = False,
        is_two_major: bool = False,
        is_pbs_fused: bool = False,
        is_kv_second_tuple: bool = False,
    ) -> "KVLayout":
        obj = int.__new__(cls, code)
        obj.name = name
        obj.code = code
        obj.is_cross_layer = is_cross_layer
        obj.is_kv_list = is_kv_list
        obj.is_layer_list = is_layer_list
        obj.is_mla = is_mla
        obj.is_hnd = is_hnd
        obj.is_fused_packed = is_fused_packed
        obj.is_two_major = is_two_major
        obj.is_pbs_fused = is_pbs_fused
        obj.is_kv_second_tuple = is_kv_second_tuple
        return obj

    @property
    def value(self) -> int:
        """Backward-compatible enum-style numeric value."""
        return int(self)

    def __repr__(self) -> str:
        return f"KVLayout.{self.name}"

    __str__ = __repr__

    def __reduce__(self) -> tuple[Callable[[str], "KVLayout"], tuple[str]]:
        return (kv_layout_from_name, (self.name,))

    @classmethod
    def all(cls) -> tuple["KVLayout", ...]:
        """Return every registered layout singleton in code order."""
        return ALL_KV_LAYOUTS

    @classmethod
    def from_code(cls, code: int) -> "KVLayout":
        """Return the layout singleton for *code*."""
        return kv_layout_from_code(code)

    @classmethod
    def from_name(cls, name: str) -> "KVLayout":
        """Return the layout singleton for *name*."""
        return kv_layout_from_name(name)

    def spec_class(self) -> type["KVFormatSpec"]:
        """Return the ``KVFormatSpec`` class that owns this layout's geometry."""
        # First Party
        from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec_class

        return get_spec_class(cast(Any, self))

    def spec(self, kv_caches: "DiscoverableKVCache") -> "KVFormatSpec":
        """Bind the layout to concrete KV tensors and return a spec instance."""
        return self.spec_class()(kv_caches)

    def describe_shape(self) -> str:
        """Return the symbolic shape description for this layout."""
        # First Party
        from lmcache.v1.gpu_connector.kv_format.specs.base import describe_shape

        return describe_shape(cast(Any, self))


def _define_layout(
    name: str,
    code: int,
    **facts: bool,
) -> KVLayout:
    layout = KVLayout(code, name=name, **facts)
    setattr(KVLayout, name, layout)
    return layout


ALL_KV_LAYOUTS = (
    _define_layout("NB_NL_TWO_BS_NH_HS", 0, is_cross_layer=True),
    _define_layout("NL_X_TWO_NB_BS_NH_HS", 1, is_layer_list=True, is_two_major=True),
    _define_layout("NL_X_NB_TWO_BS_NH_HS", 2, is_layer_list=True),
    _define_layout("NL_X_NB_BS_HS", 3, is_layer_list=True, is_mla=True),
    _define_layout("TWO_X_NL_X_NBBS_NH_HS", 4, is_kv_list=True),
    _define_layout(
        "NL_X_NBBS_ONE_HS",
        5,
        is_layer_list=True,
        is_mla=True,
        is_pbs_fused=True,
    ),
    _define_layout(
        "NL_X_TWO_NB_NH_BS_HS",
        6,
        is_layer_list=True,
        is_hnd=True,
        is_two_major=True,
    ),
    _define_layout("NL_X_NB_TWO_NH_BS_HS", 7, is_layer_list=True, is_hnd=True),
    _define_layout("NB_NL_TWO_NH_BS_HS", 8, is_cross_layer=True, is_hnd=True),
    _define_layout("TWO_X_NL_X_NB_BS_NH_HS", 9, is_kv_list=True),
    _define_layout(
        "NL_X_NB_NH_BS_TWO_HS",
        10,
        is_layer_list=True,
        is_hnd=True,
        is_fused_packed=True,
    ),
    _define_layout(
        "NL_X_NB_BS_NH_TWO_HS", 11, is_layer_list=True, is_fused_packed=True
    ),
    _define_layout(
        "NL_X_NB_NH_BS_CS",
        12,
        is_layer_list=True,
        is_hnd=True,
        is_fused_packed=True,
    ),
    _define_layout("NL_X_NB_BS_NH_CS", 13, is_layer_list=True, is_fused_packed=True),
    _define_layout("NL_X_NB_BSV_BSS", 14, is_layer_list=True, is_mla=True),
    _define_layout(
        "NL_X_TWO_NB_NH_ONE_BS_HS",
        15,
        is_layer_list=True,
        is_hnd=True,
        is_two_major=True,
    ),
    _define_layout(
        "NL_X_TWO_X_NB_BS_NH_HS",
        16,
        is_layer_list=True,
        is_kv_second_tuple=True,
    ),
)

_KV_LAYOUTS_BY_CODE = {int(layout): layout for layout in ALL_KV_LAYOUTS}
_KV_LAYOUTS_BY_NAME = {layout.name: layout for layout in ALL_KV_LAYOUTS}


def kv_layout_from_code(code: int) -> KVLayout:
    """Return the registered layout singleton for *code*."""
    try:
        return _KV_LAYOUTS_BY_CODE[int(code)]
    except KeyError as exc:
        raise ValueError(f"Unknown KV layout code: {code}") from exc


def kv_layout_from_name(name: str) -> KVLayout:
    """Return the registered layout singleton for *name*."""
    try:
        return _KV_LAYOUTS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown KV layout name: {name}") from exc


def _coerce_layout(layout: KVLayout | int) -> KVLayout:
    if isinstance(layout, KVLayout):
        return layout
    return kv_layout_from_code(int(layout))


def is_kv_list(layout: KVLayout | int) -> bool:
    return _coerce_layout(layout).is_kv_list


def is_layer_list(layout: KVLayout | int) -> bool:
    return _coerce_layout(layout).is_layer_list


def is_cross_layer(layout: KVLayout | int) -> bool:
    return _coerce_layout(layout).is_cross_layer


def is_mla(layout: KVLayout | int) -> bool:
    return _coerce_layout(layout).is_mla


def is_kv_second_tuple(layout: KVLayout | int) -> bool:
    return _coerce_layout(layout).is_kv_second_tuple


def install_on_native_module(native_module: ModuleType) -> None:
    """Replace enum-like layout exports on *native_module* with layout objects."""
    module = cast(Any, native_module)
    module.KVLayout = KVLayout
    module.EngineKVFormat = KVLayout
    module.GPUKVFormat = KVLayout
    for layout in ALL_KV_LAYOUTS:
        setattr(KVLayout, layout.name, layout)
    module.ALL_KV_LAYOUTS = ALL_KV_LAYOUTS
    module.kv_layout_from_code = kv_layout_from_code
    module.kv_layout_from_name = kv_layout_from_name
    module.is_kv_list = is_kv_list
    module.is_layer_list = is_layer_list
    module.is_cross_layer = is_cross_layer
    module.is_mla = is_mla
    module.is_kv_second_tuple = is_kv_second_tuple
