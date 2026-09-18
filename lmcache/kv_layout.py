# SPDX-License-Identifier: Apache-2.0
"""Dynamic object model for engine KV layouts.

This module intentionally does *not* hand-maintain a second copy of the layout
catalog. The per-layout ``KVFormatSpec`` classes under
``lmcache.v1.gpu_connector.kv_format.specs`` are already discovered
dynamically, and each one declares the static facts that downstream code cares
about. We therefore build one :class:`KVLayout` singleton per discovered spec at
runtime and expose those objects through ``lmcache.lmcache_native``.

Result: adding a new Python-side layout means adding one spec file; the layout
object registry, ``KVLayout.all()``, and the module/class attributes are derived
from that single definition point.
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

# First Party
from lmcache.v1.kv_layout_meta import (
    concrete_axis_groups,
    describe_axis_groups,
    parse_axis_groups,
)


class KVLayout(int):
    """One discovered KV layout singleton plus its static facts."""

    __members__: ClassVar[dict[str, "KVLayout"]]

    name: str
    code: int
    axis_groups: tuple[tuple[str, ...], ...]
    outer_axes: tuple[str, ...]
    inner_axes: tuple[str, ...]
    attention_backends: tuple[str, ...]
    is_cross_layer: bool
    is_kv_list: bool
    is_layer_list: bool
    is_mla: bool
    is_hnd: bool
    is_fused_packed: bool
    is_two_major: bool
    is_pbs_fused: bool
    is_kv_second_tuple: bool
    probe_tensor_block_axis: int | None
    _spec_class: type["KVFormatSpec"]

    def __new__(
        cls,
        code: int,
        *,
        name: str,
        axis_groups: tuple[tuple[str, ...], ...],
        spec_class: type["KVFormatSpec"],
    ) -> "KVLayout":
        obj = int.__new__(cls, code)
        obj.name = name
        obj.code = code
        obj.axis_groups = axis_groups
        obj.outer_axes = tuple("_".join(group) for group in axis_groups[:-1])
        obj.inner_axes = axis_groups[-1]
        obj.attention_backends = tuple(getattr(spec_class, "attention_backends", ()))
        obj.is_cross_layer = bool(getattr(spec_class, "is_cross_layer", False))
        obj.is_kv_list = bool(getattr(spec_class, "is_kv_list", False))
        obj.is_layer_list = bool(getattr(spec_class, "is_layer_list", False))
        obj.is_mla = bool(getattr(spec_class, "is_mla", False))
        obj.is_hnd = bool(getattr(spec_class, "is_hnd", False))
        obj.is_fused_packed = bool(getattr(spec_class, "is_fused_packed", False))
        obj.is_two_major = bool(getattr(spec_class, "is_two_major", False))
        obj.is_pbs_fused = bool(getattr(spec_class, "is_pbs_fused", False))
        obj.is_kv_second_tuple = bool(getattr(spec_class, "is_kv_second_tuple", False))
        obj.probe_tensor_block_axis = _probe_tensor_block_axis(axis_groups, obj)
        obj._spec_class = spec_class
        return obj

    @property
    def value(self) -> int:
        """Backward-compatible enum-style numeric value."""
        return int(self)

    @property
    def is_plane_tuple(self) -> bool:
        """Whether the per-layer entry is an NP plane tuple, not a fixed K/V pair."""
        return self.is_kv_second_tuple and "NP" in self.outer_axes

    @property
    def supports_dim0_block_padding(self) -> bool:
        """Whether dim-0 padding is meaningful and currently supported."""
        return self.probe_tensor_block_axis == 0 and (
            self.is_mla
            or self.is_fused_packed
            or self.is_plane_tuple
            or not self.has_inner_axis("TWO")
        )

    def __repr__(self) -> str:
        return f"KVLayout.{self.name}"

    __str__ = __repr__

    def __reduce__(self) -> tuple[Callable[[int], "KVLayout"], tuple[int]]:
        return (kv_layout_from_code, (int(self),))

    @classmethod
    def all(cls) -> tuple["KVLayout", ...]:
        """Return every discovered layout singleton in numeric-code order."""
        _ensure_registry()
        return _ALL_KV_LAYOUTS

    @classmethod
    def from_code(cls, code: int) -> "KVLayout":
        return kv_layout_from_code(code)

    @classmethod
    def from_name(cls, name: str) -> "KVLayout":
        return kv_layout_from_name(name)

    def has_outer_axis(self, axis: str) -> bool:
        return axis in self.outer_axes

    def has_inner_axis(self, axis: str) -> bool:
        return axis in self.inner_axes

    def inner_axis_index(self, axis: str) -> int | None:
        try:
            return self.inner_axes.index(axis)
        except ValueError:
            return None

    def inner_shape(self, *, nb: int, bs: int, nh: int, hs: int) -> tuple[int, ...]:
        """Return the symbolic per-tensor inner shape from ``shape_desc`` dims."""
        sizes = {
            "ONE": 1,
            "TWO": 2,
            "NBBS": nb * bs,
            "NB": nb,
            "BS": bs,
            "NH": nh,
            "HS": hs,
            "CS": hs,
            "BSV": bs,
            "BSS": bs,
        }
        try:
            return tuple(int(sizes[axis]) for axis in self.inner_axes)
        except KeyError as exc:
            raise ValueError(
                f"KV layout {self.name} uses unsupported inner axis {exc.args[0]!r}"
            ) from exc

    def paged_tensor_shape(
        self, *, nb: int, bs: int, nh: int, hs: int
    ) -> tuple[int, ...]:
        """Return the per-layer tensor shape reconstructed from ``shape_desc``.

        For fused ``..._TWO_HS`` layouts, ``shape_desc.hs`` already carries the
        packed content width (``2 * head_size``), so the physical tensor is the
        flattened 4-D leaf rather than the symbolic 5-D ``[..., TWO, HS]`` form.
        """
        if self.is_fused_packed and self.inner_axes[-2:] == ("TWO", "HS"):
            prefix = self.inner_shape(nb=nb, bs=bs, nh=nh, hs=hs)[:-2]
            return prefix + (hs,)
        return self.inner_shape(nb=nb, bs=bs, nh=nh, hs=hs)

    def num_layers(self, kv_caches: "DiscoverableKVCache") -> int:
        return self.spec(kv_caches).num_layers()

    def num_blocks(self, kv_caches: "DiscoverableKVCache") -> int:
        return self.spec(kv_caches).num_blocks()

    def block_size(self, kv_caches: "DiscoverableKVCache", layer_idx: int = 0) -> int:
        return self.spec(kv_caches).block_size(layer_idx)

    def page_buffer_size(self, kv_caches: "DiscoverableKVCache") -> int:
        return self.spec(kv_caches).page_buffer_size()

    def kv_size(self, kv_caches: "DiscoverableKVCache") -> int:
        return self.spec(kv_caches).kv_size()

    def num_heads(self, kv_caches: "DiscoverableKVCache", layer_idx: int = 0) -> int:
        return self.spec(kv_caches).num_heads(layer_idx)

    def hidden_dim(self, kv_caches: "DiscoverableKVCache", layer_idx: int = 0) -> int:
        return self.spec(kv_caches).hidden_dim(layer_idx)

    def head_size(self, kv_caches: "DiscoverableKVCache", layer_idx: int = 0) -> int:
        return self.spec(kv_caches).head_size(layer_idx)

    def tokens_per_layer(self, kv_caches: "DiscoverableKVCache") -> int:
        return self.spec(kv_caches).tokens_per_layer()

    def elements_per_layer(self, kv_caches: "DiscoverableKVCache") -> int:
        return self.spec(kv_caches).elements_per_layer()

    def dtype(self, kv_caches: "DiscoverableKVCache", layer_idx: int = 0) -> Any:
        return self.spec(kv_caches).dtype(layer_idx)

    def data_ptrs(
        self, kv_caches: "DiscoverableKVCache", layer_indices: list[int]
    ) -> list[int]:
        return self.spec(kv_caches).data_ptrs(layer_indices)

    def spec_class(self) -> type["KVFormatSpec"]:
        return cast("type[KVFormatSpec]", self._spec_class)

    def spec(self, kv_caches: "DiscoverableKVCache") -> "KVFormatSpec":
        return self.spec_class()(kv_caches)

    def describe_shape(self) -> str:
        return describe_axis_groups(self.axis_groups)

    def concrete_shape(self, size: Callable[[str], int]) -> str:
        return concrete_axis_groups(self.axis_groups, size)

    def concrete_shape_from(self, kv_caches: "DiscoverableKVCache") -> str:
        return self.spec(kv_caches).concrete_shape_str()


_ALL_KV_LAYOUTS: tuple[KVLayout, ...] = ()
_KV_LAYOUTS_BY_CODE: dict[int, KVLayout] = {}
_KV_LAYOUTS_BY_NAME: dict[str, KVLayout] = {}


def _probe_tensor_block_axis(
    axis_groups: tuple[tuple[str, ...], ...], layout: KVLayout
) -> int | None:
    if layout.is_cross_layer or layout.is_kv_list or layout.is_pbs_fused:
        return None
    inner_axes = axis_groups[-1]
    try:
        return inner_axes.index("NB")
    except ValueError:
        return None


def _ensure_registry() -> None:
    global _ALL_KV_LAYOUTS
    if _ALL_KV_LAYOUTS:
        return

    # Imported lazily so top-level ``lmcache`` import does not pull the native
    # extension before the platform package has prepared the torch runtime.
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.registry import SPECS

    layouts: list[KVLayout] = []
    for native_layout, spec_class in sorted(
        SPECS.items(), key=lambda item: int(item[0])
    ):
        layout = KVLayout(
            int(native_layout),
            name=native_layout.name,
            axis_groups=parse_axis_groups(native_layout.name),
            spec_class=spec_class,
        )
        setattr(KVLayout, layout.name, layout)
        spec_class.engine_kv_format = cast(Any, layout)
        layouts.append(layout)

    _ALL_KV_LAYOUTS = tuple(layouts)
    _KV_LAYOUTS_BY_CODE.update({int(layout): layout for layout in _ALL_KV_LAYOUTS})
    _KV_LAYOUTS_BY_NAME.update({layout.name: layout for layout in _ALL_KV_LAYOUTS})
    KVLayout.__members__ = {layout.name: layout for layout in _ALL_KV_LAYOUTS}


def kv_layout_from_code(code: int) -> KVLayout:
    _ensure_registry()
    try:
        return _KV_LAYOUTS_BY_CODE[int(code)]
    except KeyError as exc:
        raise ValueError(f"Unknown KV layout code: {code}") from exc


def kv_layout_from_name(name: str) -> KVLayout:
    _ensure_registry()
    try:
        return _KV_LAYOUTS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown KV layout name: {name}") from exc


def _coerce_layout(layout: KVLayout | int) -> KVLayout:
    _ensure_registry()
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
    """Expose discovered layout objects through ``lmcache.lmcache_native``."""
    _ensure_registry()
    module = cast(Any, native_module)
    module.KVLayout = KVLayout
    module.EngineKVFormat = KVLayout
    module.GPUKVFormat = KVLayout
    module.ALL_KV_LAYOUTS = _ALL_KV_LAYOUTS
    module.kv_layout_from_code = kv_layout_from_code
    module.kv_layout_from_name = kv_layout_from_name
    module.is_kv_list = is_kv_list
    module.is_layer_list = is_layer_list
    module.is_cross_layer = is_cross_layer
    module.is_mla = is_mla
    module.is_kv_second_tuple = is_kv_second_tuple
